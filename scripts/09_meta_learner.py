"""
STEP 9 — Stacking meta-learner (Track A ⊕ Track C) with ablation
==================================================================

Standard stacking (Wolpert 1992): the meta-learner is an L2-regularised
logistic regression on two inputs
    logit(Track A out-of-fold probability)   — produced by step 08
    Track C rule logit                       — produced by step 07
so every training variant's Track A score comes from a model that never saw
that variant. The meta-learner cannot leak.

Ablation (the paper's Track C question)
    A alone   vs   A + C
    evaluated by grouped CV on the training pool AND on every held-out test
    panel, with DeLong tests and paired bootstrap CIs on the AUC difference.

Inputs
    results/<scheme>_<feature_set>/oof_track_a.csv
    results/<scheme>_<feature_set>/predictions_<panel>.csv   (P_raw_ensemble)
    data/track_c/<scheme>_<feature_set>/{train,test}_<panel>_trackc.csv
Outputs (results/<scheme>_<feature_set>/meta/)
    meta_results.json, meta_ablation_table.csv, predictions_meta_<panel>.csv,
    meta_learner.pkl
"""

from __future__ import annotations

import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

from utils.logging_utils import get_logger
from utils.config import load_config, run_dir
from utils.schema import TARGET_COL, SINGLE_GENE_PANELS
from utils.metrics import evaluate, bootstrap_ci, delong_roc_test, paired_bootstrap_auc_diff, best_threshold_youden

log = get_logger("09_meta_learner.log")
EPS = 1e-4


def safe_logit(p):
    return logit(np.clip(np.asarray(p, float), EPS, 1 - EPS))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/pipeline.yaml")
    ap.add_argument("--scheme", default="gene", choices=["gene", "variant"])
    ap.add_argument("--feature-set", default="core", choices=["core", "full"])
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    SEED = int(cfg["seed"]); PANELS = cfg["panels"]
    N_BOOT = 200 if args.fast else int(cfg["model"]["n_bootstrap"])
    rdir = run_dir(cfg, args.scheme, args.feature_set)
    tc_dir = os.path.join(cfg["paths"]["track_c"], f"{args.scheme}_{args.feature_set}")
    out = os.path.join(rdir, "meta"); os.makedirs(out, exist_ok=True)
    log.info(f"META  scheme={args.scheme}  feature_set={args.feature_set}")

    # ── training pool: OOF (fit rows only) ⋈ Track C ────────────────────
    oof = pd.read_csv(os.path.join(rdir, "oof_track_a.csv"))
    oof = oof[oof["in_fit_set"]].copy()
    tc = pd.concat([pd.read_csv(os.path.join(tc_dir, f"train_{p}_trackc.csv")).assign(_panel=p)
                    for p in PANELS], ignore_index=True)
    oof["_variant_id"] = oof["_variant_id"].astype(str); tc["_variant_id"] = tc["_variant_id"].astype(str)
    df = oof.merge(tc[["_panel", "_variant_id", "track_c_logit", "track_c_score"]],
                   on=["_panel", "_variant_id"], how="left", validate="one_to_one")
    assert df["track_c_logit"].notna().all(), "Track C rows missing for some training variants"
    y = df[TARGET_COL].astype(int).values
    XA = safe_logit(df["oof_ensemble"]).reshape(-1, 1)
    XAC = np.column_stack([XA[:, 0], df["track_c_logit"].values])
    groups = np.array([f"{p}:{i}" if p in SINGLE_GENE_PANELS else g
                       for i, (g, p) in enumerate(zip(df["_gene"].astype(str), df["_panel"]))])
    log.info(f"  training pool: {len(df)} variants  Track A OOF AUC={roc_auc_score(y, XA[:, 0]):.4f}  "
             f"Track C AUC={roc_auc_score(y, df['track_c_logit']):.4f}")

    # ── ablation by (grouped) CV ─────────────────────────────────────────
    if args.scheme == "gene":
        folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(XAC, y, groups))
    else:
        folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(XAC, y))
    cv_pred = {"A": np.zeros(len(y)), "A+C": np.zeros(len(y))}
    for tr, va in folds:
        for name, X in (("A", XA), ("A+C", XAC)):
            lr = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED).fit(X[tr], y[tr])
            cv_pred[name][va] = lr.predict_proba(X[va])[:, 1]
    cv_auc = {k: float(roc_auc_score(y, v)) for k, v in cv_pred.items()}
    cv_test = delong_roc_test(y, cv_pred["A+C"], cv_pred["A"])
    cv_boot = paired_bootstrap_auc_diff(y, cv_pred["A+C"], cv_pred["A"], n_boot=N_BOOT, seed=SEED)
    log.info(f"  CV ablation: A={cv_auc['A']:.4f}  A+C={cv_auc['A+C']:.4f}  "
             f"Δ={cv_test['delta']:+.4f}  DeLong p={cv_test['p']:.3g}")

    # ── final meta-learner (C chosen by CV over a small grid) ───────────
    best_C, best_auc = 1.0, -1
    for C in (0.01, 0.1, 1.0, 10.0):
        pred = np.zeros(len(y))
        for tr, va in folds:
            pred[va] = LogisticRegression(C=C, max_iter=1000, random_state=SEED).fit(XAC[tr], y[tr]).predict_proba(XAC[va])[:, 1]
        a = roc_auc_score(y, pred)
        if a > best_auc:
            best_auc, best_C = a, C
    meta = LogisticRegression(C=best_C, max_iter=1000, random_state=SEED).fit(XAC, y)
    meta_A = LogisticRegression(C=best_C, max_iter=1000, random_state=SEED).fit(XA, y)
    coef = {"intercept": float(meta.intercept_[0]), "w_track_a_logit": float(meta.coef_[0][0]),
            "w_track_c_logit": float(meta.coef_[0][1]), "C": best_C}
    thr = best_threshold_youden(y, cv_pred["A+C"])
    log.info(f"  meta coefficients: {coef}")

    # ── held-out test panels ─────────────────────────────────────────────
    rows, test_res = [], {}
    for p in PANELS:
        pa = pd.read_csv(os.path.join(rdir, f"predictions_{p}.csv"))
        pc = pd.read_csv(os.path.join(tc_dir, f"test_{p}_trackc.csv"))
        pa["_variant_id"] = pa["_variant_id"].astype(str); pc["_variant_id"] = pc["_variant_id"].astype(str)
        d = pa.merge(pc[["_variant_id", "track_c_logit", "track_c_score"]], on="_variant_id", how="left", validate="one_to_one")
        assert d["track_c_logit"].notna().all()
        yt = d[TARGET_COL].astype(int).values
        if len(np.unique(yt)) < 2 or len(yt) < 10:
            log.warning(f"  TEST {p:<8} skipped — single class or < 10 variants")
            continue
        xa = safe_logit(d["P_raw_ensemble"])
        p_A = meta_A.predict_proba(xa.reshape(-1, 1))[:, 1]
        p_AC = meta.predict_proba(np.column_stack([xa, d["track_c_logit"].values]))[:, 1]
        p_C = d["track_c_score"].values
        dl = delong_roc_test(yt, p_AC, p_A)
        bs = paired_bootstrap_auc_diff(yt, p_AC, p_A, n_boot=N_BOOT, seed=SEED)
        ci = bootstrap_ci(yt, p_AC, thr, n_boot=N_BOOT, seed=SEED)
        m = evaluate(yt, p_AC, thr)
        test_res[p] = {"auc_track_a": float(roc_auc_score(yt, p_A)), "auc_track_c": float(roc_auc_score(yt, p_C)),
                       "auc_meta": m["roc_auc"], "meta_metrics": m, "meta_ci": ci,
                       "delong_meta_vs_a": dl, "bootstrap_meta_vs_a": bs}
        rows.append({"panel": p, "n": len(yt), "auc_track_a": test_res[p]["auc_track_a"],
                     "auc_track_c": test_res[p]["auc_track_c"], "auc_meta": m["roc_auc"],
                     "delta_meta_minus_a": dl["delta"], "delong_p": dl["p"],
                     "delta_ci_lo": bs["ci"][0], "delta_ci_hi": bs["ci"][1]})
        log.info(f"  TEST {p:<8} A={test_res[p]['auc_track_a']:.4f}  C={test_res[p]['auc_track_c']:.4f}  "
                 f"A+C={m['roc_auc']:.4f}  Δ={dl['delta']:+.4f} (p={dl['p']:.3g})")
        d.assign(P_meta=p_AC, P_track_a_meta=p_A, pred_meta=(p_AC >= thr).astype(int)).to_csv(
            os.path.join(out, f"predictions_meta_{p}.csv"), index=False)

    pd.DataFrame(rows).to_csv(os.path.join(out, "meta_ablation_table.csv"), index=False)
    with open(os.path.join(out, "meta_results.json"), "w") as f:
        json.dump({"scheme": args.scheme, "feature_set": args.feature_set, "n_train": int(len(y)),
                   "cv_auc": cv_auc, "cv_delong_meta_vs_a": cv_test, "cv_bootstrap_meta_vs_a": cv_boot,
                   "coefficients": coef, "threshold_youden": thr, "test": test_res}, f, indent=2, default=float)
    with open(os.path.join(out, "meta_learner.pkl"), "wb") as f:
        pickle.dump({"meta": meta, "meta_track_a_only": meta_A, "threshold": thr, "inputs": ["logit(P_raw_ensemble)", "track_c_logit"]}, f)
    log.info(f"✓ meta-learner outputs → {out}")


if __name__ == "__main__":
    main()
