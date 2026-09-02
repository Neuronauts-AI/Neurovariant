"""
STEP 8 — Track A: XGBoost + LightGBM ensemble (script version of PSR_Pipeline.ipynb)
=====================================================================================

One invocation = one cell of the experimental matrix:

    python scripts/08_train_track_a.py --scheme gene    --feature-set core   # primary
    python scripts/08_train_track_a.py --scheme gene    --feature-set full
    python scripts/08_train_track_a.py --scheme variant --feature-set core
    python scripts/08_train_track_a.py --scheme variant --feature-set full

What it does (in order)
  1. load data/enriched/<scheme>/{train,test}_<panel>_enriched.csv
  2. feature selection: schema feature set → variance filter → |r|>0.95 filter
     (both fitted on TRAINING data only)
  3. 15 % calibration hold-out (gene-grouped under the gene scheme)
  4. 5-fold CV with default hyper-parameters → OOF predictions
     (StratifiedGroupKFold by gene under the gene scheme) + fold_assignments.json
  5. LOOCV on the CFTR panel (small-sample stability check)
  6. Optuna TPE search, XGBoost and LightGBM separately (inner 3-fold CV,
     grouped under the gene scheme)
  7. final fit → isotonic calibration on the hold-out
  8. thresholds: Youden's J and sensitivity-priority (≥ 0.90), per panel with
     global fallback when the panel has < 30 calibration variants
  9. per-panel test evaluation with 2000× bootstrap 95 % CIs
 10. figures (ROC, PR, confusion matrices, calibration, Optuna history and
     importance, SHAP beeswarm / group importance / TP+FN waterfalls)
 11. error analysis (MAF, tool disagreement, Grantham by error type)
 12. artefacts: model pickle, feature list, run_config.json, environment.json,
     predictions_<panel>.csv, oof_track_a.csv (for the meta-learner)

Everything lands in results/<scheme>_<feature_set>/.
"""

from __future__ import annotations

import os
import sys
import json
import time
import pickle
import argparse
import platform
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import (StratifiedKFold, StratifiedGroupKFold,
                                     GroupShuffleSplit, train_test_split, LeaveOneOut)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.base import clone
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve,
                             average_precision_score, confusion_matrix, ConfusionMatrixDisplay)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import optuna

from utils.logging_utils import get_logger
from utils.config import load_config, run_dir
from utils.schema import (select_features, feature_group, CAT_COLS, TARGET_COL,
                          SINGLE_GENE_PANELS, group_table)
from utils.metrics import (evaluate, best_threshold_youden, best_threshold_sensitivity,
                           bootstrap_ci)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
log = get_logger("08_train_track_a.log")


# ── Helpers ──────────────────────────────────────────────────────────────────
def make_groups(df: pd.DataFrame, scheme: str) -> np.ndarray:
    """Gene groups for CV. Single-gene panels get one group per row."""
    if scheme != "gene":
        return np.arange(len(df))
    g = df["_gene"].astype(str).values
    panel = df["_panel"].astype(str).values
    return np.array([f"{p}:{i}" if p in SINGLE_GENE_PANELS else gg
                     for i, (gg, p) in enumerate(zip(g, panel))])


def cv_splitter(n_splits: int, scheme: str, seed: int):
    if scheme == "gene":
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def cv_split(splitter, X, y, groups):
    return list(splitter.split(X, y, groups)) if isinstance(splitter, StratifiedGroupKFold) \
        else list(splitter.split(X, y))


def build_preprocessor(numeric_cols, categorical_cols):
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
         categorical_cols),
    ], remainder="drop")


def feature_names_out(pre) -> list[str]:
    return [n.split("__", 1)[1] for n in pre.get_feature_names_out()]


def savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(str(path).replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


def environment_info() -> dict:
    import sklearn, xgboost, lightgbm, shap, scipy
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__, "lightgbm": lightgbm.__version__,
        "optuna": optuna.__version__, "shap": shap.__version__, "scipy": scipy.__version__,
    }


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/pipeline.yaml")
    ap.add_argument("--scheme", default="gene", choices=["gene", "variant"])
    ap.add_argument("--feature-set", default="core", choices=["core", "full"])
    ap.add_argument("--n-trials", type=int, default=None, help="Optuna trials per learner")
    ap.add_argument("--fast", action="store_true", help="smoke-test settings (15 trials, 200 bootstrap)")
    ap.add_argument("--no-shap", action="store_true")
    ap.add_argument("--skip-loocv", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    M = cfg["model"]
    SEED = int(cfg["seed"])
    PANELS = cfg["panels"]
    N_FOLDS = int(M["n_folds"])
    N_TRIALS = args.n_trials if args.n_trials is not None else (15 if args.fast else int(M["n_trials"]))
    N_BOOT = 200 if args.fast else int(M["n_bootstrap"])
    scheme, fset = args.scheme, args.feature_set

    enriched_dir = Path(cfg["paths"]["enriched"]) / scheme
    out = Path(run_dir(cfg, scheme, fset))
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log.info(f"{'=' * 70}\nTRACK A  scheme={scheme}  feature_set={fset}  trials={N_TRIALS}  → {out}\n{'=' * 70}")

    # ── 1. Load ──────────────────────────────────────────────────────────
    trains, tests = {}, {}
    for p in PANELS:
        tr = enriched_dir / f"train_{p}_enriched.csv"
        te = enriched_dir / f"test_{p}_enriched.csv"
        if not tr.exists():
            sys.exit(f"missing {tr} — run 05/06 for scheme {scheme}")
        trains[p] = pd.read_csv(tr, low_memory=False)
        tests[p] = pd.read_csv(te, low_memory=False)
        log.info(f"  {p:<8} train={trains[p].shape}  test={tests[p].shape}")
    df_train_all = pd.concat([trains[p].assign(_panel=p) for p in PANELS], ignore_index=True)
    df_train_all["_row_id"] = np.arange(len(df_train_all))
    y_all = df_train_all[TARGET_COL].astype(int)
    groups_all = make_groups(df_train_all, scheme)

    # ── 2. Feature selection (train only) ───────────────────────────────
    feature_cols = select_features(df_train_all.columns, fset)
    n0 = len(feature_cols)
    num0 = [c for c in feature_cols if c not in CAT_COLS]
    Xn = df_train_all[num0].apply(pd.to_numeric, errors="coerce")
    low_var = Xn.columns[Xn.std() < float(M["variance_threshold"])].tolist()
    feature_cols = [c for c in feature_cols if c not in low_var]
    # base columns first so that, in a correlated pair, the enrich_* duplicate is the one dropped
    num1 = sorted([c for c in feature_cols if c not in CAT_COLS], key=lambda c: c.startswith("enrich_"))
    corr = df_train_all[num1].apply(pd.to_numeric, errors="coerce").fillna(0).corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if (upper[c] > float(M["correlation_threshold"])).any()]
    feature_cols = [c for c in feature_cols if c not in to_drop]
    numeric_cols = [c for c in feature_cols if c not in CAT_COLS]
    categorical_cols = [c for c in feature_cols if c in CAT_COLS]
    assert TARGET_COL not in feature_cols and not any(c.startswith("_") for c in feature_cols)
    assert not (fset == "core" and any(feature_group(c) == "insilico" for c in feature_cols))
    log.info(f"Features: {n0} in set → −{len(low_var)} low-variance → −{len(to_drop)} |r|>"
             f"{M['correlation_threshold']} → {len(feature_cols)} "
             f"({len(numeric_cols)} numeric + {len(categorical_cols)} categorical)")
    for g, cols in sorted(group_table(feature_cols).items()):
        log.info(f"   {g:<16}{len(cols):>4}")

    # ── 3. Calibration hold-out ─────────────────────────────────────────
    X_all = df_train_all[feature_cols].copy()
    for c in numeric_cols:
        X_all[c] = pd.to_numeric(X_all[c], errors="coerce")
    cal_frac = float(M["calibration_fraction"])
    if scheme == "gene":
        gss = GroupShuffleSplit(n_splits=1, test_size=cal_frac, random_state=SEED)
        fit_idx, cal_idx = next(gss.split(X_all, y_all, groups_all))
    else:
        fit_idx, cal_idx = train_test_split(np.arange(len(X_all)), test_size=cal_frac,
                                            stratify=y_all, random_state=SEED)
    X_fit, y_fit = X_all.iloc[fit_idx].reset_index(drop=True), y_all.iloc[fit_idx].reset_index(drop=True)
    X_cal, y_cal = X_all.iloc[cal_idx].reset_index(drop=True), y_all.iloc[cal_idx].reset_index(drop=True)
    g_fit = groups_all[fit_idx]
    log.info(f"Fit set {X_fit.shape}  |  calibration set {X_cal.shape}  "
             f"(cal positives={int(y_cal.sum())})")

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    xgb_params = dict(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.9,
                      colsample_bytree=0.8, reg_lambda=1.0, objective="binary:logistic",
                      eval_metric="logloss", random_state=SEED, n_jobs=-1)
    lgbm_params = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.9,
                       colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbose=-1)

    # ── 4. 5-fold CV (default params) → OOF ─────────────────────────────
    folds = cv_split(cv_splitter(N_FOLDS, scheme, SEED), X_fit, y_fit, g_fit)
    with open(out / "fold_assignments.json", "w") as f:
        json.dump({f"fold_{i + 1}": {"train_row_ids": df_train_all["_row_id"].iloc[fit_idx[tr]].tolist(),
                                     "val_row_ids": df_train_all["_row_id"].iloc[fit_idx[va]].tolist()}
                   for i, (tr, va) in enumerate(folds)}, f)
    oof_xgb = np.zeros(len(X_fit)); oof_lgbm = np.zeros(len(X_fit)); fold_auc = []
    for i, (tr, va) in enumerate(folds, 1):
        pre = clone(preprocessor)
        Xtr = pre.fit_transform(X_fit.iloc[tr]); Xva = pre.transform(X_fit.iloc[va])
        m1 = XGBClassifier(**xgb_params).fit(Xtr, y_fit.iloc[tr])
        m2 = LGBMClassifier(**lgbm_params).fit(Xtr, y_fit.iloc[tr])
        oof_xgb[va] = m1.predict_proba(Xva)[:, 1]
        oof_lgbm[va] = m2.predict_proba(Xva)[:, 1]
        a1, a2 = roc_auc_score(y_fit.iloc[va], oof_xgb[va]), roc_auc_score(y_fit.iloc[va], oof_lgbm[va])
        fold_auc.append({"fold": i, "xgb": a1, "lgbm": a2, "n_val": len(va),
                         "n_val_genes": int(pd.Series(g_fit[va]).nunique())})
        log.info(f"  fold {i}: XGB={a1:.4f}  LGBM={a2:.4f}  (n_val={len(va)})")
    oof_ens = (oof_xgb + oof_lgbm) / 2
    cv_summary = {"oof_auc_xgb": float(roc_auc_score(y_fit, oof_xgb)),
                  "oof_auc_lgbm": float(roc_auc_score(y_fit, oof_lgbm)),
                  "oof_auc_ensemble": float(roc_auc_score(y_fit, oof_ens)),
                  "fold_auc_mean_xgb": float(np.mean([f["xgb"] for f in fold_auc])),
                  "fold_auc_std_xgb": float(np.std([f["xgb"] for f in fold_auc])),
                  "fold_auc_mean_lgbm": float(np.mean([f["lgbm"] for f in fold_auc])),
                  "fold_auc_std_lgbm": float(np.std([f["lgbm"] for f in fold_auc])),
                  "folds": fold_auc}
    log.info(f"OOF AUC  XGB={cv_summary['oof_auc_xgb']:.4f}  LGBM={cv_summary['oof_auc_lgbm']:.4f}  "
             f"Ensemble={cv_summary['oof_auc_ensemble']:.4f}")

    oof_df = df_train_all[["_row_id", "_variant_id", "_gene", "_panel", TARGET_COL]].copy()
    oof_df["in_fit_set"] = False
    oof_df.loc[fit_idx, "in_fit_set"] = True
    for name, arr in (("oof_xgb", oof_xgb), ("oof_lgbm", oof_lgbm), ("oof_ensemble", oof_ens)):
        oof_df[name] = np.nan
        oof_df.loc[fit_idx, name] = arr
    oof_df.to_csv(out / "oof_track_a.csv", index=False)

    # ── 5. LOOCV on CFTR ─────────────────────────────────────────────────
    loocv = None
    if not args.skip_loocv and "cftr" in trains:
        cf = trains["cftr"]
        Xc = cf.reindex(columns=feature_cols).copy()
        for c in numeric_cols:
            Xc[c] = pd.to_numeric(Xc[c], errors="coerce")
        yc = cf[TARGET_COL].astype(int).values
        probs = np.zeros(len(Xc))
        if len(np.unique(yc)) < 2:
            log.warning("CFTR LOOCV skipped — training panel has a single class")
            yc = None
        for tr, va in (LeaveOneOut().split(Xc) if yc is not None else []):
            pre = clone(preprocessor)
            m = XGBClassifier(**{**xgb_params, "n_estimators": 150}).fit(pre.fit_transform(Xc.iloc[tr]), yc[tr])
            probs[va] = m.predict_proba(pre.transform(Xc.iloc[va]))[:, 1]
        if yc is not None:
            loocv = {"panel": "cftr", "n": int(len(yc)), "auc": float(roc_auc_score(yc, probs))}
            log.info(f"CFTR LOOCV AUC = {loocv['auc']:.4f}  (n={loocv['n']})")

    # ── 6. Optuna ────────────────────────────────────────────────────────
    inner = cv_split(cv_splitter(3, scheme, SEED), X_fit, y_fit, g_fit)

    def objective_factory(kind):
        def objective(trial):
            if kind == "xgb":
                params = dict(n_estimators=trial.suggest_int("n_estimators", 100, 600),
                              max_depth=trial.suggest_int("max_depth", 3, 8),
                              learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                              subsample=trial.suggest_float("subsample", 0.6, 1.0),
                              colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                              reg_lambda=trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
                              reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
                              min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
                              objective="binary:logistic", eval_metric="logloss",
                              random_state=SEED, n_jobs=-1)
                Model = XGBClassifier
            else:
                params = dict(n_estimators=trial.suggest_int("n_estimators", 100, 600),
                              num_leaves=trial.suggest_int("num_leaves", 15, 127),
                              learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                              subsample=trial.suggest_float("subsample", 0.6, 1.0),
                              colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                              reg_lambda=trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
                              reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
                              min_child_samples=trial.suggest_int("min_child_samples", 5, 50),
                              random_state=SEED, n_jobs=-1, verbose=-1)
                Model = LGBMClassifier
            aucs = []
            for tr, va in inner:
                pre = clone(preprocessor)
                m = Model(**params).fit(pre.fit_transform(X_fit.iloc[tr]), y_fit.iloc[tr])
                aucs.append(roc_auc_score(y_fit.iloc[va], m.predict_proba(pre.transform(X_fit.iloc[va]))[:, 1]))
            return float(np.mean(aucs))
        return objective

    studies = {}
    for kind in ("xgb", "lgbm"):
        st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
        st.optimize(objective_factory(kind), n_trials=N_TRIALS, show_progress_bar=False)
        studies[kind] = st
        st.trials_dataframe().to_csv(out / f"optuna_trials_{kind}.csv", index=False)
        log.info(f"Optuna {kind}: best inner-CV AUC={st.best_value:.4f} (trial {st.best_trial.number})")
    best_xgb = {**studies["xgb"].best_params, "objective": "binary:logistic", "eval_metric": "logloss",
                "random_state": SEED, "n_jobs": -1}
    best_lgbm = {**studies["lgbm"].best_params, "random_state": SEED, "n_jobs": -1, "verbose": -1}

    # Optuna figures
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, kind, name, base in ((axes[0], "xgb", "XGBoost", cv_summary["oof_auc_xgb"]),
                                 (axes[1], "lgbm", "LightGBM", cv_summary["oof_auc_lgbm"])):
        vals = [t.value for t in studies[kind].trials if t.value is not None]
        ax.scatter(range(len(vals)), vals, s=12, alpha=0.4, label="trial")
        ax.plot(np.maximum.accumulate(vals), lw=2, label=f"best {max(vals):.4f}")
        ax.axhline(base, ls="--", color="red", lw=1.2, label=f"default {base:.4f}")
        ax.set(xlabel="trial", ylabel="inner-CV ROC-AUC", title=f"{name} — Optuna history")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    savefig(fig, fig_dir / "optuna_history.png")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    hp_importance = {}
    for ax, kind, name in ((axes[0], "xgb", "XGBoost"), (axes[1], "lgbm", "LightGBM")):
        try:
            imp = optuna.importance.get_param_importances(studies[kind])
            hp_importance[kind] = imp
            ks, vs = zip(*sorted(imp.items(), key=lambda x: x[1]))
            ax.barh(ks, vs); ax.set(title=f"{name} — hyper-parameter importance (fANOVA)")
        except Exception as e:  # pragma: no cover
            ax.text(0.5, 0.5, f"unavailable: {e}", ha="center", transform=ax.transAxes)
    savefig(fig, fig_dir / "optuna_param_importance.png")

    # ── 7. Final fit + calibration ──────────────────────────────────────
    pre_final = clone(preprocessor)
    Xf = pre_final.fit_transform(X_fit); Xc_ = pre_final.transform(X_cal)
    xgb_final = XGBClassifier(**best_xgb).fit(Xf, y_fit)
    lgbm_final = LGBMClassifier(**best_lgbm).fit(Xf, y_fit)
    ens_cal = (xgb_final.predict_proba(Xc_)[:, 1] + lgbm_final.predict_proba(Xc_)[:, 1]) / 2
    iso = IsotonicRegression(out_of_bounds="clip").fit(ens_cal, y_cal)
    cal_proba = iso.predict(ens_cal)
    log.info(f"Calibration set AUC  pre={roc_auc_score(y_cal, ens_cal):.4f}  post={roc_auc_score(y_cal, cal_proba):.4f}")
    pt, pp = calibration_curve(y_cal, cal_proba, n_bins=5)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(pp, pt, "o-", label="ensemble (isotonic)"); ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(xlabel="predicted probability", ylabel="observed frequency", title="Calibration (hold-out)")
    ax.legend(); ax.grid(alpha=0.3)
    savefig(fig, fig_dir / "calibration_curve.png")

    # ── 8. Thresholds ────────────────────────────────────────────────────
    min_sens = float(M["min_sensitivity"])
    g_y, g_s = best_threshold_youden(y_cal, cal_proba), best_threshold_sensitivity(y_cal, cal_proba, min_sens)
    cal_panel = df_train_all["_panel"].iloc[cal_idx].values
    thr_y, thr_s, thr_n = {}, {}, {}
    for p in PANELS:
        m = cal_panel == p
        thr_n[p] = int(m.sum())
        if m.sum() >= 30 and y_cal[m].nunique() == 2:
            thr_y[p] = best_threshold_youden(y_cal[m], cal_proba[m])
            thr_s[p] = best_threshold_sensitivity(y_cal[m], cal_proba[m], min_sens)
        else:
            thr_y[p], thr_s[p] = g_y, g_s
        log.info(f"  threshold {p:<8} Youden={thr_y[p]:.2f}  sens≥{min_sens:.2f}={thr_s[p]:.2f}  (n_cal={thr_n[p]})")

    def predict(df, raw=False):
        X = df.reindex(columns=feature_cols).copy()
        for c in numeric_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        Xp = pre_final.transform(X)
        ens = (xgb_final.predict_proba(Xp)[:, 1] + lgbm_final.predict_proba(Xp)[:, 1]) / 2
        return ens if raw else iso.predict(ens)

    # ── 9. Test evaluation ───────────────────────────────────────────────
    results, rows, probas = {}, [], {}
    for p in PANELS:
        te = tests[p]
        y_te = te[TARGET_COL].astype(int).values
        pr = predict(te)
        pd.DataFrame({"_variant_id": te["_variant_id"], "_gene": te["_gene"], "_panel": p,
                      TARGET_COL: y_te, "P_pathogenic": pr, "P_raw_ensemble": predict(te, raw=True),
                      "pred_youden": (pr >= thr_y[p]).astype(int),
                      "pred_sens90": (pr >= thr_s[p]).astype(int)}).to_csv(out / f"predictions_{p}.csv", index=False)
        if len(np.unique(y_te)) < 2 or len(y_te) < 10:
            log.warning(f"TEST {p:<8} skipped — single class or < 10 variants (n={len(y_te)})")
            results[p] = {"skipped": True, "n": int(len(y_te))}
            continue
        probas[p] = (y_te, pr)
        m_y = evaluate(y_te, pr, thr_y[p]); m_s = evaluate(y_te, pr, thr_s[p])
        ci = bootstrap_ci(y_te, pr, thr_y[p], n_boot=N_BOOT, seed=SEED)
        results[p] = {"youden": m_y, "sensitivity_priority": m_s, "ci_youden": ci,
                      "n_test_genes": int(te["_gene"].nunique())}
        rows.append({"panel": p, "n": len(y_te), "n_genes": results[p]["n_test_genes"],
                     "roc_auc": m_y["roc_auc"], "roc_auc_lo": ci["roc_auc_ci"][0], "roc_auc_hi": ci["roc_auc_ci"][1],
                     "pr_auc": m_y["pr_auc"], "f1_youden": m_y["f1"], "sens_youden": m_y["sensitivity"],
                     "spec_youden": m_y["specificity"], "mcc_youden": m_y["mcc"], "brier": m_y["brier"],
                     "thr_youden": thr_y[p], "f1_sens90": m_s["f1"], "sens_sens90": m_s["sensitivity"],
                     "spec_sens90": m_s["specificity"], "thr_sens90": thr_s[p]})
        log.info(f"TEST {p:<8} AUC={m_y['roc_auc']:.4f} ({ci['roc_auc_ci'][0]:.3f}–{ci['roc_auc_ci'][1]:.3f})  "
                 f"F1={m_y['f1']:.3f}  Sens={m_y['sensitivity']:.3f}  Spec={m_y['specificity']:.3f}")
    pd.DataFrame(rows).to_csv(out / "results_table.csv", index=False)
    EVAL_PANELS = [p for p in PANELS if p in probas]

    # ── 10. Figures ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(EVAL_PANELS), figsize=(4.6 * len(EVAL_PANELS), 4.2))
    for ax, p in zip(np.atleast_1d(axes), EVAL_PANELS):
        y_te, pr = probas[p]
        ConfusionMatrixDisplay(confusion_matrix(y_te, (pr >= thr_y[p]).astype(int)),
                               display_labels=["Benign", "Pathogenic"]).plot(cmap="Blues", ax=ax, colorbar=False)
        ax.set_title(f"{p}  AUC={roc_auc_score(y_te, pr):.3f}  t={thr_y[p]:.2f}")
    savefig(fig, fig_dir / "confusion_matrices.png")
    for kind in ("roc", "pr"):
        fig, ax = plt.subplots(figsize=(5.5, 5))
        for p in EVAL_PANELS:
            y_te, pr = probas[p]
            if kind == "roc":
                x, yv, _ = roc_curve(y_te, pr); lab = f"{p} (AUC={roc_auc_score(y_te, pr):.3f})"
            else:
                yv, x, _ = precision_recall_curve(y_te, pr); lab = f"{p} (AP={average_precision_score(y_te, pr):.3f})"
            ax.plot(x, yv, label=lab)
        if kind == "roc":
            ax.plot([0, 1], [0, 1], "--", color="gray"); ax.set(xlabel="FPR", ylabel="TPR", title="ROC — test panels")
        else:
            ax.set(xlabel="Recall", ylabel="Precision", title="Precision–recall — test panels")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        savefig(fig, fig_dir / f"{kind}_curves.png")

    # ── 11. Error analysis + SHAP (general panel) ───────────────────────
    if "general" not in probas:
        sys.exit("general panel could not be evaluated — aborting before SHAP")
    gen = tests["general"].copy()
    y_g, pr_g = probas["general"]
    pred_g = (pr_g >= thr_y["general"]).astype(int)
    gen["P_pathogenic"] = pr_g
    gen["error_type"] = np.where(pred_g == y_g, "correct", np.where(pred_g == 1, "FP", "FN"))
    err = {"counts": gen["error_type"].value_counts().to_dict(), "by_error_type": {}}
    for col, label in (("maf_log10", "log10(MAF)"), ("silico_variance", "in-silico tool variance"),
                       ("enrich_grantham_dist", "Grantham distance"), ("phylop_primate", "phyloP (primate)")):
        if col in gen.columns:
            err["by_error_type"][label] = gen.groupby("error_type")[col].apply(
                lambda s: float(pd.to_numeric(s, errors="coerce").mean())).to_dict()
    with open(out / "error_analysis.json", "w") as f:
        json.dump(err, f, indent=2)

    shap_summary = None
    if not args.no_shap:
        import shap
        Xg = gen.reindex(columns=feature_cols).copy()
        for c in numeric_cols:
            Xg[c] = pd.to_numeric(Xg[c], errors="coerce")
        Xgp = pre_final.transform(Xg)
        names = feature_names_out(pre_final)
        sv = np.array(shap.TreeExplainer(xgb_final).shap_values(Xgp))
        mean_abs = np.abs(sv).mean(axis=0)
        top = np.argsort(mean_abs)[::-1][: int(M["shap_max_display"])]
        plt.figure(figsize=(9, 7))
        shap.summary_plot(sv[:, top], Xgp[:, top], feature_names=[names[i] for i in top], show=False, plot_type="dot")
        plt.title("SHAP — XGBoost, general test panel"); plt.tight_layout()
        plt.savefig(fig_dir / "shap_beeswarm.png", dpi=300, bbox_inches="tight"); plt.close()
        # group importance uses the schema, so it is comparable across runs
        def grp(n):
            base = n
            for c in categorical_cols:          # one-hot names: ref_aa_A → ref_aa
                if n.startswith(c + "_"):
                    base = c
            return feature_group(base) or "other"
        gimp = {}
        for i, n in enumerate(names):
            gimp[grp(n)] = gimp.get(grp(n), 0.0) + float(mean_abs[i])
        gdf = pd.DataFrame(sorted(gimp.items(), key=lambda x: x[1]), columns=["group", "sum_mean_abs_shap"])
        gdf.to_csv(out / "shap_group_importance.csv", index=False)
        fig, ax = plt.subplots(figsize=(7, 3.8))
        ax.barh(gdf["group"], gdf["sum_mean_abs_shap"]); ax.set(xlabel="Σ mean |SHAP|", title="Feature-group importance")
        savefig(fig, fig_dir / "shap_group_importance.png")
        pd.DataFrame({"feature": names, "mean_abs_shap": mean_abs}).sort_values(
            "mean_abs_shap", ascending=False).to_csv(out / "shap_feature_importance.csv", index=False)
        shap_summary = {"top_features": [names[i] for i in top], "group_importance": gimp}
        # case studies: strongest TP and worst FN
        try:
            ev = np.ravel(shap.TreeExplainer(xgb_final).expected_value)[-1]
            exp = shap.Explanation(values=sv, base_values=np.full(len(sv), float(ev)),
                                   data=Xgp, feature_names=names)
            tp = int(np.argmax(np.where((y_g == 1) & (pred_g == 1), pr_g, -1)))
            fn_candidates = np.where((y_g == 1) & (pred_g == 0))[0]
            for tag, idx in (("TP", tp), ("FN", int(fn_candidates[np.argmin(pr_g[fn_candidates])]) if len(fn_candidates) else None)):
                if idx is None:
                    continue
                plt.figure(figsize=(8, 6)); shap.waterfall_plot(exp[idx], max_display=15, show=False)
                plt.title(f"SHAP waterfall — {tag} case (P={pr_g[idx]:.3f}, {gen['_gene'].iloc[idx]})")
                plt.tight_layout(); plt.savefig(fig_dir / f"shap_waterfall_{tag}.png", dpi=300, bbox_inches="tight"); plt.close()
        except Exception as e:  # pragma: no cover
            log.warning(f"waterfall plots skipped: {e}")

    # ── 12. Artefacts ────────────────────────────────────────────────────
    with open(out / "model_artefacts.pkl", "wb") as f:
        pickle.dump({"xgb_final": xgb_final, "lgbm_final": lgbm_final, "preprocessor": pre_final,
                     "isotonic": iso, "feature_cols": feature_cols, "numeric_cols": numeric_cols,
                     "categorical_cols": categorical_cols, "thresholds_youden": thr_y,
                     "thresholds_sens": thr_s, "best_xgb_params": best_xgb, "best_lgbm_params": best_lgbm}, f)
    with open(out / "feature_cols.json", "w") as f:
        json.dump({"feature_cols": feature_cols, "dropped_low_variance": low_var,
                   "dropped_correlated": to_drop, "groups": group_table(feature_cols)}, f, indent=2)
    metrics = {"scheme": scheme, "feature_set": fset, "seed": SEED, "n_trials": N_TRIALS,
               "n_train_fit": int(len(X_fit)), "n_calibration": int(len(X_cal)),
               "n_features": len(feature_cols), "cv": cv_summary, "loocv_cftr": loocv,
               "optuna_best": {"xgb": {"value": studies["xgb"].best_value, "params": studies["xgb"].best_params},
                               "lgbm": {"value": studies["lgbm"].best_value, "params": studies["lgbm"].best_params}},
               "hyperparameter_importance": hp_importance,
               "calibration": {"auc_pre": float(roc_auc_score(y_cal, ens_cal)),
                               "auc_post": float(roc_auc_score(y_cal, cal_proba)),
                               "global_threshold_youden": g_y, "global_threshold_sens": g_s,
                               "panel_thresholds_youden": thr_y, "panel_thresholds_sens": thr_s,
                               "panel_n_cal": thr_n},
               "test": results, "shap": shap_summary, "runtime_sec": round(time.time() - t0, 1)}
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    with open(out / "run_config.json", "w") as f:
        json.dump({"args": vars(args), "config": cfg}, f, indent=2)
    with open(out / "environment.json", "w") as f:
        json.dump(environment_info(), f, indent=2)
    log.info(f"\n✓ Track A done in {metrics['runtime_sec']}s → {out}")


if __name__ == "__main__":
    main()
