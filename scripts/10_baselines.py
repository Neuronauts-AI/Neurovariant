"""
STEP 10 — Published predictors as standalone baselines
=======================================================

For every test panel of a split scheme, the AUC of each published score
(REVEL, AlphaMissense, CADD, PolyPhen-2, SIFT, mean-of-tools) is computed on
the variants for which the score is available, and compared head-to-head
(same variants, DeLong test + paired bootstrap) with the NeuroVariant models
trained under the same scheme (core and full feature sets, Track A and the
meta-learner when present).

Caveat that must be stated in the paper: the published predictors were
trained on ClinVar/HGMD, so their AUCs on ClinVar-derived test variants are
optimistic (type-1 circularity). The core model has never seen these scores,
so a core-model AUC that matches them is a conservative result.

Outputs  results/<scheme>_baselines/
    baselines_table.csv     one row per (panel, baseline)
    comparisons_table.csv   one row per (panel, baseline, our model)
    baselines.json, figures/baseline_auc_<panel>.png
"""

from __future__ import annotations

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from utils.logging_utils import get_logger
from utils.config import load_config, run_dir
from utils.schema import BASELINE_SCORES, TARGET_COL
from utils.metrics import bootstrap_ci, delong_roc_test, paired_bootstrap_auc_diff

log = get_logger("10_baselines.log")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/pipeline.yaml")
    ap.add_argument("--scheme", default="gene", choices=["gene", "variant"])
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    SEED = int(cfg["seed"]); PANELS = cfg["panels"]
    N_BOOT = 200 if args.fast else int(cfg["model"]["n_bootstrap"])
    final_dir = os.path.join(cfg["paths"]["final"], args.scheme)
    out = os.path.join(cfg["paths"]["results"], f"{args.scheme}_baselines")
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)

    # our models available for this scheme
    ours = {}
    for fset in cfg["feature_sets"]:
        rd = run_dir(cfg, args.scheme, fset)
        if os.path.exists(os.path.join(rd, "predictions_general.csv")):
            ours[f"NeuroVariant-{fset} (Track A)"] = (rd, "predictions_{p}.csv", "P_pathogenic")
        if os.path.exists(os.path.join(rd, "meta", "predictions_meta_general.csv")):
            ours[f"NeuroVariant-{fset} (A+C meta)"] = (os.path.join(rd, "meta"), "predictions_meta_{p}.csv", "P_meta")
    log.info(f"Baselines  scheme={args.scheme}  our models: {list(ours)}")

    base_rows, cmp_rows, blob = [], [], {}
    for p in PANELS:
        te = pd.read_csv(os.path.join(final_dir, f"test_{p}.csv"), low_memory=False)
        te["_variant_id"] = te["_variant_id"].astype(str)
        y_all = te[TARGET_COL].astype(int).values
        preds = {}
        for name, (d, pat, col) in ours.items():
            fp = os.path.join(d, pat.format(p=p))
            if not os.path.exists(fp):
                continue
            pr = pd.read_csv(fp)
            pr["_variant_id"] = pr["_variant_id"].astype(str)
            preds[name] = te[["_variant_id"]].merge(pr[["_variant_id", col]], on="_variant_id", how="left")[col].values
        blob[p] = {}
        aucs_for_fig = {}
        for col, (disp, sign) in BASELINE_SCORES.items():
            if col not in te.columns:
                continue
            s = pd.to_numeric(te[col], errors="coerce").values * sign
            m = ~np.isnan(s)
            cov = float(m.mean())
            if m.sum() < 20 or len(np.unique(y_all[m])) < 2:
                log.info(f"  {p:<8} {disp:<16} coverage={cov:.1%} — too few rows, skipped")
                continue
            auc = float(roc_auc_score(y_all[m], s[m]))
            ci = bootstrap_ci(y_all[m], s[m], threshold=np.nanmedian(s[m]), n_boot=N_BOOT, seed=SEED)["roc_auc_ci"]
            base_rows.append({"panel": p, "baseline": disp, "column": col, "n_scored": int(m.sum()),
                              "coverage": cov, "roc_auc": auc, "roc_auc_lo": ci[0], "roc_auc_hi": ci[1]})
            aucs_for_fig[disp] = (auc, ci)
            blob[p][disp] = {"auc": auc, "ci": ci, "coverage": cov, "vs_ours": {}}
            log.info(f"  {p:<8} {disp:<16} AUC={auc:.4f} ({ci[0]:.3f}–{ci[1]:.3f})  coverage={cov:.1%}")
            for name, ours_p in preds.items():
                mm = m & ~np.isnan(ours_p)
                if mm.sum() < 20 or len(np.unique(y_all[mm])) < 2:
                    continue
                dl = delong_roc_test(y_all[mm], ours_p[mm], s[mm])
                bs = paired_bootstrap_auc_diff(y_all[mm], ours_p[mm], s[mm], n_boot=N_BOOT, seed=SEED)
                cmp_rows.append({"panel": p, "baseline": disp, "model": name, "n_common": int(mm.sum()),
                                 "auc_model": dl["auc1"], "auc_baseline": dl["auc2"], "delta": dl["delta"],
                                 "delong_p": dl["p"], "delta_ci_lo": bs["ci"][0], "delta_ci_hi": bs["ci"][1]})
                blob[p][disp]["vs_ours"][name] = {**dl, "bootstrap": bs}
        for name, ours_p in preds.items():
            mm = ~np.isnan(ours_p)
            if mm.sum() >= 20 and len(np.unique(y_all[mm])) == 2:
                a = float(roc_auc_score(y_all[mm], ours_p[mm]))
                ci = bootstrap_ci(y_all[mm], ours_p[mm], 0.5, n_boot=N_BOOT, seed=SEED)["roc_auc_ci"]
                aucs_for_fig[name] = (a, ci)
        if aucs_for_fig:
            names = list(aucs_for_fig); vals = [aucs_for_fig[n][0] for n in names]
            err = np.array([[v - aucs_for_fig[n][1][0], aucs_for_fig[n][1][1] - v] for n, v in zip(names, vals)]).T
            fig, ax = plt.subplots(figsize=(7, 0.45 * len(names) + 1.5))
            colors = ["tab:blue" if n.startswith("NeuroVariant") else "tab:gray" for n in names]
            ax.barh(names, vals, xerr=err, color=colors, capsize=3)
            ax.set(xlim=(0.5, 1.0), xlabel="ROC-AUC (95 % bootstrap CI)", title=f"{p} panel — {args.scheme} split")
            ax.grid(alpha=0.3, axis="x"); fig.tight_layout()
            fig.savefig(os.path.join(out, "figures", f"baseline_auc_{p}.png"), dpi=300); plt.close(fig)

    pd.DataFrame(base_rows).to_csv(os.path.join(out, "baselines_table.csv"), index=False)
    pd.DataFrame(cmp_rows).to_csv(os.path.join(out, "comparisons_table.csv"), index=False)
    with open(os.path.join(out, "baselines.json"), "w") as f:
        json.dump(blob, f, indent=2, default=float)
    log.info(f"✓ baselines → {out}")


if __name__ == "__main__":
    main()
