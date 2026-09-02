# Output guide — what every file in `results/` means and how to read it

```
results/
├── <scheme>_<feature_set>/          one per cell of the experimental matrix
│   ├── metrics.json                 everything numeric for this run (see §1)
│   ├── results_table.csv            one row per test panel (see §2)
│   ├── predictions_<panel>.csv      per-variant probabilities and calls
│   ├── oof_track_a.csv              out-of-fold predictions on the training pool
│   ├── fold_assignments.json        CV membership by _row_id
│   ├── feature_cols.json            retained features + drop lists + groups
│   ├── optuna_trials_{xgb,lgbm}.csv every trial with parameters and score
│   ├── shap_feature_importance.csv  mean |SHAP| per transformed feature
│   ├── shap_group_importance.csv    Σ mean |SHAP| per schema group
│   ├── error_analysis.json          FP/FN/correct means of MAF, tool variance, Grantham, phyloP
│   ├── model_artefacts.pkl          models, preprocessor, isotonic map, thresholds
│   ├── run_config.json, environment.json
│   ├── figures/                     png + pdf, 300 dpi
│   └── meta/                        stacking outputs (see §3)
├── <scheme>_baselines/              published predictors (see §4)
└── paper/                           tables 1–6, figures, summary.md, provenance.json (see §5)
```

## 1. `metrics.json`

| key | content |
|---|---|
| `scheme`, `feature_set`, `seed`, `n_trials` | run identity |
| `n_train_fit`, `n_calibration`, `n_features` | sizes after selection |
| `cv.oof_auc_{xgb,lgbm,ensemble}` | OOF ROC-AUC on the fit set |
| `cv.fold_auc_mean_*`, `cv.fold_auc_std_*`, `cv.folds[]` | per-fold AUCs, n validation variants and genes |
| `loocv_cftr` | leave-one-out AUC on the CFTR training panel (null if single-class) |
| `optuna_best.{xgb,lgbm}` | best inner-CV AUC and parameters |
| `hyperparameter_importance` | fANOVA importances |
| `calibration.auc_pre/auc_post` | hold-out AUC before/after isotonic regression |
| `calibration.panel_thresholds_{youden,sens}`, `panel_n_cal` | operating points and the number of calibration variants that determined them |
| `test.<panel>.youden`, `.sensitivity_priority` | metric dicts at each operating point |
| `test.<panel>.ci_youden` | bootstrap CIs |
| `test.<panel>.skipped` | true when a panel could not be evaluated (single class) |
| `shap.top_features`, `shap.group_importance` | interpretability summary |
| `runtime_sec` | wall-clock time |

## 2. `results_table.csv`

Columns: `panel, n, n_genes, roc_auc, roc_auc_lo, roc_auc_hi, pr_auc,
f1_youden, sens_youden, spec_youden, mcc_youden, brier, thr_youden,
f1_sens90, sens_sens90, spec_sens90, thr_sens90`. Report `roc_auc
(roc_auc_lo–roc_auc_hi)` as the headline number; report both operating
points because the sensitivity-priority point is the clinically relevant
one.

## 3. `meta/`

| file | content |
|---|---|
| `meta_results.json` | CV AUC of A and A + C, DeLong and bootstrap on the CV difference, coefficients (`w_track_a_logit`, `w_track_c_logit`, `C`), per-panel test results |
| `meta_ablation_table.csv` | `panel, n, auc_track_a, auc_track_c, auc_meta, delta_meta_minus_a, delong_p, delta_ci_lo, delta_ci_hi` |
| `predictions_meta_<panel>.csv` | per-variant stacked probabilities |
| `meta_learner.pkl` | fitted logistic regressions |

Interpretation: if `w_track_c_logit` ≈ 0 and `delta_meta_minus_a` has a CI
spanning 0, Track C adds no independent signal; state it.

## 4. `<scheme>_baselines/`

| file | content |
|---|---|
| `baselines_table.csv` | `panel, baseline, column, n_scored, coverage, roc_auc, roc_auc_lo, roc_auc_hi` |
| `comparisons_table.csv` | `panel, baseline, model, n_common, auc_model, auc_baseline, delta, delong_p, delta_ci_lo, delta_ci_hi` |
| `figures/baseline_auc_<panel>.png` | horizontal bar chart, NeuroVariant models in blue |

`coverage` < 1 means the predictor had no score for some variants; the
comparison is restricted to the common subset (`n_common`).

## 5. `paper/`

`summary.md` concatenates all tables. `provenance.json` holds, per run,
`environment.json` and `run_config.json`, plus every
`data/raw/provenance_*.json`. Cite the ClinVar `source_version`
(Last-Modified) and gnomAD release from there.

### Reading Table 3 (circularity)

* Rows *type-1*: `auc_a` = full, `auc_b` = core on the **same** test
  variants; `delta` > 0 with `p_delong` < 0.05 quantifies how much
  published-predictor features inflate performance.
* Rows *type-2*: `auc_a` = variant-level, `auc_b` = gene-disjoint;
  independent test sets, so only the conservative interval is given.
  A positive delta whose interval excludes 0 indicates gene-identity
  leakage under the variant-level protocol.

### Figures with stable names

`<scheme>_<fset>_{roc_curves, pr_curves, confusion_matrices,
calibration_curve, optuna_history, optuna_param_importance,
shap_beeswarm, shap_group_importance, shap_waterfall_TP,
shap_waterfall_FN}.png` and `baseline_auc_<panel>.png`.
