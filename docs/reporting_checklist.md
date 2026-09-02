# DOME reporting checklist → pipeline outputs

Walsh et al., *DOME: recommendations for supervised machine learning validation in biology*, Nat Methods 2021.

## Data
| question | where the answer comes from |
|---|---|
| Source, number of points per class | `results/paper/table1_dataset.csv`; `data/raw/provenance_*.json` (ClinVar release, filters) |
| How train/test were divided; separate validation set | `03_build_panels.py` (gene-disjoint + variant), 15 % calibration hold-out in `08` (`metrics.json → n_calibration`) |
| Independence of train and test | gene-disjoint scheme; `05` removes identical feature vectors; `split_manifest.json` lists test genes |
| Data and splits public | deposit `data/final/` + `fold_assignments.json` on Zenodo |

## Optimization
| question | where |
|---|---|
| Algorithm class and why | XGBoost + LightGBM ensemble (tabular, missing-value native), LR stacking |
| Uses outputs of other ML? | only in the **full** feature set (REVEL, CADD…) — reported separately from **core** |
| Encoding / preprocessing | median imputation, one-hot for 4 categorical columns (`08`, `feature_cols.json`) |
| Parameters, selection | Optuna TPE, 200 trials × inner 3-fold (grouped) — `optuna_trials_*.csv`, `metrics.json → optuna_best` |
| Feature count, selection on train only | variance + correlation filters on training data (`feature_cols.json`, `table6_features`) |
| Over/under-fitting | fold AUC spread (`metrics.json → cv`), calibration hold-out, isotonic calibration, Brier score |
| Regularisation | reg_lambda / reg_alpha / min_child_* searched; L2 in the meta-learner |
| Hyper-parameters and model files public | `run_config.json`, `model_artefacts.pkl`, `environment.json` |

## Model
| question | where |
|---|---|
| Interpretable / black box | SHAP beeswarm, group importance, TP/FN waterfalls; Track C is fully transparent (`track_c_rules`) |
| Classification / regression | binary classification, calibrated probability + two thresholds |
| Prediction time | tabular GBMs — milliseconds per variant |
| Code released | this repository (tag the commit used for the manuscript) |

## Evaluation
| question | where |
|---|---|
| Evaluation method | grouped 5-fold CV + independent gene-disjoint test panels + LOOCV (CFTR) |
| Metrics | ROC-AUC, PR-AUC, F1, sensitivity, specificity, MCC, Brier — `results_table.csv` |
| Comparison to published methods and simpler baselines | `10_baselines.py` (REVEL, AlphaMissense, CADD, PolyPhen-2, SIFT, mean-of-tools); Track C alone; Track A alone vs stacked |
| Confidence intervals and significance | 2000× bootstrap CIs, DeLong tests, paired bootstrap on AUC differences |
| Raw evaluation files public | `predictions_*.csv`, `oof_track_a.csv` |

## Limitations the manuscript must state
* PAH and CFTR panels are single-gene: no gene-disjoint evaluation is possible (labelled "within-gene").
* Published predictors were trained on ClinVar: their standalone AUCs are optimistic (type-1 circularity); a time-split test set (variants deposited after the predictors' training cut-offs) would be the stricter comparison and is not implemented here.
* Benign labels partly derive from gnomAD common variants (AF > 0.001), not from clinical assertion.
* ClinVar review-status filter (3–4 star) biases towards well-studied genes.
