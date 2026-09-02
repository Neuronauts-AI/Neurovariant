# Changelog

## 1.0.0 — 2026-09-02 — preprint pipeline

First release of the merged, script-only pipeline (data builder 01–05 from
`teknofest2026-pipeline`, model stage 06–11 from `PSR_Pipeline.ipynb` and
the Dataset3 scripts).

### Added
* `scripts/utils/schema.py` — feature groups and feature sets (`core`, `full`), baseline definitions, in-silico thresholds.
* `scripts/utils/metrics.py` — bootstrap CIs, Youden / sensitivity-priority thresholds, fast DeLong test, paired bootstrap of ΔAUC.
* `scripts/utils/config.py` and `config/pipeline.yaml` — single configuration source.
* `scripts/08_train_track_a.py` — script form of the notebook: grouped CV / grouped calibration hold-out / grouped Optuna under the gene scheme, `_row_id`-aligned OOF file, `environment.json`, `run_config.json`, stable figure names.
* `scripts/10_baselines.py` — REVEL, AlphaMissense, CADD, PolyPhen-2, SIFT, mean-of-tools evaluated on the same variants with DeLong tests.
* `scripts/11_paper_tables.py` — Tables 1–6, figures, `summary.md`, `provenance.json`.
* `run_pipeline.py` — orchestrator with `--fast`, `--schemes`, `--feature-sets`, `--skip-data-steps`.
* `tests/make_smoke_fixture.py`, `config/smoke.yaml` — offline smoke test.
* `docs/METHODS.md`, `docs/DATA_DICTIONARY.md`, `docs/OUTPUTS.md`, `docs/reporting_checklist.md`.

### Changed
* `03_build_panels.py` — rewritten: general panel excludes all specialty genes; variants assigned to both a variant-level and a gene-disjoint split; single-gene panels flagged "within-gene"; cross-panel disjointness asserted; `split_manifest.json`.
* `04*` — operate on one pool file per panel (`panel_<p>_pool_features.csv`); split columns passed through.
* `05_finalize_dataset.py` — rewritten: no anonymisation; drops rows without a parsed protein change, all-NaN and zero-variance columns, and test rows identical to training rows; recomputes in-silico summaries; writes `data/final/<scheme>/`.
* `06_enrich_features.py` — `--scheme`/`--config` aware paths; otherwise unchanged.
* `07_track_c_acmg_prior.py` — rewritten on named columns; evidence items without an ACMG counterpart renamed (CONS_HIGH, GENE_INTOL, RADICAL_SUB, CHARGE_REV, CONSERV_SUB); PP3/BP4 disabled under `core`.
* `09_meta_learner.py` — inputs are logits; grouped CV ablation; DeLong + paired bootstrap on training and test; `C` chosen by CV.

### Removed
* `06_fingerprint.py` and all `col_XXXX` handling.
* `03b_fix_single_gene_panels.py`, `03c_fix_pah_panel.py` (gnomAD AF > 1e-6 labelled benign).
* `06_train_model.py` of the data repository (superseded by 08/09).
* Diagnostic scripts (`diagnose_*`, `inspect_features`, `preview_data`, `check_alphamissense`, `generate_feature_report*`, `validate_dataset`) — kept in the original repository's history.

### Data-quality findings in the legacy Dataset3 (documented for the manuscript)
* 100 % of PAH and CFTR benign rows (train and test) lacked `ref_aa`, `alt_aa`, `aa_position` and every score — empty feature vectors that made the benign class trivially separable.
* 5–6 % of general-panel benign rows had the same defect.
* 14 / 12 / 10 exact duplicate feature vectors between train and test in the general / PAH / CFTR panels.
* The general panel shared genes (BRCA1/2, …) with the cancer panel.
* The variant-level split placed the same genes on both sides in every multi-gene panel.
