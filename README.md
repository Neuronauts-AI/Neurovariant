# NeuroVariant — leakage-controlled missense pathogenicity classification

Reproducible pipeline behind the NeuroVariant preprint
(AUGM Neuronauts AI). It classifies single-nucleotide missense variants as
pathogenic or benign and — unlike most published benchmarks — quantifies
the two circularity effects described by Grimm *et al.* (2015) as part of
its own results.

* **Methods, in Methods-section detail:** [`docs/METHODS.md`](docs/METHODS.md)
* **Every column explained:** [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md)
* **Every output file explained:** [`docs/OUTPUTS.md`](docs/OUTPUTS.md)
* **DOME reporting checklist mapped to outputs:** [`docs/reporting_checklist.md`](docs/reporting_checklist.md)
* **History and provenance of the code:** [`CHANGELOG.md`](CHANGELOG.md)

---

## Contents

1. [What the pipeline does](#1-what-the-pipeline-does)
2. [The experimental matrix](#2-the-experimental-matrix)
3. [Repository layout](#3-repository-layout)
4. [Installation](#4-installation)
5. [Running the pipeline](#5-running-the-pipeline)
6. [Configuration](#6-configuration)
7. [Outputs](#7-outputs)
8. [Smoke test without network access](#8-smoke-test-without-network-access)
9. [Testing](#9-testing)
10. [Provenance of the code](#10-provenance-of-the-code)
11. [How to report results](#11-how-to-report-results)
12. [Data availability and citation](#12-data-availability-and-citation)

---

## 1. What the pipeline does

```
ClinVar 3–4★ ──┐
2★ benign ─────┤  01/01b/02        03              04–04d            05
gnomAD AF>1e-3 ┘  fetch + label → 4 panels,   →  6 feature groups → named, cleaned
                                  2 split schemes  (+ context, scores)  train/test CSVs
                                                                            │
     ┌──────────────────────────────────────────────────────────────────────┘
     │ 06 enrichment (149 sequence-derived features)
     │ 07 Track C  — transparent rule prior
     │ 08 Track A  — XGBoost + LightGBM, Optuna, isotonic calibration, SHAP
     │ 09 stacking meta-learner + ablation (A vs A+C)
     │ 10 published predictors head-to-head (REVEL, AlphaMissense, CADD, …)
     └ 11 tables, figures, provenance for the manuscript
```

Four disease panels are built: **general** (genome-wide, excluding all
specialty genes), **cancer** (33 hereditary-cancer genes), **PAH** and
**CFTR** (single-gene). Panels are variant- and gene-disjoint from each
other.

## 2. The experimental matrix

| | `core` features | `full` features |
|---|---|---|
| **gene-disjoint split** | **primary result** — no published-predictor features, no gene on both sides | type-1 control |
| **variant-level split** | type-2 control | classical (leaky) protocol |

* *core* excludes REVEL, CADD, PolyPhen-2, SIFT, AlphaMissense and their
  aggregates; *full* includes them.
* Under the gene-disjoint scheme, cross-validation, the calibration
  hold-out and Optuna's inner folds are all grouped by gene.
* PAH and CFTR are single-gene panels: their "gene" split is a
  within-gene split and is labelled as such in every table.

The difference **full − core** on identical test variants (paired DeLong)
measures type-1 circularity; the difference **variant − gene** measures
type-2 leakage. Both are Table 3 of the manuscript.

## 3. Repository layout

```
run_pipeline.py                 orchestrator: data | model | all
config/pipeline.yaml            seed, panel sizes, trials, paths
config/smoke.yaml               overrides for the smoke test
scripts/
  utils/
    schema.py                   feature groups, feature sets, baseline definitions
    metrics.py                  bootstrap CIs, thresholds, DeLong, paired bootstrap
    config.py                   YAML loader with defaults, run-directory naming
    clinvar_utils.py            label mapping, HGVS parsing, missense filter
    gnomad_utils.py             GraphQL query, AF extraction
    protein_utils.py            three→one letter, protein-change parser
    provenance.py               provenance JSON writer, MD5
    logging_utils.py            file + stdout logger
  01_fetch_clinvar.py           ClinVar download and filtering                (network)
  01b_supplement_benign.py      2-star benign supplement                      (network)
  02_fetch_gnomad.py            gnomAD AF>0.001 missense per gene             (network, hours)
  03_build_panels.py            panels + gene-disjoint and variant splits
  04_annotate_features.py       6 feature groups
  04b_validate_panel_consistency.py
  04c_prefetch_cds.py           Ensembl CDS cache                             (network)
  04c_add_context_windows.py    ±5 aa / ±5 nt windows
  04d_add_layer1_scores.py      MyVariant.info scores                         (network)
  05_finalize_dataset.py        named columns, cleaning, data/final/<scheme>/
  06_enrich_features.py         149 enrichment features
  07_track_c_acmg_prior.py      rule prior
  08_train_track_a.py           Track A (script form of the PSR notebook)
  09_meta_learner.py            stacking + ablation
  10_baselines.py               published predictors vs NeuroVariant
  11_paper_tables.py            tables 1–6, figures, provenance
  run_tests.py, compare_results.py   regression tests for the utils
tests/
  fixtures/test_cases.json, baseline/test_results.json
  make_smoke_fixture.py         builds a fixture from the legacy Dataset3
notebooks/colab_runner.ipynb    thin Colab wrapper
docs/                           METHODS, DATA_DICTIONARY, OUTPUTS, reporting_checklist
```

## 4. Installation

Python ≥ 3.10.

```bash
git clone git@github.com:Neuronauts-AI/Neurovariant-.git neurovariant
cd neurovariant
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Optional local score files (the pipeline runs without them; the
corresponding columns stay NaN and are reported as such):

| file | put in | used by |
|---|---|---|
| `revel_all_chromosomes.csv.gz` | `data/raw/` | 04 |
| `AlphaMissense_hg38.tsv.gz` | `data/raw/` | 04 |
| `gnomad_constraints.tsv` | `data/raw/` | 04 |

## 5. Running the pipeline

```bash
# Data stage — ClinVar ~10 min, gnomAD 2–4 h (checkpointed), MyVariant ~30 min
python run_pipeline.py data
python run_pipeline.py data --skip-data-steps 02_fetch_gnomad.py   # resume-friendly

# Model stage — 4 runs × (06 → 07 → 08 → 09), then 10 per scheme, then 11
python run_pipeline.py model                # 200 Optuna trials per learner per run
python run_pipeline.py model --fast         # 15 trials, 200 bootstrap — minutes
python run_pipeline.py model --schemes gene --feature-sets core   # one cell

# Everything
python run_pipeline.py all
```

Every step is an ordinary script and can be run alone, e.g.

```bash
python scripts/08_train_track_a.py --scheme gene --feature-set core --n-trials 200
python scripts/09_meta_learner.py   --scheme gene --feature-set core
python scripts/10_baselines.py      --scheme gene
python scripts/11_paper_tables.py
```

All scripts are run from the repository root and write logs to `logs/`.

## 6. Configuration

`config/pipeline.yaml` (defaults in `scripts/utils/config.py`):

| key | default | meaning |
|---|---|---|
| `seed` | 42 | every random process |
| `general_excludes_specialty_genes` | true | keeps panels gene-disjoint |
| `panel_targets` | see file | pathogenic/benign targets per panel and side |
| `split_schemes` | `[gene, variant]` | axis 1 of the matrix |
| `feature_sets` | `[core, full]` | axis 2 |
| `model.n_folds` | 5 | outer CV |
| `model.n_trials` | 200 | Optuna trials per learner |
| `model.calibration_fraction` | 0.15 | hold-out for isotonic calibration and thresholds |
| `model.min_sensitivity` | 0.90 | sensitivity-priority operating point |
| `model.variance_threshold`, `model.correlation_threshold` | 0.01, 0.95 | feature filters |
| `model.n_bootstrap` | 2000 | CI resamples |
| `paths.*` | `data/final`, `data/enriched`, `data/track_c`, `results` | relocatable roots |

CLI flags (`--n-trials`, `--fast`, `--schemes`, `--feature-sets`) override
the file for one invocation and are recorded in `run_config.json`.

## 7. Outputs

See [`docs/OUTPUTS.md`](docs/OUTPUTS.md). In short:

* `data/final/<scheme>/{train,test}_<panel>.csv` + `feature_groups.json`,
  `dataset_summary.csv`, `split_manifest.json`
* `results/<scheme>_<feature_set>/` — `metrics.json`, `results_table.csv`,
  `predictions_*.csv`, `oof_track_a.csv`, `fold_assignments.json`,
  `feature_cols.json`, Optuna trial tables, SHAP tables, `figures/`,
  `model_artefacts.pkl`, `environment.json`, `run_config.json`, `meta/`
* `results/<scheme>_baselines/` — baseline and comparison tables + figures
* `results/paper/` — Tables 1–6 (csv + md), `figures/`, `summary.md`,
  `provenance.json`

## 8. Smoke test without network access

The model stage can be exercised end-to-end in a few minutes from the
legacy competition dataset:

```bash
python tests/make_smoke_fixture.py --dataset3 /path/to/Model_Training/Dataset3 --out data_smoke
python scripts/05_finalize_dataset.py --config config/smoke.yaml --interim-dir data_smoke/interim
python run_pipeline.py model --config config/smoke.yaml --fast --n-trials 5
open results_smoke/paper/summary.md
```

The fixture restores the anonymised columns of Dataset3 via its
`column_schema.json`, keeps real genes for test rows and assigns
pseudo-genes to training rows. It exercises every code path but is **not a
scientific dataset**: the legacy PAH/CFTR benign rows carry no protein
change and are dropped by step 05, leaving those panels single-class.

## 9. Testing

```bash
python scripts/run_tests.py         # 10 parsing/labelling cases through utils/
python scripts/compare_results.py   # compares with the committed baseline; exit 1 on drift
python -m py_compile scripts/*.py scripts/utils/*.py run_pipeline.py
```

## 10. Provenance of the code

This repository merges two earlier generations of the project:

| origin | what was taken | what changed |
|---|---|---|
| `teknofest2026-pipeline` (GitHub, PDR stage) | steps 01–04d and the utils | 03 rewritten (gene-disjoint + variant splits, panel disjointness, no low-AF hack); 05 rewritten (no anonymisation, cleaning) |
| `Model_Training/PSR_Pipeline.ipynb` + `Dataset3/06–08` (Colab, PSR stage) | enrichment (06), rule prior (07), the training procedure (08), stacking (08→09) | fingerprinting removed; 07 rewritten on named columns; notebook converted to `08_train_track_a.py` with grouped CV; 09 rewritten with DeLong/bootstrap ablation |
| new | `utils/schema.py`, `utils/metrics.py`, `utils/config.py`, `10_baselines.py`, `11_paper_tables.py`, `run_pipeline.py`, docs, smoke fixture | — |

See [`CHANGELOG.md`](CHANGELOG.md) for the full list of behavioural
differences and the data-quality issues found in the legacy dataset.

## 11. How to report results

* Headline: gene-disjoint split, core features (`results/gene_core/`),
  ROC-AUC with 95 % CI per panel, both operating points.
* Table 3: type-1 and type-2 effects — this is the methodological
  contribution; report it even (especially) when the effect is large.
* Table 5: published predictors on the same variants, with the caveat that
  they were trained on ClinVar.
* Table 4: Track C ablation — report a null result honestly.
* Label PAH/CFTR as within-gene evaluations everywhere.
* Walk through [`docs/reporting_checklist.md`](docs/reporting_checklist.md)
  (DOME) before submission.

## 12. Data availability and citation

`data/`, `results/` and `logs/` are git-ignored. For the preprint, deposit
`data/final/`, `results/*/predictions_*.csv`, `fold_assignments.json` and
`model_artefacts.pkl` on Zenodo and cite the DOI; `results/paper/provenance.json`
records package versions, configuration and the ClinVar/gnomAD download
provenance for every run. Tag the commit used for the manuscript.

Please cite the preprint (bioRxiv, in preparation) and this repository
(see `CITATION.cff`).

License: Apache-2.0.
