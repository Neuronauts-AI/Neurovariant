# Methods — step-by-step description of the NeuroVariant pipeline

This document describes every stage of the pipeline at the level of detail
expected in a Methods section: inputs, transformations, parameters, outputs,
and the reason each design choice was made. Section numbers follow the
script numbers in `scripts/`.

Contents

1. [Overview and design principles](#1-overview-and-design-principles)
2. [Data sources and labels](#2-data-sources-and-labels-steps-01-01b-02)
3. [Panel construction and split schemes](#3-panel-construction-and-split-schemes-step-03)
4. [Feature annotation](#4-feature-annotation-steps-04-04d)
5. [Dataset finalisation and cleaning](#5-dataset-finalisation-and-cleaning-step-05)
6. [Enrichment features](#6-enrichment-features-step-06)
7. [Track C — rule-based prior](#7-track-c--rule-based-prior-step-07)
8. [Track A — gradient-boosted ensemble](#8-track-a--gradient-boosted-ensemble-step-08)
9. [Stacking meta-learner and ablation](#9-stacking-meta-learner-and-ablation-step-09)
10. [Baselines: published predictors](#10-baselines-published-predictors-step-10)
11. [Result collection](#11-result-collection-step-11)
12. [Statistical procedures](#12-statistical-procedures)
13. [Reproducibility controls](#13-reproducibility-controls)
14. [Known limitations](#14-known-limitations)
15. [References](#15-references)

---

## 1. Overview and design principles

NeuroVariant classifies single-nucleotide **missense** variants as
pathogenic or benign. The pipeline has two stages:

* **Data stage (01–05)** — downloads public data, builds four disease
  panels, computes features and writes analysis-ready CSV files.
* **Model stage (06–11)** — enriches features, trains and evaluates the
  models, compares them with published predictors and collects tables and
  figures for the manuscript.

Three principles govern the design.

**Principle 1 — control both types of circularity.** Grimm *et al.* (2015)
showed that benchmarks of missense predictors are inflated by (i) *type-1
circularity*: the evaluated tool (or a feature it consumes) was trained on
the very variants used for testing, and (ii) *type-2 circularity*: the same
gene contributes variants to both training and test sets, so a model can
succeed by memorising gene identity. The pipeline addresses both with an
explicit **experimental matrix**:

| axis | levels | purpose |
|---|---|---|
| feature set | `core` — no published predictor scores; `full` — all features | isolates the type-1 effect (full − core on identical test variants) |
| split scheme | `gene` — gene-disjoint; `variant` — random variant-level | isolates the type-2 effect (variant − gene) |

The primary result of the manuscript is the `gene` × `core` cell; the other
three cells are reported as controls.

**Principle 2 — no hidden column semantics.** Every column has a name and a
documented feature group (`scripts/utils/schema.py`, `docs/DATA_DICTIONARY.md`).
The competition-era anonymisation (`col_0001…`) and the statistical
"fingerprinting" used to recover it were removed because they add no
scientific content and obscure what the model sees.

**Principle 3 — every number in the paper is produced by a script.** No
notebook state, no manual copy-paste: `11_paper_tables.py` regenerates
every table and figure from the run directories, and `provenance.json`
records data versions, package versions and configuration for each run.

---

## 2. Data sources and labels (steps 01, 01b, 02)

### 2.1 ClinVar (step 01)

`01_fetch_clinvar.py` downloads `variant_summary.txt.gz` from the NCBI FTP
site and keeps records that satisfy all of:

* assembly GRCh38;
* type "single nucleotide variant";
* HGVS name containing a protein change of the form `p.Xxx123Yyy` with two
  different standard amino acids (a **missense** substitution; synonymous,
  nonsense and frameshift are excluded by `utils.clinvar_utils.is_missense`);
* review status *reviewed by expert panel* or *practice guideline*
  (ClinVar 3–4 stars);
* clinical significance mapped by `utils.clinvar_utils.assign_label`:
  `Pathogenic`/`Likely pathogenic` → 1, `Benign`/`Likely benign` → 0;
  conflicting, uncertain and drug-response records are dropped.

The script writes `data/raw/clinvar_filtered.csv` and a provenance JSON
(`data/raw/provenance_clinvar.json`) containing the download URL, the
server `Last-Modified` header (ClinVar release surrogate), the MD5 of the
archive and the filters applied.

### 2.2 Two-star benign supplement (step 01b)

Because expert-panel benign assertions are rare, `01b_supplement_benign.py`
adds benign missense variants with review status *criteria provided,
multiple submitters, no conflicts* (2 stars). Supplemented rows carry
`label_confidence = likely_2star` so that a label-noise sensitivity analysis
can be run; the column is stripped from the modelling files.

### 2.3 gnomAD common variants (step 02)

For every gene that carries at least one pathogenic ClinVar variant,
`02_fetch_gnomad.py` queries the gnomAD GraphQL API (release v4) and keeps
missense variants with global allele frequency **AF > 0.001**. Such variants
are used as *presumed-benign* controls only when a panel lacks enough
ClinVar benign variants (see §3.3). Population-specific frequencies
(AFR, AMR, EAS, FIN, NFE, SAS) are stored for the population feature group.
The threshold follows the ACMG BS1 logic ("allele frequency greater than
expected for the disorder"); the label origin is preserved in `_source`
so the manuscript can report how many benign variants are clinically
asserted versus frequency-derived.

---

## 3. Panel construction and split schemes (step 03)

### 3.1 Panels

| panel | genes | rationale |
|---|---|---|
| `general` | all genes **except** those in the three specialty panels | genome-wide performance |
| `cancer` | 33 hereditary-cancer genes (BRCA1/2, PALB2, MMR genes, TP53, PTEN, …) | multi-gene clinical panel |
| `pah` | PAH | single-gene, metabolic (phenylketonuria) |
| `cftr` | CFTR | single-gene, small sample size |

Panels are **variant- and gene-disjoint from each other**. In the legacy
pipeline the general panel excluded only PAH and CFTR, so a BRCA1 variant
could appear in the training set of the cancer panel and the test set of
the general panel. `03_build_panels.py` now excludes every specialty gene
from the general panel and asserts that no `VariationID` occurs in more
than one panel.

### 3.2 Balanced sampling targets

Class-balanced targets per panel and side are read from
`config/pipeline.yaml` (`panel_targets`). Defaults reproduce the earlier
design (general 1500/1500 train and 1000/1000 test; cancer and PAH
200/200 + 100/100; CFTR 70/70 + 30/30). When a class has fewer candidates
than the target, all available variants are used and a warning is logged;
the achieved counts appear in `dataset_summary.csv`.

### 3.3 Benign pool policy

1. ClinVar 3–4-star (and 2-star supplemented) benign variants in the panel's genes.
2. For single-gene panels only, if fewer than 50 % of the benign target is
   reached: any-star ClinVar benign missense variants of that gene.
3. If still short: gnomAD AF > 0.001 missense variants of the panel's
   pathogenic genes, labelled `_source = gnomAD`.

The legacy step `03c_fix_pah_panel.py`, which labelled gnomAD variants with
AF > 10⁻⁶ as benign, was **removed**: rare variants of unknown significance
cannot serve as negative controls.

### 3.4 Split schemes

Both schemes are assigned to every variant at once and stored as columns
`_split_variant` and `_split_gene` (values `train`, `test`, `unused`).

* **Variant-level split.** Candidates of each class are shuffled with the
  configured seed and cut at the target counts.
* **Gene-disjoint split.** Genes are shuffled; genes are assigned to the
  test side greedily until the test side contains at least
  `test_fraction` (= test target / total target) of both the pathogenic
  and the benign candidates. Each side is then down-sampled to its class
  targets. An assertion verifies that no gene appears on both sides.
* **Single-gene panels** (PAH, CFTR) cannot be split by gene. Their
  `_split_gene` copies `_split_variant` and `_split_gene_note` records
  "within-gene"; all downstream tables carry this note.

`split_manifest.json` records the seed, the excluded genes, the test genes
of the gene-disjoint split and per-scheme counts.

---

## 4. Feature annotation (steps 04–04d)

Features are computed **once** on the panel pool (all variants selected
under either scheme), so both schemes see identical feature values.

### 4.1 Six feature groups (step 04)

`04_annotate_features.py` derives, for every variant:

| group | content | source |
|---|---|---|
| sequence | reference/alternate nucleotide and amino acid (categorical + one-hot), transition/transversion flags, C→T flag, protein position and log-position | HGVS name |
| context | placeholders for the ±5 amino-acid and ±5 nucleotide windows (filled in step 04c) | — |
| biochemical | signed and absolute differences of Kyte–Doolittle hydrophobicity, molecular weight, formal charge, polarity class, Grantham volume, aromaticity and BLOSUM62 self-score; Grantham distance; conservative / radical / charge-reversal flags | published amino-acid property tables |
| conservation | phyloP, phastCons, GERP++ (filled in step 04d); pLI, LOEUF, missense z from gnomAD constraint tables | gnomAD constraints file (optional local download) |
| population | global and six sub-population allele frequencies, log10(AF+1e-6), rarity flags (< 1e-4, < 1e-6, > 0.01) | gnomAD (step 02) |
| in-silico | CADD PHRED, REVEL, PolyPhen-2 HDIV/HVAR, SIFT, AlphaMissense; mean, variance and count of available tools | REVEL / AlphaMissense local files (optional) and MyVariant.info (step 04d) |

Metadata columns (`_variant_id`, `_gene`, `_label`, `_panel`, `_source`,
`_protein_change`, `_split_variant`, `_split_gene`, `_split_gene_note`) are
passed through untouched.

### 4.2 Consistency validation (step 04b)

Checks column-set equality across panels, NaN rates above 50 %,
near-duplicate (gene + amino-acid change) rows and class balance. Writes
`logs/panel_consistency_report.json` and exits non-zero on failure.

### 4.3 Context windows (steps 04c)

`04c_prefetch_cds.py` caches coding sequences from Ensembl REST;
`04c_add_context_windows.py` extracts the five residues on each side of the
substituted position from the UniProt/Ensembl protein sequence and the five
nucleotides on each side from the CDS, encoded one-hot (padding `X`/`N` at
termini). Step 05 converts the one-hots back to letters for the enricher.

### 4.4 In-silico and conservation scores (step 04d)

`04d_add_layer1_scores.py` queries MyVariant.info by gene symbol and
amino-acid change (coordinate-independent) and fills CADD PHRED, REVEL,
PolyPhen-2 (HDIV, HVAR), SIFT, phyloP-17way-primate,
phastCons-17way-primate and GERP++ RS. Missing scores remain NaN and are
median-imputed inside the model pipeline (training statistics only).

---

## 5. Dataset finalisation and cleaning (step 05)

`05_finalize_dataset.py` writes, for each scheme and panel,
`data/final/<scheme>/train_<panel>.csv` and `test_<panel>.csv`, each with
the target `label` and the metadata columns. Cleaning operations, all
logged with counts:

1. **Unfeaturised rows.** Rows whose `ref_aa`, `alt_aa` or `aa_position`
   is missing are removed. In the legacy competition data 100 % of PAH and
   CFTR benign rows were of this kind — empty feature vectors from which a
   model learns "missing ⇒ benign". The counts by label and source are
   written to the log for the manuscript.
2. **Context letters.** `aa_ctx_*`/`nuc_ctx_*` letters are reconstructed
   from the one-hot windows; the 240 one-hot columns are dropped.
3. **In-silico summary.** `silico_mean`, `silico_variance`,
   `silico_n_tools` are recomputed after direction-normalising each tool
   to [0, 1] with high = pathogenic (CADD/40 capped at 1, 1 − SIFT).
4. **Degenerate columns.** All-NaN and zero-variance columns are dropped.
5. **Duplicate test vectors.** A test row whose full feature vector is
   identical to a training row of the same scheme is removed from the
   test side.
6. **Gene-overlap guard.** For the gene scheme the script errors if any
   gene occurs on both sides of a multi-gene panel.

It also writes `feature_groups.json` (column → group), `dataset_summary.csv`
(rows, genes, class balance, label sources per file) and copies
`split_manifest.json`.

---

## 6. Enrichment features (step 06)

`06_enrich_features.py` appends 149 `enrich_*` features derived purely from
the 25 named sequence columns, with no external files or network access:

| block | features | basis |
|---|---|---|
| A substitution biochemistry | ref/alt/Δ/|Δ| of hydrophobicity, MW, charge, polarity, volume, aromaticity, BLOSUM62; Grantham | Grantham 1974; Kyte & Doolittle 1982; Henikoff & Henikoff 1992 |
| B substitution flags | conservative, radical, charge reversal, charge gain/loss, hydrophobic↔polar, same polarity class / charge sign | standard classifications |
| C sequence flags | transition, transversion, C→T | — |
| D position | log position | — |
| E per-position context | 7 physicochemical descriptors × 10 window positions | as A |
| F context summary | mean/SD of hydrophobicity and charge across the window | — |
| G codon flags | codon position, degeneracy of ref/alt codon | genetic code |
| H codon-usage bias | mean/max RSCU of ref/alt codons, Δ, usage pressure | human codon-usage table |
| I secondary structure | Chou–Fasman α/β/turn propensities, relative solvent accessibility, helix/sheet-breaker introduction, burial change, window propensities | Chou & Fasman 1978 |

Many block-A features duplicate step-04 biochemical features. This is
intentional (the enricher is also the competition-day feature generator);
the correlation filter in step 08 removes the duplicate, keeping the
step-04 column so that SHAP group attribution stays interpretable.

---

## 7. Track C — rule-based prior (step 07)

`07_track_c_acmg_prior.py` computes a transparent prior probability from
named columns. Evidence items are summed in log-odds units and passed
through a logistic function. Items with an ACMG/AMP counterpart keep the
ACMG code; the remaining heuristics are labelled descriptively so the
manuscript does not over-claim ACMG compliance.

| item | condition | weight |
|---|---|---|
| BA1 | global AF > 0.05 | −6 (override: returns immediately) |
| BS1 | global AF > 0.01 | −2 |
| BS2 | any sub-population AF > 0.005 | −1 |
| PP3 / BP4 | mean normalised in-silico score > 0.7 / < 0.3 (**full feature set only**) | +1 / −1 |
| CONS_HIGH | max(phyloP, phastCons, GERP) > 2.5 | +2 |
| GENE_INTOL | pLI > 0.9 | +2 |
| RADICAL_SUB | Grantham > 150 | +2 |
| CHARGE_REV | K/R ↔ D/E | +2 |
| CONSERV_SUB | Grantham < 30 and same charge and polarity class | −1 |

Outputs per panel and side: `track_c_logit`, `track_c_score` (= σ(logit)),
`track_c_n_fired`, `track_c_rules` (comma-separated list) and, where labels
exist, the standalone AUC in `track_c_report.json`. Under the `core`
feature set the PP3/BP4 items are disabled so that Track C is as
circularity-free as Track A.

---

## 8. Track A — gradient-boosted ensemble (step 08)

`08_train_track_a.py` is the script form of the `PSR_Pipeline` notebook.
One invocation trains one cell of the experimental matrix.

### 8.1 Feature selection (training data only)

1. Columns of the requested feature set (`utils.schema.select_features`);
   raw context letters are excluded because they are numerically encoded
   by blocks E/F.
2. Variance filter: numeric columns with SD < 0.01 are dropped.
3. Correlation filter: for each pair with |Pearson r| > 0.95 the later
   column (enrichment duplicates are ordered last) is dropped.
4. Four categorical columns (`ref_aa`, `alt_aa`, `ref_nuc`, `alt_nuc`) are
   one-hot encoded; numeric columns are median-imputed. The preprocessor is
   refitted inside every CV fold.

The retained columns and both drop lists are written to `feature_cols.json`.

### 8.2 Calibration hold-out

15 % of the combined training pool is held out for probability calibration
and threshold selection and is never used for fitting or tuning. Under the
gene scheme the hold-out is drawn with `GroupShuffleSplit` on gene (rows of
single-gene panels form singleton groups); under the variant scheme with a
stratified random split.

### 8.3 Cross-validation

Five-fold `StratifiedGroupKFold` on gene (gene scheme) or `StratifiedKFold`
(variant scheme) with default hyper-parameters produces out-of-fold (OOF)
predictions for XGBoost, LightGBM and their mean. Fold membership is
serialised to `fold_assignments.json`; OOF predictions to
`oof_track_a.csv` (aligned to the full training table, NaN for hold-out
rows) for the meta-learner. A leave-one-out CV on the CFTR training panel
reports small-sample stability.

### 8.4 Hyper-parameter optimisation

Optuna TPE (seeded), **200 trials per learner** (configurable; `--fast`
uses 15), objective = mean inner 3-fold ROC-AUC with the same grouping as
§8.3. Search spaces:

* XGBoost: n_estimators 100–600, max_depth 3–8, learning_rate 0.01–0.15
  (log), subsample 0.6–1, colsample_bytree 0.5–1, reg_lambda 0.1–10 (log),
  reg_alpha 1e-4–1 (log), min_child_weight 1–10.
* LightGBM: n_estimators 100–600, num_leaves 15–127, learning_rate,
  subsample, colsample_bytree, reg_lambda, reg_alpha as above,
  min_child_samples 5–50.

Trial tables (`optuna_trials_*.csv`), optimisation-history and fANOVA
importance figures are saved.

### 8.5 Final model, calibration and thresholds

Both learners are refitted on the fit set with the best parameters; the
ensemble is the mean probability. Isotonic regression fitted on the
calibration hold-out maps ensemble scores to calibrated probabilities;
the calibration curve and pre/post AUC are saved. Two operating points are
derived on the calibrated hold-out:

* **Youden's J** — maximises sensitivity + specificity − 1;
* **sensitivity-priority** — highest threshold with sensitivity ≥ 0.90
  (clinical screening setting).

Thresholds are chosen per panel when the panel has ≥ 30 calibration
variants of both classes, otherwise the global threshold is used.

### 8.6 Evaluation

For each test panel: ROC-AUC, PR-AUC, F1, precision, sensitivity,
specificity, MCC, Brier score at both operating points, with 2000-sample
bootstrap 95 % CIs. Predictions (`P_pathogenic`, raw ensemble, both
binary calls) are saved per variant. Figures: confusion matrices, ROC and
PR curves, calibration curve, Optuna history/importance, SHAP beeswarm
(top 20), SHAP group importance (schema groups), SHAP waterfalls for the
most confident true positive and the worst false negative. An error
analysis compares log10(MAF), in-silico tool variance, Grantham distance
and phyloP between correct, false-positive and false-negative predictions.

---

## 9. Stacking meta-learner and ablation (step 09)

Following Wolpert (1992), a logistic regression with L2 penalty is fitted on
two inputs: logit of the Track A **out-of-fold** probability and the Track
C logit. Because every training variant's Track A input comes from a model
that never saw it, the stack cannot leak. `C` is selected from
{0.01, 0.1, 1, 10} by grouped CV.

Ablation — *Track A alone* versus *Track A + Track C* — is evaluated (i) by
grouped 5-fold CV on the training pool and (ii) on every held-out test
panel, each with a DeLong test and a paired-bootstrap CI on the AUC
difference. Coefficients are reported; a near-zero Track C weight is a
legitimate finding and is reported as such.

---

## 10. Baselines: published predictors (step 10)

For each test panel of a scheme, `10_baselines.py` evaluates REVEL,
AlphaMissense, CADD PHRED, PolyPhen-2 HVAR, SIFT (sign-flipped) and the
mean-of-tools score on the variants where the score is available
(coverage reported), and compares each with every NeuroVariant model of
the same scheme on the **intersection** of scored variants (DeLong test,
paired bootstrap). The manuscript must state that the published predictors
were trained on ClinVar/HGMD and are therefore evaluated with type-1
circularity in their favour; the `core` model has never seen these scores.

---

## 11. Result collection (step 11)

`11_paper_tables.py` reads every run directory and writes `results/paper/`:

| file | content |
|---|---|
| table1_dataset | rows, genes, class balance and label sources per scheme/panel/side |
| table2_main_results | ROC-AUC (95 % CI) and secondary metrics per panel × (scheme, feature set), plus meta AUC |
| table3_circularity | type-1 effect (full − core, paired DeLong on identical variants) and type-2 effect (variant − gene, independent test sets) |
| table4_track_c_ablation | A vs A + C per panel and run |
| table5_baselines | published predictors vs NeuroVariant on common variants |
| table6_features | retained features per group per run |
| figures/ | all run figures with stable names |
| summary.md | all tables in Markdown |
| provenance.json | environment, configuration and data provenance per run |

---

## 12. Statistical procedures

* **Bootstrap CIs** — 2000 resamples with replacement (seeded), percentile
  2.5–97.5 for AUC, PR-AUC, F1, sensitivity, specificity.
* **DeLong test** — two-sided test for equal AUC of two scores on the
  same variants (fast algorithm of Sun & Xu 2014), used for full vs core,
  meta vs Track A and NeuroVariant vs each published predictor.
* **Paired bootstrap of ΔAUC** — 2000 paired resamples; reported CI and a
  two-sided bootstrap p.
* **Independent test sets** (variant vs gene scheme) are compared by the
  difference of point estimates with a conservative interval formed from
  the two 95 % CIs; no p-value is claimed.

---

## 13. Reproducibility controls

* Single seed (`config/pipeline.yaml: seed`) propagated to sampling,
  splits, CV, Optuna, bootstrap and models.
* `fold_assignments.json` stores every CV membership.
* `environment.json` records Python and package versions per run;
  `run_config.json` records CLI arguments and the resolved configuration.
* `data/raw/provenance_*.json` records ClinVar release surrogate, MD5 and
  filters; gnomAD query parameters and checkpoint files.
* `tests/run_tests.py` + `compare_results.py` guard the parsing utilities
  against regressions with a committed baseline.
* `tests/make_smoke_fixture.py` + `config/smoke.yaml` allow the whole model
  stage to be exercised in minutes without network access.

---

## 14. Known limitations

1. PAH and CFTR are single-gene panels: gene-disjoint evaluation is
   impossible; results are within-gene.
2. Published predictors are evaluated on ClinVar variants they may have
   been trained on; a time-split test set (variants deposited after each
   predictor's training cut-off) would be the stricter comparison and is
   not implemented.
3. Part of the benign class is frequency-derived (gnomAD AF > 0.001)
   rather than clinically asserted; the proportion is reported per panel.
4. The 3–4-star ClinVar filter over-represents well-studied genes.
5. Missing in-silico and conservation scores are median-imputed; coverage
   is reported by the baseline script.

---

## 15. References

* Grimm DG *et al.* The evaluation of tools used to predict the impact of
  missense variants is hindered by two types of circularity. *Hum Mutat*
  2015;36:513–523.
* Walsh I *et al.* DOME: recommendations for supervised machine learning
  validation in biology. *Nat Methods* 2021;18:1122–1127.
* Richards S *et al.* Standards and guidelines for the interpretation of
  sequence variants (ACMG/AMP). *Genet Med* 2015;17:405–424.
* Ioannidis NM *et al.* REVEL. *Am J Hum Genet* 2016;99:877–885.
* Cheng J *et al.* Accurate proteome-wide missense variant effect
  prediction with AlphaMissense. *Science* 2023;381:eadg7492.
* Rentzsch P *et al.* CADD. *Nucleic Acids Res* 2019;47:D886–D894.
* Grantham R. Amino acid difference formula to help explain protein
  evolution. *Science* 1974;185:862–864.
* Kyte J, Doolittle RF. A simple method for displaying the hydropathic
  character of a protein. *J Mol Biol* 1982;157:105–132.
* Henikoff S, Henikoff JG. Amino acid substitution matrices from protein
  blocks. *PNAS* 1992;89:10915–10919.
* Chou PY, Fasman GD. Prediction of the secondary structure of proteins
  from their amino acid sequence. *Adv Enzymol* 1978;47:45–148.
* Wolpert DH. Stacked generalization. *Neural Netw* 1992;5:241–259.
* Sun X, Xu W. Fast implementation of DeLong's algorithm for comparing the
  areas under correlated ROC curves. *IEEE Signal Process Lett*
  2014;21:1389–1393.
* Akiba T *et al.* Optuna: a next-generation hyperparameter optimization
  framework. *KDD* 2019.
* Lundberg SM, Lee S-I. A unified approach to interpreting model
  predictions. *NeurIPS* 2017.
