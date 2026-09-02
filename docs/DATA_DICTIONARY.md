# Data dictionary

Every column that can appear in `data/final/<scheme>/*.csv`,
`data/enriched/<scheme>/*.csv`, `data/track_c/*/*.csv` and
`results/*/predictions_*.csv`. Group membership is decided by
`scripts/utils/schema.py::feature_group`; feature sets are
`core` = all groups except *insilico*, `full` = all groups.

## 1. Metadata and target (never used as features)

| column | type | meaning |
|---|---|---|
| `_variant_id` | str | ClinVar `VariationID`, or `gnomad_<gene>_<variant_id>` for frequency-derived benign variants |
| `_gene` | str | HGNC gene symbol |
| `_panel` | str | `general`, `cancer`, `pah`, `cftr` |
| `_source` | str | `ClinVar`, `ClinVar_anystar`, `gnomAD` — origin of the label |
| `_protein_change` | str | HGVS protein change (`p.Arg408Trp`) |
| `_split_scheme` | str | `gene` or `variant` (the scheme the file belongs to) |
| `label` | int | 1 = pathogenic / likely pathogenic, 0 = benign / likely benign / frequency-derived benign |

Interim files (step 03/04) additionally carry `_label`, `_split_variant`,
`_split_gene`, `_split_gene_note`.

## 2. Group `sequence` (step 04)

| column | type | meaning |
|---|---|---|
| `ref_nuc`, `alt_nuc` | categorical A/C/G/T | reference / alternate nucleotide |
| `ref_nuc_{A,C,G,T}`, `alt_nuc_{A,C,G,T}` | 0/1 | one-hot of the above |
| `ref_aa`, `alt_aa` | categorical, 20 letters | reference / alternate amino acid |
| `ref_aa_{A…Y}`, `alt_aa_{A…Y}` | 0/1 | one-hot of the above (40 columns) |
| `is_transition` | 0/1 | purine↔purine or pyrimidine↔pyrimidine |
| `is_transversion` | 0/1 | 1 − is_transition |
| `is_ct_transition` | 0/1 | C→T (CpG-deamination-prone) |
| `aa_position` | int | 1-based residue index in the protein |
| `aa_position_log` | float | log(1 + aa_position) |

## 3. Group `context` (steps 04c, 05)

| column | type | meaning |
|---|---|---|
| `aa_ctx_up5 … aa_ctx_up1`, `aa_ctx_dn1 … aa_ctx_dn5` | letter, `X` at termini | residues −5…−1 and +1…+5 around the substitution (raw letters; excluded from the model, encoded by enrichment blocks E/F) |
| `nuc_ctx_up5 … nuc_ctx_dn5` | letter, `N` at termini / unknown CDS | nucleotides around the substituted base |
| `{aa,nuc}_ctx_*_{property}` | float | step-04 physicochemical encodings of the window (present only when step 04c letters were available at annotation time) |
| `ctx_mean_hydro`, `ctx_std_hydro`, `ctx_mean_charge`, `ctx_std_charge` | float | window summaries (step 04) |

## 4. Group `biochemical` (step 04)

Property tables: hydrophobicity (Kyte–Doolittle), MW (Da), formal charge
(R,K = +1; D,E = −1; H = 0), polarity class (0 non-polar, 1 polar
uncharged, 2 charged), volume (Å³, Grantham), aromaticity (F,H,W,Y = 1),
BLOSUM62 diagonal.

| column | meaning |
|---|---|
| `ref_{hydro,mw,charge,polar,vol,arom,blosum}` | property of the reference residue |
| `alt_{…}` | property of the alternate residue |
| `delta_{…}` | alt − ref |
| `abs_delta_{…}` | \|alt − ref\| |
| `grantham_dist` | Grantham (1974) distance, 0 for identity, 5–215 |
| `is_conservative` | same charge and same polarity class |
| `is_charge_reversal` | positive ↔ negative |
| `gains_charge`, `loses_charge` | neutral → charged / charged → neutral |
| `hydro_to_polar`, `polar_to_hydro` | sign change of hydrophobicity |

## 5. Group `enrichment` (step 06) — prefix `enrich_`

| block | columns | meaning |
|---|---|---|
| A | `enrich_{delta,abs_delta,ref,alt}_{hydro,mw,charge,polar,vol,arom,blosum}`, `enrich_grantham_dist` | as §4, recomputed from letters (duplicates of §4 are removed by the correlation filter) |
| B | `enrich_is_conservative`, `enrich_is_radical`, `enrich_is_charge_reversal`, `enrich_gains_charge`, `enrich_loses_charge`, `enrich_hydro_to_polar`, `enrich_polar_to_hydro`, `enrich_same_polarity_class`, `enrich_same_charge_sign` | substitution class flags |
| C | `enrich_is_transition`, `enrich_is_transversion`, `enrich_is_ct_transition` | nucleotide change flags |
| D | `enrich_aa_position_log` | log position |
| E | `enrich_ctx_{up5…dn5}_{hydro,charge,polar,vol,mw,arom,blosum}` | 70 per-position window descriptors |
| F | `enrich_ctx_mean_hydro`, `enrich_ctx_std_hydro`, `enrich_ctx_mean_charge`, `enrich_ctx_std_charge` | window summaries |
| G | `enrich_codon_position` (1–3), `enrich_ref_codon_degeneracy`, `enrich_alt_codon_degeneracy` | codon-level flags inferred from the nucleotide window |
| H | `enrich_{ref,alt}_mean_rscu`, `enrich_delta_mean_rscu`, `enrich_abs_delta_mean_rscu`, `enrich_{ref,alt}_max_rscu`, `enrich_delta_max_rscu`, `enrich_codon_usage_pressure` | relative synonymous codon usage (human table) |
| I | `enrich_{ref,alt}_cf_{alpha,beta,turn}`, `enrich_{ref,alt}_rsa`, `enrich_delta_cf_{alpha,beta,turn}`, `enrich_abs_delta_cf_{alpha,beta}`, `enrich_delta_rsa`, `enrich_abs_delta_rsa`, `enrich_is_helix_breaker_intro`, `enrich_is_sheet_breaker_intro`, `enrich_burial_change`, `enrich_ctx_mean_alpha`, `enrich_ctx_mean_beta`, `enrich_ctx_helix_frac`, `enrich_ctx_sheet_frac` | Chou–Fasman propensities, relative solvent accessibility, structural-breaker flags |

## 6. Group `conservation` (step 04d)

| column | range | meaning |
|---|---|---|
| `phylop_primate` | ≈ −20…+1 (primate 17-way) | per-site conservation, positive = conserved |
| `phastcons_primate` | 0–1 | probability the site is in a conserved element |
| `gerp_rs` | ≈ −12…+6 | GERP++ rejected substitutions |
| `phylop_score`, `phastcons_score`, `gerp_score`, `phylop_primate_rank` | — | alternative/legacy columns; present only if a local score file was joined |

## 7. Group `gene_constraint` (gnomAD constraint table)

| column | meaning |
|---|---|
| `pli` | probability of loss-of-function intolerance (0–1) |
| `loeuf` | LoF observed/expected upper bound (lower = more constrained) |
| `mis_z` | missense depletion z-score |

These are **constant within a gene**; under a variant-level split they act
as gene identifiers (type-2 leakage). They are legitimate under the
gene-disjoint scheme because test genes are unseen.

## 8. Group `population` (step 02)

| column | meaning |
|---|---|
| `maf_global` | gnomAD global allele frequency (0 when absent from gnomAD) |
| `maf_{afr,amr,eas,fin,nfe,sas}` | sub-population AF (NaN when unavailable) |
| `maf_log10` | log10(maf_global + 1e-6) |
| `maf_is_rare`, `maf_is_very_rare`, `maf_is_common` | AF < 1e-4, < 1e-6, > 0.01 |

## 9. Group `insilico` (step 04d; **excluded from `core`**)

| column | range | direction | source |
|---|---|---|---|
| `cadd_phred` | 0–99 | high = deleterious | CADD v1.6/1.7 via MyVariant |
| `revel_score` | 0–1 | high = pathogenic | REVEL |
| `polyphen2_hdiv`, `polyphen2_hvar` | 0–1 | high = damaging | PolyPhen-2 |
| `sift_score` | 0–1 | **low** = damaging | SIFT |
| `alphamissense` | 0–1 | high = pathogenic | AlphaMissense (local file join) |
| `cadd_raw` | — | raw CADD score, if joined locally | CADD |
| `silico_mean` | 0–1 | mean of direction-normalised available tools | step 05 |
| `silico_variance` | ≥ 0 | population variance of the normalised tools | step 05 |
| `silico_n_tools` | 0–6 | number of tools with a score | step 05 |

## 10. Track C outputs (`data/track_c/<scheme>_<fset>/*_trackc.csv`)

| column | meaning |
|---|---|
| `track_c_logit` | sum of fired evidence weights |
| `track_c_score` | σ(logit) — prior P(pathogenic) |
| `track_c_n_fired` | number of evidence items fired |
| `track_c_rules` | comma-separated item names (BA1, BS1, BS2, PP3, BP4, CONS_HIGH, GENE_INTOL, RADICAL_SUB, CHARGE_REV, CONSERV_SUB) |

## 11. Prediction files (`results/<run>/predictions_<panel>.csv`)

| column | meaning |
|---|---|
| `P_pathogenic` | isotonic-calibrated ensemble probability |
| `P_raw_ensemble` | mean of XGBoost and LightGBM probabilities before calibration (input to the meta-learner) |
| `pred_youden` | binary call at the panel's Youden threshold |
| `pred_sens90` | binary call at the sensitivity-priority threshold |
| `P_meta`, `P_track_a_meta`, `pred_meta` | (meta/ sub-directory) stacked probability, Track-A-only stacked probability, call at the meta Youden threshold |

`oof_track_a.csv` carries `_row_id`, `in_fit_set`, `oof_xgb`, `oof_lgbm`,
`oof_ensemble` for every training variant (NaN for calibration hold-out).
