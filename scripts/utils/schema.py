"""
Feature schema — the single source of truth for column semantics.

Every downstream script (05 finalize, 07 Track C, 08 Track A, 10 baselines,
11 paper tables) imports this module instead of guessing column identity.
There is no anonymisation and no fingerprinting in the preprint pipeline:
columns are what their names say they are.

FEATURE GROUPS
--------------
  sequence       ref/alt nucleotide + amino-acid identity, transition flags,
                 one-hot encodings, protein position
  context        ±5 nucleotide and ±5 amino-acid windows (raw letters and
                 physicochemical encodings)
  biochemical    physicochemical deltas (hydrophobicity, charge, volume ...),
                 Grantham, BLOSUM62, substitution flags
  enrichment     enrich_* features from 06_enrich_features.py (codon usage,
                 secondary-structure propensity, context summaries ...)
  conservation   per-site evolutionary conservation (phyloP, phastCons, GERP)
  gene_constraint gene-level intolerance (pLI, LOEUF, mis_z) — constant within
                 a gene, therefore a gene-identity proxy under variant-level
                 splits
  population     gnomAD allele frequencies and derived rarity flags
  insilico       published pathogenicity predictors trained on ClinVar/HGMD
                 (REVEL, CADD, PolyPhen-2, SIFT, AlphaMissense) — the source
                 of type-1 circularity (Grimm et al. 2015)

FEATURE SETS
------------
  core   everything EXCEPT insilico  → primary result (circularity-free)
  full   everything                  → secondary result

METADATA COLUMNS (never features)
---------------------------------
  Prefixed with "_": _variant_id, _gene, _label, _panel, _source,
  _protein_change, _split_scheme. `label` is the target.
"""

from __future__ import annotations

import re

TARGET_COL = "label"
META_PREFIX = "_"

PANELS = ["general", "cancer", "pah", "cftr"]
SINGLE_GENE_PANELS = {"pah", "cftr"}   # gene-disjoint split impossible

CAT_COLS = ["ref_aa", "alt_aa", "ref_nuc", "alt_nuc"]

CTX_RAW_COLS = (
    [f"aa_ctx_up{i}" for i in range(1, 6)] + [f"aa_ctx_dn{i}" for i in range(1, 6)]
    + [f"nuc_ctx_up{i}" for i in range(1, 6)] + [f"nuc_ctx_dn{i}" for i in range(1, 6)]
)

# ── Explicit group membership ────────────────────────────────────────────────
INSILICO_COLS = [
    "cadd_raw", "cadd_phred", "revel_score", "polyphen2_hdiv", "polyphen2_hvar",
    "sift_score", "alphamissense", "am_pathogenicity",
    "silico_mean", "silico_variance", "silico_n_tools",
]
# Standalone baselines reported in the paper (column → display name, direction)
BASELINE_SCORES = {
    "revel_score":    ("REVEL",          +1),
    "alphamissense":  ("AlphaMissense",  +1),
    "cadd_phred":     ("CADD (PHRED)",   +1),
    "polyphen2_hvar": ("PolyPhen-2 HVAR", +1),
    "sift_score":     ("SIFT",           -1),   # lower = more damaging
    "silico_mean":    ("Mean of tools",  +1),
}
# Published clinical cut-offs used by Track C PP3/BP4
INSILICO_THRESHOLDS = {
    "cadd_phred": 20.0, "revel_score": 0.75, "polyphen2_hdiv": 0.85,
    "polyphen2_hvar": 0.85, "sift_score": 0.05, "alphamissense": 0.564,
}

CONSERVATION_COLS = [
    "phylop_score", "phastcons_score", "gerp_score",
    "phylop_primate", "phastcons_primate", "gerp_rs", "phylop_primate_rank",
]
GENE_CONSTRAINT_COLS = ["pli", "loeuf", "mis_z"]

POPULATION_PREFIXES = ("maf_", "gnomad_af")

# ── Group assignment ─────────────────────────────────────────────────────────
_CTX_ENC_RE = re.compile(r"^(aa|nuc)_ctx_(up|dn)[1-5]_")
_SEQ_ONEHOT_RE = re.compile(r"^(ref|alt)_(aa|nuc)_[A-Z]$")
_BIOCHEM_RE = re.compile(
    r"^(delta_|abs_delta_|ref_(hydro|mw|charge|polar|vol|arom|blosum)$|"
    r"alt_(hydro|mw|charge|polar|vol|arom|blosum)$|grantham_dist$|"
    r"is_conservative$|is_radical$|is_charge_reversal$|gains_charge$|"
    r"loses_charge$|hydro_to_polar$|polar_to_hydro$)"
)


def feature_group(col: str) -> str | None:
    """Return the feature group of a column, or None for metadata / target."""
    if col.startswith(META_PREFIX) or col == TARGET_COL:
        return None
    if col in INSILICO_COLS:
        return "insilico"
    if col in CONSERVATION_COLS:
        return "conservation"
    if col in GENE_CONSTRAINT_COLS:
        return "gene_constraint"
    if col.startswith(POPULATION_PREFIXES):
        return "population"
    if col.startswith("enrich_"):
        return "enrichment"
    if col in CTX_RAW_COLS or _CTX_ENC_RE.match(col) or col.startswith("ctx_"):
        return "context"
    if _BIOCHEM_RE.match(col):
        return "biochemical"
    if (col in CAT_COLS or _SEQ_ONEHOT_RE.match(col)
            or col.startswith(("is_transition", "is_transversion", "is_ct_transition",
                               "aa_position", "codon_"))):
        return "sequence"
    if col.startswith("track_c_"):
        return None
    return "other"


FEATURE_SETS = {
    "core": {"sequence", "context", "biochemical", "enrichment",
             "conservation", "gene_constraint", "population", "other"},
    "full": {"sequence", "context", "biochemical", "enrichment",
             "conservation", "gene_constraint", "population", "insilico", "other"},
}


def select_features(columns, feature_set: str, drop_raw_context: bool = True) -> list[str]:
    """Columns of `columns` that belong to `feature_set`."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {feature_set!r}; choose from {list(FEATURE_SETS)}")
    allowed = FEATURE_SETS[feature_set]
    out = []
    for c in columns:
        g = feature_group(c)
        if g is None or g not in allowed:
            continue
        if drop_raw_context and c in CTX_RAW_COLS:
            continue   # raw letters are already encoded numerically
        out.append(c)
    return out


def group_table(columns) -> dict[str, list[str]]:
    """{group: [cols]} for reporting."""
    table: dict[str, list[str]] = {}
    for c in columns:
        g = feature_group(c)
        if g is not None:
            table.setdefault(g, []).append(c)
    return table
