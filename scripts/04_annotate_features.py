"""
STEP 4 — Annotate features (all 6 competition-aligned feature groups)
======================================================================
Computes the full feature matrix for every variant in each panel.

Feature groups generated:
  1. Sequence & mutation    — ref/alt nucleotide, codon, amino acid substitution
  2. Local context          — ±5 nucleotide window, ±5 amino acid window
  3. Biochemical effects    — property deltas (hydrophobicity, charge, etc.)
  4. Conservation           — phyloP / phastCons / GERP from CADD scores
  5. Population frequency   — gnomAD AF global + 6 subpopulations
  6. In silico scores       — CADD, REVEL, AlphaMissense (if files present)

All values come from static lookup tables or external score files.
No internet access required at this step.

EXTERNAL SCORE FILES (optional — improves prototype quality significantly)
---------------------------------------------------------------------------
Download these files ONCE and place in data/raw/:

  File                             URL
  ────────────────────────────────────────────────────────────────────────
  cadd_scores.tsv.gz               https://cadd.gs.washington.edu/download
                                   → "GRCh38-v1.7 SNVs" whole-genome file
                                   WARNING: ~80 GB. Use tabix to extract
                                   only the variants you need (see below).

  revel_all_chromosomes.csv.gz     https://sites.google.com/site/
                                   revelgenomics/downloads

  AlphaMissense_hg38.tsv.gz        https://zenodo.org/records/8208688
                                   → AlphaMissense_hg38.tsv.gz (~3.5 GB)

  gnomad_constraints.tsv           https://gnomad.broadinstitute.org/downloads
                                   → "Gene constraint" TSV file

TABIX EXTRACTION (for CADD — avoids downloading 80 GB):
  # Install tabix: sudo apt install tabix  OR  conda install -c bioconda tabix
  # Get just the variants in your panels:
  python scripts/04b_extract_cadd.py  # helper script — see below

The pipeline runs without these files — it just produces NaN for the
scores that require them. Even with NaN scores, the prototype is useful
for testing the pipeline and generating PSR report results.

INPUTS
------
  data/interim/panel_{name}_pool.csv   (from step 03)

OUTPUTS
-------
  data/interim/panel_{name}_pool_features.csv
"""

import os
import sys
import json
import logging
import warnings
from multiprocessing import cpu_count
from joblib import Parallel, delayed
import pandas as pd
import numpy as np

from utils.logging_utils import get_logger
from utils.protein_utils import parse_protein_change, THREE_TO_ONE

warnings.filterwarnings("ignore")
log = get_logger("04_annotate_features.log")

# ═════════════════════════════════════════════════════════════════════════════
# STATIC LOOKUP TABLES (no internet needed)
# ═════════════════════════════════════════════════════════════════════════════

AA_CODES = list("ACDEFGHIKLMNPQRSTVWY")

# Hydrophobicity — Kyte & Doolittle (1982)
HYDRO = {
    "A": 1.8,"R":-4.5,"N":-3.5,"D":-3.5,"C": 2.5,"Q":-3.5,"E":-3.5,
    "G":-0.4,"H":-3.2,"I": 4.5,"L": 3.8,"K":-3.9,"M": 1.9,"F": 2.8,
    "P":-1.6,"S":-0.8,"T":-0.7,"W":-0.9,"Y":-1.3,"V": 4.2,"X": 0.0,
}

# Molecular weight (Da)
MW = {
    "A": 89.09,"R":174.20,"N":132.12,"D":133.10,"C":121.16,
    "Q":146.15,"E":147.13,"G": 75.03,"H":155.16,"I":131.17,
    "L":131.17,"K":146.19,"M":149.21,"F":165.19,"P":115.13,
    "S":105.09,"T":119.12,"W":204.23,"Y":181.19,"V":117.15,"X":111.0,
}

# Formal charge at pH 7
CHARGE = {
    "A": 0,"R": 1,"N": 0,"D":-1,"C": 0,"Q": 0,"E":-1,"G": 0,
    "H": 0,"I": 0,"L": 0,"K": 1,"M": 0,"F": 0,"P": 0,"S": 0,
    "T": 0,"W": 0,"Y": 0,"V": 0,"X": 0,
}

# Polarity class: 0=nonpolar, 1=polar uncharged, 2=charged
POLAR = {
    "A": 0,"R": 2,"N": 1,"D": 2,"C": 1,"Q": 1,"E": 2,"G": 0,
    "H": 1,"I": 0,"L": 0,"K": 2,"M": 0,"F": 0,"P": 0,"S": 1,
    "T": 1,"W": 0,"Y": 1,"V": 0,"X": 0,
}

# Volume Å³ — Grantham (1974)
VOL = {
    "A": 31,"R":124,"N": 56,"D": 54,"C": 55,"Q": 85,"E": 83,"G":  3,
    "H": 96,"I":111,"L":111,"K":119,"M":105,"F":132,"P": 32,"S": 32,
    "T": 61,"W":170,"Y":136,"V": 84,"X": 60,
}

# Aromaticity
AROM = {"F":1,"H":1,"W":1,"Y":1}

# BLOSUM62 diagonal (conservation proxy)
BLOSUM62 = {
    "A": 4,"R": 5,"N": 6,"D": 6,"C": 9,"Q": 5,"E": 5,"G": 6,
    "H": 8,"I": 4,"L": 4,"K": 5,"M": 5,"F": 6,"P": 7,"S": 4,
    "T": 5,"W":11,"Y": 7,"V": 4,"X":-1,
}

# Grantham distance (selected pairs — Grantham 1974 Science)
_GD = {
    ("A","R"):112,("A","N"):111,("A","D"):126,("A","C"):195,("A","Q"): 91,
    ("A","E"):107,("A","G"): 60,("A","H"): 86,("A","I"): 94,("A","L"): 96,
    ("A","K"):106,("A","M"): 84,("A","F"):113,("A","P"): 27,("A","S"): 99,
    ("A","T"): 58,("A","W"):148,("A","Y"):112,("A","V"): 64,
    ("R","N"): 86,("R","D"): 96,("R","C"):180,("R","Q"): 43,("R","E"): 54,
    ("R","G"):125,("R","H"): 29,("R","I"): 97,("R","L"):102,("R","K"): 26,
    ("R","M"): 91,("R","F"): 97,("R","P"):103,("R","S"):110,("R","T"): 71,
    ("R","W"):101,("R","Y"): 77,("R","V"): 96,
    ("N","D"): 23,("N","C"):139,("N","Q"): 46,("N","E"): 42,("N","G"): 80,
    ("N","H"): 68,("N","I"):149,("N","L"):153,("N","K"): 94,("N","M"):142,
    ("N","F"):158,("N","P"): 91,("N","S"): 46,("N","T"): 65,("N","W"):174,
    ("N","Y"):143,("N","V"):133,
    ("D","C"):154,("D","Q"): 61,("D","E"): 45,("D","G"): 94,("D","H"): 81,
    ("D","I"):168,("D","L"):172,("D","K"):101,("D","M"):160,("D","F"):177,
    ("D","P"):108,("D","S"): 65,("D","T"): 85,("D","W"):181,("D","Y"):160,
    ("D","V"):152,
    ("C","Q"):154,("C","E"):170,("C","G"):159,("C","H"):174,("C","I"):198,
    ("C","L"):198,("C","K"):202,("C","M"):196,("C","F"):205,("C","P"):169,
    ("C","S"):112,("C","T"):149,("C","W"):215,("C","Y"):194,("C","V"):192,
    ("Q","E"): 29,("Q","G"): 87,("Q","H"): 24,("Q","I"):109,("Q","L"):113,
    ("Q","K"):  53,("Q","M"): 101,("Q","F"):116,("Q","P"): 76,("Q","S"): 68,
    ("Q","T"): 42,("Q","W"):130,("Q","Y"):99, ("Q","V"): 96,
    ("E","G"):98, ("E","H"): 40,("E","I"):134,("E","L"):138,("E","K"): 56,
    ("E","M"):126,("E","F"):140,("E","P"):93, ("E","S"): 80,("E","T"): 65,
    ("E","W"):152,("E","Y"):122,("E","V"):121,
    ("G","H"):98, ("G","I"):135,("G","L"):138,("G","K"):127,("G","M"):127,
    ("G","F"):153,("G","P"): 42,("G","S"): 56,("G","T"): 59,("G","W"):184,
    ("G","Y"):147,("G","V"): 109,
    ("H","I"): 94,("H","L"): 99,("H","K"):32, ("H","M"): 87,("H","F"): 100,
    ("H","P"): 77,("H","S"): 89,("H","T"): 47,("H","W"):115,("H","Y"): 83,
    ("H","V"):84,
    ("I","L"):  5,("I","K"):102,("I","M"): 10,("I","F"): 21,("I","P"):95,
    ("I","S"):142,("I","T"):89, ("I","W"): 61,("I","Y"): 33,("I","V"): 29,
    ("L","K"):107,("L","M"): 15,("L","F"): 22,("L","P"):98, ("L","S"):145,
    ("L","T"): 92,("L","W"): 61,("L","Y"): 36,("L","V"): 32,
    ("K","M"):95, ("K","F"):102,("K","P"):103,("K","S"):121,("K","T"): 78,
    ("K","W"):110,("K","Y"): 85,("K","V"):97,
    ("M","F"): 28,("M","P"):87, ("M","S"):135,("M","T"): 81,("M","W"): 67,
    ("M","Y"): 36,("M","V"): 21,
    ("F","P"):114,("F","S"):155,("F","T"):103,("F","W"): 40,("F","Y"): 22,
    ("F","V"): 50,
    ("P","S"): 74,("P","T"): 38,("P","W"):147,("P","Y"):110,("P","V"): 68,
    ("S","T"): 58,("S","W"):177,("S","Y"):144,("S","V"):124,
    ("T","W"):128,("T","Y"): 92,("T","V"): 69,
    ("W","Y"): 37,("W","V"):88,
    ("Y","V"): 55,
}

def grantham(a1: str, a2: str) -> float:
    if a1 == a2:
        return 0.0
    key = (a1, a2) if (a1, a2) in _GD else (a2, a1)
    return float(_GD.get(key, 100.0))   # 100 = fallback for unknown pairs


# THREE_TO_ONE → utils/protein_utils.py'den import edildi


# ═════════════════════════════════════════════════════════════════════════════
# EXTERNAL SCORE LOADERS
# ═════════════════════════════════════════════════════════════════════════════

def load_alphamissense(path: str) -> pd.DataFrame | None:
    """
    Load AlphaMissense scores.
    File has NO header row. Columns are positional:
      0=chrom, 1=pos, 2=ref, 3=alt, 4=genome,
      5=uniprot_id, 6=transcript_id, 7=protein_variant,
      8=am_pathogenicity, 9=am_class
    Joined without coordinates using uniprot_id + protein_variant.
    """
    if not os.path.exists(path):
        log.warning(f"AlphaMissense file not found: {path} — scores will be NaN")
        return None
    log.info(f"Loading AlphaMissense from {path} …")
    col_names = ["chrom","pos","ref","alt","genome",
                 "uniprot_id","transcript_id","protein_variant",
                 "am_pathogenicity","am_class"]
    df = pd.read_csv(path, sep="\t", header=None, names=col_names,
                     usecols=["uniprot_id","protein_variant",
                               "am_pathogenicity","am_class"],
                     comment="#",
                     low_memory=False)
    # Drop any rows where uniprot_id is NaN (residual comment lines)
    df = df[df["uniprot_id"].notna()].copy()
    # Convert score to float
    df["am_pathogenicity"] = pd.to_numeric(df["am_pathogenicity"], errors="coerce")
    # Build lookup key: "UNIPROTID_AAchange" e.g. "P04637_R175H"
    df["am_key"] = df["uniprot_id"].str.strip() + "_" + df["protein_variant"].str.strip()
    log.info(f"  Loaded {len(df):,} AlphaMissense entries")
    return df.set_index("am_key")[["am_pathogenicity","am_class"]]


def load_revel(path: str) -> pd.DataFrame | None:
    """Load REVEL scores. Indexed by chr-pos-ref-alt."""
    if not os.path.exists(path):
        log.warning(f"REVEL file not found: {path} — scores will be NaN")
        return None
    log.info(f"Loading REVEL from {path} …")
    df = pd.read_csv(path, low_memory=False)
    # Columns: chr, hg19_pos, grch38_pos, ref, alt, aaref, aaalt, REVEL, ...
    if "grch38_pos" in df.columns:
        df["revel_key"] = (
            df["chr"].astype(str) + "_" +
            df["grch38_pos"].astype(str) + "_" +
            df["ref"].astype(str) + "_" +
            df["alt"].astype(str)
        )
    else:
        df["revel_key"] = (
            df["chr"].astype(str) + "_" +
            df["hg19_pos"].astype(str) + "_" +
            df["ref"].astype(str) + "_" +
            df["alt"].astype(str)
        )
    log.info(f"  Loaded {len(df):,} REVEL entries")
    return df.set_index("revel_key")[["REVEL"]].rename(columns={"REVEL":"revel_score"})


def load_gnomad_constraints(path: str) -> pd.DataFrame | None:
    """Load gnomAD gene constraint metrics (pLI, LOEUF, mis_z)."""
    if not os.path.exists(path):
        log.warning(f"gnomAD constraint file not found: {path} — scores will be NaN")
        return None
    log.info(f"Loading gnomAD gene constraints from {path} …")
    df = pd.read_csv(path, sep="\t", low_memory=False)
    # Column names vary by version — handle both v2 and v4
    gene_col = "gene" if "gene" in df.columns else "gene_id"
    keep = [gene_col]
    col_map = {}
    # gnomAD v4 uses dot-notation: lof.pLI, lof.oe_ci.upper, mis.z_score
    # gnomAD v2 uses: pLI, oe_lof_upper, mis_z
    for src, dst in [
        ("lof.pLI",          "pli"),
        ("pLI",              "pli"),
        ("pli",              "pli"),
        ("lof.oe_ci.upper",  "loeuf"),
        ("oe_lof_upper",     "loeuf"),
        ("loeuf",            "loeuf"),
        ("mis.z_score",      "mis_z"),
        ("mis_z",            "mis_z"),
        ("z_mis",            "mis_z"),
    ]:
        if src in df.columns and dst not in col_map.values():
            keep.append(src)
            col_map[src] = dst
    df = df[keep].rename(columns=col_map | {gene_col: "GeneSymbol"})
    df = df.drop_duplicates(subset="GeneSymbol")
    log.info(f"  Loaded constraints for {len(df):,} genes")
    return df.set_index("GeneSymbol")


# parse_protein_change → utils/protein_utils.py'den import edildi


# ═════════════════════════════════════════════════════════════════════════════
# FEATURE COMPUTATION — one function per group
# ═════════════════════════════════════════════════════════════════════════════

def feat_group1_sequence(row: pd.Series) -> dict:
    """Group 1: ref nucleotide, alt nucleotide, codon, amino acid substitution."""
    ref_nuc = str(row.get("ReferenceAllele", "N")).upper()[:1]
    alt_nuc = str(row.get("AlternateAllele", "N")).upper()[:1]

    f = {}
    f["ref_nuc"] = ref_nuc
    f["alt_nuc"] = alt_nuc
    # one-hot nucleotides
    for nuc in "ATGC":
        f[f"ref_nuc_{nuc}"] = int(ref_nuc == nuc)
        f[f"alt_nuc_{nuc}"] = int(alt_nuc == nuc)

    # transition / transversion
    purines = {"A","G"}; pyrimidines = {"T","C"}
    f["is_transition"]   = int(
        (ref_nuc in purines and alt_nuc in purines) or
        (ref_nuc in pyrimidines and alt_nuc in pyrimidines)
    )
    f["is_transversion"] = 1 - f["is_transition"]
    f["is_ct_transition"]= int(ref_nuc == "C" and alt_nuc == "T")  # C→T transition (CpG-prone context)

    # amino acid substitution
    ref_aa, alt_aa, aa_pos = parse_protein_change(
        row.get("ProteinChange") or row.get("Name","")
    )
    for aa in AA_CODES:
        f[f"ref_aa_{aa}"] = int(ref_aa == aa)
        f[f"alt_aa_{aa}"] = int(alt_aa == aa)

    f["aa_position"]     = aa_pos if aa_pos > 0 else np.nan
    f["aa_position_log"] = np.log1p(aa_pos) if aa_pos > 0 else np.nan

    # store for downstream groups
    f["_ref_aa"] = ref_aa
    f["_alt_aa"] = alt_aa
    f["ref_aa"]  = ref_aa if ref_aa else None
    f["alt_aa"]  = alt_aa if alt_aa else None

    return f


def feat_group2_context(row: pd.Series) -> dict:
    """
    Group 2: local nucleotide context (±5) and amino acid context (±5).
    In the prototype these are filled with N/X placeholders because
    extracting them requires the reference genome (hg38 FASTA).
    The competition will provide these columns — our schema inference
    protocol will detect and encode them correctly.
    """
    f = {}
    # Nucleotide context window — 5 upstream + 5 downstream = 10 columns
    for i, name in enumerate(
        [f"nuc_ctx_up{5-j}" for j in range(5)] +
        [f"nuc_ctx_dn{j+1}" for j in range(5)]
    ):
        nuc = row.get(f"_nuc_ctx_{i}", "N")
        for b in "ATGC":
            f[f"{name}_{b}"] = int(nuc == b)

    # Amino acid context window — 5 upstream + 5 downstream = 10 columns
    # Each position encoded as 7 physicochemical properties
    for i, name in enumerate(
        [f"aa_ctx_up{5-j}" for j in range(5)] +
        [f"aa_ctx_dn{j+1}" for j in range(5)]
    ):
        aa = row.get(f"_aa_ctx_{i}", "X")
        f[f"{name}_hydro"]  = HYDRO.get(aa, 0.0)
        f[f"{name}_charge"] = CHARGE.get(aa, 0)
        f[f"{name}_polar"]  = POLAR.get(aa, 0)
        f[f"{name}_vol"]    = VOL.get(aa, 60)
        f[f"{name}_mw"]     = MW.get(aa, 111.0)
        f[f"{name}_arom"]   = AROM.get(aa, 0)
        f[f"{name}_blosum"] = BLOSUM62.get(aa, -1)

    # Context summary statistics (mean + std of properties across 10-AA window)
    ctx_hydro  = [HYDRO.get(row.get(f"_aa_ctx_{i}","X"), 0.0) for i in range(10)]
    ctx_charge = [CHARGE.get(row.get(f"_aa_ctx_{i}","X"), 0) for i in range(10)]
    f["ctx_mean_hydro"]  = float(np.mean(ctx_hydro))
    f["ctx_std_hydro"]   = float(np.std(ctx_hydro))
    f["ctx_mean_charge"] = float(np.mean(ctx_charge))
    f["ctx_std_charge"]  = float(np.std(ctx_charge))

    return f


def feat_group3_biochemical(ref_aa: str, alt_aa: str) -> dict:
    """Group 3: signed deltas of physicochemical properties + derived features."""
    def props(aa):
        return {
            "hydro":  HYDRO.get(aa, 0.0),
            "mw":     MW.get(aa, 111.0),
            "charge": CHARGE.get(aa, 0),
            "polar":  POLAR.get(aa, 0),
            "vol":    VOL.get(aa, 60),
            "arom":   AROM.get(aa, 0),
            "blosum": BLOSUM62.get(aa, -1),
        }

    rp = props(ref_aa)
    ap = props(alt_aa)
    f = {}

    # signed deltas (alt - ref)
    for k in rp:
        f[f"delta_{k}"] = ap[k] - rp[k]
    # absolute deltas
    for k in rp:
        f[f"abs_delta_{k}"] = abs(ap[k] - rp[k])

    # Grantham distance
    f["grantham_dist"] = grantham(ref_aa, alt_aa)

    # derived binary flags
    f["is_conservative"]  = int(
        rp["charge"] == ap["charge"] and rp["polar"] == ap["polar"]
    )
    f["is_charge_reversal"] = int(
        (rp["charge"] > 0 and ap["charge"] < 0) or
        (rp["charge"] < 0 and ap["charge"] > 0)
    )
    f["gains_charge"]       = int(rp["charge"] == 0 and ap["charge"] != 0)
    f["loses_charge"]       = int(rp["charge"] != 0 and ap["charge"] == 0)
    f["hydro_to_polar"]     = int(rp["hydro"] > 0 and ap["hydro"] <= 0)
    f["polar_to_hydro"]     = int(rp["hydro"] <= 0 and ap["hydro"] > 0)

    # ref + alt raw values (for model to use directly)
    for k, v in rp.items():
        f[f"ref_{k}"] = v
    for k, v in ap.items():
        f[f"alt_{k}"] = v

    return f


def feat_group4_conservation(row: pd.Series) -> dict:
    """Group 4: conservation scores — joined from external files or NaN."""
    return {
        "phylop_score":    row.get("phylop_score",    np.nan),
        "phastcons_score": row.get("phastcons_score", np.nan),
        "gerp_score":      row.get("gerp_score",      np.nan),
        "pli":             row.get("pli",              np.nan),
        "loeuf":           row.get("loeuf",            np.nan),
        "mis_z":           row.get("mis_z",            np.nan),
    }


def feat_group5_population(row: pd.Series) -> dict:
    """Group 5: population frequency features."""
    af_raw = float(row.get("gnomad_af_global", 0.0) or 0.0)
    f = {
        "maf_global":  af_raw,
        "maf_log10":   float(np.log10(af_raw + 1e-6)),
        "maf_is_rare": int(af_raw < 0.0001),
        "maf_is_very_rare": int(af_raw < 0.000001),
        "maf_is_common": int(af_raw > 0.01),
    }
    for pop in ["afr","amr","eas","fin","nfe","sas"]:
        f[f"maf_{pop}"] = float(row.get(f"gnomad_af_{pop}", np.nan) or np.nan)
    return f


def feat_group6_insilico(row: pd.Series) -> dict:
    """Group 6: in silico risk scores — joined from external files or NaN."""
    scores = {
        "cadd_raw":       row.get("cadd_raw",         np.nan),
        "cadd_phred":     row.get("cadd_phred",       np.nan),
        "revel_score":    row.get("revel_score",      np.nan),
        "polyphen2_hdiv": row.get("polyphen2_hdiv",   np.nan),
        "polyphen2_hvar": row.get("polyphen2_hvar",   np.nan),
        "sift_score":     row.get("sift_score",       np.nan),
        "alphamissense":  row.get("am_pathogenicity", np.nan),
    }
    # Inter-tool agreement
    vals = [v for v in scores.values() if isinstance(v, float) and not np.isnan(v)]
    scores["silico_mean"]     = float(np.mean(vals)) if vals else np.nan
    scores["silico_variance"] = float(np.var(vals))  if len(vals) > 1 else np.nan
    scores["silico_n_tools"]  = len(vals)
    return scores


# ═════════════════════════════════════════════════════════════════════════════
# EXTERNAL SCORE JOINER
# ═════════════════════════════════════════════════════════════════════════════

def load_gene_to_uniprot(path: str = "data/raw/gene_to_uniprot.tsv") -> dict:
    """
    Load gene symbol → UniProt AC mapping.
    File: data/raw/gene_to_uniprot.tsv (tab-separated: UniProtAC, GeneSymbol)
    Generated from UniProt idmapping file (see README).
    Returns empty dict if file not found — AlphaMissense falls back to aa_only.
    """
    if not os.path.exists(path):
        log.warning(f"gene_to_uniprot.tsv not found at {path} — "
                    "AlphaMissense will use aa_only fallback for all variants.")
        return {}
    df = pd.read_csv(path, sep="\t", header=None, names=["uniprot_ac","gene_symbol"],
                     low_memory=False)
    df = df.dropna().drop_duplicates(subset="gene_symbol", keep="first")
    mapping = dict(zip(df["gene_symbol"], df["uniprot_ac"]))
    log.info(f"  Loaded gene→UniProt mapping: {len(mapping):,} genes")
    return mapping


def join_external_scores(df: pd.DataFrame,
                         revel_idx: pd.DataFrame | None,
                         am_idx:    pd.DataFrame | None,
                         gncon_idx: pd.DataFrame | None) -> pd.DataFrame:
    """
    Join AlphaMissense and gnomAD constraints to panel dataframe.
    REVEL is disabled — coordinate-based join violates competition rules.
    AlphaMissense uses gene-aware join: GeneSymbol + AA change → UniProt key.
    """

    # REVEL — DISABLED (requires chr+pos+ref+alt, rules violation at test time)
    # Scores will be NaN in prototype; expected to be provided by committee.
    if revel_idx is not None:
        log.info("  REVEL join disabled — coordinate-dependent. "
                 "Expected in committee feature matrix.")

    # AlphaMissense — gene-aware join using GeneSymbol + protein_variant
    # am_idx is indexed by "UNIPROTID_AAchange" (e.g. "P04637_R175H")
    # We look up UniProt ID from gene symbol via the gene_to_uniprot map,
    # then construct the precise key. Falls back to AA-change-only median
    # if gene→UniProt mapping not available.
    if am_idx is not None:
        gene_to_uniprot = load_gene_to_uniprot()  # dict: GeneSymbol → UniProt AC

        def make_am_key_gene_aware(row):
            pc_col = row.get("ProteinChange") or row.get("Name", "")
            ref, alt, pos = parse_protein_change(str(pc_col))
            if pos < 0:
                return None, "failed"
            aa_change = f"{ref}{pos}{alt}"  # e.g. "R175H"
            gene = row.get("GeneSymbol", "")
            uniprot = gene_to_uniprot.get(gene, "")
            if uniprot:
                precise_key = f"{uniprot}_{aa_change}"
                if precise_key in am_idx.index:
                    return precise_key, "gene_aware"
            # Fallback: match on AA change only (ambiguous — multiple proteins)
            return aa_change, "aa_only"

        keys_methods = df.apply(make_am_key_gene_aware, axis=1)
        df["_am_key"]        = keys_methods.apply(lambda x: x[0])
        df["am_join_method"] = keys_methods.apply(lambda x: x[1])

        # For gene_aware keys: direct index lookup
        mask_precise = df["am_join_method"] == "gene_aware"
        if mask_precise.any():
            df.loc[mask_precise, ["am_pathogenicity","am_class"]] = (
                am_idx.reindex(df.loc[mask_precise, "_am_key"].values)
                .values
            )

        # For aa_only fallback: median across all UniProt entries with same AA change
        mask_fallback = df["am_join_method"] == "aa_only"
        if mask_fallback.any():
            am_by_variant = am_idx.copy()
            am_by_variant.index = am_by_variant.index.str.split("_").str[-1]
            # median per AA change across all proteins
            am_median = am_by_variant.groupby(level=0)["am_pathogenicity"].median()
            df.loc[mask_fallback, "am_pathogenicity"] = (
                df.loc[mask_fallback, "_am_key"].map(am_median).values
            )
            df.loc[mask_fallback, "am_class"] = np.nan  # class unreliable for fallback

        precise_hits = mask_precise.sum()
        fallback_hits = mask_fallback.sum()
        total_hits = df["am_pathogenicity"].notna().sum()
        log.info(f"  AlphaMissense: {precise_hits} gene-aware, "
                 f"{fallback_hits} aa-only fallback, "
                 f"{total_hits} total hits / {len(df)} variants")

    # gnomAD gene constraints — join by GeneSymbol
    if gncon_idx is not None:
        df = df.join(gncon_idx, on="GeneSymbol")
        pli_hits = df['pli'].notna().sum() if 'pli' in df.columns else 0
        log.info(f"  Gene constraints joined: {pli_hits} hits")

    return df


# ═════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def _process_row(row_dict: dict) -> dict:
    """Process a single variant row — runs in parallel worker."""
    row = pd.Series(row_dict)
    f = {}
    g1 = feat_group1_sequence(row)
    ref_aa = g1.pop("_ref_aa")
    alt_aa = g1.pop("_alt_aa")
    f.update(g1)
    f.update(feat_group2_context(row))
    f.update(feat_group3_biochemical(ref_aa, alt_aa))
    f.update(feat_group4_conservation(row))
    f.update(feat_group5_population(row))
    f.update(feat_group6_insilico(row))
    f["_variant_id"]      = row.get("VariationID", "")
    f["_gene"]            = row.get("GeneSymbol", "")
    # Use ProteinChange if available; fall back to Name only if it looks like HGVS (contains 'p.')
    pc_raw = str(row.get("ProteinChange", "") or "")
    if not pc_raw or pc_raw in ("nan", ""):
        name_raw = str(row.get("Name", "") or "")
        pc_raw = name_raw if "p." in name_raw else ""
    f["_protein_change"]  = pc_raw
    f["_label"]           = int(row.get("label", -1))
    f["_panel"]           = row.get("panel", "")
    f["_source"]          = row.get("source", "ClinVar")
    # split assignments from step 03 — passed through untouched
    f["_split_variant"]   = row.get("_split_variant", "")
    f["_split_gene"]      = row.get("_split_gene", "")
    f["_split_gene_note"] = row.get("_split_gene_note", "")
    return f


def annotate_panel(split_path: str, out_path: str,
                   revel_idx, am_idx, gncon_idx) -> None:
    df = pd.read_csv(split_path)
    log.info(f"  Annotating {len(df):,} variants from {split_path}")

    # Join external scores first
    df = join_external_scores(df, revel_idx, am_idx, gncon_idx)

    # Convert to list of dicts for parallel processing
    n_cores = min(cpu_count(), 8)
    log.info(f"  Using {n_cores} cores ...")

    row_dicts = df.to_dict(orient="records")
    rows = Parallel(n_jobs=n_cores, prefer="threads")(
        delayed(_process_row)(r) for r in row_dicts
    )

    out = pd.DataFrame(rows)
    n_feat = len([c for c in out.columns if not c.startswith("_")])
    miss   = out[[c for c in out.columns if not c.startswith("_")]].isnull().mean().mean() * 100
    log.info(f"  → {len(out):,} rows | {n_feat} features | {miss:.1f}% missing overall")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)


def main():
    # Load external score files (optional — NaN if absent)
    revel_idx = load_revel("data/raw/revel_all_chromosomes.csv.gz")
    am_idx    = load_alphamissense("data/raw/AlphaMissense_hg38.tsv.gz")
    gncon_idx = load_gnomad_constraints("data/raw/gnomad_constraints.tsv")

    # One pool file per panel (split schemes are materialised in step 05)
    splits = [(p, "pool") for p in ["general", "cancer", "pah", "cftr"]]

    # Process panels sequentially — large shared objects (AlphaMissense 71M rows)
    # cannot be efficiently forked across processes on macOS.
    # Row-level parallelism inside annotate_panel uses 8 threads instead.
    for panel, split in splits:
        src = f"data/interim/panel_{panel}_{split}.csv"
        dst = f"data/interim/panel_{panel}_{split}_features.csv"
        if not os.path.exists(src):
            log.warning(f"Missing: {src} — run step 03 first")
            continue
        log.info(f"\n{'─'*55}")
        log.info(f"Panel: {panel.upper()}  Split: {split.upper()}")
        annotate_panel(src, dst, revel_idx, am_idx, gncon_idx)

    log.info("\n✓  Feature annotation complete")
    log.info("   All files saved to: data/interim/")


if __name__ == "__main__":
    main()
