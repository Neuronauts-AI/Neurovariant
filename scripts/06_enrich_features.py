"""
STEP 6 — Enrichment Feature Computation
========================================
Reads the final training/test CSVs (named columns, see utils/schema.py)
and appends ~95 enrichment features derived purely from the 25 named columns.

NO external files, NO API calls, NO internet access required.
All lookup tables are hardcoded from published sources (Grantham 1974,
BLOSUM62, Kyte-Doolittle hydrophobicity, standard amino acid biochemistry).

INPUT:
    data/final/<scheme>/train_{panel}.csv
    data/final/<scheme>/test_{panel}.csv

OUTPUT:
    data/enriched/<scheme>/train_{panel}_enriched.csv
    data/enriched/<scheme>/test_{panel}_enriched.csv

The enrichment pipeline is IDENTICAL to what runs on competition day —
the committee provides their dataset with the same 25 named columns,
and this enricher appends the same features before model inference.

ENRICHMENT FEATURE GROUPS
--------------------------
EnrichA — Substitution biochemistry     (from ref_aa + alt_aa)         29 features
EnrichB — Substitution character flags  (from ref_aa + alt_aa)          9 features
EnrichC — Sequence flags                (from ref_nuc + alt_nuc)         3 features
EnrichD — Position derived              (from aa_position)               1 feature
EnrichE — Context window per-position   (from aa_ctx_up1–dn5)           70 features
EnrichF — Context window summary        (from aa_ctx_up1–dn5)            4 features
EnrichG — Codon flags                   (from ref_nuc + alt_nuc + pos)   3 features
EnrichH — Codon usage bias              (from ref_aa + alt_aa)           8 features
EnrichI — Secondary structure proxy     (from ref_aa + alt_aa + ctx)    21 features
# EnrichJ removed — redundant with existing anonymous cols

Source references:
  Grantham (1974) Science 185:862      — pairwise AA distances
  Chou & Fasman (1978) Adv.Enzymol.47  — secondary structure propensity
  Nakamura et al. (2000) NAR 28:292    — human codon usage (RSCU)
  Rost & Sander (1994)                 — relative solvent accessibility
  Kircher et al. (2014) Nat.Genet.     — CADD pathogenicity threshold
  Ioannidis et al. (2016) Nat.Methods  — REVEL pathogenicity threshold
  Adzhubei et al. (2010) Nat.Methods   — PolyPhen2 threshold
  Kumar et al. (2009) Nat.Protoc.      — SIFT threshold
  Cheng et al. (2023) Science          — AlphaMissense threshold

Total: 149 enrichment features

Usage:
    python 06_enrich_features.py [--panels general cancer pah cftr]
                                 [--input-dir data/final]
                                 [--output-dir data/enriched]
                                 [--validate]
"""

import os
import json
import re
import math
import logging
import argparse
import numpy as np
import pandas as pd

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/06_enrich_features.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
# LOOKUP TABLES — hardcoded, no external dependencies
# ═════════════════════════════════════════════════════════════════════════════

# Kyte-Doolittle hydrophobicity scale
HYDRO: dict[str, float] = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
    "X": 0.0,
}

# Residue molecular weight (Da)
MW: dict[str, float] = {
    "A": 89.09,
    "R": 174.20,
    "N": 132.12,
    "D": 133.10,
    "C": 121.16,
    "Q": 146.15,
    "E": 147.13,
    "G": 75.03,
    "H": 155.16,
    "I": 131.17,
    "L": 131.17,
    "K": 146.19,
    "M": 149.21,
    "F": 165.19,
    "P": 115.13,
    "S": 105.09,
    "T": 119.12,
    "W": 204.23,
    "Y": 181.19,
    "V": 117.15,
    "X": 111.0,
}

# Formal charge at pH 7
CHARGE: dict[str, int] = {
    "A": 0,
    "R": 1,
    "N": 0,
    "D": -1,
    "C": 0,
    "Q": 0,
    "E": -1,
    "G": 0,
    "H": 0,
    "I": 0,
    "L": 0,
    "K": 1,
    "M": 0,
    "F": 0,
    "P": 0,
    "S": 0,
    "T": 0,
    "W": 0,
    "Y": 0,
    "V": 0,
    "X": 0,
}

# Polarity class: 0=nonpolar, 1=polar uncharged, 2=charged
POLAR: dict[str, int] = {
    "A": 0,
    "R": 2,
    "N": 1,
    "D": 2,
    "C": 1,
    "Q": 1,
    "E": 2,
    "G": 0,
    "H": 1,
    "I": 0,
    "L": 0,
    "K": 2,
    "M": 0,
    "F": 0,
    "P": 0,
    "S": 1,
    "T": 1,
    "W": 0,
    "Y": 1,
    "V": 0,
    "X": 0,
}

# Volume Å³ — Grantham (1974)
VOL: dict[str, int] = {
    "A": 31,
    "R": 124,
    "N": 56,
    "D": 54,
    "C": 55,
    "Q": 85,
    "E": 83,
    "G": 3,
    "H": 96,
    "I": 111,
    "L": 111,
    "K": 119,
    "M": 105,
    "F": 132,
    "P": 32,
    "S": 32,
    "T": 61,
    "W": 170,
    "Y": 136,
    "V": 84,
    "X": 60,
}

# Aromaticity (binary)
AROM: dict[str, int] = {"F": 1, "H": 1, "W": 1, "Y": 1}

# BLOSUM62 diagonal (self-substitution conservation score)
BLOSUM62: dict[str, int] = {
    "A": 4,
    "R": 5,
    "N": 6,
    "D": 6,
    "C": 9,
    "Q": 5,
    "E": 5,
    "G": 6,
    "H": 8,
    "I": 4,
    "L": 4,
    "K": 5,
    "M": 5,
    "F": 6,
    "P": 7,
    "S": 4,
    "T": 5,
    "W": 11,
    "Y": 7,
    "V": 4,
    "X": -1,
}

# Grantham pairwise distances — Grantham (1974) Science 185:862–864
_GD: dict[tuple[str, str], int] = {
    ("A", "R"): 112,
    ("A", "N"): 111,
    ("A", "D"): 126,
    ("A", "C"): 195,
    ("A", "Q"): 91,
    ("A", "E"): 107,
    ("A", "G"): 60,
    ("A", "H"): 86,
    ("A", "I"): 94,
    ("A", "L"): 96,
    ("A", "K"): 106,
    ("A", "M"): 84,
    ("A", "F"): 113,
    ("A", "P"): 27,
    ("A", "S"): 99,
    ("A", "T"): 58,
    ("A", "W"): 148,
    ("A", "Y"): 112,
    ("A", "V"): 64,
    ("R", "N"): 86,
    ("R", "D"): 96,
    ("R", "C"): 180,
    ("R", "Q"): 43,
    ("R", "E"): 54,
    ("R", "G"): 125,
    ("R", "H"): 29,
    ("R", "I"): 97,
    ("R", "L"): 102,
    ("R", "K"): 26,
    ("R", "M"): 91,
    ("R", "F"): 97,
    ("R", "P"): 103,
    ("R", "S"): 110,
    ("R", "T"): 71,
    ("R", "W"): 101,
    ("R", "Y"): 77,
    ("R", "V"): 96,
    ("N", "D"): 23,
    ("N", "C"): 139,
    ("N", "Q"): 46,
    ("N", "E"): 42,
    ("N", "G"): 80,
    ("N", "H"): 68,
    ("N", "I"): 149,
    ("N", "L"): 153,
    ("N", "K"): 94,
    ("N", "M"): 142,
    ("N", "F"): 158,
    ("N", "P"): 91,
    ("N", "S"): 46,
    ("N", "T"): 65,
    ("N", "W"): 174,
    ("N", "Y"): 143,
    ("N", "V"): 133,
    ("D", "C"): 154,
    ("D", "Q"): 61,
    ("D", "E"): 45,
    ("D", "G"): 94,
    ("D", "H"): 81,
    ("D", "I"): 168,
    ("D", "L"): 172,
    ("D", "K"): 101,
    ("D", "M"): 160,
    ("D", "F"): 177,
    ("D", "P"): 108,
    ("D", "S"): 65,
    ("D", "T"): 85,
    ("D", "W"): 181,
    ("D", "Y"): 160,
    ("D", "V"): 152,
    ("C", "Q"): 154,
    ("C", "E"): 170,
    ("C", "G"): 159,
    ("C", "H"): 174,
    ("C", "I"): 198,
    ("C", "L"): 198,
    ("C", "K"): 202,
    ("C", "M"): 196,
    ("C", "F"): 205,
    ("C", "P"): 169,
    ("C", "S"): 112,
    ("C", "T"): 149,
    ("C", "W"): 215,
    ("C", "Y"): 194,
    ("C", "V"): 192,
    ("Q", "E"): 29,
    ("Q", "G"): 87,
    ("Q", "H"): 24,
    ("Q", "I"): 109,
    ("Q", "L"): 113,
    ("Q", "K"): 53,
    ("Q", "M"): 101,
    ("Q", "F"): 116,
    ("Q", "P"): 76,
    ("Q", "S"): 68,
    ("Q", "T"): 42,
    ("Q", "W"): 130,
    ("Q", "Y"): 99,
    ("Q", "V"): 96,
    ("E", "G"): 98,
    ("E", "H"): 40,
    ("E", "I"): 134,
    ("E", "L"): 138,
    ("E", "K"): 56,
    ("E", "M"): 126,
    ("E", "F"): 140,
    ("E", "P"): 93,
    ("E", "S"): 80,
    ("E", "T"): 65,
    ("E", "W"): 152,
    ("E", "Y"): 122,
    ("E", "V"): 121,
    ("G", "H"): 98,
    ("G", "I"): 135,
    ("G", "L"): 138,
    ("G", "K"): 127,
    ("G", "M"): 127,
    ("G", "F"): 153,
    ("G", "P"): 42,
    ("G", "S"): 56,
    ("G", "T"): 59,
    ("G", "W"): 184,
    ("G", "Y"): 147,
    ("G", "V"): 109,
    ("H", "I"): 94,
    ("H", "L"): 99,
    ("H", "K"): 32,
    ("H", "M"): 87,
    ("H", "F"): 100,
    ("H", "P"): 77,
    ("H", "S"): 89,
    ("H", "T"): 47,
    ("H", "W"): 115,
    ("H", "Y"): 83,
    ("H", "V"): 84,
    ("I", "L"): 5,
    ("I", "K"): 102,
    ("I", "M"): 10,
    ("I", "F"): 21,
    ("I", "P"): 95,
    ("I", "S"): 142,
    ("I", "T"): 89,
    ("I", "W"): 61,
    ("I", "Y"): 33,
    ("I", "V"): 29,
    ("L", "K"): 107,
    ("L", "M"): 15,
    ("L", "F"): 22,
    ("L", "P"): 98,
    ("L", "S"): 145,
    ("L", "T"): 92,
    ("L", "W"): 61,
    ("L", "Y"): 36,
    ("L", "V"): 32,
    ("K", "M"): 95,
    ("K", "F"): 102,
    ("K", "P"): 103,
    ("K", "S"): 121,
    ("K", "T"): 78,
    ("K", "W"): 110,
    ("K", "Y"): 85,
    ("K", "V"): 97,
    ("M", "F"): 28,
    ("M", "P"): 87,
    ("M", "S"): 135,
    ("M", "T"): 81,
    ("M", "W"): 67,
    ("M", "Y"): 36,
    ("M", "V"): 21,
    ("F", "P"): 114,
    ("F", "S"): 155,
    ("F", "T"): 103,
    ("F", "W"): 40,
    ("F", "Y"): 22,
    ("F", "V"): 50,
    ("P", "S"): 74,
    ("P", "T"): 38,
    ("P", "W"): 147,
    ("P", "Y"): 110,
    ("P", "V"): 68,
    ("S", "T"): 58,
    ("S", "W"): 177,
    ("S", "Y"): 144,
    ("S", "V"): 124,
    ("T", "W"): 128,
    ("T", "Y"): 92,
    ("T", "V"): 69,
    ("W", "Y"): 37,
    ("W", "V"): 88,
    ("Y", "V"): 55,
}

# ── EnrichH: Human codon usage frequency (RSCU) ───────────────────────────────
# Source: Nakamura et al. (2000) Nucleic Acids Res. 28:292 — Homo sapiens
# RSCU > 1.0 = preferred codon in highly expressed genes
# RSCU < 1.0 = rare/avoided codon
_CODON_USAGE: dict[str, float] = {
    "TTT": 0.45,
    "TTC": 1.55,
    "TTA": 0.07,
    "TTG": 0.13,
    "CTT": 0.13,
    "CTC": 0.20,
    "CTA": 0.07,
    "CTG": 1.40,
    "ATT": 0.36,
    "ATC": 1.46,
    "ATA": 0.18,
    "ATG": 1.00,
    "GTT": 0.18,
    "GTC": 0.24,
    "GTA": 0.11,
    "GTG": 1.47,
    "TCT": 0.15,
    "TCC": 0.22,
    "TCA": 0.15,
    "TCG": 0.06,
    "AGT": 0.15,
    "AGC": 0.27,
    "CCT": 0.28,
    "CCC": 0.33,
    "CCA": 0.27,
    "CCG": 0.11,
    "ACT": 0.25,
    "ACC": 0.36,
    "ACA": 0.28,
    "ACG": 0.11,
    "GCT": 0.26,
    "GCC": 0.40,
    "GCA": 0.23,
    "GCG": 0.11,
    "TAT": 0.43,
    "TAC": 1.57,
    "TAA": 0.28,
    "TAG": 0.20,
    "TGA": 0.52,
    "CAT": 0.41,
    "CAC": 1.59,
    "CAA": 0.25,
    "CAG": 1.75,
    "AAT": 0.46,
    "AAC": 1.54,
    "AAA": 0.43,
    "AAG": 1.57,
    "GAT": 0.46,
    "GAC": 1.54,
    "GAA": 0.42,
    "GAG": 1.58,
    "TGT": 0.45,
    "TGC": 1.55,
    "TGG": 1.00,
    "CGT": 0.08,
    "CGC": 0.19,
    "CGA": 0.11,
    "CGG": 0.21,
    "AGA": 0.20,
    "AGG": 0.21,
    "GGT": 0.16,
    "GGC": 0.34,
    "GGA": 0.25,
    "GGG": 0.25,
}

# ── EnrichI: Chou-Fasman secondary structure propensity ───────────────────────
# Source: Chou & Fasman (1978) Adv. Enzymol. 47:45–148
# P_alpha > 1.0 = helix former  |  P_alpha < 1.0 = helix breaker
# P_beta  > 1.0 = sheet former  |  P_beta  < 1.0 = sheet breaker
# P_turn  > 1.0 = turn former
_CF_ALPHA: dict[str, float] = {
    "A": 1.42,
    "R": 0.98,
    "N": 0.67,
    "D": 1.01,
    "C": 0.70,
    "Q": 1.11,
    "E": 1.51,
    "G": 0.57,
    "H": 1.00,
    "I": 1.08,
    "L": 1.21,
    "K": 1.16,
    "M": 1.45,
    "F": 1.13,
    "P": 0.57,
    "S": 0.77,
    "T": 0.83,
    "W": 1.08,
    "Y": 0.69,
    "V": 1.06,
    "X": 1.00,
}
_CF_BETA: dict[str, float] = {
    "A": 0.83,
    "R": 0.93,
    "N": 0.89,
    "D": 0.54,
    "C": 1.19,
    "Q": 1.10,
    "E": 0.37,
    "G": 0.75,
    "H": 0.87,
    "I": 1.60,
    "L": 1.30,
    "K": 0.74,
    "M": 1.05,
    "F": 1.38,
    "P": 0.55,
    "S": 0.75,
    "T": 1.19,
    "W": 1.37,
    "Y": 1.47,
    "V": 1.70,
    "X": 1.00,
}
_CF_TURN: dict[str, float] = {
    "A": 0.66,
    "R": 0.95,
    "N": 1.56,
    "D": 1.46,
    "C": 1.19,
    "Q": 0.98,
    "E": 0.74,
    "G": 1.56,
    "H": 0.95,
    "I": 0.47,
    "L": 0.59,
    "K": 1.01,
    "M": 0.60,
    "F": 0.60,
    "P": 1.52,
    "S": 1.43,
    "T": 0.96,
    "W": 0.96,
    "Y": 1.14,
    "V": 0.50,
    "X": 1.00,
}

# Relative solvent accessibility propensity — buried (0) vs exposed (1)
# Source: Rost & Sander (1994) derived values
_RSA: dict[str, float] = {
    "A": 0.25,
    "R": 0.84,
    "N": 0.63,
    "D": 0.62,
    "C": 0.22,
    "Q": 0.69,
    "E": 0.72,
    "G": 0.42,
    "H": 0.54,
    "I": 0.15,
    "L": 0.18,
    "K": 0.87,
    "M": 0.30,
    "F": 0.22,
    "P": 0.53,
    "S": 0.56,
    "T": 0.46,
    "W": 0.29,
    "Y": 0.42,
    "V": 0.16,
    "X": 0.50,
}

# Context window position names — must match pipeline exactly
# 04c writes: aa_ctx_up5, aa_ctx_up4, ..., aa_ctx_up1, aa_ctx_dn1, ..., aa_ctx_dn5
CTX_NAMES: list[str] = [f"up{5 - i}" for i in range(5)] + [  # up5, up4, up3, up2, up1
    f"dn{i + 1}" for i in range(5)
]  # dn1, dn2, dn3, dn4, dn5

# Standard nucleotide codon table (DNA, sense strand)
# Maps codon string → amino acid one-letter code
_CODON_TABLE: dict[str, str] = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}
# Build reverse: AA → list of codons
_AA_TO_CODONS: dict[str, list[str]] = {}
for codon, aa in _CODON_TABLE.items():
    _AA_TO_CODONS.setdefault(aa, []).append(codon)

# Precompute mean RSCU per amino acid across all its synonymous codons
_AA_MEAN_RSCU: dict[str, float] = {}
for _aa, _codons in _AA_TO_CODONS.items():
    _vals = [_CODON_USAGE.get(c, 1.0) for c in _codons if c in _CODON_USAGE]
    _AA_MEAN_RSCU[_aa] = float(sum(_vals) / len(_vals)) if _vals else 1.0


# ═════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════


def grantham(a1: str, a2: str) -> float:
    """Return Grantham distance between two amino acids (0 if identical)."""
    if a1 == a2:
        return 0.0
    key = (a1, a2) if (a1, a2) in _GD else (a2, a1)
    return float(_GD.get(key, 100.0))  # 100 = fallback for unknown pairs


def safe_aa(aa: str) -> str:
    """Normalise amino acid letter; return 'X' if unknown."""
    aa = str(aa).strip().upper()[:1] if aa else "X"
    return aa if aa in HYDRO else "X"


def safe_nuc(nuc: str) -> str:
    """Normalise nucleotide letter; return 'N' if unknown."""
    nuc = str(nuc).strip().upper()[:1] if nuc else "N"
    return nuc if nuc in "ATGC" else "N"


def codon_position_from_aa_pos(aa_pos: int) -> int:
    """
    Return codon position (1, 2 or 3) of a missense SNV.

    For a coding SNV at amino acid position P, the codon occupies
    CDS nucleotides (3P-2), (3P-1), (3P).  Without the exact CDS
    offset we can only infer the codon position from aa_pos modulo
    information — however, since all three positions are equally
    likely across a random missense, we encode it as aa_pos % 3
    mapped to {1,2,3} as an approximate proxy.
    """
    if aa_pos <= 0:
        return 0
    return ((aa_pos - 1) % 3) + 1


# ═════════════════════════════════════════════════════════════════════════════
# ENRICHMENT FEATURE FUNCTIONS
# Each function takes a pandas Series (one variant row) and returns a dict.
# ═════════════════════════════════════════════════════════════════════════════


def enrich_A_substitution_biochem(row: pd.Series) -> dict:
    """
    EnrichA — Physicochemical delta features (ref_aa + alt_aa).

    For each property: signed delta (alt − ref) and absolute delta.
    Also stores raw ref and alt property values.

    Properties: hydrophobicity, molecular weight, charge, polarity,
                volume, aromaticity, BLOSUM62 diagonal
    """
    ref = safe_aa(row.get("ref_aa", "X"))
    alt = safe_aa(row.get("alt_aa", "X"))

    def props(aa: str) -> dict:
        return {
            "hydro": HYDRO.get(aa, 0.0),
            "mw": MW.get(aa, 111.0),
            "charge": float(CHARGE.get(aa, 0)),
            "polar": float(POLAR.get(aa, 0)),
            "vol": float(VOL.get(aa, 60)),
            "arom": float(AROM.get(aa, 0)),
            "blosum": float(BLOSUM62.get(aa, -1)),
        }

    rp = props(ref)
    ap = props(alt)
    f: dict = {}

    for k in rp:
        f[f"enrich_delta_{k}"] = ap[k] - rp[k]
        f[f"enrich_abs_delta_{k}"] = abs(ap[k] - rp[k])
        f[f"enrich_ref_{k}"] = rp[k]
        f[f"enrich_alt_{k}"] = ap[k]

    f["enrich_grantham_dist"] = grantham(ref, alt)

    return f


def enrich_B_substitution_flags(row: pd.Series) -> dict:
    """
    EnrichB — Categorical substitution character flags (ref_aa + alt_aa).

    Captures charge-altering, polarity-altering, and conservative/radical
    nature of the substitution — distinct from the raw delta values.
    """
    ref = safe_aa(row.get("ref_aa", "X"))
    alt = safe_aa(row.get("alt_aa", "X"))

    rh = HYDRO.get(ref, 0.0)
    ah = HYDRO.get(alt, 0.0)
    rc = CHARGE.get(ref, 0)
    ac = CHARGE.get(alt, 0)
    rp = POLAR.get(ref, 0)
    ap = POLAR.get(alt, 0)
    gd = grantham(ref, alt)

    return {
        "enrich_is_conservative": int(rc == ac and rp == ap),
        "enrich_is_radical": int(gd > 100),
        "enrich_is_charge_reversal": int((rc > 0 and ac < 0) or (rc < 0 and ac > 0)),
        "enrich_gains_charge": int(rc == 0 and ac != 0),
        "enrich_loses_charge": int(rc != 0 and ac == 0),
        "enrich_hydro_to_polar": int(rh > 0 and ah <= 0),
        "enrich_polar_to_hydro": int(rh <= 0 and ah > 0),
        "enrich_same_polarity_class": int(rp == ap),
        "enrich_same_charge_sign": int((rc >= 0) == (ac >= 0)),
    }


def enrich_C_sequence_flags(row: pd.Series) -> dict:
    """
    EnrichC — Nucleotide substitution class flags (ref_nuc + alt_nuc).

    Transition, transversion, and C→T specific flag (relevant for
    mutational signature SBS1/SBS5 enrichment at CpG sites).
    """
    ref = safe_nuc(row.get("ref_nuc", "N"))
    alt = safe_nuc(row.get("alt_nuc", "N"))

    purines = {"A", "G"}
    pyrimidines = {"T", "C"}

    is_ts = int(
        (ref in purines and alt in purines)
        or (ref in pyrimidines and alt in pyrimidines)
    )
    return {
        "enrich_is_transition": is_ts,
        "enrich_is_transversion": 1 - is_ts,
        "enrich_is_ct_transition": int(ref == "C" and alt == "T"),
    }


def enrich_D_position_derived(row: pd.Series) -> dict:
    """
    EnrichD — Position-derived features (aa_position).

    Log-transformed position reduces the right-skewed distribution
    of variant positions in large proteins.
    """
    pos = row.get("aa_position", np.nan)
    try:
        pos = float(pos)
    except (TypeError, ValueError):
        pos = np.nan

    return {
        "enrich_aa_position_log": math.log1p(pos) if (pos and pos > 0) else np.nan,
    }


def enrich_E_context_per_position(row: pd.Series) -> dict:
    """
    EnrichE — Per-position biochemical encoding of ±5 AA context window.

    For each of the 10 context positions (up5..up1, dn1..dn5), encodes:
    hydrophobicity, charge, polarity, volume, MW, aromaticity, BLOSUM62 diagonal.

    Input columns: aa_ctx_up5, aa_ctx_up4, ..., aa_ctx_dn5  (10 named cols)
    Output: 70 features (10 positions × 7 properties)
    """
    f: dict = {}
    for name in CTX_NAMES:
        aa = safe_aa(row.get(f"aa_ctx_{name}", "X"))
        f[f"enrich_ctx_{name}_hydro"] = HYDRO.get(aa, 0.0)
        f[f"enrich_ctx_{name}_charge"] = float(CHARGE.get(aa, 0))
        f[f"enrich_ctx_{name}_polar"] = float(POLAR.get(aa, 0))
        f[f"enrich_ctx_{name}_vol"] = float(VOL.get(aa, 60))
        f[f"enrich_ctx_{name}_mw"] = MW.get(aa, 111.0)
        f[f"enrich_ctx_{name}_arom"] = float(AROM.get(aa, 0))
        f[f"enrich_ctx_{name}_blosum"] = float(BLOSUM62.get(aa, -1))
    return f


def enrich_F_context_summary(row: pd.Series) -> dict:
    """
    EnrichF — Summary statistics of ±5 AA context window.

    Mean and std of hydrophobicity and charge across the 10-position window.
    Captures local sequence environment without relying on gene identity.

    Input columns: aa_ctx_up5..dn5  (10 named cols)
    Output: 4 features
    """
    aas = [safe_aa(row.get(f"aa_ctx_{n}", "X")) for n in CTX_NAMES]
    hydros = [HYDRO.get(aa, 0.0) for aa in aas]
    charges = [float(CHARGE.get(aa, 0)) for aa in aas]

    return {
        "enrich_ctx_mean_hydro": float(np.mean(hydros)),
        "enrich_ctx_std_hydro": float(np.std(hydros)),
        "enrich_ctx_mean_charge": float(np.mean(charges)),
        "enrich_ctx_std_charge": float(np.std(charges)),
    }


def enrich_G_codon_flags(row: pd.Series) -> dict:
    """
    EnrichG — Codon position and synonymous codon count (ref_nuc + alt_nuc + aa_position).

    codon_position: approximate codon position (1,2,3) derived from aa_position % 3.
                    This is an approximation since we don't have the exact CDS offset,
                    but is informative as a relative proxy.
    ref_codon_degeneracy: number of synonymous codons for ref_aa (codon redundancy).
    alt_codon_degeneracy: number of synonymous codons for alt_aa.

    Input columns: ref_nuc, alt_nuc, aa_position, ref_aa, alt_aa
    Output: 3 features
    """
    pos = row.get("aa_position", np.nan)
    ref_aa = safe_aa(row.get("ref_aa", "X"))
    alt_aa = safe_aa(row.get("alt_aa", "X"))

    try:
        pos_int = (
            int(float(pos))
            if pos and not (isinstance(pos, float) and math.isnan(pos))
            else 0
        )
    except (TypeError, ValueError):
        pos_int = 0

    codon_pos = codon_position_from_aa_pos(pos_int)
    ref_deg = len(_AA_TO_CODONS.get(ref_aa, []))
    alt_deg = len(_AA_TO_CODONS.get(alt_aa, []))

    return {
        "enrich_codon_position": float(codon_pos),
        "enrich_ref_codon_degeneracy": float(ref_deg),
        "enrich_alt_codon_degeneracy": float(alt_deg),
    }


def enrich_H_codon_usage_bias(row: pd.Series) -> dict:
    """
    EnrichH — Codon usage bias features (ref_aa + alt_aa + ref_nuc + alt_nuc).

    Human genes under strong selection use preferred codons (high RSCU).
    A missense variant that simultaneously changes the amino acid AND switches
    to a rare codon creates two levels of disruption: protein function AND
    translational efficiency. This signal is absent from all standard tools.

    Source: Nakamura et al. (2000) Nucleic Acids Res. 28:292

    Features:
      ref_mean_rscu       — mean RSCU across all synonymous codons for ref_aa
      alt_mean_rscu       — mean RSCU across all synonymous codons for alt_aa
      delta_mean_rscu     — alt − ref (negative = substitution moves to rarer codons)
      abs_delta_mean_rscu — magnitude of change
      ref_max_rscu        — most preferred codon available for ref_aa
      alt_max_rscu        — most preferred codon available for alt_aa
      delta_max_rscu      — change in best available codon
      codon_usage_pressure— ref_max_rscu − ref_mean_rscu (how much better the
                            best codon is vs average — measures selection pressure)

    Input columns: ref_aa, alt_aa
    Output: 8 features
    """
    ref = safe_aa(row.get("ref_aa", "X"))
    alt = safe_aa(row.get("alt_aa", "X"))

    def aa_rscu_stats(aa: str) -> tuple[float, float]:
        codons = _AA_TO_CODONS.get(aa, [])
        vals = [_CODON_USAGE.get(c, 1.0) for c in codons if c in _CODON_USAGE]
        if not vals:
            return 1.0, 1.0
        return float(sum(vals) / len(vals)), float(max(vals))

    ref_mean, ref_max = aa_rscu_stats(ref)
    alt_mean, alt_max = aa_rscu_stats(alt)

    return {
        "enrich_ref_mean_rscu": ref_mean,
        "enrich_alt_mean_rscu": alt_mean,
        "enrich_delta_mean_rscu": alt_mean - ref_mean,
        "enrich_abs_delta_mean_rscu": abs(alt_mean - ref_mean),
        "enrich_ref_max_rscu": ref_max,
        "enrich_alt_max_rscu": alt_max,
        "enrich_delta_max_rscu": alt_max - ref_max,
        "enrich_codon_usage_pressure": ref_max - ref_mean,
    }


def enrich_I_secondary_structure(row: pd.Series) -> dict:
    """
    EnrichI — Secondary structure propensity features (ref_aa + alt_aa + context).

    Chou-Fasman parameters encode each amino acid's intrinsic tendency to
    adopt alpha-helix, beta-sheet, or turn conformations. A missense variant
    that introduces a helix-breaking residue (e.g. Proline, Glycine) into a
    helix-forming context is structurally disruptive in a way that charge
    and hydrophobicity deltas alone cannot capture.

    Source: Chou & Fasman (1978) Adv. Enzymol. 47:45–148
            Rost & Sander (1994) for RSA values

    Features (ref_aa + alt_aa):
      ref/alt Chou-Fasman alpha, beta, turn propensities
      signed and absolute deltas for each
      ref/alt RSA (surface accessibility propensity)
      delta_rsa, abs_delta_rsa
      is_helix_breaker_introduced  — alt has P_alpha < 0.7 (Pro, Gly)
      is_sheet_breaker_introduced  — alt has P_beta  < 0.6
      burial_change                — direction of RSA change (+1 exposed, -1 buried)

    Context window (aa_ctx_up1–dn5):
      ctx_mean_alpha  — mean helix propensity of ±5 neighbourhood
      ctx_mean_beta   — mean sheet propensity of ±5 neighbourhood
      ctx_helix_frac  — fraction of context positions with P_alpha > 1.0
      ctx_sheet_frac  — fraction of context positions with P_beta  > 1.0

    Input columns: ref_aa, alt_aa, aa_ctx_up1–dn5
    Output: 21 features
    """
    ref = safe_aa(row.get("ref_aa", "X"))
    alt = safe_aa(row.get("alt_aa", "X"))

    r_alpha = _CF_ALPHA.get(ref, 1.0)
    r_beta = _CF_BETA.get(ref, 1.0)
    r_turn = _CF_TURN.get(ref, 1.0)
    r_rsa = _RSA.get(ref, 0.5)

    a_alpha = _CF_ALPHA.get(alt, 1.0)
    a_beta = _CF_BETA.get(alt, 1.0)
    a_turn = _CF_TURN.get(alt, 1.0)
    a_rsa = _RSA.get(alt, 0.5)

    # Context window secondary structure environment
    ctx_aas = [safe_aa(row.get(f"aa_ctx_{n}", "X")) for n in CTX_NAMES]
    ctx_alphas = [_CF_ALPHA.get(aa, 1.0) for aa in ctx_aas]
    ctx_betas = [_CF_BETA.get(aa, 1.0) for aa in ctx_aas]

    ctx_mean_alpha = float(np.mean(ctx_alphas))
    ctx_mean_beta = float(np.mean(ctx_betas))
    ctx_helix_frac = float(sum(1 for v in ctx_alphas if v > 1.0) / len(ctx_alphas))
    ctx_sheet_frac = float(sum(1 for v in ctx_betas if v > 1.0) / len(ctx_betas))

    # Burial change direction: +1 if variant moves to more exposed, -1 if more buried
    burial_change = 1 if a_rsa > r_rsa else (-1 if a_rsa < r_rsa else 0)

    return {
        # Raw propensities
        "enrich_ref_cf_alpha": r_alpha,
        "enrich_ref_cf_beta": r_beta,
        "enrich_ref_cf_turn": r_turn,
        "enrich_ref_rsa": r_rsa,
        "enrich_alt_cf_alpha": a_alpha,
        "enrich_alt_cf_beta": a_beta,
        "enrich_alt_cf_turn": a_turn,
        "enrich_alt_rsa": a_rsa,
        # Deltas
        "enrich_delta_cf_alpha": a_alpha - r_alpha,
        "enrich_abs_delta_cf_alpha": abs(a_alpha - r_alpha),
        "enrich_delta_cf_beta": a_beta - r_beta,
        "enrich_abs_delta_cf_beta": abs(a_beta - r_beta),
        "enrich_delta_cf_turn": a_turn - r_turn,
        "enrich_delta_rsa": a_rsa - r_rsa,
        "enrich_abs_delta_rsa": abs(a_rsa - r_rsa),
        # Structural disruption flags
        "enrich_is_helix_breaker_intro": int(r_alpha >= 1.0 and a_alpha < 0.70),
        "enrich_is_sheet_breaker_intro": int(r_beta >= 1.0 and a_beta < 0.60),
        "enrich_burial_change": float(burial_change),
        # Context structural environment
        "enrich_ctx_mean_alpha": ctx_mean_alpha,
        "enrich_ctx_mean_beta": ctx_mean_beta,
        "enrich_ctx_helix_frac": ctx_helix_frac,
        "enrich_ctx_sheet_frac": ctx_sheet_frac,
    }


# ═════════════════════════════════════════════════════════════════════════════
# MAIN ENRICHMENT PIPELINE
# ═════════════════════════════════════════════════════════════════════════════


def enrich_J_tool_consensus(row: pd.Series) -> dict:
    """
    EnrichJ — In silico tool consensus features (from anonymous col_XXXX).

    silico_mean, silico_n_tools, silico_variance already exist as anonymous
    cols computed by 04_annotate_features.py. This group adds complementary
    features that encode per-tool pathogenicity calls and inter-tool agreement
    patterns — signals the model cannot extract from individual raw scores.

    THRESHOLDS (published clinical interpretation cutoffs):
      CADD phred   >= 20   → likely damaging  (Kircher et al. 2014)
      REVEL        >= 0.75 → likely pathogenic (Ioannidis et al. 2016)
      PolyPhen2    >= 0.85 → probably damaging (Adzhubei et al. 2010)
      SIFT         <= 0.05 → damaging          (Kumar et al. 2009)
      AlphaMissense >= 0.564 → likely pathogenic (Cheng et al. 2023)

    Column mapping (from column_schema.json — deterministic alphabetical order):
      col_0039 → cadd_phred
      col_0071 → polyphen2_hdiv
      col_0072 → polyphen2_hvar
      col_0103 → revel_score
      col_0104 → sift_score
      col_0105 → silico_mean      (already exists — used as input only)
      col_0106 → silico_n_tools   (already exists — used as input only)
      col_0107 → silico_variance  (already exists — used as input only)
      col_0008 → am_pathogenicity (AlphaMissense)

    Input columns: anonymous col_XXXX columns listed above
    Output: 16 features
    """
    # Read raw scores from anonymous columns
    cadd = row.get("col_0039", np.nan)
    pp2_h = row.get("col_0071", np.nan)
    pp2_v = row.get("col_0072", np.nan)
    revel = row.get("col_0103", np.nan)
    sift = row.get("col_0104", np.nan)
    am = row.get("col_0008", np.nan)

    # Already-computed meta-scores
    s_mean = row.get("col_0105", np.nan)
    s_ntools = row.get("col_0106", np.nan)
    s_var = row.get("col_0107", np.nan)

    def safe_float(v) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan

    cadd = safe_float(cadd)
    pp2_h = safe_float(pp2_h)
    pp2_v = safe_float(pp2_v)
    revel = safe_float(revel)
    sift = safe_float(sift)
    am = safe_float(am)
    s_var = safe_float(s_var)

    # ── Per-tool binary calls at published thresholds ─────────────────────
    call_cadd = int(cadd >= 20.0) if not np.isnan(cadd) else np.nan
    call_revel = int(revel >= 0.75) if not np.isnan(revel) else np.nan
    call_pp2h = int(pp2_h >= 0.85) if not np.isnan(pp2_h) else np.nan
    call_pp2v = int(pp2_v >= 0.85) if not np.isnan(pp2_v) else np.nan
    call_sift = int(sift <= 0.05) if not np.isnan(sift) else np.nan
    call_am = int(am >= 0.564) if not np.isnan(am) else np.nan

    # ── Consensus counting ────────────────────────────────────────────────
    calls = [
        c
        for c in [call_cadd, call_revel, call_pp2h, call_pp2v, call_sift, call_am]
        if not (isinstance(c, float) and np.isnan(c))
    ]

    n_available = len(calls)
    n_patho_calls = sum(calls)
    n_benign_calls = n_available - n_patho_calls

    # Fraction of available tools calling pathogenic
    patho_frac = float(n_patho_calls / n_available) if n_available > 0 else np.nan

    # Unanimous flags
    unanimous_patho = int(n_available >= 3 and n_patho_calls == n_available)
    unanimous_benign = int(n_available >= 3 and n_benign_calls == n_available)

    # ── CADD-REVEL agreement (two most validated tools) ───────────────────
    # Both available and both agree on direction
    cadd_revel_agree = np.nan
    if not np.isnan(cadd) and not np.isnan(revel):
        cadd_revel_agree = int(call_cadd == call_revel)

    # ── Score spread (max - min across available tools, normalised 0-1) ──
    # Normalise each score to [0,1] direction (higher = more pathogenic)
    normed = []
    if not np.isnan(cadd):
        normed.append(min(cadd / 40.0, 1.0))  # CADD max ~40
    if not np.isnan(revel):
        normed.append(revel)  # already 0-1
    if not np.isnan(pp2_h):
        normed.append(pp2_h)  # already 0-1
    if not np.isnan(pp2_v):
        normed.append(pp2_v)  # already 0-1
    if not np.isnan(sift):
        normed.append(1.0 - sift)  # invert: low=damaging
    if not np.isnan(am):
        normed.append(am)  # already 0-1

    score_range = float(max(normed) - min(normed)) if len(normed) >= 2 else np.nan
    score_max = float(max(normed)) if normed else np.nan
    score_min = float(min(normed)) if normed else np.nan

    return {
        # Per-tool binary calls
        "enrich_call_cadd": (
            float(call_cadd)
            if not (isinstance(call_cadd, float) and np.isnan(call_cadd))
            else np.nan
        ),
        "enrich_call_revel": (
            float(call_revel)
            if not (isinstance(call_revel, float) and np.isnan(call_revel))
            else np.nan
        ),
        "enrich_call_pp2": (
            float(call_pp2h)
            if not (isinstance(call_pp2h, float) and np.isnan(call_pp2h))
            else np.nan
        ),
        "enrich_call_sift": (
            float(call_sift)
            if not (isinstance(call_sift, float) and np.isnan(call_sift))
            else np.nan
        ),
        "enrich_call_am": (
            float(call_am)
            if not (isinstance(call_am, float) and np.isnan(call_am))
            else np.nan
        ),
        # Consensus
        "enrich_n_patho_calls": float(n_patho_calls),
        "enrich_n_available_tools": float(n_available),
        "enrich_patho_call_frac": patho_frac,
        "enrich_unanimous_patho": float(unanimous_patho),
        "enrich_unanimous_benign": float(unanimous_benign),
        # Agreement
        "enrich_cadd_revel_agree": cadd_revel_agree,
        # Score spread (disagreement magnitude)
        "enrich_tool_score_range": score_range,
        "enrich_tool_score_max": score_max,
        "enrich_tool_score_min": score_min,
        # Already-existing meta-scores passed through for model visibility
        # (these are in anonymous cols but naming them helps the model)
        "enrich_silico_variance_ref": safe_float(s_var),
        "enrich_n_tools_with_score": safe_float(s_ntools),
    }


ENRICHERS = [
    enrich_A_substitution_biochem,
    enrich_B_substitution_flags,
    enrich_C_sequence_flags,
    enrich_D_position_derived,
    enrich_E_context_per_position,
    enrich_F_context_summary,
    enrich_G_codon_flags,
    enrich_H_codon_usage_bias,
    enrich_I_secondary_structure,
    # EnrichJ (tool consensus) removed — redundant with col_0105/0106/0107
    # which already encode silico_mean, silico_variance, silico_n_tools
]


def enrich_row(row: pd.Series) -> dict:
    """Run all enricher functions on a single row and merge results."""
    result: dict = {}
    for fn in ENRICHERS:
        result.update(fn(row))
    return result


def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply enrichment to every row of df.
    Returns df with enrichment columns appended (original columns preserved).
    """
    enriched_records = [enrich_row(row) for _, row in df.iterrows()]
    enrich_df = pd.DataFrame(enriched_records, index=df.index)

    # Verify no column name collisions with existing columns
    overlap = set(df.columns) & set(enrich_df.columns)
    if overlap:
        log.warning(f"  Column name collision between base and enrichment: {overlap}")
        log.warning("  Enrichment columns take precedence — dropping originals.")
        df = df.drop(columns=list(overlap))

    return pd.concat([df, enrich_df], axis=1)


def validate_enrichment(
    df_base: pd.DataFrame, df_enriched: pd.DataFrame, panel: str, split: str
) -> None:
    """Sanity checks on enriched output."""
    enrich_cols = [c for c in df_enriched.columns if c.startswith("enrich_")]
    n_enrich = len(enrich_cols)
    n_rows = len(df_enriched)

    log.info(f"  [{panel}/{split}] Enrichment columns: {n_enrich}")
    log.info(f"  [{panel}/{split}] Rows: {n_rows}")

    # Check no enrichment col is 100% NaN
    all_nan = [c for c in enrich_cols if df_enriched[c].isna().all()]
    if all_nan:
        log.warning(f"  [{panel}/{split}] All-NaN enrichment cols: {all_nan}")

    # Check grantham and position_log are non-trivially populated
    for key in ["enrich_grantham_dist", "enrich_aa_position_log"]:
        if key in df_enriched.columns:
            pct = df_enriched[key].notna().mean() * 100
            log.info(f"  [{panel}/{split}] {key}: {pct:.1f}% non-NaN")

    # Check no original base columns were lost
    missing = set(df_base.columns) - set(df_enriched.columns)
    if missing:
        log.error(f"  [{panel}/{split}] BASE COLUMNS LOST: {missing}")
    else:
        log.info(
            f"  [{panel}/{split}] All {len(df_base.columns)} base columns preserved ✓"
        )


def process_panel(
    panel: str,
    input_dir: str,
    output_dir: str,
    validate: bool,
) -> None:
    log.info(f"\n{'═'*60}")
    log.info(f"Panel: {panel.upper()}")
    log.info(f"{'═'*60}")

    for split in ("train", "test"):
        in_path = os.path.join(input_dir, f"{split}_{panel}.csv")
        out_path = os.path.join(output_dir, f"{split}_{panel}_enriched.csv")

        if not os.path.exists(in_path):
            log.warning(f"  File not found, skipping: {in_path}")
            continue

        log.info(f"  Reading {in_path} ...")
        df = pd.read_csv(in_path, low_memory=False)
        log.info(f"  Rows: {len(df):,}   Base columns: {len(df.columns)}")

        log.info("  Computing enrichment features ...")
        result = enrich_dataframe(df)
        df_enriched, added_cols = result if isinstance(result, tuple) else (result, [])

        log.info(f"  Added {len(added_cols)} enrichment features")

        enrich_cols = [c for c in df_enriched.columns if c.startswith("enrich_")]
        log.info(f"  Enrichment features added: {len(enrich_cols)}")
        log.info(f"  Total columns: {len(df_enriched.columns)}")

        if validate:
            validate_enrichment(df, df_enriched, panel, split)

        df_enriched.to_csv(out_path, index=False)
        log.info(f"  Saved → {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich final dataset CSVs with features derived from named columns."
    )
    parser.add_argument(
        "--panels",
        nargs="+",
        default=["general", "cancer", "pah", "cftr"],
        help="Panels to process (default: all four)",
    )
    parser.add_argument("--config", default="config/pipeline.yaml")
    parser.add_argument(
        "--scheme", default="gene", choices=["gene", "variant"],
        help="Split scheme sub-directory (default: gene)",
    )
    parser.add_argument(
        "--input-dir", default=None,
        help="Directory containing train_*.csv and test_*.csv (default: data/final/<scheme>)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory for enriched CSVs (default: data/enriched/<scheme>)",
    )
    parser.add_argument(
        "--validate", action="store_true", help="Run sanity checks on enriched output"
    )
    args = parser.parse_args()
    from utils.config import load_config
    cfg = load_config(args.config)
    if args.input_dir is None:
        args.input_dir = os.path.join(cfg["paths"]["final"], args.scheme)
    if args.output_dir is None:
        args.output_dir = os.path.join(cfg["paths"]["enriched"], args.scheme)

    os.makedirs(args.output_dir, exist_ok=True)

    # Log feature count from a dry run on a dummy row
    dummy = pd.Series(
        {
            "ref_aa": "A",
            "alt_aa": "V",
            "ref_nuc": "C",
            "alt_nuc": "T",
            "aa_position": 100,
            **{f"aa_ctx_{n}": "G" for n in CTX_NAMES},
            **{f"nuc_ctx_{n}": "A" for n in CTX_NAMES},
        }
    )
    dummy_result = enrich_row(dummy)
    log.info(f"Enrichment feature count (dry run): {len(dummy_result)}")
    log.info(
        f"Feature groups: A(biochem deltas), B(substitution flags), "
        f"C(sequence flags), D(position), E(ctx per-pos), "
        f"F(ctx summary), G(codon)"
    )

    for panel in args.panels:
        process_panel(
            panel,
            args.input_dir,
            args.output_dir,
            args.validate,
        )

    log.info("\nDone.")


if __name__ == "__main__":
    main()
