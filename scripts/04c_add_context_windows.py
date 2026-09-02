"""
STEP 4c — Add Context Windows (±5 AA and ±5 Nucleotide)
=========================================================
Extracts the 10 surrounding amino acids and 10 surrounding nucleotides
for every variant and adds them as one-hot encoded feature columns.

This script is called IDENTICALLY on train and test files — fully symmetric.
No labels are used. No external scores. Pure sequence lookup.

AA WINDOW (200 columns):
  10 positions × 20 amino acids = 200 binary columns
  Named: aa_ctx_up5_A ... aa_ctx_dn5_Y
  Positions: up5, up4, up3, up2, up1, dn1, dn2, dn3, dn4, dn5

NUCLEOTIDE WINDOW (40 columns):
  10 positions × 4 nucleotides = 40 binary columns
  Named: nuc_ctx_up5_A ... nuc_ctx_dn5_T

PADDING CONVENTION (hardcoded — must match competition):
  Zero-padding for variants near N/C terminus.
  A 21st "is_padding" column is NOT added — zero vector used instead.
  This matches the most common convention in published variant datasets.

SOURCES:
  Protein sequences: data/raw/uniprot_human_reviewed.fasta
                     (UniProt Swiss-Prot, human, canonical isoforms)
  CDS sequences:     Not used — nucleotide window derived from
                     protein position × 3 in CDS via Ensembl REST API
                     (optional — falls back to zero if CDS unavailable)

INPUT:   data/interim/*_annotated.csv  (all panels, train and test)
OUTPUT:  same files updated in-place with 240 new columns added

USAGE:
  python scripts/04c_add_context_windows.py
  python scripts/04c_add_context_windows.py --panel general  # single panel
  python scripts/04c_add_context_windows.py --dry-run        # validate only
"""

import os
import sys
import re
import argparse
import time
import pandas as pd
import numpy as np
from Bio import SeqIO
import requests

from utils.logging_utils import get_logger
from utils.protein_utils import parse_protein_change, THREE_TO_ONE

log = get_logger("04c_add_context_windows.log")

# ── Constants ─────────────────────────────────────────────────────────────────

FASTA_PATH = "data/raw/uniprot_human_reviewed.fasta"

PANEL_FILES = {
    "general_pool": "data/interim/panel_general_pool_features.csv",
    "cancer_pool":  "data/interim/panel_cancer_pool_features.csv",
    "pah_pool":     "data/interim/panel_pah_pool_features.csv",
    "cftr_pool":    "data/interim/panel_cftr_pool_features.csv",
}

WINDOW = 5          # ±5 positions
AA_CODES = list("ACDEFGHIKLMNPQRSTVWY")   # 20 standard amino acids
NUC_CODES = list("ACGT")

# Protein change regex → utils/protein_utils.py'de tanımlı

# THREE_TO_ONE → utils/protein_utils.py'den import edildi

# Ensembl REST for CDS (rate-limited, used only if needed)
ENSEMBL_REST = "https://rest.ensembl.org"


# ── Load protein sequences ────────────────────────────────────────────────────

def load_protein_sequences(fasta_path: str) -> dict:
    """
    Load UniProt reviewed human protein sequences.
    Returns dict: gene_symbol → sequence string (canonical isoform only).
    When multiple entries exist for the same gene (isoforms), keep the longest.
    """
    if not os.path.exists(fasta_path):
        log.error(f"FASTA not found: {fasta_path}")
        sys.exit(1)

    log.info(f"Loading protein sequences from {fasta_path} ...")
    gene_to_seq = {}

    for record in SeqIO.parse(fasta_path, "fasta"):
        seq = str(record.seq)
        # UniProt FASTA header: >sp|P04637|P53_HUMAN ... GN=TP53 ...
        header = record.description
        gene = ""
        m = re.search(r"\bGN=(\S+)", header)
        if m:
            gene = m.group(1).strip()
        if not gene:
            continue
        # Keep longest sequence for each gene (canonical > isoform)
        if gene not in gene_to_seq or len(seq) > len(gene_to_seq[gene]):
            gene_to_seq[gene] = seq

    log.info(f"  Loaded sequences for {len(gene_to_seq):,} genes")
    return gene_to_seq


# parse_aa_position → utils/protein_utils.parse_protein_change ile değiştirildi


# ── Extract AA context window ─────────────────────────────────────────────────

def extract_aa_window(sequence: str, position: int, window: int = WINDOW) -> list[str]:
    """
    Extract ±window amino acids around position (1-based).
    Returns list of 2*window amino acids: [up5, up4, up3, up2, up1, dn1, dn2, dn3, dn4, dn5]
    Pads with 'X' (zero-vector in one-hot) at termini.
    """
    pos0 = position - 1  # convert to 0-based
    result = []

    # upstream (up5 ... up1)
    for i in range(window, 0, -1):
        idx = pos0 - i
        result.append(sequence[idx] if 0 <= idx < len(sequence) else "X")

    # downstream (dn1 ... dn5)
    for i in range(1, window + 1):
        idx = pos0 + i
        result.append(sequence[idx] if 0 <= idx < len(sequence) else "X")

    return result  # length = 2 * window = 10


def one_hot_aa_window(window_aas: list[str]) -> dict:
    """
    One-hot encode a list of 10 amino acids.
    Returns 200 binary columns: aa_ctx_{pos}_{AA}
    Position names: up5, up4, up3, up2, up1, dn1, dn2, dn3, dn4, dn5
    """
    pos_names = (
        [f"up{WINDOW-i}" for i in range(WINDOW)] +
        [f"dn{i+1}" for i in range(WINDOW)]
    )
    f = {}
    for pos_name, aa in zip(pos_names, window_aas):
        for code in AA_CODES:
            f[f"aa_ctx_{pos_name}_{code}"] = int(aa == code)
    return f  # 10 × 20 = 200 features


# ── Extract nucleotide context window ────────────────────────────────────────


# ── CDS file cache (populated by 04c_prefetch_cds.py) ────────────────────────

CDS_CACHE_PATH    = "data/raw/cds_cache.json"
_file_cache_loaded = False
_file_cache        = {}

def _load_file_cache():
    """Load pre-fetched CDS sequences from disk on first call."""
    global _file_cache_loaded, _file_cache
    if _file_cache_loaded:
        return
    import json
    if os.path.exists(CDS_CACHE_PATH):
        with open(CDS_CACHE_PATH) as f:
            _file_cache = json.load(f)
        n_ok = sum(1 for v in _file_cache.values() if v)
        log.info(f"  Loaded CDS file cache: {len(_file_cache)} genes, "
                 f"{n_ok} with CDS sequences")
    else:
        log.warning(f"  No CDS cache at {CDS_CACHE_PATH}. "
                    f"Run 04c_prefetch_cds.py first for fast offline operation.")
    _file_cache_loaded = True


# ── Extract nucleotide context window ────────────────────────────────────────

def fetch_cds_sequence_ensembl(gene: str, cache: dict) -> str | None:
    """
    Return canonical CDS for gene.
    Priority: in-memory cache → pre-fetched file cache → live Ensembl REST API.
    """
    _load_file_cache()

    if gene in cache:
        return cache[gene]

    # Check pre-fetched file cache first (no API call needed)
    if gene in _file_cache:
        result = _file_cache[gene]
        cache[gene] = result
        return result

    # Fall back to live API (slow — prefer running 04c_prefetch_cds.py first)
    try:
        r = requests.get(
            f"{ENSEMBL_REST}/lookup/symbol/homo_sapiens/{gene}",
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        if r.status_code != 200:
            cache[gene] = None
            return None
        gene_data = r.json()
        gene_id = gene_data.get("id", "")
        if not gene_id:
            cache[gene] = None
            return None

        r2 = requests.get(
            f"{ENSEMBL_REST}/lookup/id/{gene_id}?expand=1",
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        if r2.status_code != 200:
            cache[gene] = None
            return None
        transcripts = r2.json().get("Transcript", [])
        canonical = next((t for t in transcripts if t.get("is_canonical") == 1), None)
        if not canonical:
            cache[gene] = None
            return None
        transcript_id = canonical["id"]

        r3 = requests.get(
            f"{ENSEMBL_REST}/sequence/id/{transcript_id}?type=cds",
            headers={"Content-Type": "text/plain"},
            timeout=15,
        )
        if r3.status_code != 200:
            cache[gene] = None
            return None

        cds = r3.text.strip()
        cache[gene] = cds
        time.sleep(0.1)
        return cds

    except Exception as e:
        log.debug(f"  Ensembl CDS fetch failed for {gene}: {e}")
        cache[gene] = None
        return None

def extract_nuc_window(cds: str, aa_position: int, window: int = WINDOW) -> list[str]:
    """
    Extract ±window nucleotides around the codon for aa_position (1-based).
    The variant codon starts at (aa_position-1)*3 in the CDS.
    We take the nucleotide just before the codon start as the reference point.
    Returns list of 2*window nucleotides. Pads with 'N' at boundaries.
    """
    # Center on first nucleotide of the codon
    codon_start = (aa_position - 1) * 3
    result = []

    # upstream
    for i in range(window, 0, -1):
        idx = codon_start - i
        result.append(cds[idx].upper() if 0 <= idx < len(cds) else "N")

    # downstream (skip the codon itself — start after codon end)
    codon_end = codon_start + 3
    for i in range(window):
        idx = codon_end + i
        result.append(cds[idx].upper() if 0 <= idx < len(cds) else "N")

    return result  # length = 10


def one_hot_nuc_window(window_nucs: list[str]) -> dict:
    """
    One-hot encode a list of 10 nucleotides.
    Returns 40 binary columns: nuc_ctx_{pos}_{NUC}
    """
    pos_names = (
        [f"up{WINDOW-i}" for i in range(WINDOW)] +
        [f"dn{i+1}" for i in range(WINDOW)]
    )
    f = {}
    for pos_name, nuc in zip(pos_names, window_nucs):
        for code in NUC_CODES:
            f[f"nuc_ctx_{pos_name}_{code}"] = int(nuc == code)
    return f  # 10 × 4 = 40 features


# ── Process one panel file ────────────────────────────────────────────────────

def process_panel(path: str, gene_to_seq: dict,
                  use_nuc_window: bool = True,
                  dry_run: bool = False) -> dict:
    """
    Add context window features to one panel file.
    Returns stats dict.
    """
    log.info(f"\n  Processing: {path}")
    df = pd.read_csv(path, low_memory=False)
    n = len(df)

    # Determine protein change column — feature files use _protein_change
    pc_col = "_protein_change" if "_protein_change" in df.columns else (
             "ProteinChange"   if "ProteinChange"   in df.columns else "Name")
    gene_col = "_gene" if "_gene" in df.columns else "GeneSymbol"

    aa_hits     = 0
    aa_misses   = 0
    nuc_hits    = 0
    nuc_misses  = 0

    cds_cache   = {}   # gene → CDS string (populated lazily)
    aa_records  = []
    nuc_records = []

    for _, row in df.iterrows():
        gene = str(row.get(gene_col, "")).strip()
        pc   = str(row.get(pc_col, "")).strip()
        ref_aa, alt_aa, pos = parse_protein_change(pc)

        # ── AA window ──────────────────────────────────────────────────
        seq = gene_to_seq.get(gene, "")
        if seq and pos > 0 and pos <= len(seq):
            window_aas = extract_aa_window(seq, pos)
            aa_hits += 1
        else:
            window_aas = ["X"] * (2 * WINDOW)  # zero-padding
            aa_misses += 1
        aa_records.append(one_hot_aa_window(window_aas))

        # ── Nucleotide window ──────────────────────────────────────────
        if use_nuc_window:
            cds = fetch_cds_sequence_ensembl(gene, cds_cache) if gene else None
            if cds and pos > 0:
                window_nucs = extract_nuc_window(cds, pos)
                nuc_hits += 1
            else:
                window_nucs = ["N"] * (2 * WINDOW)  # zero-padding
                nuc_misses += 1
            nuc_records.append(one_hot_nuc_window(window_nucs))

    # Build feature DataFrames
    df_aa  = pd.DataFrame(aa_records, index=df.index)
    log.info(f"    AA window:  {aa_hits}/{n} hits "
             f"({aa_hits/n*100:.1f}%)  |  {aa_misses} zero-padded")

    if use_nuc_window:
        df_nuc = pd.DataFrame(nuc_records, index=df.index)
        log.info(f"    Nuc window: {nuc_hits}/{n} hits "
                 f"({nuc_hits/n*100:.1f}%)  |  {nuc_misses} zero-padded")

    if dry_run:
        log.info(f"    DRY RUN — file not modified")
        return {"aa_hits": aa_hits, "aa_misses": aa_misses,
                "nuc_hits": nuc_hits if use_nuc_window else None}

    # Drop old zero-variance context columns if they exist
    old_aa_cols  = [c for c in df.columns if c.startswith("aa_ctx_")]
    old_nuc_cols = [c for c in df.columns if c.startswith("nuc_ctx_")]
    if old_aa_cols:
        df.drop(columns=old_aa_cols, inplace=True)
        log.info(f"    Dropped {len(old_aa_cols)} old aa_ctx columns")
    if old_nuc_cols:
        df.drop(columns=old_nuc_cols, inplace=True)
        log.info(f"    Dropped {len(old_nuc_cols)} old nuc_ctx columns")

    # Concatenate and save
    parts = [df, df_aa]
    if use_nuc_window:
        parts.append(df_nuc)
    df_out = pd.concat(parts, axis=1)
    df_out.to_csv(path, index=False)

    n_new = len(df_aa.columns) + (len(df_nuc.columns) if use_nuc_window else 0)
    log.info(f"    Added {n_new} new columns → saved to {path}")

    return {
        "aa_hits": aa_hits, "aa_misses": aa_misses,
        "nuc_hits": nuc_hits if use_nuc_window else None,
        "nuc_misses": nuc_misses if use_nuc_window else None,
        "new_cols": n_new,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default="all",
                        help="Panel name or 'all' (default: all)")
    parser.add_argument("--no-nuc", action="store_true",
                        help="Skip nucleotide window (faster, AA only)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate only, do not modify files")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("STEP 4c — Context Window Annotation")
    log.info(f"  Mode:      {'DRY RUN' if args.dry_run else 'WRITE'}")
    log.info(f"  Nuc window: {'disabled' if args.no_nuc else 'enabled'}")
    log.info("=" * 60)

    # Load protein sequences once
    gene_to_seq = load_protein_sequences(FASTA_PATH)

    # Select panels
    if args.panel == "all":
        panels = PANEL_FILES
    elif args.panel in PANEL_FILES:
        panels = {args.panel: PANEL_FILES[args.panel]}
    else:
        # Accept partial match: "general" → both general_train and general_test
        panels = {k: v for k, v in PANEL_FILES.items() if args.panel in k}
        if not panels:
            log.error(f"Unknown panel: {args.panel}. "
                      f"Valid: {list(PANEL_FILES.keys())}")
            sys.exit(1)

    stats = {}
    for name, path in panels.items():
        if not os.path.exists(path):
            log.warning(f"  Skipping {name} — file not found: {path}")
            continue
        s = process_panel(
            path, gene_to_seq,
            use_nuc_window=not args.no_nuc,
            dry_run=args.dry_run,
        )
        stats[name] = s

    log.info(f"\n{'='*60}")
    log.info("CONTEXT WINDOW ANNOTATION COMPLETE")
    for name, s in stats.items():
        aa_pct = s['aa_hits'] / (s['aa_hits'] + s['aa_misses']) * 100 if s['aa_hits'] else 0
        log.info(f"  {name:<22} AA={aa_pct:.1f}%  "
                 f"new_cols={s.get('new_cols','dry')}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
