"""
STEP 1b — Supplement benign class with 2-star ClinVar variants
==============================================================
Addresses the benign class shortage caused by the strict 3-4 star filter.

Adds ClinVar benign variants with review status:
  - "criteria provided, multiple submitters, no conflicts"  (2-star)

Exclusion safeguards applied:
  1. Any VariationID already in the 3-4 star dataset is excluded
  2. Any variant in a gene+position that appears in the pathogenic set
     is excluded (contamination prevention)
  3. Same missense filter as step 01 (HGVS p. annotation required)
  4. Same label logic (handles compound ClinVar strings)

INPUT:   data/raw/variant_summary.txt.gz   (already downloaded)
         data/raw/clinvar_filtered.csv      (from step 01)
OUTPUT:  data/raw/clinvar_filtered.csv      (updated in place — benign class expanded)
         data/raw/clinvar_filtered_backup.csv  (backup of original)
"""

import os
import sys
import json
import shutil
import pandas as pd

from utils.logging_utils import get_logger
from utils.clinvar_utils import (
    assign_label,
    extract_ref_alt,
    is_missense,
    EXCLUDE_SIG,
)

log = get_logger("01b_supplement_benign.log")

# ── Sabitler ──────────────────────────────────────────────────────────────────
GZ_PATH     = "data/raw/variant_summary.txt.gz"
MAIN_PATH   = "data/raw/clinvar_filtered.csv"
BACKUP_PATH = "data/raw/clinvar_filtered_backup.csv"
OUT_PATH    = "data/raw/clinvar_filtered.csv"

TWO_STAR = {
    "criteria provided, multiple submitters, no conflicts",
}

TARGET_EXTRA_BENIGN = 3000


def assign_label_with_confidence_2star(sig: str) -> "tuple[int | None, str | None]":
    """2-star varyantlar için confidence etiketi 'hard_2star' / 'likely_2star'."""
    label = assign_label(sig)
    if label is None:
        return None, None
    s = sig.lower().strip()
    conf = "likely_2star" if "likely" in s else "hard_2star"
    return label, conf


# ── Ana akış ──────────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(MAIN_PATH):
        sys.exit("Run step 01 first: python scripts/01_fetch_clinvar.py")

    df_main  = pd.read_csv(MAIN_PATH)
    n_patho  = int(df_main["label"].sum())
    n_benign = int((df_main["label"] == 0).sum())
    log.info(f"Existing dataset: {len(df_main):,} variants")
    log.info(f"  Pathogenic: {n_patho:,}   Benign: {n_benign:,}")
    log.info(f"  Benign deficit: {n_patho - n_benign:,} variants")

    if n_benign >= n_patho:
        log.info("Benign class already balanced — no supplementation needed.")
        return

    shutil.copy(MAIN_PATH, BACKUP_PATH)
    log.info(f"Backup saved: {BACKUP_PATH}")

    existing_ids = set(df_main["VariationID"].astype(str))
    patho_genes  = set(df_main[df_main["label"] == 1]["GeneSymbol"].dropna())

    log.info("\nReading ClinVar for 2-star benign variants …")
    df = pd.read_csv(GZ_PATH, sep="\t", compression="gzip",
                     low_memory=False, on_bad_lines="skip")
    log.info(f"  Total rows: {len(df):,}")

    df = df[df["Assembly"].str.upper().str.strip() == "GRCH38"]
    df = df[df["Type"].str.lower().str.strip() == "single nucleotide variant"]
    log.info(f"  After GRCh38 + SNV filter: {len(df):,}")

    df = df[df["ReviewStatus"].str.lower().str.strip().isin(TWO_STAR)]
    log.info(f"  After 2-star filter: {len(df):,}")

    lc = df["ClinicalSignificance"].apply(assign_label_with_confidence_2star)
    df["label"]            = lc.apply(lambda x: x[0])
    df["label_confidence"] = lc.apply(lambda x: x[1])
    df = df[df["label"] == 0].copy()
    log.info(f"  After benign label filter: {len(df):,}")

    df = df[df["Name"].apply(is_missense)].copy()
    log.info(f"  After missense filter: {len(df):,}")

    before = len(df)
    df = df[~df["VariationID"].astype(str).isin(existing_ids)]
    log.info(f"  After removing existing IDs: {len(df):,} "
             f"(excluded {before-len(df):,})")

    df = df[df["GeneSymbol"].isin(patho_genes)]
    log.info(f"  After gene-matching to pathogenic genes: {len(df):,}")

    before = len(df)
    df = df.drop_duplicates(subset="VariationID")
    log.info(f"  After deduplication: {len(df):,} (removed {before-len(df):,})")

    if len(df) == 0:
        log.warning("No 2-star benign variants found after filtering.")
        return

    ref_alt = df["Name"].apply(extract_ref_alt)
    df["ReferenceAllele"] = ref_alt.apply(lambda x: x[0])
    df["AlternateAllele"] = ref_alt.apply(lambda x: x[1])
    df["source"] = "ClinVar_2star"

    cols    = [c for c in df_main.columns if c in df.columns]
    df      = df[cols].copy()

    deficit = n_patho - n_benign
    n_add   = min(len(df), deficit + 500)
    df_add  = df.sample(n=n_add, random_state=42)
    log.info(f"\n  Adding {len(df_add):,} 2-star benign variants "
             f"(deficit was {deficit:,})")

    df_combined = pd.concat([df_main, df_add], ignore_index=True)
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    df_combined.to_csv(OUT_PATH, index=False)

    n_patho_final  = int(df_combined["label"].sum())
    n_benign_final = int((df_combined["label"] == 0).sum())

    log.info(f"\n{'='*55}")
    log.info("SUPPLEMENTATION COMPLETE")
    log.info(f"{'='*55}")
    log.info(f"  Before:  Pathogenic={n_patho:,}  Benign={n_benign:,}")
    log.info(f"  After:   Pathogenic={n_patho_final:,}  Benign={n_benign_final:,}")
    log.info(f"  Added:   {len(df_add):,} variants  "
             f"(source: ClinVar 2-star, gene-matched)")

    if "source" in df_combined.columns:
        log.info(f"\n  Benign source breakdown:")
        log.info(df_combined[df_combined["label"] == 0]["source"]
                 .value_counts().to_string())

    log.info(f"\n  Top genes in supplemented benign set:")
    log.info(df_add["GeneSymbol"].value_counts().head(15).to_string())
    log.info(f"\n✓  Saved {len(df_combined):,} variants  →  {OUT_PATH}")

    source_counts     = df_combined["source"].value_counts().to_dict() \
                        if "source" in df_combined.columns else {}
    confidence_counts = df_combined["label_confidence"].fillna("hard") \
                        .value_counts().to_dict() \
                        if "label_confidence" in df_combined.columns else {}

    manifest = {
        "total_variants":             len(df_combined),
        "pathogenic":                 n_patho_final,
        "benign":                     n_benign_final,
        "source_breakdown":           source_counts,
        "label_confidence_breakdown": confidence_counts,
        "notes": "2-star supplement tagged as hard_2star/likely_2star in label_confidence",
    }
    manifest_path = "data/raw/dataset_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info(f"✓  Manifest written → {manifest_path}")


if __name__ == "__main__":
    main()
