"""
STEP 1 — Fetch and filter ClinVar
==================================
Downloads the ClinVar variant_summary file and filters for:

  - Assembly: GRCh38 only
  - Type: single nucleotide variant
  - Review status: Expert Panel OR Practice Guideline (3-4 star)
  - Clinical significance: Pathogenic/Likely pathogenic or Benign/Likely benign
  - Missense variants: identified via p. annotation in Name column

NOTE: ReferenceAllele and AlternateAllele columns contain "na" in ClinVar's
      tab-delimited file. Ref/alt nucleotides are extracted from the HGVS
      Name column (e.g. c.4883T>C → ref=T, alt=C).

OUTPUT:  data/raw/clinvar_filtered.csv
RUNTIME: ~3 minutes (skips download if file already exists)
"""

import os
import sys
import requests
import pandas as pd
from tqdm import tqdm

from utils.logging_utils import get_logger
from utils.clinvar_utils import (
    assign_label_with_confidence,
    extract_ref_alt,
    is_missense,
)
from utils.provenance import write_provenance, file_md5

log = get_logger("01_fetch_clinvar.log")

# ── Sabitler ──────────────────────────────────────────────────────────────────
URL       = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
GZ_PATH   = "data/raw/variant_summary.txt.gz"
OUT_PATH  = "data/raw/clinvar_filtered.csv"
PROV_PATH = "data/raw/provenance_clinvar.json"

# İndirme zaman aşımı: (bağlantı, okuma) saniye
DOWNLOAD_TIMEOUT = (10, 120)

HIGH_CONF = {
    "reviewed by expert panel",
    "practice guideline",
}


# ── İndirme ───────────────────────────────────────────────────────────────────
def download(url: str, dest: str) -> str | None:
    """
    Dosyayı indirir; kaynağın sürüm göstergesi olarak HTTP Last-Modified
    başlığını döndürür (dosya zaten varsa None).

    Sağlamlaştırma:
      - timeout: bağlantı asılı kalamaz
      - .tmp'ye indir + başarıda rename: yarım indirme asla 'tamamlanmış'
        gibi görünmez (eski davranışta bozuk .gz kalıcı olarak kilitleniyordu)
      - Content-Length doğrulaması: eksik byte → hata
    """
    if os.path.exists(dest):
        log.info(f"Already downloaded: {dest}  (delete to re-download)")
        return None
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    log.info(f"Downloading {url}")
    tmp = dest + ".tmp"
    r = requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT)
    r.raise_for_status()
    last_modified = r.headers.get("Last-Modified")
    total = int(r.headers.get("content-length", 0))
    written = 0
    try:
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in r.iter_content(8192):
                f.write(chunk)
                written += len(chunk)
                bar.update(len(chunk))
        if total and written != total:
            raise IOError(f"İndirme eksik: {written}/{total} byte")
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    log.info(f"Saved {os.path.getsize(dest)/1e6:.0f} MB  →  {dest}")
    if last_modified:
        log.info(f"ClinVar release (Last-Modified): {last_modified}")
    return last_modified


# ── Filtreleme ────────────────────────────────────────────────────────────────
def load_and_filter(gz_path: str) -> pd.DataFrame:
    log.info("Reading ClinVar (may take 1-2 min) …")
    df = pd.read_csv(gz_path, sep="\t", compression="gzip",
                     low_memory=False, on_bad_lines="skip")
    log.info(f"  Total rows:                {len(df):,}")

    df = df[df["Assembly"].str.upper().str.strip() == "GRCH38"]
    log.info(f"  After GRCh38:              {len(df):,}")

    df = df[df["Type"].str.lower().str.strip() == "single nucleotide variant"]
    log.info(f"  After SNV type:            {len(df):,}")

    df = df[df["ReviewStatus"].str.lower().str.strip().isin(HIGH_CONF)]
    log.info(f"  After high-confidence:     {len(df):,}")

    lc = df["ClinicalSignificance"].apply(assign_label_with_confidence)
    df["label"]            = lc.apply(lambda x: x[0])
    df["label_confidence"] = lc.apply(lambda x: x[1])
    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(int)
    log.info(f"  After label filter:        {len(df):,}")
    log.info(f"  Pathogenic: {df['label'].sum():,}   "
             f"Benign: {(df['label']==0).sum():,}")
    log.info(f"  label_confidence: {df['label_confidence'].value_counts().to_dict()}")

    df = df[df["Name"].apply(is_missense)].copy()
    log.info(f"  After missense filter:     {len(df):,}")

    ref_alt = df["Name"].apply(extract_ref_alt)
    df["ReferenceAllele"] = ref_alt.apply(lambda x: x[0])
    df["AlternateAllele"] = ref_alt.apply(lambda x: x[1])

    before = len(df)
    df = df.drop_duplicates(subset="VariationID")
    log.info(f"  After deduplication:       {len(df):,}  (removed {before-len(df):,})")

    cols = [
        "VariationID", "GeneSymbol", "Name", "ClinicalSignificance",
        "label", "label_confidence",
        "ReviewStatus", "Chromosome", "Start",
        "ReferenceAllele", "AlternateAllele",
    ]
    for candidate in ["ProteinChange", "ProteinAccession"]:
        if candidate in df.columns:
            cols.append(candidate)
            break
    df = df[[c for c in cols if c in df.columns]].copy()

    log.info(f"\n  ClinicalSignificance breakdown:")
    log.info(df["ClinicalSignificance"].value_counts().head(15).to_string())
    log.info(f"\n  Top 20 genes:")
    log.info(df["GeneSymbol"].value_counts().head(20).to_string())

    return df


# ── Ana akış ──────────────────────────────────────────────────────────────────
def main():
    last_modified = download(URL, GZ_PATH)
    df = load_and_filter(GZ_PATH)
    df.to_csv(OUT_PATH, index=False)
    log.info(f"\n✓  Saved {len(df):,} variants  →  {OUT_PATH}")
    log.info(f"   Pathogenic: {df['label'].sum():,}   Benign: {(df['label']==0).sum():,}")

    # Provenance: ClinVar haftalık güncellenir — sürüm ve çekim tarihi kaydı
    # olmadan veri seti tekrarlanabilir değildir.
    write_provenance(
        PROV_PATH,
        source="ClinVar variant_summary",
        url=URL,
        source_version=last_modified,   # dosya önceden mevcutsa None kalır
        input_files={os.path.basename(GZ_PATH): file_md5(GZ_PATH)},
        row_count=len(df),
        filters={
            "assembly": "GRCh38",
            "type": "single nucleotide variant",
            "review_status": sorted(HIGH_CONF),
            "missense_only": True,
        },
    )
    log.info(f"   Provenance →  {PROV_PATH}")


if __name__ == "__main__":
    main()
