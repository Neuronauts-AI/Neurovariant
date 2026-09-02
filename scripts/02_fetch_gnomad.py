"""
STEP 2 — Fetch gnomAD common missense variants (benign proxy)
==============================================================
Queries the gnomAD v4 GraphQL API for every gene that appears in the
ClinVar pathogenic set.  Keeps only common missense variants (AF > 0.001)
as a proxy for benign population variants — exactly mirroring the
competition's gnomAD benign class design.

Key design decisions:
  - Gene-matched: only genes present in the pathogenic ClinVar set
  - AF threshold: 0.001 (common in healthy populations → presumed benign)
  - Missense only: consequence == "missense_variant"
  - Saves per-gene checkpoints so the run can be safely interrupted/resumed

OUTPUT:  data/raw/gnomad_benign.csv
RUNTIME: 2-4 hours. Checkpoints allow safe resume.
"""

import os
import sys
import json
import time
import pandas as pd

from utils.logging_utils import get_logger
from utils.gnomad_utils import (
    query_gene,
    get_af,
    get_pop_af,
    clean_hgvsp,
    POPULATION_IDS,
    SLEEP_SEC,
    GNOMAD_API,
)
from utils.provenance import write_provenance

log = get_logger("02_fetch_gnomad.log")

# ── Sabitler ──────────────────────────────────────────────────────────────────
CLINVAR_PATH   = "data/raw/clinvar_filtered.csv"
CHECKPOINT_DIR = "data/raw/gnomad_checkpoints"
OUT_PATH       = "data/raw/gnomad_benign.csv"
PROV_PATH      = "data/raw/provenance_gnomad.json"
AF_THRESHOLD   = 0.001


# ── Varyant işleme ────────────────────────────────────────────────────────────
def process_gene_variants(variants: list, gene: str) -> list:
    records = []
    for v in variants:
        csq = (v.get("consequence") or "").lower()
        if "missense" not in csq:
            continue

        af_genome = get_af(v, "genome")
        af_exome  = get_af(v, "exome")
        af = af_genome if af_genome > 0 else af_exome

        if af < AF_THRESHOLD:
            continue

        rec = {
            "GeneSymbol":           gene,
            "VariationID":          f"gnomad_{v.get('variant_id', '')}",
            "Name":                 v.get("variant_id", ""),
            "ClinicalSignificance": "Benign_gnomAD",
            "label":                0,
            "label_confidence":     "gnomad_common",
            "source":               "gnomAD",
            "Chromosome":           (v.get("variant_id") or "").split("-")[0],
            "Start":                v.get("pos", ""),
            "ReferenceAllele":      v.get("ref", ""),
            "AlternateAllele":      v.get("alt", ""),
            "ProteinChange":        clean_hgvsp(v.get("hgvsp", "")),
            "gnomad_af_global":     af,
        }
        for pop in POPULATION_IDS:
            rec[f"gnomad_af_{pop}"] = get_pop_af(v, pop)

        records.append(rec)
    return records


# ── Ana akış ──────────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(CLINVAR_PATH):
        sys.exit("Run step 01 first: python scripts/01_fetch_clinvar.py")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    df_cv = pd.read_csv(CLINVAR_PATH)
    genes = sorted(
        df_cv[df_cv["label"] == 1]["GeneSymbol"].dropna().unique().tolist()
    )
    log.info(f"Querying gnomAD for {len(genes)} genes from ClinVar pathogenic set")

    all_records = []
    failed      = []

    for i, gene in enumerate(genes, 1):
        ckpt = os.path.join(CHECKPOINT_DIR, f"{gene}.json")

        if os.path.exists(ckpt):
            with open(ckpt) as f:
                records = json.load(f)
            log.info(f"[{i:3d}/{len(genes)}] {gene:12s}  "
                     f"loaded from checkpoint  ({len(records)} variants)")
        else:
            log.info(f"[{i:3d}/{len(genes)}] {gene:12s}  querying gnomAD …")
            raw = query_gene(gene)
            if raw is None:
                # GEÇİCİ ağ hatası — checkpoint YAZMA, sonraki çalıştırma
                # yeniden denesin. (Eski davranış boş checkpoint yazıyordu;
                # gen sessizce ve kalıcı olarak veri setinden düşüyordu.)
                failed.append(gene)
                log.warning(f"  → {gene}: geçici hata, checkpoint yazılmadı "
                            f"(yeniden çalıştırınca denenecek)")
                time.sleep(SLEEP_SEC)
                continue
            records = process_gene_variants(raw, gene)
            log.info(f"  → {len(records)} common missense variants kept")
            with open(ckpt, "w") as f:
                json.dump(records, f)
            time.sleep(SLEEP_SEC)

        all_records.extend(records)

    if not all_records:
        log.error("No gnomAD variants collected. Check network and API.")
        return

    df = pd.DataFrame(all_records).drop_duplicates(subset="VariationID")
    df.to_csv(OUT_PATH, index=False)
    log.info(f"\n✓  Saved {len(df):,} gnomAD variants  →  {OUT_PATH}")
    log.info(f"   Top genes:\n{df['GeneSymbol'].value_counts().head(15).to_string()}")

    # Provenance: gnomAD canlı API'dir — dataset sürümü ve çekim tarihi
    # kaydedilmezse sonuç tekrarlanamaz.
    write_provenance(
        PROV_PATH,
        source="gnomAD GraphQL API",
        url=GNOMAD_API,
        source_version="gnomad_r4 (GRCh38)",
        row_count=len(df),
        filters={
            "af_threshold": AF_THRESHOLD,
            "consequence": "missense_variant",
            "gene_matched_to_clinvar_pathogenic": True,
        },
        extra={
            "gene_count": len(genes),
            "failed_genes": failed,
        },
    )
    log.info(f"   Provenance →  {PROV_PATH}")

    if failed:
        log.warning(f"Failed genes ({len(failed)}): {failed}")
        log.warning("Bu genler için checkpoint yazılmadı — scripti yeniden "
                    "çalıştırınca otomatik olarak tekrar denenecek.")
        sys.exit(2)   # orkestrasyon/CI hatayı görebilsin


if __name__ == "__main__":
    main()
