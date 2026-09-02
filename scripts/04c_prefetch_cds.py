"""
STEP 4c-prefetch — Pre-fetch CDS sequences for all genes in panels
===================================================================
Fetches canonical CDS sequences from Ensembl REST API for every
unique gene symbol found across all panel files, then saves them
to data/raw/cds_cache.json.

04c_add_context_windows.py will load this cache automatically and
skip API calls entirely — making the nucleotide window annotation
fast and offline-capable after this step.

Unique genes across all panels: ~50-150
API calls per gene: 3 (lookup → transcript → CDS)
Estimated time: ~5-10 minutes total

USAGE:
  python scripts/04c_prefetch_cds.py
  python scripts/04c_prefetch_cds.py --dry-run
"""

import os, sys, re, json, time, logging, argparse
import pandas as pd
import requests

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.FileHandler("logs/04c_prefetch_cds.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger()

PANEL_FILES = {
    "general_pool": "data/interim/panel_general_pool_features.csv",
    "cancer_pool":  "data/interim/panel_cancer_pool_features.csv",
    "pah_pool":     "data/interim/panel_pah_pool_features.csv",
    "cftr_pool":    "data/interim/panel_cftr_pool_features.csv",
}

CACHE_PATH   = "data/raw/cds_cache.json"
ENSEMBL_REST = "https://rest.ensembl.org"
SLEEP_SEC    = 0.12   # ~8 req/sec, well within 15 req/sec limit


def fetch_cds_for_gene(gene: str) -> str | None:
    """Fetch canonical CDS for a gene via 3-step Ensembl REST call."""
    try:
        # Step 1: gene lookup → Ensembl gene ID
        r = requests.get(
            f"{ENSEMBL_REST}/lookup/symbol/homo_sapiens/{gene}",
            headers={"Content-Type": "application/json"},
            timeout=15,
        )
        if r.status_code != 200:
            return None
        gene_id = r.json().get("id", "")
        if not gene_id:
            return None
        time.sleep(SLEEP_SEC)

        # Step 2: expand gene → find canonical transcript
        r2 = requests.get(
            f"{ENSEMBL_REST}/lookup/id/{gene_id}?expand=1",
            headers={"Content-Type": "application/json"},
            timeout=15,
        )
        if r2.status_code != 200:
            return None
        transcripts = r2.json().get("Transcript", [])
        canonical = next((t for t in transcripts if t.get("is_canonical") == 1), None)
        if not canonical:
            # fallback: pick longest CDS
            canonical = max(transcripts,
                            key=lambda t: t.get("Translation", {}).get("length", 0),
                            default=None)
        if not canonical:
            return None
        transcript_id = canonical["id"]
        time.sleep(SLEEP_SEC)

        # Step 3: fetch CDS sequence
        r3 = requests.get(
            f"{ENSEMBL_REST}/sequence/id/{transcript_id}?type=cds",
            headers={"Content-Type": "text/plain"},
            timeout=20,
        )
        if r3.status_code != 200:
            return None
        time.sleep(SLEEP_SEC)

        cds = r3.text.strip()
        return cds if len(cds) >= 3 else None

    except Exception as e:
        log.debug(f"  Ensembl error for {gene}: {e}")
        return None


def collect_unique_genes() -> set:
    """Collect all unique gene symbols across all panel feature files."""
    genes = set()
    for name, path in PANEL_FILES.items():
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, usecols=["_gene"], low_memory=False)
        panel_genes = set(df["_gene"].dropna().unique())
        genes.update(panel_genes)
        log.info(f"  {name:<22} {len(panel_genes)} genes")
    return genes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Collect genes only, do not fetch CDS")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Skip genes already in cache (default: True)")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("STEP 4c-prefetch — CDS sequence pre-fetch")
    log.info("=" * 60)

    # Load existing cache
    cache = {}
    if os.path.exists(CACHE_PATH) and args.resume:
        with open(CACHE_PATH) as f:
            cache = json.load(f)
        log.info(f"Loaded existing cache: {len(cache)} genes "
                 f"({sum(1 for v in cache.values() if v)} with CDS)")

    # Collect genes
    log.info("\nCollecting unique genes from all panels ...")
    all_genes = collect_unique_genes()
    log.info(f"\nTotal unique genes: {len(all_genes)}")

    # Filter to genes not yet cached
    todo = sorted(g for g in all_genes
                  if isinstance(g, str) and g and g not in cache)
    log.info(f"Genes to fetch: {len(todo)}")

    if args.dry_run:
        log.info("DRY RUN — not fetching. Gene list:")
        for g in todo:
            log.info(f"  {g}")
        return

    if not todo:
        log.info("All genes already cached — nothing to do.")
        return

    # Fetch CDS for each gene
    log.info(f"\nFetching CDS sequences (~{len(todo)*3} API calls) ...")
    success = 0
    fail    = 0

    for i, gene in enumerate(todo, 1):
        log.info(f"  [{i:>3}/{len(todo)}] {gene:<15} ...", )
        cds = fetch_cds_for_gene(gene)
        cache[gene] = cds

        if cds:
            success += 1
            log.info(f"  [{i:>3}/{len(todo)}] {gene:<15} ✓  {len(cds)} bp")
        else:
            fail += 1
            log.info(f"  [{i:>3}/{len(todo)}] {gene:<15} ✗  not found")

        # Save cache incrementally every 10 genes
        if i % 10 == 0:
            with open(CACHE_PATH, "w") as f:
                json.dump(cache, f)
            log.info(f"  Cache saved ({i} genes processed)")

    # Final save
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)

    log.info(f"\n{'='*60}")
    log.info("CDS PRE-FETCH COMPLETE")
    log.info(f"  Success: {success}/{len(todo)}")
    log.info(f"  Failed:  {fail}/{len(todo)}")
    log.info(f"  Cache:   {CACHE_PATH}")
    log.info(f"  Total cached genes: {len(cache)}")
    log.info(f"\nNow run: python scripts/04c_add_context_windows.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
