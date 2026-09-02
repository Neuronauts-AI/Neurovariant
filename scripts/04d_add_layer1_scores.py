"""
STEP 4d — Add Layer 1 Scores via MyVariant.info (AA-based query)
================================================================
Queries MyVariant.info using gene + amino acid change (NOT genomic
coordinates) to retrieve in silico scores.

Query format:
  dbnsfp.genename:{GENE} AND dbnsfp.aa.ref:{REF} AND
  dbnsfp.aa.alt:{ALT} AND dbnsfp.aa.pos:{POS}

This is coordinate-independent — works regardless of hg19/hg38,
and matches exactly on the biological variant identity.

SCORES RETRIEVED:
  cadd.phred
  dbnsfp.revel.score
  dbnsfp.polyphen2.hdiv.score
  dbnsfp.polyphen2.hvar.score
  dbnsfp.sift.score
  dbnsfp.phylo.p17way.primate
  dbnsfp.phastcons.17way.primate
  dbnsfp.gerp++.rs

INPUT:   data/interim/panel_*_features.csv
OUTPUT:  same files updated in-place

USAGE:
  python scripts/04d_add_layer1_scores.py
  python scripts/04d_add_layer1_scores.py --panel general
  python scripts/04d_add_layer1_scores.py --dry-run
"""

import os
import sys
import re
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np

from utils.logging_utils import get_logger
from utils.protein_utils import parse_protein_change, THREE_TO_ONE

log = get_logger("04d_add_layer1_scores.log")

PANEL_FILES = {
    "general_pool": "data/interim/panel_general_pool_features.csv",
    "cancer_pool":  "data/interim/panel_cancer_pool_features.csv",
    "pah_pool":     "data/interim/panel_pah_pool_features.csv",
    "cftr_pool":    "data/interim/panel_cftr_pool_features.csv",
}

FIELDS = ",".join([
    "cadd.phred",
    "dbnsfp.revel.score",
    "dbnsfp.polyphen2.hdiv.score",
    "dbnsfp.polyphen2.hvar.score",
    "dbnsfp.sift.score",
    "dbnsfp.phylo.p17way.primate",
    "dbnsfp.phastcons.17way.primate",
    "dbnsfp.gerp++.rs",
])

SCORE_COLS = [
    "cadd_phred",
    "revel_score",
    "polyphen2_hdiv",
    "polyphen2_hvar",
    "sift_score",
    "phylop_primate",
    "phastcons_primate",
    "gerp_rs",
]

SLEEP_SEC  = 0.15   # ~6 req/sec, well within rate limit

# THREE_TO_ONE ve protein change parser → utils/protein_utils.py'den import edildi

def parse_aa_change(name_str):
    """parse_protein_change etrafındaki ince sarıcı: None tuple döndürür başarısızlıkta."""
    ref, alt, pos = parse_protein_change(name_str or "")
    if ref == "X" or alt == "X" or pos < 0:
        return None, None, None
    return ref, alt, pos


def safe_float(val):
    if val is None:
        return None
    if isinstance(val, list):
        vals = [v for v in val if v is not None]
        return float(np.median(vals)) if vals else None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def parse_hit(hit):
    """Extract scores from a MyVariant.info hit."""
    cadd   = hit.get("cadd",   {}) or {}
    dbnsfp = hit.get("dbnsfp", {}) or {}
    pp2    = dbnsfp.get("polyphen2", {}) or {}
    sift   = dbnsfp.get("sift",      {}) or {}
    phylo  = dbnsfp.get("phylo",     {}) or {}
    p17    = phylo.get("p17way",     {}) or {}
    phast  = dbnsfp.get("phastcons", {}) or {}
    p17ph  = phast.get("17way",      {}) or {}
    gerp   = dbnsfp.get("gerp++",    {}) or {}
    revel  = dbnsfp.get("revel",     {}) or {}

    return {
        "cadd_phred":       safe_float(cadd.get("phred")),
        "revel_score":      safe_float(revel.get("score")),
        "polyphen2_hdiv":   safe_float((pp2.get("hdiv") or {}).get("score")),
        "polyphen2_hvar":   safe_float((pp2.get("hvar") or {}).get("score")),
        "sift_score":       safe_float(sift.get("score")),
        "phylop_primate":   safe_float(p17.get("primate")),
        "phastcons_primate":safe_float(p17ph.get("primate")),
        "gerp_rs":          safe_float(gerp.get("rs")),
    }


def query_variant(mv, gene, ref_aa, alt_aa, pos):
    """
    Query MyVariant.info by gene + amino acid change.
    Returns scores dict or None.
    """
    q = (f"dbnsfp.genename:{gene} AND "
         f"dbnsfp.aa.ref:{ref_aa} AND "
         f"dbnsfp.aa.alt:{alt_aa} AND "
         f"dbnsfp.aa.pos:{pos}")
    try:
        result = mv.query(q, fields=FIELDS, size=1, verbose=False)
        hits = result.get("hits", [])
        if hits:
            return parse_hit(hits[0])
    except Exception as e:
        log.debug(f"  Query failed {gene} {ref_aa}{pos}{alt_aa}: {e}")
    return None


def process_panel(name, path, dry_run=False):
    log.info(f"\n  Processing: {name}  ({path})")

    try:
        import myvariant
    except ImportError:
        log.error("myvariant not installed. Run: pip install myvariant")
        sys.exit(1)

    mv = myvariant.MyVariantInfo()
    df = pd.read_csv(path, low_memory=False)
    n  = len(df)

    pc_col   = "_protein_change" if "_protein_change" in df.columns else "Name"
    gene_col = "_gene"           if "_gene"           in df.columns else "GeneSymbol"

    if dry_run:
        log.info(f"  DRY RUN — {n} variants, skipping API")
        return {"total": n, "dry_run": True}

    N_WORKERS = 8

    # Build query list
    queries = []
    for i, (_, row) in enumerate(df.iterrows()):
        gene = str(row.get(gene_col, "") or "").strip()
        pc   = str(row.get(pc_col,   "") or "").strip()
        ref_aa, alt_aa, pos = parse_aa_change(pc)
        queries.append((i, gene, ref_aa, alt_aa, pos))

    scores_list = [None] * n
    hits = 0
    completed = 0

    def fetch_one(args):
        idx, gene, ref_aa, alt_aa, pos = args
        if gene and ref_aa and alt_aa and pos:
            result = query_variant(mv, gene, ref_aa, alt_aa, pos)
            time.sleep(SLEEP_SEC / N_WORKERS)
            return idx, result
        return idx, None

    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(fetch_one, q): q[0] for q in queries}
        for future in as_completed(futures):
            idx, scores = future.result()
            scores_list[idx] = scores if scores else {col: None for col in SCORE_COLS}
            if scores:
                hits += 1
            completed += 1
            if completed % 100 == 0:
                log.info(f"    [{completed}/{n}] hits so far: {hits}")

    log.info(f"  Total hits: {hits}/{n} ({hits/n*100:.1f}%)")

    df_scores = pd.DataFrame(scores_list, index=df.index)

    for col in SCORE_COLS:
        n_hits = df_scores[col].notna().sum()
        log.info(f"    {col:<25} {n_hits:>5}/{n} ({n_hits/n*100:.1f}%)")

    # Drop old score columns if re-running
    old = [c for c in df.columns if c in SCORE_COLS]
    if old:
        df.drop(columns=old, inplace=True)

    df_out = pd.concat([df, df_scores], axis=1)
    df_out.to_csv(path, index=False)
    log.info(f"  Saved → {path}")

    return {"total": n, "hits": hits,
            "hit_rates": {col: int(df_scores[col].notna().sum())
                          for col in SCORE_COLS}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("STEP 4d — Layer 1 Scores (AA-based query, coordinate-free)")
    log.info(f"  Mode: {'DRY RUN' if args.dry_run else 'WRITE'}")
    log.info("=" * 60)

    if args.panel == "all":
        panels = PANEL_FILES
    else:
        panels = {k: v for k, v in PANEL_FILES.items() if args.panel in k}
        if not panels:
            log.error(f"Unknown panel: {args.panel}")
            sys.exit(1)

    stats = {}
    for name, path in panels.items():
        if not os.path.exists(path):
            log.warning(f"  Skipping {name} — not found")
            continue
        stats[name] = process_panel(name, path, dry_run=args.dry_run)

    log.info(f"\n{'='*60}")
    log.info("LAYER 1 ANNOTATION COMPLETE")
    for name, s in stats.items():
        if s.get("dry_run"):
            log.info(f"  {name:<22} DRY RUN")
        else:
            h = s.get("hits", 0)
            t = s.get("total", 1)
            log.info(f"  {name:<22} {h}/{t} ({h/t*100:.1f}%)")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
