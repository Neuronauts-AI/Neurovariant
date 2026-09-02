"""
STEP 3 — Build the four disease panels and assign BOTH split schemes
=====================================================================

Differences from the competition-era pipeline
---------------------------------------------
1. Panels are gene-disjoint from each other: the `general` panel excludes
   every gene that belongs to a specialty panel (cancer, PAH, CFTR). A
   variant therefore belongs to exactly one panel and can never sit in the
   training set of one panel and the test set of another.

2. Every variant is assigned to train/test under TWO schemes at once:
     _split_variant  stratified random split at the variant level
                     (the classical — and leaky — protocol)
     _split_gene     gene-disjoint split: no gene appears on both sides
                     (protects against type-2 circularity, Grimm et al. 2015)
   For single-gene panels (PAH, CFTR) a gene-disjoint split is impossible;
   there `_split_gene` copies `_split_variant` and `_split_gene_note`
   records "within-gene".

3. Features are computed once on the panel *pool* (step 04); the split
   scheme is materialised only in step 05. This guarantees identical
   feature values across schemes.

INPUTS
  data/raw/clinvar_filtered.csv
  data/raw/gnomad_benign.csv            (optional)
  data/raw/variant_summary.txt.gz       (for PAH/CFTR any-star benign)

OUTPUTS
  data/interim/panel_{name}_pool.csv    all selected variants + split columns
  data/interim/split_manifest.json      test genes / counts per scheme
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

from utils.logging_utils import get_logger
from utils.clinvar_utils import assign_label, extract_ref_alt, is_missense
from utils.config import load_config

log = get_logger("03_build_panels.log")

# ── Gene panel definitions ───────────────────────────────────────────────────
CANCER_GENES = {
    "BRCA1", "BRCA2", "PALB2", "RAD51C", "RAD51D", "BRIP1",
    "MLH1", "MSH2", "MSH6", "PMS2",
    "TP53", "PTEN", "STK11", "CDH1", "VHL", "RET", "APC",
    "SMAD4", "BMPR1A", "ATM", "CHEK2", "NBN",
    "NF1", "NF2", "MEN1",
    "SDHA", "SDHB", "SDHC", "SDHD",
    "BAP1", "MUTYH", "FLCN", "FH",
}
PAH_GENES = {"PAH"}
CFTR_GENES = {"CFTR"}
SPECIALTY_GENES = CANCER_GENES | PAH_GENES | CFTR_GENES


# ── Helpers ──────────────────────────────────────────────────────────────────
def sample_up_to(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) >= n:
        return df.sample(n=n, random_state=seed)
    if len(df) < n:
        log.warning(f"    Only {len(df)} available (wanted {n}) — using all")
    return df.copy()


def load_anystar_benign(gz_path: str, gene_set: set, existing_ids: set) -> pd.DataFrame:
    """Any-star ClinVar benign missense for genes with too few high-confidence benign."""
    if not os.path.exists(gz_path):
        log.warning(f"    {gz_path} not found — cannot supplement any-star benign")
        return pd.DataFrame()
    log.info(f"    Loading any-star ClinVar benign for {sorted(gene_set)} ...")
    df = pd.read_csv(gz_path, sep="\t", compression="gzip",
                     low_memory=False, on_bad_lines="skip")
    df = df[df["Assembly"].str.upper().str.strip() == "GRCH38"]
    df = df[df["Type"].str.lower().str.strip() == "single nucleotide variant"]
    df = df[df["GeneSymbol"].isin(gene_set)]
    df["label"] = df["ClinicalSignificance"].apply(assign_label)
    df = df[df["label"] == 0].copy()
    df = df[df["Name"].apply(is_missense)].copy()
    df = df[~df["VariationID"].astype(str).isin(existing_ids)]
    df = df.drop_duplicates(subset="VariationID")
    ref_alt = df["Name"].apply(extract_ref_alt)
    df["ReferenceAllele"] = ref_alt.apply(lambda x: x[0])
    df["AlternateAllele"] = ref_alt.apply(lambda x: x[1])
    df["source"] = "ClinVar_anystar"
    log.info(f"    Found {len(df)} any-star benign variants")
    return df


def variant_split(df: pd.DataFrame, n_train: int, n_test: int, seed: int):
    """Stratified-by-construction random split (called per class)."""
    df_s = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    if len(df_s) < n_train + n_test:          # keep the train:test ratio
        n_train = int(round(len(df_s) * n_train / (n_train + n_test)))
        n_test = len(df_s) - n_train
    return df_s.iloc[:n_train].copy(), df_s.iloc[n_train:n_train + n_test].copy()


def gene_split(df_path: pd.DataFrame, df_benign: pd.DataFrame,
               targets: dict, seed: int):
    """
    Gene-disjoint split. Genes are shuffled and assigned greedily to the
    test side until the test side holds >= test_frac of pathogenic AND of
    benign candidates. Returns (train_idx_path, test_idx_path,
    train_idx_benign, test_idx_benign, test_genes).
    """
    rng = np.random.RandomState(seed)
    total_path = targets["train_path"] + targets["test_path"]
    total_ben = targets["train_benign"] + targets["test_benign"]
    test_frac = targets["test_path"] / total_path

    genes = pd.Index(sorted(set(df_path["GeneSymbol"].dropna())
                            | set(df_benign["GeneSymbol"].dropna())))
    genes = genes[rng.permutation(len(genes))]

    n_path_gene = df_path.groupby("GeneSymbol").size()
    n_ben_gene = df_benign.groupby("GeneSymbol").size()
    need_path = test_frac * min(len(df_path), total_path)
    need_ben = test_frac * min(len(df_benign), total_ben)

    test_genes, acc_p, acc_b = [], 0, 0
    for g in genes:
        if acc_p >= need_path and acc_b >= need_ben:
            break
        test_genes.append(g)
        acc_p += int(n_path_gene.get(g, 0))
        acc_b += int(n_ben_gene.get(g, 0))
    test_genes = set(test_genes)

    p_te = df_path[df_path["GeneSymbol"].isin(test_genes)]
    p_tr = df_path[~df_path["GeneSymbol"].isin(test_genes)]
    b_te = df_benign[df_benign["GeneSymbol"].isin(test_genes)]
    b_tr = df_benign[~df_benign["GeneSymbol"].isin(test_genes)]
    return p_tr, p_te, b_tr, b_te, test_genes


# ── Panel builder ────────────────────────────────────────────────────────────
def build_panel(name: str, df_cv: pd.DataFrame, df_gnomad: pd.DataFrame | None,
                gz_path: str, gene_set: set | None, exclude_genes: set | None,
                targets: dict, seed: int) -> tuple[pd.DataFrame, dict]:
    log.info(f"\n{'─' * 60}\nPanel: {name.upper()}")

    df = df_cv.copy()
    if gene_set:
        df = df[df["GeneSymbol"].isin(gene_set)]
    elif exclude_genes:
        df = df[~df["GeneSymbol"].isin(exclude_genes)]
    log.info(f"  ClinVar variants in scope: {len(df):,}")

    df_path = df[df["label"] == 1].copy()
    df_benign_pool = df[df["label"] == 0].copy()
    log.info(f"  Pathogenic: {len(df_path):,}   Benign (ClinVar high-conf): {len(df_benign_pool):,}")

    total_path = targets["train_path"] + targets["test_path"]
    total_ben = targets["train_benign"] + targets["test_benign"]
    existing_ids = set(df_cv["VariationID"].astype(str))

    # Benign supplementation (same policy as before, now gene-aware)
    single_gene = gene_set is not None and len(gene_set) <= 2
    if single_gene and len(df_benign_pool) < total_ben * 0.5:
        df_any = load_anystar_benign(gz_path, gene_set, existing_ids)
        if len(df_any):
            shared = [c for c in df_benign_pool.columns if c in df_any.columns]
            df_benign_pool = pd.concat([df_benign_pool, df_any[shared]],
                                       ignore_index=True).drop_duplicates(subset="VariationID")
            log.info(f"  After any-star supplement: {len(df_benign_pool):,} benign")

    n_needed = total_ben - len(df_benign_pool)
    if n_needed > 0 and df_gnomad is not None:
        patho_genes = set(df_path["GeneSymbol"].dropna())
        gn = df_gnomad[df_gnomad["GeneSymbol"].isin(patho_genes)].copy()
        gn = gn[~gn["VariationID"].astype(str).isin(existing_ids)] if "VariationID" in gn.columns else gn
        df_gn = sample_up_to(gn, n_needed, seed)
        shared = [c for c in df_benign_pool.columns if c in df_gn.columns]
        df_benign_pool = pd.concat([df_benign_pool, df_gn[shared]], ignore_index=True)
        log.info(f"  Added {len(df_gn)} gnomAD common-variant benign")

    # ── Scheme 1: variant-level split ────────────────────────────────────
    path_s = sample_up_to(df_path, total_path, seed)
    ben_s = sample_up_to(df_benign_pool, total_ben, seed)
    v_tr_p, v_te_p = variant_split(path_s, targets["train_path"], targets["test_path"], seed)
    v_tr_b, v_te_b = variant_split(ben_s, targets["train_benign"], targets["test_benign"], seed)
    variant_train_ids = set(v_tr_p["VariationID"].astype(str)) | set(v_tr_b["VariationID"].astype(str))
    variant_test_ids = set(v_te_p["VariationID"].astype(str)) | set(v_te_b["VariationID"].astype(str))

    # ── Scheme 2: gene-disjoint split ────────────────────────────────────
    note = ""
    if single_gene:
        gene_train_ids, gene_test_ids, test_genes = variant_train_ids, variant_test_ids, set()
        note = "within-gene (single-gene panel: gene-disjoint split impossible)"
    else:
        p_tr, p_te, b_tr, b_te, test_genes = gene_split(df_path, df_benign_pool, targets, seed)
        p_tr = sample_up_to(p_tr, targets["train_path"], seed)
        p_te = sample_up_to(p_te, targets["test_path"], seed)
        b_tr = sample_up_to(b_tr, targets["train_benign"], seed)
        b_te = sample_up_to(b_te, targets["test_benign"], seed)
        gene_train_ids = set(p_tr["VariationID"].astype(str)) | set(b_tr["VariationID"].astype(str))
        gene_test_ids = set(p_te["VariationID"].astype(str)) | set(b_te["VariationID"].astype(str))
        assert not (set(p_tr["GeneSymbol"]) | set(b_tr["GeneSymbol"])) & test_genes

    # ── Pool = union of everything selected under either scheme ─────────
    pool = pd.concat([df_path, df_benign_pool], ignore_index=True)
    pool["VariationID"] = pool["VariationID"].astype(str)
    pool = pool.drop_duplicates(subset="VariationID")
    keep = variant_train_ids | variant_test_ids | gene_train_ids | gene_test_ids
    pool = pool[pool["VariationID"].isin(keep)].copy()

    def assign(ids_tr, ids_te):
        return np.where(pool["VariationID"].isin(ids_te), "test",
               np.where(pool["VariationID"].isin(ids_tr), "train", "unused"))

    pool["_split_variant"] = assign(variant_train_ids, variant_test_ids)
    pool["_split_gene"] = assign(gene_train_ids, gene_test_ids)
    pool["_split_gene_note"] = note
    pool["panel"] = name
    pool = pool.sample(frac=1, random_state=seed).reset_index(drop=True)

    summary = {"panel": name, "n_pool": int(len(pool)),
               "gene_split_note": note, "test_genes_gene_split": sorted(test_genes)}
    for scheme in ("variant", "gene"):
        col = f"_split_{scheme}"
        for side in ("train", "test"):
            m = pool[col] == side
            summary[f"{scheme}_{side}_path"] = int((pool.loc[m, "label"] == 1).sum())
            summary[f"{scheme}_{side}_benign"] = int((pool.loc[m, "label"] == 0).sum())
            summary[f"{scheme}_{side}_genes"] = int(pool.loc[m, "GeneSymbol"].nunique())
        log.info(f"  [{scheme:<7}] train P/B={summary[f'{scheme}_train_path']}/"
                 f"{summary[f'{scheme}_train_benign']} ({summary[f'{scheme}_train_genes']} genes)   "
                 f"test P/B={summary[f'{scheme}_test_path']}/{summary[f'{scheme}_test_benign']} "
                 f"({summary[f'{scheme}_test_genes']} genes)")
    if note:
        log.info(f"  gene split: {note}")
    return pool, summary


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/pipeline.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed = int(cfg["seed"])
    targets_cfg = cfg["panel_targets"]

    if not os.path.exists("data/raw/clinvar_filtered.csv"):
        sys.exit("Run step 01 first.")
    df_cv = pd.read_csv("data/raw/clinvar_filtered.csv", low_memory=False)
    df_cv["VariationID"] = df_cv["VariationID"].astype(str)
    log.info(f"ClinVar: {len(df_cv):,} variants (Path={int(df_cv['label'].sum()):,} "
             f"Benign={int((df_cv['label'] == 0).sum()):,})")

    df_gnomad = None
    if os.path.exists("data/raw/gnomad_benign.csv"):
        df_gnomad = pd.read_csv("data/raw/gnomad_benign.csv", low_memory=False)
        log.info(f"gnomAD benign: {len(df_gnomad):,}")
    else:
        log.warning("gnomAD file not found — benign will be ClinVar-only")

    gz_path = "data/raw/variant_summary.txt.gz"
    os.makedirs("data/interim", exist_ok=True)

    general_exclude = SPECIALTY_GENES if cfg.get("general_excludes_specialty_genes", True) \
        else (PAH_GENES | CFTR_GENES)
    panels = [
        ("cftr", CFTR_GENES, None),
        ("pah", PAH_GENES, None),
        ("cancer", CANCER_GENES, None),
        ("general", None, general_exclude),
    ]

    manifest = {"seed": seed, "general_excluded_genes": sorted(general_exclude), "panels": []}
    for name, gene_set, exclude in panels:
        pool, summary = build_panel(name, df_cv, df_gnomad, gz_path, gene_set, exclude,
                                    targets_cfg[name], seed)
        pool.to_csv(f"data/interim/panel_{name}_pool.csv", index=False)
        manifest["panels"].append(summary)

    # Cross-panel disjointness guard
    all_ids = pd.concat([pd.read_csv(f"data/interim/panel_{n}_pool.csv", usecols=["VariationID"])
                         .assign(panel=n) for n, _, _ in panels])
    dup = all_ids.duplicated(subset="VariationID", keep=False).sum()
    if dup:
        log.error(f"{dup} variants appear in more than one panel — investigate!")
    else:
        log.info("\n✓ Panels are variant-disjoint from each other")

    with open("data/interim/split_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("Manifest saved: data/interim/split_manifest.json")


if __name__ == "__main__":
    main()
