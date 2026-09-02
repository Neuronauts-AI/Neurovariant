"""
STEP 5 — Finalize datasets (named columns, both split schemes)
================================================================

Replaces the competition-era anonymisation step. Nothing is renamed;
column semantics are documented once in utils/schema.py.

For every split scheme (gene, variant) and every panel this writes
    data/final/<scheme>/train_<panel>.csv
    data/final/<scheme>/test_<panel>.csv
Both files carry the target column `label` and the metadata columns
`_variant_id, _gene, _panel, _source, _protein_change, _split_scheme`.

Also written (once, scheme-independent):
    data/final/feature_groups.json     column → feature group
    data/final/dataset_summary.csv     rows / genes / class balance per file
    data/final/split_manifest.json     copied from step 03

Cleaning performed here
-----------------------
* raw ±5 context letters are reconstructed from the one-hot windows of
  step 04c (the enricher in step 06 needs letters, not one-hots); the
  240 one-hot columns are dropped
* silico_mean / silico_variance / silico_n_tools are recomputed from the
  final in-silico columns (step 04d overwrites the raw scores)
* all-NaN and zero-variance columns are dropped (logged)
* test rows whose full feature vector is identical to a training row of
  the same scheme are removed from the test side (logged) — identical
  vectors are almost always the same variant deposited twice
"""

from __future__ import annotations

import os
import sys
import json
import shutil
import argparse
import numpy as np
import pandas as pd

from utils.logging_utils import get_logger
from utils.config import load_config
from utils.schema import (feature_group, group_table, TARGET_COL, INSILICO_COLS)

log = get_logger("05_finalize_dataset.log")

AA_CODES = list("ACDEFGHIKLMNPQRSTVWY")
NUC_CODES = list("ATGC")
CTX_POS = [f"up{i}" for i in range(5, 0, -1)] + [f"dn{i}" for i in range(1, 6)]

META_KEEP = ["_variant_id", "_gene", "_panel", "_source", "_protein_change"]
STRIP_COLS = {"am_class", "am_join_method", "label_confidence", "_ref_aa", "_alt_aa"}


# ── Cleaning helpers ─────────────────────────────────────────────────────────
def reconstruct_context_letters(df: pd.DataFrame) -> pd.DataFrame:
    """aa_ctx_up5_A..Y one-hots → aa_ctx_up5 letter (X/N when all-zero)."""
    for prefix, codes, pad in (("aa_ctx", AA_CODES, "X"), ("nuc_ctx", NUC_CODES, "N")):
        for pos in CTX_POS:
            raw = f"{prefix}_{pos}"
            onehots = [f"{raw}_{c}" for c in codes if f"{raw}_{c}" in df.columns]
            if raw in df.columns and df[raw].dtype == object:
                pass  # letters already present
            elif onehots:
                mat = df[onehots].fillna(0).to_numpy()
                letters = np.array([c.rsplit("_", 1)[1] for c in onehots])
                hit = mat.argmax(axis=1)
                df[raw] = np.where(mat.max(axis=1) > 0, letters[hit], pad)
            if onehots:
                df = df.drop(columns=onehots)
    return df


def recompute_silico_summary(df: pd.DataFrame) -> pd.DataFrame:
    tools = [c for c in ("cadd_phred", "revel_score", "polyphen2_hdiv",
                         "polyphen2_hvar", "sift_score", "alphamissense") if c in df.columns]
    if not tools:
        return df
    # direction-normalise to [0,1] pathogenic-high before averaging
    m = pd.DataFrame(index=df.index)
    for t in tools:
        v = pd.to_numeric(df[t], errors="coerce")
        if t == "cadd_phred":
            v = (v / 40.0).clip(upper=1.0)
        elif t == "sift_score":
            v = 1.0 - v
        m[t] = v
    df["silico_mean"] = m.mean(axis=1)
    df["silico_variance"] = m.var(axis=1, ddof=0)
    df["silico_n_tools"] = m.notna().sum(axis=1)
    return df


def drop_unfeaturised(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    A missense variant whose protein change could not be parsed has no
    amino-acid identity, no biochemistry and no context window. Such rows
    are empty feature vectors that a model would learn to map onto whatever
    label they carry (in the legacy competition data 100 % of PAH/CFTR
    benign rows were of this kind). They are removed here, per class, and
    the counts are logged so the paper can report them.
    """
    req = [c for c in ("ref_aa", "alt_aa", "aa_position") if c in df.columns]
    if not req:
        return df
    bad = df[req].isna().any(axis=1)
    if bad.any():
        by_label = df.loc[bad, "_label"].value_counts().to_dict() if "_label" in df else {}
        by_src = df.loc[bad, "_source"].value_counts().to_dict() if "_source" in df else {}
        log.warning(f"  [{name}] dropping {int(bad.sum())} rows without a parsed protein change "
                    f"(by label {by_label}; by source {by_src})")
    return df.loc[~bad].reset_index(drop=True)


def drop_degenerate(df: pd.DataFrame, name: str) -> pd.DataFrame:
    feat = [c for c in df.columns if feature_group(c) is not None]
    all_nan = [c for c in feat if df[c].isna().all()]
    num = df[feat].select_dtypes(include="number").columns
    zero_var = [c for c in num if c not in all_nan and df[c].std(ddof=0) == 0]
    if all_nan:
        log.info(f"  [{name}] dropping {len(all_nan)} all-NaN columns: {all_nan[:8]}{' …' if len(all_nan) > 8 else ''}")
    if zero_var:
        log.info(f"  [{name}] dropping {len(zero_var)} zero-variance columns: {zero_var[:8]}{' …' if len(zero_var) > 8 else ''}")
    return df.drop(columns=all_nan + zero_var)


def remove_test_duplicates(tr: pd.DataFrame, te: pd.DataFrame, name: str) -> pd.DataFrame:
    feat = [c for c in te.columns if feature_group(c) is not None and c in tr.columns]
    tr_keys = set(map(tuple, tr[feat].astype(str).to_numpy()))
    mask = np.array([tuple(r) in tr_keys for r in te[feat].astype(str).to_numpy()])
    if mask.any():
        log.warning(f"  [{name}] removed {int(mask.sum())} test rows identical to a training row")
    return te.loc[~mask].reset_index(drop=True)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/pipeline.yaml")
    ap.add_argument("--interim-dir", default="data/interim")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_root = cfg["paths"]["final"]
    panels = cfg["panels"]
    schemes = cfg["split_schemes"]

    pools = {}
    for panel in panels:
        path = os.path.join(args.interim_dir, f"panel_{panel}_pool_features.csv")
        if not os.path.exists(path):
            log.warning(f"Missing {path} — run steps 03/04 first")
            continue
        df = pd.read_csv(path, low_memory=False)
        log.info(f"\n{'─' * 60}\nPanel {panel.upper()}: {len(df):,} rows × {df.shape[1]} cols")
        df = df.drop(columns=[c for c in STRIP_COLS if c in df.columns])
        df = drop_unfeaturised(df, panel)
        df = reconstruct_context_letters(df)
        df = recompute_silico_summary(df)
        df = drop_degenerate(df, panel)
        pools[panel] = df
    if not pools:
        sys.exit("Nothing to finalize.")

    # Harmonise column sets across panels (union; missing → NaN)
    all_cols = sorted(set().union(*[set(d.columns) for d in pools.values()]))
    feat_cols = [c for c in all_cols if feature_group(c) is not None]
    ordered = ["_variant_id", "_gene", "_panel", "_source", "_protein_change"] + \
              [c for c in feat_cols if c not in INSILICO_COLS] + \
              [c for c in feat_cols if c in INSILICO_COLS] + [TARGET_COL]

    summary = []
    for scheme in schemes:
        out_dir = os.path.join(out_root, scheme)
        os.makedirs(out_dir, exist_ok=True)
        split_col = f"_split_{scheme}"
        for panel, df in pools.items():
            if split_col not in df.columns:
                sys.exit(f"{split_col} missing in panel {panel}: rerun step 03")
            d = df.copy()
            d[TARGET_COL] = d["_label"].astype(int)
            for c in ordered:
                if c not in d.columns:
                    d[c] = np.nan
            tr = d[d[split_col] == "train"][ordered].reset_index(drop=True)
            te = d[d[split_col] == "test"][ordered].reset_index(drop=True)
            te = remove_test_duplicates(tr, te, f"{scheme}/{panel}")
            tr["_split_scheme"] = scheme
            te["_split_scheme"] = scheme
            note = ""
            if scheme == "gene" and "_split_gene_note" in df.columns:
                v = df["_split_gene_note"].dropna().astype(str)
                note = v.iloc[0] if len(v) and v.iloc[0] not in ("", "nan") else ""
            if scheme == "gene":
                overlap = set(tr["_gene"]) & set(te["_gene"])
                if overlap and not note:
                    log.error(f"  [{scheme}/{panel}] gene overlap train/test: {sorted(overlap)[:10]}")
            tr.to_csv(os.path.join(out_dir, f"train_{panel}.csv"), index=False)
            te.to_csv(os.path.join(out_dir, f"test_{panel}.csv"), index=False)
            for side, x in (("train", tr), ("test", te)):
                summary.append({
                    "scheme": scheme, "panel": panel, "side": side, "n": len(x),
                    "n_pathogenic": int(x[TARGET_COL].sum()),
                    "n_benign": int((x[TARGET_COL] == 0).sum()),
                    "n_genes": int(x["_gene"].nunique()),
                    "sources": json.dumps(x["_source"].value_counts().to_dict()),
                    "n_features": len(feat_cols), "note": note,
                })
            log.info(f"  [{scheme:<7}/{panel:<7}] train={len(tr):>5} ({tr['_gene'].nunique():>3} genes)  "
                     f"test={len(te):>5} ({te['_gene'].nunique():>3} genes)  {note}")

    groups = group_table(feat_cols)
    with open(os.path.join(out_root, "feature_groups.json"), "w") as f:
        json.dump({"by_group": groups, "by_column": {c: feature_group(c) for c in feat_cols}}, f, indent=2)
    pd.DataFrame(summary).to_csv(os.path.join(out_root, "dataset_summary.csv"), index=False)
    manifest = os.path.join(args.interim_dir, "split_manifest.json")
    if os.path.exists(manifest):
        shutil.copy(manifest, os.path.join(out_root, "split_manifest.json"))

    log.info(f"\n{'=' * 60}\nFINALIZED: {len(feat_cols)} features")
    for g, cols in sorted(groups.items()):
        log.info(f"  {g:<16} {len(cols):>4}")
    log.info(f"Outputs under {out_root}/<scheme>/")


if __name__ == "__main__":
    main()
