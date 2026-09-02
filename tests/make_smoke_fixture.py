"""
Build a smoke-test fixture (data/interim/panel_*_pool_features.csv) from the
legacy competition Dataset3 folder, so that steps 05 → 11 can be exercised
without re-running the network-bound data stage.

    python tests/make_smoke_fixture.py --dataset3 /path/to/Model_Training/Dataset3 --out data_smoke

Anonymous columns are restored to their real names via column_schema.json.
Training rows of Dataset3 carry no gene symbol, so they receive pseudo-genes
(TRAIN_G00..TRAIN_G19); test rows keep their real genes. This is sufficient
to exercise the gene-grouped code paths but is NOT a scientific dataset.
"""

from __future__ import annotations

import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.simplefilter("ignore")

PANELS = ["general", "cancer", "pah", "cftr"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset3", required=True)
    ap.add_argument("--out", default="data_smoke")
    ap.add_argument("--max-general", type=int, default=800, help="subsample general train rows")
    args = ap.parse_args()
    d3 = args.dataset3
    schema = json.load(open(os.path.join(d3, "column_schema.json")))
    interim = os.path.join(args.out, "interim")
    os.makedirs(interim, exist_ok=True)
    rng = np.random.RandomState(0)

    for p in PANELS:
        tr = pd.read_csv(os.path.join(d3, f"train_{p}.csv"), low_memory=False)
        te = pd.read_csv(os.path.join(d3, f"test_{p}.csv"), low_memory=False)
        lab = pd.read_csv(os.path.join(d3, f"test_{p}_labels.csv"))
        te["label"] = lab["label"].values
        te["_gene"] = lab["gene"].values
        te["_source"] = lab["source"].values
        if p == "general" and len(tr) > args.max_general:
            tr = tr.sample(n=args.max_general, random_state=0)
        tr["_gene"] = [f"TRAIN_G{rng.randint(20):02d}" for _ in range(len(tr))]
        tr["_source"] = "ClinVar"
        tr["_split"] = "train"; te["_split"] = "test"
        df = pd.concat([tr, te], ignore_index=True).rename(columns=schema)
        df["_label"] = df.pop("label").astype(int)
        df["_variant_id"] = [f"{p}_{i}" for i in range(len(df))]
        df["_panel"] = p
        df["_protein_change"] = df["ref_aa"].astype(str) + df["aa_position"].fillna(0).astype(int).astype(str) + df["alt_aa"].astype(str)
        df["_split_variant"] = df.pop("_split")
        df["_split_gene"] = df["_split_variant"]
        df["_split_gene_note"] = "within-gene (smoke fixture)" if p in ("pah", "cftr") else ""
        df.to_csv(os.path.join(interim, f"panel_{p}_pool_features.csv"), index=False)
        print(f"{p:<8} {df.shape}")
    json.dump({"note": "smoke fixture built from legacy Dataset3"}, open(os.path.join(interim, "split_manifest.json"), "w"))
    print(f"fixture → {interim}")


if __name__ == "__main__":
    main()
