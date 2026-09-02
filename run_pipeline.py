#!/usr/bin/env python3
"""
NeuroVariant — end-to-end orchestrator
======================================

    python run_pipeline.py data              # 01 → 05  (network: ClinVar, gnomAD, MyVariant, Ensembl)
    python run_pipeline.py model             # 06 → 11  over the whole experimental matrix
    python run_pipeline.py all
    python run_pipeline.py model --fast      # smoke test: 15 Optuna trials, 200 bootstrap
    python run_pipeline.py model --schemes gene --feature-sets core

Every step is a plain script in scripts/; this file only sequences them and
stops at the first failure. Run from the repository root.
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
from utils.config import load_config  # noqa: E402

DATA_STEPS = [
    "01_fetch_clinvar.py",
    "01b_supplement_benign.py",
    "02_fetch_gnomad.py",
    "03_build_panels.py",
    "04_annotate_features.py",
    "04b_validate_panel_consistency.py",
    "04c_prefetch_cds.py",
    "04c_add_context_windows.py",
    "04d_add_layer1_scores.py",
    "05_finalize_dataset.py",
]


def run(cmd: list[str]) -> None:
    print(f"\n▶ {' '.join(cmd)}", flush=True)
    t = time.time()
    r = subprocess.run([sys.executable] + cmd)
    if r.returncode != 0:
        sys.exit(f"✗ step failed ({cmd[0]}) — see logs/")
    print(f"  done in {time.time() - t:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["data", "model", "all"])
    ap.add_argument("--config", default="config/pipeline.yaml")
    ap.add_argument("--schemes", nargs="+", default=None)
    ap.add_argument("--feature-sets", nargs="+", default=None)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--n-trials", type=int, default=None)
    ap.add_argument("--skip-data-steps", nargs="*", default=[], help="e.g. 02_fetch_gnomad.py")
    args = ap.parse_args()
    cfg = load_config(args.config)
    schemes = args.schemes or cfg["split_schemes"]
    fsets = args.feature_sets or cfg["feature_sets"]
    fast = ["--fast"] if args.fast else []
    trials = ["--n-trials", str(args.n_trials)] if args.n_trials else []

    if args.stage in ("data", "all"):
        for s in DATA_STEPS:
            if s in args.skip_data_steps:
                print(f"  (skipping {s})")
                continue
            run([f"scripts/{s}"] if s != "03_build_panels.py" and s != "05_finalize_dataset.py"
                else [f"scripts/{s}", "--config", args.config])

    if args.stage in ("model", "all"):
        for scheme in schemes:
            run(["scripts/06_enrich_features.py", "--scheme", scheme, "--validate", "--config", args.config])
            for fset in fsets:
                run(["scripts/07_track_c_acmg_prior.py", "--scheme", scheme, "--feature-set", fset, "--config", args.config])
                run(["scripts/08_train_track_a.py", "--scheme", scheme, "--feature-set", fset, "--config", args.config] + fast + trials)
                run(["scripts/09_meta_learner.py", "--scheme", scheme, "--feature-set", fset, "--config", args.config] + fast)
            run(["scripts/10_baselines.py", "--scheme", scheme, "--config", args.config] + fast)
        run(["scripts/11_paper_tables.py", "--config", args.config] + fast)
    print("\n✓ pipeline finished")


if __name__ == "__main__":
    main()
