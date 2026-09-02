"""
STEP 4b — Cross-Panel Feature Consistency Validation
=====================================================
Checks that all annotated panel files are consistent before finalization.

Checks performed:
  1. Column presence  — every panel has the same feature columns
  2. Value ranges     — continuous features stay within expected bounds
  3. Null rates       — flags features with >50% NaN in any panel
  4. Near-duplicates  — detects cross-source duplicates via row hashing
                        (same AA change + same gene in both ClinVar + gnomAD)
  5. Label balance    — warns if any panel deviates >10% from 50/50

OUTPUT: logs/panel_consistency_report.json
        (citable in rubric 3.4 and 4.4)
"""

import os
import sys
import json
import hashlib
import pandas as pd
import numpy as np

from utils.logging_utils import get_logger

log = get_logger("04b_validate_panel_consistency.log")

# ── Panel file paths (annotated, before finalization) ─────────────────────────
PANEL_FILES = {
    "general_pool": "data/interim/panel_general_pool_features.csv",
    "cancer_pool":  "data/interim/panel_cancer_pool_features.csv",
    "pah_pool":     "data/interim/panel_pah_pool_features.csv",
    "cftr_pool":    "data/interim/panel_cftr_pool_features.csv",
}

# Expected value ranges for key continuous features
EXPECTED_RANGES = {
    "grantham_dist":    (0, 215),
    "aa_position":      (1, 35000),
    "am_pathogenicity": (0.0, 1.0),
    "pli":              (0.0, 1.0),
    "loeuf":            (0.0, 3.0),
    "mis_z":            (-10.0, 10.0),
}

NULL_RATE_WARN_THRESHOLD = 0.50   # warn if >50% NaN
BALANCE_WARN_THRESHOLD   = 0.10   # warn if label imbalance >10%


def variant_hash(row: pd.Series) -> str:
    """Hash: GeneSymbol + protein AA change (ref+pos+alt) — for near-dup detection."""
    gene = str(row.get("GeneSymbol", "")).strip()
    pc   = str(row.get("ProteinChange", "") or row.get("Name", "")).strip()
    return hashlib.md5(f"{gene}|{pc}".encode()).hexdigest()


def check_panel(name: str, path: str, reference_cols: set | None) -> dict:
    """Run all checks on one panel file. Returns findings dict."""
    findings = {"panel": name, "path": path, "issues": []}

    if not os.path.exists(path):
        findings["issues"].append(f"FILE NOT FOUND: {path}")
        findings["status"] = "missing"
        return findings

    df = pd.read_csv(path, low_memory=False)
    findings["n_rows"]   = len(df)
    findings["n_cols"]   = len(df.columns)

    # ── 1. Column presence ────────────────────────────────────────────────
    current_cols = set(df.columns)
    if reference_cols is not None:
        missing = reference_cols - current_cols
        extra   = current_cols - reference_cols
        if missing:
            findings["issues"].append(f"Missing columns vs reference: {sorted(missing)[:10]}")
        if extra:
            findings["issues"].append(f"Extra columns vs reference: {sorted(extra)[:10]}")
    findings["columns"] = sorted(current_cols)

    # ── 2. Value ranges ───────────────────────────────────────────────────
    range_violations = []
    for feat, (lo, hi) in EXPECTED_RANGES.items():
        if feat not in df.columns:
            continue
        col = pd.to_numeric(df[feat], errors="coerce").dropna()
        if len(col) == 0:
            continue
        if col.min() < lo or col.max() > hi:
            range_violations.append(
                f"{feat}: got [{col.min():.3f}, {col.max():.3f}], "
                f"expected [{lo}, {hi}]"
            )
    if range_violations:
        findings["issues"].extend(range_violations)

    # ── 3. Null rates ─────────────────────────────────────────────────────
    high_null = {}
    for col in df.columns:
        null_rate = df[col].isna().mean()
        if null_rate > NULL_RATE_WARN_THRESHOLD:
            high_null[col] = round(null_rate, 3)
    if high_null:
        findings["high_null_features"] = high_null
        findings["issues"].append(
            f"{len(high_null)} features with >{NULL_RATE_WARN_THRESHOLD*100:.0f}% NaN"
        )

    # ── 4. Near-duplicate detection ───────────────────────────────────────
    if "GeneSymbol" in df.columns:
        df["_hash"] = df.apply(variant_hash, axis=1)
        n_dups = df["_hash"].duplicated().sum()
        if n_dups > 0:
            findings["issues"].append(
                f"Near-duplicates (same gene+AA change): {n_dups} rows"
            )
            findings["near_duplicate_count"] = int(n_dups)
        df.drop(columns=["_hash"], inplace=True)

    # ── 5. Label balance ──────────────────────────────────────────────────
    if "label" in df.columns:
        balance = df["label"].mean()
        findings["label_balance"] = round(float(balance), 4)
        if abs(balance - 0.5) > BALANCE_WARN_THRESHOLD:
            findings["issues"].append(
                f"Label imbalance: {balance:.1%} pathogenic "
                f"(expected ~50%)"
            )

    findings["status"] = "FAIL" if findings["issues"] else "OK"
    return findings


def main():
    log.info("=" * 60)
    log.info("STEP 4b — Panel Consistency Validation")
    log.info("=" * 60)

    results   = []
    ref_cols  = None
    any_fail  = False

    for name, path in PANEL_FILES.items():
        log.info(f"\nChecking: {name}")
        finding = check_panel(name, path, ref_cols)

        # Use general_train as the column reference
        if name == "general_pool" and "columns" in finding:
            ref_cols = set(finding["columns"])

        status = finding["status"]
        log.info(f"  Status: {status}  |  rows={finding.get('n_rows','?')}  "
                 f"cols={finding.get('n_cols','?')}")
        for issue in finding.get("issues", []):
            log.warning(f"  ⚠  {issue}")

        if status == "FAIL":
            any_fail = True

        results.append(finding)

    # ── Summary ───────────────────────────────────────────────────────────
    report = {
        "summary": {
            "panels_checked": len(results),
            "panels_ok":   sum(1 for r in results if r["status"] == "OK"),
            "panels_fail": sum(1 for r in results if r["status"] == "FAIL"),
            "panels_missing": sum(1 for r in results if r["status"] == "missing"),
        },
        "panels": results,
    }

    out_path = "logs/panel_consistency_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    log.info(f"\n{'='*60}")
    log.info(f"VALIDATION COMPLETE")
    log.info(f"  Panels OK:      {report['summary']['panels_ok']}")
    log.info(f"  Panels FAIL:    {report['summary']['panels_fail']}")
    log.info(f"  Panels MISSING: {report['summary']['panels_missing']}")
    log.info(f"  Report:         {out_path}")

    if any_fail:
        log.error("\n✗  Issues found — fix before running step 05.")
        sys.exit(1)
    else:
        log.info("\n✓  All panels passed consistency checks.")


if __name__ == "__main__":
    main()
