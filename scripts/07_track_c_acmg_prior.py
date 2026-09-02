"""
STEP 7 — Track C: rule-based (ACMG-inspired) prior probability
================================================================

A deterministic, fully transparent scorer. Every rule reads a NAMED column
(utils/schema.py); there is no column fingerprinting. Its output feeds the
stacking meta-learner (step 09) and is also reported standalone.

Evidence items and weights (log-odds units, summed then passed through a
sigmoid). Items with a direct ACMG/AMP counterpart keep the ACMG code;
the others are labelled descriptively so the paper does not over-claim.

  BA1             maf_global > 0.05                    stand-alone benign (override → P≈0.002)
  BS1             maf_global > 0.01                    -2
  BS2             any sub-population AF > 0.005        -1
  PP3 / BP4       mean normalised in-silico > 0.7 / < 0.3   +1 / -1   (feature-set "full" only)
  CONS_HIGH       max(conservation) > 2.5              +2   (phyloP/GERP: strongly conserved site)
  GENE_INTOL      pLI > 0.9                            +2   (gene-level intolerance to variation)
  RADICAL_SUB     Grantham > 150                       +2
  CHARGE_REV      charge reversal (K/R/H ↔ D/E)        +2
  CONSERV_SUB     Grantham < 30 and same charge/polarity   -1

USAGE
  python scripts/07_track_c_acmg_prior.py --scheme gene --feature-set core
Outputs
  data/track_c/<scheme>_<feature_set>/{train,test}_<panel>_trackc.csv
     columns: _variant_id, _gene, _panel, label, track_c_logit, track_c_score,
              track_c_n_fired, track_c_rules
"""

from __future__ import annotations

import os
import json
import argparse
import numpy as np
import pandas as pd
from scipy.special import expit

from utils.logging_utils import get_logger
from utils.config import load_config
from utils.schema import (INSILICO_THRESHOLDS, CONSERVATION_COLS, TARGET_COL)

log = get_logger("07_track_c.log")

WEIGHTS = {
    "BA1": -6.0, "BS1": -2.0, "BS2": -1.0,
    "PP3": +1.0, "BP4": -1.0,
    "CONS_HIGH": +2.0, "GENE_INTOL": +2.0,
    "RADICAL_SUB": +2.0, "CHARGE_REV": +2.0, "CONSERV_SUB": -1.0,
}
SUBPOP_COLS = [f"maf_{p}" for p in ("afr", "amr", "eas", "fin", "nfe", "sas")]
INSILICO_FOR_PP3 = ["cadd_phred", "revel_score", "polyphen2_hdiv",
                    "polyphen2_hvar", "sift_score", "alphamissense"]

# Physicochemical tables (identical to 04_annotate_features.py)
CHARGE = {"R": 1, "K": 1, "D": -1, "E": -1}                       # H neutral, as in step 04
POLAR_CLASS = {**{a: 0 for a in "AVLIMFWGP"}, **{a: 1 for a in "STCYNQH"}, **{a: 2 for a in "RKDE"}}
_GD = {
    ("A","R"):112,("A","N"):111,("A","D"):126,("A","C"):195,("A","Q"): 91,
    ("A","E"):107,("A","G"): 60,("A","H"): 86,("A","I"): 94,("A","L"): 96,
    ("A","K"):106,("A","M"): 84,("A","F"):113,("A","P"): 27,("A","S"): 99,
    ("A","T"): 58,("A","W"):148,("A","Y"):112,("A","V"): 64,
    ("R","N"): 86,("R","D"): 96,("R","C"):180,("R","Q"): 43,("R","E"): 54,
    ("R","G"):125,("R","H"): 29,("R","I"): 97,("R","L"):102,("R","K"): 26,
    ("R","M"): 91,("R","F"): 97,("R","P"):103,("R","S"):110,("R","T"): 71,
    ("R","W"):101,("R","Y"): 77,("R","V"): 96,
    ("N","D"): 23,("N","C"):139,("N","Q"): 46,("N","E"): 42,("N","G"): 80,
    ("N","H"): 68,("N","I"):149,("N","L"):153,("N","K"): 94,("N","M"):142,
    ("N","F"):158,("N","P"): 91,("N","S"): 46,("N","T"): 65,("N","W"):174,
    ("N","Y"):143,("N","V"):133,
    ("D","C"):154,("D","Q"): 61,("D","E"): 45,("D","G"): 94,("D","H"): 81,
    ("D","I"):168,("D","L"):172,("D","K"):101,("D","M"):160,("D","F"):177,
    ("D","P"):108,("D","S"): 65,("D","T"): 85,("D","W"):181,("D","Y"):160,
    ("D","V"):152,
    ("C","Q"):154,("C","E"):170,("C","G"):159,("C","H"):174,("C","I"):198,
    ("C","L"):198,("C","K"):202,("C","M"):196,("C","F"):205,("C","P"):169,
    ("C","S"):112,("C","T"):149,("C","W"):215,("C","Y"):194,("C","V"):192,
    ("Q","E"): 29,("Q","G"): 87,("Q","H"): 24,("Q","I"):109,("Q","L"):113,
    ("Q","K"):  53,("Q","M"): 101,("Q","F"):116,("Q","P"): 76,("Q","S"): 68,
    ("Q","T"): 42,("Q","W"):130,("Q","Y"):99, ("Q","V"): 96,
    ("E","G"):98, ("E","H"): 40,("E","I"):134,("E","L"):138,("E","K"): 56,
    ("E","M"):126,("E","F"):140,("E","P"):93, ("E","S"): 80,("E","T"): 65,
    ("E","W"):152,("E","Y"):122,("E","V"):121,
    ("G","H"):98, ("G","I"):135,("G","L"):138,("G","K"):127,("G","M"):127,
    ("G","F"):153,("G","P"): 42,("G","S"): 56,("G","T"): 59,("G","W"):184,
    ("G","Y"):147,("G","V"): 109,
    ("H","I"): 94,("H","L"): 99,("H","K"):32, ("H","M"): 87,("H","F"): 100,
    ("H","P"): 77,("H","S"): 89,("H","T"): 47,("H","W"):115,("H","Y"): 83,
    ("H","V"):84,
    ("I","L"):  5,("I","K"):102,("I","M"): 10,("I","F"): 21,("I","P"):95,
    ("I","S"):142,("I","T"):89, ("I","W"): 61,("I","Y"): 33,("I","V"): 29,
    ("L","K"):107,("L","M"): 15,("L","F"): 22,("L","P"):98, ("L","S"):145,
    ("L","T"): 92,("L","W"): 61,("L","Y"): 36,("L","V"): 32,
    ("K","M"):95, ("K","F"):102,("K","P"):103,("K","S"):121,("K","T"): 78,
    ("K","W"):110,("K","Y"): 85,("K","V"):97,
    ("M","F"): 28,("M","P"):87, ("M","S"):135,("M","T"): 81,("M","W"): 67,
    ("M","Y"): 36,("M","V"): 21,
    ("F","P"):114,("F","S"):155,("F","T"):103,("F","W"): 40,("F","Y"): 22,
    ("F","V"): 50,
    ("P","S"): 74,("P","T"): 38,("P","W"):147,("P","Y"):110,("P","V"): 68,
    ("S","T"): 58,("S","W"):177,("S","Y"):144,("S","V"):124,
    ("T","W"):128,("T","Y"): 92,("T","V"): 69,
    ("W","Y"): 37,("W","V"):88,
    ("Y","V"): 55,
}


def grantham(a: str, b: str) -> float:
    if a == b:
        return 0.0
    return float(_GD.get((a, b), _GD.get((b, a), 100)))


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def score_row(row: pd.Series, use_insilico: bool) -> tuple[float, list[str]]:
    fired: list[str] = []

    # ── population ─────────────────────────────────────────────────────
    maf = _num(row.get("maf_global", np.nan))
    if not np.isnan(maf) and maf > 0.05:
        return WEIGHTS["BA1"], ["BA1"]           # stand-alone override
    if not np.isnan(maf) and maf > 0.01:
        fired.append("BS1")
    sub = [_num(row.get(c, np.nan)) for c in SUBPOP_COLS if c in row.index]
    sub = [s for s in sub if not np.isnan(s)]
    if sub and max(sub) > 0.005:
        fired.append("BS2")

    # ── in silico (only when the feature set permits it) ───────────────
    if use_insilico:
        normed = []
        for c in INSILICO_FOR_PP3:
            v = _num(row.get(c, np.nan))
            if np.isnan(v):
                continue
            if c == "cadd_phred":
                normed.append(min(v / 40.0, 1.0))
            elif c == "sift_score":
                normed.append(1.0 - v)
            else:
                normed.append(v)
        if normed:
            m = float(np.mean(normed))
            if m > 0.7:
                fired.append("PP3")
            elif m < 0.3:
                fired.append("BP4")

    # ── conservation / gene constraint ─────────────────────────────────
    cons = [_num(row.get(c, np.nan)) for c in CONSERVATION_COLS if c in row.index]
    cons = [c for c in cons if not np.isnan(c)]
    if cons and max(cons) > 2.5:
        fired.append("CONS_HIGH")
    pli = _num(row.get("pli", np.nan))
    if not np.isnan(pli) and pli > 0.9:
        fired.append("GENE_INTOL")

    # ── substitution biochemistry ──────────────────────────────────────
    ref, alt = str(row.get("ref_aa", "X")), str(row.get("alt_aa", "X"))
    gd = grantham(ref, alt)
    rc, ac = CHARGE.get(ref, 0), CHARGE.get(alt, 0)
    if gd > 150:
        fired.append("RADICAL_SUB")
    if (rc > 0 and ac < 0) or (rc < 0 and ac > 0):
        fired.append("CHARGE_REV")
    if gd < 30 and rc == ac and POLAR_CLASS.get(ref, -1) == POLAR_CLASS.get(alt, -2):
        fired.append("CONSERV_SUB")

    return float(sum(WEIGHTS[r] for r in fired)), fired


def process(df: pd.DataFrame, use_insilico: bool) -> pd.DataFrame:
    logits, rules = [], []
    for _, row in df.iterrows():
        lg, fr = score_row(row, use_insilico)
        logits.append(lg)
        rules.append(",".join(fr))
    out = pd.DataFrame({
        "_variant_id": df.get("_variant_id", pd.Series(range(len(df)))).values,
        "_gene": df.get("_gene", "").values if "_gene" in df else "",
        "_panel": df.get("_panel", "").values if "_panel" in df else "",
    })
    if TARGET_COL in df.columns:
        out[TARGET_COL] = df[TARGET_COL].values
    out["track_c_logit"] = logits
    out["track_c_score"] = expit(np.array(logits))
    out["track_c_n_fired"] = [len(r.split(",")) if r else 0 for r in rules]
    out["track_c_rules"] = rules
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/pipeline.yaml")
    ap.add_argument("--scheme", default="gene", choices=["gene", "variant"])
    ap.add_argument("--feature-set", default="core", choices=["core", "full"])
    ap.add_argument("--input-dir", default=None, help="default data/enriched/<scheme>")
    ap.add_argument("--output-dir", default=None, help="default data/track_c/<scheme>_<feature_set>")
    args = ap.parse_args()
    cfg = load_config(args.config)
    in_dir = args.input_dir or os.path.join(cfg["paths"]["enriched"], args.scheme)
    out_dir = args.output_dir or os.path.join(cfg["paths"]["track_c"], f"{args.scheme}_{args.feature_set}")
    os.makedirs(out_dir, exist_ok=True)
    use_insilico = args.feature_set == "full"
    log.info(f"Track C  scheme={args.scheme}  feature_set={args.feature_set}  "
             f"in-silico rules {'ON' if use_insilico else 'OFF'}")

    from sklearn.metrics import roc_auc_score
    report = {}
    for panel in cfg["panels"]:
        for side in ("train", "test"):
            path = os.path.join(in_dir, f"{side}_{panel}_enriched.csv")
            if not os.path.exists(path):
                path = os.path.join(in_dir, f"{side}_{panel}.csv")
            if not os.path.exists(path):
                log.warning(f"  missing {path}")
                continue
            df = pd.read_csv(path, low_memory=False)
            out = process(df, use_insilico)
            out.to_csv(os.path.join(out_dir, f"{side}_{panel}_trackc.csv"), index=False)
            msg = f"  {side:<5} {panel:<8} n={len(out):>5}  mean P={out['track_c_score'].mean():.3f}"
            if TARGET_COL in out.columns and out[TARGET_COL].nunique() == 2:
                auc = roc_auc_score(out[TARGET_COL], out["track_c_score"])
                report[f"{side}_{panel}_auc"] = float(auc)
                msg += f"  standalone AUC={auc:.4f}"
            fire = pd.Series([r for rs in out["track_c_rules"] for r in rs.split(",") if r]).value_counts()
            report[f"{side}_{panel}_rule_counts"] = fire.to_dict()
            log.info(msg)
    with open(os.path.join(out_dir, "track_c_report.json"), "w") as f:
        json.dump({"weights": WEIGHTS, "use_insilico": use_insilico, **report}, f, indent=2)
    log.info(f"Outputs → {out_dir}")


if __name__ == "__main__":
    main()
