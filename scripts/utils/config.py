"""Load config/pipeline.yaml with defaults so every script shares one source of truth."""

from __future__ import annotations

import os
import copy

DEFAULTS = {
    "seed": 42,
    "panels": ["general", "cancer", "pah", "cftr"],
    "general_excludes_specialty_genes": True,
    "panel_targets": {
        "general": {"train_path": 1500, "train_benign": 1500, "test_path": 1000, "test_benign": 1000},
        "cancer":  {"train_path": 200,  "train_benign": 200,  "test_path": 100,  "test_benign": 100},
        "pah":     {"train_path": 200,  "train_benign": 200,  "test_path": 100,  "test_benign": 100},
        "cftr":    {"train_path": 70,   "train_benign": 70,   "test_path": 30,   "test_benign": 30},
    },
    "split_schemes": ["gene", "variant"],
    "feature_sets": ["core", "full"],
    "model": {
        "n_folds": 5,
        "n_trials": 200,
        "calibration_fraction": 0.15,
        "min_sensitivity": 0.90,
        "variance_threshold": 0.01,
        "correlation_threshold": 0.95,
        "n_bootstrap": 2000,
        "shap_max_display": 20,
    },
    "paths": {
        "final": "data/final",
        "enriched": "data/enriched",
        "track_c": "data/track_c",
        "results": "results",
    },
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str = "config/pipeline.yaml") -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path and os.path.exists(path):
        try:
            import yaml
        except ImportError as e:  # pragma: no cover
            raise SystemExit("pip install pyyaml") from e
        with open(path) as f:
            user = yaml.safe_load(f) or {}
        cfg = _merge(cfg, user)
    return cfg


def run_dir(cfg: dict, split: str, feature_set: str) -> str:
    """results/<split>_<feature_set>/"""
    return os.path.join(cfg["paths"]["results"], f"{split}_{feature_set}")
