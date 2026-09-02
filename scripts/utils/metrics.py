"""Evaluation helpers shared by steps 08–11 (thresholds, bootstrap CIs, DeLong)."""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score, brier_score_loss,
                             matthews_corrcoef)


def evaluate(y_true, y_proba, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "n": int(len(y_true)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0,
        "brier": float(brier_score_loss(y_true, y_proba)),
        "threshold": float(threshold),
    }


def best_threshold_youden(y_true, y_proba) -> float:
    y_true = np.asarray(y_true).astype(int)
    best_j, best_t = -1.0, 0.5
    for t in np.arange(0.05, 0.96, 0.01):
        pred = (y_proba >= t).astype(int)
        sens = recall_score(y_true, pred, zero_division=0)
        spec = recall_score(y_true, pred, pos_label=0, zero_division=0)
        if sens + spec - 1 > best_j:
            best_j, best_t = sens + spec - 1, float(t)
    return best_t


def best_threshold_sensitivity(y_true, y_proba, min_sens: float = 0.90) -> float:
    """Highest threshold that still achieves sensitivity >= min_sens."""
    y_true = np.asarray(y_true).astype(int)
    best = 0.05
    for t in np.arange(0.05, 0.96, 0.01):
        if recall_score(y_true, (y_proba >= t).astype(int), zero_division=0) >= min_sens:
            best = float(t)
    return best


def bootstrap_ci(y_true, y_proba, threshold: float, n_boot: int = 2000,
                 seed: int = 42, ci: float = 0.95) -> dict:
    rng = np.random.RandomState(seed)
    yt = np.asarray(y_true).astype(int)
    yp = np.asarray(y_proba, dtype=float)
    n = len(yt)
    store = {"roc_auc": [], "pr_auc": [], "f1": [], "sensitivity": [], "specificity": []}
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(yt[idx])) < 2:
            continue
        pred = (yp[idx] >= threshold).astype(int)
        store["roc_auc"].append(roc_auc_score(yt[idx], yp[idx]))
        store["pr_auc"].append(average_precision_score(yt[idx], yp[idx]))
        store["f1"].append(f1_score(yt[idx], pred, zero_division=0))
        store["sensitivity"].append(recall_score(yt[idx], pred, zero_division=0))
        store["specificity"].append(recall_score(yt[idx], pred, pos_label=0, zero_division=0))
    a = (1 - ci) / 2
    return {f"{k}_ci": (float(np.quantile(v, a)), float(np.quantile(v, 1 - a)))
            for k, v in store.items() if v}


# ── DeLong test (Sun & Xu 2014 fast implementation) ─────────────────────────
def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2


def _fast_delong(preds_sorted_transposed, label_1_count):
    m = label_1_count
    n = preds_sorted_transposed.shape[1] - m
    positive = preds_sorted_transposed[:, :m]
    negative = preds_sorted_transposed[:, m:]
    k = preds_sorted_transposed.shape[0]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r] = _compute_midrank(positive[r])
        ty[r] = _compute_midrank(negative[r])
        tz[r] = _compute_midrank(preds_sorted_transposed[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01); sy = np.cov(v10)
    return aucs, sx / m + sy / n


def delong_roc_test(y_true, p1, p2) -> dict:
    """Two-sided DeLong test that AUC(p1) == AUC(p2) on the SAME samples."""
    y = np.asarray(y_true).astype(int)
    order = np.argsort(-y)
    label_1_count = int(y.sum())
    if label_1_count == 0 or label_1_count == len(y):
        return {"auc1": np.nan, "auc2": np.nan, "delta": np.nan, "z": np.nan, "p": np.nan}
    preds = np.vstack([np.asarray(p1, float), np.asarray(p2, float)])[:, order]
    aucs, cov = _fast_delong(preds, label_1_count)
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return {"auc1": float(aucs[0]), "auc2": float(aucs[1]), "delta": float(diff), "z": 0.0, "p": 1.0}
    z = diff / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"auc1": float(aucs[0]), "auc2": float(aucs[1]), "delta": float(diff),
            "z": float(z), "p": float(p)}


def paired_bootstrap_auc_diff(y_true, p1, p2, n_boot: int = 2000, seed: int = 42) -> dict:
    """Bootstrap CI for AUC(p1) − AUC(p2) on paired samples."""
    rng = np.random.RandomState(seed)
    y = np.asarray(y_true).astype(int); a = np.asarray(p1, float); b = np.asarray(p2, float)
    if len(np.unique(y)) < 2:
        return {"delta": np.nan, "ci": (np.nan, np.nan), "p_boot": np.nan}
    diffs = []
    for _ in range(n_boot):
        idx = rng.choice(len(y), len(y), replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    diffs = np.array(diffs)
    return {"delta": float(roc_auc_score(y, a) - roc_auc_score(y, b)),
            "ci": (float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))),
            "p_boot": float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))}
