#!/usr/bin/env python3
"""
utils.py — Shared utilities for OOD detection benchmark.

Covers: reproducibility, metrics, score orientation, safe string names.
"""

import random
from typing import Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve


# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────

def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────
# Score orientation
# ─────────────────────────────────────────────

def ensure_ood_higher(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Flip signs if OOD mean < ID mean so that higher score ⟹ more OOD."""
    if float(np.mean(ood_scores)) < float(np.mean(id_scores)):
        return -id_scores, -ood_scores
    return id_scores, ood_scores


def orient_scores_higher_is_ood(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Same as ensure_ood_higher but also returns a `flipped` flag.
    Useful when callers need to know whether the sign was inverted.
    """
    flipped = float(ood_scores.mean()) < float(id_scores.mean())
    if flipped:
        return -id_scores, -ood_scores, True
    return id_scores, ood_scores, False


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def fpr_at_tpr(
    y_true: np.ndarray,
    scores: np.ndarray,
    tpr_level: float = 0.95,
) -> float:
    """Minimum FPR over all thresholds where TPR ≥ tpr_level."""
    fpr, tpr, _ = roc_curve(y_true, scores, pos_label=1)
    mask = tpr >= tpr_level
    if not np.any(mask):
        return float("nan")
    return float(np.min(fpr[mask]))


def compute_metrics(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
) -> Tuple[float, float]:
    """
    Returns (AUROC, FPR@95TPR).
    Assumes higher score ⟹ more OOD (call ensure_ood_higher first if needed).
    """
    scores = np.concatenate([id_scores, ood_scores])
    y_true = np.concatenate([
        np.zeros(len(id_scores), dtype=np.int64),
        np.ones(len(ood_scores), dtype=np.int64),
    ])
    auroc = float(roc_auc_score(y_true, scores))
    fpr95 = fpr_at_tpr(y_true, scores, tpr_level=0.95)
    return auroc, fpr95


def report_metrics(
    method_name: str,
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
) -> Tuple[float, float]:
    """
    Print a formatted metrics block and return (auroc, fpr95).

    Scores are automatically oriented so that higher ⟹ more OOD before
    computing metrics, so callers do not need to do this manually.
    """
    # Coerce to numpy float64
    id_scores  = np.asarray(id_scores,  dtype=np.float64)
    ood_scores = np.asarray(ood_scores, dtype=np.float64)

    id_scores, ood_scores = ensure_ood_higher(id_scores, ood_scores)
    auroc, fpr95 = compute_metrics(id_scores, ood_scores)

    print(f"\n========== {method_name.upper()} ==========")
    print(f"AUROC (OOD positive):      {auroc:.4f}")
    print(f"FPR@95%TPR (OOD positive): {fpr95:.4f}")
    print(f"(score means) ID  mean={id_scores.mean():.6f}  std={id_scores.std():.6f}")
    print(f"(score means) OOD mean={ood_scores.mean():.6f}  std={ood_scores.std():.6f}")
    print("=" * 42)

    return auroc, fpr95


# ─────────────────────────────────────────────
# Misc helpers
# ─────────────────────────────────────────────

def safe_name(s: str) -> str:
    """Replace non-alphanumeric / non-safe characters with underscores."""
    return "".join(c if (c.isalnum() or c in "_-.") else "_" for c in s)
