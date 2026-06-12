"""Evaluation metrics for both rating prediction and ranking.

Mandatory metrics required by the brief:
* RMSE   - rating-prediction accuracy
* MAP@10 - ranking quality, relevance = true rating >= 3.5

Plus optional metrics: MAE, Precision@K, Recall@K, NDCG@K, Hit-Rate, Coverage.

--- Ranking protocol (documented for the report) -------------------------------
For each test user we:
  1. take the items they rated in the *test* split,
  2. mark an item relevant iff its true rating >= ``relevance_threshold`` (3.5),
  3. ask the model for its Top-K items (excluding items already seen in TRAIN),
  4. score the ranking of those Top-K items against the relevant set.
This is an "all-unrated-items" ranking protocol: the candidate pool is every
movie the user has not rated in train, which is the realistic recommendation
setting (the model must surface the held-out liked movies out of ~thousands).
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# Rating-prediction metrics                                                   #
# --------------------------------------------------------------------------- #
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


# --------------------------------------------------------------------------- #
# Ranking metrics (per-user, then averaged)                                   #
# --------------------------------------------------------------------------- #
def average_precision_at_k(recommended: Sequence[int],
                           relevant: set, k: int = 10) -> float:
    """Average Precision @ K for a single user.

    AP@K = (1/min(|relevant|, K)) * sum_{r=1..K} P(r) * rel(r)
    where P(r) is precision at cut-off r and rel(r) indicates a hit at rank r.
    """
    if not relevant:
        return 0.0
    hits = 0
    score = 0.0
    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            hits += 1
            score += hits / rank
    return score / min(len(relevant), k)


def map_at_k(recommended_per_user: List[Sequence[int]],
             relevant_per_user: List[set], k: int = 10) -> float:
    """Mean Average Precision @ K across all evaluated users."""
    aps = [
        average_precision_at_k(rec, rel, k)
        for rec, rel in zip(recommended_per_user, relevant_per_user)
        if rel  # only score users who have at least one relevant held-out item
    ]
    return float(np.mean(aps)) if aps else 0.0


def precision_recall_at_k(recommended: Sequence[int],
                          relevant: set, k: int = 10):
    if not relevant:
        return 0.0, 0.0
    topk = list(recommended[:k])
    hits = sum(1 for it in topk if it in relevant)
    precision = hits / k
    recall = hits / len(relevant)
    return precision, recall


def ndcg_at_k(recommended: Sequence[int], relevant: set, k: int = 10) -> float:
    """Binary-relevance NDCG@K."""
    if not relevant:
        return 0.0
    dcg = 0.0
    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            dcg += 1.0 / np.log2(rank + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(r + 1) for r in range(1, ideal_hits + 1))
    return float(dcg / idcg) if idcg > 0 else 0.0


def hit_rate_at_k(recommended_per_user: List[Sequence[int]],
                  relevant_per_user: List[set], k: int = 10) -> float:
    """Fraction of users with at least one relevant item in their Top-K."""
    hits = 0
    total = 0
    for rec, rel in zip(recommended_per_user, relevant_per_user):
        if not rel:
            continue
        total += 1
        if any(it in rel for it in rec[:k]):
            hits += 1
    return hits / total if total else 0.0


# --------------------------------------------------------------------------- #
# Aggregators used by scripts                                                 #
# --------------------------------------------------------------------------- #
def evaluate_rating(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {"rmse": rmse(y_true, y_pred), "mae": mae(y_true, y_pred)}


def evaluate_ranking(recommended_per_user: List[Sequence[int]],
                     relevant_per_user: List[set],
                     k: int = 10,
                     n_items: int = 0) -> Dict[str, float]:
    precisions, recalls, ndcgs = [], [], []
    catalogue = set()
    for rec, rel in zip(recommended_per_user, relevant_per_user):
        catalogue.update(rec[:k])
        if not rel:
            continue
        p, r = precision_recall_at_k(rec, rel, k)
        precisions.append(p)
        recalls.append(r)
        ndcgs.append(ndcg_at_k(rec, rel, k))
    coverage = (len(catalogue) / n_items) if n_items else 0.0
    return {
        f"map@{k}": map_at_k(recommended_per_user, relevant_per_user, k),
        f"precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"hit_rate@{k}": hit_rate_at_k(recommended_per_user, relevant_per_user, k),
        "catalogue_coverage": coverage,
    }
