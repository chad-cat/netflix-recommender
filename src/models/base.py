"""Common interface shared by every recommender.

A model must implement:
* ``fit(dataset)``                          -> learn parameters from train data
* ``predict(u_idx, i_idx)``                 -> vectorised rating predictions
* ``recommend(u_idx, k, seen)``             -> Top-K item indices for one user

The default ``recommend`` simply scores every item with ``predict`` and takes
the top K after masking already-seen items, which is correct for all models
here; subclasses may override it for speed.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np


class BaseRecommender:
    name: str = "base"

    def __init__(self, config: dict):
        self.config = config
        self.n_users: int = 0
        self.n_items: int = 0
        self.global_mean: float = 0.0
        self._all_items: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ fit
    def fit(self, dataset) -> "BaseRecommender":
        raise NotImplementedError

    # -------------------------------------------------------------- predict
    def predict(self, u_idx: np.ndarray, i_idx: np.ndarray) -> np.ndarray:
        """Return predicted ratings for paired (u_idx, i_idx) arrays."""
        raise NotImplementedError

    def _clip(self, preds: np.ndarray) -> np.ndarray:
        return np.clip(preds, 1.0, 5.0)

    # ------------------------------------------------------------ recommend
    def scores_for_user(self, u: int) -> np.ndarray:
        """Score every item for a single user index. Override for speed."""
        items = np.arange(self.n_items, dtype=np.int32)
        users = np.full(self.n_items, u, dtype=np.int32)
        return self.predict(users, items)

    def recommend(self, u: int, k: int = 10,
                  seen: Optional[Iterable[int]] = None) -> np.ndarray:
        """Return the Top-K item indices for user ``u`` (excluding ``seen``)."""
        scores = self.scores_for_user(u).astype(np.float64)
        if seen is not None:
            seen = np.fromiter(seen, dtype=np.int64)
            if seen.size:
                scores[seen] = -np.inf
        if k >= scores.size:
            order = np.argsort(-scores)
        else:
            # argpartition is O(n) vs full sort O(n log n)
            part = np.argpartition(-scores, k)[:k]
            order = part[np.argsort(-scores[part])]
        return order.astype(np.int32)
