"""User-based collaborative filtering (pure NumPy, no SciPy dependency).

Symmetric counterpart to item-based CF: cosine similarity is computed between
users on their mean-centred residual vectors, and a target rating is predicted
from the k most similar users who rated the item::

    r_hat(u, i) = mu + b_u + b_i
                  + sum_v sim(u, v) * resid(v, i) / sum_v |sim(u, v)|

Unlike the item model we do NOT materialise the full n_users x n_users matrix
(it would be huge); instead each user's neighbour similarities are computed on
demand with a single dense mat-vec. User-based CF is included mainly for the
Task-C comparison: it is typically costlier and less stable than item-based CF
on Netflix-scale data because users are numerous and their taste drifts.
"""
from __future__ import annotations

import numpy as np

from .base import BaseRecommender
from .baseline import BaselineModel
from ..utils import LOG


class UserBasedCF(BaseRecommender):
    name = "user_cf"

    def fit(self, dataset) -> "UserBasedCF":
        cfg = self.config["models"]["user_cf"]
        self.k = int(cfg["k_neighbors"])
        self.shrinkage = float(cfg["shrinkage"])
        self.min_support = int(cfg.get("min_support", 1))

        self.n_users = dataset.n_users
        self.n_items = dataset.n_items
        self.global_mean = dataset.global_mean
        self.base = BaselineModel(self.config).fit(dataset)

        u, i, r = dataset.train_arrays()
        base_pred = self.base.global_mean + self.base.b_u[u] + self.base.b_i[i]
        resid = (r - base_pred).astype(np.float32)

        # Dense residual (R) and rating-mask (M): users x items.
        self.R = np.zeros((self.n_users, self.n_items), dtype=np.float32)
        self.M = np.zeros((self.n_users, self.n_items), dtype=np.float32)
        self.R[u, i] = resid
        self.M[u, i] = 1.0

        norms = np.sqrt((self.R * self.R).sum(axis=1))
        norms[norms == 0] = 1e-9
        self.norms = norms
        LOG.info("UserCF fit: %d users x %d items.", self.n_users, self.n_items)
        return self

    def _neighbour_sims(self, u: int) -> np.ndarray:
        """Cosine similarity of user u to all users, shrinkage + top-k pruned."""
        sims = (self.R @ self.R[u]) / (self.norms * self.norms[u])
        support = self.M @ self.M[u]              # # items co-rated
        sims = sims * (support / (support + self.shrinkage))
        sims[support < self.min_support] = 0.0
        sims[u] = 0.0
        np.nan_to_num(sims, copy=False)
        if self.k < sims.size:
            keep = np.argpartition(-np.abs(sims), self.k)[: self.k]
            pruned = np.zeros_like(sims)
            pruned[keep] = sims[keep]
            sims = pruned
        return sims

    def _base_all(self, u: int) -> np.ndarray:
        b = self.base
        return b.global_mean + b.b_u[u] + b.b_i

    def scores_for_user(self, u: int) -> np.ndarray:
        sims = self._neighbour_sims(int(u))
        num = sims @ self.R                        # (n_items,)
        den = np.abs(sims) @ self.M                # (n_items,)
        den[den < 1e-8] = 1e-8
        return self._clip(self._base_all(u) + num / den)

    def predict(self, u_idx: np.ndarray, i_idx: np.ndarray) -> np.ndarray:
        """Predict by computing each unique user's item scores once."""
        u_idx = np.asarray(u_idx, np.int64)
        i_idx = np.asarray(i_idx, np.int64)
        preds = np.empty(len(u_idx), dtype=np.float64)
        for u in np.unique(u_idx):
            sel = u_idx == u
            scores = self.scores_for_user(int(u))
            preds[sel] = scores[i_idx[sel]]
        return preds  # already clipped in scores_for_user
