"""Item-based collaborative filtering (pure NumPy, no SciPy dependency).

We build a dense *item x user* matrix of mean-centred residuals (rating minus
the regularised bias baseline) and compute item-item cosine similarity::

    sim(i, j) = (r_i . r_j) / (||r_i|| * ||r_j||)

Similarities are damped by a shrinkage term based on the number of users who
co-rated both items, then pruned to the top-k neighbours per item. Predictions
blend the bias baseline with a similarity-weighted neighbourhood term::

    r_hat(u, i) = mu + b_u + b_i
                  + sum_j sim(i, j) * resid(u, j) / sum_j |sim(i, j)|

Item-based CF is the classic Netflix choice: movies are fewer and far more
stable than users, so the item-item similarity matrix is smaller and the
neighbourhoods are more reliable. Because it materialises dense matrices it is
meant for the *sub-sampled* dataset configured in ``config.yaml`` (which the
brief explicitly allows).
"""
from __future__ import annotations

import numpy as np

from .base import BaseRecommender
from .baseline import BaselineModel
from ..utils import LOG


class ItemBasedCF(BaseRecommender):
    name = "item_cf"

    def fit(self, dataset) -> "ItemBasedCF":
        cfg = self.config["models"]["item_cf"]
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

        # Dense residual (R) and rating-mask (M) matrices: items x users.
        self.R = np.zeros((self.n_items, self.n_users), dtype=np.float32)
        self.M = np.zeros((self.n_items, self.n_users), dtype=np.float32)
        self.R[i, u] = resid
        self.M[i, u] = 1.0

        norms = np.sqrt((self.R * self.R).sum(axis=1))
        norms[norms == 0] = 1e-9
        self.norms = norms

        # Item-item cosine similarity with co-support shrinkage.
        sim = (self.R @ self.R.T) / np.outer(norms, norms)
        support = self.M @ self.M.T               # # users co-rating (i, j)
        sim *= support / (support + self.shrinkage)
        np.fill_diagonal(sim, 0.0)
        sim[support < self.min_support] = 0.0

        # Keep only the top-k neighbours per item (sparsify rows).
        if self.k < self.n_items:
            for it in range(self.n_items):
                row = sim[it]
                nz = np.count_nonzero(row)
                if nz > self.k:
                    drop = np.argpartition(-np.abs(row), self.k)[self.k:]
                    row[drop] = 0.0
        self.sim = sim.astype(np.float32)
        LOG.info("ItemCF fit: %d items, similarity matrix %s",
                 self.n_items, self.sim.shape)
        return self

    def _base_all(self, u: int) -> np.ndarray:
        b = self.base
        return b.global_mean + b.b_u[u] + b.b_i

    def predict(self, u_idx: np.ndarray, i_idx: np.ndarray,
                batch: int = 4096) -> np.ndarray:
        """Vectorised, batched rating prediction for paired arrays."""
        u_idx = np.asarray(u_idx, np.int64)
        i_idx = np.asarray(i_idx, np.int64)
        preds = np.empty(len(u_idx), dtype=np.float64)
        base = self.base
        for s in range(0, len(u_idx), batch):
            us = u_idx[s:s + batch]
            is_ = i_idx[s:s + batch]
            sims = self.sim[is_]                  # (B, n_items)
            user_resid = self.R[:, us].T          # (B, n_items)
            mask = self.M[:, us].T                # (B, n_items)
            w = sims                              # neighbour weights
            num = np.einsum("bn,bn->b", w, user_resid)
            den = np.einsum("bn,bn->b", np.abs(w), mask)
            b = base.global_mean + base.b_u[us] + base.b_i[is_]
            safe = den > 1e-8
            out = b.astype(np.float64)
            out[safe] = b[safe] + num[safe] / den[safe]
            preds[s:s + batch] = out
        return self._clip(preds)

    def scores_for_user(self, u: int) -> np.ndarray:
        ur = self.R[:, u]                         # residuals (0 where unrated)
        rated = self.M[:, u]
        num = self.sim @ ur
        den = np.abs(self.sim) @ rated
        den[den < 1e-8] = 1e-8
        return self._clip(self._base_all(u) + num / den)

    # bonus: nearest items for the "identify similarities" requirement
    def similar_items(self, i: int, top: int = 10) -> np.ndarray:
        sims = self.sim[i].copy()
        return np.argsort(-sims)[:top].astype(np.int32)
