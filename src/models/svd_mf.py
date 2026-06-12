"""Biased matrix factorization trained with mini-batch SGD ("Funk SVD").

This is the model family that won the Netflix Prize. We factor the rating
matrix into latent user/item factors plus bias terms::

    r_hat(u, i) = mu + b_u + b_i + p_u . q_i

Parameters are learned by minimising regularised squared error::

    min  sum (r_ui - r_hat)^2 + reg * (||p_u||^2 + ||q_i||^2 + b_u^2 + b_i^2)

We use *mini-batch* gradient descent (vectorised with ``np.add.at`` scatter
updates) so a few-million-rating subset trains in seconds without a Python
per-sample loop. Latent factors also give us free user/item similarity for the
"identify similarities" requirement.
"""
from __future__ import annotations

import numpy as np

from .base import BaseRecommender
from ..utils import LOG


class SVDModel(BaseRecommender):
    name = "svd"

    def fit(self, dataset) -> "SVDModel":
        cfg = self.config["models"]["svd"]
        self.n_factors = int(cfg["n_factors"])
        self.n_epochs = int(cfg["n_epochs"])
        self.lr = float(cfg["lr"])
        self.reg = float(cfg["reg"])
        self.batch_size = int(cfg.get("batch_size", 100_000))
        seed = self.config.get("seed", 42)

        self.n_users = dataset.n_users
        self.n_items = dataset.n_items
        self.global_mean = dataset.global_mean

        u, i, r = dataset.train_arrays()
        n = len(r)
        rng = np.random.default_rng(seed)

        scale = 0.1
        self.P = rng.normal(0, scale, (self.n_users, self.n_factors)).astype(np.float32)
        self.Q = rng.normal(0, scale, (self.n_items, self.n_factors)).astype(np.float32)
        self.b_u = np.zeros(self.n_users, dtype=np.float32)
        self.b_i = np.zeros(self.n_items, dtype=np.float32)

        for epoch in range(self.n_epochs):
            perm = rng.permutation(n)
            sq_err = 0.0
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                bu, bi, br = u[idx], i[idx], r[idx]
                pu, qi = self.P[bu], self.Q[bi]
                pred = (self.global_mean + self.b_u[bu] + self.b_i[bi]
                        + np.einsum("ij,ij->i", pu, qi))
                err = br - pred
                sq_err += float(np.dot(err, err))

                # Gradients (note: err already carries the sign).
                e = err[:, None]
                grad_p = e * qi - self.reg * pu
                grad_q = e * pu - self.reg * qi

                # Scatter-add updates (handles repeated users/items in a batch).
                np.add.at(self.P, bu, self.lr * grad_p)
                np.add.at(self.Q, bi, self.lr * grad_q)
                np.add.at(self.b_u, bu, self.lr * (err - self.reg * self.b_u[bu]))
                np.add.at(self.b_i, bi, self.lr * (err - self.reg * self.b_i[bi]))

            rmse = np.sqrt(sq_err / n)
            LOG.info("SVD epoch %2d/%d | train RMSE=%.4f", epoch + 1, self.n_epochs, rmse)
        return self

    def predict(self, u_idx: np.ndarray, i_idx: np.ndarray) -> np.ndarray:
        dot = np.einsum("ij,ij->i", self.P[u_idx], self.Q[i_idx])
        preds = self.global_mean + self.b_u[u_idx] + self.b_i[i_idx] + dot
        return self._clip(preds)

    def scores_for_user(self, u: int) -> np.ndarray:
        dot = self.Q @ self.P[u]
        return self._clip(self.global_mean + self.b_u[u] + self.b_i + dot)

    # -- bonus: latent-factor similarity for the "find similar items" task --
    def similar_items(self, i: int, top: int = 10) -> np.ndarray:
        q = self.Q[i]
        norms = np.linalg.norm(self.Q, axis=1) * (np.linalg.norm(q) + 1e-9)
        sims = (self.Q @ q) / (norms + 1e-9)
        sims[i] = -np.inf
        return np.argsort(-sims)[:top].astype(np.int32)
