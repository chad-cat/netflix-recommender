"""Regularised bias baseline.

Prediction:  r_hat(u, i) = mu + b_u + b_i

where the user/item biases are estimated with damped means (the standard
Netflix-Prize baseline). It is fast, has no hyper-parameter search, and gives
a strong RMSE reference that every fancier model must beat.
"""
from __future__ import annotations

import numpy as np

from .base import BaseRecommender
from ..utils import LOG


class BaselineModel(BaseRecommender):
    name = "baseline"

    def fit(self, dataset) -> "BaselineModel":
        cfg = self.config["models"]["baseline"]
        reg_u = float(cfg["reg_user"])
        reg_i = float(cfg["reg_item"])

        self.n_users = dataset.n_users
        self.n_items = dataset.n_items
        self.global_mean = dataset.global_mean
        u, i, r = dataset.train_arrays()
        dev = r - self.global_mean

        # Item bias first (damped mean of deviations from global mean).
        item_sum = np.bincount(i, weights=dev, minlength=self.n_items)
        item_cnt = np.bincount(i, minlength=self.n_items)
        self.b_i = item_sum / (reg_i + item_cnt)

        # User bias on residuals after removing the item bias.
        resid = dev - self.b_i[i]
        user_sum = np.bincount(u, weights=resid, minlength=self.n_users)
        user_cnt = np.bincount(u, minlength=self.n_users)
        self.b_u = user_sum / (reg_u + user_cnt)

        LOG.info("Baseline fit: mu=%.4f", self.global_mean)
        return self

    def predict(self, u_idx: np.ndarray, i_idx: np.ndarray) -> np.ndarray:
        preds = self.global_mean + self.b_u[u_idx] + self.b_i[i_idx]
        return self._clip(preds)

    def scores_for_user(self, u: int) -> np.ndarray:
        return self._clip(self.global_mean + self.b_u[u] + self.b_i)
