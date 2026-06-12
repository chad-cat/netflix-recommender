"""Filtering, id-remapping and train/test splitting.

The ``Dataset`` object is the single source of truth passed to every model and
evaluator. It stores:

* contiguous integer ids (``u_idx`` / ``i_idx``) for fast matrix ops,
* the train and test interaction frames,
* lookup maps back to the original Netflix user/movie ids.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd

from ..utils import LOG


def filter_dataset(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Drop sparse users/movies and optionally cap the number of users.

    Filtering is applied iteratively because removing low-activity users can
    push some movies below the movie threshold and vice-versa.
    """
    data_cfg = config["data"]
    min_u = data_cfg["min_ratings_per_user"]
    min_i = data_cfg["min_ratings_per_movie"]
    max_users = data_cfg.get("max_users")

    prev = -1
    while len(df) != prev:
        prev = len(df)
        uc = df["userId"].value_counts()
        keep_u = uc[uc >= min_u].index
        df = df[df["userId"].isin(keep_u)]
        ic = df["movieId"].value_counts()
        keep_i = ic[ic >= min_i].index
        df = df[df["movieId"].isin(keep_i)]

    if max_users is not None:
        # Keep the `max_users` most active users (deterministic given the data).
        top_users = df["userId"].value_counts().head(int(max_users)).index
        df = df[df["userId"].isin(top_users)]
        # Re-apply the movie threshold after the user cap.
        ic = df["movieId"].value_counts()
        df = df[df["movieId"].isin(ic[ic >= min_i].index)]

    df = df.reset_index(drop=True)
    LOG.info(
        "After filtering: %d ratings | %d users | %d movies",
        len(df), df["userId"].nunique(), df["movieId"].nunique(),
    )
    return df


@dataclass
class Dataset:
    train: pd.DataFrame
    test: pd.DataFrame
    user_map: Dict[int, int]
    item_map: Dict[int, int]
    global_mean: float
    n_users: int
    n_items: int
    inv_user_map: Dict[int, int] = field(default_factory=dict)
    inv_item_map: Dict[int, int] = field(default_factory=dict)

    def __post_init__(self):
        self.inv_user_map = {v: k for k, v in self.user_map.items()}
        self.inv_item_map = {v: k for k, v in self.item_map.items()}

    # -- convenience views -------------------------------------------------
    def train_arrays(self):
        return (
            self.train["u_idx"].to_numpy(np.int32),
            self.train["i_idx"].to_numpy(np.int32),
            self.train["rating"].to_numpy(np.float32),
        )

    def test_arrays(self):
        return (
            self.test["u_idx"].to_numpy(np.int32),
            self.test["i_idx"].to_numpy(np.int32),
            self.test["rating"].to_numpy(np.float32),
        )


def _remap_ids(df: pd.DataFrame):
    users = np.sort(df["userId"].unique())
    items = np.sort(df["movieId"].unique())
    user_map = {int(u): i for i, u in enumerate(users)}
    item_map = {int(m): i for i, m in enumerate(items)}
    df = df.copy()
    df["u_idx"] = df["userId"].map(user_map).astype(np.int32)
    df["i_idx"] = df["movieId"].map(item_map).astype(np.int32)
    return df, user_map, item_map


def train_test_split(df: pd.DataFrame, config: Dict) -> Dataset:
    """Per-user hold-out split.

    *temporal* (default): for each user the most recent ``test_quota`` fraction
    of their ratings is held out. This mirrors a real deployment where we train
    on the past and predict the future, and avoids look-ahead leakage.

    *random*: a per-user random hold-out of the same size.

    Users always keep at least ``min_train_per_user`` ratings in train.
    """
    split_cfg = config["split"]
    seed = config.get("seed", 42)
    quota = split_cfg["test_quota"]
    min_train = split_cfg["min_train_per_user"]
    strategy = split_cfg["strategy"]

    df, user_map, item_map = _remap_ids(df)

    rng = np.random.default_rng(seed)
    train_parts, test_parts = [], []

    for _, grp in df.groupby("u_idx", sort=False):
        n = len(grp)
        n_test = int(np.floor(n * quota))
        # Guarantee enough training history.
        n_test = min(n_test, max(0, n - min_train))
        if n_test <= 0:
            train_parts.append(grp)
            continue
        if strategy == "temporal":
            grp_sorted = grp.sort_values("date")
            train_parts.append(grp_sorted.iloc[:-n_test])
            test_parts.append(grp_sorted.iloc[-n_test:])
        else:  # random
            perm = rng.permutation(n)
            test_idx = perm[:n_test]
            mask = np.zeros(n, dtype=bool)
            mask[test_idx] = True
            test_parts.append(grp.iloc[mask])
            train_parts.append(grp.iloc[~mask])

    train = pd.concat(train_parts, ignore_index=True)
    test = (
        pd.concat(test_parts, ignore_index=True)
        if test_parts else df.iloc[0:0].copy()
    )

    global_mean = float(train["rating"].mean())
    LOG.info(
        "Split (%s): train=%d test=%d | users=%d items=%d | global_mean=%.4f",
        strategy, len(train), len(test), len(user_map), len(item_map), global_mean,
    )
    return Dataset(
        train=train,
        test=test,
        user_map=user_map,
        item_map=item_map,
        global_mean=global_mean,
        n_users=len(user_map),
        n_items=len(item_map),
    )
