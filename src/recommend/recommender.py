"""Top-K recommendation generation + helpers for ranking evaluation.

The key contract used everywhere:
* ``seen`` items (rated in TRAIN) are excluded from recommendations,
* ``relevant`` items are the TEST items with rating >= threshold,
* recommendations are produced over the full catalogue of unseen items.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..utils import LOG


def build_seen_map(dataset) -> Dict[int, np.ndarray]:
    """user_idx -> array of item_idx the user rated in TRAIN."""
    grp = dataset.train.groupby("u_idx")["i_idx"].apply(lambda s: s.to_numpy(np.int32))
    return grp.to_dict()


def build_relevant_map(dataset, threshold: float) -> Dict[int, set]:
    """user_idx -> set of item_idx rated >= threshold in TEST (the hits)."""
    test = dataset.test
    rel = test[test["rating"] >= threshold]
    return rel.groupby("u_idx")["i_idx"].apply(set).to_dict()


def generate_topk_for_users(
    model,
    users: Sequence[int],
    k: int,
    seen_map: Dict[int, np.ndarray],
) -> Dict[int, np.ndarray]:
    """Return {user_idx: top-k item_idx array} excluding seen items."""
    recs: Dict[int, np.ndarray] = {}
    for n, u in enumerate(users):
        seen = seen_map.get(int(u))
        recs[int(u)] = model.recommend(int(u), k=k, seen=seen)
        if (n + 1) % 2000 == 0:
            LOG.info("  generated recs for %d/%d users", n + 1, len(users))
    return recs


def recommendations_to_frame(
    recs: Dict[int, np.ndarray],
    dataset,
    titles: Optional[pd.DataFrame] = None,
    scores_model=None,
) -> pd.DataFrame:
    """Flatten recommendations into a tidy, human-readable DataFrame."""
    title_map = {}
    year_map = {}
    if titles is not None and not titles.empty:
        title_map = dict(zip(titles["movieId"], titles["title"]))
        year_map = dict(zip(titles["movieId"], titles["year"]))

    rows: List[dict] = []
    for u_idx, items in recs.items():
        orig_user = dataset.inv_user_map[u_idx]
        scores = None
        if scores_model is not None:
            user_arr = np.full(len(items), u_idx, dtype=np.int32)
            scores = scores_model.predict(user_arr, np.asarray(items, np.int32))
        for rank, it in enumerate(items, start=1):
            movie_id = dataset.inv_item_map[int(it)]
            rows.append({
                "userId": orig_user,
                "rank": rank,
                "movieId": movie_id,
                "title": title_map.get(movie_id, ""),
                "year": year_map.get(movie_id, ""),
                "pred_rating": round(float(scores[rank - 1]), 3) if scores is not None else "",
            })
    return pd.DataFrame(rows)
