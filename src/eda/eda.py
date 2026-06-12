"""Exploratory Data Analysis for the Netflix Prize ratings.

Produces the figures and summary statistics required by Task A:

* user activity patterns,
* content popularity trends,
* rating distribution,
* data sparsity characteristics,
* temporal trends,

and writes a machine-readable ``eda_summary.json`` plus PNG figures.
"""
from __future__ import annotations

import os
from typing import Dict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless backend for servers / CI
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
except ImportError:  # seaborn is optional
    sns = None

from ..utils import LOG, save_json


def _savefig(fig, figures_dir: str, name: str) -> None:
    os.makedirs(figures_dir, exist_ok=True)
    path = os.path.join(figures_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    LOG.info("Saved figure -> %s", path)


def run_eda(df: pd.DataFrame, titles: pd.DataFrame, config: Dict) -> Dict:
    """Compute summary stats + figures. Returns the summary dict."""
    figures_dir = config["paths"]["figures_dir"]
    outputs_dir = config["paths"]["outputs_dir"]

    n_ratings = len(df)
    n_users = df["userId"].nunique()
    n_movies = df["movieId"].nunique()
    if n_users == 0 or n_movies == 0:
        raise ValueError(
            "No ratings left after filtering. Lower min_ratings_per_user / "
            "min_ratings_per_movie in config.yaml or use a denser dataset."
        )
    sparsity = 1.0 - n_ratings / (n_users * n_movies)

    user_counts = df["userId"].value_counts()
    movie_counts = df["movieId"].value_counts()
    rating_dist = df["rating"].value_counts().sort_index()

    summary = {
        "n_ratings": int(n_ratings),
        "n_users": int(n_users),
        "n_movies": int(n_movies),
        "sparsity": float(sparsity),
        "density_pct": float(100 * (1 - sparsity)),
        "global_mean_rating": float(df["rating"].mean()),
        "rating_distribution": {int(k): int(v) for k, v in rating_dist.items()},
        "ratings_per_user": {
            "min": int(user_counts.min()),
            "median": float(user_counts.median()),
            "mean": float(user_counts.mean()),
            "max": int(user_counts.max()),
        },
        "ratings_per_movie": {
            "min": int(movie_counts.min()),
            "median": float(movie_counts.median()),
            "mean": float(movie_counts.mean()),
            "max": int(movie_counts.max()),
        },
    }

    # --- Figure 1: rating distribution ------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(rating_dist.index.astype(str), rating_dist.values, color="#E50914")
    ax.set_title("Rating distribution")
    ax.set_xlabel("Rating (stars)")
    ax.set_ylabel("Count")
    _savefig(fig, figures_dir, "01_rating_distribution.png")

    # --- Figure 2: ratings-per-user (log-log popularity tail) -------------
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(user_counts.values, bins=60, color="#564d4d")
    ax.set_yscale("log")
    ax.set_title("User activity (ratings per user)")
    ax.set_xlabel("# ratings by a user")
    ax.set_ylabel("# users (log)")
    _savefig(fig, figures_dir, "02_user_activity.png")

    # --- Figure 3: movie popularity --------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(movie_counts.values, bins=60, color="#831010")
    ax.set_yscale("log")
    ax.set_title("Content popularity (ratings per movie)")
    ax.set_xlabel("# ratings for a movie")
    ax.set_ylabel("# movies (log)")
    _savefig(fig, figures_dir, "03_movie_popularity.png")

    # --- Figure 4: ratings over time -------------------------------------
    if "date" in df.columns:
        by_month = (
            df.set_index("date").resample("ME")["rating"].count()
            if hasattr(pd.Series(), "resample") else None
        )
        try:
            ts = df.set_index("date").resample("M")["rating"].count()
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(ts.index, ts.values, color="#E50914")
            ax.set_title("Ratings volume over time")
            ax.set_xlabel("Month")
            ax.set_ylabel("# ratings")
            _savefig(fig, figures_dir, "04_ratings_over_time.png")
            summary["first_rating_date"] = str(df["date"].min().date())
            summary["last_rating_date"] = str(df["date"].max().date())
        except Exception as exc:  # pragma: no cover
            LOG.warning("Temporal plot skipped: %s", exc)

    # --- Top / bottom movies by popularity & mean rating -----------------
    if not titles.empty:
        title_map = dict(zip(titles["movieId"], titles["title"]))
        stats = (
            df.groupby("movieId")["rating"]
            .agg(["count", "mean"]).rename(columns={"count": "n", "mean": "avg"})
        )
        stats["title"] = stats.index.map(title_map)
        popular = stats.sort_values("n", ascending=False).head(10)
        # Best-rated among reasonably popular movies (>= 90th pct support).
        thresh = stats["n"].quantile(0.9)
        best = stats[stats["n"] >= thresh].sort_values("avg", ascending=False).head(10)
        summary["top_10_most_rated"] = [
            {"movieId": int(i), "title": r["title"], "n": int(r["n"]), "avg": round(float(r["avg"]), 3)}
            for i, r in popular.iterrows()
        ]
        summary["top_10_best_rated_popular"] = [
            {"movieId": int(i), "title": r["title"], "n": int(r["n"]), "avg": round(float(r["avg"]), 3)}
            for i, r in best.iterrows()
        ]

    save_json(summary, os.path.join(outputs_dir, "eda_summary.json"))
    LOG.info(
        "EDA done | sparsity=%.5f%% | mean=%.3f",
        100 * sparsity, summary["global_mean_rating"],
    )
    return summary
