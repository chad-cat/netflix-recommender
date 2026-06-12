"""Generate a tiny *synthetic* Netflix-format dataset for smoke-testing.

This lets you run the full pipeline end-to-end without downloading the 2 GB
Kaggle data. It writes ``combined_data_1.txt`` and ``movie_titles.csv`` into
the raw dir using the exact Netflix Prize format, with latent-factor structure
so the models have a real signal to learn.

Usage:
    python -m src.make_synthetic --raw-dir data/raw --users 800 --movies 300
"""
from __future__ import annotations

import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--users", type=int, default=800)
    ap.add_argument("--movies", type=int, default=300)
    ap.add_argument("--density", type=float, default=0.18)
    ap.add_argument("--factors", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.raw_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Latent structure so ratings are predictable.
    P = rng.normal(0, 1, (args.users, args.factors))
    Q = rng.normal(0, 1, (args.movies, args.factors))
    bu = rng.normal(0, 0.5, args.users)
    bi = rng.normal(0, 0.5, args.movies)
    mu = 3.6

    # Group ratings by movie to match the combined_data format.
    by_movie = {m: [] for m in range(args.movies)}
    base_date = np.datetime64("2004-01-01")
    for u in range(args.users):
        n = rng.binomial(args.movies, args.density)
        movies = rng.choice(args.movies, size=max(n, 5), replace=False)
        for m in movies:
            raw = mu + bu[u] + bi[m] + P[u] @ Q[m]
            rating = int(np.clip(round(raw + rng.normal(0, 0.4)), 1, 5))
            day = int(rng.integers(0, 700))
            date = (base_date + np.timedelta64(day, "D")).astype(str)
            by_movie[m].append((u + 1, rating, date))  # +1 -> 1-based ids

    path = os.path.join(args.raw_dir, "combined_data_1.txt")
    with open(path, "w", encoding="utf-8") as fh:
        for m in range(args.movies):
            rows = by_movie[m]
            if not rows:
                continue
            fh.write(f"{m + 1}:\n")
            for uid, rating, date in rows:
                fh.write(f"{uid},{rating},{date}\n")

    titles_path = os.path.join(args.raw_dir, "movie_titles.csv")
    with open(titles_path, "w", encoding="latin-1") as fh:
        for m in range(args.movies):
            year = int(rng.integers(1990, 2005))
            fh.write(f"{m + 1},{year},Synthetic Movie {m + 1}\n")

    print(f"Wrote synthetic dataset -> {path}")
    print(f"Wrote titles            -> {titles_path}")


if __name__ == "__main__":
    main()
