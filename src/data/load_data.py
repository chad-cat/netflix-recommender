"""Parse the raw Netflix Prize files into a tidy ratings DataFrame.

The Netflix Prize release ships four files (``combined_data_1.txt`` ...
``combined_data_4.txt``) using a compact custom format::

    1:                      <- movie id followed by a colon
    1488844,3,2005-09-06    <- userId,rating,date
    822109,5,2005-05-13
    ...
    2:
    2059652,4,2005-09-05
    ...

Movie metadata lives in ``movie_titles.csv`` (Latin-1 encoded)::

    1,2003,Dinosaur Planet
    2,2004,Isle of Man TT 2004 Review

This module turns those files into a single DataFrame with columns
``[userId, movieId, rating, date]`` and caches the result as parquet so the
(slow) text parsing only happens once.
"""
from __future__ import annotations

import os
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from ..utils import LOG


def _parse_combined_file(path: str) -> pd.DataFrame:
    """Stream a single combined_data_*.txt file into a DataFrame.

    We do a single pass over the file. Lines ending with ``:`` switch the
    current movie id; every other line is a ``user,rating,date`` triple.
    """
    user_ids: List[int] = []
    movie_ids: List[int] = []
    ratings: List[np.int8] = []
    dates: List[str] = []

    current_movie = -1
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.endswith(":"):
                current_movie = int(line[:-1])
                continue
            uid, rating, date = line.split(",")
            user_ids.append(int(uid))
            movie_ids.append(current_movie)
            ratings.append(int(rating))
            dates.append(date)

    df = pd.DataFrame(
        {
            "userId": np.asarray(user_ids, dtype=np.int32),
            "movieId": np.asarray(movie_ids, dtype=np.int32),
            "rating": np.asarray(ratings, dtype=np.int8),
            "date": pd.to_datetime(dates),
        }
    )
    LOG.info("Parsed %s -> %d ratings", os.path.basename(path), len(df))
    return df


def load_ratings(raw_dir: str, combined_files: Iterable[int]) -> pd.DataFrame:
    """Load and concatenate the requested combined_data files."""
    frames = []
    for idx in combined_files:
        path = os.path.join(raw_dir, f"combined_data_{idx}.txt")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {path}. Download the Netflix Prize data from Kaggle "
                "and place the combined_data_*.txt files in the raw dir."
            )
        frames.append(_parse_combined_file(path))
    df = pd.concat(frames, ignore_index=True)
    return df


def load_movie_titles(raw_dir: str) -> pd.DataFrame:
    """Load movie_titles.csv -> DataFrame[movieId, year, title].

    Titles can contain commas, so we cap the split at two separators.
    """
    path = os.path.join(raw_dir, "movie_titles.csv")
    if not os.path.exists(path):
        LOG.warning("movie_titles.csv not found in %s; titles unavailable.", raw_dir)
        return pd.DataFrame(columns=["movieId", "year", "title"])

    rows = []
    with open(path, "r", encoding="latin-1") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(",", 2)
            if len(parts) < 3:
                continue
            movie_id, year, title = parts
            year_val = int(year) if year.isdigit() else None
            rows.append((int(movie_id), year_val, title))
    titles = pd.DataFrame(rows, columns=["movieId", "year", "title"])
    LOG.info("Loaded %d movie titles", len(titles))
    return titles


def build_or_load_dataset(config: Dict) -> pd.DataFrame:
    """Return the ratings DataFrame, using a parquet cache when possible."""
    paths = config["paths"]
    data_cfg = config["data"]
    os.makedirs(paths["processed_dir"], exist_ok=True)

    tag = "-".join(str(i) for i in data_cfg["combined_files"])
    cache = os.path.join(paths["processed_dir"], f"ratings_{tag}.parquet")

    if os.path.exists(cache):
        LOG.info("Loading cached ratings from %s", cache)
        try:
            return pd.read_parquet(cache)
        except Exception as exc:  # parquet engine may be missing
            LOG.warning("Could not read parquet cache (%s); re-parsing.", exc)

    df = load_ratings(paths["raw_dir"], data_cfg["combined_files"])
    try:
        df.to_parquet(cache, index=False)
        LOG.info("Cached ratings -> %s", cache)
    except Exception as exc:  # pragma: no cover
        LOG.warning("Could not write parquet cache (%s).", exc)
    return df
