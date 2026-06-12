"""CLI: run Exploratory Data Analysis and dump figures + summary.

Usage:
    python -m scripts.run_eda --config config.yaml
"""
from __future__ import annotations

import argparse

from src.utils import load_config, set_seed, ensure_dirs, LOG
from src.data import build_or_load_dataset, filter_dataset
from src.data.load_data import load_movie_titles
from src.eda import run_eda


def main():
    ap = argparse.ArgumentParser(description="Run EDA on the Netflix Prize data.")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    ensure_dirs(config)

    df = build_or_load_dataset(config)
    df = filter_dataset(df, config)
    titles = load_movie_titles(config["paths"]["raw_dir"])
    run_eda(df, titles, config)
    LOG.info("EDA complete. See %s and %s.",
             config["paths"]["figures_dir"], config["paths"]["outputs_dir"])


if __name__ == "__main__":
    main()
