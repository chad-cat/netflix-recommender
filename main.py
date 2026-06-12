"""End-to-end pipeline runner: EDA -> train -> evaluate -> recommend.

This is the single entry point graders can run to reproduce everything::

    python main.py --config config.yaml --models all

Use ``--synthetic`` to first generate a tiny synthetic dataset so the whole
pipeline runs in seconds without the 2 GB Kaggle download (great for a smoke
test / CI).
"""
from __future__ import annotations

import argparse
import os
import runpy
import sys

from src.utils import load_config, set_seed, ensure_dirs, LOG
from src.data import build_or_load_dataset, filter_dataset, train_test_split
from src.data.load_data import load_movie_titles
from src.eda import run_eda
from src.models import MODEL_REGISTRY
from scripts.train import train_models
from scripts.evaluate import evaluate_models


def maybe_make_synthetic(config, users, movies):
    raw_dir = config["paths"]["raw_dir"]
    if os.path.exists(os.path.join(raw_dir, "combined_data_1.txt")):
        LOG.info("Raw data already present; skipping synthetic generation.")
        return
    LOG.info("Generating synthetic dataset (users=%d, movies=%d)", users, movies)
    sys.argv = ["make_synthetic", "--raw-dir", raw_dir,
                "--users", str(users), "--movies", str(movies)]
    runpy.run_module("src.make_synthetic", run_name="__main__")


def main():
    ap = argparse.ArgumentParser(description="Run the full recommendation pipeline.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--models", nargs="+", default=["all"])
    ap.add_argument("--synthetic", action="store_true",
                    help="Generate a small synthetic dataset if raw data is absent.")
    ap.add_argument("--synth-users", type=int, default=800)
    ap.add_argument("--synth-movies", type=int, default=300)
    ap.add_argument("--max-eval-users", type=int, default=3000)
    ap.add_argument("--skip-eda", action="store_true")
    args = ap.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    ensure_dirs(config)

    if args.synthetic:
        maybe_make_synthetic(config, args.synth_users, args.synth_movies)

    # 1) Load + filter
    df = build_or_load_dataset(config)
    df = filter_dataset(df, config)
    titles = load_movie_titles(config["paths"]["raw_dir"])

    # 2) EDA
    if not args.skip_eda:
        LOG.info("########## STEP 1/4: EDA ##########")
        run_eda(df, titles, config)

    # 3) Split + train
    LOG.info("########## STEP 2/4: TRAIN ##########")
    dataset = train_test_split(df, config)
    names = list(MODEL_REGISTRY) if "all" in args.models else args.models
    train_models(config, names, dataset=dataset)

    # 4) Evaluate
    LOG.info("########## STEP 3/4: EVALUATE ##########")
    evaluate_models(config, names, max_eval_users=args.max_eval_users)

    # 5) Recommend (best traditional + MF) for inspection
    LOG.info("########## STEP 4/4: RECOMMEND ##########")
    rec_model = "svd" if "svd" in names else names[0]
    sys.argv = ["recommend", "--config", args.config, "--model", rec_model]
    runpy.run_module("scripts.recommend", run_name="__main__")

    LOG.info("Pipeline complete. See outputs/ and figures/.")


if __name__ == "__main__":
    main()
