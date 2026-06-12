"""CLI: train one or more models and cache them to the artifacts dir.

Usage:
    python -m scripts.train --config config.yaml --models baseline svd
    python -m scripts.train --models all
"""
from __future__ import annotations

import argparse
import os
import pickle

from src.utils import load_config, set_seed, ensure_dirs, LOG, Timer
from src.data import build_or_load_dataset, filter_dataset, train_test_split
from src.models import MODEL_REGISTRY


def build_dataset(config):
    df = build_or_load_dataset(config)
    df = filter_dataset(df, config)
    return train_test_split(df, config)


def train_models(config, model_names, dataset=None):
    if dataset is None:
        dataset = build_dataset(config)
    artifacts_dir = config["paths"]["artifacts_dir"]
    os.makedirs(artifacts_dir, exist_ok=True)

    # Persist the dataset split so evaluate/recommend reuse the exact same one.
    with open(os.path.join(artifacts_dir, "dataset.pkl"), "wb") as fh:
        pickle.dump(dataset, fh)

    trained = {}
    for name in model_names:
        if name not in MODEL_REGISTRY:
            LOG.warning("Unknown model '%s' (skipping).", name)
            continue
        LOG.info("=== Training %s ===", name)
        model = MODEL_REGISTRY[name](config)
        with Timer(f"train[{name}]") as t:
            model.fit(dataset)
        with open(os.path.join(artifacts_dir, f"model_{name}.pkl"), "wb") as fh:
            pickle.dump(model, fh)
        trained[name] = {"model": model, "train_seconds": t.seconds}
    return dataset, trained


def main():
    ap = argparse.ArgumentParser(description="Train recommendation models.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--models", nargs="+", default=["all"],
                    help="baseline item_cf user_cf svd | all")
    args = ap.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    ensure_dirs(config)

    names = list(MODEL_REGISTRY) if "all" in args.models else args.models
    train_models(config, names)
    LOG.info("Training complete -> %s", config["paths"]["artifacts_dir"])


if __name__ == "__main__":
    main()
