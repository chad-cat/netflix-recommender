"""CLI: evaluate trained models on RMSE + MAP@10 (and optional metrics).

Usage:
    python -m scripts.evaluate --config config.yaml --models all \
        --max-eval-users 3000

Produces ``outputs/metrics.json`` and a printed comparison table.
"""
from __future__ import annotations

import argparse
import os
import pickle

import numpy as np

from src.utils import load_config, set_seed, ensure_dirs, LOG, save_json, Timer
from src.models import MODEL_REGISTRY
from src.evaluation import evaluate_rating, evaluate_ranking
from src.recommend import build_seen_map, build_relevant_map, generate_topk_for_users


def _load(artifacts_dir, name):
    with open(os.path.join(artifacts_dir, f"model_{name}.pkl"), "rb") as fh:
        return pickle.load(fh)


def evaluate_models(config, model_names, max_eval_users=None):
    artifacts_dir = config["paths"]["artifacts_dir"]
    with open(os.path.join(artifacts_dir, "dataset.pkl"), "rb") as fh:
        dataset = pickle.load(fh)

    k = config["evaluation"]["k"]
    thresh = config["evaluation"]["relevance_threshold"]

    seen_map = build_seen_map(dataset)
    relevant_map = build_relevant_map(dataset, thresh)

    # Users we score for ranking = those with >=1 relevant held-out item.
    eval_users = np.array(sorted(relevant_map.keys()), dtype=np.int32)
    rng = np.random.default_rng(config.get("seed", 42))
    if max_eval_users and len(eval_users) > max_eval_users:
        eval_users = rng.choice(eval_users, size=max_eval_users, replace=False)
        eval_users.sort()
    LOG.info("Ranking evaluation on %d users (relevance>=%.1f, K=%d)",
             len(eval_users), thresh, k)

    u_test, i_test, r_test = dataset.test_arrays()
    results = {}
    for name in model_names:
        if name not in MODEL_REGISTRY:
            continue
        model = _load(artifacts_dir, name)
        LOG.info("=== Evaluating %s ===", name)

        # --- rating accuracy (RMSE / MAE) on all test interactions ---
        preds = model.predict(u_test, i_test)
        rating_metrics = evaluate_rating(r_test, preds)

        # --- ranking (MAP@K etc.) over the candidate catalogue ---
        with Timer(f"rank[{name}]"):
            recs = generate_topk_for_users(model, eval_users, k, seen_map)
        rec_lists = [recs[int(u)] for u in eval_users]
        rel_lists = [relevant_map[int(u)] for u in eval_users]
        ranking_metrics = evaluate_ranking(rec_lists, rel_lists, k, dataset.n_items)

        results[name] = {**rating_metrics, **ranking_metrics}
        LOG.info("%s -> RMSE=%.4f  MAP@%d=%.4f",
                 name, rating_metrics["rmse"], k, ranking_metrics[f"map@{k}"])

    save_json(results, os.path.join(config["paths"]["outputs_dir"], "metrics.json"))
    _print_table(results, k)
    return results


def _print_table(results, k):
    if not results:
        return
    cols = ["rmse", "mae", f"map@{k}", f"precision@{k}",
            f"recall@{k}", f"ndcg@{k}", f"hit_rate@{k}", "catalogue_coverage"]
    header = "model".ljust(12) + "".join(c.rjust(14) for c in cols)
    print("\n" + header)
    print("-" * len(header))
    for name, m in results.items():
        row = name.ljust(12) + "".join(f"{m.get(c, 0):14.4f}" for c in cols)
        print(row)
    print()


def main():
    ap = argparse.ArgumentParser(description="Evaluate recommendation models.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--models", nargs="+", default=["all"])
    ap.add_argument("--max-eval-users", type=int, default=3000)
    args = ap.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    ensure_dirs(config)
    names = list(MODEL_REGISTRY) if "all" in args.models else args.models
    evaluate_models(config, names, args.max_eval_users)


if __name__ == "__main__":
    main()
