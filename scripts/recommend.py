"""CLI: generate Top-K recommendations and a success/failure case analysis.

Usage:
    python -m scripts.recommend --config config.yaml --model svd \
        --k 10 --n-users 20

Writes:
    outputs/recommendations_<model>.csv   (tidy Top-K per sampled user)
    outputs/case_analysis_<model>.json     (success + failure examples)
"""
from __future__ import annotations

import argparse
import os
import pickle

import numpy as np

from src.utils import load_config, set_seed, ensure_dirs, LOG, save_json
from src.data.load_data import load_movie_titles
from src.recommend import (
    build_seen_map, build_relevant_map,
    generate_topk_for_users, recommendations_to_frame,
)
from src.evaluation import average_precision_at_k


def main():
    ap = argparse.ArgumentParser(description="Generate Top-K recommendations.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", default="svd")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--n-users", type=int, default=20)
    args = ap.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    ensure_dirs(config)

    k = args.k or config["evaluation"]["k"]
    thresh = config["evaluation"]["relevance_threshold"]
    artifacts_dir = config["paths"]["artifacts_dir"]
    outputs_dir = config["paths"]["outputs_dir"]

    with open(os.path.join(artifacts_dir, "dataset.pkl"), "rb") as fh:
        dataset = pickle.load(fh)
    with open(os.path.join(artifacts_dir, f"model_{args.model}.pkl"), "rb") as fh:
        model = pickle.load(fh)
    titles = load_movie_titles(config["paths"]["raw_dir"])

    seen_map = build_seen_map(dataset)
    relevant_map = build_relevant_map(dataset, thresh)

    # Sample users who have held-out relevant items so we can judge quality.
    candidate_users = np.array(sorted(relevant_map.keys()), dtype=np.int32)
    rng = np.random.default_rng(config.get("seed", 42))
    sample = rng.choice(candidate_users,
                        size=min(args.n_users, len(candidate_users)),
                        replace=False)
    sample.sort()

    recs = generate_topk_for_users(model, sample, k, seen_map)
    frame = recommendations_to_frame(recs, dataset, titles, scores_model=model)
    csv_path = os.path.join(outputs_dir, f"recommendations_{args.model}.csv")
    frame.to_csv(csv_path, index=False)
    LOG.info("Wrote %s", csv_path)

    # --- success / failure case analysis -------------------------------
    title_map = dict(zip(titles["movieId"], titles["title"])) if not titles.empty else {}
    per_user = []
    for u in sample:
        rel = relevant_map[int(u)]
        rec_items = recs[int(u)]
        ap = average_precision_at_k(rec_items, rel, k)
        hits = [int(it) for it in rec_items if it in rel]
        per_user.append({
            "userId": int(dataset.inv_user_map[int(u)]),
            "ap@k": round(ap, 4),
            "n_relevant_heldout": len(rel),
            "n_hits_in_topk": len(hits),
            "hit_titles": [title_map.get(int(dataset.inv_item_map[h]), "") for h in hits],
        })

    per_user.sort(key=lambda d: d["ap@k"], reverse=True)
    analysis = {
        "model": args.model,
        "k": k,
        "relevance_threshold": thresh,
        "success_cases": per_user[:5],
        "failure_cases": [p for p in per_user if p["n_hits_in_topk"] == 0][:5],
        "mean_ap@k_sample": round(float(np.mean([p["ap@k"] for p in per_user])), 4),
    }
    save_json(analysis, os.path.join(outputs_dir, f"case_analysis_{args.model}.json"))
    LOG.info("Done. Sample mean AP@%d=%.4f", k, analysis["mean_ap@k_sample"])


if __name__ == "__main__":
    main()
