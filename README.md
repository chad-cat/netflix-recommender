# 🎬 Netflix Prize — Personalized Content Discovery

A clean, reproducible recommendation-system codebase built on the **Netflix
Prize Dataset**. It learns user preferences, predicts unseen ratings, generates
Top‑K personalized recommendations, and reports the mandatory **RMSE** and
**MAP@10** metrics (plus optional ranking metrics).

The repository implements and compares **four** recommendation approaches:

| Model | Family | Idea |
|-------|--------|------|
| `baseline` | Bias model | `mu + b_u + b_i` (regularised damped means) — strong RMSE reference |
| `item_cf`  | Neighbourhood CF | Item–item cosine similarity on mean‑centred residuals |
| `user_cf`  | Neighbourhood CF | User–user cosine similarity on mean‑centred residuals |
| `svd`      | Latent factor | Biased matrix factorization ("Funk SVD") trained with mini‑batch SGD |

---

## 1. Repository structure

```
netflix-recommender/
├── main.py                  # one-command end-to-end pipeline
├── config.yaml              # all knobs live here (reproducibility)
├── requirements.txt
├── data/
│   ├── raw/                 # put combined_data_*.txt + movie_titles.csv here
│   └── processed/           # parquet cache (auto-generated)
├── src/
│   ├── data/                # parsing, filtering, train/test split
│   │   ├── load_data.py
│   │   └── preprocess.py
│   ├── eda/                 # Task A: exploratory data analysis + figures
│   │   └── eda.py
│   ├── models/              # Task B/C: the four recommenders
│   │   ├── base.py
│   │   ├── baseline.py
│   │   ├── item_cf.py
│   │   ├── user_cf.py
│   │   └── svd_mf.py
│   ├── evaluation/          # Task E: RMSE, MAP@10, P/R, NDCG, coverage
│   │   └── metrics.py
│   ├── recommend/           # Task D: Top-K generation + case analysis
│   │   └── recommender.py
│   ├── make_synthetic.py    # tiny synthetic dataset for smoke tests
│   └── utils.py
└── scripts/                 # individual CLI stages
    ├── run_eda.py
    ├── train.py
    ├── evaluate.py
    └── recommend.py
```

---

## 2. Setup

```bash
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

Python 3.9+ recommended. Core deps: numpy, pandas, scipy, scikit‑learn,
matplotlib, seaborn, PyYAML, tqdm.

---

## 3. Get the data

Download the dataset from Kaggle:
<https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data>

Place these files in `data/raw/`:

```
data/raw/combined_data_1.txt
data/raw/combined_data_2.txt   # optional
data/raw/combined_data_3.txt   # optional
data/raw/combined_data_4.txt   # optional
data/raw/movie_titles.csv
```

`config.yaml` controls how much data is loaded (`combined_files`) and how it is
sub‑sampled (`min_ratings_per_user`, `min_ratings_per_movie`, `max_users`) so
the pipeline runs comfortably on a laptop, as the brief allows.

### No download? Run the synthetic smoke test

You can verify everything works without the 2 GB download — a small synthetic
dataset in the exact Netflix format is generated for you:

```bash
python main.py --synthetic --models all
```

---

## 4. Reproduce all results (one command)

```bash
python main.py --config config.yaml --models all
```

This runs the full pipeline: **EDA → train → evaluate → recommend** and writes:

```
figures/   01_rating_distribution.png, 02_user_activity.png, ...
outputs/   eda_summary.json, metrics.json,
           recommendations_svd.csv, case_analysis_svd.json
artifacts/ dataset.pkl, model_*.pkl   (cached split + trained models)
```

### Run stages individually

```bash
python -m scripts.run_eda    --config config.yaml
python -m scripts.train      --config config.yaml --models baseline item_cf user_cf svd
python -m scripts.evaluate   --config config.yaml --models all --max-eval-users 3000
python -m scripts.recommend  --config config.yaml --model svd --k 10 --n-users 20
```

---

## 5. Methodology (how the mandatory metrics are computed)

**Train/test split** — *per-user temporal hold-out* (`split.strategy: temporal`).
For every user, their most recent `test_quota` (default 20%) ratings go to the
test set; the rest is training. This mimics a real deployment (train on the
past, predict the future) and prevents look‑ahead leakage. Each user keeps at
least `min_train_per_user` ratings in train. A `random` per-user hold-out is
also available.

**Relevance definition** — a test item is *relevant* iff its true rating is
**≥ 3.5** (configurable via `evaluation.relevance_threshold`).

**Top-10 generation** — for each evaluated user we score **every catalogue item
not rated in train**, mask the seen items, and take the 10 highest-scoring
items (`argpartition` for speed). This "all-unrated-items" protocol is the
realistic recommendation setting.

**RMSE** — computed over all held-out test interactions:
`sqrt(mean((r - r_hat)^2))`.

**MAP@10** — mean over users of Average Precision@10, where
`AP@K = (1/min(|relevant|,K)) * Σ_r P(r)·rel(r)`. Only users with at least one
relevant held-out item are scored.

Optional metrics reported alongside: **MAE, Precision@10, Recall@10, NDCG@10,
Hit-Rate@10, Catalogue Coverage**.

---

## 6. Model design notes & trade-offs

* **Baseline** — closed-form, milliseconds to fit, surprisingly strong RMSE.
  Establishes the bias terms reused by the CF models.
* **Item-based CF** — preferred neighbourhood method for Netflix-scale data
  (fewer, more stable items than users). Shrinkage damps similarities built on
  little co-rating support.
* **User-based CF** — included for the Task-C comparison; generally costlier and
  noisier than item-based CF here.
* **SVD (matrix factorization)** — the Netflix-Prize-winning family. Learns
  dense latent factors via mini-batch SGD; usually the best RMSE **and**
  ranking, and its latent factors give free item/user similarity
  (`SVDModel.similar_items`).

**RMSE vs ranking trade-off:** a model can have great RMSE yet mediocre MAP@10
(accurate on observed ratings ≠ good at surfacing the few liked items out of
thousands). We therefore report both and discuss them together — exactly what
the brief asks for.

---

## 7. Mapping to the deliverables

| Task | Where |
|------|-------|
| A. EDA | `src/eda/eda.py`, `outputs/eda_summary.json`, `figures/` |
| B. Model development | `src/models/*` |
| C. Model comparison | `scripts/evaluate.py` → `outputs/metrics.json` table |
| D. Top-K + case analysis | `scripts/recommend.py` → `recommendations_*.csv`, `case_analysis_*.json` |
| E. Evaluation (RMSE, MAP@10) | `src/evaluation/metrics.py` |

Use the JSON/CSV/PNG outputs to populate the **Technical Report (PDF, ≤10 pp)**
and **Presentation (PDF, ≤8 slides)**.

---

## 8. Reproducibility

* Single global `seed` in `config.yaml` seeds NumPy/Python.
* The exact train/test split is cached to `artifacts/dataset.pkl` and reused by
  every evaluation/recommendation step.
* All hyper-parameters live in `config.yaml`; no magic numbers in code.

---

## 9. Future improvements

* Alternating Least Squares (ALS) and implicit-feedback models.
* Neural Collaborative Filtering / two-tower models.
* Hybrid (content + collaborative) to mitigate cold-start.
* Diversity / novelty-aware re-ranking.
* Time-aware factors (rating date already parsed and available).

## License

Released for educational use with the Netflix Prize dataset. See dataset terms
on Kaggle.
