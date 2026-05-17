from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score

import lightgbm as lgb


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

train_path = DATA_DIR / "train_features_lean_extra_v2_best.parquet"

# top70 feature list from previous importance file
importance_path = OUTPUT_DIR / "model_b_feature_importance.csv"

result_path = OUTPUT_DIR / "robust_check_split2024_v2_top70_fclick_results.csv"

print("Train path exists:", train_path.exists())
print("Importance path exists:", importance_path.exists())


def mean_ndcg_at_5(df, label_col="relevance", score_col="score"):
    scores = []

    for _, group in df.groupby("srch_id"):
        y_true = group[label_col].to_numpy()
        y_score = group[score_col].to_numpy()

        score = ndcg_score(
            y_true.reshape(1, -1),
            y_score.reshape(1, -1),
            k=5
        )

        scores.append(score)

    return float(np.mean(scores))


def normalize_within_search(df, score_col):
    g = df.groupby("srch_id")[score_col]
    mean = g.transform("mean")
    std = g.transform("std").replace(0, 1)
    return ((df[score_col] - mean) / std).fillna(0)


# ============================================================
# 1. Load features
# ============================================================

train_fe = pd.read_parquet(train_path)
print("Train features:", train_fe.shape)

train_fe["click_rank_label"] = (train_fe["relevance"] >= 1).astype("int8")


# ============================================================
# 2. split 2024
# ============================================================

unique_srch_ids = train_fe["srch_id"].unique()

train_ids, val_ids = train_test_split(
    unique_srch_ids,
    test_size=0.2,
    random_state=2024
)

train_part = train_fe[train_fe["srch_id"].isin(train_ids)].copy()
val_part = train_fe[train_fe["srch_id"].isin(val_ids)].copy()

print("Train part:", train_part.shape)
print("Validation part:", val_part.shape)


# ============================================================
# 3. Feature columns
# ============================================================

drop_cols = [
    "srch_id",
    "prop_id",
    "relevance",
    "click_rank_label",
]

feature_cols = [c for c in train_fe.columns if c not in drop_cols]

print("Number of all features:", len(feature_cols))


# Try to get top70 from feature importance
if importance_path.exists():
    importance_df = pd.read_csv(importance_path)
    top70_features = (
        importance_df["feature"]
        .head(70)
        .tolist()
    )
    top70_features = [c for c in top70_features if c in feature_cols]
else:
    # fallback: use all features if importance file missing
    top70_features = feature_cols

print("Number of top70 features:", len(top70_features))


# ============================================================
# 4. Sort and prepare matrices
# ============================================================

train_part = train_part.sort_values("srch_id").reset_index(drop=True)
val_part = val_part.sort_values("srch_id").reset_index(drop=True)

X_train_all = (
    train_part[feature_cols]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(-999)
)

X_val_all = (
    val_part[feature_cols]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(-999)
)

X_train_top70 = (
    train_part[top70_features]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(-999)
)

X_val_top70 = (
    val_part[top70_features]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(-999)
)

y_rank_train = train_part["relevance"]
y_rank_val = val_part["relevance"]

y_click_train = train_part["click_rank_label"]
y_click_val = val_part["click_rank_label"]

train_group = train_part.groupby("srch_id").size().to_numpy()
val_group = val_part.groupby("srch_id").size().to_numpy()

print("Train group rows check:", train_group.sum(), len(train_part))
print("Val group rows check:", val_group.sum(), len(val_part))


# ============================================================
# 5. Train helper
# ============================================================

def train_ranker(X_train, y_train, X_val, y_val, seed, name):
    print("=" * 70)
    print("Training", name, "seed =", seed)
    print("=" * 70)

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=95,
        min_child_samples=100,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=seed,
        n_jobs=-1,
    )

    ranker.fit(
        X_train,
        y_train,
        group=train_group,
        eval_set=[(X_val, y_val)],
        eval_group=[val_group],
        eval_at=[5],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50),
        ],
    )

    best_iter = ranker.best_iteration_
    if best_iter is None or best_iter <= 0:
        best_iter = 700

    preds = ranker.predict(X_val, num_iteration=best_iter)

    return preds, best_iter


# ============================================================
# 6. Train models
# ============================================================

val_part["v2_seed42"], iter_v2_42 = train_ranker(
    X_train_all,
    y_rank_train,
    X_val_all,
    y_rank_val,
    seed=42,
    name="v2 all-features ranker"
)

val_part["v2_seed3407"], iter_v2_3407 = train_ranker(
    X_train_all,
    y_rank_train,
    X_val_all,
    y_rank_val,
    seed=3407,
    name="v2 all-features ranker"
)

val_part["top70_seed42"], iter_top70_42 = train_ranker(
    X_train_top70,
    y_rank_train,
    X_val_top70,
    y_rank_val,
    seed=42,
    name="top70 ranker"
)

val_part["f_click_seed42"], iter_f_42 = train_ranker(
    X_train_all,
    y_click_train,
    X_val_all,
    y_click_val,
    seed=42,
    name="F click ranker"
)


# ============================================================
# 7. Individual scores
# ============================================================

individual_cols = [
    "v2_seed42",
    "v2_seed3407",
    "top70_seed42",
    "f_click_seed42",
]

print("=" * 70)
print("Individual model NDCG@5 on split 2024")
print("=" * 70)

for col in individual_cols:
    val_part["score"] = val_part[col]
    ndcg = mean_ndcg_at_5(val_part)
    print(col, ":", ndcg)


# ============================================================
# 8. Build candidate ensembles
# ============================================================

val_part["v2_seed_avg"] = (
    0.60 * val_part["v2_seed42"]
    + 0.40 * val_part["v2_seed3407"]
)

val_part["v2_top70_blend"] = (
    0.92 * val_part["v2_seed_avg"]
    + 0.08 * val_part["top70_seed42"]
)

val_part["v2_top70_blend_norm"] = (
    0.92 * normalize_within_search(val_part, "v2_seed_avg")
    + 0.08 * normalize_within_search(val_part, "top70_seed42")
)


# ============================================================
# 9. Search F-click blend weights
# ============================================================

results = []

base_candidates = [
    "v2_seed42",
    "v2_seed_avg",
    "v2_top70_blend",
    "v2_top70_blend_norm",
]

for base_col in base_candidates:
    for w_base in np.arange(0.960, 1.0001, 0.005):
        w_f = 1.0 - w_base

        val_part["score"] = (
            w_base * val_part[base_col]
            + w_f * val_part["f_click_seed42"]
        )

        raw_ndcg = mean_ndcg_at_5(val_part)

        val_part["score"] = (
            w_base * normalize_within_search(val_part, base_col)
            + w_f * normalize_within_search(val_part, "f_click_seed42")
        )

        norm_ndcg = mean_ndcg_at_5(val_part)

        print(
            "base:", base_col,
            "w_base:", round(w_base, 3),
            "w_f:", round(w_f, 3),
            "raw:", raw_ndcg,
            "norm:", norm_ndcg,
        )

        results.append({
            "base_model": base_col,
            "w_base": w_base,
            "w_f": w_f,
            "raw_ndcg5": raw_ndcg,
            "norm_ndcg5": norm_ndcg,
        })


results_df = pd.DataFrame(results)

print("=" * 70)
print("Robust split2024 v2/top70/F results")
print("Iterations:")
print("v2_seed42:", iter_v2_42)
print("v2_seed3407:", iter_v2_3407)
print("top70_seed42:", iter_top70_42)
print("f_click_seed42:", iter_f_42)

print("Top raw:")
print(results_df.sort_values("raw_ndcg5", ascending=False).head(20))

print("Top norm:")
print(results_df.sort_values("norm_ndcg5", ascending=False).head(20))
print("=" * 70)

results_df.to_csv(result_path, index=False)
print("Saved results to:", result_path)