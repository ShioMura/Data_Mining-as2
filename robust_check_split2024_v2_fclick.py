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
result_path = OUTPUT_DIR / "robust_check_split2024_v2_fclick_results.csv"

print("Train path exists:", train_path.exists())


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
# 2. New validation split
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
# 3. Features
# ============================================================

drop_cols = [
    "srch_id",
    "prop_id",
    "relevance",
    "click_rank_label",
]

feature_cols = [c for c in train_fe.columns if c not in drop_cols]

print("Number of features:", len(feature_cols))

train_part = train_part.sort_values("srch_id").reset_index(drop=True)
val_part = val_part.sort_values("srch_id").reset_index(drop=True)

X_train = (
    train_part[feature_cols]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(-999)
)

X_val = (
    val_part[feature_cols]
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
# 4. Train normal v2 ranker
# ============================================================

ranker = lgb.LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    n_estimators=1200,
    learning_rate=0.03,
    num_leaves=95,
    min_child_samples=100,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=42,
    n_jobs=-1,
)

print("=" * 70)
print("Training v2 normal ranker on split 2024")
print("=" * 70)

ranker.fit(
    X_train,
    y_rank_train,
    group=train_group,
    eval_set=[(X_val, y_rank_val)],
    eval_group=[val_group],
    eval_at=[5],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=50)
    ]
)

ranker_iter = ranker.best_iteration_
if ranker_iter is None or ranker_iter <= 0:
    ranker_iter = 780

val_part["v2_score"] = ranker.predict(X_val, num_iteration=ranker_iter)

temp = val_part[["srch_id", "prop_id", "relevance", "v2_score"]].copy()
temp = temp.rename(columns={"v2_score": "score"})
v2_ndcg = mean_ndcg_at_5(temp)

print("v2 best iteration:", ranker_iter)
print("v2 split2024 NDCG@5:", v2_ndcg)


# ============================================================
# 5. Train F click ranker
# ============================================================

f_ranker = lgb.LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    n_estimators=1200,
    learning_rate=0.03,
    num_leaves=95,
    min_child_samples=100,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=42,
    n_jobs=-1,
)

print("=" * 70)
print("Training F click ranker on split 2024")
print("=" * 70)

f_ranker.fit(
    X_train,
    y_click_train,
    group=train_group,
    eval_set=[(X_val, y_click_val)],
    eval_group=[val_group],
    eval_at=[5],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=50)
    ]
)

f_iter = f_ranker.best_iteration_
if f_iter is None or f_iter <= 0:
    f_iter = 592

val_part["f_click_score"] = f_ranker.predict(X_val, num_iteration=f_iter)

temp = val_part[["srch_id", "prop_id", "relevance", "f_click_score"]].copy()
temp = temp.rename(columns={"f_click_score": "score"})
f_ndcg = mean_ndcg_at_5(temp)

print("F click best iteration:", f_iter)
print("F click split2024 original-relevance NDCG@5:", f_ndcg)


# ============================================================
# 6. Blend v2 + F click
# ============================================================

val_part["v2_score_norm"] = normalize_within_search(val_part, "v2_score")
val_part["f_click_score_norm"] = normalize_within_search(val_part, "f_click_score")

results = []

for w_v2 in np.arange(0.970, 1.0001, 0.005):
    w_f = 1.0 - w_v2

    val_part["score"] = (
        w_v2 * val_part["v2_score"]
        + w_f * val_part["f_click_score"]
    )

    raw_ndcg = mean_ndcg_at_5(val_part)

    val_part["score"] = (
        w_v2 * val_part["v2_score_norm"]
        + w_f * val_part["f_click_score_norm"]
    )

    norm_ndcg = mean_ndcg_at_5(val_part)

    print(
        "w_v2:", round(w_v2, 3),
        "w_f:", round(w_f, 3),
        "raw:", raw_ndcg,
        "norm:", norm_ndcg
    )

    results.append({
        "w_v2": w_v2,
        "w_f": w_f,
        "raw_ndcg5": raw_ndcg,
        "norm_ndcg5": norm_ndcg,
    })


results_df = pd.DataFrame(results)

print("=" * 70)
print("Robust split2024 results")
print("v2 single:", v2_ndcg)
print("F click:", f_ndcg)
print("Top raw blend:")
print(results_df.sort_values("raw_ndcg5", ascending=False).head(10))
print("Top norm blend:")
print(results_df.sort_values("norm_ndcg5", ascending=False).head(10))
print("=" * 70)

results_df.to_csv(result_path, index=False)
print("Saved results to:", result_path)