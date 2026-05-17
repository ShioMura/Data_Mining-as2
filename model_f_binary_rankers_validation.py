from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score

import lightgbm as lgb


# ============================================================
# 0. Paths
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

train_path = DATA_DIR / "train_features_lean_extra_v2_best.parquet"

val_score_path = OUTPUT_DIR / "model_f_val_scores_binary_rankers.csv"
result_path = OUTPUT_DIR / "model_f_binary_ranker_weight_results.csv"

print("Train path exists:", train_path.exists())


# ============================================================
# 1. Load v2 features
# ============================================================

train_fe = pd.read_parquet(train_path)

print("Train features:", train_fe.shape)


# ============================================================
# 2. Create binary rank labels
# ============================================================

train_fe["booking_rank_label"] = (train_fe["relevance"] == 5).astype("int8")
train_fe["click_rank_label"] = (train_fe["relevance"] >= 1).astype("int8")

print("Booking label positive rate:", train_fe["booking_rank_label"].mean())
print("Click label positive rate:", train_fe["click_rank_label"].mean())


# ============================================================
# 3. Split by srch_id
# ============================================================

unique_srch_ids = train_fe["srch_id"].unique()

train_ids, val_ids = train_test_split(
    unique_srch_ids,
    test_size=0.2,
    random_state=42
)

train_part = train_fe[train_fe["srch_id"].isin(train_ids)].copy()
val_part = train_fe[train_fe["srch_id"].isin(val_ids)].copy()

print("Train part:", train_part.shape)
print("Validation part:", val_part.shape)


# ============================================================
# 4. Feature columns
# ============================================================

drop_cols = [
    "srch_id",
    "prop_id",
    "relevance",
    "booking_rank_label",
    "click_rank_label",
]

feature_cols = [c for c in train_fe.columns if c not in drop_cols]

print("Number of features:", len(feature_cols))


# ============================================================
# 5. Sort by query and create groups
# ============================================================

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

y_booking_train = train_part["booking_rank_label"]
y_booking_val = val_part["booking_rank_label"]

y_click_train = train_part["click_rank_label"]
y_click_val = val_part["click_rank_label"]

train_group = train_part.groupby("srch_id").size().to_numpy()
val_group = val_part.groupby("srch_id").size().to_numpy()

print("Train group rows check:", train_group.sum(), len(train_part))
print("Val group rows check:", val_group.sum(), len(val_part))


# ============================================================
# 6. Metric using original relevance
# ============================================================

def mean_ndcg_at_5(df, label_col="relevance", score_col="model_score"):
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


# ============================================================
# 7. Train booking binary ranker
# ============================================================

booking_ranker = lgb.LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    n_estimators=1200,
    learning_rate=0.03,
    num_leaves=95,
    min_child_samples=100,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=42,
    n_jobs=-1
)

print("=" * 70)
print("Training booking binary ranker")
print("=" * 70)

booking_ranker.fit(
    X_train,
    y_booking_train,
    group=train_group,
    eval_set=[(X_val, y_booking_val)],
    eval_group=[val_group],
    eval_at=[5],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=50)
    ]
)

booking_best_iter = booking_ranker.best_iteration_
if booking_best_iter is None or booking_best_iter <= 0:
    booking_best_iter = 600

val_part["booking_ranker_score"] = booking_ranker.predict(
    X_val,
    num_iteration=booking_best_iter
)

temp = val_part[["srch_id", "prop_id", "relevance", "booking_ranker_score"]].copy()
temp = temp.rename(columns={"booking_ranker_score": "model_score"})
booking_ndcg = mean_ndcg_at_5(temp)

print("Booking ranker best iteration:", booking_best_iter)
print("Booking ranker original-relevance NDCG@5:", booking_ndcg)


# ============================================================
# 8. Train click binary ranker
# ============================================================

click_ranker = lgb.LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    n_estimators=1200,
    learning_rate=0.03,
    num_leaves=95,
    min_child_samples=100,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=42,
    n_jobs=-1
)

print("=" * 70)
print("Training click binary ranker")
print("=" * 70)

click_ranker.fit(
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

click_best_iter = click_ranker.best_iteration_
if click_best_iter is None or click_best_iter <= 0:
    click_best_iter = 600

val_part["click_ranker_score"] = click_ranker.predict(
    X_val,
    num_iteration=click_best_iter
)

temp = val_part[["srch_id", "prop_id", "relevance", "click_ranker_score"]].copy()
temp = temp.rename(columns={"click_ranker_score": "model_score"})
click_ndcg = mean_ndcg_at_5(temp)

print("Click ranker best iteration:", click_best_iter)
print("Click ranker original-relevance NDCG@5:", click_ndcg)


# ============================================================
# 9. Search blend weights
# ============================================================

results = []

candidate_weights = [
    (1.0, 0.0),
    (0.9, 0.1),
    (0.8, 0.2),
    (0.7, 0.3),
    (0.6, 0.4),
    (0.5, 0.5),
    (0.4, 0.6),
    (0.3, 0.7),
    (0.2, 0.8),
    (0.1, 0.9),
    (0.0, 1.0),
]

for w_booking, w_click in candidate_weights:
    print(
        "Testing weights:",
        "booking =", w_booking,
        "click =", w_click
    )

    val_part["model_score"] = (
        w_booking * val_part["booking_ranker_score"]
        + w_click * val_part["click_ranker_score"]
    )

    ndcg = mean_ndcg_at_5(val_part)

    print("NDCG@5:", ndcg)

    results.append({
        "w_booking": w_booking,
        "w_click": w_click,
        "val_ndcg5": ndcg
    })

results_df = pd.DataFrame(results).sort_values("val_ndcg5", ascending=False)
results_df.to_csv(result_path, index=False)

print("=" * 70)
print("MODEL F BINARY RANKER RESULTS")
print(results_df)
print("=" * 70)

best = results_df.iloc[0]
best_w_booking = float(best["w_booking"])
best_w_click = float(best["w_click"])
best_val = float(best["val_ndcg5"])

print("Best booking ranker iter:", booking_best_iter)
print("Best click ranker iter:", click_best_iter)
print("Best w_booking:", best_w_booking)
print("Best w_click:", best_w_click)
print("Best validation NDCG@5:", best_val)


# ============================================================
# 10. Save validation scores
# ============================================================

val_part["model_score"] = (
    best_w_booking * val_part["booking_ranker_score"]
    + best_w_click * val_part["click_ranker_score"]
)

val_scores = val_part[
    [
        "srch_id",
        "prop_id",
        "relevance",
        "booking_ranker_score",
        "click_ranker_score",
        "model_score",
    ]
].copy()

val_scores.to_csv(val_score_path, index=False)

print("Saved validation scores to:", val_score_path)
print("Saved weight results to:", result_path)