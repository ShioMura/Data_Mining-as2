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
test_path = DATA_DIR / "test_features_lean_extra_v2_best.parquet"

submission_path = OUTPUT_DIR / "submission_model_b_lgbm_ranker_extra_v2_random_weighted.csv"
importance_path = OUTPUT_DIR / "model_b_feature_importance_extra_v2_random_weighted.csv"

val_score_path = OUTPUT_DIR / "model_b_val_scores_extra_v2_random_weighted.csv"
test_score_path = OUTPUT_DIR / "model_b_test_scores_extra_v2_random_weighted.csv"

print("ROOT_DIR:", ROOT_DIR)
print("DATA_DIR:", DATA_DIR)
print("Train feature path exists:", train_path.exists())
print("Test feature path exists:", test_path.exists())


# ============================================================
# 1. Load extra v2 features
# ============================================================

train_fe = pd.read_parquet(train_path)
test_fe = pd.read_parquet(test_path)

print("Train features:", train_fe.shape)
print("Test features:", test_fe.shape)

if "random_bool" not in train_fe.columns:
    raise ValueError("random_bool not found in train features.")


# ============================================================
# 2. Split by srch_id
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

print("Train random_bool distribution:")
print(train_part["random_bool"].value_counts(dropna=False))

print("Validation random_bool distribution:")
print(val_part["random_bool"].value_counts(dropna=False))


# ============================================================
# 3. Feature columns
# ============================================================

drop_cols = ["srch_id", "prop_id", "relevance"]
feature_cols = [c for c in train_fe.columns if c not in drop_cols]

print("Number of features:", len(feature_cols))


# ============================================================
# 4. Sort by query and create groups
# ============================================================

train_part = train_part.sort_values("srch_id").reset_index(drop=True)
val_part = val_part.sort_values("srch_id").reset_index(drop=True)

X_train = (
    train_part[feature_cols]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(-999)
)

y_train = train_part["relevance"]

X_val = (
    val_part[feature_cols]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(-999)
)

y_val = val_part["relevance"]

train_group = train_part.groupby("srch_id").size().to_numpy()
val_group = val_part.groupby("srch_id").size().to_numpy()

print("Train group rows check:", train_group.sum(), len(train_part))
print("Val group rows check:", val_group.sum(), len(val_part))


# ============================================================
# 5. Sample weights using random_bool
# ============================================================

# First experiment:
# random_bool == 1 gets higher weight.
random_weight = 2.0

sample_weight = np.where(
    train_part["random_bool"].fillna(0).to_numpy() == 1,
    random_weight,
    1.0
).astype("float32")

print("Sample weight summary:")
print(pd.Series(sample_weight).describe())
print("random_bool=1 weight:", random_weight)
print("random_bool=0 weight:", 1.0)


# ============================================================
# 6. Train random-weighted LightGBM LambdaRank model
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
    n_jobs=-1
)

ranker.fit(
    X_train,
    y_train,
    group=train_group,
    sample_weight=sample_weight,
    eval_set=[(X_val, y_val)],
    eval_group=[val_group],
    eval_at=[5],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=50)
    ]
)


# ============================================================
# 7. Validation NDCG@5
# ============================================================

best_iter = ranker.best_iteration_
if best_iter is None or best_iter <= 0:
    best_iter = 780

val_part["model_b_score"] = ranker.predict(
    X_val,
    num_iteration=best_iter
)


def mean_ndcg_at_5(df, label_col="relevance", score_col="model_b_score"):
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


val_ndcg5 = mean_ndcg_at_5(val_part)

val_scores = val_part[["srch_id", "prop_id", "relevance", "model_b_score"]].copy()
val_scores.to_csv(val_score_path, index=False)

print("=" * 70)
print("RANDOM_BOOL WEIGHTED MODEL RESULT")
print("random_bool=1 weight:", random_weight)
print("Best iteration:", best_iter)
print("Validation NDCG@5:", val_ndcg5)
print("=" * 70)

print("Saved validation scores to:", val_score_path)


# ============================================================
# 8. Feature importance
# ============================================================

importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": ranker.feature_importances_
}).sort_values("importance", ascending=False)

importance_df.to_csv(importance_path, index=False)

print("Top 30 features:")
print(importance_df.head(30))
print("Saved feature importance to:", importance_path)


# ============================================================
# 9. Train final model on full training data
# ============================================================

train_full = train_fe.sort_values("srch_id").reset_index(drop=True)

X_full = (
    train_full[feature_cols]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(-999)
)

y_full = train_full["relevance"]
full_group = train_full.groupby("srch_id").size().to_numpy()

full_sample_weight = np.where(
    train_full["random_bool"].fillna(0).to_numpy() == 1,
    random_weight,
    1.0
).astype("float32")

print("Training final random-weighted model with n_estimators =", best_iter)

final_ranker = lgb.LGBMRanker(
    objective="lambdarank",
    metric="ndcg",

    n_estimators=best_iter,
    learning_rate=0.03,
    num_leaves=95,
    min_child_samples=100,
    subsample=0.85,
    colsample_bytree=0.85,

    random_state=42,
    n_jobs=-1
)

final_ranker.fit(
    X_full,
    y_full,
    group=full_group,
    sample_weight=full_sample_weight
)


# ============================================================
# 10. Predict test and create submission
# ============================================================

test_sorted = test_fe.sort_values("srch_id").reset_index(drop=True)

X_test = (
    test_sorted[feature_cols]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(-999)
)

test_sorted["model_b_score"] = final_ranker.predict(X_test)

test_scores = test_sorted[["srch_id", "prop_id", "model_b_score"]].copy()
test_scores.to_csv(test_score_path, index=False)

print("Saved test scores to:", test_score_path)

submission = (
    test_sorted
    .sort_values(["srch_id", "model_b_score"], ascending=[True, False])
    [["srch_id", "prop_id"]]
)

submission.to_csv(submission_path, index=False)

print("Saved submission to:", submission_path)
print("Submission shape:", submission.shape)
print(submission.head(30))

print("Unique srch_id:", submission["srch_id"].nunique())
print("Missing values:")
print(submission.isna().sum())
print("Duplicated srch_id-prop_id:", submission.duplicated(["srch_id", "prop_id"]).sum())