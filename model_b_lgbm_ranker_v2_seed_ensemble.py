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

submission_path = OUTPUT_DIR / "submission_model_b_lgbm_ranker_extra_v2_seed_ensemble.csv"
importance_path = OUTPUT_DIR / "model_b_feature_importance_extra_v2_seed_ensemble.csv"

val_score_path = OUTPUT_DIR / "model_b_val_scores_extra_v2_seed_ensemble.csv"
test_score_path = OUTPUT_DIR / "model_b_test_scores_extra_v2_seed_ensemble.csv"

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

X_train = train_part[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(-999)
y_train = train_part["relevance"]

X_val = val_part[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(-999)
y_val = val_part["relevance"]

train_group = train_part.groupby("srch_id").size().to_numpy()
val_group = val_part.groupby("srch_id").size().to_numpy()

print("Train group rows check:", train_group.sum(), len(train_part))
print("Val group rows check:", val_group.sum(), len(val_part))


# ============================================================
# 5. Metric
# ============================================================

def mean_ndcg_at_5(df, label_col="relevance", score_col="ensemble_score"):
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
# 6. Train several seeds on train/validation
# ============================================================

seeds = [42, 2024, 3407]

val_pred_matrix = []
best_iterations = []
importance_list = []

for seed in seeds:
    print("=" * 70)
    print("Training validation model with seed =", seed)
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
        n_jobs=-1
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
            lgb.log_evaluation(period=50)
        ]
    )

    best_iter = ranker.best_iteration_
    if best_iter is None or best_iter <= 0:
        best_iter = 780

    best_iterations.append(best_iter)

    val_pred = ranker.predict(X_val, num_iteration=best_iter)
    val_pred_matrix.append(val_pred)

    importance_list.append(ranker.feature_importances_)

    val_part[f"score_seed_{seed}"] = val_pred
    val_part["single_seed_score"] = val_pred

    single_ndcg = mean_ndcg_at_5(
        val_part,
        label_col="relevance",
        score_col="single_seed_score"
    )

    print("Seed:", seed)
    print("Best iteration:", best_iter)
    print("Single seed validation NDCG@5:", single_ndcg)


# ============================================================
# 7. Validation ensemble score
# ============================================================

val_pred_matrix = np.vstack(val_pred_matrix)
val_part["ensemble_score"] = val_pred_matrix.mean(axis=0)

ensemble_ndcg = mean_ndcg_at_5(
    val_part,
    label_col="relevance",
    score_col="ensemble_score"
)

print("=" * 70)
print("ENSEMBLE VALIDATION RESULT")
print("Seeds:", seeds)
print("Best iterations:", best_iterations)
print("Validation NDCG@5:", ensemble_ndcg)
print("=" * 70)

val_scores = val_part[["srch_id", "prop_id", "relevance", "ensemble_score"]].copy()
val_scores.to_csv(val_score_path, index=False)
print("Saved validation ensemble scores to:", val_score_path)


# ============================================================
# 8. Average feature importance
# ============================================================

importance_arr = np.vstack(importance_list)
importance_mean = importance_arr.mean(axis=0)

importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": importance_mean
}).sort_values("importance", ascending=False)

importance_df.to_csv(importance_path, index=False)

print("Top 30 features:")
print(importance_df.head(30))
print("Saved feature importance to:", importance_path)


# ============================================================
# 9. Train final models on full train and predict test
# ============================================================

train_full = train_fe.sort_values("srch_id").reset_index(drop=True)
test_sorted = test_fe.sort_values("srch_id").reset_index(drop=True)

X_full = train_full[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(-999)
y_full = train_full["relevance"]
full_group = train_full.groupby("srch_id").size().to_numpy()

X_test = test_sorted[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(-999)

test_pred_matrix = []

for seed, n_estimators in zip(seeds, best_iterations):
    print("=" * 70)
    print("Training final full model with seed =", seed)
    print("n_estimators =", n_estimators)
    print("=" * 70)

    final_ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=n_estimators,
        learning_rate=0.03,
        num_leaves=95,
        min_child_samples=100,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=seed,
        n_jobs=-1
    )

    final_ranker.fit(
        X_full,
        y_full,
        group=full_group
    )

    test_pred = final_ranker.predict(X_test)
    test_pred_matrix.append(test_pred)


# ============================================================
# 10. Create ensemble submission
# ============================================================

test_pred_matrix = np.vstack(test_pred_matrix)
test_sorted["ensemble_score"] = test_pred_matrix.mean(axis=0)

test_scores = test_sorted[["srch_id", "prop_id", "ensemble_score"]].copy()
test_scores.to_csv(test_score_path, index=False)
print("Saved test ensemble scores to:", test_score_path)

submission = (
    test_sorted
    .sort_values(["srch_id", "ensemble_score"], ascending=[True, False])
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