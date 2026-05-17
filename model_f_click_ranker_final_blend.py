from pathlib import Path

import numpy as np
import pandas as pd
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

weighted_test_path = OUTPUT_DIR / "model_b_test_scores_extra_v2_weighted_seed_ensemble.csv"
top70_test_path = OUTPUT_DIR / "model_b_test_scores_extra_v2_pruned_top70_seed_ensemble.csv"
v2_test_path = OUTPUT_DIR / "model_b_test_scores_extra_v2_best.csv"

test_score_path = OUTPUT_DIR / "model_f_click_ranker_final_test_scores.csv"
submission_path = OUTPUT_DIR / "submission_currentbest_model_f_click_blend_finetuned.csv"

print("Train exists:", train_path.exists())
print("Test exists:", test_path.exists())
print("weighted test exists:", weighted_test_path.exists())
print("top70 test exists:", top70_test_path.exists())
print("v2 test exists:", v2_test_path.exists())


# ============================================================
# 1. Helpers
# ============================================================

def normalize_within_search(df, score_col):
    g = df.groupby("srch_id")[score_col]
    mean = g.transform("mean")
    std = g.transform("std").replace(0, 1)
    return ((df[score_col] - mean) / std).fillna(0)


def find_score_col(df):
    for col in ["blend_score", "ensemble_score", "model_b_score", "model_score"]:
        if col in df.columns:
            return col
    raise ValueError(f"No score column found: {df.columns.tolist()}")


# ============================================================
# 2. Load v2 features
# ============================================================

train_fe = pd.read_parquet(train_path)
test_fe = pd.read_parquet(test_path)

print("Train features:", train_fe.shape)
print("Test features:", test_fe.shape)

train_fe["click_rank_label"] = (train_fe["relevance"] >= 1).astype("int8")


# ============================================================
# 3. Feature columns
# ============================================================

drop_cols = ["srch_id", "prop_id", "relevance", "click_rank_label"]
feature_cols = [c for c in train_fe.columns if c not in drop_cols]

print("Number of features:", len(feature_cols))


# ============================================================
# 4. Prepare full train and test
# ============================================================

train_full = train_fe.sort_values("srch_id").reset_index(drop=True)
test_sorted = test_fe.sort_values("srch_id").reset_index(drop=True)

X_full = (
    train_full[feature_cols]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(-999)
)

y_full = train_full["click_rank_label"]
full_group = train_full.groupby("srch_id").size().to_numpy()

X_test = (
    test_sorted[feature_cols]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(-999)
)

print("Full group rows check:", full_group.sum(), len(train_full))


# ============================================================
# 5. Train final click binary ranker
# From validation:
# click ranker best iteration = 592
# ============================================================

click_best_iter = 592

print("=" * 70)
print("Training final Model F click binary ranker")
print("n_estimators =", click_best_iter)
print("=" * 70)

click_ranker = lgb.LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    n_estimators=click_best_iter,
    learning_rate=0.03,
    num_leaves=95,
    min_child_samples=100,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=42,
    n_jobs=-1,
)

click_ranker.fit(
    X_full,
    y_full,
    group=full_group,
)

test_sorted["model_f_click_score"] = click_ranker.predict(X_test)

model_f_test_scores = test_sorted[["srch_id", "prop_id", "model_f_click_score"]].copy()
model_f_test_scores.to_csv(test_score_path, index=False)

print("Saved Model F click test scores to:", test_score_path)


# ============================================================
# 6. Reconstruct current best test score
# current best before Model F:
# current_base = 0.80 * weighted_seed + 0.20 * top70_seed
# final_current = 0.95 * current_base_norm + 0.05 * v2_single_norm
# ============================================================

weighted_test = pd.read_csv(weighted_test_path)
top70_test = pd.read_csv(top70_test_path)
v2_test = pd.read_csv(v2_test_path)

weighted_test = weighted_test.rename(columns={"blend_score": "weighted_seed_score"})
top70_test = top70_test.rename(columns={"ensemble_score": "top70_seed_score"})

v2_score_col = find_score_col(v2_test)
v2_test = v2_test.rename(columns={v2_score_col: "v2_single_score"})

test = weighted_test[["srch_id", "prop_id", "weighted_seed_score"]].merge(
    top70_test[["srch_id", "prop_id", "top70_seed_score"]],
    on=["srch_id", "prop_id"],
    how="inner",
)

test = test.merge(
    v2_test[["srch_id", "prop_id", "v2_single_score"]],
    on=["srch_id", "prop_id"],
    how="inner",
)

test = test.merge(
    model_f_test_scores,
    on=["srch_id", "prop_id"],
    how="inner",
)

print("Final blend test data:", test.shape)

test["current_base_score"] = (
    0.80 * test["weighted_seed_score"]
    + 0.20 * test["top70_seed_score"]
)

test["current_base_score_norm"] = normalize_within_search(test, "current_base_score")
test["v2_single_score_norm"] = normalize_within_search(test, "v2_single_score")

test["current_best_score"] = (
    0.95 * test["current_base_score_norm"]
    + 0.05 * test["v2_single_score_norm"]
)


# ============================================================
# 7. Apply Model F blend
# Best validation:
# 0.99 * current_best + 0.01 * Model F click ranker
# ============================================================

test["blend_score"] = (
    0.992 * test["current_best_score"]
    + 0.008 * test["model_f_click_score"]
)

submission = (
    test
    .sort_values(["srch_id", "blend_score"], ascending=[True, False])
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