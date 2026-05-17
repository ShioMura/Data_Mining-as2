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

submission_path = OUTPUT_DIR / "submission_model_b_lgbm_ranker_extra_v2_weighted_seed_ensemble.csv"
val_score_path = OUTPUT_DIR / "model_b_val_scores_extra_v2_weighted_seed_ensemble.csv"
test_score_path = OUTPUT_DIR / "model_b_test_scores_extra_v2_weighted_seed_ensemble.csv"
weight_result_path = OUTPUT_DIR / "weighted_seed_ensemble_results.csv"

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
# 4. Sort and groups
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
# 5. Metric
# ============================================================

def mean_ndcg_at_5(df, label_col="relevance", score_col="blend_score"):
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
# 6. Train validation models for each seed
# ============================================================

seeds = [42, 2024, 3407]

val_pred_dict = {}
best_iterations = {}

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

    best_iterations[seed] = best_iter

    score_col = f"score_seed_{seed}"
    val_part[score_col] = ranker.predict(X_val, num_iteration=best_iter)
    val_pred_dict[seed] = score_col

    val_part["blend_score"] = val_part[score_col]
    single_ndcg = mean_ndcg_at_5(val_part)

    print("Seed:", seed)
    print("Best iteration:", best_iter)
    print("Single seed validation NDCG@5:", single_ndcg)


# ============================================================
# 7. Normalize scores within search
# ============================================================

for seed in seeds:
    score_col = f"score_seed_{seed}"
    norm_col = f"{score_col}_norm"
    val_part[norm_col] = normalize_within_search(val_part, score_col)


# ============================================================
# 8. Search seed weights - fast candidate version
# ============================================================

results = []

candidate_weights = [
    (1/3, 1/3, 1/3),   # equal ensemble
    (0.45, 0.10, 0.45),
    (0.50, 0.10, 0.40),
    (0.55, 0.10, 0.35),
    (0.60, 0.10, 0.30),
    (0.65, 0.10, 0.25),
    (0.70, 0.05, 0.25),
    (0.65, 0.05, 0.30),
    (0.60, 0.05, 0.35),
    (0.55, 0.05, 0.40),
    (0.50, 0.05, 0.45),
    (0.75, 0.05, 0.20),
    (0.80, 0.05, 0.15),
]

for w42, w2024, w3407 in candidate_weights:
    print(
        "Testing weights:",
        "w42 =", round(w42, 3),
        "w2024 =", round(w2024, 3),
        "w3407 =", round(w3407, 3)
    )

    # Raw score blend only first
    val_part["blend_score"] = (
        w42 * val_part["score_seed_42"]
        + w2024 * val_part["score_seed_2024"]
        + w3407 * val_part["score_seed_3407"]
    )

    raw_ndcg = mean_ndcg_at_5(val_part)

    results.append({
        "w42": w42,
        "w2024": w2024,
        "w3407": w3407,
        "raw_ndcg5": raw_ndcg,
        "norm_ndcg5": np.nan
    })

    print("NDCG@5:", raw_ndcg)

results_df = pd.DataFrame(results)

print("=" * 70)
print("Top weighted seed ensemble:")
print(results_df.sort_values("raw_ndcg5", ascending=False).head(20))
print("=" * 70)

results_df.to_csv(weight_result_path, index=False)
print("Saved weight search results to:", weight_result_path)

best = results_df.sort_values("raw_ndcg5", ascending=False).iloc[0]
use_norm = False
best_val = float(best["raw_ndcg5"])

best_w42 = float(best["w42"])
best_w2024 = float(best["w2024"])
best_w3407 = float(best["w3407"])

print("Using raw weighted seed ensemble")
print("Best weights:")
print("w42:", best_w42)
print("w2024:", best_w2024)
print("w3407:", best_w3407)
print("Best validation NDCG@5:", best_val)

val_part["blend_score"] = (
    best_w42 * val_part["score_seed_42"]
    + best_w2024 * val_part["score_seed_2024"]
    + best_w3407 * val_part["score_seed_3407"]
)

val_scores = val_part[["srch_id", "prop_id", "relevance", "blend_score"]].copy()
val_scores.to_csv(val_score_path, index=False)
print("Saved validation scores to:", val_score_path)


# ============================================================
# 9. Train final models on full train
# ============================================================

train_full = train_fe.sort_values("srch_id").reset_index(drop=True)
test_sorted = test_fe.sort_values("srch_id").reset_index(drop=True)

X_full = (
    train_full[feature_cols]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(-999)
)

y_full = train_full["relevance"]
full_group = train_full.groupby("srch_id").size().to_numpy()

X_test = (
    test_sorted[feature_cols]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(-999)
)

test_score_cols = []

for seed in seeds:
    n_estimators = best_iterations[seed]

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

    score_col = f"score_seed_{seed}"
    test_sorted[score_col] = final_ranker.predict(X_test)
    test_score_cols.append(score_col)


# ============================================================
# 10. Test weighted blend
# ============================================================

if use_norm:
    for seed in seeds:
        score_col = f"score_seed_{seed}"
        norm_col = f"{score_col}_norm"
        test_sorted[norm_col] = normalize_within_search(test_sorted, score_col)

    test_sorted["blend_score"] = (
        best_w42 * test_sorted["score_seed_42_norm"]
        + best_w2024 * test_sorted["score_seed_2024_norm"]
        + best_w3407 * test_sorted["score_seed_3407_norm"]
    )
else:
    test_sorted["blend_score"] = (
        best_w42 * test_sorted["score_seed_42"]
        + best_w2024 * test_sorted["score_seed_2024"]
        + best_w3407 * test_sorted["score_seed_3407"]
    )

test_scores = test_sorted[["srch_id", "prop_id", "blend_score"]].copy()
test_scores.to_csv(test_score_path, index=False)
print("Saved test weighted seed scores to:", test_score_path)

submission = (
    test_sorted
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