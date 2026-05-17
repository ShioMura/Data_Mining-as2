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

# Existing strong model scores
weighted_val_path = OUTPUT_DIR / "model_b_val_scores_extra_v2_weighted_seed_ensemble.csv"
weighted_test_path = OUTPUT_DIR / "model_b_test_scores_extra_v2_weighted_seed_ensemble.csv"

top70_val_path = OUTPUT_DIR / "model_b_val_scores_extra_v2_pruned_top70_seed_ensemble.csv"
top70_test_path = OUTPUT_DIR / "model_b_test_scores_extra_v2_pruned_top70_seed_ensemble.csv"

# New outputs
val_score_path = OUTPUT_DIR / "model_b_val_scores_extra_v2_additional_seed_blend.csv"
test_score_path = OUTPUT_DIR / "model_b_test_scores_extra_v2_additional_seed_blend.csv"
result_path = OUTPUT_DIR / "additional_seed_blend_results.csv"
submission_path = OUTPUT_DIR / "submission_blend_currentbest_additional_seeds.csv"

print("Train feature path exists:", train_path.exists())
print("Test feature path exists:", test_path.exists())
print("weighted val exists:", weighted_val_path.exists())
print("weighted test exists:", weighted_test_path.exists())
print("top70 val exists:", top70_val_path.exists())
print("top70 test exists:", top70_test_path.exists())


# ============================================================
# 1. Helpers
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


# ============================================================
# 2. Load features
# ============================================================

train_fe = pd.read_parquet(train_path)
test_fe = pd.read_parquet(test_path)

print("Train features:", train_fe.shape)
print("Test features:", test_fe.shape)


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

drop_cols = ["srch_id", "prop_id", "relevance"]
feature_cols = [c for c in train_fe.columns if c not in drop_cols]

print("Number of features:", len(feature_cols))


# ============================================================
# 5. Prepare train/val matrices
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
# 6. Load current best validation scores
#    current_best = 0.80 weighted_seed + 0.20 top70_seed
# ============================================================

weighted_val = pd.read_csv(weighted_val_path)
top70_val = pd.read_csv(top70_val_path)

weighted_val = weighted_val.rename(columns={"blend_score": "weighted_seed_score"})
top70_val = top70_val.rename(columns={"ensemble_score": "top70_seed_score"})

current_val = weighted_val[["srch_id", "prop_id", "relevance", "weighted_seed_score"]].merge(
    top70_val[["srch_id", "prop_id", "top70_seed_score"]],
    on=["srch_id", "prop_id"],
    how="inner"
)

current_val["current_best_score"] = (
    0.80 * current_val["weighted_seed_score"]
    + 0.20 * current_val["top70_seed_score"]
)

print("Current best val score data:", current_val.shape)

current_val["blend_score"] = current_val["current_best_score"]
current_best_ndcg = mean_ndcg_at_5(current_val)

print("Current best reconstructed NDCG@5:", current_best_ndcg)


# ============================================================
# 7. Train new seed validation models
# ============================================================

new_seeds = [777, 1001]

best_iterations = {}

for seed in new_seeds:
    print("=" * 70)
    print("Training additional validation model with seed =", seed)
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

    temp_val = val_part[["srch_id", "prop_id", "relevance", score_col]].copy()
    temp_val = temp_val.rename(columns={score_col: "blend_score"})
    single_ndcg = mean_ndcg_at_5(temp_val)

    print("Seed:", seed)
    print("Best iteration:", best_iter)
    print("Single seed validation NDCG@5:", single_ndcg)


# Merge new seed scores into current_val
for seed in new_seeds:
    score_col = f"score_seed_{seed}"
    current_val = current_val.merge(
        val_part[["srch_id", "prop_id", score_col]],
        on=["srch_id", "prop_id"],
        how="inner"
    )

print("Validation blend data:", current_val.shape)


# ============================================================
# 8. Search blend weights
# ============================================================

results = []

# current_best should keep large weight
# remaining weight split between seed777 and seed1001
candidate_current_weights = np.arange(0.70, 1.001, 0.05)
split_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]

for w_current in candidate_current_weights:
    remaining = 1.0 - w_current

    for r in split_ratios:
        w777 = remaining * r
        w1001 = remaining * (1.0 - r)

        print(
            "Testing weights:",
            "current =", round(w_current, 3),
            "seed777 =", round(w777, 3),
            "seed1001 =", round(w1001, 3)
        )

        current_val["blend_score"] = (
            w_current * current_val["current_best_score"]
            + w777 * current_val["score_seed_777"]
            + w1001 * current_val["score_seed_1001"]
        )

        ndcg = mean_ndcg_at_5(current_val)

        print("NDCG@5:", ndcg)

        results.append({
            "w_current": w_current,
            "w777": w777,
            "w1001": w1001,
            "val_ndcg5": ndcg
        })

results_df = pd.DataFrame(results).sort_values("val_ndcg5", ascending=False)
results_df.to_csv(result_path, index=False)

print("=" * 70)
print("Additional seed blend results:")
print(results_df.head(20))
print("=" * 70)

best = results_df.iloc[0]

best_w_current = float(best["w_current"])
best_w777 = float(best["w777"])
best_w1001 = float(best["w1001"])
best_val = float(best["val_ndcg5"])

print("Best w_current:", best_w_current)
print("Best w777:", best_w777)
print("Best w1001:", best_w1001)
print("Best validation NDCG@5:", best_val)

current_val["blend_score"] = (
    best_w_current * current_val["current_best_score"]
    + best_w777 * current_val["score_seed_777"]
    + best_w1001 * current_val["score_seed_1001"]
)

val_scores = current_val[["srch_id", "prop_id", "relevance", "blend_score"]].copy()
val_scores.to_csv(val_score_path, index=False)
print("Saved validation blend scores to:", val_score_path)


# ============================================================
# 9. Train new seed final models on full train
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

for seed in new_seeds:
    n_estimators = best_iterations[seed]

    print("=" * 70)
    print("Training additional full model with seed =", seed)
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

    test_sorted[f"score_seed_{seed}"] = final_ranker.predict(X_test)


# ============================================================
# 10. Load current best test scores and apply best blend
# ============================================================

weighted_test = pd.read_csv(weighted_test_path)
top70_test = pd.read_csv(top70_test_path)

weighted_test = weighted_test.rename(columns={"blend_score": "weighted_seed_score"})
top70_test = top70_test.rename(columns={"ensemble_score": "top70_seed_score"})

current_test = weighted_test[["srch_id", "prop_id", "weighted_seed_score"]].merge(
    top70_test[["srch_id", "prop_id", "top70_seed_score"]],
    on=["srch_id", "prop_id"],
    how="inner"
)

current_test["current_best_score"] = (
    0.80 * current_test["weighted_seed_score"]
    + 0.20 * current_test["top70_seed_score"]
)

for seed in new_seeds:
    score_col = f"score_seed_{seed}"
    current_test = current_test.merge(
        test_sorted[["srch_id", "prop_id", score_col]],
        on=["srch_id", "prop_id"],
        how="inner"
    )

print("Test blend data:", current_test.shape)

current_test["blend_score"] = (
    best_w_current * current_test["current_best_score"]
    + best_w777 * current_test["score_seed_777"]
    + best_w1001 * current_test["score_seed_1001"]
)

test_scores = current_test[["srch_id", "prop_id", "blend_score"]].copy()
test_scores.to_csv(test_score_path, index=False)
print("Saved test blend scores to:", test_score_path)

submission = (
    current_test
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