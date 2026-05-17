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

train_path = DATA_DIR / "train_features_lean_extra_v2_best.parquet"

weighted_val_path = OUTPUT_DIR / "model_b_val_scores_extra_v2_weighted_seed_ensemble.csv"
top70_val_path = OUTPUT_DIR / "model_b_val_scores_extra_v2_pruned_top70_seed_ensemble.csv"
v2_val_path = OUTPUT_DIR / "model_b_val_scores_extra_v2_best.csv"
f_multi_val_path = OUTPUT_DIR / "model_f_click_multiseed_best_val_scores.csv"

result_path = OUTPUT_DIR / "model_f_click_seed777_validation_results.csv"
val_score_path = OUTPUT_DIR / "model_f_click_seed777_val_scores.csv"

print("Train exists:", train_path.exists())
print("weighted val exists:", weighted_val_path.exists())
print("top70 val exists:", top70_val_path.exists())
print("v2 val exists:", v2_val_path.exists())
print("F multiseed val exists:", f_multi_val_path.exists())


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


def normalize_within_search(df, score_col):
    g = df.groupby("srch_id")[score_col]
    mean = g.transform("mean")
    std = g.transform("std").replace(0, 1)
    return ((df[score_col] - mean) / std).fillna(0)


def find_score_col(df):
    for col in ["blend_score", "ensemble_score", "model_b_score", "model_score"]:
        if col in df.columns:
            return col

    raise ValueError(f"No score column found. Columns = {df.columns.tolist()}")


# ============================================================
# 2. Load v2 train and split
# ============================================================

train_fe = pd.read_parquet(train_path)

print("Train features:", train_fe.shape)

train_fe["click_rank_label"] = (train_fe["relevance"] >= 1).astype("int8")

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
# 3. Feature columns and matrices
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

y_train = train_part["click_rank_label"]
y_val = val_part["click_rank_label"]

train_group = train_part.groupby("srch_id").size().to_numpy()
val_group = val_part.groupby("srch_id").size().to_numpy()

print("Train group rows check:", train_group.sum(), len(train_part))
print("Val group rows check:", val_group.sum(), len(val_part))


# ============================================================
# 4. Train F-click seed777
# ============================================================

seed = 777

print("=" * 70)
print("Training Model F click ranker seed =", seed)
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
    best_iter = 592

val_part["f_click_seed777"] = ranker.predict(
    X_val,
    num_iteration=best_iter
)

temp = val_part[["srch_id", "prop_id", "relevance", "f_click_seed777"]].copy()
temp = temp.rename(columns={"f_click_seed777": "blend_score"})

seed777_ndcg = mean_ndcg_at_5(temp)

print("=" * 70)
print("Seed777 result")
print("Best iteration:", best_iter)
print("Seed777 original-relevance NDCG@5:", seed777_ndcg)
print("=" * 70)


# ============================================================
# 5. Reconstruct current_pre_f and existing F scores
# ============================================================

weighted_val = pd.read_csv(weighted_val_path)
top70_val = pd.read_csv(top70_val_path)
v2_val = pd.read_csv(v2_val_path)
f_multi_val = pd.read_csv(f_multi_val_path)

weighted_val = weighted_val.rename(columns={"blend_score": "weighted_seed_score"})
top70_val = top70_val.rename(columns={"ensemble_score": "top70_seed_score"})

v2_col = find_score_col(v2_val)
v2_val = v2_val.rename(columns={v2_col: "v2_single_score"})

required_f_cols = [
    "f_click_seed42",
    "f_click_seed2024",
    "f_click_seed3407",
    "f_click_weighted_b",
]

missing = [c for c in required_f_cols if c not in f_multi_val.columns]
if missing:
    raise ValueError(f"Missing columns in F multiseed val file: {missing}")

val = weighted_val[["srch_id", "prop_id", "relevance", "weighted_seed_score"]].merge(
    top70_val[["srch_id", "prop_id", "top70_seed_score"]],
    on=["srch_id", "prop_id"],
    how="inner"
)

val = val.merge(
    v2_val[["srch_id", "prop_id", "v2_single_score"]],
    on=["srch_id", "prop_id"],
    how="inner"
)

val = val.merge(
    f_multi_val[
        [
            "srch_id",
            "prop_id",
            "f_click_seed42",
            "f_click_seed2024",
            "f_click_seed3407",
            "f_click_weighted_b",
        ]
    ],
    on=["srch_id", "prop_id"],
    how="inner"
)

val = val.merge(
    val_part[["srch_id", "prop_id", "f_click_seed777"]],
    on=["srch_id", "prop_id"],
    how="inner"
)

print("Validation blend data:", val.shape)

val["current_base_score"] = (
    0.80 * val["weighted_seed_score"]
    + 0.20 * val["top70_seed_score"]
)

val["current_base_score_norm"] = normalize_within_search(
    val,
    "current_base_score"
)

val["v2_single_score_norm"] = normalize_within_search(
    val,
    "v2_single_score"
)

val["current_pre_f_score"] = (
    0.95 * val["current_base_score_norm"]
    + 0.05 * val["v2_single_score_norm"]
)

val["blend_score"] = val["current_pre_f_score"]
pre_f_ndcg = mean_ndcg_at_5(val)

print("Current pre-F NDCG@5:", pre_f_ndcg)


# ============================================================
# 6. Build candidate F ensembles including seed777
# ============================================================

# Current best F ensemble:
# f_click_weighted_b = 0.60 seed42 + 0.20 seed2024 + 0.20 seed3407

val["f_click_equal4"] = (
    val["f_click_seed42"]
    + val["f_click_seed2024"]
    + val["f_click_seed3407"]
    + val["f_click_seed777"]
) / 4.0

val["f_click_weighted_d"] = (
    0.50 * val["f_click_seed42"]
    + 0.20 * val["f_click_seed2024"]
    + 0.20 * val["f_click_seed3407"]
    + 0.10 * val["f_click_seed777"]
)

val["f_click_weighted_e"] = (
    0.55 * val["f_click_seed42"]
    + 0.15 * val["f_click_seed2024"]
    + 0.15 * val["f_click_seed3407"]
    + 0.15 * val["f_click_seed777"]
)

val["f_click_weighted_f"] = (
    0.40 * val["f_click_seed42"]
    + 0.20 * val["f_click_seed2024"]
    + 0.20 * val["f_click_seed3407"]
    + 0.20 * val["f_click_seed777"]
)

val["f_click_seed777_only"] = val["f_click_seed777"]

candidate_f_cols = [
    "f_click_weighted_b",
    "f_click_seed777_only",
    "f_click_equal4",
    "f_click_weighted_d",
    "f_click_weighted_e",
    "f_click_weighted_f",
]


# ============================================================
# 7. Search F weight with seed777 candidates
# ============================================================

results = []

for f_col in candidate_f_cols:
    for w_f in np.arange(0.004, 0.0201, 0.0005):
        w_current = 1.0 - w_f

        val["blend_score"] = (
            w_current * val["current_pre_f_score"]
            + w_f * val[f_col]
        )

        ndcg = mean_ndcg_at_5(val)

        print(
            "F model:", f_col,
            "w_current:", round(w_current, 4),
            "w_f:", round(w_f, 4),
            "NDCG@5:", ndcg
        )

        results.append({
            "f_model": f_col,
            "w_current": w_current,
            "w_f": w_f,
            "val_ndcg5": ndcg,
        })


results_df = pd.DataFrame(results).sort_values("val_ndcg5", ascending=False)
results_df.to_csv(result_path, index=False)

print("=" * 70)
print("Seed777 F-click validation results:")
print(results_df.head(40))
print("=" * 70)

best = results_df.iloc[0]

print("Best result:")
print(best.to_dict())

# Save validation scores with all F candidates
val_scores = val[
    [
        "srch_id",
        "prop_id",
        "relevance",
        "current_pre_f_score",
        "f_click_seed42",
        "f_click_seed2024",
        "f_click_seed3407",
        "f_click_seed777",
        "f_click_weighted_b",
        "f_click_equal4",
        "f_click_weighted_d",
        "f_click_weighted_e",
        "f_click_weighted_f",
    ]
].copy()

val_scores.to_csv(val_score_path, index=False)

print("Saved validation scores to:", val_score_path)
print("Saved results to:", result_path)