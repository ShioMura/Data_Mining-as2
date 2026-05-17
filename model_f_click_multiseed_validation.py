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
model_f_seed42_val_path = OUTPUT_DIR / "model_f_val_scores_binary_rankers.csv"

result_path = OUTPUT_DIR / "model_f_click_multiseed_validation_results.csv"
val_score_path = OUTPUT_DIR / "model_f_click_multiseed_best_val_scores.csv"

print("Train exists:", train_path.exists())
print("weighted val exists:", weighted_val_path.exists())
print("top70 val exists:", top70_val_path.exists())
print("v2 val exists:", v2_val_path.exists())
print("F seed42 val exists:", model_f_seed42_val_path.exists())


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
    raise ValueError(f"No score column found: {df.columns.tolist()}")


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

drop_cols = ["srch_id", "prop_id", "relevance", "click_rank_label"]
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
# 3. Load current_pre_f validation score
# current_base = 0.80 weighted_seed + 0.20 top70_seed
# current_pre_f = 0.95 current_base_norm + 0.05 v2_single_norm
# ============================================================

weighted_val = pd.read_csv(weighted_val_path)
top70_val = pd.read_csv(top70_val_path)
v2_val = pd.read_csv(v2_val_path)
f42_val = pd.read_csv(model_f_seed42_val_path)

weighted_val = weighted_val.rename(columns={"blend_score": "weighted_seed_score"})
top70_val = top70_val.rename(columns={"ensemble_score": "top70_seed_score"})

v2_score_col = find_score_col(v2_val)
v2_val = v2_val.rename(columns={v2_score_col: "v2_single_score"})

if "click_ranker_score" not in f42_val.columns:
    raise ValueError("model_f_val_scores_binary_rankers.csv has no click_ranker_score column.")

f42_val = f42_val.rename(columns={"click_ranker_score": "f_click_seed42"})

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
    f42_val[["srch_id", "prop_id", "f_click_seed42"]],
    on=["srch_id", "prop_id"],
    how="inner"
)

val["current_base_score"] = (
    0.80 * val["weighted_seed_score"]
    + 0.20 * val["top70_seed_score"]
)

val["current_base_score_norm"] = normalize_within_search(val, "current_base_score")
val["v2_single_score_norm"] = normalize_within_search(val, "v2_single_score")

val["current_pre_f_score"] = (
    0.95 * val["current_base_score_norm"]
    + 0.05 * val["v2_single_score_norm"]
)

val["blend_score"] = val["current_pre_f_score"]
pre_f_ndcg = mean_ndcg_at_5(val)

print("Current pre-F NDCG@5:", pre_f_ndcg)


# ============================================================
# 4. Train additional F click rankers
# ============================================================

extra_seeds = [2024, 3407]
best_iterations = {}

for seed in extra_seeds:
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

    best_iterations[seed] = best_iter

    score_col = f"f_click_seed{seed}"

    val_part[score_col] = ranker.predict(
        X_val,
        num_iteration=best_iter
    )

    temp = val_part[["srch_id", "prop_id", "relevance", score_col]].copy()
    temp = temp.rename(columns={score_col: "blend_score"})

    single_ndcg = mean_ndcg_at_5(temp)

    print("Seed:", seed)
    print("Best iteration:", best_iter)
    print("Original relevance NDCG@5:", single_ndcg)

    val = val.merge(
        val_part[["srch_id", "prop_id", score_col]],
        on=["srch_id", "prop_id"],
        how="inner"
    )


print("Validation blend data:", val.shape)
print("Best iterations:", best_iterations)


# ============================================================
# 5. Candidate F click seed ensembles
# ============================================================

val["f_click_equal3"] = (
    val["f_click_seed42"]
    + val["f_click_seed2024"]
    + val["f_click_seed3407"]
) / 3.0

val["f_click_weighted_a"] = (
    0.50 * val["f_click_seed42"]
    + 0.25 * val["f_click_seed2024"]
    + 0.25 * val["f_click_seed3407"]
)

val["f_click_weighted_b"] = (
    0.60 * val["f_click_seed42"]
    + 0.20 * val["f_click_seed2024"]
    + 0.20 * val["f_click_seed3407"]
)

val["f_click_weighted_c"] = (
    0.40 * val["f_click_seed42"]
    + 0.30 * val["f_click_seed2024"]
    + 0.30 * val["f_click_seed3407"]
)

candidate_f_cols = [
    "f_click_seed42",
    "f_click_seed2024",
    "f_click_seed3407",
    "f_click_equal3",
    "f_click_weighted_a",
    "f_click_weighted_b",
    "f_click_weighted_c",
]


# ============================================================
# 6. Search current + F blend
# ============================================================

results = []

for f_col in candidate_f_cols:
    for w_current in np.arange(0.985, 0.9971, 0.001):
        w_f = 1.0 - w_current

        val["blend_score"] = (
            w_current * val["current_pre_f_score"]
            + w_f * val[f_col]
        )

        ndcg = mean_ndcg_at_5(val)

        print(
            "F:", f_col,
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
print("MODEL F CLICK MULTISEED VALIDATION RESULTS")
print(results_df.head(30))
print("=" * 70)

best = results_df.iloc[0]
best_f_model = str(best["f_model"])
best_w_current = float(best["w_current"])
best_w_f = float(best["w_f"])
best_val = float(best["val_ndcg5"])

print("Best F model:", best_f_model)
print("Best w_current:", best_w_current)
print("Best w_f:", best_w_f)
print("Best validation NDCG@5:", best_val)

val["blend_score"] = (
    best_w_current * val["current_pre_f_score"]
    + best_w_f * val[best_f_model]
)

val_scores = val[
    [
        "srch_id",
        "prop_id",
        "relevance",
        "current_pre_f_score",
        "f_click_seed42",
        "f_click_seed2024",
        "f_click_seed3407",
        best_f_model,
        "blend_score",
    ]
].copy()

val_scores.to_csv(val_score_path, index=False)

print("Saved validation scores to:", val_score_path)
print("Saved results to:", result_path)