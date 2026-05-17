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

# Use clean extra v2 best features, not the already-TE version
train_path = DATA_DIR / "train_features_lean_extra_v2_best.parquet"

print("ROOT_DIR:", ROOT_DIR)
print("DATA_DIR:", DATA_DIR)
print("Train path exists:", train_path.exists())


# ============================================================
# 1. Load extra v2 features
# ============================================================

train_fe = pd.read_parquet(train_path)

print("Original train features:", train_fe.shape)


# ============================================================
# 2. Split by srch_id first
# ============================================================

unique_srch_ids = train_fe["srch_id"].unique()

train_ids, val_ids = train_test_split(
    unique_srch_ids,
    test_size=0.2,
    random_state=42
)

train_part = train_fe[train_fe["srch_id"].isin(train_ids)].copy()
val_part = train_fe[train_fe["srch_id"].isin(val_ids)].copy()

print("Train part before TE:", train_part.shape)
print("Validation part before TE:", val_part.shape)


# ============================================================
# 3. Create conservative target inside train_part only
# ============================================================

# Conservative target:
# clicked or booked = 1
# otherwise = 0
train_part["target_click"] = (train_part["relevance"] >= 1).astype("float32")

keys = [
    "prop_id",
    "srch_destination_id",
]

alpha = 300.0
prior = float(train_part["target_click"].mean())

print("Strict conservative TE prior:", prior)
print("Smoothing alpha:", alpha)


# ============================================================
# 4. Strict conservative TE functions
# ============================================================

def add_strict_click_te(train_df, val_df, key_col):
    """
    Strict TE:
    - statistics are computed only from train_df
    - then mapped to both train_df and val_df
    - validation labels are never used
    """
    out_col = f"{key_col}_click_rate_strict_cons"
    count_col = f"{key_col}_hist_count_log_strict_cons"

    print(f"Adding strict conservative TE: {out_col}")

    stats = (
        train_df.groupby(key_col, sort=False)["target_click"]
        .agg(["sum", "count"])
        .reset_index()
    )

    stats[out_col] = (
        (stats["sum"] + alpha * prior)
        / (stats["count"] + alpha)
    ).astype("float32")

    # Apply TE to train rows
    train_df[out_col] = (
        train_df[[key_col]]
        .merge(stats[[key_col, out_col]], on=key_col, how="left")[out_col]
        .fillna(prior)
        .astype("float32")
    )

    # Apply TE to validation rows using train-only stats
    val_df[out_col] = (
        val_df[[key_col]]
        .merge(stats[[key_col, out_col]], on=key_col, how="left")[out_col]
        .fillna(prior)
        .astype("float32")
    )

    # Historical exposure count based only on train_df
    count_stats = (
        train_df.groupby(key_col, sort=False)
        .size()
        .reset_index(name=count_col)
    )

    count_stats[count_col] = np.log1p(count_stats[count_col]).astype("float32")

    train_df[count_col] = (
        train_df[[key_col]]
        .merge(count_stats[[key_col, count_col]], on=key_col, how="left")[count_col]
        .fillna(0)
        .astype("float32")
    )

    val_df[count_col] = (
        val_df[[key_col]]
        .merge(count_stats[[key_col, count_col]], on=key_col, how="left")[count_col]
        .fillna(0)
        .astype("float32")
    )

    print(f"Added count feature: {count_col}")


# ============================================================
# 5. Apply strict conservative TE
# ============================================================

for key_col in keys:
    if key_col not in train_part.columns:
        print(f"Skipping {key_col}, not found.")
        continue

    add_strict_click_te(train_part, val_part, key_col)

# Remove temporary target column
train_part.drop(columns=["target_click"], inplace=True)

print("Train part after TE:", train_part.shape)
print("Validation part after TE:", val_part.shape)


# ============================================================
# 6. Feature columns
# ============================================================

drop_cols = ["srch_id", "prop_id", "relevance"]
feature_cols = [c for c in train_part.columns if c not in drop_cols]

print("Number of features:", len(feature_cols))


# ============================================================
# 7. Sort by query and create groups
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
# 8. Train LightGBM LambdaRank model
# ============================================================

ranker = lgb.LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    n_estimators=1000,
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
    eval_set=[(X_val, y_val)],
    eval_group=[val_group],
    eval_at=[5],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=50)
    ]
)


# ============================================================
# 9. Manual validation NDCG@5
# ============================================================

val_part["model_b_score"] = ranker.predict(
    X_val,
    num_iteration=ranker.best_iteration_
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

print("=" * 70)
print("STRICT CONSERVATIVE TE VALIDATION RESULT")
print("Best iteration:", ranker.best_iteration_)
print("Strict conservative TE validation NDCG@5:", val_ndcg5)
print("=" * 70)


# ============================================================
# 10. Feature importance
# ============================================================

importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": ranker.feature_importances_
}).sort_values("importance", ascending=False)

importance_path = OUTPUT_DIR / "model_b_feature_importance_strict_conservative_te_validation.csv"
importance_df.to_csv(importance_path, index=False)

print("Top 30 features:")
print(importance_df.head(30))
print("Saved feature importance to:", importance_path)