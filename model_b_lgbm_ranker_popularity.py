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

train_path = DATA_DIR / "train_features_lean.parquet"
test_path = DATA_DIR / "test_features_lean.parquet"

submission_path = OUTPUT_DIR / "submission_model_b_lgbm_ranker_popularity.csv"
importance_path = OUTPUT_DIR / "model_b_feature_importance_popularity.csv"

print("ROOT_DIR:", ROOT_DIR)
print("DATA_DIR:", DATA_DIR)
print("Train feature path exists:", train_path.exists())
print("Test feature path exists:", test_path.exists())


# ============================================================
# 1. Load engineered features
# ============================================================

train_fe = pd.read_parquet(train_path)
test_fe = pd.read_parquet(test_path)

print("Train features:", train_fe.shape)
print("Test features:", test_fe.shape)


# ============================================================
# 2. Helper functions for property popularity
# ============================================================

def build_property_popularity(base_df, alpha=10):
    """
    Build smoothed property-level popularity features.

    Important:
    For validation, base_df should only be train_part,
    not the full train set, to avoid leakage.
    """

    global_ctr = (base_df["relevance"] >= 1).mean()
    global_booking_rate = (base_df["relevance"] == 5).mean()

    prop_pop = (
        base_df
        .groupby("prop_id")
        .agg(
            prop_presentations=("relevance", "size"),
            prop_clicks=("relevance", lambda x: (x >= 1).sum()),
            prop_bookings=("relevance", lambda x: (x == 5).sum())
        )
        .reset_index()
    )

    prop_pop["prop_ctr"] = (
        (prop_pop["prop_clicks"] + alpha * global_ctr)
        / (prop_pop["prop_presentations"] + alpha)
    ).astype("float32")

    prop_pop["prop_booking_rate"] = (
        (prop_pop["prop_bookings"] + alpha * global_booking_rate)
        / (prop_pop["prop_presentations"] + alpha)
    ).astype("float32")

    prop_pop["prop_popularity_log"] = np.log1p(
        prop_pop["prop_presentations"]
    ).astype("float32")

    max_pop = prop_pop["prop_popularity_log"].max()
    if max_pop > 0:
        prop_pop["prop_popularity_log"] = (
            prop_pop["prop_popularity_log"] / max_pop
        ).astype("float32")

    prop_pop = prop_pop[
        [
            "prop_id",
            "prop_ctr",
            "prop_booking_rate",
            "prop_popularity_log"
        ]
    ]

    return prop_pop, global_ctr, global_booking_rate


def add_property_popularity(df, prop_pop, global_ctr, global_booking_rate):
    """
    Merge popularity features into df.
    Properties not seen in base data are filled with global values.
    """

    df = df.merge(prop_pop, on="prop_id", how="left")

    df["prop_ctr"] = df["prop_ctr"].fillna(global_ctr).astype("float32")
    df["prop_booking_rate"] = (
        df["prop_booking_rate"]
        .fillna(global_booking_rate)
        .astype("float32")
    )
    df["prop_popularity_log"] = (
        df["prop_popularity_log"]
        .fillna(0)
        .astype("float32")
    )

    return df


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

print("Train part before pop:", train_part.shape)
print("Validation part before pop:", val_part.shape)


# ============================================================
# 4. Add popularity features without validation leakage
# ============================================================

# Important:
# only use train_part to build popularity features for validation.
prop_pop_fold, fold_global_ctr, fold_global_booking_rate = build_property_popularity(
    train_part,
    alpha=10
)

train_part = add_property_popularity(
    train_part,
    prop_pop_fold,
    fold_global_ctr,
    fold_global_booking_rate
)

val_part = add_property_popularity(
    val_part,
    prop_pop_fold,
    fold_global_ctr,
    fold_global_booking_rate
)

print("Train part after pop:", train_part.shape)
print("Validation part after pop:", val_part.shape)

print("Popularity features added:")
print(["prop_ctr", "prop_booking_rate", "prop_popularity_log"])


# ============================================================
# 5. Feature columns
# ============================================================

drop_cols = ["srch_id", "prop_id", "relevance"]
feature_cols = [c for c in train_part.columns if c not in drop_cols]

print("Number of features:", len(feature_cols))

print("New popularity columns in features:")
for col in ["prop_ctr", "prop_booking_rate", "prop_popularity_log"]:
    print(col, col in feature_cols)


# ============================================================
# 6. Sort by query and create groups
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
# 7. Train LightGBM LambdaRank model
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
# 8. Validation NDCG@5
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

print("Validation NDCG@5:", val_ndcg5)


# ============================================================
# 9. Feature importance
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
# 10. Train final model on full training data
# ============================================================

# For final test prediction, it is okay to use full train
# to build popularity features.
prop_pop_full, full_global_ctr, full_global_booking_rate = build_property_popularity(
    train_fe,
    alpha=10
)

train_full = add_property_popularity(
    train_fe.copy(),
    prop_pop_full,
    full_global_ctr,
    full_global_booking_rate
)

train_full = train_full.sort_values("srch_id").reset_index(drop=True)

X_full = train_full[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(-999)
y_full = train_full["relevance"]

full_group = train_full.groupby("srch_id").size().to_numpy()

best_n_estimators = ranker.best_iteration_
if best_n_estimators is None or best_n_estimators <= 0:
    best_n_estimators = 300

print("Training final model with n_estimators =", best_n_estimators)

final_ranker = lgb.LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    n_estimators=best_n_estimators,
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
    group=full_group
)


# ============================================================
# 11. Predict test and create submission
# ============================================================

test_with_pop = add_property_popularity(
    test_fe.copy(),
    prop_pop_full,
    full_global_ctr,
    full_global_booking_rate
)

test_sorted = test_with_pop.sort_values("srch_id").reset_index(drop=True)

X_test = test_sorted[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(-999)

test_sorted["model_b_score"] = final_ranker.predict(X_test)

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