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

# Prefer seed ensemble importance if available
importance_candidates = [
    OUTPUT_DIR / "model_b_feature_importance_extra_v2_seed_ensemble_best.csv",
    OUTPUT_DIR / "model_b_feature_importance_extra_v2_seed_ensemble.csv",
    OUTPUT_DIR / "model_b_feature_importance_extra_v2_best.csv",
    OUTPUT_DIR / "model_b_feature_importance.csv",
]

importance_path = None
for p in importance_candidates:
    if p.exists():
        importance_path = p
        break

if importance_path is None:
    raise FileNotFoundError("No feature importance file found.")

print("ROOT_DIR:", ROOT_DIR)
print("DATA_DIR:", DATA_DIR)
print("Train feature path exists:", train_path.exists())
print("Test feature path exists:", test_path.exists())
print("Using importance file:", importance_path)


# ============================================================
# 1. Load data
# ============================================================

train_fe = pd.read_parquet(train_path)
test_fe = pd.read_parquet(test_path)
importance_df = pd.read_csv(importance_path)

print("Train features:", train_fe.shape)
print("Test features:", test_fe.shape)
print("Importance:", importance_df.shape)


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
# 3. Feature ranking
# ============================================================

drop_cols = ["srch_id", "prop_id", "relevance"]
all_feature_cols = [c for c in train_fe.columns if c not in drop_cols]

importance_df = importance_df[importance_df["feature"].isin(all_feature_cols)].copy()
importance_df = importance_df.sort_values("importance", ascending=False)

ranked_features = importance_df["feature"].tolist()

# Add any missing features at the end, just in case
for col in all_feature_cols:
    if col not in ranked_features:
        ranked_features.append(col)

print("Total available features:", len(all_feature_cols))
print("Ranked features:", len(ranked_features))
print("Top 20 ranked features:")
print(ranked_features[:20])


# ============================================================
# 4. Metric
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
# 5. Sort and groups
# ============================================================

train_part = train_part.sort_values("srch_id").reset_index(drop=True)
val_part = val_part.sort_values("srch_id").reset_index(drop=True)

y_train = train_part["relevance"]
y_val = val_part["relevance"]

train_group = train_part.groupby("srch_id").size().to_numpy()
val_group = val_part.groupby("srch_id").size().to_numpy()

print("Train group rows check:", train_group.sum(), len(train_part))
print("Val group rows check:", val_group.sum(), len(val_part))


# ============================================================
# 6. Pruning search
# ============================================================

top_k_list = [55, 60, 65, 70, 75, 80, 85, 90, 91]

results = []

for top_k in top_k_list:
    feature_cols = ranked_features[:top_k]

    print("=" * 70)
    print("Training pruned model with top_k =", top_k)
    print("=" * 70)

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

    val_part["model_score"] = ranker.predict(
        X_val,
        num_iteration=best_iter
    )

    val_ndcg = mean_ndcg_at_5(val_part)

    print("top_k:", top_k)
    print("best_iter:", best_iter)
    print("Validation NDCG@5:", val_ndcg)

    results.append({
        "top_k": top_k,
        "best_iter": best_iter,
        "val_ndcg5": val_ndcg,
    })


# ============================================================
# 7. Save search result
# ============================================================

results_df = pd.DataFrame(results).sort_values("val_ndcg5", ascending=False)

result_path = OUTPUT_DIR / "pruning_search_extra_v2_results.csv"
results_df.to_csv(result_path, index=False)

print("=" * 70)
print("PRUNING SEARCH RESULTS")
print(results_df)
print("Saved pruning results to:", result_path)
print("=" * 70)


# ============================================================
# 8. Train final best pruned model
# ============================================================

best = results_df.iloc[0]
best_top_k = int(best["top_k"])
best_iter = int(best["best_iter"])
best_val = float(best["val_ndcg5"])

print("Best top_k:", best_top_k)
print("Best iter:", best_iter)
print("Best validation NDCG@5:", best_val)

best_features = ranked_features[:best_top_k]

selected_features_path = OUTPUT_DIR / f"selected_features_extra_v2_top{best_top_k}.csv"
pd.DataFrame({"feature": best_features}).to_csv(selected_features_path, index=False)
print("Saved selected features to:", selected_features_path)


# Only train final submission for best top_k
train_full = train_fe.sort_values("srch_id").reset_index(drop=True)
test_sorted = test_fe.sort_values("srch_id").reset_index(drop=True)

X_full = (
    train_full[best_features]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(-999)
)

y_full = train_full["relevance"]
full_group = train_full.groupby("srch_id").size().to_numpy()

X_test = (
    test_sorted[best_features]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(-999)
)

print("Training final pruned model...")
print("top_k =", best_top_k)
print("n_estimators =", best_iter)

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
    group=full_group
)

test_sorted["model_score"] = final_ranker.predict(X_test)

test_score_path = OUTPUT_DIR / f"model_b_test_scores_extra_v2_pruned_top{best_top_k}.csv"
test_sorted[["srch_id", "prop_id", "model_score"]].to_csv(test_score_path, index=False)

submission = (
    test_sorted
    .sort_values(["srch_id", "model_score"], ascending=[True, False])
    [["srch_id", "prop_id"]]
)

submission_path = OUTPUT_DIR / f"submission_model_b_lgbm_ranker_extra_v2_pruned_top{best_top_k}.csv"
submission.to_csv(submission_path, index=False)

print("Saved test scores to:", test_score_path)
print("Saved submission to:", submission_path)
print("Submission shape:", submission.shape)
print(submission.head(30))

print("Unique srch_id:", submission["srch_id"].nunique())
print("Missing values:")
print(submission.isna().sum())
print("Duplicated srch_id-prop_id:", submission.duplicated(["srch_id", "prop_id"]).sum())