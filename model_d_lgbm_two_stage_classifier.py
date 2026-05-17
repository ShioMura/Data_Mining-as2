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

submission_path = OUTPUT_DIR / "submission_model_d_two_stage_classifier.csv"
importance_booking_path = OUTPUT_DIR / "model_d_booking_importance.csv"
importance_click_path = OUTPUT_DIR / "model_d_click_importance.csv"

val_score_path = OUTPUT_DIR / "model_d_val_scores.csv"
test_score_path = OUTPUT_DIR / "model_d_test_scores.csv"

print("ROOT_DIR:", ROOT_DIR)
print("DATA_DIR:", DATA_DIR)
print("Train feature path exists:", train_path.exists())
print("Test feature path exists:", test_path.exists())


# ============================================================
# 1. Load features
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
# 4. Prepare X/y
# ============================================================

X_train = train_part[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(-999)
X_val = val_part[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(-999)

# Two targets
# booked target: only relevance 5 is positive
# clicked target: relevance 1 or 5 is positive
y_train_booking = (train_part["relevance"] == 5).astype(int)
y_val_booking = (val_part["relevance"] == 5).astype(int)

y_train_click = (train_part["relevance"] >= 1).astype(int)
y_val_click = (val_part["relevance"] >= 1).astype(int)

print("Booking positive rate train:", y_train_booking.mean())
print("Click/booking positive rate train:", y_train_click.mean())


# ============================================================
# 5. Train booking classifier
# ============================================================

booking_clf = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=1500,
    learning_rate=0.03,
    num_leaves=95,
    min_child_samples=100,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=1.0,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

booking_clf.fit(
    X_train,
    y_train_booking,
    eval_set=[(X_val, y_val_booking)],
    eval_metric="binary_logloss",
    callbacks=[
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(period=100)
    ]
)


# ============================================================
# 6. Train click classifier
# ============================================================

click_clf = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=1500,
    learning_rate=0.03,
    num_leaves=95,
    min_child_samples=100,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=1.0,
    class_weight="balanced",
    random_state=43,
    n_jobs=-1
)

click_clf.fit(
    X_train,
    y_train_click,
    eval_set=[(X_val, y_val_click)],
    eval_metric="binary_logloss",
    callbacks=[
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(period=100)
    ]
)


# ============================================================
# 7. Validation scores and search best booking/click weight
# ============================================================

val_part["booking_prob"] = booking_clf.predict_proba(
    X_val,
    num_iteration=booking_clf.best_iteration_
)[:, 1]

val_part["click_prob"] = click_clf.predict_proba(
    X_val,
    num_iteration=click_clf.best_iteration_
)[:, 1]


def mean_ndcg_at_5(df, label_col="relevance", score_col="model_d_score"):
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


results = []

for w_booking in np.arange(0.50, 1.01, 0.05):
    w_click = 1.0 - w_booking

    val_part["model_d_score"] = (
        w_booking * val_part["booking_prob"]
        + w_click * val_part["click_prob"]
    )

    score = mean_ndcg_at_5(val_part)

    results.append({
        "w_booking": w_booking,
        "w_click": w_click,
        "val_ndcg5": score
    })

results_df = pd.DataFrame(results).sort_values("val_ndcg5", ascending=False)

print("Top Model D internal weights:")
print(results_df.head(10))

best = results_df.iloc[0]
best_w_booking = float(best["w_booking"])
best_w_click = float(best["w_click"])

print("Best w_booking:", best_w_booking)
print("Best w_click:", best_w_click)
print("Best Model D validation NDCG@5:", best["val_ndcg5"])

val_part["model_d_score"] = (
    best_w_booking * val_part["booking_prob"]
    + best_w_click * val_part["click_prob"]
)

val_scores = val_part[
    ["srch_id", "prop_id", "relevance", "booking_prob", "click_prob", "model_d_score"]
].copy()

val_scores.to_csv(val_score_path, index=False)
print("Saved Model D validation scores to:", val_score_path)


# ============================================================
# 8. Feature importance
# ============================================================

booking_importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": booking_clf.feature_importances_
}).sort_values("importance", ascending=False)

click_importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": click_clf.feature_importances_
}).sort_values("importance", ascending=False)

booking_importance.to_csv(importance_booking_path, index=False)
click_importance.to_csv(importance_click_path, index=False)

print("Top 20 booking features:")
print(booking_importance.head(20))

print("Top 20 click features:")
print(click_importance.head(20))


# ============================================================
# 9. Train final classifiers on full training data
# ============================================================

X_full = train_fe[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(-999)

y_full_booking = (train_fe["relevance"] == 5).astype(int)
y_full_click = (train_fe["relevance"] >= 1).astype(int)

best_booking_iter = booking_clf.best_iteration_
if best_booking_iter is None or best_booking_iter <= 0:
    best_booking_iter = 500

best_click_iter = click_clf.best_iteration_
if best_click_iter is None or best_click_iter <= 0:
    best_click_iter = 500

print("Final booking n_estimators:", best_booking_iter)
print("Final click n_estimators:", best_click_iter)

final_booking_clf = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=best_booking_iter,
    learning_rate=0.03,
    num_leaves=95,
    min_child_samples=100,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=1.0,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

final_click_clf = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=best_click_iter,
    learning_rate=0.03,
    num_leaves=95,
    min_child_samples=100,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=1.0,
    class_weight="balanced",
    random_state=43,
    n_jobs=-1
)

final_booking_clf.fit(X_full, y_full_booking)
final_click_clf.fit(X_full, y_full_click)


# ============================================================
# 10. Predict test and create submission
# ============================================================

test_scored = test_fe.copy()
X_test = test_scored[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(-999)

test_scored["booking_prob"] = final_booking_clf.predict_proba(X_test)[:, 1]
test_scored["click_prob"] = final_click_clf.predict_proba(X_test)[:, 1]

test_scored["model_d_score"] = (
    best_w_booking * test_scored["booking_prob"]
    + best_w_click * test_scored["click_prob"]
)

test_scores = test_scored[
    ["srch_id", "prop_id", "booking_prob", "click_prob", "model_d_score"]
].copy()

test_scores.to_csv(test_score_path, index=False)
print("Saved Model D test scores to:", test_score_path)

submission = (
    test_scored
    .sort_values(["srch_id", "model_d_score"], ascending=[True, False])
    [["srch_id", "prop_id"]]
)

submission.to_csv(submission_path, index=False)

print("Saved Model D submission to:", submission_path)
print("Submission shape:", submission.shape)
print(submission.head(30))

print("Unique srch_id:", submission["srch_id"].nunique())
print("Missing values:")
print(submission.isna().sum())
print("Duplicated srch_id-prop_id:", submission.duplicated(["srch_id", "prop_id"]).sum())