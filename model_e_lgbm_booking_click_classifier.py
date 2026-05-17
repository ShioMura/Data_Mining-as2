from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score, roc_auc_score

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

val_score_path = OUTPUT_DIR / "model_e_val_scores_booking_click_classifier.csv"
test_score_path = OUTPUT_DIR / "model_e_test_scores_booking_click_classifier.csv"
result_path = OUTPUT_DIR / "model_e_booking_click_weight_results.csv"
submission_path = OUTPUT_DIR / "submission_model_e_booking_click_classifier.csv"

print("Train path exists:", train_path.exists())
print("Test path exists:", test_path.exists())


# ============================================================
# 1. Load v2 features
# ============================================================

train_fe = pd.read_parquet(train_path)
test_fe = pd.read_parquet(test_path)

print("Train features:", train_fe.shape)
print("Test features:", test_fe.shape)


# ============================================================
# 2. Create binary targets from relevance
# relevance: 0 = none, 1 = click, 5 = booking
# ============================================================

train_fe["target_booking"] = (train_fe["relevance"] == 5).astype("int8")
train_fe["target_click"] = (train_fe["relevance"] >= 1).astype("int8")

print("Booking positive rate:", train_fe["target_booking"].mean())
print("Click positive rate:", train_fe["target_click"].mean())


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

drop_cols = [
    "srch_id",
    "prop_id",
    "relevance",
    "target_booking",
    "target_click",
]

feature_cols = [c for c in train_fe.columns if c not in drop_cols]

print("Number of features:", len(feature_cols))


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

X_test = (
    test_fe[feature_cols]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(-999)
)

y_booking_train = train_part["target_booking"]
y_booking_val = val_part["target_booking"]

y_click_train = train_part["target_click"]
y_click_val = val_part["target_click"]


# ============================================================
# 5. Helper metric
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


def get_scale_pos_weight(y):
    pos = float(np.sum(y == 1))
    neg = float(np.sum(y == 0))
    return neg / max(pos, 1.0)


booking_spw = get_scale_pos_weight(y_booking_train)
click_spw = get_scale_pos_weight(y_click_train)

print("Booking scale_pos_weight:", booking_spw)
print("Click scale_pos_weight:", click_spw)


# ============================================================
# 6. Train booking classifier
# ============================================================

booking_clf = lgb.LGBMClassifier(
    objective="binary",
    metric="auc",
    n_estimators=3000,
    learning_rate=0.03,
    num_leaves=95,
    min_child_samples=100,
    subsample=0.85,
    colsample_bytree=0.85,
    scale_pos_weight=booking_spw,
    random_state=42,
    n_jobs=-1,
)

print("=" * 70)
print("Training booking classifier")
print("=" * 70)

booking_clf.fit(
    X_train,
    y_booking_train,
    eval_set=[(X_val, y_booking_val)],
    eval_metric="auc",
    callbacks=[
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(period=100),
    ],
)

booking_best_iter = booking_clf.best_iteration_
if booking_best_iter is None or booking_best_iter <= 0:
    booking_best_iter = 1000

val_part["booking_score"] = booking_clf.predict_proba(
    X_val,
    num_iteration=booking_best_iter,
)[:, 1]

booking_auc = roc_auc_score(y_booking_val, val_part["booking_score"])

print("Booking best iteration:", booking_best_iter)
print("Booking validation AUC:", booking_auc)


# ============================================================
# 7. Train click classifier
# ============================================================

click_clf = lgb.LGBMClassifier(
    objective="binary",
    metric="auc",
    n_estimators=3000,
    learning_rate=0.03,
    num_leaves=95,
    min_child_samples=100,
    subsample=0.85,
    colsample_bytree=0.85,
    scale_pos_weight=click_spw,
    random_state=42,
    n_jobs=-1,
)

print("=" * 70)
print("Training click classifier")
print("=" * 70)

click_clf.fit(
    X_train,
    y_click_train,
    eval_set=[(X_val, y_click_val)],
    eval_metric="auc",
    callbacks=[
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(period=100),
    ],
)

click_best_iter = click_clf.best_iteration_
if click_best_iter is None or click_best_iter <= 0:
    click_best_iter = 1000

val_part["click_score"] = click_clf.predict_proba(
    X_val,
    num_iteration=click_best_iter,
)[:, 1]

click_auc = roc_auc_score(y_click_val, val_part["click_score"])

print("Click best iteration:", click_best_iter)
print("Click validation AUC:", click_auc)


# ============================================================
# 8. Search booking/click score weights
# ============================================================

results = []

booking_weights = [
    1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20
]

for w_booking in booking_weights:
    w_click = 1.0

    val_part["model_score"] = (
        w_booking * val_part["booking_score"]
        + w_click * val_part["click_score"]
    )

    ndcg = mean_ndcg_at_5(val_part)

    print(
        "Testing weights:",
        "booking =", w_booking,
        "click =", w_click,
        "NDCG@5 =", ndcg,
    )

    results.append({
        "w_booking": w_booking,
        "w_click": w_click,
        "val_ndcg5": ndcg,
    })

results_df = pd.DataFrame(results).sort_values("val_ndcg5", ascending=False)
results_df.to_csv(result_path, index=False)

print("=" * 70)
print("Booking/click classifier weight results:")
print(results_df)
print("=" * 70)

best = results_df.iloc[0]
best_w_booking = float(best["w_booking"])
best_w_click = float(best["w_click"])
best_val = float(best["val_ndcg5"])

print("Best w_booking:", best_w_booking)
print("Best w_click:", best_w_click)
print("Best validation NDCG@5:", best_val)


val_part["model_score"] = (
    best_w_booking * val_part["booking_score"]
    + best_w_click * val_part["click_score"]
)

val_scores = val_part[
    ["srch_id", "prop_id", "relevance", "booking_score", "click_score", "model_score"]
].copy()

val_scores.to_csv(val_score_path, index=False)

print("Saved validation scores to:", val_score_path)


# ============================================================
# 9. Train final classifiers on full train
# ============================================================

train_full = train_fe.copy()

X_full = (
    train_full[feature_cols]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(-999)
)

y_booking_full = train_full["target_booking"]
y_click_full = train_full["target_click"]

booking_spw_full = get_scale_pos_weight(y_booking_full)
click_spw_full = get_scale_pos_weight(y_click_full)

print("Full booking scale_pos_weight:", booking_spw_full)
print("Full click scale_pos_weight:", click_spw_full)


print("=" * 70)
print("Training final booking classifier")
print("n_estimators =", booking_best_iter)
print("=" * 70)

final_booking_clf = lgb.LGBMClassifier(
    objective="binary",
    metric="auc",
    n_estimators=booking_best_iter,
    learning_rate=0.03,
    num_leaves=95,
    min_child_samples=100,
    subsample=0.85,
    colsample_bytree=0.85,
    scale_pos_weight=booking_spw_full,
    random_state=42,
    n_jobs=-1,
)

final_booking_clf.fit(X_full, y_booking_full)


print("=" * 70)
print("Training final click classifier")
print("n_estimators =", click_best_iter)
print("=" * 70)

final_click_clf = lgb.LGBMClassifier(
    objective="binary",
    metric="auc",
    n_estimators=click_best_iter,
    learning_rate=0.03,
    num_leaves=95,
    min_child_samples=100,
    subsample=0.85,
    colsample_bytree=0.85,
    scale_pos_weight=click_spw_full,
    random_state=42,
    n_jobs=-1,
)

final_click_clf.fit(X_full, y_click_full)


# ============================================================
# 10. Predict test and create submission
# ============================================================

test_scores = test_fe[["srch_id", "prop_id"]].copy()

test_scores["booking_score"] = final_booking_clf.predict_proba(X_test)[:, 1]
test_scores["click_score"] = final_click_clf.predict_proba(X_test)[:, 1]

test_scores["model_score"] = (
    best_w_booking * test_scores["booking_score"]
    + best_w_click * test_scores["click_score"]
)

test_scores.to_csv(test_score_path, index=False)

submission = (
    test_scores
    .sort_values(["srch_id", "model_score"], ascending=[True, False])
    [["srch_id", "prop_id"]]
)

submission.to_csv(submission_path, index=False)

print("Saved test scores to:", test_score_path)
print("Saved submission to:", submission_path)

print("Submission shape:", submission.shape)
print(submission.head(30))

print("Unique srch_id:", submission["srch_id"].nunique())
print("Missing values:")
print(submission.isna().sum())
print("Duplicated srch_id-prop_id:", submission.duplicated(["srch_id", "prop_id"]).sum())