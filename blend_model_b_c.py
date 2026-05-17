from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score


ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs"

model_b_val_path = OUTPUT_DIR / "model_b_val_scores.csv"
model_c_val_path = OUTPUT_DIR / "model_c_val_scores.csv"

model_b_test_path = OUTPUT_DIR / "model_b_test_scores.csv"
model_c_test_path = OUTPUT_DIR / "model_c_test_scores.csv"

submission_path = OUTPUT_DIR / "submission_blend_model_b_c_finetuned.csv"


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
    """
    Normalize scores within each srch_id.
    This is important because Model B and Model C scores may have different scales.
    """
    g = df.groupby("srch_id")[score_col]

    mean = g.transform("mean")
    std = g.transform("std").replace(0, 1)

    return ((df[score_col] - mean) / std).fillna(0)


# ============================================================
# 1. Load validation scores
# ============================================================

b_val = pd.read_csv(model_b_val_path)
c_val = pd.read_csv(model_c_val_path)

val = b_val.merge(
    c_val[["srch_id", "prop_id", "model_c_score"]],
    on=["srch_id", "prop_id"],
    how="inner"
)

print("Validation blend data:", val.shape)

val["model_b_score_norm"] = normalize_within_search(val, "model_b_score")
val["model_c_score_norm"] = normalize_within_search(val, "model_c_score")


# ============================================================
# 2. Search best blend weight on validation
# ============================================================

results = []

for w_b in np.arange(0.80, 0.901, 0.01):
    w_c = 1.0 - w_b

    val["blend_score"] = (
        w_b * val["model_b_score_norm"]
        + w_c * val["model_c_score_norm"]
    )

    score = mean_ndcg_at_5(val)

    results.append({
        "w_b": w_b,
        "w_c": w_c,
        "val_ndcg5": score
    })

results_df = pd.DataFrame(results).sort_values("val_ndcg5", ascending=False)

print("Top blend weights:")
print(results_df.head(10))

best = results_df.iloc[0]
best_w_b = float(best["w_b"])
best_w_c = float(best["w_c"])

print("Best w_b:", best_w_b)
print("Best w_c:", best_w_c)
print("Best validation NDCG@5:", best["val_ndcg5"])


# ============================================================
# 3. Load test scores and create blended submission
# ============================================================

b_test = pd.read_csv(model_b_test_path)
c_test = pd.read_csv(model_c_test_path)

test = b_test.merge(
    c_test[["srch_id", "prop_id", "model_c_score"]],
    on=["srch_id", "prop_id"],
    how="inner"
)

print("Test blend data:", test.shape)

test["model_b_score_norm"] = normalize_within_search(test, "model_b_score")
test["model_c_score_norm"] = normalize_within_search(test, "model_c_score")

test["blend_score"] = (
    best_w_b * test["model_b_score_norm"]
    + best_w_c * test["model_c_score_norm"]
)

submission = (
    test
    .sort_values(["srch_id", "blend_score"], ascending=[True, False])
    [["srch_id", "prop_id"]]
)

submission.to_csv(submission_path, index=False)

print("Saved blended submission to:", submission_path)
print("Submission shape:", submission.shape)
print(submission.head(30))

print("Unique srch_id:", submission["srch_id"].nunique())
print("Missing values:")
print(submission.isna().sum())
print("Duplicated srch_id-prop_id:", submission.duplicated(["srch_id", "prop_id"]).sum())