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

submission_path = OUTPUT_DIR / "submission_blend_model_b_c_rank.csv"


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


def add_rank_features(df):
    """
    Within each srch_id:
    higher model score should get better rank.
    rank 1 = best.
    Then convert to negative rank score because larger blend_score should be better.
    """
    df["model_b_rank"] = (
        df.groupby("srch_id")["model_b_score"]
        .rank(method="average", ascending=False)
    )

    df["model_c_rank"] = (
        df.groupby("srch_id")["model_c_score"]
        .rank(method="average", ascending=False)
    )

    return df


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

print("Validation data:", val.shape)

val = add_rank_features(val)


# ============================================================
# 2. Search best rank-blend weight
# ============================================================

results = []

for w_b in np.arange(0.50, 1.001, 0.01):
    w_c = 1.0 - w_b

    # lower rank is better, so use negative weighted rank as score
    val["blend_score"] = -(
        w_b * val["model_b_rank"]
        + w_c * val["model_c_rank"]
    )

    score = mean_ndcg_at_5(val)

    results.append({
        "w_b": w_b,
        "w_c": w_c,
        "val_ndcg5": score
    })

results_df = pd.DataFrame(results).sort_values("val_ndcg5", ascending=False)

print("Top rank-blend weights:")
print(results_df.head(15))

best = results_df.iloc[0]
best_w_b = float(best["w_b"])
best_w_c = float(best["w_c"])

print("Best w_b:", best_w_b)
print("Best w_c:", best_w_c)
print("Best validation NDCG@5:", best["val_ndcg5"])


# ============================================================
# 3. Load test scores and create rank-blend submission
# ============================================================

b_test = pd.read_csv(model_b_test_path)
c_test = pd.read_csv(model_c_test_path)

test = b_test.merge(
    c_test[["srch_id", "prop_id", "model_c_score"]],
    on=["srch_id", "prop_id"],
    how="inner"
)

print("Test data:", test.shape)

test = add_rank_features(test)

test["blend_score"] = -(
    best_w_b * test["model_b_rank"]
    + best_w_c * test["model_c_rank"]
)

submission = (
    test
    .sort_values(["srch_id", "blend_score"], ascending=[True, False])
    [["srch_id", "prop_id"]]
)

submission.to_csv(submission_path, index=False)

print("Saved rank-blend submission to:", submission_path)
print("Submission shape:", submission.shape)
print(submission.head(30))

print("Unique srch_id:", submission["srch_id"].nunique())
print("Missing values:")
print(submission.isna().sum())
print("Duplicated srch_id-prop_id:", submission.duplicated(["srch_id", "prop_id"]).sum())