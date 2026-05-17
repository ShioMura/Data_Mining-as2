from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score


ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs"

weighted_val_path = OUTPUT_DIR / "model_b_val_scores_extra_v2_weighted_seed_ensemble.csv"
weighted_test_path = OUTPUT_DIR / "model_b_test_scores_extra_v2_weighted_seed_ensemble.csv"

top70_val_path = OUTPUT_DIR / "model_b_val_scores_extra_v2_pruned_top70_seed_ensemble.csv"
top70_test_path = OUTPUT_DIR / "model_b_test_scores_extra_v2_pruned_top70_seed_ensemble.csv"

result_path = OUTPUT_DIR / "blend_weighted_seed_top70_results.csv"
submission_path = OUTPUT_DIR / "submission_blend_weighted_seed_top70.csv"

print("weighted val exists:", weighted_val_path.exists())
print("weighted test exists:", weighted_test_path.exists())
print("top70 val exists:", top70_val_path.exists())
print("top70 test exists:", top70_test_path.exists())


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
# 1. Load validation scores
# ============================================================

weighted_val = pd.read_csv(weighted_val_path)
top70_val = pd.read_csv(top70_val_path)

weighted_val = weighted_val.rename(columns={"blend_score": "weighted_seed_score"})
top70_val = top70_val.rename(columns={"ensemble_score": "top70_seed_score"})

val = weighted_val[["srch_id", "prop_id", "relevance", "weighted_seed_score"]].merge(
    top70_val[["srch_id", "prop_id", "top70_seed_score"]],
    on=["srch_id", "prop_id"],
    how="inner"
)

print("Validation blend data:", val.shape)


# ============================================================
# 2. Search small blend weights
# ============================================================

results = []

# weighted seed is stronger, so search mostly near 1.0
for w_weighted in np.arange(0.80, 1.001, 0.02):
    w_top70 = 1.0 - w_weighted

    print(
        "Testing:",
        "w_weighted =", round(w_weighted, 2),
        "w_top70 =", round(w_top70, 2)
    )

    val["blend_score"] = (
        w_weighted * val["weighted_seed_score"]
        + w_top70 * val["top70_seed_score"]
    )

    ndcg = mean_ndcg_at_5(val)

    print("NDCG@5:", ndcg)

    results.append({
        "w_weighted": w_weighted,
        "w_top70": w_top70,
        "val_ndcg5": ndcg
    })

results_df = pd.DataFrame(results).sort_values("val_ndcg5", ascending=False)
results_df.to_csv(result_path, index=False)

print("=" * 70)
print("Blend results:")
print(results_df.head(20))
print("=" * 70)

best = results_df.iloc[0]

best_w_weighted = float(best["w_weighted"])
best_w_top70 = float(best["w_top70"])
best_val = float(best["val_ndcg5"])

print("Best w_weighted:", best_w_weighted)
print("Best w_top70:", best_w_top70)
print("Best validation NDCG@5:", best_val)


# ============================================================
# 3. Apply best weights to test
# ============================================================

weighted_test = pd.read_csv(weighted_test_path)
top70_test = pd.read_csv(top70_test_path)

weighted_test = weighted_test.rename(columns={"blend_score": "weighted_seed_score"})
top70_test = top70_test.rename(columns={"ensemble_score": "top70_seed_score"})

test = weighted_test[["srch_id", "prop_id", "weighted_seed_score"]].merge(
    top70_test[["srch_id", "prop_id", "top70_seed_score"]],
    on=["srch_id", "prop_id"],
    how="inner"
)

print("Test blend data:", test.shape)

test["blend_score"] = (
    best_w_weighted * test["weighted_seed_score"]
    + best_w_top70 * test["top70_seed_score"]
)

submission = (
    test
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