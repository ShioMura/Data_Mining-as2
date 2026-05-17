from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score


# ============================================================
# 0. Paths
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs"

v2_val_path = OUTPUT_DIR / "model_b_val_scores_extra_v2_seed_ensemble.csv"
v2_test_path = OUTPUT_DIR / "model_b_test_scores_extra_v2_seed_ensemble.csv"

top70_val_path = OUTPUT_DIR / "model_b_val_scores_extra_v2_pruned_top70_seed_ensemble.csv"
top70_test_path = OUTPUT_DIR / "model_b_test_scores_extra_v2_pruned_top70_seed_ensemble.csv"

submission_path = OUTPUT_DIR / "submission_blend_v2_seed_top70_seed.csv"
blend_result_path = OUTPUT_DIR / "blend_v2_seed_top70_seed_results.csv"

print("v2 val exists:", v2_val_path.exists())
print("v2 test exists:", v2_test_path.exists())
print("top70 val exists:", top70_val_path.exists())
print("top70 test exists:", top70_test_path.exists())


# ============================================================
# 1. Helper functions
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


# ============================================================
# 2. Load validation scores
# ============================================================

v2_val = pd.read_csv(v2_val_path)
top70_val = pd.read_csv(top70_val_path)

# Standardize score column names
v2_val = v2_val.rename(columns={"ensemble_score": "v2_seed_score"})
top70_val = top70_val.rename(columns={"ensemble_score": "top70_seed_score"})

val = v2_val[["srch_id", "prop_id", "relevance", "v2_seed_score"]].merge(
    top70_val[["srch_id", "prop_id", "top70_seed_score"]],
    on=["srch_id", "prop_id"],
    how="inner"
)

print("Validation blend data:", val.shape)

val["v2_seed_score_norm"] = normalize_within_search(val, "v2_seed_score")
val["top70_seed_score_norm"] = normalize_within_search(val, "top70_seed_score")


# ============================================================
# 3. Search blend weights
# ============================================================

results = []

for w_v2 in np.arange(0.70, 1.001, 0.02):
    w_top70 = 1.0 - w_v2

    # raw score blend
    val["blend_score"] = (
        w_v2 * val["v2_seed_score"]
        + w_top70 * val["top70_seed_score"]
    )

    raw_ndcg = mean_ndcg_at_5(val)

    # normalized score blend
    val["blend_score"] = (
        w_v2 * val["v2_seed_score_norm"]
        + w_top70 * val["top70_seed_score_norm"]
    )

    norm_ndcg = mean_ndcg_at_5(val)

    results.append({
        "w_v2": w_v2,
        "w_top70": w_top70,
        "raw_ndcg5": raw_ndcg,
        "norm_ndcg5": norm_ndcg
    })

results_df = pd.DataFrame(results)

best_raw = results_df.sort_values("raw_ndcg5", ascending=False).iloc[0]
best_norm = results_df.sort_values("norm_ndcg5", ascending=False).iloc[0]

print("=" * 70)
print("Top raw blend weights:")
print(results_df.sort_values("raw_ndcg5", ascending=False).head(15))
print("=" * 70)
print("Top normalized blend weights:")
print(results_df.sort_values("norm_ndcg5", ascending=False).head(15))
print("=" * 70)

results_df.to_csv(blend_result_path, index=False)
print("Saved blend results to:", blend_result_path)

# Choose whichever validation is higher
if best_norm["norm_ndcg5"] > best_raw["raw_ndcg5"]:
    use_norm = True
    best_w_v2 = float(best_norm["w_v2"])
    best_w_top70 = float(best_norm["w_top70"])
    best_val = float(best_norm["norm_ndcg5"])
    print("Using normalized blend")
else:
    use_norm = False
    best_w_v2 = float(best_raw["w_v2"])
    best_w_top70 = float(best_raw["w_top70"])
    best_val = float(best_raw["raw_ndcg5"])
    print("Using raw blend")

print("Best w_v2:", best_w_v2)
print("Best w_top70:", best_w_top70)
print("Best validation NDCG@5:", best_val)


# ============================================================
# 4. Load test scores and create submission
# ============================================================

v2_test = pd.read_csv(v2_test_path)
top70_test = pd.read_csv(top70_test_path)

v2_test = v2_test.rename(columns={"ensemble_score": "v2_seed_score"})
top70_test = top70_test.rename(columns={"ensemble_score": "top70_seed_score"})

test = v2_test[["srch_id", "prop_id", "v2_seed_score"]].merge(
    top70_test[["srch_id", "prop_id", "top70_seed_score"]],
    on=["srch_id", "prop_id"],
    how="inner"
)

print("Test blend data:", test.shape)

if use_norm:
    test["v2_seed_score_norm"] = normalize_within_search(test, "v2_seed_score")
    test["top70_seed_score_norm"] = normalize_within_search(test, "top70_seed_score")

    test["blend_score"] = (
        best_w_v2 * test["v2_seed_score_norm"]
        + best_w_top70 * test["top70_seed_score_norm"]
    )
else:
    test["blend_score"] = (
        best_w_v2 * test["v2_seed_score"]
        + best_w_top70 * test["top70_seed_score"]
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