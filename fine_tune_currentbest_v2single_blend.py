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

v2_val_path = OUTPUT_DIR / "model_b_val_scores_extra_v2_best.csv"
v2_test_path = OUTPUT_DIR / "model_b_test_scores_extra_v2_best.csv"

result_path = OUTPUT_DIR / "fine_tune_currentbest_v2single_results.csv"
submission_path = OUTPUT_DIR / "submission_fine_tune_currentbest_v2single.csv"


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
    raise ValueError(f"No score column found. Columns = {df.columns.tolist()}")


print("weighted val exists:", weighted_val_path.exists())
print("top70 val exists:", top70_val_path.exists())
print("v2 val exists:", v2_val_path.exists())


# ============================================================
# 1. Build current best validation score
# current_best = 0.80 weighted_seed + 0.20 top70_seed
# ============================================================

weighted_val = pd.read_csv(weighted_val_path)
top70_val = pd.read_csv(top70_val_path)
v2_val = pd.read_csv(v2_val_path)

weighted_val = weighted_val.rename(columns={"blend_score": "weighted_seed_score"})
top70_val = top70_val.rename(columns={"ensemble_score": "top70_seed_score"})

v2_score_col = find_score_col(v2_val)
v2_val = v2_val.rename(columns={v2_score_col: "v2_single_score"})

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

val["current_best_score"] = (
    0.80 * val["weighted_seed_score"]
    + 0.20 * val["top70_seed_score"]
)

val["current_best_score_norm"] = normalize_within_search(val, "current_best_score")
val["v2_single_score_norm"] = normalize_within_search(val, "v2_single_score")

val["blend_score"] = val["current_best_score_norm"]
current_ndcg = mean_ndcg_at_5(val)

print("Current best norm-only NDCG@5:", current_ndcg)
print("Validation data:", val.shape)


# ============================================================
# 2. Fine search around 0.98 / 0.02
# ============================================================

results = []

for w_current in np.arange(0.950, 1.0001, 0.005):
    w_v2 = 1.0 - w_current

    print(
        "Testing:",
        "w_current =", round(w_current, 3),
        "w_v2 =", round(w_v2, 3)
    )

    val["blend_score"] = (
        w_current * val["current_best_score_norm"]
        + w_v2 * val["v2_single_score_norm"]
    )

    ndcg = mean_ndcg_at_5(val)

    print("NDCG@5:", ndcg)

    results.append({
        "w_current": w_current,
        "w_v2": w_v2,
        "val_ndcg5": ndcg
    })

results_df = pd.DataFrame(results).sort_values("val_ndcg5", ascending=False)
results_df.to_csv(result_path, index=False)

print("=" * 70)
print("Fine tune results:")
print(results_df)
print("=" * 70)

best = results_df.iloc[0]
best_w_current = float(best["w_current"])
best_w_v2 = float(best["w_v2"])
best_val = float(best["val_ndcg5"])

print("Best w_current:", best_w_current)
print("Best w_v2:", best_w_v2)
print("Best validation NDCG@5:", best_val)


# ============================================================
# 3. Apply to test
# ============================================================

weighted_test = pd.read_csv(weighted_test_path)
top70_test = pd.read_csv(top70_test_path)
v2_test = pd.read_csv(v2_test_path)

weighted_test = weighted_test.rename(columns={"blend_score": "weighted_seed_score"})
top70_test = top70_test.rename(columns={"ensemble_score": "top70_seed_score"})

v2_test_score_col = find_score_col(v2_test)
v2_test = v2_test.rename(columns={v2_test_score_col: "v2_single_score"})

test = weighted_test[["srch_id", "prop_id", "weighted_seed_score"]].merge(
    top70_test[["srch_id", "prop_id", "top70_seed_score"]],
    on=["srch_id", "prop_id"],
    how="inner"
)

test = test.merge(
    v2_test[["srch_id", "prop_id", "v2_single_score"]],
    on=["srch_id", "prop_id"],
    how="inner"
)

test["current_best_score"] = (
    0.80 * test["weighted_seed_score"]
    + 0.20 * test["top70_seed_score"]
)

test["current_best_score_norm"] = normalize_within_search(test, "current_best_score")
test["v2_single_score_norm"] = normalize_within_search(test, "v2_single_score")

test["blend_score"] = (
    best_w_current * test["current_best_score_norm"]
    + best_w_v2 * test["v2_single_score_norm"]
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