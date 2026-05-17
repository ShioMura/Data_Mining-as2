from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score


ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs"

# Current best components
weighted_val_path = OUTPUT_DIR / "model_b_val_scores_extra_v2_weighted_seed_ensemble.csv"
weighted_test_path = OUTPUT_DIR / "model_b_test_scores_extra_v2_weighted_seed_ensemble.csv"

top70_val_path = OUTPUT_DIR / "model_b_val_scores_extra_v2_pruned_top70_seed_ensemble.csv"
top70_test_path = OUTPUT_DIR / "model_b_test_scores_extra_v2_pruned_top70_seed_ensemble.csv"

# Candidate extra models
candidate_files = {
    "stronger_tuned": {
        "val": OUTPUT_DIR / "model_b_val_scores_extra_v2_tuned_stronger.csv",
        "test": OUTPUT_DIR / "model_b_test_scores_extra_v2_tuned_stronger.csv",
    },
    "conservative_tuned": {
        "val": OUTPUT_DIR / "model_b_val_scores_extra_v2_tuned.csv",
        "test": OUTPUT_DIR / "model_b_test_scores_extra_v2_tuned.csv",
    },
    "v2_single": {
        "val": OUTPUT_DIR / "model_b_val_scores_extra_v2_best.csv",
        "test": OUTPUT_DIR / "model_b_test_scores_extra_v2_best.csv",
    },
}

result_path = OUTPUT_DIR / "blend_currentbest_with_tuned_results.csv"
submission_path = OUTPUT_DIR / "submission_blend_currentbest_with_tuned.csv"


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
    possible_cols = [
        "blend_score",
        "ensemble_score",
        "model_b_score",
        "model_score",
    ]

    for col in possible_cols:
        if col in df.columns:
            return col

    raise ValueError(f"No known score column found. Columns = {df.columns.tolist()}")


print("weighted val exists:", weighted_val_path.exists())
print("weighted test exists:", weighted_test_path.exists())
print("top70 val exists:", top70_val_path.exists())
print("top70 test exists:", top70_test_path.exists())


# ============================================================
# 1. Build current best validation score
# current best = 0.80 weighted_seed + 0.20 top70_seed
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

val["current_best_score"] = (
    0.80 * val["weighted_seed_score"]
    + 0.20 * val["top70_seed_score"]
)

val["blend_score"] = val["current_best_score"]
current_best_ndcg = mean_ndcg_at_5(val)

print("Current best reconstructed NDCG@5:", current_best_ndcg)


# ============================================================
# 2. Load candidate validation scores if available
# ============================================================

available_candidates = []

for name, paths in candidate_files.items():
    val_path = paths["val"]
    test_path = paths["test"]

    print(name, "val exists:", val_path.exists(), "test exists:", test_path.exists())

    if not val_path.exists() or not test_path.exists():
        print("Skipping", name)
        continue

    cand_val = pd.read_csv(val_path)
    score_col = find_score_col(cand_val)

    cand_val = cand_val.rename(columns={score_col: f"{name}_score"})

    val = val.merge(
        cand_val[["srch_id", "prop_id", f"{name}_score"]],
        on=["srch_id", "prop_id"],
        how="inner"
    )

    available_candidates.append(name)

print("Available candidates:", available_candidates)
print("Validation blend data:", val.shape)

if not available_candidates:
    raise RuntimeError("No candidate score files found.")


# ============================================================
# 3. Search current_best + one candidate at a time
# ============================================================

results = []

for name in available_candidates:
    cand_col = f"{name}_score"

    for w_current in np.arange(0.80, 1.001, 0.02):
        w_cand = 1.0 - w_current

        print(
            "Testing raw:",
            "candidate =", name,
            "w_current =", round(w_current, 2),
            "w_cand =", round(w_cand, 2)
        )

        val["blend_score"] = (
            w_current * val["current_best_score"]
            + w_cand * val[cand_col]
        )

        ndcg = mean_ndcg_at_5(val)

        print("NDCG@5:", ndcg)

        results.append({
            "candidate": name,
            "mode": "raw",
            "w_current": w_current,
            "w_candidate": w_cand,
            "val_ndcg5": ndcg
        })


# ============================================================
# 4. Optional normalized blend, one candidate at a time
# ============================================================

val["current_best_score_norm"] = normalize_within_search(val, "current_best_score")

for name in available_candidates:
    cand_col = f"{name}_score"
    cand_norm_col = f"{name}_score_norm"

    val[cand_norm_col] = normalize_within_search(val, cand_col)

    for w_current in np.arange(0.80, 1.001, 0.02):
        w_cand = 1.0 - w_current

        print(
            "Testing norm:",
            "candidate =", name,
            "w_current =", round(w_current, 2),
            "w_cand =", round(w_cand, 2)
        )

        val["blend_score"] = (
            w_current * val["current_best_score_norm"]
            + w_cand * val[cand_norm_col]
        )

        ndcg = mean_ndcg_at_5(val)

        print("NDCG@5:", ndcg)

        results.append({
            "candidate": name,
            "mode": "norm",
            "w_current": w_current,
            "w_candidate": w_cand,
            "val_ndcg5": ndcg
        })


# ============================================================
# 5. Choose best
# ============================================================

results_df = pd.DataFrame(results).sort_values("val_ndcg5", ascending=False)
results_df.to_csv(result_path, index=False)

print("=" * 70)
print("Blend current best with tuned results:")
print(results_df.head(30))
print("=" * 70)

best = results_df.iloc[0]

best_candidate = str(best["candidate"])
best_mode = str(best["mode"])
best_w_current = float(best["w_current"])
best_w_candidate = float(best["w_candidate"])
best_val = float(best["val_ndcg5"])

print("Best candidate:", best_candidate)
print("Best mode:", best_mode)
print("Best w_current:", best_w_current)
print("Best w_candidate:", best_w_candidate)
print("Best validation NDCG@5:", best_val)


# ============================================================
# 6. Build current best test score
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

test["current_best_score"] = (
    0.80 * test["weighted_seed_score"]
    + 0.20 * test["top70_seed_score"]
)


# ============================================================
# 7. Load best candidate test score
# ============================================================

best_test_path = candidate_files[best_candidate]["test"]
cand_test = pd.read_csv(best_test_path)
score_col = find_score_col(cand_test)

cand_test = cand_test.rename(columns={score_col: f"{best_candidate}_score"})

test = test.merge(
    cand_test[["srch_id", "prop_id", f"{best_candidate}_score"]],
    on=["srch_id", "prop_id"],
    how="inner"
)

print("Test blend data:", test.shape)


# ============================================================
# 8. Apply best blend to test
# ============================================================

if best_mode == "norm":
    test["current_best_score_norm"] = normalize_within_search(test, "current_best_score")
    test[f"{best_candidate}_score_norm"] = normalize_within_search(
        test,
        f"{best_candidate}_score"
    )

    test["blend_score"] = (
        best_w_current * test["current_best_score_norm"]
        + best_w_candidate * test[f"{best_candidate}_score_norm"]
    )
else:
    test["blend_score"] = (
        best_w_current * test["current_best_score"]
        + best_w_candidate * test[f"{best_candidate}_score"]
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