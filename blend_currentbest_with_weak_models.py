from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score


ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs"

# Base current best components
weighted_val_path = OUTPUT_DIR / "model_b_val_scores_extra_v2_weighted_seed_ensemble.csv"
top70_val_path = OUTPUT_DIR / "model_b_val_scores_extra_v2_pruned_top70_seed_ensemble.csv"
v2_val_path = OUTPUT_DIR / "model_b_val_scores_extra_v2_best.csv"
f_multiseed_val_path = OUTPUT_DIR / "model_f_click_multiseed_best_val_scores.csv"

# Weak model validation scores
candidate_files = {
    "label_gain": OUTPUT_DIR / "model_b_val_scores_extra_v2_label_gain.csv",
    "v4_search": OUTPUT_DIR / "model_b_val_scores_extra_v4_search_competition.csv",
    "v5_competitor": OUTPUT_DIR / "model_b_val_scores_extra_v5_competitor.csv",
    "random_weighted": OUTPUT_DIR / "model_b_val_scores_extra_v2_random_weighted.csv",
}

result_path = OUTPUT_DIR / "blend_currentbest_with_weak_models_results.csv"


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
print("F multiseed val exists:", f_multiseed_val_path.exists())


# ============================================================
# 1. Reconstruct current best validation score
# current_base = 0.80 weighted_seed + 0.20 top70_seed
# current_pre_f = 0.95 current_base_norm + 0.05 v2_single_norm
# final_current = 0.988 current_pre_f + 0.012 f_click_weighted_b
# ============================================================

weighted_val = pd.read_csv(weighted_val_path)
top70_val = pd.read_csv(top70_val_path)
v2_val = pd.read_csv(v2_val_path)
f_val = pd.read_csv(f_multiseed_val_path)

weighted_val = weighted_val.rename(columns={"blend_score": "weighted_seed_score"})
top70_val = top70_val.rename(columns={"ensemble_score": "top70_seed_score"})

v2_score_col = find_score_col(v2_val)
v2_val = v2_val.rename(columns={v2_score_col: "v2_single_score"})

if "f_click_weighted_b" not in f_val.columns:
    raise ValueError("f_click_weighted_b not found in Model F multiseed validation file.")

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

val = val.merge(
    f_val[["srch_id", "prop_id", "f_click_weighted_b"]],
    on=["srch_id", "prop_id"],
    how="inner"
)

val["current_base_score"] = (
    0.80 * val["weighted_seed_score"]
    + 0.20 * val["top70_seed_score"]
)

val["current_base_score_norm"] = normalize_within_search(val, "current_base_score")
val["v2_single_score_norm"] = normalize_within_search(val, "v2_single_score")

val["current_pre_f_score"] = (
    0.95 * val["current_base_score_norm"]
    + 0.05 * val["v2_single_score_norm"]
)

val["current_best_score"] = (
    0.988 * val["current_pre_f_score"]
    + 0.012 * val["f_click_weighted_b"]
)

val["blend_score"] = val["current_best_score"]
current_ndcg = mean_ndcg_at_5(val)

print("Reconstructed current best NDCG@5:", current_ndcg)
print("Validation data before weak models:", val.shape)


# ============================================================
# 2. Load available weak model scores
# ============================================================

available = []

for name, path in candidate_files.items():
    print(name, "exists:", path.exists())

    if not path.exists():
        continue

    df = pd.read_csv(path)
    score_col = find_score_col(df)
    df = df.rename(columns={score_col: f"{name}_score"})

    val = val.merge(
        df[["srch_id", "prop_id", f"{name}_score"]],
        on=["srch_id", "prop_id"],
        how="inner"
    )

    val[f"{name}_score_norm"] = normalize_within_search(val, f"{name}_score")
    available.append(name)

print("Available weak models:", available)
print("Validation data after weak models:", val.shape)

if not available:
    raise RuntimeError("No weak model validation files found.")


# ============================================================
# 3. Try one weak model at a time
# ============================================================

results = []

for name in available:
    raw_col = f"{name}_score"
    norm_col = f"{name}_score_norm"

    for w_current in np.arange(0.970, 1.0001, 0.005):
        w_weak = 1.0 - w_current

        val["blend_score"] = (
            w_current * val["current_best_score"]
            + w_weak * val[raw_col]
        )

        raw_ndcg = mean_ndcg_at_5(val)

        val["blend_score"] = (
            w_current * val["current_best_score"]
            + w_weak * val[norm_col]
        )

        norm_ndcg = mean_ndcg_at_5(val)

        print(
            "single weak:",
            name,
            "w_current:", round(w_current, 3),
            "w_weak:", round(w_weak, 3),
            "raw:", raw_ndcg,
            "norm:", norm_ndcg
        )

        results.append({
            "mode": "single_raw",
            "weak_model": name,
            "w_current": w_current,
            "w_weak": w_weak,
            "raw_ndcg5": raw_ndcg,
            "norm_ndcg5": np.nan,
            "val_ndcg5": raw_ndcg,
        })

        results.append({
            "mode": "single_norm",
            "weak_model": name,
            "w_current": w_current,
            "w_weak": w_weak,
            "raw_ndcg5": np.nan,
            "norm_ndcg5": norm_ndcg,
            "val_ndcg5": norm_ndcg,
        })


# ============================================================
# 4. Try small equal weak ensemble
# ============================================================

raw_cols = [f"{name}_score" for name in available]
norm_cols = [f"{name}_score_norm" for name in available]

val["weak_raw_equal"] = val[raw_cols].mean(axis=1)
val["weak_norm_equal"] = val[norm_cols].mean(axis=1)

for w_current in np.arange(0.970, 1.0001, 0.005):
    w_weak = 1.0 - w_current

    val["blend_score"] = (
        w_current * val["current_best_score"]
        + w_weak * val["weak_raw_equal"]
    )

    raw_ndcg = mean_ndcg_at_5(val)

    val["blend_score"] = (
        w_current * val["current_best_score"]
        + w_weak * val["weak_norm_equal"]
    )

    norm_ndcg = mean_ndcg_at_5(val)

    print(
        "weak equal ensemble",
        "w_current:", round(w_current, 3),
        "w_weak:", round(w_weak, 3),
        "raw:", raw_ndcg,
        "norm:", norm_ndcg
    )

    results.append({
        "mode": "weak_equal_raw",
        "weak_model": "weak_equal",
        "w_current": w_current,
        "w_weak": w_weak,
        "raw_ndcg5": raw_ndcg,
        "norm_ndcg5": np.nan,
        "val_ndcg5": raw_ndcg,
    })

    results.append({
        "mode": "weak_equal_norm",
        "weak_model": "weak_equal",
        "w_current": w_current,
        "w_weak": w_weak,
        "raw_ndcg5": np.nan,
        "norm_ndcg5": norm_ndcg,
        "val_ndcg5": norm_ndcg,
    })


# ============================================================
# 5. Save results
# ============================================================

results_df = pd.DataFrame(results).sort_values("val_ndcg5", ascending=False)
results_df.to_csv(result_path, index=False)

print("=" * 70)
print("Top weak model blend results:")
print(results_df.head(30))
print("=" * 70)

best = results_df.iloc[0]

print("Best result:")
print(best.to_dict())
print("Saved results to:", result_path)