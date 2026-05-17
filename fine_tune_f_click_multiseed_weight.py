from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score


# ============================================================
# 0. Paths
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs"

weighted_val_path = OUTPUT_DIR / "model_b_val_scores_extra_v2_weighted_seed_ensemble.csv"
top70_val_path = OUTPUT_DIR / "model_b_val_scores_extra_v2_pruned_top70_seed_ensemble.csv"
v2_val_path = OUTPUT_DIR / "model_b_val_scores_extra_v2_best.csv"
f_multi_val_path = OUTPUT_DIR / "model_f_click_multiseed_best_val_scores.csv"

result_path = OUTPUT_DIR / "fine_tune_f_click_multiseed_weight_results.csv"


print("weighted val exists:", weighted_val_path.exists())
print("top70 val exists:", top70_val_path.exists())
print("v2 val exists:", v2_val_path.exists())
print("F multiseed val exists:", f_multi_val_path.exists())


# ============================================================
# 1. Helpers
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


def find_score_col(df):
    for col in ["blend_score", "ensemble_score", "model_b_score", "model_score"]:
        if col in df.columns:
            return col

    raise ValueError(f"No score column found. Columns = {df.columns.tolist()}")


# ============================================================
# 2. Load validation score files
# ============================================================

weighted_val = pd.read_csv(weighted_val_path)
top70_val = pd.read_csv(top70_val_path)
v2_val = pd.read_csv(v2_val_path)
f_val = pd.read_csv(f_multi_val_path)

weighted_val = weighted_val.rename(columns={"blend_score": "weighted_seed_score"})
top70_val = top70_val.rename(columns={"ensemble_score": "top70_seed_score"})

v2_col = find_score_col(v2_val)
v2_val = v2_val.rename(columns={v2_col: "v2_single_score"})

if "f_click_weighted_b" not in f_val.columns:
    raise ValueError(
        "f_click_weighted_b not found in model_f_click_multiseed_best_val_scores.csv"
    )


# ============================================================
# 3. Reconstruct current_pre_f validation score
#
# current_base = 0.80 weighted_seed + 0.20 top70_seed
# current_pre_f = 0.95 current_base_norm + 0.05 v2_single_norm
# ============================================================

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

print("Validation data:", val.shape)

val["current_base_score"] = (
    0.80 * val["weighted_seed_score"]
    + 0.20 * val["top70_seed_score"]
)

val["current_base_score_norm"] = normalize_within_search(
    val,
    "current_base_score"
)

val["v2_single_score_norm"] = normalize_within_search(
    val,
    "v2_single_score"
)

val["current_pre_f_score"] = (
    0.95 * val["current_base_score_norm"]
    + 0.05 * val["v2_single_score_norm"]
)

val["blend_score"] = val["current_pre_f_score"]
pre_f_ndcg = mean_ndcg_at_5(val)

print("Pre-F NDCG@5:", pre_f_ndcg)


# ============================================================
# 4. Fine search F-click multiseed weight
# ============================================================

results = []

for w_f in np.arange(0.004, 0.0201, 0.0005):
    w_current = 1.0 - w_f

    val["blend_score"] = (
        w_current * val["current_pre_f_score"]
        + w_f * val["f_click_weighted_b"]
    )

    ndcg = mean_ndcg_at_5(val)

    print(
        "w_current:", round(w_current, 4),
        "w_f:", round(w_f, 4),
        "NDCG@5:", ndcg
    )

    results.append({
        "w_current": w_current,
        "w_f": w_f,
        "val_ndcg5": ndcg,
    })


# ============================================================
# 5. Save results
# ============================================================

results_df = pd.DataFrame(results).sort_values("val_ndcg5", ascending=False)
results_df.to_csv(result_path, index=False)

print("=" * 70)
print("Fine tune F-click multiseed weight results:")
print(results_df.head(40))
print("=" * 70)

best = results_df.iloc[0]

print("Best result:")
print(best.to_dict())
print("Saved results to:", result_path)