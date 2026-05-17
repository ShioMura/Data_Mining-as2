from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score


ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs"

weighted_val_path = OUTPUT_DIR / "model_b_val_scores_extra_v2_weighted_seed_ensemble.csv"
top70_val_path = OUTPUT_DIR / "model_b_val_scores_extra_v2_pruned_top70_seed_ensemble.csv"
v2_val_path = OUTPUT_DIR / "model_b_val_scores_extra_v2_best.csv"
model_f_val_path = OUTPUT_DIR / "model_f_val_scores_binary_rankers.csv"

result_path = OUTPUT_DIR / "fine_tune_currentbest_model_f_click_results.csv"


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
    raise ValueError(f"No score column found: {df.columns.tolist()}")


print("weighted val exists:", weighted_val_path.exists())
print("top70 val exists:", top70_val_path.exists())
print("v2 val exists:", v2_val_path.exists())
print("model_f val exists:", model_f_val_path.exists())


# ============================================================
# 1. Load score files
# ============================================================

weighted_val = pd.read_csv(weighted_val_path)
top70_val = pd.read_csv(top70_val_path)
v2_val = pd.read_csv(v2_val_path)
model_f_val = pd.read_csv(model_f_val_path)

weighted_val = weighted_val.rename(columns={"blend_score": "weighted_seed_score"})
top70_val = top70_val.rename(columns={"ensemble_score": "top70_seed_score"})

v2_score_col = find_score_col(v2_val)
v2_val = v2_val.rename(columns={v2_score_col: "v2_single_score"})

# Model F: use click ranker score because validation selected click only
if "click_ranker_score" in model_f_val.columns:
    model_f_val = model_f_val.rename(columns={"click_ranker_score": "model_f_click_score"})
else:
    f_score_col = find_score_col(model_f_val)
    model_f_val = model_f_val.rename(columns={f_score_col: "model_f_click_score"})


# ============================================================
# 2. Reconstruct current best before Model F
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
    model_f_val[["srch_id", "prop_id", "model_f_click_score"]],
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

val["model_f_click_score_norm"] = normalize_within_search(val, "model_f_click_score")

print("Validation data:", val.shape)

val["blend_score"] = val["current_pre_f_score"]
pre_f_ndcg = mean_ndcg_at_5(val)
print("Current pre-F NDCG@5:", pre_f_ndcg)


# ============================================================
# 3. Fine search around 0.99 / 0.01
# ============================================================

results = []

for w_current in np.arange(0.985, 0.9971, 0.001):
    w_f = 1.0 - w_current

    # raw F score blend
    val["blend_score"] = (
        w_current * val["current_pre_f_score"]
        + w_f * val["model_f_click_score"]
    )

    raw_ndcg = mean_ndcg_at_5(val)

    # normalized F score blend
    val["blend_score"] = (
        w_current * val["current_pre_f_score"]
        + w_f * val["model_f_click_score_norm"]
    )

    norm_ndcg = mean_ndcg_at_5(val)

    print(
        "w_current:", round(w_current, 4),
        "w_f:", round(w_f, 4),
        "raw:", raw_ndcg,
        "norm:", norm_ndcg
    )

    results.append({
        "w_current": w_current,
        "w_model_f": w_f,
        "raw_ndcg5": raw_ndcg,
        "norm_ndcg5": norm_ndcg,
    })


results_df = pd.DataFrame(results)
results_df.to_csv(result_path, index=False)

print("=" * 70)
print("Top raw:")
print(results_df.sort_values("raw_ndcg5", ascending=False).head(20))
print("=" * 70)
print("Top norm:")
print(results_df.sort_values("norm_ndcg5", ascending=False).head(20))
print("=" * 70)

best_raw = results_df.sort_values("raw_ndcg5", ascending=False).iloc[0]
best_norm = results_df.sort_values("norm_ndcg5", ascending=False).iloc[0]

print("Best raw:", best_raw.to_dict())
print("Best norm:", best_norm.to_dict())
print("Saved results to:", result_path)