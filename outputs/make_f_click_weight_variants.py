from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs"

weighted_test_path = OUTPUT_DIR / "model_b_test_scores_extra_v2_weighted_seed_ensemble.csv"
top70_test_path = OUTPUT_DIR / "model_b_test_scores_extra_v2_pruned_top70_seed_ensemble.csv"
v2_test_path = OUTPUT_DIR / "model_b_test_scores_extra_v2_best.csv"
f_multi_test_path = OUTPUT_DIR / "model_f_click_multiseed_final_test_scores.csv"


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


print("weighted test exists:", weighted_test_path.exists())
print("top70 test exists:", top70_test_path.exists())
print("v2 test exists:", v2_test_path.exists())
print("F multiseed test exists:", f_multi_test_path.exists())


# ============================================================
# 1. Load score files
# ============================================================

weighted_test = pd.read_csv(weighted_test_path)
top70_test = pd.read_csv(top70_test_path)
v2_test = pd.read_csv(v2_test_path)
f_test = pd.read_csv(f_multi_test_path)

weighted_test = weighted_test.rename(columns={"blend_score": "weighted_seed_score"})
top70_test = top70_test.rename(columns={"ensemble_score": "top70_seed_score"})

v2_col = find_score_col(v2_test)
v2_test = v2_test.rename(columns={v2_col: "v2_single_score"})

if "f_click_weighted_b" not in f_test.columns:
    raise ValueError("f_click_weighted_b not found in model_f_click_multiseed_final_test_scores.csv")


# ============================================================
# 2. Reconstruct current_pre_f
# current_base = 0.80 weighted_seed + 0.20 top70_seed
# current_pre_f = 0.95 current_base_norm + 0.05 v2_single_norm
# ============================================================

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

test = test.merge(
    f_test[["srch_id", "prop_id", "f_click_weighted_b"]],
    on=["srch_id", "prop_id"],
    how="inner"
)

print("Merged test data:", test.shape)

test["current_base_score"] = (
    0.80 * test["weighted_seed_score"]
    + 0.20 * test["top70_seed_score"]
)

test["current_base_score_norm"] = normalize_within_search(test, "current_base_score")
test["v2_single_score_norm"] = normalize_within_search(test, "v2_single_score")

test["current_pre_f_score"] = (
    0.95 * test["current_base_score_norm"]
    + 0.05 * test["v2_single_score_norm"]
)


# ============================================================
# 3. Generate F-click weight variants
# ============================================================

weights = [
    0.005,  # conservative
    0.008,  # previous fine-tuned seed42-like
    0.010,  # conservative multiseed
    0.012,  # current best
    0.015,  # slightly aggressive
    0.020,  # aggressive
]

for w_f in weights:
    w_current = 1.0 - w_f

    test["blend_score"] = (
        w_current * test["current_pre_f_score"]
        + w_f * test["f_click_weighted_b"]
    )

    suffix = str(w_f).replace(".", "p")

    submission_path = OUTPUT_DIR / f"submission_f_click_multiseed_weight_{suffix}.csv"

    submission = (
        test
        .sort_values(["srch_id", "blend_score"], ascending=[True, False])
        [["srch_id", "prop_id"]]
    )

    submission.to_csv(submission_path, index=False)

    print("=" * 70)
    print("Saved:", submission_path)
    print("w_current:", w_current)
    print("w_f:", w_f)
    print("Submission shape:", submission.shape)
    print("Unique srch_id:", submission["srch_id"].nunique())
    print("Missing values:")
    print(submission.isna().sum())
    print("Duplicated srch_id-prop_id:", submission.duplicated(["srch_id", "prop_id"]).sum())