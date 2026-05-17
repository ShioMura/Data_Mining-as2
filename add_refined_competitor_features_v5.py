from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 0. Paths
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"

train_v2_path = DATA_DIR / "train_features_lean_extra_v2_best.parquet"
test_v2_path = DATA_DIR / "test_features_lean_extra_v2_best.parquet"

raw_train_path = DATA_DIR / "training_set_VU_DM.csv"
raw_test_path = DATA_DIR / "test_set_VU_DM.csv"

out_train_path = DATA_DIR / "train_features_lean_extra_v5_competitor.parquet"
out_test_path = DATA_DIR / "test_features_lean_extra_v5_competitor.parquet"


# ============================================================
# 1. Feature function
# ============================================================

def make_comp_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-6

    rate_cols = [f"comp{i}_rate" for i in range(1, 9)]
    inv_cols = [f"comp{i}_inv" for i in range(1, 9)]
    diff_cols = [f"comp{i}_rate_percent_diff" for i in range(1, 9)]

    out = pd.DataFrame(index=raw_df.index)

    rate = raw_df[rate_cols].copy()
    inv = raw_df[inv_cols].copy()
    diff = raw_df[diff_cols].copy()

    # -------------------------
    # 1. Competitor price relation counts
    # comp_rate:
    # -1 means Expedia is cheaper than competitor
    #  0 means same price
    #  1 means Expedia is more expensive
    # -------------------------

    out["comp_rate_known_count_v5"] = rate.notna().sum(axis=1).astype("float32")

    out["comp_expedia_cheaper_count_v5"] = (
        (rate == -1).sum(axis=1)
    ).astype("float32")

    out["comp_expedia_same_count_v5"] = (
        (rate == 0).sum(axis=1)
    ).astype("float32")

    out["comp_expedia_more_expensive_count_v5"] = (
        (rate == 1).sum(axis=1)
    ).astype("float32")

    out["comp_expedia_cheaper_ratio_v5"] = (
        out["comp_expedia_cheaper_count_v5"]
        / (out["comp_rate_known_count_v5"] + eps)
    ).astype("float32")

    out["comp_expedia_more_expensive_ratio_v5"] = (
        out["comp_expedia_more_expensive_count_v5"]
        / (out["comp_rate_known_count_v5"] + eps)
    ).astype("float32")

    out["comp_has_expedia_cheaper_v5"] = (
        out["comp_expedia_cheaper_count_v5"] > 0
    ).astype("int8")

    out["comp_has_expedia_more_expensive_v5"] = (
        out["comp_expedia_more_expensive_count_v5"] > 0
    ).astype("int8")

    out["comp_price_position_score_v5"] = (
        out["comp_expedia_cheaper_count_v5"]
        - out["comp_expedia_more_expensive_count_v5"]
    ).astype("float32")

    # -------------------------
    # 2. Competitor inventory features
    # comp_inv:
    # 1 often means competitor has no availability
    # 0 means competitor has availability
    # -------------------------

    out["comp_inv_known_count_v5"] = inv.notna().sum(axis=1).astype("float32")

    out["comp_unavailable_count_v5"] = (
        (inv == 1).sum(axis=1)
    ).astype("float32")

    out["comp_available_count_v5"] = (
        (inv == 0).sum(axis=1)
    ).astype("float32")

    out["comp_unavailable_ratio_v5"] = (
        out["comp_unavailable_count_v5"]
        / (out["comp_inv_known_count_v5"] + eps)
    ).astype("float32")

    out["comp_has_unavailable_v5"] = (
        out["comp_unavailable_count_v5"] > 0
    ).astype("int8")

    # -------------------------
    # 3. Percent-difference statistics
    # -------------------------

    out["comp_percent_diff_known_count_v5"] = (
        diff.notna().sum(axis=1)
    ).astype("float32")

    out["comp_percent_diff_mean_v5"] = (
        diff.mean(axis=1).fillna(0)
    ).astype("float32")

    out["comp_percent_diff_min_v5"] = (
        diff.min(axis=1).fillna(0)
    ).astype("float32")

    out["comp_percent_diff_max_v5"] = (
        diff.max(axis=1).fillna(0)
    ).astype("float32")

    out["comp_percent_diff_std_v5"] = (
        diff.std(axis=1).fillna(0)
    ).astype("float32")

    positive_diff = diff.where(diff > 0)
    negative_diff = diff.where(diff < 0)

    out["comp_positive_percent_diff_mean_v5"] = (
        positive_diff.mean(axis=1).fillna(0)
    ).astype("float32")

    out["comp_negative_percent_diff_mean_v5"] = (
        negative_diff.mean(axis=1).fillna(0)
    ).astype("float32")

    out["comp_positive_percent_diff_count_v5"] = (
        (diff > 0).sum(axis=1)
    ).astype("float32")

    out["comp_negative_percent_diff_count_v5"] = (
        (diff < 0).sum(axis=1)
    ).astype("float32")

    # -------------------------
    # 4. Combined competitor strength
    # -------------------------

    out["comp_price_advantage_strength_v5"] = (
        out["comp_expedia_cheaper_ratio_v5"]
        + 0.1 * out["comp_unavailable_ratio_v5"]
    ).astype("float32")

    out["comp_price_disadvantage_strength_v5"] = (
        out["comp_expedia_more_expensive_ratio_v5"]
        + 0.01 * out["comp_positive_percent_diff_mean_v5"].clip(lower=0)
    ).astype("float32")

    return out


# ============================================================
# 2. Load v2 features
# ============================================================

print("Loading v2 features...")

train = pd.read_parquet(train_v2_path)
test = pd.read_parquet(test_v2_path)

print("Train v2:", train.shape)
print("Test v2:", test.shape)


# ============================================================
# 3. Load raw competitor columns
# ============================================================

comp_cols = []

for i in range(1, 9):
    comp_cols += [
        f"comp{i}_rate",
        f"comp{i}_inv",
        f"comp{i}_rate_percent_diff",
    ]

print("Loading raw competitor columns...")

raw_train_comp = pd.read_csv(raw_train_path, usecols=comp_cols)
raw_test_comp = pd.read_csv(raw_test_path, usecols=comp_cols)

print("Raw train comp:", raw_train_comp.shape)
print("Raw test comp:", raw_test_comp.shape)


# ============================================================
# 4. Safety check: row alignment
# ============================================================

if len(raw_train_comp) != len(train):
    raise ValueError(
        f"Train row count mismatch: raw={len(raw_train_comp)}, v2={len(train)}"
    )

if len(raw_test_comp) != len(test):
    raise ValueError(
        f"Test row count mismatch: raw={len(raw_test_comp)}, v2={len(test)}"
    )


# ============================================================
# 5. Create v5 competitor features
# ============================================================

print("Creating refined competitor features...")

train_comp = make_comp_features(raw_train_comp)
test_comp = make_comp_features(raw_test_comp)

print("Train comp features:", train_comp.shape)
print("Test comp features:", test_comp.shape)

print("New competitor feature columns:")
print(train_comp.columns.tolist())


# ============================================================
# 6. Avoid duplicate names
# ============================================================

duplicate_train = [c for c in train_comp.columns if c in train.columns]
duplicate_test = [c for c in test_comp.columns if c in test.columns]

if duplicate_train or duplicate_test:
    raise ValueError(
        f"Duplicate feature names found: {duplicate_train + duplicate_test}"
    )


# ============================================================
# 7. Merge with v2 parquet
# ============================================================

train_v5 = pd.concat(
    [train.reset_index(drop=True), train_comp.reset_index(drop=True)],
    axis=1
)

test_v5 = pd.concat(
    [test.reset_index(drop=True), test_comp.reset_index(drop=True)],
    axis=1
)

print("Final train v5:", train_v5.shape)
print("Final test v5:", test_v5.shape)


# ============================================================
# 8. Save
# ============================================================

train_v5.to_parquet(out_train_path, index=False)
test_v5.to_parquet(out_test_path, index=False)

print("Saved train to:", out_train_path)
print("Saved test to:", out_test_path)