from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"

train_path = DATA_DIR / "train_features_lean_extra_v2_best.parquet"
test_path = DATA_DIR / "test_features_lean_extra_v2_best.parquet"

out_train_path = DATA_DIR / "train_features_lean_extra_v2_te.parquet"
out_test_path = DATA_DIR / "test_features_lean_extra_v2_te.parquet"

print("Loading v2 features...")
train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

print("Train:", train.shape)
print("Test:", test.shape)

# Targets used only for encoding, not kept as model features directly
train["target_booking"] = (train["relevance"] == 5).astype("float32")
train["target_click"] = (train["relevance"] >= 1).astype("float32")
train["target_relevance"] = train["relevance"].astype("float32")

targets = [
    ("target_booking", "booking_rate", 50.0),
    ("target_click", "click_rate", 50.0),
    ("target_relevance", "rel_mean", 100.0),
]

keys = [
    "prop_id",
    "srch_destination_id",
]

global_means = {
    target_col: float(train[target_col].mean())
    for target_col, _, _ in targets
}

groups = train["srch_id"].to_numpy()
gkf = GroupKFold(n_splits=5)


def add_oof_single_key(train_df, test_df, key_col, target_col, out_col, alpha):
    prior = global_means[target_col]

    train_df[out_col] = np.nan

    print(f"Creating OOF feature: {out_col}")

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(train_df, groups=groups), start=1):
        tr = train_df.iloc[tr_idx]
        val_keys = train_df.iloc[val_idx][[key_col]].copy()

        stats = (
            tr.groupby(key_col, sort=False)[target_col]
            .agg(["sum", "count"])
            .reset_index()
        )

        stats[out_col] = (
            (stats["sum"] + alpha * prior)
            / (stats["count"] + alpha)
        ).astype("float32")

        mapped = val_keys.merge(
            stats[[key_col, out_col]],
            on=key_col,
            how="left"
        )[out_col].fillna(prior).astype("float32")

        train_df.loc[train_df.index[val_idx], out_col] = mapped.to_numpy()

        print(f"  fold {fold} done")

    train_df[out_col] = train_df[out_col].fillna(prior).astype("float32")

    # Full-train encoding for test
    full_stats = (
        train_df.groupby(key_col, sort=False)[target_col]
        .agg(["sum", "count"])
        .reset_index()
    )

    full_stats[out_col] = (
        (full_stats["sum"] + alpha * prior)
        / (full_stats["count"] + alpha)
    ).astype("float32")

    test_df[out_col] = (
        test_df[[key_col]]
        .merge(full_stats[[key_col, out_col]], on=key_col, how="left")[out_col]
        .fillna(prior)
        .astype("float32")
    )

    # Count feature, useful but not target leakage
    count_col = f"{key_col}_hist_count_log"
    count_stats = (
        train_df.groupby(key_col, sort=False)
        .size()
        .reset_index(name=count_col)
    )
    count_stats[count_col] = np.log1p(count_stats[count_col]).astype("float32")

    train_df[count_col] = (
        train_df[[key_col]]
        .merge(count_stats, on=key_col, how="left")[count_col]
        .fillna(0)
        .astype("float32")
    )

    test_df[count_col] = (
        test_df[[key_col]]
        .merge(count_stats, on=key_col, how="left")[count_col]
        .fillna(0)
        .astype("float32")
    )


for key_col in keys:
    if key_col not in train.columns or key_col not in test.columns:
        print(f"Skipping {key_col}, not found.")
        continue

    for target_col, suffix, alpha in targets:
        out_col = f"{key_col}_{suffix}_oof"
        add_oof_single_key(train, test, key_col, target_col, out_col, alpha)


# Remove temporary target columns before saving
train.drop(
    columns=["target_booking", "target_click", "target_relevance"],
    inplace=True
)

print("Final train:", train.shape)
print("Final test:", test.shape)

train.to_parquet(out_train_path, index=False)
test.to_parquet(out_test_path, index=False)

print("Saved train to:", out_train_path)
print("Saved test to:", out_test_path)