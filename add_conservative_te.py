from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"

# Always start from the clean extra v2 best features
train_path = DATA_DIR / "train_features_lean_extra_v2_best.parquet"
test_path = DATA_DIR / "test_features_lean_extra_v2_best.parquet"

out_train_path = DATA_DIR / "train_features_lean_extra_v2_te_conservative.parquet"
out_test_path = DATA_DIR / "test_features_lean_extra_v2_te_conservative.parquet"

print("Loading extra v2 best features...")
train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

print("Train:", train.shape)
print("Test:", test.shape)

# Conservative target: click/book indicator only
# relevance >= 1 means clicked or booked
train["target_click"] = (train["relevance"] >= 1).astype("float32")

key_cols = [
    "prop_id",
    "srch_destination_id",
]

alpha = 300.0
prior = float(train["target_click"].mean())

print("Global click/book prior:", prior)
print("Smoothing alpha:", alpha)

groups = train["srch_id"].to_numpy()
gkf = GroupKFold(n_splits=5)


def add_oof_click_te(train_df, test_df, key_col):
    out_col = f"{key_col}_click_rate_oof_cons"
    count_col = f"{key_col}_hist_count_log_cons"

    print(f"Creating conservative OOF feature: {out_col}")

    train_df[out_col] = np.nan

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(train_df, groups=groups), start=1):
        tr = train_df.iloc[tr_idx]
        val_keys = train_df.iloc[val_idx][[key_col]].copy()

        stats = (
            tr.groupby(key_col, sort=False)["target_click"]
            .agg(["sum", "count"])
            .reset_index()
        )

        stats[out_col] = (
            (stats["sum"] + alpha * prior)
            / (stats["count"] + alpha)
        ).astype("float32")

        mapped = (
            val_keys
            .merge(stats[[key_col, out_col]], on=key_col, how="left")[out_col]
            .fillna(prior)
            .astype("float32")
        )

        train_df.loc[train_df.index[val_idx], out_col] = mapped.to_numpy()

        print(f"  fold {fold} done")

    train_df[out_col] = train_df[out_col].fillna(prior).astype("float32")

    # Full train encoding for test
    full_stats = (
        train_df.groupby(key_col, sort=False)["target_click"]
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

    # Historical exposure count, no target leakage
    count_stats = (
        train_df.groupby(key_col, sort=False)
        .size()
        .reset_index(name=count_col)
    )

    count_stats[count_col] = np.log1p(count_stats[count_col]).astype("float32")

    train_df[count_col] = (
        train_df[[key_col]]
        .merge(count_stats[[key_col, count_col]], on=key_col, how="left")[count_col]
        .fillna(0)
        .astype("float32")
    )

    test_df[count_col] = (
        test_df[[key_col]]
        .merge(count_stats[[key_col, count_col]], on=key_col, how="left")[count_col]
        .fillna(0)
        .astype("float32")
    )

    print(f"Added count feature: {count_col}")


for key_col in key_cols:
    if key_col not in train.columns or key_col not in test.columns:
        print(f"Skipping {key_col}, not found.")
        continue

    add_oof_click_te(train, test, key_col)


train.drop(columns=["target_click"], inplace=True)

print("Final train:", train.shape)
print("Final test:", test.shape)

train.to_parquet(out_train_path, index=False)
test.to_parquet(out_test_path, index=False)

print("Saved train to:", out_train_path)
print("Saved test to:", out_test_path)