from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"

train_path = DATA_DIR / "train_features_lean_extra_v2_best.parquet"
test_path = DATA_DIR / "test_features_lean_extra_v2_best.parquet"

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

print("Train shape:", train.shape)
print("Test shape:", test.shape)

comp_cols = []
for i in range(1, 9):
    for suffix in ["rate", "inv", "rate_percent_diff"]:
        col = f"comp{i}_{suffix}"
        comp_cols.append(col)

existing_train = [c for c in comp_cols if c in train.columns]
existing_test = [c for c in comp_cols if c in test.columns]

print("\nExisting competitor columns in train:")
print(existing_train)

print("\nExisting competitor columns in test:")
print(existing_test)

print("\nMissing competitor columns:")
print([c for c in comp_cols if c not in train.columns])

print("\nAll columns containing 'comp':")
print([c for c in train.columns if "comp" in c.lower()])