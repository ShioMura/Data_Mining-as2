from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"

train_csv = DATA_DIR / "training_set_VU_DM.csv"
test_csv = DATA_DIR / "test_set_VU_DM.csv"

print("Train csv exists:", train_csv.exists())
print("Test csv exists:", test_csv.exists())

train_cols = pd.read_csv(train_csv, nrows=5).columns.tolist()
test_cols = pd.read_csv(test_csv, nrows=5).columns.tolist()

comp_cols = []
for i in range(1, 9):
    for suffix in ["rate", "inv", "rate_percent_diff"]:
        comp_cols.append(f"comp{i}_{suffix}")

print("\nExisting competitor columns in raw train:")
print([c for c in comp_cols if c in train_cols])

print("\nExisting competitor columns in raw test:")
print([c for c in comp_cols if c in test_cols])

print("\nAll raw train columns containing 'comp':")
print([c for c in train_cols if "comp" in c.lower()])