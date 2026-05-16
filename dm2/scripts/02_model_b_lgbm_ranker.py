from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

train_path = DATA_DIR / "train_features_lean.parquet"
test_path = DATA_DIR / "test_features_lean.parquet"
submission_path = OUTPUT_DIR / "submission_model_b_lgbm_ranker.csv"
print("ROOT_DIR:", ROOT_DIR)
print("DATA_DIR:", DATA_DIR)
print("OUTPUT_DIR:", OUTPUT_DIR)

print("Train feature path exists:", train_path.exists())
print("Test feature path exists:", test_path.exists())
print("Submission path:", submission_path)
