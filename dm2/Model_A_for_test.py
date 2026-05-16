import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# 0. 设置路径
# ============================================================

DATA_DIR = Path(r"A:\dm2")

train_path = DATA_DIR / "train_features_lean.parquet"
test_path = DATA_DIR / "test_features_lean.parquet"

submission_path = DATA_DIR / "submission_model_a_hybrid_recommender.csv"


# ============================================================
# 1. 读取 train features 和 test features
# ============================================================

train_fe = pd.read_parquet(train_path)
test_fe = pd.read_parquet(test_path)

print("Train shape:", train_fe.shape)
print("Test shape:", test_fe.shape)

print("Train columns:", len(train_fe.columns))
print("Test columns:", len(test_fe.columns))


# ============================================================
# 2. 检查必要列是否存在
# ============================================================

required_train_cols = [
    "prop_id",
    "relevance"
]

required_test_cols = [
    "srch_id",
    "prop_id",
    "review_high_rank_pct",
    "loc1_high_rank_pct",
    "loc2_high_rank_pct",
    "star_high_rank_pct",
    "promotion_flag",
    "price_cheap_rank_pct",
    "price_ratio_to_srch_median"
]

missing_train_cols = [c for c in required_train_cols if c not in train_fe.columns]
missing_test_cols = [c for c in required_test_cols if c not in test_fe.columns]

if missing_train_cols:
    raise ValueError(f"Missing columns in train_fe: {missing_train_cols}")

if missing_test_cols:
    raise ValueError(f"Missing columns in test_fe: {missing_test_cols}")

print("Required columns exist.")


# ============================================================
# 3. 用完整 train_fe 计算 property-level popularity features
# ============================================================
# 注意：
# 这里是给 test set 生成最终 submission，
# 所以可以使用完整 train_fe 来计算历史 CTR / CVR / popularity。
#
# relevance:
# 5 = booked
# 1 = clicked
# 0 = no interaction
#
# CTR = clicked_or_booked / presentations
# CVR = booked / clicked_or_booked
# 加 alpha smoothing，避免出现次数很少的酒店 CTR/CVR 过于极端。
# ============================================================

alpha = 10

full_global_ctr = (train_fe["relevance"] >= 1).mean()

full_global_cvr = (
    (train_fe["relevance"] == 5).sum()
    / max((train_fe["relevance"] >= 1).sum(), 1)
)

print("Full global CTR:", full_global_ctr)
print("Full global CVR:", full_global_cvr)

full_prop_pop = (
    train_fe
    .groupby("prop_id")
    .agg(
        prop_presentations=("relevance", "size"),
        prop_clicks=("relevance", lambda x: (x >= 1).sum()),
        prop_bookings=("relevance", lambda x: (x == 5).sum())
    )
    .reset_index()
)

full_prop_pop["prop_ctr"] = (
    (full_prop_pop["prop_clicks"] + alpha * full_global_ctr)
    / (full_prop_pop["prop_presentations"] + alpha)
)

full_prop_pop["prop_cvr"] = (
    (full_prop_pop["prop_bookings"] + alpha * full_global_cvr)
    / (full_prop_pop["prop_clicks"] + alpha)
)

full_prop_pop["prop_popularity_log"] = np.log1p(full_prop_pop["prop_presentations"])

max_pop = full_prop_pop["prop_popularity_log"].max()
if max_pop > 0:
    full_prop_pop["prop_popularity_log"] = (
        full_prop_pop["prop_popularity_log"] / max_pop
    )

print("Popularity feature table shape:", full_prop_pop.shape)
print(full_prop_pop.head())


# ============================================================
# 4. merge popularity features 到 test
# ============================================================

test_df = test_fe.merge(full_prop_pop, on="prop_id", how="left")

# test 里有些酒店可能在 train 里没有出现过，所以需要填补
test_df["prop_ctr"] = test_df["prop_ctr"].fillna(full_global_ctr)
test_df["prop_cvr"] = test_df["prop_cvr"].fillna(full_global_cvr)
test_df["prop_popularity_log"] = test_df["prop_popularity_log"].fillna(0)

print("Test after merge shape:", test_df.shape)

print(
    test_df[
        ["srch_id", "prop_id", "prop_ctr", "prop_cvr", "prop_popularity_log"]
    ].head()
)


# ============================================================
# 5. 定义辅助函数
# ============================================================

def safe_col(df, col, default=0):
    """
    如果 df 里有这个列，就返回该列并填补缺失值；
    如果没有这个列，就返回默认值 Series。
    """
    if col in df.columns:
        return df[col].fillna(default)
    return pd.Series(default, index=df.index)


def query_minmax(df, col):
    """
    在每个 srch_id 内部做 min-max normalization。
    因为最终任务是在同一个 search query 内部排序酒店。
    """
    g = df.groupby("srch_id")[col]
    min_v = g.transform("min")
    max_v = g.transform("max")

    return (df[col] - min_v) / (max_v - min_v + 1e-9)


# ============================================================
# 6. 给 test 计算 Model A 的子分数
# ============================================================
# 这里使用你验证集中表现最好的 A3:
#
# Final Model A = quality + price + popularity
#
# model_a_final_score =
#     0.45 * quality_score
#   + 0.30 * price_score
#   + 0.25 * popularity_score
#
# 不再加入 user_match_score，
# 因为 validation 里 A3 比 A4 略高。
# ============================================================

def add_model_a_final_score(df, global_ctr, global_cvr):
    df = df.copy()

    # ------------------------------------------------------------
    # 1. quality_score_raw：酒店质量分
    # ------------------------------------------------------------
    df["quality_score_raw"] = (
        0.30 * safe_col(df, "review_high_rank_pct")
        + 0.25 * safe_col(df, "loc2_high_rank_pct")
        + 0.20 * safe_col(df, "loc1_high_rank_pct")
        + 0.15 * safe_col(df, "star_high_rank_pct")
        + 0.10 * safe_col(df, "promotion_flag")
    )

    # ------------------------------------------------------------
    # 2. price_score_raw：价格吸引力分
    # ------------------------------------------------------------
    price_ratio = safe_col(df, "price_ratio_to_srch_median", default=1)
    price_ratio_penalty = price_ratio.clip(0, 5) / 5

    df["price_score_raw"] = (
        0.80 * safe_col(df, "price_cheap_rank_pct")
        - 0.20 * price_ratio_penalty
    )

    # ------------------------------------------------------------
    # 3. popularity_score_raw：历史群体偏好分
    # ------------------------------------------------------------
    df["popularity_score_raw"] = (
        0.45 * safe_col(df, "prop_ctr", global_ctr)
        + 0.45 * safe_col(df, "prop_cvr", global_cvr)
        + 0.10 * safe_col(df, "prop_popularity_log")
    )

    # ------------------------------------------------------------
    # 4. query-level normalization
    # ------------------------------------------------------------
    raw_score_cols = [
        "quality_score_raw",
        "price_score_raw",
        "popularity_score_raw"
    ]

    for col in raw_score_cols:
        normalized_col = col.replace("_raw", "")
        df[normalized_col] = query_minmax(df, col)

    # ------------------------------------------------------------
    # 5. 最终 Model A score
    # ------------------------------------------------------------
    df["model_a_final_score"] = (
        0.45 * df["quality_score"]
        + 0.30 * df["price_score"]
        + 0.25 * df["popularity_score"]
    )

    return df


test_scored = add_model_a_final_score(
    test_df,
    global_ctr=full_global_ctr,
    global_cvr=full_global_cvr
)

print(
    test_scored[
        [
            "srch_id",
            "prop_id",
            "quality_score",
            "price_score",
            "popularity_score",
            "model_a_final_score"
        ]
    ].head(20)
)


# ============================================================
# 7. 生成 submission
# ============================================================
# Kaggle 要求格式:
#
# SearchId,PropertyId
# 2,7771
# 2,26540
# ...
#
# 每个 srch_id 内部按照 model_a_final_score 从高到低排序。
# ============================================================

submission = (
    test_scored
    .sort_values(
        ["srch_id", "model_a_final_score"],
        ascending=[True, False]
    )
    [["srch_id", "prop_id"]]
    .rename(
        columns={
            "srch_id": "SearchId",
            "prop_id": "PropertyId"
        }
    )
)

submission.to_csv(submission_path, index=False)

print("Submission saved to:", submission_path)
print("Submission shape:", submission.shape)
print(submission.head(30))


# ============================================================
# 8. 简单检查 submission 是否合理
# ============================================================

print("Number of unique SearchId:", submission["SearchId"].nunique())
print("Number of rows:", len(submission))

# 检查是否有缺失
print("Missing values:")
print(submission.isna().sum())

# 检查每个 SearchId 下是否有重复 PropertyId
dup_count = submission.duplicated(["SearchId", "PropertyId"]).sum()
print("Duplicated SearchId-PropertyId pairs:", dup_count)