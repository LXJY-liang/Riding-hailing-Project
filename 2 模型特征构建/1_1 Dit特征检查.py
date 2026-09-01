# -*- coding: utf-8 -*-

import pandas as pd


DIT_FEATURE_FILE = "1 Dit特征构建（结果）.parquet"

# TARGET_REGION_ID = "17031770602"
# TARGET_T = pd.Timestamp("2024-06-06 03:15:00")

TARGET_REGION_ID = "17031833000"
# TARGET_REGION_ID = "17031839100"
TARGET_T = pd.Timestamp("2024-06-01 10:00:00")


df = pd.read_parquet(DIT_FEATURE_FILE)

df["region_id"] = df["region_id"].astype("string")
df["t"] = pd.to_datetime(df["t"], errors="coerce")


result = df[
    (df["region_id"] == TARGET_REGION_ID)
    & (df["t"] == TARGET_T)
].copy()


pd.set_option("display.max_columns", None)
pd.set_option("display.width", 400)
pd.set_option("display.max_colwidth", 120)


if result.empty:
    print("未找到符合条件的样本")
else:
    print(result.to_string(index=False))