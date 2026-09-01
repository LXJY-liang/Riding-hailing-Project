# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ============================================================
# 1. 路径设置
# ============================================================

INPUT_CSV = "/Users/liangxue/Documents/8、Python文件-网约车数据/Riding hailing_Project/1 数据清洗/1 经纬度恢复及删除多余变量_保留订单.csv"

OUTPUT_PARQUET = "/Users/liangxue/Documents/8、Python文件-网约车数据/Riding hailing_Project/1 数据清洗/1_2 经纬度恢复及删除多余变量_保留订单.parquet"


# ============================================================
# 2. 参数设置
# ============================================================

CHUNKSIZE = 500000

STRING_COLS = [
    "Trip ID",
    "Pickup Census Tract",
    "Dropoff Census Tract",
]

TIME_COLS = [
    "Trip Start Timestamp",
    "Trip End Timestamp",
]

NUMERIC_COLS = [
    "Trip Seconds",
    "Trip Miles",
    "Fare",
    "Additional Charges",
    "Trip Total fare",
]


# ============================================================
# 3. 工具函数
# ============================================================

def clean_numeric(s):
    """把可能存在的 $、逗号、空格清理掉，并转成数值。"""
    return pd.to_numeric(
        s.astype(str)
         .str.replace("$", "", regex=False)
         .str.replace(",", "", regex=False)
         .str.strip()
         .replace({"": np.nan, "nan": np.nan, "None": np.nan, "<NA>": np.nan}),
        errors="coerce"
    )


# ============================================================
# 4. CSV 分块转 Parquet
# ============================================================

if os.path.exists(OUTPUT_PARQUET):
    os.remove(OUTPUT_PARQUET)

writer = None
total_rows = 0

reader = pd.read_csv(
    INPUT_CSV,
    chunksize=CHUNKSIZE,
    low_memory=False,
    dtype={col: "string" for col in STRING_COLS}
)

for chunk_id, df in enumerate(reader, start=1):

    # ---------- 字符串字段 ----------
    for col in STRING_COLS:
        df[col] = df[col].astype("string")

    # ---------- 时间字段 ----------
    for col in TIME_COLS:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # ---------- 数值字段 ----------
    for col in NUMERIC_COLS:
        df[col] = clean_numeric(df[col])

    # ---------- 写入 parquet ----------
    table = pa.Table.from_pandas(df, preserve_index=False)

    if writer is None:
        writer = pq.ParquetWriter(
            OUTPUT_PARQUET,
            table.schema,
            compression="snappy"
        )

    writer.write_table(table)

    total_rows += len(df)

    print(f"chunk {chunk_id} 完成，写入 {len(df)} 行")


if writer is not None:
    writer.close()


print("完成")
print(f"总行数：{total_rows}")
print(f"Parquet 文件：{OUTPUT_PARQUET}")