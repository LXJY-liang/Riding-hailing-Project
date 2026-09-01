# -*- coding: utf-8 -*-

# 根据订单ID检查删除订单的合理性。是否是拼车？是否有上下车点的经纬度缺失？
# 通过输入订单ID，终端输出该ID的所有信息。
# 人工审核是否符合要求

import pandas as pd

RAW_DATA_FILE = "/Users/liangxue/Documents/8、Python文件-网约车数据/Transportation_Network_Providers_-_Trips_(2023-2024)_20260307.csv"
# 输入订单ID，终端输出该ID的所有信息
ORDER_ID = "42c109f58768235ed50173bbff1b54d6d0c3cba9"

CHUNKSIZE = 300000

# 显示完整内容，避免终端省略字段
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 300)

found = False

for df in pd.read_csv(
    RAW_DATA_FILE,
    chunksize=CHUNKSIZE,
    low_memory=False,
    dtype={"Trip ID": "string"}
):
    matched = df[df["Trip ID"].astype("string") == ORDER_ID]

    if len(matched) > 0:
        found = True

        # 一条订单横向字段太多，转置输出更清楚：字段名在左，字段值在右
        print(matched.iloc[0].to_frame(name="原始数据值").to_string())

        break


if not found:
    print(f"原始数据集中没有找到该订单ID：{ORDER_ID}")