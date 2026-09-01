# -*- coding: utf-8 -*-

"""随机检查最终的芝加哥内部 OD 口径区域-15min完整面板。"""

import pandas as pd


# ============================================================
# 1. 路径设置
# ============================================================

OUTPUT_FILE = "/Users/liangxue/Documents/8、Python文件-网约车数据/Riding hailing_Project/1 数据清洗/3_3 区域-15min聚合（补全空值）结果.parquet"


# ============================================================
# 2. 读取补全后的完整面板
# ============================================================

df = pd.read_parquet(OUTPUT_FILE)

df["t"] = pd.to_datetime(df["t"], errors="coerce")


# ============================================================
# 3. 随机抽取10个样本
# random_state 固定后，每次运行输出结果一致，方便反复检查
# ============================================================

sample_df = df.sample(n=10, random_state=78).sort_values(["region_id", "t"])


# ============================================================
# 4. 终端完整显示10个样本
# ============================================================

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 300)
pd.set_option("display.max_colwidth", 100)

print("补全后文件样本数：", len(df))
print("补全后区域数量：", df["region_id"].nunique())
print("补全后时间段数量：", df["t"].nunique())
print("补全后时间范围：", df["t"].min(), "到", df["t"].max())
print("补全后样本数是否801×2880：", len(df) == 801 * 2880)
print("重复(region_id,t)数：", int(df.duplicated(["region_id", "t"]).sum()))
print("D_origin_it总和：", int(df["D_origin_it"].sum()))
print("D_dest_it总和：", int(df["D_dest_it"].sum()))

if "day_type" in df.columns:
    print("日期类型分布：")
    print(df["day_type"].value_counts(dropna=False).to_string())

print("\n随机抽取的10个样本：")
print(sample_df.to_string(index=False))
