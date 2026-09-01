# -*- coding: utf-8 -*-

"""
芝加哥官方 census tract 数量： 801
OD-15min 聚合结果总样本数： 4667146;OD-15min 聚合表里一共有 4,667,146 行，每一行是一个有订单出现过的 (i,j,t)，不是一笔订单。
OD-15min 聚合结果 i 区域数量： 1216;原始 OD 聚合表里，出发区域出现了 1216 个，到达区域出现了 1224 个不同 census tract。
OD-15min 聚合结果 j 区域数量： 1224
属于芝加哥内部 OD 的样本数： 4465517;有 4,465,517 行满足 i 和 j 都属于芝加哥 801 个 census tract。
不属于芝加哥内部 OD 的样本数： 201629;有 201,629 行不满足 Chicago 内部 OD 条件。包括三类：i在j不在；i不在j在；ij都不在
属于芝加哥内部 OD 的 OD 对数量： 206772;在保留下来的 Chicago 内部样本里，共出现过 206,772 个不同的 (i,j) 组合。
不属于芝加哥内部 OD 的 OD 对数量： 45431;被剔除的样本里，共涉及 45,431 个不同 OD 对。
原始 D_ijt_observed 总和： 6061035;OD聚合表对应的原始订单总数是 6,061,035 笔
属于芝加哥内部 OD 的 D_ijt_observed 总和： 5822811;有5,822,811 笔订单是 i 和 j 都在芝加哥 801 个 census tract 内部的订单。
不属于芝加哥内部 OD 的 D_ijt_observed 总和： 238224;有 238,224 笔订单涉及芝加哥边界外区域
"""

import os
import pandas as pd


# ============================================================
# 1. 路径设置
# ============================================================

# OD-15min 聚合结果文件
OD_FILE = "/Users/liangxue/Documents/8、Python文件-网约车数据/Riding hailing_Project/1 数据清洗/2 OD-15min 聚合结果.parquet"

# 芝加哥官方 census tract 编号文件
# 其中 GEOID10 是芝加哥 801 个 census tract 的完整编号
CHICAGO_TRACT_FILE = "/Users/liangxue/Documents/8、Python文件-网约车数据/Riding hailing_Project/0 芝加哥census tract编号及对应的community/Census Tracts TIGER 2010.csv"

# 输出目录
OUTPUT_DIR = "/Users/liangxue/Documents/8、Python文件-网约车数据/Riding hailing_Project/1 数据清洗"

# 输出文件：i 和 j 都属于芝加哥 801 个 census tract 的 OD-15min 样本
OUTPUT_CHICAGO_FILE = os.path.join(
    OUTPUT_DIR,
    "2_2 OD-15min聚合_属于芝加哥.parquet"
)

# 输出文件：i 或 j 至少有一个不属于芝加哥 801 个 census tract 的 OD-15min 样本
OUTPUT_NON_CHICAGO_FILE = os.path.join(
    OUTPUT_DIR,
    "2_2 OD-15min聚合_不属于芝加哥.parquet"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. 工具函数：统一 census tract 编号格式
# ============================================================

def clean_region_id(series):
    """
    统一 census tract 编号格式：
    1. 转为字符串；
    2. 去除前后空格；
    3. 去除可能出现的 .0；
    4. 提取数字部分；
    5. 补齐为 11 位 GEOID。
    """
    s = series.astype("string").str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    s = s.str.extract(r"(\d+)", expand=False)
    s = s.str.zfill(11)
    return s


# ============================================================
# 3. 读取芝加哥官方 801 个 census tract 编号
# ============================================================

chicago_tract = pd.read_csv(
    CHICAGO_TRACT_FILE,
    dtype="string",
    encoding="utf-8-sig"
)

# 优先使用 GEOID10 字段；如果字段名不存在，则使用第五列
if "GEOID10" in chicago_tract.columns:
    chicago_id_col = "GEOID10"
else:
    chicago_id_col = chicago_tract.columns[4]

chicago_tract["region_id"] = clean_region_id(chicago_tract[chicago_id_col])

chicago_region_ids = (
    chicago_tract["region_id"]
    .dropna()
    .drop_duplicates()
    .sort_values()
    .tolist()
)

chicago_region_set = set(chicago_region_ids)

if len(chicago_region_set) != 801:
    raise ValueError(f"芝加哥官方 census tract 数量为 {len(chicago_region_set)}，不是预期的 801。")


# ============================================================
# 4. 读取 OD-15min 聚合结果
# ============================================================

od = pd.read_parquet(OD_FILE)

if "i" not in od.columns or "j" not in od.columns:
    raise ValueError("OD-15min 聚合结果中缺少 i 或 j 字段。")

od["i"] = clean_region_id(od["i"])
od["j"] = clean_region_id(od["j"])

if "t" in od.columns:
    od["t"] = pd.to_datetime(od["t"], errors="coerce")


# ============================================================
# 5. 判断 OD 是否属于芝加哥内部 OD
# ============================================================

# 属于芝加哥内部 OD 的定义：
# 出发区域 i 属于芝加哥 801 个 census tract；
# 到达区域 j 也属于芝加哥 801 个 census tract。
is_i_chicago = od["i"].isin(chicago_region_set)
is_j_chicago = od["j"].isin(chicago_region_set)

is_chicago_od = is_i_chicago & is_j_chicago

od_chicago = od[is_chicago_od].copy()
od_non_chicago = od[~is_chicago_od].copy()


# ============================================================
# 6. 保存筛选结果
# ============================================================

if os.path.exists(OUTPUT_CHICAGO_FILE):
    os.remove(OUTPUT_CHICAGO_FILE)

if os.path.exists(OUTPUT_NON_CHICAGO_FILE):
    os.remove(OUTPUT_NON_CHICAGO_FILE)

od_chicago.to_parquet(
    OUTPUT_CHICAGO_FILE,
    index=False,
    engine="pyarrow",
    compression="snappy"
)

od_non_chicago.to_parquet(
    OUTPUT_NON_CHICAGO_FILE,
    index=False,
    engine="pyarrow",
    compression="snappy"
)


# ============================================================
# 7. 终端输出检查结果
# ============================================================

print("完成")
print("芝加哥官方 census tract 数量：", len(chicago_region_set))
print("OD-15min 聚合结果总样本数：", len(od))
print("OD-15min 聚合结果 i 区域数量：", od["i"].nunique())
print("OD-15min 聚合结果 j 区域数量：", od["j"].nunique())
print("属于芝加哥内部 OD 的样本数：", len(od_chicago))
print("不属于芝加哥内部 OD 的样本数：", len(od_non_chicago))
print("属于芝加哥内部 OD 的 OD 对数量：", od_chicago[["i", "j"]].drop_duplicates().shape[0])
print("不属于芝加哥内部 OD 的 OD 对数量：", od_non_chicago[["i", "j"]].drop_duplicates().shape[0])

if "D_ijt_observed" in od.columns:
    print("原始 D_ijt_observed 总和：", int(od["D_ijt_observed"].sum()))
    print("属于芝加哥内部 OD 的 D_ijt_observed 总和：", int(od_chicago["D_ijt_observed"].sum()))
    print("不属于芝加哥内部 OD 的 D_ijt_observed 总和：", int(od_non_chicago["D_ijt_observed"].sum()))

print("输出文件1：", OUTPUT_CHICAGO_FILE)
print("输出文件2：", OUTPUT_NON_CHICAGO_FILE)