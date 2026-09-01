# -*- coding: utf-8 -*-
"""
邻近关系记录数： 9144
region_id 数量： 801
neighbor_region_id 数量： 801
每个区域最少邻居数： 1
每个区域最多邻居数： 29
每个区域平均邻居数： 11.4157
权重和最小值： 0.99999994
权重和最大值： 1.0
使用兜底最近邻的区域数： 0
"""
"""
1、随机抽取10个region，查看其邻近区域有哪些
2、只有最少邻居数（1个）区域的3个样本
3、有最多邻居数（29个）区域的2个样本
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt


# ============================================================
# 1. 路径设置
# ============================================================

# Chicago 801个 census tract 边界数据
TRACT_FILE = "/Users/liangxue/Documents/8、Python文件-网约车数据/Riding hailing_Project/0 芝加哥census tract编号及对应的community/Census Tracts TIGER 2010.csv"

# 输出文件：census tract 邻近关系表
OUTPUT_FILE = "/Users/liangxue/Documents/8、Python文件-网约车数据/Riding hailing_Project/1 数据清洗/4 邻近区域关系表构造结果.parquet"

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)


# ============================================================
# 2. 参数设置
# ============================================================

# 最大邻近距离阈值，单位：km
# 非边界相邻区域只有在质心距离不超过该阈值时才进入邻居集合
MAX_NEIGHBOR_DISTANCE_KM = 1.5

# 距离衰减参数，单位：km
# h 越小，距离远的邻居权重下降越快
DISTANCE_DECAY_H_KM = 0.75

# 共享边界长度阈值，单位：米
# 大于该阈值才认为两个 tract 是边界相邻，避免只在一个点接触也被当成邻居
MIN_SHARED_BOUNDARY_M = 1.0


# ============================================================
# 3. 工具函数：统一 census tract 编号格式
# ============================================================

def clean_region_id(series):
    """
    统一 census tract 编号格式：
    1. 转为字符串；
    2. 去除前后空格；
    3. 去除可能出现的 .0；
    4. 只提取数字；
    5. 补齐为 11 位 GEOID。
    """
    s = series.astype("string").str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    s = s.str.extract(r"(\d+)", expand=False)
    s = s.str.zfill(11)
    return s


# ============================================================
# 4. 读取 census tract 边界数据
# ============================================================

tracts_raw = pd.read_csv(TRACT_FILE, dtype="string", encoding="utf-8-sig")

# 自动识别几何字段
geom_candidates = ["the_geom", "THE_GEOM", "geometry", "Geometry"]
geom_col = None
for col in geom_candidates:
    if col in tracts_raw.columns:
        geom_col = col
        break

if geom_col is None:
    raise ValueError(f"没有找到几何字段。当前字段为：{list(tracts_raw.columns)}")

# 自动识别 census tract ID 字段
id_candidates = ["GEOID10", "geoid10", "GEOID", "geoid"]
id_col = None
for col in id_candidates:
    if col in tracts_raw.columns:
        id_col = col
        break

if id_col is None:
    # 如果字段名无法识别，则使用第五列作为 census tract 编号
    id_col = tracts_raw.columns[4]

tracts_raw["region_id"] = clean_region_id(tracts_raw[id_col])

# 将 WKT 字符串转换为空间几何对象
tracts_raw["geometry"] = tracts_raw[geom_col].apply(wkt.loads)

tracts = gpd.GeoDataFrame(
    tracts_raw[["region_id", "geometry"]].copy(),
    geometry="geometry",
    crs="EPSG:4326"
)

# 去重，确保每个 census tract 只保留一条边界
tracts = (
    tracts.drop_duplicates(subset=["region_id"])
    .sort_values("region_id")
    .reset_index(drop=True)
)

if len(tracts) != 801:
    raise ValueError(f"读取到的 census tract 数量为 {len(tracts)}，不是预期的801。")


# ============================================================
# 5. 转换到米制投影坐标系
# ============================================================

# EPSG:26916 为 NAD83 / UTM zone 16N，适合芝加哥地区距离计算，单位为米
tracts_proj = tracts.to_crs("EPSG:26916").copy()

# 计算每个 tract 的质心，用于计算质心距离
tracts_proj["centroid"] = tracts_proj.geometry.centroid

region_ids = tracts_proj["region_id"].tolist()
n = len(region_ids)


# ============================================================
# 6. 计算所有 census tract 之间的质心距离
# ============================================================

centroid_x = tracts_proj["centroid"].x.to_numpy()
centroid_y = tracts_proj["centroid"].y.to_numpy()

dx = centroid_x[:, None] - centroid_x[None, :]
dy = centroid_y[:, None] - centroid_y[None, :]

distance_m = np.sqrt(dx ** 2 + dy ** 2)
distance_km = distance_m / 1000.0


# ============================================================
# 7. 计算边界相邻关系
# ============================================================

# is_boundary_neighbor_matrix[i, j] = True 表示 i 和 j 共享边界
is_boundary_neighbor_matrix = np.zeros((n, n), dtype=bool)

# 使用空间索引减少不必要的两两几何计算
spatial_index = tracts_proj.sindex

for i in range(n):
    geom_i = tracts_proj.geometry.iloc[i]

    # 先用边界框筛选可能相交的区域
    candidate_indices = list(spatial_index.query(geom_i))

    for j in candidate_indices:
        if j <= i:
            continue

        geom_j = tracts_proj.geometry.iloc[j]

        # 共享边界长度：只统计边界线重合长度，避免仅点接触
        shared_boundary = geom_i.boundary.intersection(geom_j.boundary)
        shared_length_m = shared_boundary.length

        if shared_length_m > MIN_SHARED_BOUNDARY_M:
            is_boundary_neighbor_matrix[i, j] = True
            is_boundary_neighbor_matrix[j, i] = True


# ============================================================
# 8. 构造邻近候选关系
# ============================================================

records = []

for i in range(n):
    region_id = region_ids[i]

    # 条件1：边界相邻
    boundary_mask = is_boundary_neighbor_matrix[i]

    # 条件2：质心距离不超过阈值
    distance_mask = distance_km[i] <= MAX_NEIGHBOR_DISTANCE_KM

    # 排除自己
    self_mask = np.zeros(n, dtype=bool)
    self_mask[i] = True

    # 邻居集合 = 边界相邻区域 + 距离阈值内区域
    neighbor_mask = (boundary_mask | distance_mask) & (~self_mask)

    neighbor_indices = np.where(neighbor_mask)[0].tolist()

    # 如果没有任何邻居，则使用最近的1个区域兜底
    fallback_flag = 0
    if len(neighbor_indices) == 0:
        nearest_idx = np.argsort(distance_km[i])[1]
        neighbor_indices = [nearest_idx]
        fallback_flag = 1

    for j in neighbor_indices:
        records.append(
            {
                "region_id": region_id,
                "neighbor_region_id": region_ids[j],
                "is_boundary_neighbor": int(is_boundary_neighbor_matrix[i, j]),
                "distance_km": float(distance_km[i, j]),
                "neighbor_fallback_flag": fallback_flag,
            }
        )

neighbor_df = pd.DataFrame(records)


# ============================================================
# 9. 计算距离衰减权重
# ============================================================

# 原始影响分数：距离越近，score越大
neighbor_df["score"] = np.exp(
    -neighbor_df["distance_km"] / DISTANCE_DECAY_H_KM
)

# 对每个 region_id 内部归一化，使该区域所有邻居权重和为1
neighbor_df["weight"] = (
    neighbor_df["score"]
    / neighbor_df.groupby("region_id")["score"].transform("sum")
)


# ============================================================
# 10. 增加邻居数量与距离统计字段
# ============================================================

neighbor_stats = (
    neighbor_df.groupby("region_id", as_index=False)
    .agg(
        neighbor_count_i=("neighbor_region_id", "count"),
        neighbor_mean_distance_i=("distance_km", "mean"),
        neighbor_max_distance_i=("distance_km", "max"),
        boundary_neighbor_count_i=("is_boundary_neighbor", "sum"),
    )
)

neighbor_df = neighbor_df.merge(
    neighbor_stats,
    on="region_id",
    how="left"
)


# ============================================================
# 11. 整理字段顺序与数据类型
# ============================================================

final_cols = [
    "region_id",
    "neighbor_region_id",
    "is_boundary_neighbor",
    "distance_km",
    "weight",
    "neighbor_count_i",
    "neighbor_mean_distance_i",
    "neighbor_max_distance_i",
    "boundary_neighbor_count_i",
    "neighbor_fallback_flag",
]

neighbor_df = neighbor_df[final_cols].copy()

neighbor_df["region_id"] = neighbor_df["region_id"].astype("string")
neighbor_df["neighbor_region_id"] = neighbor_df["neighbor_region_id"].astype("string")
neighbor_df["is_boundary_neighbor"] = neighbor_df["is_boundary_neighbor"].astype("int8")
neighbor_df["neighbor_count_i"] = neighbor_df["neighbor_count_i"].astype("int16")
neighbor_df["boundary_neighbor_count_i"] = neighbor_df["boundary_neighbor_count_i"].astype("int16")
neighbor_df["neighbor_fallback_flag"] = neighbor_df["neighbor_fallback_flag"].astype("int8")

float_cols = [
    "distance_km",
    "weight",
    "neighbor_mean_distance_i",
    "neighbor_max_distance_i",
]

for col in float_cols:
    neighbor_df[col] = pd.to_numeric(neighbor_df[col], errors="coerce").astype("float32")


# ============================================================
# 12. 保存邻近关系表
# ============================================================

if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

neighbor_df.to_parquet(
    OUTPUT_FILE,
    index=False,
    engine="pyarrow",
    compression="snappy"
)


# ============================================================
# 13. 终端输出检查结果
# ============================================================

weight_check = (
    neighbor_df.groupby("region_id")["weight"]
    .sum()
    .reset_index(name="weight_sum")
)

print("完成")
print("输出文件：", OUTPUT_FILE)
print("census tract 数量：", len(region_ids))
print("邻近关系记录数：", len(neighbor_df))
print("每个区域最少邻居数：", int(neighbor_df.groupby("region_id").size().min()))
print("每个区域最多邻居数：", int(neighbor_df.groupby("region_id").size().max()))
print("每个区域平均邻居数：", round(float(neighbor_df.groupby("region_id").size().mean()), 2))
print("使用兜底最近邻的区域数：", int(neighbor_df.groupby("region_id")["neighbor_fallback_flag"].max().sum()))
print("权重和最小值：", round(float(weight_check["weight_sum"].min()), 6))
print("权重和最大值：", round(float(weight_check["weight_sum"].max()), 6))