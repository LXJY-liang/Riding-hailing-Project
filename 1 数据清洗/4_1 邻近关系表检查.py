# -*- coding: utf-8 -*-

import pandas as pd


# ============================================================
# 1. 路径设置
# ============================================================

NEIGHBOR_FILE = "/Users/liangxue/Documents/8、Python文件-网约车数据/Riding hailing_Project/1 数据清洗/4 邻近区域关系表构造结果.parquet"


# ============================================================
# 2. 读取邻近关系表
# ============================================================

neighbor = pd.read_parquet(NEIGHBOR_FILE)

neighbor["region_id"] = neighbor["region_id"].astype("string")
neighbor["neighbor_region_id"] = neighbor["neighbor_region_id"].astype("string")


# ============================================================
# 3. 设置终端显示格式
# ============================================================

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 300)
pd.set_option("display.max_colwidth", 100)


# ============================================================
# 4. 基础检查
# ============================================================

neighbor_count = (
    neighbor.groupby("region_id", as_index=False)
    .agg(
        neighbor_count=("neighbor_region_id", "count"),
        weight_sum=("weight", "sum"),
        mean_distance=("distance_km", "mean"),
        max_distance=("distance_km", "max"),
        boundary_neighbor_count=("is_boundary_neighbor", "sum"),
        fallback_flag=("neighbor_fallback_flag", "max"),
    )
)

min_neighbor_count = neighbor_count["neighbor_count"].min()
max_neighbor_count = neighbor_count["neighbor_count"].max()

min_regions = neighbor_count.loc[
    neighbor_count["neighbor_count"] == min_neighbor_count,
    "region_id"
].tolist()

max_regions = neighbor_count.loc[
    neighbor_count["neighbor_count"] == max_neighbor_count,
    "region_id"
].tolist()

print("=" * 120)
print("一、邻近关系表基础检查")
print("=" * 120)
print("邻近关系记录数：", len(neighbor))
print("region_id 数量：", neighbor["region_id"].nunique())
print("neighbor_region_id 数量：", neighbor["neighbor_region_id"].nunique())
print("每个区域最少邻居数：", int(min_neighbor_count))
print("每个区域最多邻居数：", int(max_neighbor_count))
print("每个区域平均邻居数：", round(float(neighbor_count["neighbor_count"].mean()), 4))
print("权重和最小值：", round(float(neighbor_count["weight_sum"].min()), 8))
print("权重和最大值：", round(float(neighbor_count["weight_sum"].max()), 8))
print("使用兜底最近邻的区域数：", int(neighbor_count["fallback_flag"].sum()))


# ============================================================
# 5. 随机输出10个区域的完整邻近关系样本
# ============================================================

random_regions = (
    neighbor_count["region_id"]
    .sample(n=10, random_state=42)
    .tolist()
)

random_sample = (
    neighbor[neighbor["region_id"].isin(random_regions)]
    .sort_values(["region_id", "distance_km", "neighbor_region_id"])
    .copy()
)

print("\n" + "=" * 120)
print("二、随机10个区域的邻近区域完整样本")
print("=" * 120)
print("随机抽取的10个 region_id：")
print(random_regions)

print("\n随机10个区域的邻近关系明细：")
print(random_sample.to_string(index=False))


# ============================================================
# 6. 输出只有最少邻居数的区域完整样本
# ============================================================

min_sample = (
    neighbor[neighbor["region_id"].isin(min_regions)]
    .sort_values(["region_id", "distance_km", "neighbor_region_id"])
    .copy()
)

print("\n" + "=" * 120)
print("三、只有最少邻居数的区域完整样本")
print("=" * 120)
print("最少邻居数：", int(min_neighbor_count))
print("最少邻居数区域数量：", len(min_regions))
print("最少邻居数 region_id：")
print(min_regions)

print("\n最少邻居数区域的邻近关系明细：")
print(min_sample.to_string(index=False))


# ============================================================
# 7. 输出有最多邻居数的区域完整样本
# ============================================================

max_sample = (
    neighbor[neighbor["region_id"].isin(max_regions)]
    .sort_values(["region_id", "distance_km", "neighbor_region_id"])
    .copy()
)

print("\n" + "=" * 120)
print("四、有最多邻居数的区域完整样本")
print("=" * 120)
print("最多邻居数：", int(max_neighbor_count))
print("最多邻居数区域数量：", len(max_regions))
print("最多邻居数 region_id：")
print(max_regions)

print("\n最多邻居数区域的邻近关系明细：")
print(max_sample.to_string(index=False))