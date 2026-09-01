# -*- coding: utf-8 -*-
"""
2_3 OD-15min聚合（补全时间、不管没出现的OD）.py

功能：
1. 读取已筛选的 Chicago 内部 OD-15min 聚合数据；
2. 提取历史上至少出现过一次订单的 OD 对；
3. 只对这些历史活跃 OD 对补全 2024年6月全部 2880 个15min时间段；
4. 缺失的 OD-时段订单数 D_ijt_observed 补0；
5. 计算 OD_id、same_region_ij、dist_ij；
6. 融合时间特征和天气特征；
7. 分块写出，避免内存溢出。

输出：
2_3 OD-15min聚合（补全时间、不管没出现的OD）结果.parquet
"""

"""
运行结果

输出文件： /Users/liangxue/Documents/8、Python文件-网约车数据/Riding hailing_Project/1 数据清洗/2_3 OD-15min聚合（补全时间、不管没出现的OD）结果.parquet
历史活跃 OD 对数量： 206772
时间段数量： 2880
理论完整面板行数： 595503360
实际写入行数： 595503360
原始 OD-15min 样本数： 4465517
原始 D_ijt_observed 总和： 5822811
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import pyarrow as pa
import pyarrow.parquet as pq
from shapely import wkt


# ============================================================
# 1. 路径设置
# ============================================================

OD_CHICAGO_FILE = "/Users/liangxue/Documents/8、Python文件-网约车数据/Riding hailing_Project/1 数据清洗/2_2 OD-15min聚合_属于芝加哥.parquet"

TRACT_FILE = "/Users/liangxue/Documents/8、Python文件-网约车数据/Riding hailing_Project/0 芝加哥census tract编号及对应的community/Census Tracts TIGER 2010.csv"

WEATHER_FILE = "/Users/liangxue/Documents/8、Python文件-网约车数据/Riding hailing_Project/0 天气数据获取/chicago_weather_2024_06_hourly.csv"

OUTPUT_FILE = "/Users/liangxue/Documents/8、Python文件-网约车数据/Riding hailing_Project/1 数据清洗/2_3 OD-15min聚合（补全时间、不管没出现的OD）结果.parquet"

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)


# ============================================================
# 2. 参数设置
# ============================================================

CHUNK_OD_SIZE = 5000          # 每次处理500个OD对；如果内存够，可以调大到1000
PROJECTED_CRS = "EPSG:26916"


# ============================================================
# 3. 工具函数
# ============================================================

def clean_region_id(series):
    """
    将 census tract 编号统一为11位字符串。
    """
    s = series.astype("string").str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    s = s.str.extract(r"(\d+)", expand=False)
    s = s.str.zfill(11)
    return s


# ============================================================
# 4. 读取 OD-15min 聚合数据
# ============================================================

od = pd.read_parquet(OD_CHICAGO_FILE)

od["i"] = clean_region_id(od["i"])
od["j"] = clean_region_id(od["j"])
od["t"] = pd.to_datetime(od["t"], errors="coerce")

od = od[
    (od["t"] >= pd.Timestamp("2024-06-01 00:00:00"))
    & (od["t"] <= pd.Timestamp("2024-06-30 23:45:00"))
].copy()

if od[["i", "j", "t"]].duplicated().sum() > 0:
    raise ValueError("OD-15min 聚合数据中存在重复的 i + j + t。")

od["OD_id"] = od["i"].astype(str) + "_" + od["j"].astype(str)


# ============================================================
# 5. 提取历史活跃 OD 对
# ============================================================

active_od = (
    od[["i", "j", "OD_id"]]
    .drop_duplicates()
    .sort_values(["i", "j"])
    .reset_index(drop=True)
)

active_od["same_region_ij"] = (active_od["i"] == active_od["j"]).astype("int8")


# ============================================================
# 6. 构造 2024年6月完整15min时间表
# ============================================================

time_features = pd.DataFrame({
    "t": pd.date_range(
        start="2024-06-01 00:00:00",
        end="2024-06-30 23:45:00",
        freq="15min"
    )
})

if len(time_features) != 2880:
    raise ValueError(f"时间段数量为 {len(time_features)}，不是预期的2880。")

time_features["day_index"] = (
    time_features["t"].dt.normalize() - pd.Timestamp("2024-06-01")
).dt.days + 1

time_features["slot_in_day"] = (
    time_features["t"].dt.hour * 4 + time_features["t"].dt.minute // 15
)

time_features["dow"] = time_features["t"].dt.dayofweek

holiday_dates = {pd.Timestamp("2024-06-19").date()}

time_features["is_weekend_or_holiday"] = (
    time_features["dow"].isin([5, 6]) | time_features["t"].dt.date.isin(holiday_dates)
).astype("int8")

time_features["is_peak_morning"] = (
    (time_features["t"].dt.hour >= 7) & (time_features["t"].dt.hour < 9)
).astype("int8")

time_features["is_peak_evening"] = (
    (time_features["t"].dt.hour >= 16) & (time_features["t"].dt.hour < 19)
).astype("int8")

time_features["is_night"] = (
    (time_features["t"].dt.hour >= 22) | (time_features["t"].dt.hour < 6)
).astype("int8")


# ============================================================
# 7. 融合天气特征
# 优先从现有 OD 聚合表按 t 提取天气；如果有缺失，再用天气文件补
# ============================================================

weather_cols = [
    "temperature_2m",
    "relative_humidity_2m",
    "rain",
    "weather_code",
    "wind_speed_10m",
]

od_has_weather = all(col in od.columns for col in weather_cols)

if od_has_weather:
    weather_from_od = (
        od[["t"] + weather_cols]
        .drop_duplicates(subset=["t"])
        .sort_values("t")
        .reset_index(drop=True)
    )

    time_features = time_features.merge(
        weather_from_od,
        on="t",
        how="left"
    )
else:
    for col in weather_cols:
        time_features[col] = np.nan

if time_features[weather_cols].isna().sum().sum() > 0:
    weather = pd.read_csv(WEATHER_FILE)
    weather["time"] = pd.to_datetime(weather["time"], errors="coerce")

    weather = weather[["time"] + weather_cols].copy()

    for col in weather_cols:
        weather[col] = pd.to_numeric(weather[col], errors="coerce")

    time_features["weather_hour"] = time_features["t"].dt.floor("h")

    time_features = time_features.merge(
        weather,
        left_on="weather_hour",
        right_on="time",
        how="left",
        suffixes=("", "_from_file")
    )

    for col in weather_cols:
        time_features[col] = time_features[col].fillna(time_features[f"{col}_from_file"])

    drop_cols = ["weather_hour", "time"] + [f"{col}_from_file" for col in weather_cols]
    time_features = time_features.drop(columns=drop_cols)

if time_features[weather_cols].isna().sum().sum() > 0:
    raise ValueError("时间天气表仍然存在天气缺失。")


# ============================================================
# 8. 读取 census tract 边界并计算质心
# ============================================================

tract_raw = pd.read_csv(TRACT_FILE, dtype="string", encoding="utf-8-sig")

tract_id_col = "GEOID10" if "GEOID10" in tract_raw.columns else tract_raw.columns[4]

if "the_geom" in tract_raw.columns:
    geom_col = "the_geom"
elif "THE_GEOM" in tract_raw.columns:
    geom_col = "THE_GEOM"
else:
    raise ValueError(f"没有找到 the_geom 字段。当前字段为：{list(tract_raw.columns)}")

tract_raw["region_id"] = clean_region_id(tract_raw[tract_id_col])
tract_raw["geometry"] = tract_raw[geom_col].apply(wkt.loads)

tract_gdf = gpd.GeoDataFrame(
    tract_raw[["region_id", "geometry"]].copy(),
    geometry="geometry",
    crs="EPSG:4326"
)

tract_gdf = (
    tract_gdf.drop_duplicates(subset=["region_id"])
    .sort_values("region_id")
    .reset_index(drop=True)
)

if len(tract_gdf) != 801:
    raise ValueError(f"census tract 数量为 {len(tract_gdf)}，不是预期的801。")

tract_proj = tract_gdf.to_crs(PROJECTED_CRS).copy()
tract_proj["centroid"] = tract_proj.geometry.centroid

centroid = pd.DataFrame({
    "region_id": tract_proj["region_id"].astype("string"),
    "x": tract_proj["centroid"].x,
    "y": tract_proj["centroid"].y,
})


# ============================================================
# 9. 计算每个历史活跃 OD 对的 dist_ij
# ============================================================

active_od = active_od.merge(
    centroid.rename(columns={"region_id": "i", "x": "x_i", "y": "y_i"}),
    on="i",
    how="left"
)

active_od = active_od.merge(
    centroid.rename(columns={"region_id": "j", "x": "x_j", "y": "y_j"}),
    on="j",
    how="left"
)

if active_od[["x_i", "y_i", "x_j", "y_j"]].isna().sum().sum() > 0:
    raise ValueError("存在无法匹配质心坐标的 OD 对。")

active_od["dist_ij"] = (
    np.sqrt((active_od["x_i"] - active_od["x_j"]) ** 2 + (active_od["y_i"] - active_od["y_j"]) ** 2)
    / 1000.0
).astype("float32")

active_od = active_od[
    ["i", "j", "OD_id", "dist_ij", "same_region_ij"]
].copy()


# ============================================================
# 10. 整理原 OD 观测字段
# 时间、天气、OD定位字段后面统一生成，不从原表重复保留
# ============================================================

time_cols = [
    "day_index",
    "slot_in_day",
    "dow",
    "is_weekend_or_holiday",
    "is_peak_morning",
    "is_peak_evening",
    "is_night",
]

exclude_cols = ["i", "j", "t", "OD_id"] + time_cols + weather_cols

obs_value_cols = [c for c in od.columns if c not in exclude_cols]

od_obs = od[["OD_id", "t"] + obs_value_cols].copy()

if "D_ijt_observed" not in od_obs.columns:
    raise ValueError("原 OD 聚合数据中缺少 D_ijt_observed 字段。")

for col in obs_value_cols:
    od_obs[col] = pd.to_numeric(od_obs[col], errors="coerce")


# ============================================================
# 11. 分块构造完整 OD-15min 面板
# ============================================================

if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

writer = None
total_rows = 0

time_n = len(time_features)
time_arrays = {
    col: time_features[col].to_numpy()
    for col in time_features.columns
}

for start in range(0, len(active_od), CHUNK_OD_SIZE):
    end = min(start + CHUNK_OD_SIZE, len(active_od))

    od_chunk = active_od.iloc[start:end].copy()
    od_n = len(od_chunk)

    # --------------------------------------------------------
    # 11.1 用 repeat/tile 构造笛卡尔积，比 merge(_key) 快很多
    # --------------------------------------------------------

    panel = pd.DataFrame({
        "i": np.repeat(od_chunk["i"].to_numpy(), time_n),
        "j": np.repeat(od_chunk["j"].to_numpy(), time_n),
        "OD_id": np.repeat(od_chunk["OD_id"].to_numpy(), time_n),
        "dist_ij": np.repeat(od_chunk["dist_ij"].to_numpy(), time_n),
        "same_region_ij": np.repeat(od_chunk["same_region_ij"].to_numpy(), time_n),
        "t": np.tile(time_arrays["t"], od_n),
    })

    for col in time_cols + weather_cols:
        panel[col] = np.tile(time_arrays[col], od_n)

    # --------------------------------------------------------
    # 11.2 合并已有 OD-15min 观测值
    # --------------------------------------------------------

    chunk_od_ids = set(od_chunk["OD_id"].astype(str))
    od_obs_chunk = od_obs[od_obs["OD_id"].astype(str).isin(chunk_od_ids)].copy()

    panel = panel.merge(
        od_obs_chunk,
        on=["OD_id", "t"],
        how="left"
    )

    # --------------------------------------------------------
    # 11.3 没有订单的 OD-时段，D_ijt_observed 补0
    # 其他订单发生后才有意义的均值/分位数字段保留 NaN
    # --------------------------------------------------------

    panel["D_ijt_observed"] = (
        pd.to_numeric(panel["D_ijt_observed"], errors="coerce")
        .fillna(0)
        .astype("int32")
    )

    # --------------------------------------------------------
    # 11.4 字段顺序
    # --------------------------------------------------------

    first_cols = [
        "i",
        "j",
        "t",
        "OD_id",
        "dist_ij",
        "same_region_ij",
    ]

    final_cols = (
        first_cols
        + time_cols
        + ["D_ijt_observed"]
        + [c for c in obs_value_cols if c != "D_ijt_observed"]
        + weather_cols
    )

    panel = panel[final_cols].copy()

    # --------------------------------------------------------
    # 11.5 类型压缩
    # --------------------------------------------------------

    panel["i"] = panel["i"].astype("string")
    panel["j"] = panel["j"].astype("string")
    panel["OD_id"] = panel["OD_id"].astype("string")
    panel["t"] = pd.to_datetime(panel["t"], errors="coerce")

    int_cols = [
        "same_region_ij",
        "day_index",
        "slot_in_day",
        "dow",
        "is_weekend_or_holiday",
        "is_peak_morning",
        "is_peak_evening",
        "is_night",
        "D_ijt_observed",
    ]

    for col in int_cols:
        panel[col] = pd.to_numeric(panel[col], errors="coerce").fillna(0).astype("int32")

    float_cols = [
        c for c in panel.columns
        if c not in ["i", "j", "t", "OD_id"] + int_cols
    ]

    for col in float_cols:
        panel[col] = pd.to_numeric(panel[col], errors="coerce").astype("float32")

    # --------------------------------------------------------
    # 11.6 追加写入 Parquet
    # --------------------------------------------------------

    table = pa.Table.from_pandas(panel, preserve_index=False)

    if writer is None:
        writer = pq.ParquetWriter(
            OUTPUT_FILE,
            table.schema,
            compression="snappy"
        )

    writer.write_table(
        table,
        row_group_size=len(panel)
    )

    total_rows += len(panel)

    print(f"已处理 OD 对：{start} - {end - 1}；本块行数：{len(panel)}；累计行数：{total_rows}")

    del panel, table, od_obs_chunk

if writer is not None:
    writer.close()


# ============================================================
# 12. 输出检查信息
# ============================================================

print("完成")
print("输出文件：", OUTPUT_FILE)
print("历史活跃 OD 对数量：", len(active_od))
print("时间段数量：", time_n)
print("理论完整面板行数：", len(active_od) * time_n)
print("实际写入行数：", total_rows)
print("原始 OD-15min 样本数：", len(od))
print("原始 D_ijt_observed 总和：", int(od["D_ijt_observed"].sum()))