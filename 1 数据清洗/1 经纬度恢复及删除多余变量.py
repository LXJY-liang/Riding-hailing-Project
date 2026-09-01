# -*- coding: utf-8 -*-

# 1、删除所有拼车订单
# 2、无上下车点census tract，用经纬度恢复tract；已有tract不覆盖；
# 3、删除无上下车点census tract且无法通过经纬度恢复的订单
# 4、删除所有无关变量，只保留10个关键变量

# 代码运行结果
# 总订单数：7616926；由于拼车删除的订单数：432273
# 修复前上车点坐标缺失总行数：2869328；成功修复上车坐标的总行数：2277595；无法恢复数：591733
# 修复前下车点坐标缺失总行数：2886119；成功修复下车坐标的总行数：2269773；无法恢复数：616346
# 清洗完成后最终保留有效行数：6061035；累计删除数：1555891

import os
import re
import glob
from decimal import Decimal, InvalidOperation

import numpy as np
import pandas as pd
import geopandas as gpd


# ============================================================
# 1. 路径设置
# ============================================================

RAW_DATA_FILE = "/Users/liangxue/Documents/8、Python文件-网约车数据/Transportation_Network_Providers_-_Trips_(2023-2024)_20260307.csv"

TRACT_SHAPE_DIR = "/Users/liangxue/Documents/8、Python文件-网约车数据/Riding hailing_Project/0 芝加哥经纬度恢复/tl_2010_17_tract10"

OUTPUT_DIR = "/Users/liangxue/Documents/8、Python文件-网约车数据/Riding hailing_Project/1 数据清洗"

OUTPUT_NAME = "1 经纬度恢复及删除多余变量"

OUTPUT_KEPT_CSV = os.path.join(OUTPUT_DIR, f"{OUTPUT_NAME}_保留订单.csv")
OUTPUT_DELETED_CSV = os.path.join(OUTPUT_DIR, f"{OUTPUT_NAME}_删除订单.csv")
OUTPUT_SUMMARY_CSV = os.path.join(OUTPUT_DIR, f"{OUTPUT_NAME}_处理统计.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. 参数设置
# ============================================================

CHUNKSIZE = 300000

LAT_MIN, LAT_MAX = 41.0, 43.0
LON_MIN, LON_MAX = -89.0, -86.0

PROJECTED_CRS = "EPSG:26916"
NEAREST_MAX_DISTANCE_METERS = 10

READ_COLS = [
    "Trip ID",
    "Trip Start Timestamp",
    "Trip End Timestamp",
    "Trip Seconds",
    "Trip Miles",
    "Pickup Census Tract",
    "Dropoff Census Tract",
    "Fare",
    "Additional Charges",
    "Shared Trip Authorized",
    "Pickup Centroid Latitude",
    "Pickup Centroid Longitude",
    "Pickup Centroid Location",
    "Dropoff Centroid Latitude",
    "Dropoff Centroid Longitude",
    "Dropoff Centroid Location",
]

FINAL_COLS = [
    "Trip ID",
    "Trip Start Timestamp",
    "Trip End Timestamp",
    "Trip Seconds",
    "Trip Miles",
    "Pickup Census Tract",
    "Dropoff Census Tract",
    "Fare",
    "Additional Charges",
    "Trip Total fare",
]


# ============================================================
# 3. 工具函数
# ============================================================

def find_shp_file(folder):
    """找到 shapefile 文件夹中的 .shp 文件。"""
    files = glob.glob(os.path.join(folder, "*.shp"))
    if not files:
        files = glob.glob(os.path.join(folder, "**", "*.shp"), recursive=True)
    if not files:
        raise FileNotFoundError(f"没有找到 .shp 文件：{folder}")
    return files[0]


def clean_number(s):
    """清洗金额、秒数、英里数：去掉 $、逗号、空格后转成数值。"""
    return pd.to_numeric(
        s.astype(str)
         .str.replace("$", "", regex=False)
         .str.replace(",", "", regex=False)
         .str.strip()
         .replace({"": np.nan, "nan": np.nan, "None": np.nan, "<NA>": np.nan}),
        errors="coerce"
    )


def clean_tract(x):
    """把 Census Tract 清洗成 11 位字符串，避免 .0 和科学计数法。"""
    if pd.isna(x):
        return pd.NA

    x = str(x).strip()

    if x == "" or x.lower() in ["nan", "none", "null", "<na>"]:
        return pd.NA

    try:
        if "." in x or "e" in x.lower():
            x = str(int(Decimal(x)))
    except (InvalidOperation, ValueError):
        return pd.NA

    x = re.sub(r"\D", "", x)

    if x == "":
        return pd.NA

    x = x.zfill(11)

    return x if len(x) == 11 else pd.NA


def parse_time(s):
    """解析时间，并四舍五入到最接近的 15 分钟。"""
    t = pd.to_datetime(s, errors="coerce", format="%m/%d/%Y %I:%M:%S %p")
    bad = t.isna()

    if bad.any():
        t.loc[bad] = pd.to_datetime(s.loc[bad], errors="coerce")

    return t.dt.round("15min")


def parse_shared_true(s):
    """判断 Shared Trip Authorized 是否为 True。"""
    return s.astype(str).str.strip().str.lower().isin(["true", "t", "1", "yes", "y"])


def parse_point_location(df, lat_col, lon_col, point_col):
    """经纬度为空时，从 POINT(lon lat) 字段中恢复经纬度。"""
    if point_col not in df.columns:
        return df

    need = df[lat_col].isna() | df[lon_col].isna()

    if need.sum() == 0:
        return df

    temp = df.loc[need, point_col].astype(str).str.extract(
        r"POINT\s*\(\s*([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s*\)"
    )

    idx = df.loc[need].index

    lon = pd.to_numeric(temp[0], errors="coerce")
    lat = pd.to_numeric(temp[1], errors="coerce")

    df.loc[idx, lon_col] = df.loc[idx, lon_col].fillna(pd.Series(lon.values, index=idx))
    df.loc[idx, lat_col] = df.loc[idx, lat_col].fillna(pd.Series(lat.values, index=idx))

    return df


def recover_tract(df, tract_col, lat_col, lon_col, tracts_4326, tracts_projected):
    """只对缺失 Census Tract 的订单，用经纬度恢复 tract；已有 tract 不覆盖。"""
    result = df[tract_col].copy()

    need = (
        result.isna()
        & df[lat_col].notna()
        & df[lon_col].notna()
        & df[lat_col].between(LAT_MIN, LAT_MAX)
        & df[lon_col].between(LON_MIN, LON_MAX)
    )

    if need.sum() == 0:
        return result

    idx = df.index[need]

    points = gpd.GeoDataFrame(
        {"row_id": idx},
        geometry=gpd.points_from_xy(df.loc[idx, lon_col], df.loc[idx, lat_col]),
        crs="EPSG:4326"
    ).set_index("row_id")

    joined = gpd.sjoin(
        points,
        tracts_4326[["GEOID", "geometry"]],
        how="left",
        predicate="within"
    )

    joined = joined[~joined.index.duplicated(keep="first")]
    hit = joined["GEOID"].notna()

    if hit.any():
        result.loc[joined.index[hit]] = joined.loc[hit, "GEOID"].astype(str).values

    still_need = (
        result.isna()
        & df[lat_col].notna()
        & df[lon_col].notna()
        & df[lat_col].between(LAT_MIN, LAT_MAX)
        & df[lon_col].between(LON_MIN, LON_MAX)
    )

    if still_need.sum() == 0:
        return result

    idx2 = df.index[still_need]

    points2 = gpd.GeoDataFrame(
        {"row_id": idx2},
        geometry=gpd.points_from_xy(df.loc[idx2, lon_col], df.loc[idx2, lat_col]),
        crs="EPSG:4326"
    ).set_index("row_id").to_crs(PROJECTED_CRS)

    nearest = gpd.sjoin_nearest(
        points2,
        tracts_projected[["GEOID", "geometry"]],
        how="left",
        max_distance=NEAREST_MAX_DISTANCE_METERS,
        distance_col="distance_m"
    )

    nearest = nearest[~nearest.index.duplicated(keep="first")]
    hit2 = nearest["GEOID"].notna()

    if hit2.any():
        result.loc[nearest.index[hit2]] = nearest.loc[hit2, "GEOID"].astype(str).values

    return result


def format_output(df):
    """只保留10个关键变量，并把时间强制保存为完整日期时间格式。"""
    out = df[FINAL_COLS].copy()

    out["Trip Start Timestamp"] = pd.to_datetime(out["Trip Start Timestamp"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    out["Trip End Timestamp"] = pd.to_datetime(out["Trip End Timestamp"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")

    return out


# ============================================================
# 4. 读取 Census Tract 地图
# ============================================================

shp_file = find_shp_file(TRACT_SHAPE_DIR)

tracts = gpd.read_file(shp_file)

if "GEOID10" in tracts.columns:
    geoid_col = "GEOID10"
elif "GEOID" in tracts.columns:
    geoid_col = "GEOID"
else:
    raise ValueError(f"shapefile 中没有 GEOID10 或 GEOID 字段，当前字段为：{list(tracts.columns)}")

tracts = tracts[[geoid_col, "geometry"]].rename(columns={geoid_col: "GEOID"}).copy()
tracts["GEOID"] = tracts["GEOID"].astype(str).str.zfill(11)

tracts_4326 = tracts.to_crs("EPSG:4326")
tracts_projected = tracts_4326.to_crs(PROJECTED_CRS)

valid_tracts = set(tracts_4326["GEOID"])


# ============================================================
# 5. 删除旧输出文件
# ============================================================

for file in [OUTPUT_KEPT_CSV, OUTPUT_DELETED_CSV, OUTPUT_SUMMARY_CSV]:
    if os.path.exists(file):
        os.remove(file)


# ============================================================
# 6. 分块清洗、恢复 tract、输出保留订单和删除订单
# ============================================================

kept_header_written = False
deleted_header_written = False

summary_rows = []

reader = pd.read_csv(
    RAW_DATA_FILE,
    usecols=lambda c: c in READ_COLS,
    chunksize=CHUNKSIZE,
    low_memory=False,
    dtype={
        "Trip ID": "string",
        "Pickup Census Tract": "string",
        "Dropoff Census Tract": "string",
    }
)

for chunk_id, df in enumerate(reader, start=1):
    raw_rows = len(df)

    for col in READ_COLS:
        if col not in df.columns:
            df[col] = np.nan

    # ---------- 时间清洗 ----------
    df["Trip Start Timestamp"] = parse_time(df["Trip Start Timestamp"])
    df["Trip End Timestamp"] = parse_time(df["Trip End Timestamp"])

    # ---------- 数值清洗 ----------
    df["Trip Seconds"] = clean_number(df["Trip Seconds"])
    df["Trip Miles"] = clean_number(df["Trip Miles"])
    df["Fare"] = clean_number(df["Fare"])
    df["Additional Charges"] = clean_number(df["Additional Charges"])

    # ---------- 拼车订单标记 ----------
    shared_true = parse_shared_true(df["Shared Trip Authorized"])

    # ---------- Census Tract 原始值清洗 ----------
    df["Pickup Census Tract"] = df["Pickup Census Tract"].apply(clean_tract)
    df["Dropoff Census Tract"] = df["Dropoff Census Tract"].apply(clean_tract)

    pickup_missing_before = int(df["Pickup Census Tract"].isna().sum())
    dropoff_missing_before = int(df["Dropoff Census Tract"].isna().sum())

    # ---------- 经纬度清洗 ----------
    df["pickup_lat"] = pd.to_numeric(df["Pickup Centroid Latitude"], errors="coerce")
    df["pickup_lon"] = pd.to_numeric(df["Pickup Centroid Longitude"], errors="coerce")
    df["dropoff_lat"] = pd.to_numeric(df["Dropoff Centroid Latitude"], errors="coerce")
    df["dropoff_lon"] = pd.to_numeric(df["Dropoff Centroid Longitude"], errors="coerce")

    df = parse_point_location(df, "pickup_lat", "pickup_lon", "Pickup Centroid Location")
    df = parse_point_location(df, "dropoff_lat", "dropoff_lon", "Dropoff Centroid Location")

    # ---------- 用经纬度恢复缺失 Census Tract ----------
    df["Pickup Census Tract"] = recover_tract(
        df, "Pickup Census Tract", "pickup_lat", "pickup_lon", tracts_4326, tracts_projected
    )

    df["Dropoff Census Tract"] = recover_tract(
        df, "Dropoff Census Tract", "dropoff_lat", "dropoff_lon", tracts_4326, tracts_projected
    )

    pickup_missing_after = int(df["Pickup Census Tract"].isna().sum())
    dropoff_missing_after = int(df["Dropoff Census Tract"].isna().sum())

    # ---------- 新变量：Trip Total fare = Fare + Additional Charges ----------
    df["Trip Total fare"] = df["Fare"] + df["Additional Charges"]

    # ---------- 删除条件 ----------
    keep = (
        (~shared_true)
        & df["Trip ID"].notna()
        & df["Trip Start Timestamp"].notna()
        & df["Trip Seconds"].notna()
        & (df["Trip Seconds"] > 0)
        & df["Trip Miles"].notna()
        & (df["Trip Miles"] >= 0)
        & df["Fare"].notna()
        & (df["Fare"] >= 0)
        & df["Additional Charges"].notna()
        & (df["Additional Charges"] >= 0)
        & df["Pickup Census Tract"].notna()
        & df["Dropoff Census Tract"].notna()
        & df["Pickup Census Tract"].isin(valid_tracts)
        & df["Dropoff Census Tract"].isin(valid_tracts)
    )

    kept = format_output(df.loc[keep])
    deleted = format_output(df.loc[~keep])

    # ---------- 输出保留订单 ----------
    if len(kept) > 0:
        kept.to_csv(
            OUTPUT_KEPT_CSV,
            mode="a",
            index=False,
            header=not kept_header_written,
            encoding="utf-8-sig"
        )
        kept_header_written = True

    # ---------- 输出删除订单 ----------
    if len(deleted) > 0:
        deleted.to_csv(
            OUTPUT_DELETED_CSV,
            mode="a",
            index=False,
            header=not deleted_header_written,
            encoding="utf-8-sig"
        )
        deleted_header_written = True

    # ---------- 统计 ----------
    summary_rows.append({
        "chunk_id": chunk_id,
        "raw_rows": raw_rows,
        "shared_true_deleted": int(shared_true.sum()),
        "pickup_missing_before_recovery": pickup_missing_before,
        "pickup_missing_after_recovery": pickup_missing_after,
        "pickup_recovered": pickup_missing_before - pickup_missing_after,
        "dropoff_missing_before_recovery": dropoff_missing_before,
        "dropoff_missing_after_recovery": dropoff_missing_after,
        "dropoff_recovered": dropoff_missing_before - dropoff_missing_after,
        "kept_rows": len(kept),
        "deleted_rows": len(deleted),
    })

    print(
        f"chunk {chunk_id}: 原始={raw_rows}, "
        f"拼车删除={int(shared_true.sum())}, "
        f"上车tract缺失 {pickup_missing_before}->{pickup_missing_after}, "
        f"下车tract缺失 {dropoff_missing_before}->{dropoff_missing_after}, "
        f"保留={len(kept)}, 删除={len(deleted)}"
    )


# ============================================================
# 7. 输出统计文件
# ============================================================

summary_df = pd.DataFrame(summary_rows)

total_row = {"chunk_id": "TOTAL"}

for col in summary_df.columns:
    if col != "chunk_id":
        total_row[col] = summary_df[col].sum()

summary_df = pd.concat([summary_df, pd.DataFrame([total_row])], ignore_index=True)

summary_df.to_csv(
    OUTPUT_SUMMARY_CSV,
    index=False,
    encoding="utf-8-sig"
)

print("完成")
print(f"保留订单CSV：{OUTPUT_KEPT_CSV}")
print(f"删除订单CSV：{OUTPUT_DELETED_CSV}")
print(f"处理统计CSV：{OUTPUT_SUMMARY_CSV}")