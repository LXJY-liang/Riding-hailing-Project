# -*- coding: utf-8 -*-
"""
4 tauijt特征构造.py

功能：
1. 基于由“1_4异常订单清洗结果”重建的
   2_3 OD-15min聚合（补全时间、不管没出现的OD）结果.parquet 构造 tau_ijt 特征；
2. 每个样本单位为 (i, j, t)；
3. tau_ijt_label = tau_bar_ijt，但仅当 D_ijt_observed > 0 时有效，否则为 NaN；
4. 使用矩阵方式快速构造同一OD短期历史服务时间特征；
5. 使用矩阵方式快速构造同一OD长期服务时间特征；
6. 使用全月相似距离统计构造兜底特征；
7. 输出 parquet 文件。
"""

import os
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ============================================================
# 1. 路径设置
# ============================================================

# 该文件已由最新的1_4异常订单清洗结果重新聚合生成。
INPUT_FILE = "/Users/liangxue/Documents/8、Python文件-网约车数据/Riding hailing_Project/1 数据清洗/2_3 OD-15min聚合（补全时间、不管没出现的OD）结果.parquet"

OUTPUT_DIR = "/Users/liangxue/Documents/8、Python文件-网约车数据/Riding hailing_Project/2 模型特征构建"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "4 tauijt特征构造（结果）.parquet"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. 参数设置
# ============================================================

TIME_N = 2880        # 2024年6月共2880个15min
DAY_N = 30           # 2024年6月共30天
SLOT_N = 96          # 每天96个15min
DIST_BIN_WIDTH = 0.5 # 相似距离分箱宽度，单位km


# ============================================================
# 3. 字段设置
# ============================================================

base_cols = [
    "i",
    "j",
    "t",
    "OD_id",
    "dist_ij",
    "same_region_ij",
    "day_index",
    "slot_in_day",
    "dow",
    "is_weekend_or_holiday",
    "is_peak_morning",
    "is_peak_evening",
    "is_night",
    "temperature_2m",
    "relative_humidity_2m",
    "rain",
    "weather_code",
    "wind_speed_10m",
    "D_ijt_observed",
    "tau_bar_ijt",
    "tau_q50_ijt",
]


# ============================================================
# 4. 工具函数
# ============================================================

def safe_divide(numerator, denominator):
    """分母为0时返回NaN。"""
    numerator = numerator.astype("float32")
    denominator = denominator.astype("float32")
    out = numerator / np.where(denominator == 0, np.nan, denominator)
    return out.astype("float32")


def lag_array(arr, lag):
    """
    对二维矩阵按时间维度做滞后。
    arr: shape = (n_od, 2880)
    """
    out = np.full_like(arr, np.nan, dtype="float32")
    out[:, lag:] = arr[:, :-lag]
    return out


def rolling_prev_nanmean(arr, window):
    """
    计算过去window个时段的均值，不包含当前时段，自动忽略NaN。
    arr: shape = (n_od, 2880)
    """
    valid = np.isfinite(arr)
    values = np.where(valid, arr, 0.0).astype("float32")
    counts = valid.astype("float32")

    csum = np.concatenate(
        [np.zeros((arr.shape[0], 1), dtype="float32"), np.cumsum(values, axis=1)],
        axis=1
    )

    ccount = np.concatenate(
        [np.zeros((arr.shape[0], 1), dtype="float32"), np.cumsum(counts, axis=1)],
        axis=1
    )

    idx = np.arange(arr.shape[1])
    start = np.maximum(0, idx - window)
    end = idx

    window_sum = csum[:, end] - csum[:, start]
    window_count = ccount[:, end] - ccount[:, start]

    return safe_divide(window_sum, window_count)


def previous_days_weighted_mean(sum_daily, count_daily, window):
    """
    计算过去window天的订单数加权平均，不包含当天。
    sum_daily: shape = (n_od, 30)
    count_daily: shape = (n_od, 30)
    """
    csum = np.concatenate(
        [np.zeros((sum_daily.shape[0], 1), dtype="float32"), np.cumsum(sum_daily, axis=1)],
        axis=1
    )

    ccount = np.concatenate(
        [np.zeros((count_daily.shape[0], 1), dtype="float32"), np.cumsum(count_daily, axis=1)],
        axis=1
    )

    idx = np.arange(sum_daily.shape[1])
    start = np.maximum(0, idx - window)
    end = idx

    prev_sum = csum[:, end] - csum[:, start]
    prev_count = ccount[:, end] - ccount[:, start]

    return safe_divide(prev_sum, prev_count), prev_count.astype("float32")


def broadcast_daily_to_15min(daily_arr):
    """
    将 shape=(n_od, 30) 的日度特征扩展为 shape=(n_od, 2880)。
    """
    return np.repeat(daily_arr, SLOT_N, axis=1).astype("float32")


def fill_by_fallback(target, fallback_list, global_mean):
    """
    对目标特征逐级填补：
    先用同一OD长期特征，再用相似距离兜底特征，最后用全局均值。
    """
    out = target.copy()

    for fb in fallback_list:
        out = np.where(np.isnan(out), fb, out)

    out = np.where(np.isnan(out), global_mean, out)

    return out.astype("float32")


# ============================================================
# 5. 第一遍：快速计算相似距离兜底统计
#    注意：这里用全月统计，速度快；严格时间切分实验时，应只用训练集统计。
# ============================================================

pf = pq.ParquetFile(INPUT_FILE)

dist_slot_summary = []
dist_summary = []

global_tau_sum = 0.0
global_order_count = 0.0

for rg in range(pf.num_row_groups):
    chunk = pf.read_row_group(
        rg,
        columns=["dist_ij", "slot_in_day", "D_ijt_observed", "tau_bar_ijt"]
    ).to_pandas()

    chunk["dist_ij"] = pd.to_numeric(chunk["dist_ij"], errors="coerce")
    chunk["slot_in_day"] = pd.to_numeric(chunk["slot_in_day"], errors="coerce").astype("int16")
    chunk["D_ijt_observed"] = pd.to_numeric(chunk["D_ijt_observed"], errors="coerce").fillna(0)
    chunk["tau_bar_ijt"] = pd.to_numeric(chunk["tau_bar_ijt"], errors="coerce")

    valid = (chunk["D_ijt_observed"] > 0) & chunk["tau_bar_ijt"].notna()

    obs = chunk.loc[valid].copy()

    if len(obs) > 0:
        obs["dist_bin_id"] = np.floor(obs["dist_ij"] / DIST_BIN_WIDTH).astype("int32")
        obs["tau_sum"] = obs["tau_bar_ijt"] * obs["D_ijt_observed"]

        dist_slot = (
            obs.groupby(["dist_bin_id", "slot_in_day"], as_index=False)
            .agg(
                tau_sum=("tau_sum", "sum"),
                order_count=("D_ijt_observed", "sum"),
            )
        )

        dist_only = (
            obs.groupby(["dist_bin_id"], as_index=False)
            .agg(
                tau_sum=("tau_sum", "sum"),
                order_count=("D_ijt_observed", "sum"),
            )
        )

        dist_slot_summary.append(dist_slot)
        dist_summary.append(dist_only)

        global_tau_sum += float(obs["tau_sum"].sum())
        global_order_count += float(obs["D_ijt_observed"].sum())

    print(f"第一遍兜底统计：row group {rg + 1}/{pf.num_row_groups}")

if global_order_count == 0:
    raise ValueError("没有找到任何 tau_bar_ijt 非空且 D_ijt_observed > 0 的样本。")

global_tau_mean = global_tau_sum / global_order_count

dist_slot_all = pd.concat(dist_slot_summary, ignore_index=True)
dist_slot_all = (
    dist_slot_all.groupby(["dist_bin_id", "slot_in_day"], as_index=False)
    .agg(
        tau_sum=("tau_sum", "sum"),
        order_count=("order_count", "sum"),
    )
)

dist_slot_all["global_tau_by_dist_slot"] = (
    dist_slot_all["tau_sum"] / dist_slot_all["order_count"]
)

dist_slot_all["dist_slot_key"] = (
    dist_slot_all["dist_bin_id"].astype("int64") * 100
    + dist_slot_all["slot_in_day"].astype("int64")
)

global_tau_by_dist_slot_map = pd.Series(
    dist_slot_all["global_tau_by_dist_slot"].values,
    index=dist_slot_all["dist_slot_key"].values
)

dist_all = pd.concat(dist_summary, ignore_index=True)
dist_all = (
    dist_all.groupby(["dist_bin_id"], as_index=False)
    .agg(
        tau_sum=("tau_sum", "sum"),
        order_count=("order_count", "sum"),
    )
)

dist_all["similar_OD_tau_mean"] = (
    dist_all["tau_sum"] / dist_all["order_count"]
)

similar_OD_tau_mean_map = pd.Series(
    dist_all["similar_OD_tau_mean"].values,
    index=dist_all["dist_bin_id"].values
)


# ============================================================
# 6. 第二遍：按 row group 构造 tauijt 特征并写出
# ============================================================

if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

writer = None
total_rows = 0
total_trainable_rows = 0

for rg in range(pf.num_row_groups):
    df = pf.read_row_group(rg, columns=base_cols).to_pandas()

    # --------------------------------------------------------
    # 6.1 基础字段清洗
    # --------------------------------------------------------

    df["i"] = df["i"].astype("string")
    df["j"] = df["j"].astype("string")
    df["OD_id"] = df["OD_id"].astype("string")
    df["t"] = pd.to_datetime(df["t"], errors="coerce")

    df["D_ijt_observed"] = (
        pd.to_numeric(df["D_ijt_observed"], errors="coerce")
        .fillna(0)
        .astype("int32")
    )

    df["tau_bar_ijt"] = pd.to_numeric(df["tau_bar_ijt"], errors="coerce").astype("float32")
    df["tau_q50_ijt"] = pd.to_numeric(df["tau_q50_ijt"], errors="coerce").astype("float32")
    df["dist_ij"] = pd.to_numeric(df["dist_ij"], errors="coerce").astype("float32")

    # --------------------------------------------------------
    # 6.2 检查当前 row group 是否由完整 OD 序列组成
    # --------------------------------------------------------

    if len(df) % TIME_N != 0:
        raise ValueError(
            f"row group {rg} 的行数不是2880的整数倍。"
            "请确认 2_3 文件写出时使用 writer.write_table(table, row_group_size=len(panel))。"
        )

    n_od = len(df) // TIME_N

    od_block = df["OD_id"].astype(str).to_numpy().reshape(n_od, TIME_N)

    if not np.all(od_block == od_block[:, [0]]):
        raise ValueError(
            f"row group {rg} 中存在 OD_id 没有连续存放的情况。"
            "请重新生成 2_3 文件，并确保每个 OD 的2880行连续写入。"
        )

    # --------------------------------------------------------
    # 6.3 构造 tau 标签矩阵
    #     D=0 时 tau 不可观测，标签为 NaN
    # --------------------------------------------------------

    D = df["D_ijt_observed"].to_numpy(dtype="float32").reshape(n_od, TIME_N)
    tau = df["tau_bar_ijt"].to_numpy(dtype="float32").reshape(n_od, TIME_N)
    tau_q50 = df["tau_q50_ijt"].to_numpy(dtype="float32").reshape(n_od, TIME_N)

    tau_label = np.where((D > 0) & np.isfinite(tau), tau, np.nan).astype("float32")
    tau_q50_obs = np.where((D > 0) & np.isfinite(tau_q50), tau_q50, np.nan).astype("float32")

    # --------------------------------------------------------
    # 6.4 同一 OD 短期历史服务时间特征
    # --------------------------------------------------------

    tau_ij_lag1 = lag_array(tau_label, 1)
    tau_ij_lag2 = lag_array(tau_label, 2)

    tau_ij_roll4_mean = rolling_prev_nanmean(tau_label, 4)
    tau_ij_roll8_mean = rolling_prev_nanmean(tau_label, 8)

    tau_ij_same_slot_last_day = lag_array(tau_label, 96)
    tau_ij_same_slot_last_week = lag_array(tau_label, 672)

    # --------------------------------------------------------
    # 6.5 同一 OD 长期服务时间特征
    #     按订单数加权，不能直接对 tau_bar_ijt 简单平均
    # --------------------------------------------------------

    valid_tau = np.isfinite(tau_label)
    D_valid = np.where(valid_tau, D, 0.0).astype("float32")

    tau_sum_15min = np.where(valid_tau, tau_label * D, 0.0).astype("float32")
    tau_q50_sum_15min = np.where(np.isfinite(tau_q50_obs), tau_q50_obs * D, 0.0).astype("float32")

    tau_sum_daily = tau_sum_15min.reshape(n_od, DAY_N, SLOT_N).sum(axis=2)
    tau_q50_sum_daily = tau_q50_sum_15min.reshape(n_od, DAY_N, SLOT_N).sum(axis=2)
    count_daily = D_valid.reshape(n_od, DAY_N, SLOT_N).sum(axis=2)

    tau_mean_daily = safe_divide(tau_sum_daily, count_daily)
    tau_q50_mean_daily = safe_divide(tau_q50_sum_daily, count_daily)

    tau_ij_mean_last_day_daily = np.full_like(tau_mean_daily, np.nan, dtype="float32")
    tau_ij_mean_last_day_daily[:, 1:] = tau_mean_daily[:, :-1]

    tau_ij_median_last_day_proxy_daily = np.full_like(tau_q50_mean_daily, np.nan, dtype="float32")
    tau_ij_median_last_day_proxy_daily[:, 1:] = tau_q50_mean_daily[:, :-1]

    tau_ij_mean_last3_daily, _ = previous_days_weighted_mean(tau_sum_daily, count_daily, 3)
    tau_ij_mean_last7_daily, tau_ij_count_last7_daily = previous_days_weighted_mean(tau_sum_daily, count_daily, 7)

    tau_ij_mean_last_day = broadcast_daily_to_15min(tau_ij_mean_last_day_daily)
    tau_ij_median_last_day_proxy = broadcast_daily_to_15min(tau_ij_median_last_day_proxy_daily)
    tau_ij_mean_last3 = broadcast_daily_to_15min(tau_ij_mean_last3_daily)
    tau_ij_mean_last7 = broadcast_daily_to_15min(tau_ij_mean_last7_daily)
    tau_ij_count_last7 = broadcast_daily_to_15min(tau_ij_count_last7_daily)

    # --------------------------------------------------------
    # 6.6 相似距离兜底特征
    # --------------------------------------------------------

    dist_bin_id = np.floor(df["dist_ij"].to_numpy(dtype="float32") / DIST_BIN_WIDTH).astype("int32")
    slot_in_day = df["slot_in_day"].to_numpy(dtype="int32")

    dist_slot_key = dist_bin_id.astype("int64") * 100 + slot_in_day.astype("int64")

    global_tau_by_dist_slot = (
        pd.Series(dist_slot_key)
        .map(global_tau_by_dist_slot_map)
        .to_numpy(dtype="float32")
    )

    similar_OD_tau_mean = (
        pd.Series(dist_bin_id)
        .map(similar_OD_tau_mean_map)
        .to_numpy(dtype="float32")
    )

    global_tau_by_dist_slot = np.where(
        np.isnan(global_tau_by_dist_slot),
        global_tau_mean,
        global_tau_by_dist_slot
    ).astype("float32")

    similar_OD_tau_mean = np.where(
        np.isnan(similar_OD_tau_mean),
        global_tau_mean,
        similar_OD_tau_mean
    ).astype("float32")

    global_tau_by_dist_slot_m = global_tau_by_dist_slot.reshape(n_od, TIME_N)
    similar_OD_tau_mean_m = similar_OD_tau_mean.reshape(n_od, TIME_N)

    # --------------------------------------------------------
    # 6.7 构造同一 OD 历史缺失标记
    # --------------------------------------------------------

    own_missing = (
        np.isnan(tau_ij_lag1)
        & np.isnan(tau_ij_lag2)
        & np.isnan(tau_ij_roll4_mean)
        & np.isnan(tau_ij_roll8_mean)
        & np.isnan(tau_ij_same_slot_last_day)
        & np.isnan(tau_ij_same_slot_last_week)
        & np.isnan(tau_ij_mean_last_day)
        & np.isnan(tau_ij_mean_last3)
        & np.isnan(tau_ij_mean_last7)
    ).astype("int8")

    # --------------------------------------------------------
    # 6.8 对短期历史特征逐级填补
    # --------------------------------------------------------

    fallback_list = [
        tau_ij_mean_last_day,
        tau_ij_mean_last3,
        tau_ij_mean_last7,
        global_tau_by_dist_slot_m,
        similar_OD_tau_mean_m,
    ]

    tau_ij_lag1 = fill_by_fallback(tau_ij_lag1, fallback_list, global_tau_mean)
    tau_ij_lag2 = fill_by_fallback(tau_ij_lag2, fallback_list, global_tau_mean)
    tau_ij_roll4_mean = fill_by_fallback(tau_ij_roll4_mean, fallback_list, global_tau_mean)
    tau_ij_roll8_mean = fill_by_fallback(tau_ij_roll8_mean, fallback_list, global_tau_mean)
    tau_ij_same_slot_last_day = fill_by_fallback(tau_ij_same_slot_last_day, fallback_list, global_tau_mean)
    tau_ij_same_slot_last_week = fill_by_fallback(tau_ij_same_slot_last_week, fallback_list, global_tau_mean)

    # 长期特征自身也补齐，避免输入出现缺失
    tau_ij_mean_last_day = fill_by_fallback(tau_ij_mean_last_day, [global_tau_by_dist_slot_m, similar_OD_tau_mean_m], global_tau_mean)
    tau_ij_median_last_day_proxy = fill_by_fallback(tau_ij_median_last_day_proxy, [global_tau_by_dist_slot_m, similar_OD_tau_mean_m], global_tau_mean)
    tau_ij_mean_last3 = fill_by_fallback(tau_ij_mean_last3, [global_tau_by_dist_slot_m, similar_OD_tau_mean_m], global_tau_mean)
    tau_ij_mean_last7 = fill_by_fallback(tau_ij_mean_last7, [global_tau_by_dist_slot_m, similar_OD_tau_mean_m], global_tau_mean)

    tau_ij_count_last7 = np.where(
        np.isnan(tau_ij_count_last7),
        0.0,
        tau_ij_count_last7
    ).astype("float32")

    # --------------------------------------------------------
    # 6.9 整理输出结果
    # --------------------------------------------------------

    result = df[
        [
            "i",
            "j",
            "t",
            "OD_id",
            "dist_ij",
            "same_region_ij",
            "day_index",
            "slot_in_day",
            "dow",
            "is_weekend_or_holiday",
            "is_peak_morning",
            "is_peak_evening",
            "is_night",
            "temperature_2m",
            "relative_humidity_2m",
            "rain",
            "weather_code",
            "wind_speed_10m",
        ]
    ].copy()

    result["tau_ij_lag1"] = tau_ij_lag1.reshape(-1)
    result["tau_ij_lag2"] = tau_ij_lag2.reshape(-1)
    result["tau_ij_roll4_mean"] = tau_ij_roll4_mean.reshape(-1)
    result["tau_ij_roll8_mean"] = tau_ij_roll8_mean.reshape(-1)
    result["tau_ij_same_slot_last_day"] = tau_ij_same_slot_last_day.reshape(-1)
    result["tau_ij_same_slot_last_week"] = tau_ij_same_slot_last_week.reshape(-1)

    result["tau_ij_mean_last_day"] = tau_ij_mean_last_day.reshape(-1)
    result["tau_ij_median_last_day_proxy"] = tau_ij_median_last_day_proxy.reshape(-1)
    result["tau_ij_mean_last3"] = tau_ij_mean_last3.reshape(-1)
    result["tau_ij_mean_last7"] = tau_ij_mean_last7.reshape(-1)
    result["tau_ij_count_last7"] = tau_ij_count_last7.reshape(-1)

    result["od_tau_missing_flag"] = own_missing.reshape(-1)
    result["global_tau_by_dist_slot"] = global_tau_by_dist_slot
    result["similar_OD_tau_mean"] = similar_OD_tau_mean

    result["tau_ijt_label"] = tau_label.reshape(-1)

    # --------------------------------------------------------
    # 6.10 类型压缩
    # --------------------------------------------------------

    result["i"] = result["i"].astype("string")
    result["j"] = result["j"].astype("string")
    result["OD_id"] = result["OD_id"].astype("string")

    int_cols = [
        "same_region_ij",
        "day_index",
        "slot_in_day",
        "dow",
        "is_weekend_or_holiday",
        "is_peak_morning",
        "is_peak_evening",
        "is_night",
        "od_tau_missing_flag",
    ]

    for col in int_cols:
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0).astype("int32")

    float_cols = [
        c for c in result.columns
        if c not in ["i", "j", "t", "OD_id"] + int_cols
    ]

    for col in float_cols:
        result[col] = pd.to_numeric(result[col], errors="coerce").astype("float32")

    # --------------------------------------------------------
    # 6.11 写出当前 row group
    # --------------------------------------------------------

    table = pa.Table.from_pandas(result, preserve_index=False)

    if writer is None:
        writer = pq.ParquetWriter(
            OUTPUT_FILE,
            table.schema,
            compression="snappy"
        )

    writer.write_table(table, row_group_size=len(result))

    total_rows += len(result)
    total_trainable_rows += int(result["tau_ijt_label"].notna().sum())

    print(
        f"第二遍特征构造：row group {rg + 1}/{pf.num_row_groups}；"
        f"累计样本数：{total_rows}"
    )

    del df, result, table


if writer is not None:
    writer.close()


# ============================================================
# 7. 输出检查信息
# ============================================================

print("完成")
print("输出文件：", OUTPUT_FILE)
print("总样本数：", total_rows)
print("tau_ijt_label 非空样本数：", total_trainable_rows)
print("全局平均服务时间：", float(global_tau_mean))
