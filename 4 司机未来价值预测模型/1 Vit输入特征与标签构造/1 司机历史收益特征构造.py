# -*- coding: utf-8 -*-
"""
1 司机历史收益特征构造.py

功能：
1. 根据订单级 Fare、Trip Miles、Trip Seconds 构造历史司机净收益代理；
2. 生成上一时段、过去1小时、过去2小时、前一天同一时段、前一周同一时段的
   单位服务时间净收益和平均每单净收益，共10个特征。

说明：
1. 只使用 Fare，不使用 Additional Charges 和 Tips；
2. 历史司机净收益代理：
   driver_net_profit = (1 - rho) * Fare - service_cost；
3. service_cost = 0.67 * Trip Miles + 0.27 * Trip Minutes；
4. 没有历史订单时，对应收益特征保留 NaN。
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. 路径与参数
# ============================================================

project_root = Path(
    "/Users/liangxue/Documents/8、Python文件-网约车数据/Riding hailing_Project"
)

code_dir = project_root / "4 司机未来价值预测模型/1 Vit输入特征与标签构造"

order_file = (
    project_root
    / "1 数据清洗"
    / "1 经纬度恢复及删除多余变量_保留订单.csv"
)

region_panel_file = (
    project_root
    / "1 数据清洗"
    / "3_3 区域-15min聚合（补全空值）结果.parquet"
)

output_file = code_dir / "1 司机历史收益特征构造（结果）.parquet"

rho = 0.25
cost_per_mile = 0.67
cost_per_minute = 0.27
chunk_size = 500_000


# ============================================================
# 2. 按块聚合订单收益
# ============================================================

usecols = [
    "Trip Start Timestamp",
    "Trip Seconds",
    "Trip Miles",
    "Pickup Census Tract",
    "Fare",
]

parts = []

for chunk in pd.read_csv(
    order_file,
    usecols=usecols,
    dtype={"Pickup Census Tract": "string"},
    chunksize=chunk_size,
):
    chunk["t"] = pd.to_datetime(
        chunk["Trip Start Timestamp"]
    ).dt.floor("15min")

    chunk["region_id"] = chunk[
        "Pickup Census Tract"
    ].astype("string")

    trip_seconds = pd.to_numeric(
        chunk["Trip Seconds"],
        errors="coerce",
    )

    trip_miles = pd.to_numeric(
        chunk["Trip Miles"],
        errors="coerce",
    )

    fare = pd.to_numeric(
        chunk["Fare"],
        errors="coerce",
    )

    trip_minutes = trip_seconds / 60.0

    # 删除异常订单
    mask = (
            (trip_seconds > 0)
            &
            (trip_miles > 0)
            &
            (fare > 0)
            &
            (trip_minutes >= 3)
    )

    chunk = chunk.loc[mask].copy()

    trip_minutes = trip_minutes.loc[mask]
    trip_miles = trip_miles.loc[mask]
    fare = fare.loc[mask]

    service_cost = (
        cost_per_mile * trip_miles
        + cost_per_minute * trip_minutes
    )

    chunk["driver_net_profit"] = (
        (1.0 - rho) * fare
        - service_cost
    )

    chunk["service_minutes"] = trip_minutes

    part = (
        chunk.groupby(
            ["region_id", "t"],
            as_index=False,
        )
        .agg(
            profit_sum_it=("driver_net_profit", "sum"),
            service_minutes_sum_it=("service_minutes", "sum"),
            order_count_it=("Fare", "count"),
        )
    )

    parts.append(part)

order_agg = (
    pd.concat(parts, ignore_index=True)
    .groupby(
        ["region_id", "t"],
        as_index=False,
    )
    .agg(
        profit_sum_it=("profit_sum_it", "sum"),
        service_minutes_sum_it=("service_minutes_sum_it", "sum"),
        order_count_it=("order_count_it", "sum"),
    )
)


# ============================================================
# 3. 合并完整区域-15min面板
# ============================================================

panel = pd.read_parquet(
    region_panel_file,
    columns=["region_id", "t"],
)

panel["region_id"] = panel["region_id"].astype("string")
panel["t"] = pd.to_datetime(panel["t"])

df = panel.merge(
    order_agg,
    on=["region_id", "t"],
    how="left",
)

df[
    [
        "profit_sum_it",
        "service_minutes_sum_it",
        "order_count_it",
    ]
] = df[
    [
        "profit_sum_it",
        "service_minutes_sum_it",
        "order_count_it",
    ]
].fillna(0.0)

df = df.sort_values(
    ["region_id", "t"]
).reset_index(drop=True)

g = df.groupby("region_id", sort=False)


# ============================================================
# 4. 收益率计算函数
# ============================================================

def safe_ratio(numerator, denominator):
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)

    return np.divide(
        numerator,
        denominator,
        out=np.full(
            numerator.shape,
            np.nan,
            dtype=np.float64,
        ),
        where=denominator > 0,
    )


# ============================================================
# 5. 上一时段
# ============================================================

profit_lag1 = g["profit_sum_it"].shift(1)
minutes_lag1 = g["service_minutes_sum_it"].shift(1)
orders_lag1 = g["order_count_it"].shift(1)

df["driver_profit_per_min_i_lag1"] = safe_ratio(
    profit_lag1,
    minutes_lag1,
)

df["driver_profit_per_order_i_lag1"] = safe_ratio(
    profit_lag1,
    orders_lag1,
)


# ============================================================
# 6. 过去1小时和过去2小时
# ============================================================

for window, name in [
    (4, "roll4"),
    (8, "roll8"),
]:
    profit_roll = g["profit_sum_it"].transform(
        lambda s: s.shift(1).rolling(
            window=window,
            min_periods=1,
        ).sum()
    )

    minutes_roll = g[
        "service_minutes_sum_it"
    ].transform(
        lambda s: s.shift(1).rolling(
            window=window,
            min_periods=1,
        ).sum()
    )

    orders_roll = g["order_count_it"].transform(
        lambda s: s.shift(1).rolling(
            window=window,
            min_periods=1,
        ).sum()
    )

    df[
        f"driver_profit_per_min_i_{name}"
    ] = safe_ratio(
        profit_roll,
        minutes_roll,
    )

    df[
        f"driver_profit_per_order_i_{name}"
    ] = safe_ratio(
        profit_roll,
        orders_roll,
    )


# ============================================================
# 7. 前一天和前一周相同时段
# ============================================================

for periods, name in [
    (96, "same_slot_last_day"),
    (672, "same_slot_last_week"),
]:
    profit_shift = g["profit_sum_it"].shift(periods)
    minutes_shift = g[
        "service_minutes_sum_it"
    ].shift(periods)
    orders_shift = g[
        "order_count_it"
    ].shift(periods)

    df[
        f"driver_profit_per_min_i_{name}"
    ] = safe_ratio(
        profit_shift,
        minutes_shift,
    )

    df[
        f"driver_profit_per_order_i_{name}"
    ] = safe_ratio(
        profit_shift,
        orders_shift,
    )

# ============================================================
# 8. 收益特征截尾处理
# ============================================================

def winsorize_series(s, lower=0.01, upper=0.99):
    q_low = s.quantile(lower)
    q_high = s.quantile(upper)

    return s.clip(
        lower=q_low,
        upper=q_high,
    )


profit_feature_columns = [
    "driver_profit_per_min_i_lag1",
    "driver_profit_per_order_i_lag1",
    "driver_profit_per_min_i_roll4",
    "driver_profit_per_order_i_roll4",
    "driver_profit_per_min_i_roll8",
    "driver_profit_per_order_i_roll8",
    "driver_profit_per_min_i_same_slot_last_day",
    "driver_profit_per_order_i_same_slot_last_day",
    "driver_profit_per_min_i_same_slot_last_week",
    "driver_profit_per_order_i_same_slot_last_week",
]


for col in profit_feature_columns:
    df[col] = winsorize_series(
        df[col]
    )

# ============================================================
# 9. 保存结果
# ============================================================

feature_columns = [
    "region_id",
    "t",
    "driver_profit_per_min_i_lag1",
    "driver_profit_per_order_i_lag1",
    "driver_profit_per_min_i_roll4",
    "driver_profit_per_order_i_roll4",
    "driver_profit_per_min_i_roll8",
    "driver_profit_per_order_i_roll8",
    "driver_profit_per_min_i_same_slot_last_day",
    "driver_profit_per_order_i_same_slot_last_day",
    "driver_profit_per_min_i_same_slot_last_week",
    "driver_profit_per_order_i_same_slot_last_week",
]

result = df[feature_columns].copy()

for col in feature_columns[2:]:
    result[col] = result[col].astype("float32")

code_dir.mkdir(parents=True, exist_ok=True)

result.to_parquet(
    output_file,
    index=False,
    compression="snappy",
)
