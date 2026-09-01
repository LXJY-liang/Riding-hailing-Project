# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd


# ============================================================
# 1. 路径设置
# ============================================================

ORDER_FILE = "/Users/liangxue/Documents/8、Python文件-网约车数据/Riding hailing_Project/1 数据清洗/1_4 异常订单清洗_保留订单.parquet"

WEATHER_FILE = "/Users/liangxue/Documents/8、Python文件-网约车数据/Riding hailing_Project/0 天气数据获取/chicago_weather_2024_06_hourly.csv"

OUTPUT_PARQUET = "/Users/liangxue/Documents/8、Python文件-网约车数据/Riding hailing_Project/1 数据清洗/2 OD-15min 聚合结果.parquet"

os.makedirs(os.path.dirname(OUTPUT_PARQUET), exist_ok=True)


# ============================================================
# 2. 只读取聚合需要的订单字段
# ============================================================

use_cols = [
    "Trip Start Timestamp",
    "Trip Seconds",
    "Trip Miles",
    "Pickup Census Tract",
    "Dropoff Census Tract",
    "Trip Total fare",
]

df = pd.read_parquet(ORDER_FILE, columns=use_cols)


# ============================================================
# 3. 基础字段清洗
# ============================================================

df["t"] = pd.to_datetime(df["Trip Start Timestamp"], errors="coerce")

df["i"] = df["Pickup Census Tract"].astype("string")
df["j"] = df["Dropoff Census Tract"].astype("string")

df["Trip Seconds"] = pd.to_numeric(df["Trip Seconds"], errors="coerce")
df["Trip Miles"] = pd.to_numeric(df["Trip Miles"], errors="coerce")
df["Trip Total fare"] = pd.to_numeric(df["Trip Total fare"], errors="coerce")

df["tau_minutes"] = df["Trip Seconds"] / 60.0

df = df[
    df["t"].notna()
    & df["i"].notna()
    & df["j"].notna()
    & df["Trip Total fare"].notna()
    & df["tau_minutes"].notna()
    & df["Trip Miles"].notna()
].copy()


# ============================================================
# 4. OD-15min 分组键
# ============================================================

group_cols = ["i", "j", "t"]
g = df.groupby(group_cols, sort=False)


# ============================================================
# 5. 计算订单数、均值
# ============================================================

base = g.agg(
    D_ijt_observed=("Trip Total fare", "size"),
    p_bar_ijt=("Trip Total fare", "mean"),
    tau_bar_ijt=("tau_minutes", "mean"),
    trip_miles_bar_ijt=("Trip Miles", "mean"),
).reset_index()


# ============================================================
# 6. 一次性计算价格分位数
# ============================================================

quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]

p_q = (
    df.groupby(group_cols, sort=False)["Trip Total fare"]
      .quantile(quantiles)
      .unstack()
      .reset_index()
)

p_q = p_q.rename(columns={
    0.10: "p_q10_ijt",
    0.25: "p_q25_ijt",
    0.50: "p_q50_ijt",
    0.75: "p_q75_ijt",
    0.90: "p_q90_ijt",
})


# ============================================================
# 7. 一次性计算行程时间分位数
# ============================================================

tau_q = (
    df.groupby(group_cols, sort=False)["tau_minutes"]
      .quantile(quantiles)
      .unstack()
      .reset_index()
)

tau_q = tau_q.rename(columns={
    0.10: "tau_q10_ijt",
    0.25: "tau_q25_ijt",
    0.50: "tau_q50_ijt",
    0.75: "tau_q75_ijt",
    0.90: "tau_q90_ijt",
})

# ============================================================
# 7.1 一次性计算行程里程分位数
# ============================================================

# 说明：
# Trip Miles 是订单行程距离，单位为英里。
# 这里按同一个 OD-15min 分组，即 (i, j, t)，计算该组订单行程里程的分位数。
# 如果某个 OD-15min 只有 1 条订单，那么 q10、q25、q50、q75、q90 会相同，这是正常现象。

miles_q = (
    df.groupby(group_cols, sort=False)["Trip Miles"]
      .quantile(quantiles)
      .unstack()
      .reset_index()
)

# 将分位数列重命名为模型中更清楚的变量名
miles_q = miles_q.rename(columns={
    0.10: "trip_miles_q10_ijt",
    0.25: "trip_miles_q25_ijt",
    0.50: "trip_miles_q50_ijt",
    0.75: "trip_miles_q75_ijt",
    0.90: "trip_miles_q90_ijt",
})

# ============================================================
# 8. 合并聚合结果
# ============================================================

# base：订单数、平均价格、平均行程时间、平均行程里程
# p_q：价格分位数
# tau_q：行程时间分位数
# miles_q：行程里程分位数
od = base.merge(p_q, on=group_cols, how="left")
od = od.merge(tau_q, on=group_cols, how="left")
od = od.merge(miles_q, on=group_cols, how="left")

# 删除中间变量，释放内存
del df, base, p_q, tau_q, miles_q


# ============================================================
# 9. 时间特征
# ============================================================

od["hour"] = od["t"].dt.hour
od["minute"] = od["t"].dt.minute
od["date"] = od["t"].dt.date

od["day_index"] = (od["t"].dt.normalize() - pd.Timestamp("2024-06-01")).dt.days + 1

# 时间段0～95，共一天共96个
od["slot_in_day"] = od["hour"] * 4 + (od["minute"] // 15)

# 星期几0～6
od["dow"] = od["t"].dt.dayofweek

holiday_dates = {pd.Timestamp("2024-06-19").date()}

od["is_weekend_or_holiday"] = (
    od["dow"].isin([5, 6]) | od["date"].isin(holiday_dates)
).astype("int8")

od["is_peak_morning"] = (
    (od["hour"] >= 7) & (od["hour"] < 9)
).astype("int8")

od["is_peak_evening"] = (
    (od["hour"] >= 16) & (od["hour"] < 19)
).astype("int8")

od["is_night"] = (
    (od["hour"] >= 22) | (od["hour"] < 6)
).astype("int8")


# ============================================================
# 10. 合并小时级天气
# ============================================================

weather = pd.read_csv(WEATHER_FILE)

weather["time"] = pd.to_datetime(weather["time"], errors="coerce")

weather = weather[
    [
        "time",
        "temperature_2m",
        "relative_humidity_2m",
        "rain",
        "weather_code",
        "wind_speed_10m",
    ]
].copy()

od["weather_hour"] = od["t"].dt.floor("h")

od = od.merge(
    weather,
    left_on="weather_hour",
    right_on="time",
    how="left"
)

od = od.drop(columns=["time", "weather_hour", "date", "hour", "minute"])


# ============================================================
# 11. 调整字段顺序
# ============================================================

final_cols = [
    "i",
    "j",
    "t",
    "day_index",
    "slot_in_day",
    "dow",
    "is_weekend_or_holiday",
    "is_peak_morning",
    "is_peak_evening",
    "is_night",

    "D_ijt_observed",

    "p_bar_ijt",
    "p_q10_ijt",
    "p_q25_ijt",
    "p_q50_ijt",
    "p_q75_ijt",
    "p_q90_ijt",

    "tau_bar_ijt",
    "tau_q10_ijt",
    "tau_q25_ijt",
    "tau_q50_ijt",
    "tau_q75_ijt",
    "tau_q90_ijt",

    "trip_miles_bar_ijt",
    "trip_miles_q10_ijt",
    "trip_miles_q25_ijt",
    "trip_miles_q50_ijt",
    "trip_miles_q75_ijt",
    "trip_miles_q90_ijt",

    "temperature_2m",
    "relative_humidity_2m",
    "rain",
    "weather_code",
    "wind_speed_10m",
]

od = od[final_cols].copy()


# ============================================================
# 12. 降低存储体积
# ============================================================

od["day_index"] = od["day_index"].astype("int16")
od["slot_in_day"] = od["slot_in_day"].astype("int16")
od["dow"] = od["dow"].astype("int8")
od["D_ijt_observed"] = od["D_ijt_observed"].astype("int32")

float_cols = [
    "p_bar_ijt",
    "p_q10_ijt",
    "p_q25_ijt",
    "p_q50_ijt",
    "p_q75_ijt",
    "p_q90_ijt",
    "tau_bar_ijt",
    "tau_q10_ijt",
    "tau_q25_ijt",
    "tau_q50_ijt",
    "tau_q75_ijt",
    "tau_q90_ijt",
    "trip_miles_bar_ijt",
    "trip_miles_q10_ijt",
    "trip_miles_q25_ijt",
    "trip_miles_q50_ijt",
    "trip_miles_q75_ijt",
    "trip_miles_q90_ijt",
    "temperature_2m",
    "relative_humidity_2m",
    "rain",
    "weather_code",
    "wind_speed_10m",
]

for col in float_cols:
    od[col] = pd.to_numeric(od[col], errors="coerce").astype("float32")


# ============================================================
# 13. 保存 parquet
# ============================================================

if os.path.exists(OUTPUT_PARQUET):
    os.remove(OUTPUT_PARQUET)

od.to_parquet(
    OUTPUT_PARQUET,
    index=False,
    engine="pyarrow",
    compression="snappy"
)

print("完成")
print(f"OD-15min 样本数：{len(od)}")
print(f"输出文件：{OUTPUT_PARQUET}")
