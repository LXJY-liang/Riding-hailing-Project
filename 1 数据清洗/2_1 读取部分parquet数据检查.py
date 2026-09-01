# -*- coding: utf-8 -*-

# 导入 pandas，用于读取 parquet 文件、处理时间字段、筛选数据和终端打印
import pandas as pd

# ============================================================
# 1. 设置 OD-15min 聚合结果文件路径
PARQUET_FILE = "/Users/liangxue/Documents/8、Python文件-网约车数据/Riding hailing_Project/1 数据清洗/2 OD-15min 聚合结果.parquet"

# ============================================================
# 2. 读取 parquet 文件中的部分字段
# 这里不是读取全部字段，而是只读取当前检查需要的字段。

df = pd.read_parquet(
    PARQUET_FILE,
    columns=[
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
)


# ============================================================
# 3. 确保 t 字段是 datetime 类型
# 有些 parquet 文件中的 t 可能已经是 datetime 类型；
# 但为了保证后面可以准确按时间筛选，这里统一再转一次。
# 如果不转成 datetime，后面用 "2024-06-01 08:00:00" 筛选时可能匹配失败。

df["t"] = pd.to_datetime(df["t"])

# ============================================================
# 4. 筛选指定 15min 时间段的数据

# 这里筛选的是所有在 2024-06-01 08:00:00 这个时间段发生的 OD 样本。
# 这不是筛选某一笔订单，而是筛选该 15min 时段内所有有订单的 OD 组合。
# 例如：
# i=17031010100, j=17031020100, t=2024-06-01 08:00:00 是一行；
# i=17031010200, j=17031030100, t=2024-06-01 08:00:00 也是一行。
# head(100) 表示只取前 100 行，避免终端输出太多。

sample = df[df["t"] == "2024-06-06 18:15:00"].head(100)
# ============================================================
# 4.1 统计指定 15min 时间段的总订单数
# ============================================================

CHECK_TIME = "2024-06-06 18:15:00"

time_df = df[df["t"] == CHECK_TIME].copy()

# D_ijt_observed 是每个 OD 在该 15min 内的订单数；
# 对所有 OD 的 D_ijt_observed 求和，就是该 15min 全部订单数。
total_orders = time_df["D_ijt_observed"].sum()

# 该时间段有多少个有订单的 OD 组合。
total_od_pairs = len(time_df)

print(f"{CHECK_TIME} 这个15min时间段的总订单数：{total_orders}")
print(f"{CHECK_TIME} 这个15min时间段有订单的OD组合数：{total_od_pairs}")
# ============================================================
# 5. 在终端输出筛选结果

# to_string(index=False) 的作用：
# 1. 输出完整表格；
# 2. 不显示 pandas 默认的行索引；
# 3. 方便直接在终端查看 OD-15min 聚合结果。

print(sample.to_string(index=False))