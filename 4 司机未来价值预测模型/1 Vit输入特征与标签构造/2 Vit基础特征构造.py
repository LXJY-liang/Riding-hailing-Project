# -*- coding: utf-8 -*-
"""
2 Vit基础特征构造.py

功能：
1. 以区域-15min完整面板为基础；
2. 保留区域、时间和天气特征；
3. 将天气代码转换为对应的天气文本；
4. 合并10个历史司机收益机会特征；
5. 输出后续与随机变量预测结果拼接的 Vit 基础特征表。
"""

from pathlib import Path

import pandas as pd


# ============================================================
# 1. 路径
# ============================================================

project_root = Path(
    "/Users/liangxue/Documents/8、Python文件-网约车数据/Riding hailing_Project"
)

code_dir = (
    project_root
    / "4 司机未来价值预测模型"
    / "1 Vit输入特征与标签构造"
)

region_panel_file = (
    project_root
    / "1 数据清洗"
    / "3_3 区域-15min聚合（补全空值）结果.parquet"
)

profit_feature_file = (
    code_dir
    / "1 司机历史收益特征构造（结果）.parquet"
)

output_file = (
    code_dir
    / "2 Vit基础特征构造（结果）.parquet"
)


# ============================================================
# 2. 天气代码映射
# ============================================================

weather_map = {
    0: "晴朗",
    1: "大致晴朗",
    2: "局部多云",
    3: "阴天",
    45: "有雾",
    48: "雾凇",
    51: "小毛毛雨",
    53: "中等毛毛雨",
    55: "强毛毛雨",
    56: "轻微冻毛毛雨",
    57: "强冻毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    66: "轻微冻雨",
    67: "强冻雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    77: "米雪",
    80: "小阵雨",
    81: "中等阵雨",
    82: "强阵雨",
    85: "小阵雪",
    86: "强阵雪",
    95: "雷暴",
    96: "雷暴伴轻微冰雹",
    99: "雷暴伴强冰雹",
}


# ============================================================
# 3. 基础区域、时间和天气特征
# ============================================================

base_columns = [
    "region_id",
    "t",
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

base = pd.read_parquet(
    region_panel_file,
    columns=base_columns,
)

base["region_id"] = (
    base["region_id"]
    .astype("string")
)

base["t"] = pd.to_datetime(
    base["t"]
)

base["weather_code"] = (
    base["weather_code"]
    .map(weather_map)
    .astype("string")
)


# ============================================================
# 4. 合并历史司机收益机会
# ============================================================

profit = pd.read_parquet(
    profit_feature_file
)

profit["region_id"] = (
    profit["region_id"]
    .astype("string")
)

profit["t"] = pd.to_datetime(
    profit["t"]
)

result = base.merge(
    profit,
    on=[
        "region_id",
        "t",
    ],
    how="left",
)


# ============================================================
# 5. 保存
# ============================================================

code_dir.mkdir(
    parents=True,
    exist_ok=True,
)

result.to_parquet(
    output_file,
    index=False,
    compression="snappy",
)