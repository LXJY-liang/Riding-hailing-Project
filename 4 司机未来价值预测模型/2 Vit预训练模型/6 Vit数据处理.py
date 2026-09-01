# -*- coding: utf-8 -*-
"""
6 Vit数据处理.py

功能：
1. 读取Vit的train/valid/test输入特征并合并Vit_label；
2. 使用训练集标签均值和标准差进行标准化；
3. 将全部Vit输入特征转换为固定结构的中文自然语言；
4. 保存后续Tokenizer使用的文本与标签。
"""

import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. 加载配置
# ============================================================

module_dir = Path(__file__).resolve().parent

spec = importlib.util.spec_from_file_location(
    "vit_config",
    module_dir / "5 Vit配置.py",
)

cfg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)


# ============================================================
# 2. 文本辅助函数
# ============================================================

weekday_map = {
    0: "星期一",
    1: "星期二",
    2: "星期三",
    3: "星期四",
    4: "星期五",
    5: "星期六",
    6: "星期日",
}


def number_text(value, digits=2):
    """将数值转换为文本，缺失值写为不可用。"""

    if pd.isna(value):
        return "不可用"

    return f"{float(value):.{digits}f}"


def percent_text(value, digits=2):
    """将0-1概率转换为百分比文本。"""

    if pd.isna(value):
        return "不可用"

    return f"{float(value) * 100:.{digits}f}%"


def time_period_text(row):
    """将时段标记转换为自然语言。"""

    periods = []

    if int(row["is_peak_morning"]) == 1:
        periods.append("早高峰")

    if int(row["is_peak_evening"]) == 1:
        periods.append("晚高峰")

    if int(row["is_night"]) == 1:
        periods.append("夜间")

    if not periods:
        periods.append("普通时段")

    if int(row["is_weekend_or_holiday"]) == 1:
        periods.append("周末或节假日")
    else:
        periods.append("工作日")

    return "、".join(periods)


def distribution_text(row, prefix, unit):
    """将20个5%分位区间均值和7个关键分位数转换为文本。"""

    bin_values = []

    for bin_index in range(20):
        lower = bin_index * 5
        upper = (bin_index + 1) * 5

        bin_values.append(
            number_text(
                row[
                    f"{prefix}_qbin_"
                    f"{lower:02d}_"
                    f"{upper:02d}_mean"
                ]
            )
        )

    key_text = "，".join(
        [
            f"q0.005={number_text(row[f'{prefix}_q_0.005'])}{unit}",
            f"q0.05={number_text(row[f'{prefix}_q_0.050'])}{unit}",
            f"q0.10={number_text(row[f'{prefix}_q_0.100'])}{unit}",
            f"q0.50={number_text(row[f'{prefix}_q_0.500'])}{unit}",
            f"q0.90={number_text(row[f'{prefix}_q_0.900'])}{unit}",
            f"q0.95={number_text(row[f'{prefix}_q_0.950'])}{unit}",
            f"q0.995={number_text(row[f'{prefix}_q_0.995'])}{unit}",
        ]
    )

    return (
        "20个5%分位区间均值从低到高依次为["
        + "，".join(bin_values)
        + f"]{unit}；关键分位数为{key_text}"
    )


def destination_text(row):
    """将Top-10目的地概率和对应服务时间转换为文本。"""

    destinations = []

    for rank in range(1, 11):
        destinations.append(
            (
                f"第{rank}位目的地区域为{row[f'dest_{rank}_region_id']}，"
                f"条件概率为{percent_text(row[f'dest_{rank}_pi'])}，"
                f"服务时间q0.10、q0.50、q0.90分别为"
                f"{number_text(row[f'dest_{rank}_tau_q0.10'])}、"
                f"{number_text(row[f'dest_{rank}_tau_q0.50'])}、"
                f"{number_text(row[f'dest_{rank}_tau_q0.90'])}分钟"
            )
        )

    return (
        f"Top-10目的地累计概率为{percent_text(row['pi_top10_mass'])}，"
        f"其余目的地累计概率为{percent_text(row['pi_other_mass'])}，"
        f"订单目的地仍为当前区域的概率为{percent_text(row['pi_same_region'])}；"
        + "；".join(destinations)
    )


def neighbor_text(row, neighbor_ranks):
    """将所有直接邻近区域的需求和供给状态转换为文本。"""

    neighbors = []

    for rank in neighbor_ranks:
        region_id = row[f"neighbor_{rank}_region_id"]

        if pd.isna(region_id):
            continue

        neighbors.append(
            (
                f"邻近区域{region_id}："
                f"预测需求q0.10、q0.50、q0.90分别为"
                f"{number_text(row[f'neighbor_{rank}_D_q0.10'])}、"
                f"{number_text(row[f'neighbor_{rank}_D_q0.50'])}、"
                f"{number_text(row[f'neighbor_{rank}_D_q0.90'])}单，"
                f"预测司机供给q0.10、q0.50、q0.90分别为"
                f"{number_text(row[f'neighbor_{rank}_S_q0.10'])}、"
                f"{number_text(row[f'neighbor_{rank}_S_q0.50'])}、"
                f"{number_text(row[f'neighbor_{rank}_S_q0.90'])}名司机"
            )
        )

    return "；".join(neighbors)


# ============================================================
# 3. 将一条Vit样本转换为自然语言
# ============================================================

def build_text(row, neighbor_ranks):
    """将全部Vit输入特征转换为固定结构的中文自然语言。"""

    timestamp = pd.Timestamp(row[cfg.time_column])
    dow = int(row["dow"])

    return "\n".join(
        [
            (
                "任务：根据当前区域与时间状态、天气、历史司机收益机会、"
                "预测需求与供给分布、司机获得订单概率、目的地结构、"
                "目的地服务时间和邻近区域状态，预测司机未来收益机会价值。"
                "预训练阶段以未来4小时折现收益作为该价值的监督代理。"
            ),
            "当前区域与时间状态：",
            (
                f"当前区域编号为{row[cfg.region_column]}，时间为"
                f"{timestamp.strftime('%Y年%m月%d日 %H:%M')}，"
                f"{weekday_map.get(dow, f'星期编码{dow}')}，"
                f"这是数据期第{int(row['day_index'])}天、"
                f"当日第{int(row['slot_in_day'])}个15分钟时段，"
                f"当前属于{time_period_text(row)}。"
            ),
            "当前天气状态：",
            (
                f"气温为{number_text(row['temperature_2m'])}摄氏度，"
                f"相对湿度为{number_text(row['relative_humidity_2m'])}%，"
                f"降雨量为{number_text(row['rain'])}毫米，"
                f"天气状况为{row['weather_code']}，"
                f"10米高度风速为{number_text(row['wind_speed_10m'])}千米/小时。"
            ),
            "历史司机收益机会：",
            (
                f"上一15分钟司机单位服务时间净收益为"
                f"{number_text(row['driver_profit_per_min_i_lag1'])}美元/分钟，"
                f"平均每单净收益为"
                f"{number_text(row['driver_profit_per_order_i_lag1'])}美元/单；"
                f"过去1小时司机单位服务时间净收益为"
                f"{number_text(row['driver_profit_per_min_i_roll4'])}美元/分钟，"
                f"平均每单净收益为"
                f"{number_text(row['driver_profit_per_order_i_roll4'])}美元/单；"
                f"过去2小时司机单位服务时间净收益为"
                f"{number_text(row['driver_profit_per_min_i_roll8'])}美元/分钟，"
                f"平均每单净收益为"
                f"{number_text(row['driver_profit_per_order_i_roll8'])}美元/单；"
                f"前一天同一时段司机单位服务时间净收益为"
                f"{number_text(row['driver_profit_per_min_i_same_slot_last_day'])}美元/分钟，"
                f"平均每单净收益为"
                f"{number_text(row['driver_profit_per_order_i_same_slot_last_day'])}美元/单；"
                f"前一周同一时段司机单位服务时间净收益为"
                f"{number_text(row['driver_profit_per_min_i_same_slot_last_week'])}美元/分钟，"
                f"平均每单净收益为"
                f"{number_text(row['driver_profit_per_order_i_same_slot_last_week'])}美元/单。"
            ),
            "当前预测订单需求分布：",
            distribution_text(row, "D", "单") + "。",
            "当前预测司机供给分布：",
            distribution_text(row, "S", "名司机") + "。",
            "司机获得订单机会：",
            (
                f"根据预测需求和预测司机供给的中位数计算，"
                f"代表性司机在当前15分钟获得至少一笔订单的概率为"
                f"{percent_text(row['a_it'])}，"
                f"未获得订单的概率为"
                f"{percent_text(row['one_minus_a_it'])}。"
            ),
            "获得订单后的条件目的地结构与服务时间：",
            destination_text(row) + "。",
            "直接邻近区域状态：",
            neighbor_text(row, neighbor_ranks) + "。",
        ]
    )


# ============================================================
# 4. 读取数据并合并标签
# ============================================================

label = pd.read_parquet(
    cfg.label_file,
    columns=[
        cfg.region_column,
        cfg.time_column,
        cfg.label_column,
    ],
)

label[cfg.region_column] = label[
    cfg.region_column
].astype("string")

label[cfg.time_column] = pd.to_datetime(
    label[cfg.time_column]
)


def read_split(feature_file):
    """读取一个数据划分并合并Vit标签。"""

    data = pd.read_parquet(feature_file)

    data[cfg.region_column] = data[
        cfg.region_column
    ].astype("string")

    data[cfg.time_column] = pd.to_datetime(
        data[cfg.time_column]
    )

    return data.merge(
        label,
        on=[
            cfg.region_column,
            cfg.time_column,
        ],
        how="inner",
    )


train = read_split(cfg.train_feature_file)
valid = read_split(cfg.valid_feature_file)
test = read_split(cfg.test_feature_file)

label_mean = float(
    train[cfg.label_column].mean()
)

label_std = float(
    train[cfg.label_column].std(ddof=0)
)

neighbor_ranks = sorted(
    {
        int(
            re.match(
                r"neighbor_(\d+)_region_id",
                column,
            ).group(1)
        )
        for column in train.columns
        if re.match(
            r"neighbor_(\d+)_region_id",
            column,
        )
    }
)


# ============================================================
# 5. 构造文本并保存
# ============================================================

def prepare_split(data, output_file):
    """生成预训练使用的自然语言文本和标准化标签。"""

    data = data.sort_values(
        [
            cfg.time_column,
            cfg.region_column,
        ]
    ).reset_index(drop=True)

    data["label_scaled"] = (
        (
            data[cfg.label_column].to_numpy(dtype=np.float32)
            - label_mean
        )
        / label_std
    ).astype(np.float32)

    data["text"] = [
        build_text(row, neighbor_ranks)
        for _, row in data.iterrows()
    ]

    data[
        [
            cfg.region_column,
            cfg.time_column,
            cfg.label_column,
            "label_scaled",
            "text",
        ]
    ].to_parquet(
        output_file,
        index=False,
        compression="snappy",
    )


cfg.data_dir.mkdir(parents=True, exist_ok=True)
cfg.model_dir.mkdir(parents=True, exist_ok=True)

prepare_split(train, cfg.train_data_file)
prepare_split(valid, cfg.valid_data_file)
prepare_split(test, cfg.test_data_file)

with open(
    cfg.label_scaler_file,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        {
            "label_mean": label_mean,
            "label_std": label_std,
            "use_standardization": cfg.use_standardization,
        },
        file,
        ensure_ascii=False,
        indent=2,
    )
