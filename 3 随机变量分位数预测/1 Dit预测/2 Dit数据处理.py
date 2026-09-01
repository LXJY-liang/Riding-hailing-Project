"""
运行结果

训练集样本数：1845504
验证集样本数：230688
测试集样本数：230688
标签均值：0.560831
标签标准差：0.941579
"""

from pathlib import Path
import importlib.util
import json

import numpy as np
import pandas as pd


# =========================
# 1. 读取配置
# =========================

def load_config():
    config_path = Path(__file__).resolve().parent / "1 Dit配置.py"
    spec = importlib.util.spec_from_file_location("dit_config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


config = load_config()


# =========================
# 2. 文本辅助函数
# =========================

weekday_map = {
    0: "星期一",
    1: "星期二",
    2: "星期三",
    3: "星期四",
    4: "星期五",
    5: "星期六",
    6: "星期日",
}

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


def number_text(value, digits=2):
    """将数值写入自然语言；缺失值写为历史不可用。"""

    if pd.isna(value):
        return "历史不可用"

    return f"{float(value):.{digits}f}"


def history_text(value, available=True, digits=2, unit=""):
    """历史可用时输出数值和单位；不可用时只输出“历史不可用”。"""

    if not available or pd.isna(value):
        return "历史不可用"

    return f"{float(value):.{digits}f}{unit}"


def time_period_text(row):
    """描述当前时段类型。"""

    descriptions = []

    if int(row["is_peak_morning"]) == 1:
        descriptions.append("早高峰")

    if int(row["is_peak_evening"]) == 1:
        descriptions.append("晚高峰")

    if int(row["is_night"]) == 1:
        descriptions.append("夜间")

    if not descriptions:
        descriptions.append("普通时段")

    if int(row["is_weekend_or_holiday"]) == 1:
        descriptions.append("周末或节假日")
    else:
        descriptions.append("工作日")

    return "、".join(descriptions)


# =========================
# 3. 将一条样本转换为自然语言
# =========================

def build_text(row):
    """将Dit数值特征按参考优先级转换为固定结构的中文自然语言。"""

    timestamp = pd.Timestamp(row[config.time_column])
    day_index = int(row["day_index"])
    dow = int(row["dow"])
    weather_code = int(row["weather_code"])

    return "\n".join(
        [
            "任务：预测当前区域当前时段的出发订单需求分布。",
            (
                "参考优先级：一级为本区域短期和同一时段历史需求；"
                "二级为本区域长期需求活跃度；三级为邻近区域历史需求。"
                "各级特征均保留，前级信息更重要，后级用于补充稀疏信息。"
            ),
            "区域与时间状态：",
            (
                f"当前区域编号为{row[config.region_column]}，时间为"
                f"{timestamp.strftime('%Y年%m月%d日 %H:%M')}，"
                f"{weekday_map.get(dow, f'星期编码{dow}')}，"
                f"这是当日第{int(row['slot_in_day'])}个15分钟时段。"
            ),
            f"当前属于{time_period_text(row)}。",
            "天气状态：",
            (
                f"气温为{number_text(row['temperature_2m'])}摄氏度，"
                f"相对湿度为{number_text(row['relative_humidity_2m'])}%，"
                f"降雨量为{number_text(row['rain'])}毫米，"
                f"天气状况为{weather_map.get(weather_code, f'天气代码{weather_code}')}，"
                f"10米高度风速为{number_text(row['wind_speed_10m'])}千米/小时。"
            ),
            "一级参考——本区域短期和同一时段历史需求：",
            (
                f"上一15分钟出发订单量："
                f"{history_text(row['D_origin_i_lag1'], unit='单')}；"
                f"上两个15分钟时段前出发订单量："
                f"{history_text(row['D_origin_i_lag2'], unit='单')}。"
            ),
            (
                f"过去1小时出发订单量均值："
                f"{history_text(row['D_origin_i_roll4_mean'], unit='单')}；"
                f"过去2小时出发订单量均值："
                f"{history_text(row['D_origin_i_roll8_mean'], unit='单')}。"
            ),
            (
                f"前一天同一时段出发订单量："
                f"{history_text(row['D_origin_i_same_slot_last_day'], day_index >= 2, unit='单')}；"
                f"前一周同一时段出发订单量："
                f"{history_text(row['D_origin_i_same_slot_last_week'], day_index >= 8, unit='单')}。"
            ),
            "二级参考——本区域长期需求活跃度：",
            (
                f"前一天全天出发订单总量："
                f"{history_text(row['D_origin_i_total_last_day'], day_index >= 2, unit='单')}；"
                f"过去3天日均出发订单量："
                f"{history_text(row['D_origin_i_daily_avg_last3'], day_index >= 4, unit='单')}；"
                f"过去7天日均出发订单量："
                f"{history_text(row['D_origin_i_daily_avg_last7'], day_index >= 8, unit='单')}。"
            ),
            "三级参考——邻近区域历史需求：",
            (
                f"邻近区域上一15分钟平均出发订单量："
                f"{history_text(row['neighbor_D_origin_lag1'], unit='单')}；"
                f"邻近区域过去1小时平均出发订单量："
                f"{history_text(row['neighbor_D_origin_roll4_mean'], unit='单')}；"
                f"邻近区域过去2小时平均出发订单量："
                f"{history_text(row['neighbor_D_origin_roll8_mean'], unit='单')}。"
            ),
        ]
    )


def build_text_batch(batch):
    """将datasets的一个批次转换为自然语言文本。"""

    batch_size = len(batch[config.region_column])
    texts = []

    for index in range(batch_size):
        row = {
            column: batch[column][index]
            for column in config.feature_columns
        }
        texts.append(build_text(row))

    return texts


# =========================
# 4. 标签变换
# =========================

def transform_label(values, label_mean, label_std):
    """对需求标签依次进行log1p和标准化。"""

    values = np.asarray(values, dtype=np.float32)

    if config.use_log1p:
        values = np.log1p(values)

    if config.use_standardization:
        values = (values - label_mean) / label_std

    return values.astype(np.float32)


# =========================
# 5. 数据划分与保存
# =========================

def prepare_data():
    """读取完整Dit特征，并按照日期划分训练、验证和测试集。"""

    data = pd.read_parquet(
        config.feature_data_path,
        columns=config.all_columns,
    )

    data[config.time_column] = pd.to_datetime(data[config.time_column])
    data[config.region_column] = data[config.region_column].astype(str)
    data = data.sort_values(
        [config.time_column, config.region_column]
    ).reset_index(drop=True)

    train_data = data[
        data[config.time_column].between(
            pd.Timestamp(config.train_start),
            pd.Timestamp(config.train_end),
            inclusive="both",
        )
    ].copy()

    valid_data = data[
        data[config.time_column].between(
            pd.Timestamp(config.valid_start),
            pd.Timestamp(config.valid_end),
            inclusive="both",
        )
    ].copy()

    test_data = data[
        data[config.time_column].between(
            pd.Timestamp(config.test_start),
            pd.Timestamp(config.test_end),
            inclusive="both",
        )
    ].copy()

    train_label = train_data[config.label_column].to_numpy(dtype=np.float32)

    if config.use_log1p:
        scaler_label = np.log1p(train_label)
    else:
        scaler_label = train_label

    if config.use_standardization:
        label_mean = float(scaler_label.mean())
        label_std = max(float(scaler_label.std(ddof=0)), 1e-6)
    else:
        label_mean = 0.0
        label_std = 1.0

    for split_data in [train_data, valid_data, test_data]:
        split_data["label_scaled"] = transform_label(
            split_data[config.label_column].to_numpy(dtype=np.float32),
            label_mean,
            label_std,
        )

    config.data_output_dir.mkdir(parents=True, exist_ok=True)
    config.model_output_dir.mkdir(parents=True, exist_ok=True)

    train_data.to_parquet(config.train_data_path, index=False)
    valid_data.to_parquet(config.valid_data_path, index=False)
    test_data.to_parquet(config.test_data_path, index=False)

    scaler = {
        "label_mean": label_mean,
        "label_std": label_std,
        "use_log1p": config.use_log1p,
        "use_standardization": config.use_standardization,
    }

    with open(config.label_scaler_path, "w", encoding="utf-8") as file:
        json.dump(scaler, file, ensure_ascii=False, indent=2)

    print(f"训练集样本数：{len(train_data)}")
    print(f"验证集样本数：{len(valid_data)}")
    print(f"测试集样本数：{len(test_data)}")
    print(f"标签均值：{label_mean:.6f}")
    print(f"标签标准差：{label_std:.6f}")


if __name__ == "__main__":
    prepare_data()
