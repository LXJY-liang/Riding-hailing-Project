from pathlib import Path
from datetime import datetime, timezone
import importlib.util
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

"""
运行结果：

训练集样本数：3549356
验证集样本数：380811
测试集样本数：527658
标签均值：2.743131
标签标准差：0.589868
数据来源与样本数清单：/Users/liangxue/Documents/8、Python文件-网约车数据/Riding hailing_Project/3 随机变量分位数预测/4 tauijt预测/output/data/tauijt_data_manifest.json
"""
# =========================
# 1. 读取配置
# =========================

def load_config():
    config_path = Path(__file__).resolve().parent / "1 tauijt配置.py"
    spec = importlib.util.spec_from_file_location("tauijt_config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


config = load_config()


def validate_latest_feature_input():
    """阻止旧 tauijt 特征覆盖最新异常订单清洗链。"""

    required_paths = [
        config.cleaned_od_panel_path,
        config.feature_data_path,
        config.feature_source_manifest_path,
    ]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "tauijt 数据处理缺少最新上游产物，请先运行特征构造：\n"
            + "\n".join(missing)
        )

    if config.feature_data_path.stat().st_mtime < config.cleaned_od_panel_path.stat().st_mtime:
        raise RuntimeError(
            "tauijt特征文件早于最新OD面板，继续运行会使用旧清洗样本。"
            "请先重新运行‘2 模型特征构建/4 tauijt特征构造.py’。"
        )

    with open(config.feature_source_manifest_path, "r", encoding="utf-8") as file:
        source_manifest = json.load(file)
    if source_manifest.get("od_panel_mtime_ns") != config.cleaned_od_panel_path.stat().st_mtime_ns:
        raise RuntimeError(
            "tauijt特征来源清单与当前OD面板不一致。请重新运行 tauijt 特征构造。"
        )


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
    """将数值写入自然语言；缺失值写为“未知”。"""

    if pd.isna(value):
        return "未知"

    return f"{float(value):.{digits}f}"


def history_text(value, available=True, digits=2, unit=""):
    """历史可用时输出数值和单位；不可用时输出“未知”。"""

    if not available or pd.isna(value):
        return "未知"

    return f"{float(value):.{digits}f}{unit}"


def special_period_text(row):
    """仅描述早高峰、晚高峰和夜间等特殊时段。"""

    descriptions = []

    if int(row["is_peak_morning"]) == 1:
        descriptions.append("早高峰")

    if int(row["is_peak_evening"]) == 1:
        descriptions.append("晚高峰")

    if int(row["is_night"]) == 1:
        descriptions.append("夜间")

    return "、".join(descriptions)


# =========================
# 3. 将一条样本转换为自然语言
# =========================

def build_text(row):
    """将 tau_ijt 特征按参考优先级转换为中文自然语言。"""

    timestamp = pd.Timestamp(row[config.time_column])
    day_index = int(row["day_index"])
    dow = int(row["dow"])
    weather_code = int(row["weather_code"])
    day_type = (
        "周末或节假日"
        if int(row["is_weekend_or_holiday"]) == 1
        else "工作日"
    )
    region_relation = (
        "是区域内订单"
        if int(row["same_region_ij"]) == 1
        else "不是区域内订单"
    )
    special_period = special_period_text(row)
    distance_text = (
        f"区域质心距离为{number_text(row['dist_ij'], 4)}公里，"
        f"{region_relation}"
    )
    if special_period:
        distance_text += f"，当前为{special_period}"
    distance_text += "。"

    return "\n".join(
        [
            "任务：预测当前OD当前时段的订单服务时间分布。",
            (
                "参考优先级：一级为同一OD短期和同一时段历史服务时间；"
                "二级为同一OD长期历史服务时间；"
                "三级为相似距离范围的服务时间兜底。"
            ),
            "OD与时间状态：",
            (
                f"出发区域为{row[config.i_column]}，"
                f"到达区域为{row[config.j_column]}，"
                f"时间为{timestamp.strftime('%Y年%m月%d日 %H:%M')}，"
                f"{weekday_map.get(dow, f'星期编码{dow}')}，"
                f"当前为{day_type}。"
            ),
            distance_text,
            "天气状态：",
            (
                f"气温为{number_text(row['temperature_2m'])}摄氏度，"
                f"相对湿度为{number_text(row['relative_humidity_2m'])}%，"
                f"降雨量为{number_text(row['rain'])}毫米，"
                f"天气状况为{weather_map.get(weather_code, f'天气代码{weather_code}')}，"
                f"10米高度风速为{number_text(row['wind_speed_10m'])}千米/小时。"
            ),
            "一级参考——同一OD短期和同一时段历史服务时间：",
            (
                f"上一15分钟平均服务时间："
                f"{history_text(row['tau_ij_lag1'], unit='分钟')}；"
                f"上两个15分钟时段前平均服务时间："
                f"{history_text(row['tau_ij_lag2'], unit='分钟')}。"
            ),
            (
                f"过去1小时平均服务时间："
                f"{history_text(row['tau_ij_roll4_mean'], unit='分钟')}；"
                f"过去2小时平均服务时间："
                f"{history_text(row['tau_ij_roll8_mean'], unit='分钟')}。"
            ),
            (
                f"前一天同一时段平均服务时间："
                f"{history_text(row['tau_ij_same_slot_last_day'], day_index >= 2, unit='分钟')}；"
                f"前一周同一时段平均服务时间："
                f"{history_text(row['tau_ij_same_slot_last_week'], day_index >= 8, unit='分钟')}。"
            ),
            "二级参考——同一OD长期历史服务时间：",
            (
                f"前一天平均服务时间："
                f"{history_text(row['tau_ij_mean_last_day'], day_index >= 2, unit='分钟')}；"
                f"前一天中位服务时间："
                f"{history_text(row['tau_ij_median_last_day_proxy'], day_index >= 2, unit='分钟')}。"
            ),
            (
                f"过去3天平均服务时间："
                f"{history_text(row['tau_ij_mean_last3'], day_index >= 4, unit='分钟')}；"
                f"过去7天平均服务时间："
                f"{history_text(row['tau_ij_mean_last7'], day_index >= 8, unit='分钟')}；"
                f"过去7天历史订单样本数："
                f"{history_text(row['tau_ij_count_last7'], day_index >= 8, 0, '单')}。"
            ),
            "三级参考——相似距离范围服务时间兜底：",
            (
                f"相似距离且同一时段全局平均服务时间："
                f"{history_text(row['global_tau_by_dist_slot'], unit='分钟')}；"
                f"相似距离OD历史平均服务时间："
                f"{history_text(row['similar_OD_tau_mean'], unit='分钟')}。"
            ),
        ]
    )


def build_text_batch(batch):
    """将datasets的一个批次转换为自然语言文本。"""

    batch_size = len(batch[config.i_column])
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
    """对运行时间标签依次进行log1p和标准化。"""

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
    """
    分块读取完整tau_ijt特征，并按照日期划分训练、验证和测试集。
    只改变大文件读取方式，其余处理流程与Dit一致。
    """

    validate_latest_feature_input()

    train_start = pd.Timestamp(config.train_start)
    train_end = pd.Timestamp(config.train_end)
    valid_start = pd.Timestamp(config.valid_start)
    valid_end = pd.Timestamp(config.valid_end)
    test_start = pd.Timestamp(config.test_start)
    test_end = pd.Timestamp(config.test_end)

    parquet_file = pq.ParquetFile(config.feature_data_path)

    # 先只读取时间和标签，计算训练集标签变换参数
    label_count = 0
    label_sum = 0.0
    label_square_sum = 0.0

    for batch in parquet_file.iter_batches(
        batch_size=200000,
        columns=[config.time_column, config.label_column],
    ):
        data = batch.to_pandas()
        data[config.time_column] = pd.to_datetime(data[config.time_column])

        mask = (
            data[config.time_column].between(
                train_start,
                train_end,
                inclusive="both",
            )
            & data[config.label_column].notna()
        )

        labels = data.loc[
            mask,
            config.label_column,
        ].to_numpy(dtype=np.float64)

        if config.use_log1p:
            labels = np.log1p(labels)

        label_count += len(labels)
        label_sum += labels.sum()
        label_square_sum += np.square(labels).sum()

    if label_count == 0:
        raise ValueError("最新特征数据中没有可用于训练的非空 tauijt 标签。")

    if config.use_standardization:
        label_mean = label_sum / label_count
        label_variance = (
            label_square_sum / label_count
            - label_mean ** 2
        )
        label_std = max(
            float(np.sqrt(max(label_variance, 0.0))),
            1e-6,
        )
    else:
        label_mean = 0.0
        label_std = 1.0

    config.data_output_dir.mkdir(parents=True, exist_ok=True)
    config.model_output_dir.mkdir(parents=True, exist_ok=True)

    writers = {
        "train": None,
        "valid": None,
        "test": None,
    }

    output_paths = {
        "train": config.train_data_path,
        "valid": config.valid_data_path,
        "test": config.test_data_path,
    }

    sample_counts = {
        "train": 0,
        "valid": 0,
        "test": 0,
    }

    # 再分块读取完整特征，完成数据划分、标签变换和写出
    for batch in parquet_file.iter_batches(
        batch_size=50000,
        columns=config.all_columns,
    ):
        data = batch.to_pandas()
        data[config.time_column] = pd.to_datetime(data[config.time_column])

        split_masks = {
            "train": data[config.time_column].between(
                train_start,
                train_end,
                inclusive="both",
            ),
            "valid": data[config.time_column].between(
                valid_start,
                valid_end,
                inclusive="both",
            ),
            "test": data[config.time_column].between(
                test_start,
                test_end,
                inclusive="both",
            ),
        }

        for split_name, split_mask in split_masks.items():
            split_data = data.loc[
                split_mask
                & data[config.label_column].notna()
            ].copy()

            if split_data.empty:
                continue

            split_data[config.i_column] = (
                split_data[config.i_column].astype(str)
            )
            split_data[config.j_column] = (
                split_data[config.j_column].astype(str)
            )
            split_data[config.od_id_column] = (
                split_data[config.od_id_column].astype(str)
            )

            split_data["label_scaled"] = transform_label(
                split_data[config.label_column].to_numpy(dtype=np.float32),
                label_mean,
                label_std,
            )

            table = pa.Table.from_pandas(
                split_data,
                preserve_index=False,
            )

            if writers[split_name] is None:
                writers[split_name] = pq.ParquetWriter(
                    output_paths[split_name],
                    table.schema,
                    compression="snappy",
                )

            writers[split_name].write_table(table)
            sample_counts[split_name] += len(split_data)

    for writer in writers.values():
        if writer is not None:
            writer.close()

    scaler = {
        "label_mean": float(label_mean),
        "label_std": float(label_std),
        "use_log1p": config.use_log1p,
        "use_standardization": config.use_standardization,
    }

    with open(
        config.label_scaler_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            scaler,
            file,
            ensure_ascii=False,
            indent=2,
        )

    data_manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_data_path": str(config.feature_data_path),
        "feature_data_mtime_ns": config.feature_data_path.stat().st_mtime_ns,
        "cleaned_od_panel_path": str(config.cleaned_od_panel_path),
        "cleaned_od_panel_mtime_ns": config.cleaned_od_panel_path.stat().st_mtime_ns,
        "split_counts": {name: int(count) for name, count in sample_counts.items()},
        "split_ranges": {
            "train": [config.train_start, config.train_end],
            "valid": [config.valid_start, config.valid_end],
            "test": [config.test_start, config.test_end],
        },
        "label_scaler": scaler,
    }
    with open(config.data_manifest_path, "w", encoding="utf-8") as file:
        json.dump(data_manifest, file, ensure_ascii=False, indent=2)

    print(f"训练集样本数：{sample_counts['train']}")
    print(f"验证集样本数：{sample_counts['valid']}")
    print(f"测试集样本数：{sample_counts['test']}")
    print(f"标签均值：{label_mean:.6f}")
    print(f"标签标准差：{label_std:.6f}")
    print(f"数据来源与样本数清单：{config.data_manifest_path}")


if __name__ == "__main__":
    prepare_data()
