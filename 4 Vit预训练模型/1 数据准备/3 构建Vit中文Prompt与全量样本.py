# -*- coding: utf-8 -*-
"""将Vit特征写成完整中文自然语言Prompt，并与6月1—24日全量标签合并。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("vit_pretrain_config", HERE / "1 Vit预训练配置.py")
cfg = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(cfg)

WEEKDAY_NAMES = ("星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日")

WEATHER_NAMES = {
    0: "晴朗", 1: "大致晴朗", 2: "局部多云", 3: "阴天", 45: "有雾", 48: "雾凇",
    51: "小毛毛雨", 53: "中等毛毛雨", 55: "强毛毛雨", 56: "轻微冻毛毛雨",
    57: "强冻毛毛雨", 61: "小雨", 63: "中雨", 65: "大雨", 66: "轻微冻雨",
    67: "强冻雨", 71: "小雪", 73: "中雪", 75: "大雪", 77: "米雪",
    80: "小阵雨", 81: "中等阵雨", 82: "强阵雨", 85: "小阵雪", 86: "强阵雪",
    95: "雷暴", 96: "雷暴伴轻微冰雹", 99: "雷暴伴强冰雹",
}

PROFIT_WINDOWS = (
    ("lag1", "上一15分钟"),
    ("roll4", "过去1小时"),
    ("roll8", "过去2小时"),
    ("same_slot_last_day", "前一天同时段"),
    ("same_slot_last_week", "前一周同时段"),
)

# Prompt保留10%、50%、90%三个关键分位数，并将相邻5%区间合并为10个10%区间。
PROMPT_KEY_LEVELS = (0.10, 0.50, 0.90)
PROMPT_TOP_DESTINATIONS = 3
PROMPT_TOP_NEIGHBORS = 3


def number(value: object, digits: int = 2) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "缺失"
    if not np.isfinite(value):
        return "缺失"
    # Prompt保留最多两位小数，但删除无意义的末尾0：0.00→0、3.00→3、3.50→3.5。
    text = f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return "0" if text in {"-0", "+0"} else text


def weather_name(value: object) -> str:
    """兼容上游已经文本化的天气名称和原始WMO数字代码。"""

    if isinstance(value, str):
        text = value.strip()
        if text and not text.replace(".", "", 1).isdigit():
            return text
    try:
        code = int(float(value))
    except (TypeError, ValueError):
        return "天气信息不可用"
    return WEATHER_NAMES.get(code, f"天气代码{code}")


def distribution_paragraph(
    row: pd.Series,
    prefix: str,
    chinese_name: str,
) -> str:
    intervals: list[str] = []
    for left in range(0, 100, 10):
        lower_half = float(
            row[f"{prefix}_percentile_{left:02d}_{left + 5:02d}_mean"]
        )
        upper_half = float(
            row[f"{prefix}_percentile_{left + 5:02d}_{left + 10:02d}_mean"]
        )
        intervals.append(number((lower_half + upper_half) / 2.0))
    keys: list[str] = []
    for level in PROMPT_KEY_LEVELS:
        value = row[f"{prefix}_quantile_{level:.3f}"]
        keys.append(number(value))
    return (
        f"{chinese_name}：\n"
        "0%—10%、10%—20%、……、90%—100%分位区间的均值为："
        f"[{', '.join(intervals)}]\n"
        "10%、50%、90%分位数值为："
        f"[{', '.join(keys)}]"
    )


def period_text(row: pd.Series) -> str | None:
    descriptions: list[str] = []
    if int(row["is_weekend_or_holiday"]):
        descriptions.append("周末或节假日")
    if int(row["is_peak_morning"]):
        descriptions.append("早高峰")
    if int(row["is_peak_evening"]):
        descriptions.append("晚高峰")
    if int(row["is_night"]):
        descriptions.append("夜间")
    return "、".join(descriptions) if descriptions else None


def profit_paragraph(row: pd.Series) -> str:
    labels = "、".join(name for _, name in PROFIT_WINDOWS)
    per_minute = [
        number(row[f"driver_profit_per_min_i_{suffix}"])
        for suffix, _ in PROFIT_WINDOWS
    ]
    per_order = [
        number(row[f"driver_profit_per_order_i_{suffix}"])
        for suffix, _ in PROFIT_WINDOWS
    ]
    return (
        "历史机会收益：\n"
        f"时间窗口顺序=[{labels}]\n"
        f"每分钟服务净收益=[{', '.join(per_minute)}]\n"
        f"每单净收益=[{', '.join(per_order)}]"
    )


def destination_paragraph(row: pd.Series) -> str:
    items: list[str] = []
    for rank in range(1, min(PROMPT_TOP_DESTINATIONS, cfg.TOP_K_DESTINATIONS) + 1):
        region = str(row[f"destination_{rank}_region_id"] or "")
        if not region:
            continue
        items.append(
            f"{region}｜{number(row[f'destination_{rank}_probability'])}｜"
            f"{number(row[f'destination_{rank}_travel_time_q10'])}/"
            f"{number(row[f'destination_{rank}_travel_time_q50'])}/"
            f"{number(row[f'destination_{rank}_travel_time_q90'])}"
        )
    if not items:
        return "历史主要目的地及行程时间：\n目标时点之前没有可用的目的地历史。"
    return "\n".join(
        [
            "历史主要目的地及行程时间=[目的地区域｜历史到达概率｜行程时间q10/q50/q90，单位分钟]：",
            *items,
        ]
    )


def neighbor_paragraph(row: pd.Series) -> str:
    items: list[str] = []
    for rank in range(1, min(PROMPT_TOP_NEIGHBORS, cfg.TOP_K_NEIGHBORS) + 1):
        region = str(row[f"neighbor_{rank}_region_id"] or "")
        if not region:
            continue
        items.append(
            f"{region}｜{number(row[f'neighbor_{rank}_weight'])}｜"
            f"{number(row[f'neighbor_{rank}_demand_q10'])}/"
            f"{number(row[f'neighbor_{rank}_demand_q50'])}/"
            f"{number(row[f'neighbor_{rank}_demand_q90'])}｜"
            f"{number(row[f'neighbor_{rank}_supply_q10'])}/"
            f"{number(row[f'neighbor_{rank}_supply_q50'])}/"
            f"{number(row[f'neighbor_{rank}_supply_q90'])}"
        )
    return "\n".join(
        [
            "邻近区域=[区域｜空间权重｜需求分位数q10/q50/q90｜供给代理q10/q50/q90分位数]：",
            *items,
        ]
    )


def build_prompt(row: pd.Series) -> str:
    timestamp = pd.Timestamp(row["t"])
    dow = int(row["dow"])
    sections = [
        "任务：预测一名代表性司机从当前区域出发，在基于历史订单机会、无单原地等待、接单后随订单转移的被动基准策略下，未来4小时折现净收益的101个分位数",
        (
            f"时空=[区域{row['region_id']}, {timestamp:%Y-%m-%d %H:%M}, "
            f"{WEEKDAY_NAMES[dow] if 0 <= dow < len(WEEKDAY_NAMES) else f'星期编码{dow}'}]"
        ),
        (
            f"天气=[{weather_name(row['weather_code'])}, {number(row['temperature_2m'])}℃, "
            f"相对湿度{number(row['relative_humidity_2m'])}%, "
            f"降雨{number(row['rain'])}mm, "
            f"10米风速{number(row['wind_speed_10m'])}km/h]"
        ),
        distribution_paragraph(row, "demand", "订单需求/15min"),
        distribution_paragraph(row, "supply", "司机数/15min"),
        profit_paragraph(row),
        destination_paragraph(row),
        neighbor_paragraph(row),
        "输出目标：输出未来4小时折现净收益的101个分位数，分位水平依次为0.5%、1%、2%、……、99%和99.5%",
    ]
    return "\n\n".join(sections)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="构建Vit中文Prompt与全量监督样本，或只预览一条Prompt。"
    )
    parser.add_argument(
        "--preview-one",
        action="store_true",
        help="只从现有输入特征中随机生成并打印一条Prompt，不写全量训练文件。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="--preview-one使用的随机种子，默认42。",
    )
    parser.add_argument(
        "--split", choices=("train", "valid"), default="train",
        help="train构建6月1—24日；valid构建6月25—27日独立验证样本。",
    )
    return parser.parse_args()


def preview_one(seed: int, split: str) -> None:
    """从指定数据段输入特征中等概率抽一行，仅预览Prompt。"""

    days = cfg.days(split)
    files = [
        cfg.INPUT_FEATURE_DIR / f"vit_input_features_{day:%Y-%m-%d}.parquet"
        for day in days
    ]
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少Vit输入特征：\n" + "\n".join(missing[:5]))

    rng = np.random.default_rng(seed)
    file_index = int(rng.integers(0, len(files)))
    frame = pd.read_parquet(files[file_index])
    row_index = int(rng.integers(0, len(frame)))
    row = frame.iloc[row_index]
    prompt = build_prompt(row)
    report = "\n".join(
        [
            f"Vit单条Prompt预览：seed={seed}",
            f"来源文件：{files[file_index]}",
            f"文件内行号：{row_index}",
            f"决策时点：{pd.Timestamp(row['t'])}",
            f"区域：{row['region_id']}",
            "",
            "完整Prompt：",
            prompt,
            "",
        ]
    )
    output = cfg.DATA_PREPARATION_LOG_DIR / f"vit_prompt_preview_seed{seed}.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(report)
    print(f"预览已保存：{output}")


def assert_completed_labels(
    diagnostics_file: Path,
    temporary_file: Path,
    split: str,
    expected_rows: int,
) -> None:
    if temporary_file.exists():
        raise RuntimeError(
            f"检测到标签仍在生成：{temporary_file}。"
            "请等1000条轨迹程序完成并自动替换正式文件后再运行。"
        )
    diagnostics = json.loads(diagnostics_file.read_text(encoding="utf-8"))
    if int(diagnostics.get("monte_carlo_samples", 0)) != 1000:
        raise ValueError("当前正式Vit标签不是1000条轨迹版，拒绝继续。")
    if diagnostics.get("split", "train") != split:
        raise ValueError(f"Vit标签split错误：{diagnostics.get('split')} != {split}")
    if int(diagnostics.get("rows", 0)) != expected_rows:
        raise ValueError(
            f"{split}标签行数应为{expected_rows:,}，"
            f"实际{int(diagnostics.get('rows', 0)):,}。"
        )
    if diagnostics.get("negative_or_zero_net_order_policy") == "reject_and_wait_one_slot":
        raise ValueError("当前Vit标签使用了非正收益订单拒绝口径，与历史成交轨迹口径不一致。")


def main() -> None:
    args = parse_args()
    cfg.ensure_directories()
    if args.preview_one:
        preview_one(args.seed, args.split)
        return
    label_file, diagnostics_file, temporary_file = (
        (
            cfg.SOURCE_VIT_LABELS_101,
            cfg.SOURCE_VIT_LABELS_DIAGNOSTICS,
            cfg.SOURCE_VIT_LABELS_TEMP,
        )
        if args.split == "train"
        else (
            cfg.SOURCE_VIT_VALID_LABELS_101,
            cfg.SOURCE_VIT_VALID_LABELS_DIAGNOSTICS,
            cfg.SOURCE_VIT_VALID_LABELS_TEMP,
        )
    )
    expected_rows = (
        cfg.EXPECTED_TRAIN_ROWS if args.split == "train" else cfg.EXPECTED_VALID_ROWS
    )
    assert_completed_labels(
        diagnostics_file, temporary_file, args.split, expected_rows
    )
    mapping = pd.read_csv(cfg.REGION_MAP_FILE, dtype={"region_id": "string"})[
        ["region_id", "region_index"]
    ]
    mapping["region_id"] = mapping["region_id"].astype(str)
    labels = ds.dataset(label_file, format="parquet")
    rename = dict(zip(cfg.SOURCE_LABEL_COLUMNS, cfg.QUANTILE_COLUMNS))

    output_part_dir = cfg.TRAIN_PART_DIR if args.split == "train" else cfg.VALID_PART_DIR
    for day in cfg.days(args.split):
        feature_file = cfg.INPUT_FEATURE_DIR / f"vit_input_features_{day:%Y-%m-%d}.parquet"
        if not feature_file.exists():
            raise FileNotFoundError(f"缺少Vit输入特征：{feature_file}")
        frame = pd.read_parquet(feature_file)
        frame["region_id"] = frame["region_id"].astype(str)
        frame["t"] = pd.to_datetime(frame["t"])
        frame = frame.merge(mapping, on="region_id", how="left", validate="many_to_one")
        frame["text"] = [build_prompt(row) for _, row in frame.iterrows()]

        demand_cols = (
            [f"demand_percentile_{left:02d}_{left + 5:02d}_mean" for left in range(0, 100, 5)]
            + [f"demand_quantile_{level:.3f}" for level in cfg.KEY_LEVELS]
        )
        demand = np.log1p(frame[demand_cols].clip(lower=0).to_numpy(np.float32))
        frame[cfg.DEMAND_FEATURE_COLUMNS] = demand
        frame["log_supply"] = np.log1p(
            frame["supply_quantile_0.500"].clip(lower=0).to_numpy(np.float32)
        )

        table = labels.to_table(
            filter=(ds.field("t") >= day.to_pydatetime())
            & (ds.field("t") < (day + pd.Timedelta(days=1)).to_pydatetime())
        ).to_pandas()
        table["region_id"] = table["region_id"].astype(str)
        table["t"] = pd.to_datetime(table["t"])
        table = table.rename(columns=rename)
        sample = frame.merge(
            table[["region_id", "region_index", "t", *cfg.QUANTILE_COLUMNS]],
            on=["region_id", "region_index", "t"], how="inner", validate="one_to_one",
        ).sort_values(["t", "region_index"])
        expected = cfg.TIME_STEPS_PER_DAY * len(mapping)
        if len(sample) != expected:
            raise ValueError(f"{day.date()}样本应为{expected:,}行，实际{len(sample):,}行。")

        state_columns = [
            "region_id", "region_index", "t", "text", "log_supply",
            *cfg.DEMAND_FEATURE_COLUMNS,
        ]
        sample[state_columns].to_parquet(
            cfg.STATE_TEXT_DIR / f"vit_state_{day:%Y-%m-%d}.parquet",
            index=False, compression="zstd",
        )
        sample[state_columns + cfg.QUANTILE_COLUMNS].to_parquet(
            output_part_dir / f"vit_{args.split}_{day:%Y-%m-%d}.parquet",
            index=False, compression="zstd",
        )
        print(
            f"已构建 {day.date()} {args.split}中文Prompt与监督样本：{len(sample):,}行"
        )


if __name__ == "__main__":
    main()
