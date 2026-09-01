# -*- coding: utf-8 -*-
"""将Vit全部历史状态特征先写成文本Prompt，再构建监督预训练样本。"""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as pds


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("vit_pretrain_config", HERE / "1 Vit预训练配置.py")
cfg = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(cfg)

KEY_LEVELS = cfg.DEMAND_KEY_LEVELS
HISTORY_COLUMNS = [
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


def number(value: object, digits: int = 3) -> str:
    """将数值紧凑地写入Prompt，缺失值显式标记为NA。"""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{numeric:.{digits}f}" if np.isfinite(numeric) else "NA"


def text_value(row: pd.Series, *names: str, default: str = "NA") -> str:
    for name in names:
        if name in row.index and pd.notna(row[name]):
            return str(row[name])
    return default


def numeric_value(row: pd.Series, *names: str) -> str:
    for name in names:
        if name in row.index and pd.notna(row[name]):
            return number(row[name])
    return "NA"


def distribution_text(row: pd.Series, prefix: str) -> str:
    bins = [
        number(row[f"{prefix}_qbin_{left:02d}_{left + 5:02d}_mean"])
        for left in range(0, 100, 5)
    ]
    keys = [number(row[f"{prefix}_q_{level:.3f}"]) for level in KEY_LEVELS]
    count = int(row[f"{prefix[0]}_hist_sample_count"])
    return f"bins5=[{','.join(bins)}];keys005/05/10/50/90/95/995=[{','.join(keys)}];n={count}"


def destination_text(row: pd.Series) -> str:
    items = []
    for rank in range(1, cfg.TOP_K_DESTINATIONS + 1):
        region = text_value(row, f"dest_{rank}_region_id", default="")
        items.append(
            f"{rank}:{region or 'NA'},{numeric_value(row, f'dest_{rank}_pi')},"
            f"{numeric_value(row, f'dest_{rank}_tau_q0.10')},"
            f"{numeric_value(row, f'dest_{rank}_tau_q0.50')},"
            f"{numeric_value(row, f'dest_{rank}_tau_q0.90')}"
        )
    return ";".join(items)


def neighbor_text(row: pd.Series) -> str:
    items = []
    for rank in range(1, cfg.TOP_K_NEIGHBORS + 1):
        region = text_value(row, f"neighbor_{rank}_region_id", default="")
        items.append(
            f"{rank}:{region or 'NA'},{numeric_value(row, f'neighbor_{rank}_weight')},"
            f"{numeric_value(row, f'neighbor_{rank}_D_q0.10')},"
            f"{numeric_value(row, f'neighbor_{rank}_D_q0.50')},"
            f"{numeric_value(row, f'neighbor_{rank}_D_q0.90')},"
            f"{numeric_value(row, f'neighbor_{rank}_S_q0.10')},"
            f"{numeric_value(row, f'neighbor_{rank}_S_q0.50')},"
            f"{numeric_value(row, f'neighbor_{rank}_S_q0.90')}"
        )
    return ";".join(items)


def build_prompt(row: pd.Series) -> str:
    """构建Qwen的唯一输入：所有结构化特征先显式文本化。"""

    history = ",".join(number(row[name]) for name in HISTORY_COLUMNS)
    return "\n".join(
        [
            "任务：预测代表性司机从当前区域和时段出发、未来4小时折现净收益的101维分位数。",
            f"时空：region={row['region_id']};day_index={int(row['day_index'])};"
            f"slot={int(row['slot_in_day'])};dow={int(row['dow'])};"
            f"weekend_holiday={int(row['is_weekend_or_holiday'])};"
            f"peak_morning={int(row['is_peak_morning'])};"
            f"peak_evening={int(row['is_peak_evening'])};night={int(row['is_night'])}。",
            f"天气：temperature_2m={numeric_value(row, 'temperature_2m', 'temperature')};"
            f"relative_humidity_2m={numeric_value(row, 'relative_humidity_2m', 'humidity')};"
            f"rain={numeric_value(row, 'rain')};weather={text_value(row, 'weather', 'weather_code')};"
            f"wind_speed_10m={numeric_value(row, 'wind_speed_10m', 'wind')}。",
            f"历史需求经验分布D_it：{distribution_text(row, 'D_hist')}。",
            f"历史可服务供给代理分布S_it：{distribution_text(row, 'S_hist')};"
            f"fallback={int(row['history_fallback_level'])}。",
            f"Top10历史目的地[rank:region,pi,tau10,tau50,tau90]：{destination_text(row)}。",
            f"目的地概率质量：top10_mass={number(row['pi_top10_mass'])};"
            f"other_mass={number(row['pi_other_mass'])};same_region={number(row['pi_same_region'])};"
            f"available={int(row['pi_history_available'])}。",
            "历史司机机会收益["
            "ppm_lag1,ppo_lag1,ppm_roll4,ppo_roll4,ppm_roll8,ppo_roll8,"
            "ppm_day,ppo_day,ppm_week,ppo_week]（1%-99% winsorization）="
            f"[{history}]。",
            f"Top5邻近区域[rank:region,weight,D10,D50,D90,S10,S50,S90]：{neighbor_text(row)}。",
            f"关联区域滞后特征：related_dest_demand_lag1={number(row['related_dest_demand_lag1'])};"
            f"related_origin_inflow_lag1={number(row['related_origin_inflow_lag1'])};"
            f"related_pressure_lag1={number(row['related_pressure_lag1'])}。",
        ]
    )


def build_day(day: pd.Timestamp, mapping: pd.DataFrame) -> pd.DataFrame:
    source = cfg.HISTORICAL_FEATURE_DIR / f"vit_history_{day:%Y-%m-%d}.parquet"
    if not source.exists():
        raise FileNotFoundError(f"缺少Vit历史特征：{source}")
    state = pd.read_parquet(source)
    state["region_id"] = state["region_id"].astype(str)
    state["t"] = pd.to_datetime(state["t"])
    state = state.merge(mapping, on="region_id", how="left", validate="many_to_one")
    if state["region_index"].isna().any():
        raise ValueError("历史特征中存在无法映射的区域。")
    state["region_index"] = state["region_index"].astype(np.int16)
    state["log_supply"] = np.log1p(state["S_hist_q_0.500"].clip(lower=0)).astype(np.float32)
    demand_source_columns = [
        f"D_hist_qbin_{left:02d}_{left + 5:02d}_mean"
        for left in range(0, 100, 5)
    ] + [f"D_hist_q_{level:.3f}" for level in KEY_LEVELS]
    demand_summary = np.log1p(
        state[demand_source_columns].clip(lower=0).to_numpy(np.float32)
    )
    state[cfg.DEMAND_FEATURE_COLUMNS] = demand_summary
    # Qwen看到的不是原始表格行，而是这里构建的完整文本Prompt。
    state["text"] = state.apply(build_prompt, axis=1)
    return state[
        ["region_id", "region_index", "t", "log_supply", "text"]
        + cfg.DEMAND_FEATURE_COLUMNS
    ]


def build_split(
    split: str,
    label_dataset: pds.Dataset,
    mapping: pd.DataFrame,
    rng: np.random.Generator,
) -> None:
    start, end = map(pd.Timestamp, cfg.SPLITS[split])
    days = pd.date_range(start.normalize(), end.normalize(), freq="D")
    per_day = max(1, math.ceil(cfg.MAX_SAMPLES[split] / len(days)))
    samples = []
    for day in days:
        state = build_day(day, mapping)
        state.to_parquet(
            cfg.STATE_TEXT_DIR / f"vit_state_{day:%Y-%m-%d}.parquet",
            index=False,
            compression="snappy",
        )
        next_day = day + pd.Timedelta(days=1)
        labels = label_dataset.to_table(
            filter=(pds.field("t") >= day.to_pydatetime())
            & (pds.field("t") < next_day.to_pydatetime())
        ).to_pandas()
        labels["region_id"] = labels["region_id"].astype(str)
        labels["t"] = pd.to_datetime(labels["t"])
        merged = state.merge(
            labels,
            on=["region_id", "region_index", "t"],
            how="inner",
            validate="one_to_one",
        )
        if merged.empty:
            continue
        take = min(per_day, len(merged))
        samples.append(merged.iloc[rng.choice(len(merged), size=take, replace=False)])
        print(
            f"{split}：已构建 {day:%Y-%m-%d} 的 {len(state):,} 个(i,t)文本Prompt"
        )

    if not samples:
        raise ValueError(f"{split}没有与Vit标签匹配的状态。")
    result = pd.concat(samples, ignore_index=True)
    if len(result) > cfg.MAX_SAMPLES[split]:
        result = result.iloc[rng.choice(len(result), size=cfg.MAX_SAMPLES[split], replace=False)]
    result = result.sort_values(["t", "region_index"], kind="mergesort").reset_index(drop=True)
    result.to_parquet(cfg.PRETRAIN_FILES[split], index=False, compression="snappy")
    print(f"{split}：预训练样本 {len(result):,} 行。")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", choices=tuple(cfg.SPLITS), default=list(cfg.SPLITS))
    args = parser.parse_args()
    cfg.ensure_directories()
    mapping = pd.read_csv(cfg.REGION_MAP_FILE, dtype={"region_id": "string"})[
        ["region_id", "region_index"]
    ]
    mapping["region_id"] = mapping["region_id"].astype(str)
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    for split in args.splits:
        labels = pds.dataset(cfg.LABEL_FILES[split], format="parquet")
        build_split(split, labels, mapping, rng)


if __name__ == "__main__":
    main()
