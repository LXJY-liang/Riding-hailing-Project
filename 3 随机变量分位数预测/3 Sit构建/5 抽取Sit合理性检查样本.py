# -*- coding: utf-8 -*-
"""抽取 15 个 (i,t) 样本，用于人工检查构造的 S_it 是否合理。

默认自动读取当前 S_it 文件的实际时间范围，并在该范围内抽取：

1. 历史成交量 D_origin_it 最大的 5 个样本；
2. 从 D_origin_it = 0 的决策点中随机抽取 5 个样本；
3. 从 D_origin_it > 0 且不属于前 5 的决策点中随机抽取 5 个样本。

随机抽样使用固定种子，因此输入文件不变时可以复现相同样本。输出同时包含
历史订单流、S_it 的代表性分位数、U_it 摘要和若干供需检查指标。
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
OUTPUT_DIR = HERE / "output"

REGION_PANEL_FILE = (
    PROJECT_ROOT / "1 数据清洗" / "3_3 区域-15min聚合（补全空值）结果.parquet"
)
PREVIOUS_DEPARTURE_FILE = OUTPUT_DIR / "M_out_i_t_minus_1.parquet"
CURRENT_ARRIVAL_FILE = OUTPUT_DIR / "M_arrive_it.parquet"

DEFAULT_START = pd.Timestamp("2024-06-11 00:00:00")
MAXIMUM_END = pd.Timestamp("2024-06-30 23:45:00")

KEY_COLUMNS = ["region_id", "t"]
SUPPLY_QUANTILES = [
    "q_0.005",
    "q_0.050",
    "q_0.100",
    "q_0.250",
    "q_0.500",
    "q_0.750",
    "q_0.900",
    "q_0.950",
    "q_0.995",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="抽取样本检查 S_it 构造合理性。")
    parser.add_argument("--start", default=str(DEFAULT_START))
    parser.add_argument(
        "--input-prefix",
        default="learned_U",
        help="待检查的 S/U 文件前缀，默认 learned_U。",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="抽样结束时刻；默认使用当前 S_it 文件中的最大时刻。",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="每一类抽取的样本数，默认 5，总计 15 个样本。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20240611,
        help="两组随机样本的固定随机种子。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已经存在的检查结果。",
    )
    return parser.parse_args()


def build_paths(prefix: str) -> tuple[Path, Path, Path, Path, Path]:
    if not prefix.replace("_", "").replace("-", "").isalnum():
        raise ValueError("input-prefix 只能包含字母、数字、下划线和短横线。")
    if not prefix.replace("_", "").replace("-", "").isalnum():
        raise ValueError("input-prefix 只能包含字母、数字、下划线和短横线。")
    supply = OUTPUT_DIR / f"S_it_{prefix}_quantiles.parquet"
    u_summary = OUTPUT_DIR / f"U_it_{prefix}_summary.parquet"
    suffix = "" if prefix == "learned_U" else f"_{prefix}"
    csv = OUTPUT_DIR / f"Sit_reasonableness_check_15_samples{suffix}.csv"
    parquet = OUTPUT_DIR / f"Sit_reasonableness_check_15_samples{suffix}.parquet"
    metadata = OUTPUT_DIR / f"Sit_reasonableness_check_15_samples_metadata{suffix}.json"
    return supply, u_summary, csv, parquet, metadata


def validate_args(
    args: argparse.Namespace,
    supply_file: Path,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(args.start)
    supply_times = pd.to_datetime(
        pd.read_parquet(supply_file, columns=["t"])["t"], errors="raise"
    )
    supply_start = pd.Timestamp(supply_times.min())
    supply_end = pd.Timestamp(supply_times.max())
    end = supply_end if args.end is None else pd.Timestamp(args.end)
    if start < DEFAULT_START or end > MAXIMUM_END or end < start:
        raise ValueError("抽样范围必须位于 2024-06-11 至 2024-06-30。")
    if start < supply_start or end > supply_end:
        raise ValueError(
            f"抽样范围超出当前 S_it 文件的覆盖范围：{supply_start} 至 {supply_end}。"
        )
    if start.minute % 15 or end.minute % 15:
        raise ValueError("起止时间必须对齐到 15 分钟。")
    if args.sample_size < 1:
        raise ValueError("sample-size 必须为正整数。")
    return start, end


def check_inputs_and_outputs(
    overwrite: bool,
    supply_file: Path,
    u_summary_file: Path,
    output_files: list[Path],
) -> None:
    required = [
        REGION_PANEL_FILE,
        supply_file,
        u_summary_file,
        PREVIOUS_DEPARTURE_FILE,
        CURRENT_ARRIVAL_FILE,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少输入文件：\n" + "\n".join(missing))

    existing = [str(path) for path in output_files if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "检查结果已经存在；如需重建请使用 --overwrite：\n"
            + "\n".join(existing)
        )


def normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["region_id"] = frame["region_id"].astype(str)
    frame["t"] = pd.to_datetime(frame["t"], errors="raise")
    return frame


def select_samples(
    start: pd.Timestamp,
    end: pd.Timestamp,
    sample_size: int,
    seed: int,
) -> pd.DataFrame:
    panel_columns = [
        "region_id",
        "t",
        "day_type",
        "slot_in_day",
        "is_peak_morning",
        "is_peak_evening",
        "is_night",
        "D_origin_it",
        "D_dest_it",
        "net_flow_it",
    ]
    panel = pd.read_parquet(REGION_PANEL_FILE, columns=panel_columns)
    panel = normalize_keys(panel)
    panel["day_of_week"] = panel["t"].dt.dayofweek.astype(np.int8)
    panel = panel.loc[panel["t"].between(start, end)].copy()
    if panel.empty:
        raise ValueError("指定时间范围内没有历史成交数据。")
    if panel[KEY_COLUMNS].duplicated().any():
        raise ValueError("历史区域面板中存在重复的 (region_id,t)。")

    # 稳定排序使前 5 个样本在成交量并列时仍然可复现。
    ranked = panel.sort_values(
        ["D_origin_it", "t", "region_id"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    top = ranked.head(sample_size).copy()
    if len(top) < sample_size:
        raise ValueError("历史区域面板不足以抽取指定数量的高成交样本。")
    top["sample_group"] = "历史成交量前5"
    top["sample_rank_in_group"] = np.arange(1, len(top) + 1, dtype=np.int16)

    zero_candidates = panel.loc[panel["D_origin_it"].eq(0)].copy()
    if len(zero_candidates) < sample_size:
        raise ValueError("历史成交量为 0 的候选样本不足。")
    zero = zero_candidates.sample(n=sample_size, random_state=seed).copy()
    zero = zero.sort_values(["t", "region_id"], kind="mergesort")
    zero["sample_group"] = "历史成交量为0随机样本"
    zero["sample_rank_in_group"] = np.arange(1, len(zero) + 1, dtype=np.int16)

    top_keys = pd.MultiIndex.from_frame(top[KEY_COLUMNS])
    panel_keys = pd.MultiIndex.from_frame(panel[KEY_COLUMNS])
    positive_candidates = panel.loc[
        panel["D_origin_it"].gt(0) & ~panel_keys.isin(top_keys)
    ].copy()
    if len(positive_candidates) < sample_size:
        raise ValueError("非前 5 的正成交量候选样本不足。")
    positive = positive_candidates.sample(
        n=sample_size,
        random_state=seed + 1,
    ).copy()
    positive = positive.sort_values(["t", "region_id"], kind="mergesort")
    positive["sample_group"] = "正成交量非前5随机样本"
    positive["sample_rank_in_group"] = np.arange(
        1, len(positive) + 1, dtype=np.int16
    )

    samples = pd.concat([top, zero, positive], ignore_index=True)
    samples.insert(0, "sample_id", np.arange(1, len(samples) + 1, dtype=np.int16))
    return samples


def read_only_selected_rows(
    path: Path,
    columns: list[str],
    sample_keys: pd.MultiIndex,
) -> pd.DataFrame:
    """只保留抽中键；读取时限制列数，避免加载 101 个完整分位数列。"""
    frame = pd.read_parquet(path, columns=columns)
    frame = normalize_keys(frame)
    frame_keys = pd.MultiIndex.from_frame(frame[KEY_COLUMNS])
    selected = frame.loc[frame_keys.isin(sample_keys)].copy()
    if selected[KEY_COLUMNS].duplicated().any():
        raise ValueError(f"{path.name} 存在重复的 (region_id,t)。")
    if len(selected) != len(sample_keys):
        raise ValueError(
            f"{path.name} 只匹配到 {len(selected)} 个样本，预期 {len(sample_keys)} 个。"
        )
    del frame, frame_keys
    gc.collect()
    return selected


def add_source_data(
    samples: pd.DataFrame,
    supply_file: Path,
    u_summary_file: Path,
) -> pd.DataFrame:
    sample_keys = pd.MultiIndex.from_frame(samples[KEY_COLUMNS])

    supply = read_only_selected_rows(
        supply_file,
        [*KEY_COLUMNS, *SUPPLY_QUANTILES],
        sample_keys,
    )
    u_summary_columns = [
        *KEY_COLUMNS,
        "historical_departures_it",
        "reference_supply_profile",
        "response_speed",
        "pre_u_supply_mean",
        "mean_reversion_u_mean",
        "seasonal_u_mean",
        "conditional_u_mean",
        "u_mean",
        "u_std",
        "nonnegative_projection_rate",
        "historical_feasibility_common_shift",
        "historical_feasibility_tail_clip_rate",
        "historical_feasibility_topup_mean",
        "historical_feasibility_topup_rate",
        "historical_departure_infeasibility_rate",
        "historical_flow_deficit_rate",
    ]
    u_summary = read_only_selected_rows(
        u_summary_file,
        u_summary_columns,
        sample_keys,
    )
    previous_departures = read_only_selected_rows(
        PREVIOUS_DEPARTURE_FILE,
        [*KEY_COLUMNS, "source_t", "source_available", "M_out_i_t_minus_1"],
        sample_keys,
    )
    arrivals = read_only_selected_rows(
        CURRENT_ARRIVAL_FILE,
        [*KEY_COLUMNS, "M_arrive_it"],
        sample_keys,
    )

    result = samples.merge(supply, on=KEY_COLUMNS, how="left", validate="one_to_one")
    result = result.merge(u_summary, on=KEY_COLUMNS, how="left", validate="one_to_one")
    result = result.merge(
        previous_departures,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    result = result.merge(arrivals, on=KEY_COLUMNS, how="left", validate="one_to_one")
    return result


def add_audit_metrics(result: pd.DataFrame) -> pd.DataFrame:
    result = result.copy()
    result["historical_driver_net_flow"] = (
        result["M_arrive_it"] - result["M_out_i_t_minus_1"]
    )
    result["supply_q50_minus_transactions"] = (
        result["q_0.500"] - result["D_origin_it"]
    )
    result["supply_q05_minus_transactions"] = (
        result["q_0.050"] - result["D_origin_it"]
    )
    result["supply_q95_minus_transactions"] = (
        result["q_0.950"] - result["D_origin_it"]
    )
    result["supply_q90_interval_width"] = result["q_0.950"] - result["q_0.050"]
    result["transactions_le_supply_q05"] = (
        result["D_origin_it"] <= result["q_0.050"]
    )
    result["transactions_le_supply_q50"] = (
        result["D_origin_it"] <= result["q_0.500"]
    )
    result["transactions_le_supply_q95"] = (
        result["D_origin_it"] <= result["q_0.950"]
    )

    positive_demand = result["D_origin_it"].gt(0)
    positive_supply = result["q_0.500"].gt(0)
    result["supply_q50_to_transactions_ratio"] = np.where(
        positive_demand,
        result["q_0.500"] / result["D_origin_it"],
        np.nan,
    )
    result["historical_utilization_at_q50"] = np.where(
        positive_supply,
        result["D_origin_it"] / result["q_0.500"],
        np.nan,
    )
    return result


def ordered_output_columns() -> list[str]:
    return [
        "sample_id",
        "sample_group",
        "sample_rank_in_group",
        "region_id",
        "t",
        "day_type",
        "day_of_week",
        "slot_in_day",
        "is_peak_morning",
        "is_peak_evening",
        "is_night",
        "D_origin_it",
        "D_dest_it",
        "net_flow_it",
        "source_t",
        "source_available",
        "M_out_i_t_minus_1",
        "M_arrive_it",
        "historical_driver_net_flow",
        "historical_departures_it",
        "reference_supply_profile",
        "response_speed",
        "pre_u_supply_mean",
        "mean_reversion_u_mean",
        "seasonal_u_mean",
        "conditional_u_mean",
        "u_mean",
        "u_std",
        *SUPPLY_QUANTILES,
        "supply_q90_interval_width",
        "supply_q05_minus_transactions",
        "supply_q50_minus_transactions",
        "supply_q95_minus_transactions",
        "supply_q50_to_transactions_ratio",
        "historical_utilization_at_q50",
        "transactions_le_supply_q05",
        "transactions_le_supply_q50",
        "transactions_le_supply_q95",
        "nonnegative_projection_rate",
        "historical_feasibility_common_shift",
        "historical_feasibility_tail_clip_rate",
        "historical_feasibility_topup_mean",
        "historical_feasibility_topup_rate",
        "historical_departure_infeasibility_rate",
        "historical_flow_deficit_rate",
    ]


def main() -> None:
    args = parse_args()
    (
        supply_file,
        u_summary_file,
        output_csv,
        output_parquet,
        output_json,
    ) = build_paths(args.input_prefix)
    check_inputs_and_outputs(
        args.overwrite,
        supply_file,
        u_summary_file,
        [output_csv, output_parquet, output_json],
    )
    start, end = validate_args(args, supply_file)

    samples = select_samples(start, end, args.sample_size, args.seed)
    result = add_source_data(samples, supply_file, u_summary_file)
    result = add_audit_metrics(result)
    result = result.loc[:, ordered_output_columns()]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False, encoding="utf-8-sig")
    result.to_parquet(output_parquet, index=False)

    group_summary = (
        result.groupby("sample_group", sort=False)
        .agg(
            sample_count=("sample_id", "size"),
            historical_transactions_min=("D_origin_it", "min"),
            historical_transactions_mean=("D_origin_it", "mean"),
            historical_transactions_max=("D_origin_it", "max"),
            reference_supply_profile_mean=("reference_supply_profile", "mean"),
            supply_q50_mean=("q_0.500", "mean"),
            supply_q50_max=("q_0.500", "max"),
            q50_covers_transactions_rate=("transactions_le_supply_q50", "mean"),
            q95_covers_transactions_rate=("transactions_le_supply_q95", "mean"),
        )
        .reset_index()
    )
    metadata = {
        "definition_of_historical_transactions": (
            "D_origin_it: region-origin completed trips in the 15-minute decision slot"
        ),
        "sample_start": str(start),
        "sample_end": str(end),
        "sample_size_per_group": int(args.sample_size),
        "total_sample_count": int(len(result)),
        "random_seed": int(args.seed),
        "input_prefix": args.input_prefix,
        "sampling_groups": [
            "top D_origin_it decision points",
            "random D_origin_it == 0 decision points",
            "random D_origin_it > 0 excluding the top group",
        ],
        "group_summary": group_summary.to_dict(orient="records"),
        "files": {
            "supply_input": str(supply_file),
            "u_summary_input": str(u_summary_file),
            "csv": str(output_csv),
            "parquet": str(output_parquet),
        },
    }
    with output_json.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print(result.to_string(index=False))
    print("\n分组摘要：")
    print(group_summary.to_string(index=False))
    print("\n输出文件：")
    print(output_csv)
    print(output_parquet)
    print(output_json)


if __name__ == "__main__":
    main()
