# -*- coding: utf-8 -*-
"""基于已经学习的共享 U_it 分布，重新生成随机 S_it 路径和分位数。

默认从 2024-06-11 00:00 开始，使用 6 月 10 日 23:45 的历史校准参考供给
作为初始状态。正式生成阶段：

* 固定使用历史订单形成的司机流出和到达，不作缩放；
* U 依赖学习到的区域组/日期类型/时段分布和随机冲击；
* 使用重构历史 S 得到的区域/日期类型/时段供给曲线作为均值恢复基准；
* 使用从伪 U 中校准的响应速度抑制长期正漂移和供给爆炸；
* 日内 U 均值按“区域组/日期类型”中心化，避免固定偏移被逐时段累积；
* 随机项使用历史供给基准缩放，不随已经偏大的当前供给继续放大；
* 使用上一时段历史流出和当前历史到达推进外生供给场景；
* 当期历史流出只用于事后诊断，不作为供给下界，不做共同平移或截断；
* 最终保存每个 (i,t) 的 101 维 S_it 分位数和 U_it 审计摘要。

该脚本构造的是用于离线 SAC 的外生随机供给场景，不是实时司机供给预测。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]

OUTPUT_DIR = HERE / "output"
PREVIOUS_DEPARTURE_FILE = OUTPUT_DIR / "M_out_i_t_minus_1.parquet"
CURRENT_ARRIVAL_FILE = OUTPUT_DIR / "M_arrive_it.parquet"
REGION_PANEL_FILE = (
    PROJECT_ROOT / "1 数据清洗" / "3_3 区域-15min聚合（补全空值）结果.parquet"
)
MODEL_DIR = OUTPUT_DIR / "u_distribution_model"
REGION_MODEL_FILE = MODEL_DIR / "u_region_groups.parquet"
GROUP_MODEL_FILE = MODEL_DIR / "u_group_distribution.parquet"
INITIAL_SUPPLY_FILE = MODEL_DIR / "supply_at_2024-06-10_2345.parquet"
SUPPLY_PROFILE_FILE = MODEL_DIR / "historical_supply_profile.parquet"
METADATA_FILE = MODEL_DIR / "u_distribution_metadata.json"

DEFAULT_START = pd.Timestamp("2024-06-11 00:00:00")
DEFAULT_END = pd.Timestamp("2024-06-30 23:45:00")
INITIAL_STATE_TIME = DEFAULT_START - pd.Timedelta(minutes=15)
TIME_STEP_MINUTES = 15
STEPS_PER_DAY = 96
EXPECTED_REGION_COUNT = 801

QUANTILE_LEVELS = np.asarray(
    [0.005] + [level / 100 for level in range(1, 100)] + [0.995],
    dtype=np.float32,
)
QUANTILE_COLUMNS = [f"q_{level:.3f}" for level in QUANTILE_LEVELS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从学习到的 U 分布生成 6 月 11 日以后的随机 S_it 分布。"
    )
    parser.add_argument("--start", default=str(DEFAULT_START))
    parser.add_argument("--end", default=str(DEFAULT_END))
    parser.add_argument(
        "--num-scenarios",
        type=int,
        default=500,
        help="随机供给路径数量，默认 500。",
    )
    parser.add_argument("--seed", type=int, default=20240611)
    parser.add_argument(
        "--output-prefix",
        default="learned_U",
        help="输出文件前缀；仅允许字母、数字、下划线和短横线。",
    )
    parser.add_argument(
        "--save-daily-scenarios",
        action="store_true",
        help="额外按日保存完整 S/U 场景路径；默认只保存分位数和摘要。",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_and_validate_period(args: argparse.Namespace) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    if start != DEFAULT_START:
        raise ValueError(
            "当前 U 模型只保存了 6 月 10 日 23:45 的初始供给，"
            "因此 --start 必须为 2024-06-11 00:00:00。"
        )
    if end < start or end > DEFAULT_END:
        raise ValueError("--end 必须位于 2024-06-11 至 2024-06-30。")
    if start.minute % 15 or end.minute % 15:
        raise ValueError("起止时间必须对齐到 15 分钟。")
    if args.num_scenarios < 20:
        raise ValueError("num-scenarios 至少为 20。")
    if not args.output_prefix.replace("_", "").replace("-", "").isalnum():
        raise ValueError("output-prefix 只能包含字母、数字、下划线和短横线。")
    return start, end


def output_paths(prefix: str) -> tuple[Path, Path, Path, Path]:
    supply = OUTPUT_DIR / f"S_it_{prefix}_quantiles.parquet"
    u_summary = OUTPUT_DIR / f"U_it_{prefix}_summary.parquet"
    diagnostics = OUTPUT_DIR / f"S_it_{prefix}_diagnostics.json"
    scenario_dir = OUTPUT_DIR / f"daily_scenario_paths_{prefix}"
    return supply, u_summary, diagnostics, scenario_dir


def prepare_outputs(
    args: argparse.Namespace,
    supply_path: Path,
    u_path: Path,
    diagnostics_path: Path,
    scenario_dir: Path,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = [path for path in (supply_path, u_path, diagnostics_path) if path.exists()]
    scenario_files = (
        sorted(scenario_dir.glob("scenarios_*.npz")) if scenario_dir.exists() else []
    )
    if args.save_daily_scenarios:
        existing.extend(scenario_files)
    if existing and not args.overwrite:
        raise FileExistsError(
            "输出已存在；如需重建请使用 --overwrite：\n"
            + "\n".join(str(path) for path in existing[:20])
        )
    if args.overwrite:
        for path in existing:
            path.unlink()
    if args.save_daily_scenarios:
        scenario_dir.mkdir(parents=True, exist_ok=True)


def load_ordered_frame(
    path: Path,
    value_columns: list[str],
    times: pd.DatetimeIndex,
    regions: np.ndarray,
) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=["region_id", "t", *value_columns])
    frame["region_id"] = frame["region_id"].astype(str)
    frame["t"] = pd.to_datetime(frame["t"], errors="raise")
    frame = frame.loc[frame["t"].between(times[0], times[-1])].copy()
    if frame[["region_id", "t"]].duplicated().any():
        raise ValueError(f"{path.name} 存在重复的 (region_id,t)。")
    frame = frame.sort_values(["t", "region_id"], kind="mergesort").reset_index(drop=True)
    expected_rows = len(times) * len(regions)
    if len(frame) != expected_rows:
        raise ValueError(f"{path.name} 行数为 {len(frame):,}，预期 {expected_rows:,}。")
    if not np.array_equal(frame["t"].to_numpy(), np.repeat(times.to_numpy(), len(regions))):
        raise ValueError(f"{path.name} 时间网格不完整或顺序异常。")
    if not np.array_equal(
        frame["region_id"].to_numpy(), np.tile(regions, len(times))
    ):
        raise ValueError(f"{path.name} 区域网格不完整或顺序异常。")
    return frame


def load_model():
    required = [
        REGION_MODEL_FILE,
        GROUP_MODEL_FILE,
        INITIAL_SUPPLY_FILE,
        SUPPLY_PROFILE_FILE,
        METADATA_FILE,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "缺少 U 分布模型；请先运行“3 学习U分布.py”：\n"
            + "\n".join(missing)
        )
    region_model = pd.read_parquet(REGION_MODEL_FILE)
    group_model = pd.read_parquet(GROUP_MODEL_FILE)
    initial = pd.read_parquet(INITIAL_SUPPLY_FILE)
    supply_profile = pd.read_parquet(SUPPLY_PROFILE_FILE)
    metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    return region_model, group_model, initial, supply_profile, metadata


def load_inputs(
    start: pd.Timestamp,
    end: pd.Timestamp,
    region_model: pd.DataFrame,
    initial: pd.DataFrame,
):
    times = pd.date_range(start, end, freq=f"{TIME_STEP_MINUTES}min")
    region_model = region_model.copy()
    region_model["region_id"] = region_model["region_id"].astype(str)
    initial = initial.copy()
    initial["region_id"] = initial["region_id"].astype(str)
    initial["t"] = pd.to_datetime(initial["t"], errors="raise")
    regions = np.asarray(sorted(region_model["region_id"].unique()), dtype=str)
    if len(regions) != EXPECTED_REGION_COUNT:
        raise ValueError(f"U 区域模型有 {len(regions)} 个区域，预期 801。")
    region_model = region_model.set_index("region_id").loc[regions].reset_index()
    initial = initial.set_index("region_id").loc[regions].reset_index()
    if initial["t"].nunique() != 1 or initial["t"].iloc[0] != INITIAL_STATE_TIME:
        raise ValueError("初始供给文件的时间不是 2024-06-10 23:45。")

    required = [PREVIOUS_DEPARTURE_FILE, CURRENT_ARRIVAL_FILE, REGION_PANEL_FILE]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少供给递推输入：\n" + "\n".join(missing))
    previous = load_ordered_frame(
        PREVIOUS_DEPARTURE_FILE, ["M_out_i_t_minus_1"], times, regions
    )
    arrivals = load_ordered_frame(
        CURRENT_ARRIVAL_FILE, ["M_arrive_it"], times, regions
    )
    panel = load_ordered_frame(
        REGION_PANEL_FILE,
        ["D_origin_it"],
        times,
        regions,
    )
    shape = (len(times), len(regions))
    previous_values = previous["M_out_i_t_minus_1"].to_numpy(np.float32).reshape(shape)
    arrival_values = arrivals["M_arrive_it"].to_numpy(np.float32).reshape(shape)
    current_departures = panel["D_origin_it"].to_numpy(np.float32).reshape(shape)
    day_type = np.asarray(times.dayofweek, dtype=np.int8)
    groups = region_model["region_group"].to_numpy(np.int16)
    initial_supply = initial["S_reference_it"].to_numpy(np.float32)
    return (
        times,
        regions,
        previous_values,
        arrival_values,
        day_type,
        groups,
        initial_supply,
        current_departures,
    )


def build_supply_profile_array(
    times: pd.DatetimeIndex,
    regions: np.ndarray,
    day_type: np.ndarray,
    supply_profile: pd.DataFrame,
) -> np.ndarray:
    required = {"region_id", "day_type", "slot_in_day", "S_reference_profile"}
    missing = sorted(required - set(supply_profile.columns))
    if missing:
        raise ValueError(f"历史供给基准缺少字段：{missing}")
    profile = supply_profile.copy()
    profile["region_id"] = profile["region_id"].astype(str)
    lookup = profile.set_index(["region_id", "day_type", "slot_in_day"])
    values = np.empty((len(times), len(regions)), dtype=np.float32)
    slots = np.asarray(times.hour * 4 + times.minute // 15, dtype=np.int16)
    for time_index, (day, slot) in enumerate(zip(day_type, slots, strict=True)):
        keys = pd.MultiIndex.from_arrays(
            [
                regions,
                np.full(len(regions), day, dtype=np.int8),
                np.full(len(regions), slot, dtype=np.int16),
            ],
            names=["region_id", "day_type", "slot_in_day"],
        )
        selected = lookup.reindex(keys)["S_reference_profile"]
        if selected.isna().any():
            raise ValueError(
                f"历史供给基准缺少 day_type={day}, slot={slot} 的区域。"
            )
        values[time_index] = selected.to_numpy(np.float32)
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("历史供给基准包含负数、NaN 或 inf。")
    return values


def build_distribution_arrays(
    times: pd.DatetimeIndex,
    groups: np.ndarray,
    day_type: np.ndarray,
    group_model: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = {
        "region_group",
        "day_type",
        "slot_in_day",
        "seasonal_mean_normalized",
        "residual_std",
        "response_speed",
    }
    missing = sorted(required - set(group_model.columns))
    if missing:
        raise ValueError(f"U 分组模型缺少字段：{missing}")
    lookup = group_model.set_index(["region_group", "day_type", "slot_in_day"])
    mean_values = np.empty((len(times), len(groups)), dtype=np.float32)
    std_values = np.empty_like(mean_values)
    response_speed_values = np.empty_like(mean_values)
    slots = np.asarray(times.hour * 4 + times.minute // 15, dtype=np.int16)
    for time_index, (day, slot) in enumerate(zip(day_type, slots, strict=True)):
        keys = pd.MultiIndex.from_arrays(
            [
                groups,
                np.full(len(groups), day, dtype=np.int8),
                np.full(len(groups), slot, dtype=np.int16),
            ],
            names=["region_group", "day_type", "slot_in_day"],
        )
        selected = lookup.reindex(keys)
        if selected[
            [
                "seasonal_mean_normalized",
                "residual_std",
                "response_speed",
            ]
        ].isna().any().any():
            raise ValueError(f"U 分组模型缺少 day_type={day}, slot={slot} 的区域组。")
        mean_values[time_index] = selected[
            "seasonal_mean_normalized"
        ].to_numpy(np.float32)
        std_values[time_index] = selected["residual_std"].to_numpy(np.float32)
        response_speed_values[time_index] = selected["response_speed"].to_numpy(
            np.float32
        )
    return mean_values, std_values, response_speed_values


def supply_quantiles(values: np.ndarray) -> np.ndarray:
    result = np.quantile(values, QUANTILE_LEVELS, axis=0).T.astype(np.float32)
    return np.maximum.accumulate(np.maximum(result, 0.0), axis=1)


def write_supply_day(
    writer: pq.ParquetWriter | None,
    path: Path,
    day_times: list[pd.Timestamp],
    regions: np.ndarray,
    day_quantiles: list[np.ndarray],
) -> pq.ParquetWriter:
    values = np.stack(day_quantiles, axis=0).reshape(-1, len(QUANTILE_LEVELS))
    frame = pd.DataFrame(values, columns=QUANTILE_COLUMNS)
    frame.insert(0, "t", np.repeat(np.asarray(day_times, dtype="datetime64[ns]"), len(regions)))
    frame.insert(0, "region_id", np.tile(regions, len(day_times)))
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema, compression="zstd")
    writer.write_table(table)
    return writer


def write_u_day(
    writer: pq.ParquetWriter | None,
    path: Path,
    day_times: list[pd.Timestamp],
    regions: np.ndarray,
    summaries: list[dict[str, np.ndarray]],
) -> pq.ParquetWriter:
    columns = [
        "historical_departures_it",
        "reference_supply_profile",
        "response_speed",
        "pre_u_supply_mean",
        "mean_reversion_u_mean",
        "seasonal_u_mean",
        "conditional_u_mean",
        "u_mean",
        "u_std",
        "u_q05",
        "u_q50",
        "u_q95",
        "nonnegative_projection_rate",
        "historical_feasibility_common_shift",
        "historical_feasibility_tail_clip_rate",
        "historical_feasibility_topup_mean",
        "historical_feasibility_topup_rate",
        "historical_departure_infeasibility_rate",
        "historical_flow_deficit_rate",
    ]
    data = {
        column: np.stack([row[column] for row in summaries], axis=0).reshape(-1)
        for column in columns
    }
    frame = pd.DataFrame(data, dtype=np.float32)
    frame.insert(0, "t", np.repeat(np.asarray(day_times, dtype="datetime64[ns]"), len(regions)))
    frame.insert(0, "region_id", np.tile(regions, len(day_times)))
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema, compression="zstd")
    writer.write_table(table)
    return writer


def main() -> None:
    args = parse_args()
    start, end = parse_and_validate_period(args)
    supply_path, u_path, diagnostics_path, scenario_dir = output_paths(args.output_prefix)
    region_model, group_model, initial, supply_profile, metadata = load_model()
    (
        times,
        regions,
        previous,
        arrivals,
        day_type,
        groups,
        initial_supply,
        current_departures,
    ) = load_inputs(start, end, region_model, initial)
    historical_supply_profile = build_supply_profile_array(
        times, regions, day_type, supply_profile
    )
    seasonal_mean, residual_std, response_speed = build_distribution_arrays(
        times, groups, day_type, group_model
    )

    if not np.isfinite(response_speed).all() or not (
        (response_speed > 0.0) & (response_speed <= 1.0)
    ).all():
        raise ValueError("学习到的 response_speed 必须位于 (0,1]。")
    profile_scale = np.sqrt(
        np.maximum(historical_supply_profile, 1.0)
    ).astype(np.float32)
    prepare_outputs(args, supply_path, u_path, diagnostics_path, scenario_dir)

    learned = metadata["learned_parameters"]
    rho = float(learned["temporal_ar"])
    common_share = float(learned["common_shock_share"])
    scenario_count = int(args.num_scenarios)
    region_count = len(regions)
    current_supply = np.repeat(
        initial_supply[None, :], scenario_count, axis=0
    ).astype(np.float32)
    latent_shock = np.zeros_like(current_supply, dtype=np.float32)
    rng = np.random.default_rng(args.seed)

    supply_writer: pq.ParquetWriter | None = None
    u_writer: pq.ParquetWriter | None = None
    day_times: list[pd.Timestamp] = []
    day_quantiles: list[np.ndarray] = []
    day_u_summaries: list[dict[str, np.ndarray]] = []
    day_s_paths: list[np.ndarray] = []
    day_u_paths: list[np.ndarray] = []
    total_projection = 0
    total_feasibility_topup = 0
    total_feasibility_topup_amount = 0.0
    total_tail_clip = 0
    total_common_shift_region_time = 0
    total_region_times = 0
    total_departure_infeasibility = 0
    total_flow_deficit = 0
    total_states = 0
    maximum_supply = 0.0
    maximum_city_supply_mean = 0.0
    minimum_city_supply_mean = float("inf")

    try:
        for time_index, timestamp in enumerate(times):
            departure_infeasibility = (
                current_supply < previous[time_index][None, :]
            )
            raw_pre = (
                current_supply
                - previous[time_index][None, :]
                + arrivals[time_index][None, :]
            )
            flow_deficit = raw_pre < 0.0
            pre_u = np.maximum(raw_pre, 0.0).astype(np.float32)

            common = rng.standard_normal((scenario_count, 1), dtype=np.float32)
            local = rng.standard_normal((scenario_count, region_count), dtype=np.float32)
            innovation = (
                np.sqrt(common_share) * common
                + np.sqrt(1.0 - common_share) * local
            )
            latent_shock = (
                rho * latent_shock
                + np.sqrt(1.0 - rho**2) * innovation
            ).astype(np.float32)
            mean_reversion_u = response_speed[time_index][None, :] * (
                historical_supply_profile[time_index][None, :] - pre_u
            )
            seasonal_u = (
                profile_scale[time_index] * seasonal_mean[time_index]
            )[None, :]
            stochastic_u = (
                profile_scale[time_index][None, :]
                * residual_std[time_index][None, :]
                * latent_shock
            )
            proposed_u = np.rint(
                mean_reversion_u + seasonal_u + stochastic_u
            ).astype(np.float32)
            proposed_supply = pre_u + proposed_u
            nonnegative_projection = proposed_supply < 0.0
            nonnegative_supply = np.maximum(proposed_supply, 0.0)
            common_shift = np.zeros(region_count, dtype=np.float32)
            tail_topup = np.zeros_like(nonnegative_supply, dtype=np.float32)
            feasibility_topup = np.zeros_like(
                nonnegative_supply, dtype=np.float32
            )
            current_supply = (
                nonnegative_supply + feasibility_topup
            ).astype(np.float32)
            actual_u = (current_supply - pre_u).astype(np.float32)

            if not np.isfinite(current_supply).all() or (current_supply < 0).any():
                raise FloatingPointError(f"{timestamp} 的 S 场景包含无效值。")
            maximum_supply = max(maximum_supply, float(current_supply.max()))
            city_supply_mean = float(current_supply.sum(axis=1).mean())
            maximum_city_supply_mean = max(maximum_city_supply_mean, city_supply_mean)
            minimum_city_supply_mean = min(minimum_city_supply_mean, city_supply_mean)
            total_projection += int(nonnegative_projection.sum())
            total_feasibility_topup += int((feasibility_topup > 0.0).sum())
            total_feasibility_topup_amount += float(feasibility_topup.sum())
            total_tail_clip += int((tail_topup > 0.0).sum())
            total_common_shift_region_time += int((common_shift > 0.0).sum())
            total_region_times += int(region_count)
            total_tail_clip += int((tail_topup > 0.0).sum())
            total_common_shift_region_time += int((common_shift > 0.0).sum())
            total_region_times += int(region_count)
            total_departure_infeasibility += int(departure_infeasibility.sum())
            total_flow_deficit += int(flow_deficit.sum())
            total_states += int(current_supply.size)

            expected_u = (
                mean_reversion_u.mean(axis=0) + seasonal_u[0]
            ).astype(np.float32)
            day_times.append(pd.Timestamp(timestamp))
            day_quantiles.append(supply_quantiles(current_supply))
            day_u_summaries.append(
                {
                    "historical_departures_it": current_departures[time_index],
                    "reference_supply_profile": historical_supply_profile[time_index],
                    "response_speed": response_speed[time_index],
                    "pre_u_supply_mean": pre_u.mean(axis=0).astype(np.float32),
                    "mean_reversion_u_mean": mean_reversion_u.mean(axis=0).astype(np.float32),
                    "seasonal_u_mean": seasonal_u[0].astype(np.float32),
                    "conditional_u_mean": expected_u,
                    "u_mean": actual_u.mean(axis=0).astype(np.float32),
                    "u_std": actual_u.std(axis=0).astype(np.float32),
                    "u_q05": np.quantile(actual_u, 0.05, axis=0).astype(np.float32),
                    "u_q50": np.quantile(actual_u, 0.50, axis=0).astype(np.float32),
                    "u_q95": np.quantile(actual_u, 0.95, axis=0).astype(np.float32),
                    "nonnegative_projection_rate": nonnegative_projection.mean(axis=0).astype(np.float32),
                    "historical_feasibility_common_shift": common_shift,
                    "historical_feasibility_tail_clip_rate": (tail_topup > 0.0).mean(axis=0).astype(np.float32),
                    "historical_feasibility_topup_mean": feasibility_topup.mean(axis=0).astype(np.float32),
                    "historical_feasibility_topup_rate": (feasibility_topup > 0.0).mean(axis=0).astype(np.float32),
                    "historical_departure_infeasibility_rate": departure_infeasibility.mean(axis=0).astype(np.float32),
                    "historical_flow_deficit_rate": flow_deficit.mean(axis=0).astype(np.float32),
                }
            )
            if args.save_daily_scenarios:
                day_s_paths.append(current_supply.copy())
                day_u_paths.append(actual_u.copy())

            day_finished = (time_index + 1) % STEPS_PER_DAY == 0
            final_step = time_index == len(times) - 1
            if day_finished or final_step:
                supply_writer = write_supply_day(
                    supply_writer, supply_path, day_times, regions, day_quantiles
                )
                u_writer = write_u_day(
                    u_writer, u_path, day_times, regions, day_u_summaries
                )
                if args.save_daily_scenarios:
                    day = day_times[0].date()
                    np.savez_compressed(
                        scenario_dir / f"scenarios_{day}.npz",
                        date=np.asarray(str(day)),
                        region_ids=regions,
                        times=np.asarray(day_times, dtype="datetime64[ns]"),
                        scenario_ids=np.arange(scenario_count, dtype=np.int32),
                        S=np.stack(day_s_paths, axis=0).astype(np.float32),
                        U=np.stack(day_u_paths, axis=0).astype(np.float32),
                    )
                print(
                    f"已写入 {day_times[0].date()}："
                    f"{len(day_times)} 个时段，{scenario_count} 条场景。"
                )
                day_times.clear()
                day_quantiles.clear()
                day_u_summaries.clear()
                day_s_paths.clear()
                day_u_paths.clear()
    finally:
        if supply_writer is not None:
            supply_writer.close()
        if u_writer is not None:
            u_writer.close()

    if supply_writer is None or u_writer is None:
        raise RuntimeError("没有生成任何 S 或 U 输出。")
    diagnostics = {
        "supply_output": str(supply_path),
        "u_summary_output": str(u_path),
        "date_start": str(start),
        "date_end": str(end),
        "region_count": int(region_count),
        "time_count": int(len(times)),
        "num_scenarios": scenario_count,
        "quantile_count": int(len(QUANTILE_LEVELS)),
        "seed": int(args.seed),
        "temporal_ar": rho,
        "common_shock_share": common_share,
        "response_speed_minimum": float(response_speed.min()),
        "response_speed_median": float(np.median(response_speed)),
        "response_speed_maximum": float(response_speed.max()),
        "supply_target_definition": (
            "median reconstructed historical S by "
            "region/day_type/15-minute slot"
        ),
        "historical_supply_profile_city_minimum": float(
            historical_supply_profile.sum(axis=1).min()
        ),
        "historical_supply_profile_city_median": float(
            np.median(historical_supply_profile.sum(axis=1))
        ),
        "historical_supply_profile_city_maximum": float(
            historical_supply_profile.sum(axis=1).max()
        ),
        "historical_supply_profile_maximum": float(
            historical_supply_profile.max()
        ),
        "seasonal_mean_centered_within_group_day_type": True,
        "permanent_region_offset_reused_during_generation": False,
        "stochastic_scale_uses_historical_supply_profile": True,
        "artificial_supply_cap_or_abort_threshold_used": False,
        "nonnegative_projection_rate": float(total_projection / total_states),
        "historical_feasibility_topup_rate": float(
            total_feasibility_topup / total_states
        ),
        "historical_feasibility_topup_mean": float(
            total_feasibility_topup_amount / total_states
        ),
        "historical_feasibility_shift_quantile": None,
        "historical_feasibility_common_shift_region_time_rate": float(
            total_common_shift_region_time / total_region_times
        ),
        "historical_feasibility_tail_clip_rate": float(
            total_tail_clip / total_states
        ),
        "historical_departure_infeasibility_rate": float(
            total_departure_infeasibility / total_states
        ),
        "historical_flow_deficit_rate": float(total_flow_deficit / total_states),
        "maximum_supply": maximum_supply,
        "minimum_city_supply_mean": minimum_city_supply_mean,
        "maximum_city_supply_mean": maximum_city_supply_mean,
        "current_historical_departure_floor_used": False,
        "future_historical_departure_used_during_generation": False,
        "predicted_D_used_during_generation": False,
        "realized_current_departure_used_for_feasibility": False,
        "full_month_history_used_for_U_calibration": True,
        "daily_scenarios_saved": bool(args.save_daily_scenarios),
        "scenario_output_dir": str(scenario_dir) if args.save_daily_scenarios else None,
        "u_model_metadata": str(METADATA_FILE),
    }
    diagnostics_path.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
