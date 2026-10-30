# -*- coding: utf-8 -*-
"""使用完整 2024 年 6 月数据学习共享的 U_it 条件分布。

本脚本不会为每个 (i,t) 单独学习分布，而是：

1. 使用完整历史成交司机流重构一条确定性的参考 S_it 路径；
2. 由供给动力学残差反推 U_it 伪标签；
3. 将 801 个区域按月均成交活跃度划分为少量组；
4. 按“区域组 × 星期一至星期日 × 15 分钟时段”共享 U 分布；
5. 从参考 S 提取“区域 × 日期类型 × 15 分钟时段”的历史供给基准；
6. 学习 U 对“历史供给基准 - 当前 U 前供给”的均值恢复速度；
7. 学习组内残差波动、时间相关性和全市共同冲击比例；
8. 保存 6 月 10 日 23:45 的参考供给作为随机路径初始状态。

注意：这里学习的是基于重构 S_it 得到的 U 伪分布，不是真实可观测的司机
上线、下线或空驶迁移分布。完整 6 月数据仅用于校准仿真环境。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]

OUTPUT_DIR = HERE / "output"
PREVIOUS_DEPARTURE_FILE = OUTPUT_DIR / "M_out_i_t_minus_1.parquet"
CURRENT_ARRIVAL_FILE = OUTPUT_DIR / "M_arrive_it.parquet"
HISTORICAL_REFERENCE_SUPPLY_FILE = OUTPUT_DIR / "S_it_historical_reference.parquet"
REGION_PANEL_FILE = (
    PROJECT_ROOT / "1 数据清洗" / "3_3 区域-15min聚合（补全空值）结果.parquet"
)

MODEL_DIR = OUTPUT_DIR / "u_distribution_model"
REGION_MODEL_FILE = MODEL_DIR / "u_region_groups.parquet"
GROUP_MODEL_FILE = MODEL_DIR / "u_group_distribution.parquet"
INITIAL_SUPPLY_FILE = MODEL_DIR / "supply_at_2024-06-10_2345.parquet"
SUPPLY_PROFILE_FILE = MODEL_DIR / "historical_supply_profile.parquet"
METADATA_FILE = MODEL_DIR / "u_distribution_metadata.json"
PSEUDO_LABEL_FILE = MODEL_DIR / "u_pseudo_labels.parquet"

DATA_START = pd.Timestamp("2024-06-01 00:00:00")
DATA_END = pd.Timestamp("2024-06-30 23:45:00")
SAC_START = pd.Timestamp("2024-06-11 00:00:00")
BOUNDARY_TIME = SAC_START - pd.Timedelta(minutes=15)
TIME_STEP_MINUTES = 15
STEPS_PER_DAY = 96
EXPECTED_REGION_COUNT = 801
EXPECTED_TIME_COUNT = 30 * STEPS_PER_DAY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用完整 6 月数据重构 U 伪标签并学习共享条件分布。"
    )
    parser.add_argument("--region-groups", type=int, default=5)
    parser.add_argument("--initial-utilization", type=float, default=0.50)
    parser.add_argument("--target-utilization", type=float, default=0.75)
    parser.add_argument("--adjustment-speed", type=float, default=0.50)
    parser.add_argument(
        "--lookahead-slots",
        type=int,
        default=2,
        help="仅用于全月历史参考路径重构；2 表示使用未来 30 分钟历史需求。",
    )
    parser.add_argument(
        "--minimum-normalized-std",
        type=float,
        default=0.05,
        help="组内标准化 U 残差标准差下限，避免生成阶段退化为确定值。",
    )
    parser.add_argument(
        "--save-pseudo-labels",
        action="store_true",
        help="额外保存约 230 万行 U 伪标签，便于审计但会增加磁盘占用。",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 2 <= args.region_groups <= 20:
        raise ValueError("region-groups 必须位于 2 至 20。")
    if not 0 < args.initial_utilization <= 1:
        raise ValueError("initial-utilization 必须位于 (0,1]。")
    if not 0 < args.target_utilization <= 1:
        raise ValueError("target-utilization 必须位于 (0,1]。")
    if not 0 <= args.adjustment_speed <= 1:
        raise ValueError("adjustment-speed 必须位于 [0,1]。")
    if args.lookahead_slots < 0:
        raise ValueError("lookahead-slots 不能为负数。")
    if args.minimum_normalized_std <= 0:
        raise ValueError("minimum-normalized-std 必须为正数。")


def prepare_outputs(args: argparse.Namespace) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        HISTORICAL_REFERENCE_SUPPLY_FILE,
        REGION_MODEL_FILE,
        GROUP_MODEL_FILE,
        INITIAL_SUPPLY_FILE,
        SUPPLY_PROFILE_FILE,
        METADATA_FILE,
    ]
    if args.save_pseudo_labels:
        outputs.append(PSEUDO_LABEL_FILE)
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "U 分布模型输出已存在；如需重建请使用 --overwrite：\n"
            + "\n".join(str(path) for path in existing)
        )
    if args.overwrite:
        for path in existing:
            path.unlink()


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
    expected_t = np.repeat(times.to_numpy(), len(regions))
    expected_region = np.tile(regions, len(times))
    if not np.array_equal(frame["t"].to_numpy(), expected_t):
        raise ValueError(f"{path.name} 时间网格不完整或顺序异常。")
    if not np.array_equal(frame["region_id"].to_numpy(), expected_region):
        raise ValueError(f"{path.name} 区域网格不完整或顺序异常。")
    return frame


def load_inputs() -> tuple[
    pd.DatetimeIndex,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    required = [REGION_PANEL_FILE, PREVIOUS_DEPARTURE_FILE, CURRENT_ARRIVAL_FILE]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少输入文件：\n" + "\n".join(missing))

    times = pd.date_range(DATA_START, DATA_END, freq=f"{TIME_STEP_MINUTES}min")
    panel_regions = pd.read_parquet(REGION_PANEL_FILE, columns=["region_id"])
    regions = np.asarray(sorted(panel_regions["region_id"].astype(str).unique()), dtype=str)
    if len(times) != EXPECTED_TIME_COUNT or len(regions) != EXPECTED_REGION_COUNT:
        raise ValueError("完整 6 月时间或区域数量不符合预期。")

    panel = load_ordered_frame(
        REGION_PANEL_FILE,
        ["D_origin_it"],
        times,
        regions,
    )
    previous = load_ordered_frame(
        PREVIOUS_DEPARTURE_FILE, ["M_out_i_t_minus_1"], times, regions
    )
    arrival = load_ordered_frame(
        CURRENT_ARRIVAL_FILE, ["M_arrive_it"], times, regions
    )
    shape = (len(times), len(regions))
    demand = panel["D_origin_it"].to_numpy(np.float32).reshape(shape)
    previous_values = previous["M_out_i_t_minus_1"].to_numpy(np.float32).reshape(shape)
    arrival_values = arrival["M_arrive_it"].to_numpy(np.float32).reshape(shape)
    day_type = np.asarray(times.dayofweek, dtype=np.int8)
    for name, values in (
        ("D_origin_it", demand),
        ("M_out_i_t_minus_1", previous_values),
        ("M_arrive_it", arrival_values),
    ):
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{name} 包含负数、NaN 或 inf。")
    return times, regions, demand, previous_values, arrival_values, day_type


def allocate_initial_supply(
    demand: np.ndarray,
    initial_utilization: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    city_departures = int(np.ceil(float(demand[0].sum())))
    total_supply = int(np.ceil(city_departures / initial_utilization))
    lower = np.ceil(demand[0]).astype(np.int64)
    remaining = total_supply - int(lower.sum())
    if remaining < 0:
        raise ValueError("初始总供给小于初始成交出发量。")
    weights = demand[:4].mean(axis=0, dtype=np.float64)
    if float(weights.sum()) <= 0:
        weights = np.ones_like(weights)
    weights /= weights.sum()
    exact = remaining * weights
    extra = np.floor(exact).astype(np.int64)
    leftovers = remaining - int(extra.sum())
    if leftovers:
        order = np.argsort(-(exact - extra), kind="mergesort")
        extra[order[:leftovers]] += 1
    supply = (lower + extra).astype(np.float32)
    metadata = {
        "initial_city_departures": city_departures,
        "initial_city_supply": int(supply.sum()),
        "initial_utilization": float(initial_utilization),
        "initial_nonzero_regions": int((supply > 0).sum()),
        "initial_supply_max": float(supply.max()),
    }
    return supply, metadata


def future_max(values: np.ndarray, slots: int) -> np.ndarray:
    result = values.copy()
    for offset in range(1, slots + 1):
        result[:-offset] = np.maximum(result[:-offset], values[offset:])
    return result


def reconstruct_reference_supply(
    demand: np.ndarray,
    previous_departures: np.ndarray,
    arrivals: np.ndarray,
    initial_supply: np.ndarray,
    target_utilization: float,
    adjustment_speed: float,
    lookahead_slots: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """用完整历史需求重构确定性参考 S，并反推动力学残差 U。"""

    target_demand = future_max(demand, lookahead_slots)
    target_supply = target_demand / float(target_utilization)
    reference = np.empty_like(demand, dtype=np.float32)
    pre_u = np.empty_like(demand, dtype=np.float32)
    pseudo_u = np.zeros_like(demand, dtype=np.float32)
    projected = np.zeros_like(demand, dtype=bool)
    reference[0] = initial_supply
    pre_u[0] = initial_supply

    for time_index in range(1, demand.shape[0]):
        raw_pre = (
            reference[time_index - 1]
            - previous_departures[time_index]
            + arrivals[time_index]
        )
        if float(raw_pre.min()) < -1e-5:
            raise AssertionError(
                f"参考路径在时间索引 {time_index} 的 U 前供给为负；"
                "初始供给或历史成交可行性异常。"
            )
        pre_u[time_index] = np.maximum(raw_pre, 0.0)
        center = adjustment_speed * (
            target_supply[time_index] - pre_u[time_index]
        )
        proposed = np.rint(pre_u[time_index] + center).astype(np.float32)
        projected[time_index] = proposed < demand[time_index]
        reference[time_index] = np.maximum(proposed, demand[time_index])
        pseudo_u[time_index] = reference[time_index] - pre_u[time_index]

    return reference, pre_u, pseudo_u, projected


def assign_region_groups(activity: np.ndarray, group_count: int) -> np.ndarray:
    """按活跃度排序等频分组；相同活跃度用稳定区域顺序打破并列。"""

    rank = pd.Series(activity).rank(method="first")
    groups = pd.qcut(rank, q=group_count, labels=False).to_numpy(np.int16)
    if len(np.unique(groups)) != group_count:
        raise ValueError("区域活跃度无法形成指定数量的分组。")
    return groups


def learn_distribution(
    times: pd.DatetimeIndex,
    regions: np.ndarray,
    demand: np.ndarray,
    day_type: np.ndarray,
    reference: np.ndarray,
    pre_u: np.ndarray,
    pseudo_u: np.ndarray,
    region_groups: np.ndarray,
    minimum_std: float,
):
    """学习历史供给基准恢复速度、日内残差、AR 和共同冲击占比。"""

    time_count, region_count = demand.shape
    slot = np.asarray(times.hour * 4 + times.minute // 15, dtype=np.int16)
    # t=0 没有动力学残差，排除该时段。
    labels = pd.DataFrame(
        {
            "region_id": np.tile(regions, time_count - 1),
            "t": np.repeat(times[1:].to_numpy(), region_count),
            "region_group": np.tile(region_groups, time_count - 1),
            "day_type": np.repeat(day_type[1:], region_count),
            "slot_in_day": np.repeat(slot[1:], region_count),
            "D_origin_it": demand[1:].reshape(-1),
            "S_reference_it": reference[1:].reshape(-1),
            "S_pre_u_it": pre_u[1:].reshape(-1),
            "U_pseudo_it": pseudo_u[1:].reshape(-1),
        }
    )
    keys = ["region_group", "day_type", "slot_in_day"]

    supply_profile = (
        labels.groupby(
            ["region_id", "day_type", "slot_in_day"], observed=True
        )["S_reference_it"]
        .median()
        .rename("S_reference_profile")
        .reset_index()
    )
    labels = labels.merge(
        supply_profile,
        on=["region_id", "day_type", "slot_in_day"],
        how="left",
        validate="many_to_one",
    )
    labels["reference_supply_gap"] = (
        labels["S_reference_profile"] - labels["S_pre_u_it"]
    ).astype(np.float32)
    labels["gap_squared"] = np.square(
        labels["reference_supply_gap"].to_numpy(np.float64)
    )
    labels["gap_times_u"] = (
        labels["reference_supply_gap"].to_numpy(np.float64)
        * labels["U_pseudo_it"].to_numpy(np.float64)
    )
    global_denominator = float(labels["gap_squared"].sum())
    if global_denominator <= 0.0:
        raise ValueError("没有可用于校准历史供给基准恢复速度的有效样本。")
    global_response_speed = float(
        labels["gap_times_u"].sum() / global_denominator
    )
    global_response_speed = float(np.clip(global_response_speed, 0.05, 1.0))
    response = (
        labels.groupby(keys, observed=True)
        .agg(
            response_sample_count=("reference_supply_gap", "size"),
            response_numerator=("gap_times_u", "sum"),
            response_denominator=("gap_squared", "sum"),
        )
        .reset_index()
    )
    response["response_speed_raw"] = (
        response["response_numerator"]
        / np.maximum(response["response_denominator"], 1e-12)
    )
    # 向全局响应速度做样本量收缩，防止个别组时段由少量极端点决定。
    response_weight = (
        response["response_sample_count"]
        / (response["response_sample_count"] + 200.0)
    )
    response["response_speed"] = np.clip(
        response_weight * response["response_speed_raw"]
        + (1.0 - response_weight) * global_response_speed,
        0.05,
        1.0,
    ).astype(np.float32)
    labels = labels.merge(
        response[keys + ["response_speed"]],
        on=keys,
        how="left",
        validate="many_to_one",
    )
    labels["response_speed"] = labels["response_speed"].fillna(
        global_response_speed
    )
    labels["U_response_center"] = (
        labels["response_speed"] * labels["reference_supply_gap"]
    ).astype(np.float32)
    labels["U_residual_it"] = (
        labels["U_pseudo_it"] - labels["U_response_center"]
    ).astype(np.float32)
    labels["supply_scale"] = np.sqrt(
        np.maximum(labels["S_reference_profile"].to_numpy(np.float32), 1.0)
    )
    labels["U_normalized"] = (
        labels["U_residual_it"] / labels["supply_scale"]
    ).astype(np.float32)

    base = (
        labels.groupby(keys, observed=True)["U_normalized"]
        .agg(sample_count="size", base_mean_normalized="mean")
        .reset_index()
    )
    labels = labels.merge(base, on=keys, how="left", validate="many_to_one")
    labels["base_residual"] = (
        labels["U_normalized"] - labels["base_mean_normalized"]
    )
    offsets = (
        labels.groupby("region_id", observed=True)["base_residual"]
        .mean()
        .rename("region_offset_normalized")
    )
    labels = labels.merge(offsets, on="region_id", how="left", validate="many_to_one")
    labels["distribution_residual"] = (
        labels["U_normalized"]
        - labels["base_mean_normalized"]
        - labels["region_offset_normalized"]
    )

    residual_stats = (
        labels.groupby(keys, observed=True)["distribution_residual"]
        .agg(residual_mean="mean", residual_std="std")
        .reset_index()
    )
    quantiles = (
        labels.groupby(keys, observed=True)["distribution_residual"]
        .quantile([0.05, 0.25, 0.50, 0.75, 0.95])
        .unstack()
        .rename(
            columns={
                0.05: "residual_q05",
                0.25: "residual_q25",
                0.50: "residual_q50",
                0.75: "residual_q75",
                0.95: "residual_q95",
            }
        )
        .reset_index()
    )
    group_model = base.merge(residual_stats, on=keys, validate="one_to_one")
    group_model = group_model.merge(quantiles, on=keys, validate="one_to_one")
    group_model = group_model.merge(
        response[
            keys
            + [
                "response_sample_count",
                "response_speed_raw",
                "response_speed",
            ]
        ],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    group_model["response_speed"] = group_model["response_speed"].fillna(
        global_response_speed
    ).astype(np.float32)
    day_keys = ["region_group", "day_type"]
    group_model["daily_mean_normalized"] = (
        group_model.groupby(day_keys, observed=True)["base_mean_normalized"]
        .transform("mean")
        .astype(np.float32)
    )
    group_model["seasonal_mean_normalized"] = (
        group_model["base_mean_normalized"]
        - group_model["daily_mean_normalized"]
    ).astype(np.float32)
    group_model["residual_std"] = (
        group_model["residual_std"]
        .fillna(minimum_std)
        .clip(lower=minimum_std)
    )
    labels = labels.drop(columns=["residual_mean", "residual_std"], errors="ignore")
    labels = labels.merge(
        group_model[keys + ["residual_std"]],
        on=keys,
        how="left",
        validate="many_to_one",
    )
    labels["standardized_residual"] = (
        labels["distribution_residual"] / labels["residual_std"]
    ).astype(np.float32)

    z = labels["standardized_residual"].to_numpy(np.float32).reshape(
        time_count - 1, region_count
    )
    previous_z = z[:-1]
    current_z = z[1:]
    denominator = float(np.square(previous_z, dtype=np.float64).sum())
    rho = (
        float((previous_z.astype(np.float64) * current_z).sum() / denominator)
        if denominator > 0
        else 0.0
    )
    rho = float(np.clip(rho, -0.95, 0.95))

    total_variance = float(np.var(z, dtype=np.float64))
    city_mean_variance = float(np.var(z.mean(axis=1), dtype=np.float64))
    if total_variance <= 0:
        common_share = 0.0
    else:
        # 在等相关近似下，Var(mean)/Var = c + (1-c)/N。
        ratio = city_mean_variance / total_variance
        common_share = (region_count * ratio - 1.0) / (region_count - 1.0)
        common_share = float(np.clip(common_share, 0.0, 0.95))

    activity = demand.mean(axis=0, dtype=np.float64)
    region_model = pd.DataFrame(
        {
            "region_id": regions,
            "region_group": region_groups,
            "monthly_mean_D_origin": activity.astype(np.float32),
        }
    ).merge(offsets.reset_index(), on="region_id", validate="one_to_one")
    labels.attrs["global_response_speed"] = global_response_speed
    return region_model, group_model, supply_profile, labels, rho, common_share


def main() -> None:
    args = parse_args()
    validate_args(args)
    prepare_outputs(args)
    times, regions, demand, previous, arrivals, day_type = load_inputs()
    initial, initial_metadata = allocate_initial_supply(
        demand, args.initial_utilization
    )
    reference, pre_u, pseudo_u, projected = reconstruct_reference_supply(
        demand,
        previous,
        arrivals,
        initial,
        args.target_utilization,
        args.adjustment_speed,
        args.lookahead_slots,
    )
    pd.DataFrame(
        {
            "region_id": np.tile(regions, len(times)),
            "t": np.repeat(times.to_numpy(), len(regions)),
            "S_reference_it": reference.reshape(-1),
        }
    ).to_parquet(
        HISTORICAL_REFERENCE_SUPPLY_FILE,
        index=False,
        compression="zstd",
    )
    region_groups = assign_region_groups(
        demand.mean(axis=0, dtype=np.float64), args.region_groups
    )
    (
        region_model,
        group_model,
        supply_profile,
        labels,
        rho,
        common_share,
    ) = learn_distribution(
        times,
        regions,
        demand,
        day_type,
        reference,
        pre_u,
        pseudo_u,
        region_groups,
        args.minimum_normalized_std,
    )
    global_response_speed = float(labels.attrs["global_response_speed"])

    boundary_index = int(np.flatnonzero(times == BOUNDARY_TIME)[0])
    boundary_supply = reference[boundary_index].astype(np.float32)
    region_model["boundary_supply"] = boundary_supply
    region_model.to_parquet(REGION_MODEL_FILE, index=False, compression="zstd")
    group_model.to_parquet(GROUP_MODEL_FILE, index=False, compression="zstd")
    supply_profile.to_parquet(SUPPLY_PROFILE_FILE, index=False, compression="zstd")
    pd.DataFrame(
        {
            "region_id": regions,
            "t": np.repeat(BOUNDARY_TIME.to_datetime64(), len(regions)),
            "S_reference_it": boundary_supply,
        }
    ).to_parquet(INITIAL_SUPPLY_FILE, index=False, compression="zstd")
    if args.save_pseudo_labels:
        labels.to_parquet(PSEUDO_LABEL_FILE, index=False, compression="zstd")

    metadata = {
        "model_type": "pooled_grouped_mean_reverting_gaussian_U_distribution",
        "calibration_start": str(DATA_START),
        "calibration_end": str(DATA_END),
        "simulation_start": str(SAC_START),
        "region_count": int(len(regions)),
        "time_count": int(len(times)),
        "pseudo_label_count": int(len(labels)),
        "region_group_count": int(args.region_groups),
        "group_distribution_count": int(len(group_model)),
        "day_type_definition": "0=Monday, ..., 6=Sunday",
        "initial": initial_metadata,
        "reference_reconstruction": {
            "target_utilization": float(args.target_utilization),
            "adjustment_speed": float(args.adjustment_speed),
            "lookahead_slots": int(args.lookahead_slots),
            "historical_current_demand_floor_used": True,
            "historical_future_information_used_for_calibration": True,
        },
        "learned_parameters": {
            "temporal_ar": rho,
            "common_shock_share": common_share,
            "minimum_normalized_std": float(args.minimum_normalized_std),
            "global_response_speed": global_response_speed,
            "group_response_speed_minimum": float(group_model["response_speed"].min()),
            "group_response_speed_median": float(group_model["response_speed"].median()),
            "group_response_speed_maximum": float(group_model["response_speed"].max()),
            "response_speed_definition": (
                "two-sided historical-profile-gap OLS by "
                "region_group/day_type/slot, "
                "shrunk toward the global estimate"
            ),
            "seasonal_mean_centered_within_group_day_type": True,
            "region_offset_saved_for_audit_only": True,
            "region_offset_reused_during_generation": False,
        },
        "historical_supply_profile": {
            "definition": (
                "median reconstructed S_reference by "
                "region/day_type/15-minute slot"
            ),
            "row_count": int(len(supply_profile)),
            "minimum": float(supply_profile["S_reference_profile"].min()),
            "maximum": float(supply_profile["S_reference_profile"].max()),
            "uses_predicted_D": False,
        },
        "generation_supply_target": {
            "definition": "historical reconstructed supply profile",
            "uses_predicted_D": False,
            "mean_reversion_prevents_cumulative_supply_drift": True,
        },
        "reference_projection_rate": float(projected[1:].mean()),
        "pseudo_u_mean": float(pseudo_u[1:].mean()),
        "pseudo_u_std": float(pseudo_u[1:].std()),
        "reference_supply_min": float(reference.min()),
        "reference_supply_max": float(reference.max()),
        "pseudo_labels_saved": bool(args.save_pseudo_labels),
        "files": {
            "historical_reference_supply": str(HISTORICAL_REFERENCE_SUPPLY_FILE),
            "region_model": str(REGION_MODEL_FILE),
            "group_model": str(GROUP_MODEL_FILE),
            "initial_supply": str(INITIAL_SUPPLY_FILE),
            "historical_supply_profile": str(SUPPLY_PROFILE_FILE),
            "pseudo_labels": str(PSEUDO_LABEL_FILE) if args.save_pseudo_labels else None,
        },
    }
    METADATA_FILE.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
