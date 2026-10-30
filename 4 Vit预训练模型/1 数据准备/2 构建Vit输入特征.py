# -*- coding: utf-8 -*-
"""构建Vit的全量输入特征。

6月11—24日的D_it和S_it直接读取新随机变量预测的101维分位数。
6月1—10日由于新随机变量文件尚未覆盖，只用目标日之前的历史数据
构造因果经验分布；6月1日没有更早数据，显式标记为不可用并填0。
"""

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


def summary_columns(prefix: str) -> list[str]:
    return (
        [f"{prefix}_percentile_{left:02d}_{left + 5:02d}_mean" for left in range(0, 100, 5)]
        + [f"{prefix}_quantile_{level:.3f}" for level in cfg.KEY_LEVELS]
    )


def compress(quantiles: np.ndarray) -> np.ndarray:
    """将101维分位数压缩为20个5%区间均值和7个关键分位数。"""

    pieces = []
    for index in range(20):
        mask = (cfg.QUANTILE_LEVELS > index * 0.05) & (
            cfg.QUANTILE_LEVELS <= (index + 1) * 0.05
        )
        pieces.append(quantiles[..., mask].mean(axis=-1))
    for level in cfg.KEY_LEVELS:
        position = int(np.argmin(np.abs(cfg.QUANTILE_LEVELS - level)))
        pieces.append(quantiles[..., position])
    return np.stack(pieces, axis=-1).astype(np.float32)


def read_day(path: Path, day: pd.Timestamp, extra: list[str] | None = None) -> pd.DataFrame:
    columns = ["region_id", "t", *cfg.RANDOM_VARIABLE_QUANTILE_COLUMNS]
    if extra:
        columns += extra
    dataset = ds.dataset(path, format="parquet")
    table = dataset.to_table(
        columns=columns,
        filter=(ds.field("t") >= day.to_pydatetime())
        & (ds.field("t") < (day + pd.Timedelta(days=1)).to_pydatetime()),
    )
    frame = table.to_pandas()
    frame["region_id"] = frame["region_id"].astype(str)
    frame["t"] = pd.to_datetime(frame["t"])
    return frame


def make_cube(
    frame: pd.DataFrame, value: str, dates: pd.Index, region_position: dict[str, int]
) -> np.ndarray:
    cube = np.zeros((len(dates), cfg.TIME_STEPS_PER_DAY, len(region_position)), dtype=np.float32)
    day_position = pd.Series(np.arange(len(dates)), index=dates)
    d = frame["t"].dt.normalize().map(day_position).to_numpy(np.int16)
    slot = ((frame["t"].dt.hour * 60 + frame["t"].dt.minute) // 15).to_numpy(np.int16)
    region = frame["region_id"].map(region_position).to_numpy(np.int16)
    cube[d, slot, region] = frame[value].to_numpy(np.float32)
    return cube


def causal_empirical_quantiles(
    cube: np.ndarray,
    day_types: np.ndarray,
    target_day: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回[96,region,101]的因果经验分布、样本数和可用标记。"""

    regions = cube.shape[-1]
    result = np.zeros((cfg.TIME_STEPS_PER_DAY, regions, len(cfg.QUANTILE_LEVELS)), np.float32)
    counts = np.zeros(cfg.TIME_STEPS_PER_DAY, np.int16)
    available = np.zeros(cfg.TIME_STEPS_PER_DAY, np.int8)
    if target_day == 0:
        return result, counts, available
    prior = np.arange(target_day)
    same_type = prior[day_types[:target_day] == day_types[target_day]]
    for slot in range(cfg.TIME_STEPS_PER_DAY):
        left = max(0, slot - cfg.HISTORY_SLOT_RADIUS)
        right = min(cfg.TIME_STEPS_PER_DAY, slot + cfg.HISTORY_SLOT_RADIUS + 1)
        sample = cube[same_type, left:right].reshape(-1, regions)
        if len(sample) < cfg.HISTORY_MIN_SAMPLES:
            sample = cube[:target_day, left:right].reshape(-1, regions)
        if len(sample):
            result[slot] = np.quantile(sample, cfg.QUANTILE_LEVELS, axis=0).T
            counts[slot] = len(sample)
            available[slot] = 1
    return result, counts, available


def neighbor_table(region_ids: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    relation = pd.read_parquet(cfg.NEIGHBOR_FILE, columns=["region_id", "neighbor_region_id", "weight"])
    relation["region_id"] = relation["region_id"].astype(str)
    relation["neighbor_region_id"] = relation["neighbor_region_id"].astype(str)
    relation = relation.sort_values(
        ["region_id", "weight", "neighbor_region_id"], ascending=[True, False, True]
    )
    position = {value: index for index, value in enumerate(region_ids)}
    result = []
    for region_id in region_ids:
        part = relation.loc[relation["region_id"] == region_id].head(cfg.TOP_K_NEIGHBORS)
        indices = part["neighbor_region_id"].map(position).fillna(-1).to_numpy(np.int16)
        weights = part["weight"].to_numpy(np.float32)
        missing = cfg.TOP_K_NEIGHBORS - len(indices)
        if missing:
            indices = np.r_[indices, np.full(missing, -1, np.int16)]
            weights = np.r_[weights, np.zeros(missing, np.float32)]
        result.append((indices, weights))
    return result


def historical_destination_prior(
    od: pd.DataFrame, day: pd.Timestamp, day_type: int, region_ids: np.ndarray
) -> dict[int, pd.DataFrame]:
    """仅用目标日前的OD，构造各时段Top-10目的地与行程时长。"""

    prior = od.loc[od["t"] < day].copy()
    if prior.empty:
        return {}
    same = prior.loc[prior["is_weekend_or_holiday"] == day_type]
    if not same.empty:
        prior = same
    expanded = []
    for offset in range(-cfg.HISTORY_SLOT_RADIUS, cfg.HISTORY_SLOT_RADIUS + 1):
        part = prior.copy()
        part["target_slot"] = part["slot_in_day"] - offset
        expanded.append(part.loc[part["target_slot"].between(0, cfg.TIME_STEPS_PER_DAY - 1)])
    data = pd.concat(expanded, ignore_index=True)
    weight = data["D_ijt_observed"].to_numpy(np.float64)
    for source, target in (
        ("tau_q10_ijt", "tau10_sum"),
        ("tau_q50_ijt", "tau50_sum"),
        ("tau_q90_ijt", "tau90_sum"),
    ):
        data[target] = data[source].fillna(data["tau_bar_ijt"]).to_numpy(np.float64) * weight
    grouped = data.groupby(["target_slot", "i", "j"], observed=True, sort=False).agg(
        flow=("D_ijt_observed", "sum"),
        tau10_sum=("tau10_sum", "sum"),
        tau50_sum=("tau50_sum", "sum"),
        tau90_sum=("tau90_sum", "sum"),
    ).reset_index()
    grouped = grouped.loc[grouped["flow"] > 0]
    for column in ("tau10", "tau50", "tau90"):
        grouped[column] = grouped[f"{column}_sum"] / grouped["flow"]
    return {int(slot): part for slot, part in grouped.groupby("target_slot", sort=False)}


def add_destinations(rows: pd.DataFrame, priors: dict[int, pd.DataFrame], region_ids: np.ndarray) -> None:
    position = {value: index for index, value in enumerate(region_ids)}
    row_region = np.tile(np.arange(len(region_ids)), cfg.TIME_STEPS_PER_DAY)
    row_slot = np.repeat(np.arange(cfg.TIME_STEPS_PER_DAY), len(region_ids))
    for rank in range(1, cfg.TOP_K_DESTINATIONS + 1):
        rows[f"destination_{rank}_region_id"] = ""
        rows[f"destination_{rank}_probability"] = np.float32(0)
        for name in ("travel_time_q10", "travel_time_q50", "travel_time_q90"):
            rows[f"destination_{rank}_{name}"] = np.float32(np.nan)
    rows["destination_history_available"] = np.int8(0)
    for slot, prior in priors.items():
        prior = prior.copy()
        prior["i"] = prior["i"].astype(str)
        prior["j"] = prior["j"].astype(str)
        prior["probability"] = prior["flow"] / prior.groupby("i", observed=True)["flow"].transform("sum")
        prior = prior.sort_values(["i", "probability", "j"], ascending=[True, False, True])
        prior["rank"] = prior.groupby("i", observed=True).cumcount() + 1
        top = prior.loc[prior["rank"] <= cfg.TOP_K_DESTINATIONS]
        mask_slot = row_slot == slot
        base_index = np.flatnonzero(mask_slot)
        rows.loc[base_index, "destination_history_available"] = np.asarray(
            pd.Series(region_ids).isin(top["i"].unique()), dtype=np.int8
        )
        for _, item in top.iterrows():
            origin = position.get(str(item["i"]))
            if origin is None:
                continue
            index = base_index[origin]
            rank = int(item["rank"])
            rows.at[index, f"destination_{rank}_region_id"] = str(item["j"])
            rows.at[index, f"destination_{rank}_probability"] = np.float32(item["probability"])
            rows.at[index, f"destination_{rank}_travel_time_q10"] = np.float32(item["tau10"])
            rows.at[index, f"destination_{rank}_travel_time_q50"] = np.float32(item["tau50"])
            rows.at[index, f"destination_{rank}_travel_time_q90"] = np.float32(item["tau90"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建Vit训练或验证输入特征。")
    parser.add_argument(
        "--split", choices=("train", "valid", "all"), default="train",
        help="默认只构建6月1—24日训练特征；valid只构建6月25—27日。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg.ensure_directories()
    split_names = ("train", "valid") if args.split == "all" else (args.split,)
    target_days = [day for name in split_names for day in cfg.days(name)]
    dit_files = [cfg.DIT_PREDICTION_FILE]
    if "valid" in split_names:
        dit_files.append(cfg.DIT_VALID_PREDICTION_FILE)
    required = [
        cfg.BASE_FEATURE_FILE, cfg.DRIVER_HISTORY_METADATA_FILE,
        cfg.REGION_HISTORY_FILE, cfg.OD_HISTORY_FILE,
        *dit_files, cfg.SIT_PREDICTION_FILE,
        cfg.SIT_HISTORICAL_REFERENCE_FILE, cfg.REGION_MAP_FILE,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少Vit输入：\n" + "\n".join(missing))
    history_metadata = json.loads(cfg.DRIVER_HISTORY_METADATA_FILE.read_text(encoding="utf-8"))
    if history_metadata.get("uses_full_month_quantile_winsorization") is not False:
        raise ValueError("司机历史收益特征仍是全月分位数截尾旧口径，请重新构建。")

    mapping = pd.read_csv(cfg.REGION_MAP_FILE, dtype={"region_id": "string"}).sort_values("region_index")
    region_ids = mapping["region_id"].astype(str).to_numpy()
    region_position = {value: index for index, value in enumerate(region_ids)}
    base = pd.read_parquet(cfg.BASE_FEATURE_FILE)
    region = pd.read_parquet(
        cfg.REGION_HISTORY_FILE,
        columns=["region_id", "t", "D_origin_it", "is_weekend_or_holiday"],
    )
    supply_history = pd.read_parquet(
        cfg.SIT_HISTORICAL_REFERENCE_FILE, columns=["region_id", "t", "S_reference_it"]
    )
    for frame in (base, region, supply_history):
        frame["region_id"] = frame["region_id"].astype(str)
        frame["t"] = pd.to_datetime(frame["t"])
    dates = pd.Index(sorted(region["t"].dt.normalize().unique()))
    day_types = (
        region.groupby(region["t"].dt.normalize())["is_weekend_or_holiday"]
        .first().reindex(dates).to_numpy(np.int8)
    )
    demand_cube = make_cube(region, "D_origin_it", dates, region_position)
    supply_cube = make_cube(supply_history, "S_reference_it", dates, region_position)
    neighbors = neighbor_table(region_ids)

    od = pd.read_parquet(
        cfg.OD_HISTORY_FILE,
        columns=[
            "i", "j", "t", "slot_in_day", "is_weekend_or_holiday", "D_ijt_observed",
            "tau_bar_ijt", "tau_q10_ijt", "tau_q50_ijt", "tau_q90_ijt",
        ],
    )
    od["i"] = od["i"].astype(str)
    od["j"] = od["j"].astype(str)
    od["t"] = pd.to_datetime(od["t"])

    for day in target_days:
        date_index = int(dates.get_loc(day))
        use_prediction = day >= pd.Timestamp(cfg.PREDICT_START).normalize()
        if use_prediction:
            dit_path = (
                cfg.DIT_VALID_PREDICTION_FILE
                if day >= pd.Timestamp(cfg.VALID_START).normalize()
                else cfg.DIT_PREDICTION_FILE
            )
            demand_frame = read_day(dit_path, day)
            supply_frame = read_day(cfg.SIT_PREDICTION_FILE, day)
            for frame, name in ((demand_frame, "Dit"), (supply_frame, "Sit")):
                if len(frame) != cfg.TIME_STEPS_PER_DAY * len(region_ids):
                    raise ValueError(f"{day.date()} {name}预测分布行数不完整：{len(frame):,}")
            demand_frame = demand_frame.sort_values(["t", "region_id"])
            supply_frame = supply_frame.sort_values(["t", "region_id"])
            demand_q = demand_frame[cfg.RANDOM_VARIABLE_QUANTILE_COLUMNS].to_numpy(np.float32).reshape(
                cfg.TIME_STEPS_PER_DAY, len(region_ids), -1
            )
            supply_q = supply_frame[cfg.RANDOM_VARIABLE_QUANTILE_COLUMNS].to_numpy(np.float32).reshape(
                cfg.TIME_STEPS_PER_DAY, len(region_ids), -1
            )
            demand_count = np.full(cfg.TIME_STEPS_PER_DAY, -1, np.int16)
            supply_count = np.full(cfg.TIME_STEPS_PER_DAY, -1, np.int16)
            demand_available = supply_available = np.ones(cfg.TIME_STEPS_PER_DAY, np.int8)
            demand_source = "Dit预测模型的101维分位数"
            supply_source = "基于U分布重构的Sit随机供给101维分位数"
        else:
            demand_q, demand_count, demand_available = causal_empirical_quantiles(
                demand_cube, day_types, date_index
            )
            supply_q, supply_count, supply_available = causal_empirical_quantiles(
                supply_cube, day_types, date_index
            )
            demand_source = "仅由目标日之前历史成交量构造的经验分布"
            supply_source = "仅由目标日之前历史供给代理构造的经验分布"

        times = [day + pd.Timedelta(minutes=15 * slot) for slot in range(cfg.TIME_STEPS_PER_DAY)]
        rows = pd.MultiIndex.from_product([times, region_ids], names=["t", "region_id"]).to_frame(index=False)
        demand_summary = compress(demand_q).reshape(-1, cfg.SUMMARY_FEATURE_DIM)
        supply_summary = compress(supply_q).reshape(-1, cfg.SUMMARY_FEATURE_DIM)
        for index, column in enumerate(summary_columns("demand")):
            rows[column] = demand_summary[:, index]
        for index, column in enumerate(summary_columns("supply")):
            rows[column] = supply_summary[:, index]
        rows["demand_distribution_source"] = demand_source
        rows["supply_distribution_source"] = supply_source
        rows["demand_history_sample_count"] = np.repeat(demand_count, len(region_ids))
        rows["supply_history_sample_count"] = np.repeat(supply_count, len(region_ids))
        rows["demand_distribution_available"] = np.repeat(demand_available, len(region_ids))
        rows["supply_distribution_available"] = np.repeat(supply_available, len(region_ids))

        key_positions = [int(np.argmin(np.abs(cfg.QUANTILE_LEVELS - level))) for level in (0.10, 0.50, 0.90)]
        for rank in range(cfg.TOP_K_NEIGHBORS):
            indices = np.asarray([item[0][rank] for item in neighbors], np.int16)
            weights = np.asarray([item[1][rank] for item in neighbors], np.float32)
            safe = np.maximum(indices, 0)
            rows[f"neighbor_{rank + 1}_region_id"] = np.tile(
                np.where(indices >= 0, region_ids[safe], ""), cfg.TIME_STEPS_PER_DAY
            )
            rows[f"neighbor_{rank + 1}_weight"] = np.tile(weights, cfg.TIME_STEPS_PER_DAY)
            for pos, level in zip(key_positions, (10, 50, 90)):
                d = demand_q[:, safe, pos]
                s = supply_q[:, safe, pos]
                d[:, indices < 0] = np.nan
                s[:, indices < 0] = np.nan
                rows[f"neighbor_{rank + 1}_demand_q{level}"] = d.reshape(-1)
                rows[f"neighbor_{rank + 1}_supply_q{level}"] = s.reshape(-1)

        priors = historical_destination_prior(od, day, int(day_types[date_index]), region_ids)
        add_destinations(rows, priors, region_ids)
        base_day = base.loc[base["t"].dt.normalize() == day]
        result = rows.merge(base_day, on=["region_id", "t"], how="left", validate="one_to_one")
        if result["day_index"].isna().any() or len(result) != cfg.TIME_STEPS_PER_DAY * len(region_ids):
            raise ValueError(f"{day.date()} Vit基础特征不完整。")
        output = cfg.INPUT_FEATURE_DIR / f"vit_input_features_{day:%Y-%m-%d}.parquet"
        result.to_parquet(output, index=False, compression="zstd")
        print(f"已构建 {output.name}：{len(result):,}行；D={demand_source}；S={supply_source}")


if __name__ == "__main__":
    main()
