# -*- coding: utf-8 -*-
"""仅用目标日之前的历史，构建D/S经验分布、基础状态和Top-5邻近区域特征。"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("vit_pretrain_config", HERE / "1 Vit预训练配置.py")
cfg = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(cfg)

KEY_LEVELS = (0.005, 0.05, 0.10, 0.50, 0.90, 0.95, 0.995)
NEIGHBOR_LEVELS = (0.10, 0.50, 0.90)


def compressed_columns(prefix: str) -> list[str]:
    bins = [f"{prefix}_qbin_{left:02d}_{left + 5:02d}_mean" for left in range(0, 100, 5)]
    keys = [f"{prefix}_q_{level:.3f}" for level in KEY_LEVELS]
    return bins + keys


def compress_quantiles(values: np.ndarray) -> np.ndarray:
    """将101维分位数压缩为20个5%区间均值和7个关键分位数。"""

    pieces = []
    for index in range(20):
        mask = (cfg.QUANTILE_LEVELS > index * 0.05) & (
            cfg.QUANTILE_LEVELS <= (index + 1) * 0.05
        )
        pieces.append(values[..., mask].mean(axis=-1))
    for level in KEY_LEVELS:
        position = int(np.argmin(np.abs(cfg.QUANTILE_LEVELS - level)))
        pieces.append(values[..., position])
    return np.stack(pieces, axis=-1).astype(np.float32)


def make_cube(
    frame: pd.DataFrame,
    value_column: str,
    dates: pd.Index,
    region_index: dict[str, int],
) -> np.ndarray:
    cube = np.zeros((len(dates), cfg.TIME_STEPS_PER_DAY, len(region_index)), dtype=np.float32)
    date_position = pd.Series(np.arange(len(dates)), index=dates)
    day = frame["t"].dt.normalize().map(date_position).to_numpy(np.int16)
    slot = ((frame["t"].dt.hour * 60 + frame["t"].dt.minute) // 15).to_numpy(np.int16)
    region = frame["region_id"].astype(str).map(region_index).to_numpy(np.int16)
    cube[day, slot, region] = frame[value_column].to_numpy(np.float32)
    return cube


def empirical_distribution(
    cube: np.ndarray,
    day_types: np.ndarray,
    target_day: int,
    slot: int,
) -> tuple[np.ndarray, int, int]:
    """优先同日期类型和相邻时段，样本不足时回退到全部过去日期。"""

    left = max(0, slot - cfg.HISTORY_SLOT_RADIUS)
    right = min(cfg.TIME_STEPS_PER_DAY, slot + cfg.HISTORY_SLOT_RADIUS + 1)
    prior = np.arange(target_day)
    selected = prior[day_types[:target_day] == day_types[target_day]]
    fallback = 0
    sample = cube[selected, left:right].reshape(-1, cube.shape[-1])
    if len(sample) < cfg.HISTORY_MIN_SAMPLES:
        sample = cube[:target_day, left:right].reshape(-1, cube.shape[-1])
        fallback = 1
    if len(sample) == 0:
        raise ValueError("目标日之前没有可用历史。")
    quantiles = np.quantile(sample, cfg.QUANTILE_LEVELS, axis=0).T.astype(np.float32)
    return quantiles, len(sample), fallback


def neighbor_mapping(region_ids: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    relation = pd.read_parquet(
        cfg.NEIGHBOR_FILE,
        columns=["region_id", "neighbor_region_id", "weight"],
    )
    relation["region_id"] = relation["region_id"].astype(str)
    relation["neighbor_region_id"] = relation["neighbor_region_id"].astype(str)
    relation = relation.sort_values(
        ["region_id", "weight", "neighbor_region_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    position = {region_id: index for index, region_id in enumerate(region_ids)}
    result = []
    for region_id in region_ids:
        rows = relation.loc[relation["region_id"] == region_id].head(cfg.TOP_K_NEIGHBORS)
        index = rows["neighbor_region_id"].map(position).fillna(-1).to_numpy(np.int16)
        weight = rows["weight"].to_numpy(np.float32)
        if len(index) < cfg.TOP_K_NEIGHBORS:
            missing = cfg.TOP_K_NEIGHBORS - len(index)
            index = np.r_[index, np.full(missing, -1, dtype=np.int16)]
            weight = np.r_[weight, np.zeros(missing, dtype=np.float32)]
        result.append((index, weight))
    return result


def build_day(
    day: pd.Timestamp,
    dates: pd.Index,
    day_types: np.ndarray,
    demand_cube: np.ndarray,
    supply_cube: np.ndarray,
    region_ids: np.ndarray,
    base: pd.DataFrame,
    neighbors: list[tuple[np.ndarray, np.ndarray]],
) -> None:
    target_day = int(dates.get_loc(day))
    demand_parts = []
    supply_parts = []
    demand_count = []
    supply_count = []
    fallback = []
    demand_neighbor = []
    supply_neighbor = []

    key_positions = [int(np.argmin(np.abs(cfg.QUANTILE_LEVELS - level))) for level in NEIGHBOR_LEVELS]
    for slot in range(cfg.TIME_STEPS_PER_DAY):
        demand_q, d_count, d_fallback = empirical_distribution(
            demand_cube, day_types, target_day, slot
        )
        supply_q, s_count, s_fallback = empirical_distribution(
            supply_cube, day_types, target_day, slot
        )
        demand_parts.append(compress_quantiles(demand_q))
        supply_parts.append(compress_quantiles(supply_q))
        demand_count.extend([d_count] * len(region_ids))
        supply_count.extend([s_count] * len(region_ids))
        fallback.extend([max(d_fallback, s_fallback)] * len(region_ids))

        d_key = demand_q[:, key_positions]
        s_key = supply_q[:, key_positions]
        d_neighbors = np.full((len(region_ids), cfg.TOP_K_NEIGHBORS, 3), np.nan, dtype=np.float32)
        s_neighbors = np.full_like(d_neighbors, np.nan)
        for origin, (indices, _) in enumerate(neighbors):
            valid = indices >= 0
            d_neighbors[origin, valid] = d_key[indices[valid]]
            s_neighbors[origin, valid] = s_key[indices[valid]]
        demand_neighbor.append(d_neighbors)
        supply_neighbor.append(s_neighbors)

    rows = pd.MultiIndex.from_product(
        [[day + pd.Timedelta(minutes=15 * slot) for slot in range(cfg.TIME_STEPS_PER_DAY)], region_ids],
        names=["t", "region_id"],
    ).to_frame(index=False)
    demand_values = np.concatenate(demand_parts, axis=0)
    supply_values = np.concatenate(supply_parts, axis=0)
    for index, column in enumerate(compressed_columns("D_hist")):
        rows[column] = demand_values[:, index]
    for index, column in enumerate(compressed_columns("S_hist")):
        rows[column] = supply_values[:, index]
    rows["D_hist_sample_count"] = np.asarray(demand_count, dtype=np.int16)
    rows["S_hist_sample_count"] = np.asarray(supply_count, dtype=np.int16)
    rows["history_fallback_level"] = np.asarray(fallback, dtype=np.int8)

    d_neighbor = np.concatenate(demand_neighbor, axis=0)
    s_neighbor = np.concatenate(supply_neighbor, axis=0)
    for rank in range(cfg.TOP_K_NEIGHBORS):
        index = np.asarray([item[0][rank] for item in neighbors], dtype=np.int16)
        weight = np.asarray([item[1][rank] for item in neighbors], dtype=np.float32)
        tiled_index = np.tile(index, cfg.TIME_STEPS_PER_DAY)
        rows[f"neighbor_{rank + 1}_region_id"] = np.where(
            tiled_index >= 0, region_ids[np.maximum(tiled_index, 0)], ""
        )
        rows[f"neighbor_{rank + 1}_weight"] = np.tile(weight, cfg.TIME_STEPS_PER_DAY)
        for level_position, level in enumerate(NEIGHBOR_LEVELS):
            rows[f"neighbor_{rank + 1}_D_q{level:.2f}"] = d_neighbor[:, rank, level_position]
            rows[f"neighbor_{rank + 1}_S_q{level:.2f}"] = s_neighbor[:, rank, level_position]

    base_day = base.loc[base["t"].dt.normalize() == day].copy()
    result = rows.merge(base_day, on=["region_id", "t"], how="left", validate="one_to_one")
    if result["dow"].isna().any():
        raise ValueError(f"{day.date()}基础时间、天气或历史收益特征不完整。")
    output = cfg.REGION_HISTORY_FEATURE_DIR / f"vit_region_history_{day:%Y-%m-%d}.parquet"
    result.to_parquet(output, index=False, compression="snappy")
    print(f"已构建：{output.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", choices=tuple(cfg.SPLITS), default=list(cfg.SPLITS))
    args = parser.parse_args()
    cfg.ensure_directories()

    mapping = pd.read_csv(cfg.REGION_MAP_FILE, dtype={"region_id": "string"}).sort_values("region_index")
    region_ids = mapping["region_id"].astype(str).to_numpy()
    region_index = {region_id: index for index, region_id in enumerate(region_ids)}
    region = pd.read_parquet(
        cfg.REGION_HISTORY_FILE,
        columns=["region_id", "t", "is_weekend_or_holiday", "D_origin_it"],
    )
    supply = pd.read_parquet(cfg.SUPPLY_HISTORY_FILE, columns=["region_id", "t", "S_bar_it"])
    base = pd.read_parquet(cfg.BASE_FEATURE_FILE)
    for frame in (region, supply, base):
        frame["region_id"] = frame["region_id"].astype(str)
        frame["t"] = pd.to_datetime(frame["t"])

    dates = pd.Index(sorted(region["t"].dt.normalize().unique()))
    day_types = (
        region.groupby(region["t"].dt.normalize(), sort=True)["is_weekend_or_holiday"]
        .first()
        .reindex(dates)
        .to_numpy(np.int8)
    )
    demand_cube = make_cube(region, "D_origin_it", dates, region_index)
    supply_cube = make_cube(supply, "S_bar_it", dates, region_index)
    neighbors = neighbor_mapping(region_ids)

    requested_days = sorted(
        {
            day
            for split in args.splits
            for day in pd.date_range(
                pd.Timestamp(cfg.SPLITS[split][0]).normalize(),
                pd.Timestamp(cfg.SPLITS[split][1]).normalize(),
                freq="D",
            )
        }
    )
    for day in requested_days:
        build_day(
            day, dates, day_types, demand_cube, supply_cube,
            region_ids, base, neighbors,
        )


if __name__ == "__main__":
    main()
