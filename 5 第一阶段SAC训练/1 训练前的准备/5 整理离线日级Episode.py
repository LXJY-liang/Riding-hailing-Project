# -*- coding: utf-8 -*-
"""把预测与滚动历史特征整理为外生状态离线SAC的日级episode。"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as pds
from scipy import sparse


HERE = Path(__file__).resolve().parent


def load_module(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


cfg = load_module("1 第一阶段SAC配置.py", "stage1_config")
qtool = load_module("2 分位数计算工具.py", "stage1_quantiles")


def unique_prediction_file(folder: Path, prefix: str, split: str) -> Path:
    exact = folder / f"{prefix}_{split}_predictions.parquet"
    matches = [exact] if exact.exists() else sorted(folder.glob(f"{prefix}_{split}_predictions*.parquet"))
    if len(matches) != 1:
        raise FileNotFoundError(f"{folder}中缺少唯一的{prefix}_{split}_predictions.parquet。")
    return matches[0]


def read_day(dataset, day: pd.Timestamp, columns: list[str]) -> pd.DataFrame:
    end = day + pd.Timedelta(days=1)
    table = dataset.to_table(
        columns=columns,
        filter=(pds.field("t") >= day.to_pydatetime()) & (pds.field("t") < end.to_pydatetime()),
    )
    frame = table.to_pandas()
    frame["t"] = pd.to_datetime(frame["t"])
    return frame


def quantile_cube(frame, region_index, columns, lower=0.0) -> np.ndarray:
    expected = cfg.STEPS_PER_DAY * len(region_index)
    if len(frame) != expected or frame[["region_id", "t"]].duplicated().any():
        raise ValueError(f"区域-时段预测应为{expected:,}行，实际{len(frame):,}行。")
    cube = np.zeros((cfg.STEPS_PER_DAY, len(region_index), len(columns)), dtype=np.float32)
    slots = ((frame["t"].dt.hour * 60 + frame["t"].dt.minute) // 15).to_numpy(np.int16)
    regions = frame["region_id"].astype(str).map(region_index).to_numpy(np.int32)
    if (regions < 0).any():
        raise ValueError("Dit中存在未知区域。")
    cube[slots, regions] = qtool.enforce_quantiles(
        frame[columns].to_numpy(np.float32), lower=lower
    )
    return cube


def extract_pi(od, region_index, time_index) -> np.ndarray:
    raw = np.zeros(len(od), dtype=np.float32)
    i_idx = od["i"].astype(str).map(region_index).to_numpy(np.int32)
    j_idx = od["j"].astype(str).map(region_index).to_numpy(np.int32)
    t_idx = od["t"].map(time_index).to_numpy(np.int32)
    if (i_idx < 0).any() or (j_idx < 0).any() or (t_idx < 0).any():
        raise ValueError("OD包含未知区域或时间索引。")
    for origin in np.unique(i_idx):
        mask = i_idx == origin
        matrix = sparse.load_npz(cfg.PI_PARTS_DIR / f"origin-{origin:04d}.npz")
        raw[mask] = np.asarray(matrix[t_idx[mask], j_idx[mask]]).reshape(-1)
    group_code = pd.factorize(pd.MultiIndex.from_arrays([od["t"], od["i"]]), sort=False)[0]
    totals = np.bincount(group_code, weights=raw)
    result = np.divide(raw, totals[group_code], out=np.zeros_like(raw), where=totals[group_code] > 0)
    fallback = totals[group_code] <= 0
    if fallback.any():
        sizes = np.bincount(group_code)
        result[fallback] = 1.0 / sizes[group_code[fallback]]
    return result


def group_structure(od):
    key = od[["slot", "i_index"]].to_numpy(np.int32)
    starts = np.r_[0, np.flatnonzero(np.any(key[1:] != key[:-1], axis=1)) + 1]
    return (
        np.r_[starts, len(od)].astype(np.int64),
        od["i_index"].to_numpy(np.int16)[starts],
        od["slot"].to_numpy(np.int16)[starts],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", choices=tuple(cfg.SPLITS), default=list(cfg.SPLITS))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg.ensure_directories()
    if not cfg.SUPPLY_FILE.exists():
        raise FileNotFoundError(f"请先构造滚动供给容量：{cfg.SUPPLY_FILE}")

    region_map = pd.read_csv(cfg.PI_REGION_MAP, dtype={"region_id": "string"}).sort_values("region_index")
    region_ids = region_map["region_id"].astype(str).to_numpy()
    region_index = dict(zip(region_ids, region_map["region_index"]))
    time_map = pd.read_parquet(cfg.PI_TIME_MAP, columns=["time_index", "t"])
    time_map["t"] = pd.to_datetime(time_map["t"])
    time_index = dict(zip(time_map["t"], time_map["time_index"]))

    original_files = [unique_prediction_file(cfg.DIT_PREDICTION_DIR, "dit", split) for split in args.splits]
    potential_files = [
        cfg.POTENTIAL_DEMAND_DIR / f"dit_potential_{split}_predictions.parquet"
        for split in args.splits
    ]
    missing_potential = [path for path in potential_files if not path.exists()]
    if missing_potential:
        raise FileNotFoundError(f"请先转换潜在需求，缺少：{missing_potential[0]}")
    tau_files = [unique_prediction_file(cfg.TAU_PREDICTION_DIR, "tauijt", split) for split in args.splits]

    dit_transaction_ds = pds.dataset(original_files, format="parquet")
    dit_potential_ds = pds.dataset(potential_files, format="parquet")
    tau_ds = pds.dataset(tau_files, format="parquet")
    pe_ds = pds.dataset(cfg.PE_DATASET_DIR, format="parquet")
    od_ds = pds.dataset(cfg.OD_FILE, format="parquet")
    supply_ds = pds.dataset(cfg.SUPPLY_FILE, format="parquet")
    region_ds = pds.dataset(cfg.REGION_FILE, format="parquet")

    cost = pd.read_parquet(cfg.COST_FILE, columns=["i", "j", "d_ij_miles", "c_ij_usd"])
    cost["i"] = cost["i"].astype("string")
    cost["j"] = cost["j"].astype("string")

    days = [pd.Timestamp(str(day)) for day in cfg.days_for_splits(args.splits)]
    for day in days:
        output = cfg.EPISODE_DIR / f"episode_{day:%Y-%m-%d}.npz"
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"{output}已存在；需要覆盖时加入--overwrite。")

        od = read_day(
            od_ds, day,
            ["i", "j", "t", "D_ijt_observed", "p_bar_ijt", "is_weekend_or_holiday"],
        )
        # 当前候选OD仍沿用已有tau和p^e覆盖的历史活跃OD集合；p_ref不再使用目标时段p_bar。
        od = od.loc[od["D_ijt_observed"] > 0].copy()
        od["i"] = od["i"].astype("string")
        od["j"] = od["j"].astype("string")

        tau = read_day(tau_ds, day, ["i", "j", "t", *cfg.QUANTILE_COLUMNS])
        pe = read_day(
            pe_ds, day,
            [
                "i", "j", "t", "available_flag", "insufficient_sample_flag",
                "sample_count", "max_match_level", *cfg.PE_QUANTILE_COLUMNS,
            ],
        )
        for frame in (tau, pe):
            frame["i"] = frame["i"].astype("string")
            frame["j"] = frame["j"].astype("string")
        tau = tau.rename(columns={name: f"tau_{name}" for name in cfg.QUANTILE_COLUMNS})
        pe = pe.rename(
            columns={name: f"pe_{qname}" for name, qname in zip(cfg.PE_QUANTILE_COLUMNS, cfg.QUANTILE_COLUMNS)}
        )
        active_od_count = len(od)
        od = od.merge(tau, on=["i", "j", "t"], how="inner", validate="one_to_one")
        od = od.merge(pe, on=["i", "j", "t"], how="inner", validate="one_to_one")
        od = od.merge(cost, on=["i", "j"], how="left", validate="many_to_one")
        if len(od) != active_od_count:
            raise ValueError(f"{day.date()}的tau或p^e没有覆盖全部候选OD。")
        if od[["d_ij_miles", "c_ij_usd"]].isna().any().any():
            raise ValueError(f"{day.date()}存在缺失的c_ij。")

        # p^e构造器用available_flag=0和全NaN曲线明确表示“历史价格样本不足”。
        # 不能用目标日真实价格填补，否则会泄露；也不能把NaN静默替换成任意低价。
        pe_columns = [f"pe_{name}" for name in cfg.QUANTILE_COLUMNS]
        pe_finite = np.isfinite(od[pe_columns].to_numpy(np.float32)).all(axis=1)
        pe_available = od["available_flag"].to_numpy(np.int16) == 1
        inconsistent = pe_available != pe_finite
        if inconsistent.any():
            raise ValueError(
                f"{day.date()}有{int(inconsistent.sum()):,}行p^e可用标记与数值不一致。"
            )
        excluded_pe_count = int((~pe_available).sum())
        od = od.loc[pe_available].copy()
        if od.empty:
            raise ValueError(f"{day.date()}没有可用于定价的历史价格分布OD。")

        od["i_index"] = od["i"].astype(str).map(region_index).astype(np.int16)
        od["j_index"] = od["j"].astype(str).map(region_index).astype(np.int16)
        od["slot"] = ((od["t"].dt.hour * 60 + od["t"].dt.minute) // 15).astype(np.int16)
        od["pi_ijt"] = extract_pi(od, region_index, time_index)
        od = od.sort_values(["slot", "i_index", "j_index"], kind="mergesort").reset_index(drop=True)
        pointers, group_origin, group_slot = group_structure(od)

        transaction = read_day(
            dit_transaction_ds, day, ["region_id", "t", *cfg.QUANTILE_COLUMNS]
        )
        potential = read_day(
            dit_potential_ds, day, ["region_id", "t", *cfg.QUANTILE_COLUMNS]
        )
        for frame in (transaction, potential):
            frame["region_id"] = frame["region_id"].astype("string")
        demand_q_transaction = quantile_cube(transaction, region_index, cfg.QUANTILE_COLUMNS)
        demand_q_potential = quantile_cube(potential, region_index, cfg.QUANTILE_COLUMNS)
        expected_ratio = 1.0 / cfg.DEMAND_COMPLETION_RATE
        mask = demand_q_transaction > 1e-5
        if mask.any() and not np.allclose(
            demand_q_potential[mask] / demand_q_transaction[mask], expected_ratio,
            rtol=2e-4, atol=2e-4,
        ):
            raise ValueError(f"{day.date()}潜在需求与完成率配置不一致。")

        supply = read_day(
            supply_ds, day,
            ["region_id", "t", "S_it_capacity", "fallback_level", "history_sample_count"],
        )
        supply["region_id"] = supply["region_id"].astype("string")
        supply["slot"] = ((supply["t"].dt.hour * 60 + supply["t"].dt.minute) // 15).astype(np.int16)
        supply["region"] = supply["region_id"].astype(str).map(region_index).astype(np.int16)
        shape = (cfg.STEPS_PER_DAY, len(region_ids))
        supply_capacity = np.zeros(shape, dtype=np.float32)
        supply_level = np.zeros(shape, dtype=np.int8)
        supply_count = np.zeros(shape, dtype=np.int16)
        supply_capacity[supply["slot"], supply["region"]] = supply["S_it_capacity"].to_numpy(np.float32)
        supply_level[supply["slot"], supply["region"]] = supply["fallback_level"].to_numpy(np.int8)
        supply_count[supply["slot"], supply["region"]] = supply["history_sample_count"].to_numpy(np.int16)

        region_columns = [
            "region_id", "t", "dow", "is_weekend_or_holiday",
            "is_peak_morning", "is_peak_evening", "is_night", "temperature_2m",
            "relative_humidity_2m", "rain", "weather_code", "wind_speed_10m",
        ]
        region = read_day(region_ds, day, region_columns)
        region["region_id"] = region["region_id"].astype("string")
        region["slot"] = ((region["t"].dt.hour * 60 + region["t"].dt.minute) // 15).astype(np.int16)
        region["region"] = region["region_id"].astype(str).map(region_index).astype(np.int16)
        weather_columns = ["temperature_2m", "relative_humidity_2m", "rain", "weather_code", "wind_speed_10m"]
        weather = np.zeros((*shape, len(weather_columns)), dtype=np.float32)
        weather[region["slot"], region["region"]] = region[weather_columns].fillna(0.0).to_numpy(np.float32)

        tau_q = qtool.enforce_quantiles(
            od[[f"tau_{name}" for name in cfg.QUANTILE_COLUMNS]].to_numpy(np.float32), lower=0.01
        )
        pe_q = qtool.enforce_quantiles(
            od[[f"pe_{name}" for name in cfg.QUANTILE_COLUMNS]].to_numpy(np.float32), lower=0.01
        )
        q50 = int(np.argmin(np.abs(cfg.QUANTILE_LEVELS - 0.50)))
        # p_ref使用目标日前滚动历史p^e分布的中位数，避免读取目标时段真实成交均价。
        p_ref = np.maximum(pe_q[:, q50], 0.01).astype(np.float32)

        np.savez_compressed(
            output,
            date=np.asarray(str(day.date())),
            split=np.asarray(cfg.split_for_time(day)),
            completion_rate=np.asarray(cfg.DEMAND_COMPLETION_RATE, dtype=np.float32),
            active_od_count=np.asarray(active_od_count, dtype=np.int64),
            excluded_missing_pe_count=np.asarray(excluded_pe_count, dtype=np.int64),
            region_ids=np.asarray(region_ids, dtype=np.str_),
            demand_q_transaction=demand_q_transaction.astype(np.float16),
            demand_q_potential=demand_q_potential.astype(np.float16),
            weather=weather.astype(np.float16),
            supply_capacity=supply_capacity,
            supply_fallback_level=supply_level,
            supply_history_count=supply_count,
            od_i=od["i_index"].to_numpy(np.int16),
            od_j=od["j_index"].to_numpy(np.int16),
            od_slot=od["slot"].to_numpy(np.int16),
            group_ptr=pointers,
            group_origin=group_origin,
            group_slot=group_slot,
            observed_orders=od["D_ijt_observed"].to_numpy(np.float32),
            observed_price=od["p_bar_ijt"].to_numpy(np.float32),
            p_ref=p_ref,
            pi=od["pi_ijt"].to_numpy(np.float32),
            distance=od["d_ij_miles"].to_numpy(np.float32),
            cost=od["c_ij_usd"].to_numpy(np.float32),
            tau_q=tau_q.astype(np.float16),
            pe_q=pe_q.astype(np.float16),
            pe_history_sample_count=od["sample_count"].to_numpy(np.int16),
            pe_history_match_level=od["max_match_level"].to_numpy(np.int8),
            pe_history_insufficient_flag=od["insufficient_sample_flag"].to_numpy(np.int8),
            is_weekend=np.asarray(int(od["is_weekend_or_holiday"].iloc[0]), dtype=np.int8),
        )
        print(
            f"已整理：{output.name}，可定价OD={len(od):,}，"
            f"因p^e缺失剔除={excluded_pe_count:,}/{active_od_count:,}"
        )


if __name__ == "__main__":
    main()
