# -*- coding: utf-8 -*-
"""用目标日前的历史成交出发量滚动构造固定供给容量代理。

四层回退始终只使用 d'<d：
1. 同日期类型、完全相同时段；
2. 同日期类型、目标时段前后各1格；
3. 所有日期类型、完全相同时段；
4. 所有日期类型、目标时段前后各2格。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent


def load_config():
    path = HERE / "1 第一阶段SAC配置.py"
    spec = importlib.util.spec_from_file_location("stage1_config", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


cfg = load_config()


def candidate_samples(
    cube: np.ndarray,
    prior_days: np.ndarray,
    same_type_days: np.ndarray,
    slot: int,
) -> list[np.ndarray]:
    """返回四层[N,R]样本矩阵；R为区域数。"""

    radius1 = np.arange(max(0, slot - 1), min(cfg.STEPS_PER_DAY, slot + 2))
    radius2 = np.arange(max(0, slot - 2), min(cfg.STEPS_PER_DAY, slot + 3))
    return [
        cube[same_type_days, slot, :].reshape(-1, cube.shape[2]),
        cube[np.ix_(same_type_days, radius1, np.arange(cube.shape[2]))].reshape(-1, cube.shape[2]),
        cube[prior_days, slot, :].reshape(-1, cube.shape[2]),
        cube[np.ix_(prior_days, radius2, np.arange(cube.shape[2]))].reshape(-1, cube.shape[2]),
    ]


def main() -> None:
    if not 0.0 < cfg.SUPPLY_UTILIZATION <= 1.0:
        raise ValueError("SUPPLY_UTILIZATION必须位于(0,1]。")
    if not 0.0 < cfg.SUPPLY_QUANTILE < 1.0:
        raise ValueError("SUPPLY_QUANTILE必须位于(0,1)。")
    cfg.ensure_directories()

    columns = [
        "region_id", "t", "slot_in_day", "is_weekend_or_holiday", "D_origin_it"
    ]
    panel = pd.read_parquet(cfg.REGION_FILE, columns=columns)
    panel["region_id"] = panel["region_id"].astype("string")
    panel["t"] = pd.to_datetime(panel["t"], errors="raise")
    panel["date"] = panel["t"].dt.normalize()
    if panel[["region_id", "t"]].duplicated().any():
        raise ValueError("区域历史面板存在重复的(region_id,t)。")

    regions = np.asarray(sorted(panel["region_id"].astype(str).unique()), dtype=str)
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    region_index = {value: index for index, value in enumerate(regions)}
    date_index = {value: index for index, value in enumerate(dates)}
    expected = len(dates) * cfg.STEPS_PER_DAY * len(regions)
    if len(panel) != expected:
        raise ValueError(
            f"区域历史面板必须是日期×96×区域的完整网格：期望{expected:,}行，实际{len(panel):,}行。"
        )

    cube = np.zeros((len(dates), cfg.STEPS_PER_DAY, len(regions)), dtype=np.float32)
    d_idx = panel["date"].map(date_index).to_numpy(np.int16)
    s_idx = panel["slot_in_day"].to_numpy(np.int16)
    r_idx = panel["region_id"].astype(str).map(region_index).to_numpy(np.int16)
    values = panel["D_origin_it"].fillna(0.0).to_numpy(np.float32)
    if (values < 0).any():
        raise ValueError("历史成交出发量不能为负。")
    cube[d_idx, s_idx, r_idx] = values
    day_type = (
        panel.groupby("date", sort=True)["is_weekend_or_holiday"].first()
        .reindex(dates).to_numpy(np.int8)
    )

    first_target = min(np.datetime64(start, "D") for start, _ in cfg.SPLITS.values())
    last_target = max(np.datetime64(end, "D") for _, end in cfg.SPLITS.values())
    target_positions = [
        index for index, date in enumerate(dates)
        if first_target <= np.datetime64(date.date()) <= last_target
    ]
    if not target_positions:
        raise ValueError("历史区域面板没有覆盖SAC目标日期。")

    pieces: list[pd.DataFrame] = []
    audit_rows: list[dict] = []
    for day in target_positions:
        prior = np.arange(day, dtype=np.int16)
        if len(prior) == 0:
            raise ValueError(f"{dates[day].date()}之前没有历史日期，无法构造滚动供给。")
        same_type = prior[day_type[:day] == day_type[day]]
        day_capacity = np.zeros((cfg.STEPS_PER_DAY, len(regions)), dtype=np.float32)
        day_upper = np.zeros_like(day_capacity)
        day_level = np.zeros_like(day_capacity, dtype=np.int8)
        day_count = np.zeros_like(day_capacity, dtype=np.int16)
        day_low_confidence = np.zeros_like(day_capacity, dtype=bool)

        for slot in range(cfg.STEPS_PER_DAY):
            layers = candidate_samples(cube, prior, same_type, slot)
            unresolved = np.ones(len(regions), dtype=bool)
            for level, samples in enumerate(layers, start=1):
                finite = np.isfinite(samples)
                counts = finite.sum(axis=0)
                eligible = unresolved & (counts >= cfg.SUPPLY_MIN_SAMPLES)
                if eligible.any():
                    safe = np.where(finite[:, eligible], samples[:, eligible], np.nan)
                    upper = np.nanquantile(safe, cfg.SUPPLY_QUANTILE, axis=0)
                    day_upper[slot, eligible] = upper.astype(np.float32)
                    day_count[slot, eligible] = counts[eligible].astype(np.int16)
                    day_level[slot, eligible] = level
                    unresolved[eligible] = False
            # 第四层仍不足时，使用其全部历史样本并明确标记低置信度。
            if unresolved.any():
                samples = layers[-1]
                finite = np.isfinite(samples)
                counts = finite.sum(axis=0)
                if (counts[unresolved] == 0).any():
                    raise ValueError("第四层仍存在完全无历史样本的区域。")
                safe = np.where(finite[:, unresolved], samples[:, unresolved], np.nan)
                upper = np.nanquantile(safe, cfg.SUPPLY_QUANTILE, axis=0)
                day_upper[slot, unresolved] = upper.astype(np.float32)
                day_count[slot, unresolved] = counts[unresolved].astype(np.int16)
                day_level[slot, unresolved] = 4
                day_low_confidence[slot, unresolved] = True
            day_capacity[slot] = np.ceil(
                day_upper[slot] / cfg.SUPPLY_UTILIZATION
            ).astype(np.float32)

        timestamp = dates[day] + pd.to_timedelta(np.arange(cfg.STEPS_PER_DAY) * 15, unit="min")
        pieces.append(
            pd.DataFrame(
                {
                    "region_id": np.tile(regions, cfg.STEPS_PER_DAY),
                    "t": np.repeat(timestamp.to_numpy(), len(regions)),
                    "date": np.repeat(dates[day].to_datetime64(), cfg.STEPS_PER_DAY * len(regions)),
                    "slot_in_day": np.repeat(np.arange(cfg.STEPS_PER_DAY), len(regions)),
                    "is_weekend_or_holiday": int(day_type[day]),
                    "historical_upper_completed": day_upper.reshape(-1),
                    "S_it_capacity": day_capacity.reshape(-1),
                    "fallback_level": day_level.reshape(-1),
                    "history_sample_count": day_count.reshape(-1),
                    "history_unique_days": len(prior),
                    "history_end_date": dates[day - 1],
                    "low_confidence": day_low_confidence.reshape(-1),
                    "supply_quantile": cfg.SUPPLY_QUANTILE,
                    "supply_utilization": cfg.SUPPLY_UTILIZATION,
                }
            )
        )
        unique, counts = np.unique(day_level, return_counts=True)
        audit_rows.append(
            {
                "date": str(dates[day].date()),
                "history_end_date": str(dates[day - 1].date()),
                "history_days": len(prior),
                "capacity_mean": float(day_capacity.mean()),
                "capacity_p95": float(np.quantile(day_capacity, 0.95)),
                "zero_capacity_rate": float((day_capacity == 0).mean()),
                "low_confidence_rate": float(day_low_confidence.mean()),
                **{f"level_{int(k)}_rows": int(v) for k, v in zip(unique, counts)},
            }
        )
        print(
            f"{dates[day].date()}：历史截至{dates[day-1].date()}，"
            f"平均容量={day_capacity.mean():.3f}"
        )

    result = pd.concat(pieces, ignore_index=True)
    result.to_parquet(cfg.SUPPLY_FILE, index=False, compression="snappy")
    pd.DataFrame(audit_rows).to_csv(cfg.SUPPLY_AUDIT_FILE, index=False)
    print(f"滚动供给容量：{cfg.SUPPLY_FILE}")
    print(f"构造审计：{cfg.SUPPLY_AUDIT_FILE}")


if __name__ == "__main__":
    main()
