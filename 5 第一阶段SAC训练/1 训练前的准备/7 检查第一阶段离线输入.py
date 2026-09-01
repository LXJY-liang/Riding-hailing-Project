# -*- coding: utf-8 -*-
"""检查episode、512维编码和冻结101维Vit查表是否完全对齐。"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def load_config():
    spec = importlib.util.spec_from_file_location("stage1_config", HERE / "1 第一阶段SAC配置.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


cfg = load_config()


def required_state_masks(split: str) -> dict[str, np.ndarray]:
    """构造单一split内的跨日状态需求，禁止跨数据划分边界。"""

    days = [str(day) for day in cfg.days_for_splits([split])]
    day_set = set(days)
    masks = {}
    reference_regions = None
    episodes = {}
    for day in days:
        path = cfg.EPISODE_DIR / f"episode_{day}.npz"
        if not path.exists():
            raise FileNotFoundError(f"{day}缺少episode：{path}")
        with np.load(path, allow_pickle=False) as data:
            regions = data["region_ids"].astype(str)
            episodes[day] = {name: data[name] for name in (
                "group_origin", "group_slot", "od_j", "od_slot", "tau_q"
            )}
        if reference_regions is None:
            reference_regions = regions
        elif not np.array_equal(reference_regions, regions):
            raise ValueError(f"{day}的区域顺序与同一split其他日期不一致。")
        masks[day] = np.zeros((cfg.STEPS_PER_DAY, len(regions)), dtype=bool)

    for day, episode in episodes.items():
        base_day = np.datetime64(day, "D")
        origin = episode["group_origin"].astype(np.int64)
        group_slot = episode["group_slot"].astype(np.int64)
        masks[day][group_slot, origin] = True
        waiting_raw = group_slot + 1
        for offset in np.unique(waiting_raw // cfg.STEPS_PER_DAY):
            selected = waiting_raw // cfg.STEPS_PER_DAY == offset
            target_day = str(base_day + np.timedelta64(int(offset), "D"))
            if target_day in day_set:
                masks[target_day][waiting_raw[selected] % cfg.STEPS_PER_DAY, origin[selected]] = True

        destination = episode["od_j"].astype(np.int64)
        departure = episode["od_slot"].astype(np.int64)
        tau = episode["tau_q"].astype(np.float32)
        minimum = np.maximum(np.ceil(tau.min(axis=1) / 15.0), 1).astype(np.int16)
        maximum = np.maximum(np.ceil(tau.max(axis=1) / 15.0), minimum).astype(np.int16)
        for step in range(1, int(maximum.max(initial=1)) + 1):
            selected = (minimum <= step) & (step <= maximum)
            if not selected.any():
                continue
            arrival_raw = departure[selected] + step
            selected_destination = destination[selected]
            for offset in np.unique(arrival_raw // cfg.STEPS_PER_DAY):
                within_offset = arrival_raw // cfg.STEPS_PER_DAY == offset
                target_day = str(base_day + np.timedelta64(int(offset), "D"))
                if target_day in day_set:
                    masks[target_day][
                        arrival_raw[within_offset] % cfg.STEPS_PER_DAY,
                        selected_destination[within_offset],
                    ] = True
    return masks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", choices=tuple(cfg.SPLITS), default=["train"])
    args = parser.parse_args()
    checked = 0
    required_by_day = {}
    for split in args.splits:
        for day, mask in required_state_masks(split).items():
            if day in required_by_day:
                required_by_day[day] |= mask
            else:
                required_by_day[day] = mask
    for day64 in cfg.days_for_splits(args.splits):
        day = str(day64)
        paths = {
            "episode": cfg.EPISODE_DIR / f"episode_{day}.npz",
            "encoding": cfg.VIT_STATE_FEATURE_DIR / f"vit_features_{day}.npz",
            "vit": cfg.FROZEN_VIT_DIR / f"vit_quantiles_{day}.npz",
        }
        for label, path in paths.items():
            if not path.exists():
                raise FileNotFoundError(f"{day}缺少{label}：{path}")
        with np.load(paths["episode"], allow_pickle=False) as episode:
            regions = episode["region_ids"].astype(str)
            expected_region = (cfg.STEPS_PER_DAY, len(regions))
            for key in ("demand_q_transaction", "demand_q_potential"):
                curve = episode[key].astype(np.float32)
                if curve.shape != (*expected_region, len(cfg.QUANTILE_LEVELS)):
                    raise ValueError(f"{day}的{key}形状错误：{curve.shape}")
                if not np.isfinite(curve).all() or (np.diff(curve, axis=-1) < -1e-5).any():
                    raise ValueError(f"{day}的{key}存在非有限值或分位数交叉。")
            rate = float(episode["completion_rate"])
            if not np.isclose(rate, cfg.DEMAND_COMPLETION_RATE):
                raise ValueError(f"{day}的完成率{rate}与配置不一致。")
            if episode["supply_capacity"].shape != expected_region:
                raise ValueError(f"{day}的固定供给容量形状错误。")
            if (episode["supply_capacity"] < 0).any():
                raise ValueError(f"{day}的固定供给容量出现负值。")
            required = required_by_day[day]
        with np.load(paths["encoding"], allow_pickle=True) as encoding:
            if not np.array_equal(encoding["region_ids"].astype(str), regions):
                raise ValueError(f"{day}的512维编码区域顺序不一致。")
            if encoding["features"].shape != (*expected_region, cfg.VIT_STATE_FEATURE_DIM):
                raise ValueError(f"{day}的512维编码形状错误。")
            encoding_available = encoding["available"].astype(bool)
            missing = required & ~encoding_available
            if missing.any():
                raise ValueError(
                    f"{day}的512维编码缺少{int(missing.sum()):,}个新episode必需状态；"
                    "请先增量补全状态编码。"
                )
        with np.load(paths["vit"], allow_pickle=False) as vit:
            if not np.array_equal(vit["region_ids"].astype(str), regions):
                raise ValueError(f"{day}的冻结Vit区域顺序不一致。")
            values = vit["quantiles"].astype(np.float32)
            if values.shape != (*expected_region, len(cfg.QUANTILE_LEVELS)):
                raise ValueError(f"{day}的冻结Vit形状错误：{values.shape}")
            if not np.isfinite(values).all() or (np.diff(values, axis=-1) < -1e-3).any():
                raise ValueError(f"{day}冻结Vit存在非有限值或分位数交叉。")
            vit_available = vit["available"].astype(bool)
            missing = required & ~vit_available
            if missing.any():
                raise ValueError(
                    f"{day}的冻结Vit缓存缺少{int(missing.sum()):,}个必需状态；"
                    "请在补全512维编码后重新导出冻结Vit。"
                )
        checked += 1
    print(f"第一阶段离线输入检查通过：{checked}天。")


if __name__ == "__main__":
    main()
