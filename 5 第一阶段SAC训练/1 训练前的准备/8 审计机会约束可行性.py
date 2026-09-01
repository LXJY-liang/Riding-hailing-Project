# -*- coding: utf-8 -*-
"""审计第一阶段离线SAC中逐OD乘客/司机机会约束的结构可行性。

本脚本不修改训练数据、模型或超参数，只读取已经整理好的训练episode与
冻结Vit。输出用于回答：

1. 在最低允许价格下，哪些OD仍无法达到乘客接受率目标；
2. 在满足乘客约束的最高价格下，补贴上限取不同参考价比例时，哪些OD仍
   无法达到司机最终接受率目标；
3. 若暂时忽略乘客约束并把价格提到动作上界，司机侧是否仍不可行；
4. 每条司机约束最少需要多高的补贴/参考价比例。

司机场景严格沿用完成本轮训练时的设置：32条tau/Vit场景、固定随机种子、
日末越界价值置零。跨日逻辑补齐并重新导出冻结Vit后，应再次运行本脚本，
重新得到最终统计。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
TRAIN_DIR = HERE.parent / "2 训练与验证"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


cfg = load_module(HERE / "1 第一阶段SAC配置.py", "stage1_audit_config")
qtool = load_module(HERE / "2 分位数计算工具.py", "stage1_audit_quantiles")


def passenger_acceptance(pe_q: np.ndarray, price: np.ndarray) -> np.ndarray:
    return 1.0 - qtool.cdf_at(pe_q, price)


def final_driver_acceptance(
    price: np.ndarray,
    subsidy_ratio: float,
    p_ref: np.ndarray,
    cost: np.ndarray,
    tau_steps: np.ndarray,
    vj_sample: np.ndarray,
    wait_sample: np.ndarray,
    dispatch: int,
) -> np.ndarray:
    if dispatch <= 0:
        return np.zeros(len(price), dtype=np.float32)
    reward = (
        (1.0 - cfg.PLATFORM_RATE) * price
        + float(subsidy_ratio) * p_ref
        - cost
    )
    value = (
        reward[:, None]
        + np.power(cfg.DRIVER_GAMMA, tau_steps) * vj_sample
        - cfg.DRIVER_GAMMA * wait_sample
    )
    single = (value >= 0.0).mean(axis=1)
    return (1.0 - np.power(1.0 - single, dispatch)).astype(np.float32)


def required_subsidy_ratio(
    price: np.ndarray,
    p_ref: np.ndarray,
    cost: np.ndarray,
    tau_steps: np.ndarray,
    vj_sample: np.ndarray,
    wait_sample: np.ndarray,
    dispatch: int,
) -> np.ndarray:
    """返回离散32场景下达到司机最终接受率目标所需的最小补贴比。"""

    if dispatch <= 0:
        return np.full(len(price), np.inf, dtype=np.float32)
    target_final = 1.0 - cfg.DRIVER_BETA
    target_single = 1.0 - np.power(1.0 - target_final, 1.0 / dispatch)
    required_successes = int(np.ceil(target_single * cfg.ACCEPTANCE_SCENARIOS - 1e-12))
    required_successes = min(max(required_successes, 1), cfg.ACCEPTANCE_SCENARIOS)
    scenario_subsidy = (
        cost[:, None]
        - (1.0 - cfg.PLATFORM_RATE) * price[:, None]
        - np.power(cfg.DRIVER_GAMMA, tau_steps) * vj_sample
        + cfg.DRIVER_GAMMA * wait_sample
    )
    threshold = np.sort(scenario_subsidy, axis=1)[:, required_successes - 1]
    return (np.maximum(threshold, 0.0) / np.maximum(p_ref, 0.01)).astype(np.float32)


def scenario_values(
    episode,
    vit_quantiles: np.ndarray,
    group_id: int,
    left: int,
    right: int,
    episode_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """复现训练时固定的32条tau/Vit场景（日末越界价值为0）。"""

    rows = right - left
    slot = int(episode["group_slot"][group_id])
    destinations = episode["od_j"][left:right].astype(np.int64)
    date = str(episode["date"])
    date_ordinal = int(pd.Timestamp(date).toordinal())
    rng = np.random.default_rng(
        np.random.SeedSequence([int(episode_seed), date_ordinal, int(group_id)])
    )
    # 与环境的随机数消耗顺序一致。
    rng.random()
    tau_uniform = rng.random((rows, cfg.ACCEPTANCE_SCENARIOS)).astype(np.float32)
    vi_index = rng.integers(
        0, 101, (rows, cfg.ACCEPTANCE_SCENARIOS), dtype=np.uint8
    ).astype(np.int64)
    vj_index = rng.integers(
        0, 101, (rows, cfg.ACCEPTANCE_SCENARIOS), dtype=np.uint8
    ).astype(np.int64)

    tau_q = episode["tau_q"][left:right].astype(np.float32)
    tau_minutes = qtool.sample_quantile_scenarios(tau_q, tau_uniform)
    tau_steps = np.maximum(
        np.ceil(tau_minutes / cfg.TIME_STEP_MINUTES).astype(np.int16), 1
    )
    future_raw = slot + tau_steps.astype(np.int32)
    future_valid = future_raw < cfg.STEPS_PER_DAY
    future_slot = np.minimum(future_raw, cfg.STEPS_PER_DAY - 1)
    vj_q = vit_quantiles[future_slot, destinations[:, None]].astype(np.float32)
    vj_q[~future_valid] = 0.0
    vj_sample = np.take_along_axis(vj_q, vj_index[..., None], axis=2).squeeze(2)

    origin = int(episode["group_origin"][group_id])
    if slot + 1 < cfg.STEPS_PER_DAY:
        wait_q = vit_quantiles[slot + 1, origin].astype(np.float32)
    else:
        wait_q = np.zeros(len(cfg.QUANTILE_LEVELS), dtype=np.float32)
    wait_sample = np.repeat(wait_q[None], rows, axis=0)
    wait_sample = np.take_along_axis(wait_sample, vi_index, axis=1)
    return tau_steps, vj_sample, wait_sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=tuple(cfg.SPLITS), default="train")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = cfg.EVALUATION_DIR / "chance_constraint_structural_audit" / args.split
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"输出目录已有结果：{output_dir}；覆盖需加--overwrite。")
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_files = []
    for path in sorted(cfg.EPISODE_DIR.glob("episode_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            if str(data["split"]) == args.split:
                episode_files.append(path)
    if not episode_files:
        raise FileNotFoundError(f"没有{args.split}集episode。")

    q30_index = int(np.argmin(np.abs(cfg.QUANTILE_LEVELS - cfg.PASSENGER_ALPHA)))
    passenger_target = 1.0 - cfg.PASSENGER_ALPHA
    driver_target = 1.0 - cfg.DRIVER_BETA
    subsidy_grid = (0.50, 0.75, 1.00, 1.50, 2.00)
    passenger_bad_parts: list[pd.DataFrame] = []
    driver_bad_parts: list[pd.DataFrame] = []
    day_rows: list[dict] = []

    for ordinal, episode_path in enumerate(episode_files, start=1):
        with np.load(episode_path, allow_pickle=False) as loaded:
            episode = {name: loaded[name] for name in loaded.files}
        date = str(episode["date"])
        region_ids = episode["region_ids"].astype(str)
        vit_path = cfg.FROZEN_VIT_DIR / f"vit_quantiles_{date}.npz"
        with np.load(vit_path, allow_pickle=False) as vit:
            vit_quantiles = vit["quantiles"].astype(np.float32)

        p_ref = episode["p_ref"].astype(np.float32)
        pe_q = episode["pe_q"].astype(np.float32)
        p_min = cfg.PRICE_RATIO_MIN * p_ref
        p_max = cfg.PRICE_RATIO_MAX * p_ref
        p_passenger_cap = pe_q[:, q30_index]
        p_joint_max = np.minimum(p_max, p_passenger_cap)
        passenger_at_min = passenger_acceptance(pe_q, p_min)
        passenger_bad = passenger_at_min < passenger_target - 1e-7
        timestamps = (
            pd.Timestamp(date)
            + pd.to_timedelta(episode["od_slot"].astype(np.int32) * 15, unit="m")
        )
        base = pd.DataFrame(
            {
                "date": date,
                "t": timestamps,
                "slot": episode["od_slot"].astype(np.int16),
                "i": region_ids[episode["od_i"].astype(np.int64)],
                "j": region_ids[episode["od_j"].astype(np.int64)],
                "p_ref_q50": p_ref,
                "p_min": p_min,
                "p_max": p_max,
                "passenger_price_cap_q30": p_passenger_cap,
                "observed_price": episode["observed_price"].astype(np.float32),
                "observed_orders": episode["observed_orders"].astype(np.float32),
                "pe_q005": pe_q[:, 0],
                "pe_q010": pe_q[:, 1],
                "pe_q030": pe_q[:, q30_index],
                "pe_q050": pe_q[:, int(np.argmin(np.abs(cfg.QUANTILE_LEVELS - 0.50)))],
                "pe_q090": pe_q[:, int(np.argmin(np.abs(cfg.QUANTILE_LEVELS - 0.90)))],
                "pe_q995": pe_q[:, -1],
                "pe_history_sample_count": episode["pe_history_sample_count"],
                "pe_history_match_level": episode["pe_history_match_level"],
                "pe_history_insufficient_flag": episode["pe_history_insufficient_flag"],
                "distance_miles": episode["distance"].astype(np.float32),
                "service_cost": episode["cost"].astype(np.float32),
                "pi_ijt": episode["pi"].astype(np.float32),
                "passenger_acceptance_at_pmin": passenger_at_min,
                "passenger_shortfall_at_pmin": np.maximum(
                    passenger_target - passenger_at_min, 0.0
                ),
                "pmin_to_passenger_cap_ratio": p_min / np.maximum(p_passenger_cap, 0.01),
                "p_ref_to_observed_price_ratio": p_ref / np.maximum(
                    episode["observed_price"].astype(np.float32), 0.01
                ),
            }
        )
        if passenger_bad.any():
            passenger_bad_parts.append(base.loc[passenger_bad].copy())

        # 逐(i,t)组复现司机场景；只保存kappa=0.5仍不可行或需要更高补贴的行。
        driver_records: list[pd.DataFrame] = []
        group_ptr = episode["group_ptr"].astype(np.int64)
        for group_id in range(len(group_ptr) - 1):
            left, right = int(group_ptr[group_id]), int(group_ptr[group_id + 1])
            if left == right:
                continue
            tau_steps, vj_sample, wait_sample = scenario_values(
                episode,
                vit_quantiles,
                group_id,
                left,
                right,
                cfg.RANDOM_SEED + ordinal - 1,
            )
            supply = float(
                episode["supply_capacity"][
                    int(episode["group_slot"][group_id]),
                    int(episode["group_origin"][group_id]),
                ]
            )
            dispatch = min(cfg.MAX_DISPATCH, max(int(np.floor(supply)), 0))
            joint_price = p_joint_max[left:right]
            joint_passenger_feasible = p_passenger_cap[left:right] >= p_min[left:right] - 1e-6
            required_ratio = required_subsidy_ratio(
                joint_price,
                p_ref[left:right],
                episode["cost"][left:right].astype(np.float32),
                tau_steps,
                vj_sample,
                wait_sample,
                dispatch,
            )
            acceptance = {
                value: final_driver_acceptance(
                    joint_price,
                    value,
                    p_ref[left:right],
                    episode["cost"][left:right].astype(np.float32),
                    tau_steps,
                    vj_sample,
                    wait_sample,
                    dispatch,
                )
                for value in subsidy_grid
            }
            driver_only_at_action_max = final_driver_acceptance(
                p_max[left:right],
                cfg.SUBSIDY_MAX_RATIO,
                p_ref[left:right],
                episode["cost"][left:right].astype(np.float32),
                tau_steps,
                vj_sample,
                wait_sample,
                dispatch,
            )
            hard = (
                (~joint_passenger_feasible)
                | (acceptance[0.50] < driver_target - 1e-7)
            )
            if not hard.any():
                continue
            part = base.iloc[left:right].copy()
            part["fixed_supply"] = supply
            part["dispatch_count"] = dispatch
            part["joint_max_price"] = joint_price
            part["passenger_feasible_in_action_domain"] = joint_passenger_feasible
            part["required_subsidy_ratio_for_driver_target"] = required_ratio
            for value in subsidy_grid:
                part[f"driver_final_at_kappa_{value:.2f}"] = acceptance[value]
            part["driver_final_at_price_max_kappa_0.50"] = driver_only_at_action_max
            part["joint_infeasible_at_kappa_0.50"] = hard
            driver_records.append(part.loc[hard])

        driver_bad = (
            pd.concat(driver_records, ignore_index=True)
            if driver_records
            else pd.DataFrame()
        )
        if not driver_bad.empty:
            driver_bad_parts.append(driver_bad)
        day_rows.append(
            {
                "date": date,
                "od_rows": len(base),
                "passenger_infeasible_rows": int(passenger_bad.sum()),
                "driver_or_joint_infeasible_rows_kappa_0.50": len(driver_bad),
            }
        )
        print(
            f"[{ordinal}/{len(episode_files)}] {date}: OD={len(base):,}, "
            f"乘客不可行={int(passenger_bad.sum()):,}, "
            f"联合/司机不可行(kappa=0.5)={len(driver_bad):,}"
        )

    passenger_bad = (
        pd.concat(passenger_bad_parts, ignore_index=True)
        if passenger_bad_parts
        else pd.DataFrame()
    )
    driver_bad = (
        pd.concat(driver_bad_parts, ignore_index=True)
        if driver_bad_parts
        else pd.DataFrame()
    )
    daily = pd.DataFrame(day_rows)
    passenger_path = output_dir / "passenger_infeasible_od.csv"
    driver_path = output_dir / "driver_or_joint_infeasible_od.csv"
    daily_path = output_dir / "daily_summary.csv"
    passenger_bad.to_csv(passenger_path, index=False, encoding="utf-8-sig")
    driver_bad.to_csv(driver_path, index=False, encoding="utf-8-sig")
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")

    summary = {
        "split": args.split,
        "episode_days": len(episode_files),
        "od_rows": int(daily["od_rows"].sum()),
        "passenger_target": passenger_target,
        "driver_target": driver_target,
        "price_ratio_min": cfg.PRICE_RATIO_MIN,
        "price_ratio_max": cfg.PRICE_RATIO_MAX,
        "current_subsidy_max_ratio": cfg.SUBSIDY_MAX_RATIO,
        "passenger_infeasible_rows_at_price_min": int(len(passenger_bad)),
        "passenger_infeasible_rate": float(
            len(passenger_bad) / max(int(daily["od_rows"].sum()), 1)
        ),
        "driver_or_joint_infeasible_rows_at_kappa_0_50": int(len(driver_bad)),
        "driver_audit_semantics": "completed_main20_run_cross_day_value_zero",
        "subsidy_sensitivity": {},
    }
    if not driver_bad.empty:
        for value in subsidy_grid:
            column = f"driver_final_at_kappa_{value:.2f}"
            feasible_passenger = driver_bad["passenger_feasible_in_action_domain"].astype(bool)
            summary["subsidy_sensitivity"][f"kappa_{value:.2f}"] = {
                "driver_infeasible_rows_among_listed_passenger_feasible": int(
                    (
                        feasible_passenger
                        & (driver_bad[column] < driver_target - 1e-7)
                    ).sum()
                )
            }
        summary["driver_only_infeasible_at_absolute_price_max_kappa_0_50"] = int(
            (driver_bad["driver_final_at_price_max_kappa_0.50"] < driver_target - 1e-7).sum()
        )
        finite = driver_bad["required_subsidy_ratio_for_driver_target"].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        if len(finite):
            summary["required_subsidy_ratio_quantiles_among_listed"] = {
                str(level): float(finite.quantile(level))
                for level in (0.50, 0.75, 0.90, 0.95, 0.99, 1.00)
            }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
