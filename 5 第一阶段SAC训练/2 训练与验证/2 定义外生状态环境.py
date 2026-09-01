# -*- coding: utf-8 -*-
"""第一阶段外生历史状态环境：动作不改变下一时段状态。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PREP_DIR = HERE.parent / "1 训练前的准备"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


cfg = load(PREP_DIR / "1 第一阶段SAC配置.py", "stage1_config")
qtool = load(PREP_DIR / "2 分位数计算工具.py", "stage1_quantiles")

SUMMARY_DIM = 27
LOCAL_STATE_FEATURE_NAMES = (
    [f"potential_demand_summary_{i:02d}" for i in range(SUMMARY_DIM)]
    + [f"tau_summary_{i:02d}" for i in range(SUMMARY_DIM)]
    + [f"pe_summary_{i:02d}" for i in range(SUMMARY_DIM)]
    + [
        "log_fixed_supply", "log_potential_demand_scenario", "log_p_ref", "pi",
        "cost_to_price_ratio", "distance_scaled", "sin_slot", "cos_slot",
        "sin_dow", "cos_dow", "is_weekend", "is_peak_morning", "is_peak_evening",
        "is_night", "temperature_scaled", "humidity_scaled", "log_rain",
        "weather_code_scaled", "wind_speed_scaled",
    ]
)


def state_feature_name(index: int) -> str:
    if index < len(LOCAL_STATE_FEATURE_NAMES):
        return LOCAL_STATE_FEATURE_NAMES[index]
    index -= len(LOCAL_STATE_FEATURE_NAMES)
    if index < cfg.VIT_STATE_FEATURE_DIM:
        return f"qwen_history_feature_{index:03d}"
    index -= cfg.VIT_STATE_FEATURE_DIM
    if index < SUMMARY_DIM:
        return f"vit_origin_summary_{index:02d}"
    return f"vit_destination_summary_{index-SUMMARY_DIM:02d}"


def require_finite(name, value, date, slot, origin):
    if not np.isfinite(np.asarray(value)).all():
        raise FloatingPointError(f"{name}出现NaN/inf：date={date}, slot={slot}, i={origin}")


def region_context(supply, demand, accepted, completed, ratio, price, subsidy):
    accepted_total = float(accepted.sum())
    denominator = max(accepted_total, 1e-6)
    return np.asarray(
        [
            np.log1p(max(supply, 0.0)), np.log1p(max(demand, 0.0)),
            np.log1p(max(accepted_total, 0.0)), np.log1p(max(float(completed.sum()), 0.0)),
            ratio, np.log1p(max(float(np.sum(accepted * price) / denominator), 0.0)),
            np.log1p(max(float(np.sum(accepted * subsidy) / denominator), 0.0)),
        ], dtype=np.float32,
    )


class OfflineSACEpisode:
    """一天96步；step只推进历史时钟，不更新需求、供给或Vit。

    `available_dates`限定本次训练或评估数据划分内允许读取的自然日。这样日末
    订单可以读取同一数据划分内下一天的冻结Vit，但训练集最后一天不会越界读取
    验证集，避免时间泄漏。
    """

    def __init__(
        self,
        episode_file: Path,
        seed: int = 42,
        available_dates: set[str] | None = None,
    ):
        with np.load(episode_file, allow_pickle=False) as loaded:
            for name in loaded.files:
                setattr(self, name, loaded[name])
        self.date = str(self.date)
        self.available_dates = (
            {str(pd.Timestamp(day).date()) for day in available_dates}
            if available_dates is not None else {self.date}
        )
        if self.date not in self.available_dates:
            raise ValueError(f"当前日期{self.date}不在允许读取的日期集合中。")
        self.region_ids = self.region_ids.astype(str)
        self.region_count = len(self.region_ids)
        self.seed = int(seed)
        self.date_ordinal = int(pd.Timestamp(self.date).toordinal())
        feature_path = cfg.VIT_STATE_FEATURE_DIR / f"vit_features_{self.date}.npz"
        vit_path = cfg.FROZEN_VIT_DIR / f"vit_quantiles_{self.date}.npz"
        with np.load(feature_path, allow_pickle=True) as data:
            if not np.array_equal(data["region_ids"].astype(str), self.region_ids):
                raise ValueError(f"{feature_path.name}区域顺序不一致。")
            self.history_features = data["features"].astype(np.float32)
            self.history_available = data["available"].astype(bool)
        with np.load(vit_path, allow_pickle=False) as data:
            if not np.array_equal(data["region_ids"].astype(str), self.region_ids):
                raise ValueError(f"{vit_path.name}区域顺序不一致。")
            self.vit_quantiles = data["quantiles"].astype(np.float32)
            self.vit_available = data["available"].astype(bool)
        self._vit_cache = {
            self.date: (self.vit_quantiles, self.vit_available)
        }
        self.reset()

    def _date_after(self, day_offset: int) -> str:
        return str((pd.Timestamp(self.date) + pd.Timedelta(days=int(day_offset))).date())

    def _vit_for_offset(self, day_offset: int):
        """读取同一数据划分内未来自然日的冻结Vit；越过划分边界时返回None。"""

        target_date = self._date_after(day_offset)
        if target_date not in self.available_dates:
            return None
        if target_date not in self._vit_cache:
            path = cfg.FROZEN_VIT_DIR / f"vit_quantiles_{target_date}.npz"
            if not path.exists():
                raise FileNotFoundError(
                    f"跨日价值需要{target_date}的冻结Vit，但文件不存在：{path}"
                )
            with np.load(path, allow_pickle=False) as data:
                if not np.array_equal(data["region_ids"].astype(str), self.region_ids):
                    raise ValueError(f"{path.name}区域顺序与{self.date}不一致。")
                quantiles = data["quantiles"].astype(np.float32)
                available = data["available"].astype(bool)
            self._vit_cache[target_date] = (quantiles, available)
        return self._vit_cache[target_date]

    def reset(self) -> None:
        self.slot = 0
        self.scenario_cache: dict[int, dict] = {}

    def groups_for_slot(self, slot=None):
        return np.flatnonzero(self.group_slot == (self.slot if slot is None else slot))

    def _scenarios(self, group_id, rows):
        if group_id not in self.scenario_cache:
            # 按(date, group_id, seed)无状态生成；抽样顺序或epoch不会改变历史场景。
            rng = np.random.default_rng(
                np.random.SeedSequence([self.seed, self.date_ordinal, int(group_id)])
            )
            self.scenario_cache[group_id] = {
                "demand_uniform": float(rng.random()),
                "tau_uniform": rng.random((rows, cfg.ACCEPTANCE_SCENARIOS)).astype(np.float32),
                "vi_index": rng.integers(0, 101, (rows, cfg.ACCEPTANCE_SCENARIOS), dtype=np.uint8),
                "vj_index": rng.integers(0, 101, (rows, cfg.ACCEPTANCE_SCENARIOS), dtype=np.uint8),
            }
        return self.scenario_cache[group_id]

    def build_group(self, group_id: int) -> dict:
        """按固定group_id重建历史状态；结果只由episode和seed决定。"""

        slot = int(self.group_slot[group_id])
        dow = pd.Timestamp(self.date).dayofweek
        left, right = int(self.group_ptr[group_id]), int(self.group_ptr[group_id + 1])
        rows = np.arange(left, right)
        origin = int(self.group_origin[group_id])
        destinations = self.od_j[rows].astype(np.int64)
        count = len(rows)
        if not self.history_available[slot, origin]:
            raise KeyError(f"512维历史编码未覆盖：date={self.date}, slot={slot}, i={origin}")
        if not self.vit_available[slot, origin]:
            raise KeyError(f"冻结Vit未覆盖当前状态：date={self.date}, slot={slot}, i={origin}")
        scenario = self._scenarios(int(group_id), count)
        potential_q = self.demand_q_potential[slot, origin].astype(np.float32)
        demand_summary = np.log1p(qtool.compress_quantiles(potential_q[None]))[0]
        demand_realized = float(qtool.sample_quantiles(
            potential_q[None], np.asarray([scenario["demand_uniform"]], dtype=np.float32)
        )[0])
        tau_q = self.tau_q[rows].astype(np.float32)
        tau_minutes = qtool.sample_quantile_scenarios(tau_q, scenario["tau_uniform"])
        tau_steps = np.maximum(np.ceil(tau_minutes / cfg.TIME_STEP_MINUTES).astype(np.int16), 1)
        # 以绝对15分钟序号计算到达日期和日内时段。日末订单不再夹到23:45，
        # 也不再把跨日目的地价值置零；只有越过当前数据划分末日才按终止处理。
        future_raw = slot + tau_steps.astype(np.int32)
        future_day_offset = future_raw // cfg.STEPS_PER_DAY
        future_slot = future_raw % cfg.STEPS_PER_DAY
        vj_scenario_q = np.zeros(
            (*future_slot.shape, len(cfg.QUANTILE_LEVELS)), dtype=np.float32
        )
        for day_offset in np.unique(future_day_offset):
            selected = future_day_offset == day_offset
            future_vit = self._vit_for_offset(int(day_offset))
            if future_vit is None:
                continue
            quantiles, available = future_vit
            row_index, scenario_index = np.nonzero(selected)
            target_slot = future_slot[row_index, scenario_index]
            target_region = destinations[row_index]
            covered = available[target_slot, target_region]
            if not covered.all():
                missing = int(np.flatnonzero(~covered)[0])
                raise KeyError(
                    "冻结Vit未覆盖跨日到达状态："
                    f"date={self._date_after(int(day_offset))}, "
                    f"slot={int(target_slot[missing])}, j={int(target_region[missing])}"
                )
            vj_scenario_q[row_index, scenario_index] = quantiles[target_slot, target_region]

        wait_raw = slot + 1
        wait_day_offset, wait_slot = divmod(wait_raw, cfg.STEPS_PER_DAY)
        wait_vit = self._vit_for_offset(wait_day_offset)
        if wait_vit is None:
            wait_q = np.zeros(len(cfg.QUANTILE_LEVELS), dtype=np.float32)
        else:
            wait_quantiles, wait_available = wait_vit
            if not wait_available[wait_slot, origin]:
                raise KeyError(
                    "冻结Vit未覆盖跨日等待状态："
                    f"date={self._date_after(wait_day_offset)}, slot={wait_slot}, i={origin}"
                )
            wait_q = wait_quantiles[wait_slot, origin].astype(np.float32)
        vi_q = self.vit_quantiles[slot, origin].astype(np.float32)
        vj_mixture = qtool.mixture_quantiles(vj_scenario_q)
        vi_compressed = qtool.compress_quantiles(vi_q[None])
        vi_summary = np.sign(vi_compressed) * np.log1p(np.abs(vi_compressed))
        vj_compressed = qtool.compress_quantiles(vj_mixture)
        vj_summary = np.sign(vj_compressed) * np.log1p(np.abs(vj_compressed))
        p_ref = self.p_ref[rows].astype(np.float32)
        pe_q = self.pe_q[rows].astype(np.float32)
        weather = self.weather[slot, origin].astype(np.float32)
        angle = 2 * np.pi * slot / cfg.STEPS_PER_DAY
        dow_angle = 2 * np.pi * dow / 7
        hour = slot / 4.0
        scalar = np.column_stack(
            [
                np.full(count, np.log1p(max(float(self.supply_capacity[slot, origin]), 0.0))),
                np.full(count, np.log1p(max(demand_realized, 0.0))),
                np.log1p(p_ref), self.pi[rows], self.cost[rows] / np.maximum(p_ref, 0.01),
                self.distance[rows] / 20.0, np.full(count, np.sin(angle)), np.full(count, np.cos(angle)),
                np.full(count, np.sin(dow_angle)), np.full(count, np.cos(dow_angle)),
                np.full(count, float(self.is_weekend)), np.full(count, float(7 <= hour < 10)),
                np.full(count, float(16 <= hour < 19)), np.full(count, float(hour < 6 or hour >= 22)),
                np.full(count, weather[0] / 40.0), np.full(count, weather[1] / 100.0),
                np.full(count, np.log1p(max(weather[2], 0.0))), np.full(count, weather[3] / 100.0),
                np.full(count, weather[4] / 50.0),
            ]
        ).astype(np.float32)
        local = np.concatenate(
            [
                np.repeat(demand_summary[None], count, axis=0),
                np.log1p(qtool.compress_quantiles(tau_q)),
                qtool.compress_quantiles(pe_q) / np.maximum(p_ref[:, None], 0.01),
                scalar,
            ], axis=1,
        )
        history = np.repeat(self.history_features[slot, origin][None], count, axis=0)
        state = np.concatenate(
            [local, history, np.repeat(vi_summary, count, axis=0), vj_summary], axis=1
        ).astype(np.float32)
        require_finite("state", state, self.date, slot, origin)
        return {
            "date": self.date, "slot": slot, "group_id": int(group_id), "rows": rows,
            "state": state, "i": np.full(count, origin, dtype=np.int64), "j": destinations,
            "p_ref": p_ref, "pe_q": pe_q, "tau_steps": tau_steps,
            "vi_index": scenario["vi_index"].astype(np.int64),
            "vj_index": scenario["vj_index"].astype(np.int64),
            "wait_q": np.repeat(wait_q[None], count, axis=0),
            "vj_scenario_q": vj_scenario_q,
            "supply": float(self.supply_capacity[slot, origin]),
            "demand_realized": demand_realized,
            "base_demand": (demand_realized * self.pi[rows]).astype(np.float32),
            "cost": self.cost[rows].astype(np.float32), "pi": self.pi[rows].astype(np.float32),
        }

    def build_groups(self) -> list[dict]:
        return [self.build_group(int(group_id)) for group_id in self.groups_for_slot(self.slot)]

    def step(self, groups: list[dict], actions: list[np.ndarray]) -> list[dict]:
        results = []
        for group, action in zip(groups, actions):
            action = np.asarray(action, dtype=np.float32)
            price, subsidy = action[:, 0], action[:, 1]
            passenger = 1.0 - qtool.cdf_at(group["pe_q"], price)
            wait_sample = np.take_along_axis(group["wait_q"], group["vi_index"], axis=1)
            vj_sample = np.take_along_axis(
                group["vj_scenario_q"], group["vj_index"][..., None], axis=2
            ).squeeze(2)
            driver_reward = (1.0 - cfg.PLATFORM_RATE) * price + subsidy - group["cost"]
            accept_value = (
                driver_reward[:, None]
                + np.power(cfg.DRIVER_GAMMA, group["tau_steps"]) * vj_sample
                - cfg.DRIVER_GAMMA * wait_sample
            )
            driver_single = (accept_value >= 0).mean(axis=1).astype(np.float32)
            dispatch = min(cfg.MAX_DISPATCH, max(int(np.floor(group["supply"])), 0))
            driver_final = 1.0 - np.power(1.0 - driver_single, dispatch)
            max_driver_reward = (
                (1.0 - cfg.PLATFORM_RATE) * price
                + cfg.SUBSIDY_MAX_RATIO * group["p_ref"]
                - group["cost"]
            )
            max_accept_value = (
                max_driver_reward[:, None]
                + np.power(cfg.DRIVER_GAMMA, group["tau_steps"]) * vj_sample
                - cfg.DRIVER_GAMMA * wait_sample
            )
            max_single = (max_accept_value >= 0).mean(axis=1).astype(np.float32)
            max_final = 1.0 - np.power(1.0 - max_single, dispatch)
            hard_infeasible = max_final < (1.0 - cfg.DRIVER_BETA)
            accepted = group["base_demand"] * passenger * driver_final
            if cfg.HARD_INFEASIBLE_ORDER_LOSS:
                accepted = np.where(hard_infeasible, 0.0, accepted)
            accepted_total = float(accepted.sum())
            ratio = min(1.0, group["supply"] / accepted_total) if accepted_total > 0 else 0.0
            completed = accepted * ratio
            demand_mask = group["base_demand"] > 0
            serviceable = demand_mask & (group["supply"] > 0)
            g_passenger_raw = np.where(demand_mask, 1.0 - cfg.PASSENGER_ALPHA - passenger, 0.0)
            g_driver_raw = np.where(serviceable, 1.0 - cfg.DRIVER_BETA - driver_final, 0.0)
            profit = (cfg.PLATFORM_RATE * price - subsidy) * completed
            context = region_context(
                group["supply"], group["demand_realized"], accepted, completed,
                ratio, price, subsidy,
            )
            results.append(
                {
                    **group, "action": action, "context": context,
                    "price": price, "subsidy": subsidy,
                    "bilateral_accepted_demand": accepted.astype(np.float32),
                    "completed": completed.astype(np.float32), "profit": profit.astype(np.float32),
                    "g_passenger_raw": g_passenger_raw.astype(np.float32),
                    "g_driver_raw": g_driver_raw.astype(np.float32),
                    "g_passenger": np.maximum(g_passenger_raw, 0).astype(np.float32),
                    "g_driver": np.maximum(g_driver_raw, 0).astype(np.float32),
                    "passenger_constraint_active": demand_mask,
                    "driver_constraint_active": serviceable,
                    "passenger_acceptance": passenger.astype(np.float32),
                    "driver_single": driver_single, "driver_final": driver_final.astype(np.float32),
                    "hard_infeasible": hard_infeasible.astype(np.int8),
                    "supply_binding": np.asarray(ratio < 1.0 - 1e-6, dtype=np.int8),
                }
            )
        # 唯一的状态转移：读取episode中的历史下一时段；动作不写回任何状态。
        self.slot += 1
        return results
