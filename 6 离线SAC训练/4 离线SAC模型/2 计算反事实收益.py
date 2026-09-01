# -*- coding: utf-8 -*-
"""任务 2：在冻结历史状态上计算接受率、成交量、约束和平台反事实收益。

输入是同一个 (i,t) 下全部有订单目的地 j 及其动作。计算器共同重算乘客接受概率、
司机接受概率、供给约束前需求、区域供给比例缩放、成交量和平台收益。它只返回当前
时段结果，绝不改变下一时段的 S、V 或其他历史状态。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


def load_config():
    path = Path(__file__).resolve().parents[1] / "1 数据准备" / "1 离线SAC统一配置.py"
    spec = importlib.util.spec_from_file_location("offline_sac_config", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


cfg = load_config()

SubsidyPolicy = Literal["actor", "force_zero", "minimum", "fixed"]


class CounterfactualReward:
    """用历史代理分布评估一组 OD 联合动作。"""

    PE_LEVELS = np.asarray([0.10, 0.25, 0.50, 0.75, 0.90], dtype=np.float32)
    FIVE_LEVELS = PE_LEVELS

    def __init__(self, num_scenarios: int, vit_file: Path | None = None):
        self.num_scenarios = int(num_scenarios)
        if self.num_scenarios < 8:
            raise ValueError("反事实场景数至少为 8。")
        self.vit_file = cfg.VIT_OUTPUT if vit_file is None else Path(vit_file)
        self._load_vit()

    def _load_vit(self) -> None:
        """把 V_proxy 放入稠密的 [time,region,quantile] 查询数组。"""

        columns = ["region_index", "t", "V_proxy_available"] + cfg.VIT_QUANTILE_COLUMNS
        vit = pd.read_parquet(self.vit_file, columns=columns)
        vit = vit.loc[vit["V_proxy_available"] == 1].copy()
        vit["t"] = pd.to_datetime(vit["t"])
        self.v_times = pd.date_range(vit["t"].min(), vit["t"].max(), freq=f"{cfg.TIME_STEP_MINUTES}min")
        self.time_to_index = {timestamp: index for index, timestamp in enumerate(self.v_times)}
        region_count = int(vit["region_index"].max()) + 1
        self.v_values = np.full(
            (len(self.v_times), region_count, len(cfg.VIT_QUANTILE_COLUMNS)),
            np.nan,
            dtype=np.float32,
        )
        time_index = vit["t"].map(self.time_to_index).to_numpy(dtype=np.int64)
        region_index = vit["region_index"].to_numpy(dtype=np.int64)
        self.v_values[time_index, region_index] = vit[cfg.VIT_QUANTILE_COLUMNS].to_numpy(
            dtype=np.float32
        )

    def _value_quantiles(self, region_index: int, timestamp: pd.Timestamp) -> np.ndarray:
        index = self.time_to_index.get(pd.Timestamp(timestamp))
        if index is None or region_index >= self.v_values.shape[1]:
            raise KeyError(f"缺少 V_proxy：region_index={region_index}, t={timestamp}")
        values = self.v_values[index, int(region_index)]
        if np.isnan(values).any():
            raise KeyError(f"V_proxy 不完整：region_index={region_index}, t={timestamp}")
        return values

    def uniforms(self, state_id: int, salt: int) -> np.ndarray:
        """生成确定性分层分位数场景，使同一状态的候选动作使用共同随机数。"""

        base = (np.arange(self.num_scenarios, dtype=np.float64) + 0.5) / self.num_scenarios
        # 只改变排列，不改变分层点集合；不同随机变量使用不同 salt，避免完全正相关。
        shift = int((int(state_id) * 2654435761 + salt * 2246822519) % self.num_scenarios)
        return np.roll(base, shift)

    @staticmethod
    def sample_quantiles(values: np.ndarray, levels: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
        values = np.maximum.accumulate(np.asarray(values, dtype=np.float64))
        return np.interp(uniforms, levels, values, left=values[0], right=values[-1])

    @staticmethod
    def normalized_price_action_for_ratio(ratio: float) -> float:
        u = (float(ratio) - cfg.PRICE_RATIO_MIN) / (
            cfg.PRICE_RATIO_MAX - cfg.PRICE_RATIO_MIN
        )
        return float(np.clip(2.0 * u - 1.0, -1.0, 1.0))

    def evaluate_group(
        self,
        rows: pd.DataFrame,
        normalized_actions: np.ndarray,
        subsidy_policy: SubsidyPolicy = "actor",
        fixed_subsidy_ratio: float = 0.0,
    ) -> pd.DataFrame:
        """评估一个完整 (i,t) 组；rows 中必须包含该组全部有订单的 j。"""

        if rows["group_id"].nunique() != 1:
            raise ValueError("evaluate_group 每次只能接收一个完整 (i,t) 组。")
        actions = np.clip(np.asarray(normalized_actions, dtype=np.float32), -1.0, 1.0)
        if actions.shape != (len(rows), 2):
            raise ValueError(f"动作维度应为 {(len(rows), 2)}，实际为 {actions.shape}。")

        row_results: list[dict[str, float | int | str | pd.Timestamp]] = []
        potential_demand = np.zeros(len(rows), dtype=np.float64)

        for position, (_, row) in enumerate(rows.iterrows()):
            state_id = int(row["state_id"])
            price_u = (float(actions[position, 0]) + 1.0) / 2.0
            subsidy_u = (float(actions[position, 1]) + 1.0) / 2.0
            price_ratio = cfg.PRICE_RATIO_MIN + price_u * (
                cfg.PRICE_RATIO_MAX - cfg.PRICE_RATIO_MIN
            )
            price = float(row["p_ref_ijt"]) * price_ratio
            subsidy_max = cfg.SUBSIDY_MAX_RATIO * float(row["p_ref_ijt"])

            pe_values = row[
                ["pe_q10_ijt", "pe_q25_ijt", "pe_q50_ijt", "pe_q75_ijt", "pe_q90_ijt"]
            ].to_numpy(dtype=np.float64)
            pe_samples = self.sample_quantiles(
                pe_values, self.PE_LEVELS, self.uniforms(state_id, 11)
            )
            passenger_acceptance = float(np.mean(price <= pe_samples))

            tau_values = row[
                ["tau_q10_ijt", "tau_q25_ijt", "tau_q50_ijt", "tau_q75_ijt", "tau_q90_ijt"]
            ].to_numpy(dtype=np.float64)
            tau_samples = self.sample_quantiles(
                tau_values, self.FIVE_LEVELS, self.uniforms(state_id, 23)
            )
            travel_steps = np.maximum(
                np.ceil(tau_samples / cfg.TIME_STEP_MINUTES).astype(np.int16), 1
            )

            cost_values = row[
                ["c_q10_ijt", "c_q25_ijt", "c_q50_ijt", "c_q75_ijt", "c_q90_ijt"]
            ].to_numpy(dtype=np.float64)
            cost_samples = self.sample_quantiles(
                cost_values, self.FIVE_LEVELS, self.uniforms(state_id, 37)
            )

            reject_time = pd.Timestamp(row["t"]) + pd.Timedelta(minutes=cfg.TIME_STEP_MINUTES)
            reject_q = self._value_quantiles(int(row["i_index"]), reject_time)
            reject_v = self.sample_quantiles(
                reject_q, cfg.VIT_QUANTILE_LEVELS, self.uniforms(state_id, 41)
            )

            destination_v = np.empty(self.num_scenarios, dtype=np.float64)
            for step in np.unique(travel_steps):
                mask = travel_steps == step
                arrival_time = pd.Timestamp(row["t"]) + pd.Timedelta(
                    minutes=int(step) * cfg.TIME_STEP_MINUTES
                )
                destination_q = self._value_quantiles(int(row["j_index"]), arrival_time)
                all_samples = self.sample_quantiles(
                    destination_q,
                    cfg.VIT_QUANTILE_LEVELS,
                    self.uniforms(state_id + int(step), 53),
                )
                destination_v[mask] = all_samples[mask]

            q_reject = cfg.DRIVER_GAMMA * reject_v
            gap_without_subsidy = (
                (1.0 - cfg.PLATFORM_COMMISSION_RATE) * price
                - cost_samples
                + np.power(cfg.DRIVER_GAMMA, travel_steps) * destination_v
                - q_reject
            )

            supply = float(row["S_it_hist"])
            dispatch_count = int(min(cfg.MAX_DISPATCH, np.floor(max(supply, 0.0))))
            if dispatch_count > 0:
                required_single_acceptance = 1.0 - cfg.DRIVER_BETA ** (1.0 / dispatch_count)
                required_subsidy_samples = np.maximum(-gap_without_subsidy, 0.0)
                subsidy_min = float(
                    np.quantile(required_subsidy_samples, required_single_acceptance, method="higher")
                )
                b0 = float(np.mean(gap_without_subsidy >= 0.0))
                bmax = float(np.mean(gap_without_subsidy + subsidy_max >= 0.0))
                bbar0 = 1.0 - (1.0 - b0) ** dispatch_count
                bbar_max = 1.0 - (1.0 - bmax) ** dispatch_count
            else:
                required_single_acceptance = 1.0
                subsidy_min = subsidy_max
                bbar0 = 0.0
                bbar_max = 0.0

            hard_infeasible = dispatch_count == 0 or bbar_max < 1.0 - cfg.DRIVER_BETA
            if subsidy_policy == "actor":
                if dispatch_count == 0:
                    subsidy = 0.0
                elif bbar0 >= 1.0 - cfg.DRIVER_BETA:
                    subsidy = 0.0
                elif hard_infeasible:
                    subsidy = subsidy_max
                else:
                    subsidy_min = min(subsidy_min, subsidy_max)
                    subsidy = subsidy_min + subsidy_u * (subsidy_max - subsidy_min)
            elif subsidy_policy == "force_zero":
                subsidy = 0.0
            elif subsidy_policy == "minimum":
                subsidy = 0.0 if bbar0 >= 1.0 - cfg.DRIVER_BETA else min(subsidy_min, subsidy_max)
            elif subsidy_policy == "fixed":
                subsidy = min(max(fixed_subsidy_ratio, 0.0) * float(row["p_ref_ijt"]), subsidy_max)
            else:
                raise ValueError(f"未知 subsidy_policy：{subsidy_policy}")

            single_driver_acceptance = float(np.mean(gap_without_subsidy + subsidy >= 0.0))
            if dispatch_count > 0:
                final_driver_acceptance = 1.0 - (
                    1.0 - single_driver_acceptance
                ) ** dispatch_count
                driver_failure_probability = (1.0 - single_driver_acceptance) ** dispatch_count
                driver_violation = max(driver_failure_probability - cfg.DRIVER_BETA, 0.0)
            else:
                final_driver_acceptance = 0.0
                driver_failure_probability = 1.0
                # 文档规定 S=0 时不进入司机机会约束。
                driver_violation = 0.0

            passenger_violation = max(
                (1.0 - cfg.PASSENGER_ALPHA) - passenger_acceptance, 0.0
            )

            # “最高补贴仍无法达标”时，Actor/最低补贴策略按文档硬设 M=0。外部比较基线
            # 保留其模型成交量并报告违反度，以便看到无补贴或固定规则的真实代价。
            hard_zero_volume = hard_infeasible and subsidy_policy in {"actor", "minimum"}
            service_indicator = 0.0 if hard_zero_volume or supply <= 0 else 1.0
            potential = (
                float(row["D_ijt_observed"])
                * passenger_acceptance
                * final_driver_acceptance
                * service_indicator
            )
            potential_demand[position] = potential

            row_results.append(
                {
                    "state_id": state_id,
                    "group_id": int(row["group_id"]),
                    "i": str(row["i"]),
                    "j": str(row["j"]),
                    "t": pd.Timestamp(row["t"]),
                    "p_ijt": price,
                    "price_ratio": price_ratio,
                    "b_ijt": subsidy,
                    "b_min_ijt": subsidy_min,
                    "b_max_ijt": subsidy_max,
                    "A_ijt": passenger_acceptance,
                    "B_ijt": single_driver_acceptance,
                    "Bbar_ijt": final_driver_acceptance,
                    "driver_failure_probability": driver_failure_probability,
                    "g_passenger": passenger_violation,
                    "g_driver": driver_violation,
                    "dispatch_count": dispatch_count,
                    "hard_infeasible": int(hard_infeasible),
                    "potential_demand_ijt": potential,
                    "S_it_hist": supply,
                    "D_ijt_observed": float(row["D_ijt_observed"]),
                    "p_ref_ijt": float(row["p_ref_ijt"]),
                    "outside_q10_q90_support": int(
                        price < float(row["p_q10_ijt"]) or price > float(row["p_q90_ijt"])
                    ),
                }
            )

        result = pd.DataFrame(row_results)
        supply = float(rows["S_it_hist"].iloc[0])
        total_potential = float(potential_demand.sum())
        if total_potential <= 0.0:
            supply_scale = 0.0
        else:
            supply_scale = min(1.0, supply / total_potential)
        result["supply_scale"] = supply_scale
        result["M_ijt"] = result["potential_demand_ijt"] * supply_scale
        result["platform_profit"] = (
            cfg.PLATFORM_COMMISSION_RATE * result["p_ijt"] - result["b_ijt"]
        ) * result["M_ijt"]
        return result
