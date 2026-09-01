# -*- coding: utf-8 -*-
"""任务 6：在验证集或测试集上评估 SAC，并可选择是否运行比较基线。

默认模式只评估 SAC，避免在仅查看 Actor 价格和补贴时误运行耗时的二维网格搜索。
测试集必须显式传入 ``--split test``，防止在模型选择阶段提前查看测试结果。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


MODULE_DIR = Path(__file__).resolve().parent


def load_module(file_name: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, MODULE_DIR / file_name)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


cfg_path = MODULE_DIR.parents[0] / "1 数据准备" / "1 离线SAC统一配置.py"
cfg_spec = importlib.util.spec_from_file_location("offline_sac_config", cfg_path)
cfg = importlib.util.module_from_spec(cfg_spec)
assert cfg_spec.loader is not None
cfg_spec.loader.exec_module(cfg)
reward_module = load_module("2 计算反事实收益.py", "counterfactual_reward")
network_module = load_module("3 定义SAC网络.py", "sac_networks")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估离线 SAC 策略及可选比较基线")
    parser.add_argument(
        "--split",
        choices=["valid", "test"],
        default="valid",
        help="评估数据段；默认为验证集，测试集必须显式指定",
    )
    parser.add_argument(
        "--mode",
        choices=["sac_only", "simple_baselines", "all"],
        default="sac_only",
        help=(
            "sac_only 仅输出 SAC；simple_baselines 增加三个简单基线；"
            "all 再增加非常耗时的二维网格搜索"
        ),
    )
    parser.add_argument(
        "--max-groups", type=int, default=0, help="只评估前 N 个完整 (i,t) 组；0 为完整数据段"
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=cfg.MODEL_CHECKPOINT, help="要评估的 SAC 检查点"
    )
    parser.add_argument(
        "--output-tag", type=str, default=cfg.DEFAULT_RUN_NAME, help="输出文件名后缀"
    )
    return parser.parse_args()


def device_for_evaluation() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalized_price(engine: reward_module.CounterfactualReward, ratio: float, rows: int) -> np.ndarray:
    value = engine.normalized_price_action_for_ratio(ratio)
    return np.full(rows, value, dtype=np.float32)


def coordinate_grid_actions(
    rows: pd.DataFrame,
    engine: reward_module.CounterfactualReward,
    lambda_passenger: float,
    lambda_driver: float,
) -> np.ndarray:
    """对每个 OD 搜索价格×补贴二维网格。

    第一版中 D 使用历史已成交量且 S 被重构为不小于区域历史总成交量，所以任意
    `D*A*Bbar` 的组内总和都不会超过 S，供给缩放恒为 1。此时逐 OD 二维搜索与完整组
    联合坐标搜索等价，但计算量从 O(J^2) 降为 O(J)。
    """

    actions = np.column_stack(
        [normalized_price(engine, 1.0, len(rows)), np.full(len(rows), -1.0, dtype=np.float32)]
    )
    for _ in range(cfg.GRID_COORDINATE_PASSES):
        for position in range(len(rows)):
            one_row = rows.iloc[[position]].copy()
            best_score = -np.inf
            best_action = actions[position].copy()
            for price_ratio in cfg.GRID_PRICE_RATIOS:
                price_action = engine.normalized_price_action_for_ratio(float(price_ratio))
                for subsidy_fraction in cfg.GRID_SUBSIDY_FRACTIONS:
                    candidate_action = np.asarray(
                        [[price_action, 2.0 * float(subsidy_fraction) - 1.0]], dtype=np.float32
                    )
                    outcome = engine.evaluate_group(
                        one_row, candidate_action, subsidy_policy="actor"
                    )
                    score = float(
                        outcome["platform_profit"].sum()
                        - lambda_passenger * outcome["g_passenger"].sum()
                        - lambda_driver * outcome["g_driver"].sum()
                    )
                    if score > best_score:
                        best_score = score
                        best_action = candidate_action[0].copy()
            actions[position] = best_action
    return actions


def summarize(output: pd.DataFrame) -> dict[str, float | int]:
    completed = float(output["M_ijt"].sum())
    observed = float(output["D_ijt_observed"].sum())
    return {
        "od_rows": int(len(output)),
        "platform_total_profit": float(output["platform_profit"].sum()),
        "profit_per_completed_order": float(output["platform_profit"].sum() / max(completed, 1e-8)),
        "historical_observed_orders": observed,
        "model_completed_orders": completed,
        "model_completion_ratio_to_observed": completed / max(observed, 1e-8),
        "passenger_acceptance_mean": float(output["A_ijt"].mean()),
        "driver_single_acceptance_mean": float(output["B_ijt"].mean()),
        "driver_final_acceptance_mean": float(output["Bbar_ijt"].mean()),
        "passenger_violation_rate": float((output["g_passenger"] > 0).mean()),
        "driver_violation_rate": float((output["g_driver"] > 0).mean()),
        "passenger_violation_mean": float(output["g_passenger"].mean()),
        "driver_violation_mean": float(output["g_driver"].mean()),
        "average_historical_price": float(output["p_ref_ijt"].mean()),
        "average_price": float(output["p_ijt"].mean()),
        "average_price_ratio": float(output["price_ratio"].mean()),
        "average_price_difference": float(output["price_difference"].mean()),
        "average_price_change_percent": float(output["price_change_percent"].mean()),
        "median_price_change_percent": float(output["price_change_percent"].median()),
        "price_increase_share": float((output["price_difference"] > 1e-6).mean()),
        "price_decrease_share": float((output["price_difference"] < -1e-6).mean()),
        "price_unchanged_share": float((output["price_difference"].abs() <= 1e-6).mean()),
        "average_subsidy": float(output["b_ijt"].mean()),
        "median_subsidy": float(output["b_ijt"].median()),
        "average_subsidy_ratio_to_historical_price": float(
            output["subsidy_ratio_to_historical_price"].mean()
        ),
        "subsidized_od_share": float((output["b_ijt"] > 1e-8).mean()),
        "outside_q10_q90_support_rate": float(output["outside_q10_q90_support"].mean()),
        "hard_infeasible_rate": float(output["hard_infeasible"].mean()),
        "supply_binding_rate": float(
            (
                (output["supply_scale"] < 1.0 - 1e-8)
                & (output["potential_demand_ijt"] > 0)
            ).mean()
        ),
    }


def main() -> None:
    args = parse_args()
    cfg.ensure_output_directories()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    scaler = np.load(cfg.STATE_SCALER_OUTPUT, allow_pickle=True)
    mean = scaler["mean"].astype(np.float32)
    std = scaler["std"].astype(np.float32)

    required = list(
        dict.fromkeys(
            [
                "state_id", "group_id", "i", "j", "t", "split", "i_index", "j_index",
                "p_ref_ijt", "D_ijt_observed", "S_it_hist", "p_q10_ijt", "p_q90_ijt",
                "pe_q10_ijt", "pe_q25_ijt", "pe_q50_ijt", "pe_q75_ijt", "pe_q90_ijt",
                "tau_q10_ijt", "tau_q25_ijt", "tau_q50_ijt", "tau_q75_ijt", "tau_q90_ijt",
                "c_q10_ijt", "c_q25_ijt", "c_q50_ijt", "c_q75_ijt", "c_q90_ijt",
            ]
            + cfg.STATE_NUMERIC_COLUMNS
        )
    )
    # 只读取指定数据段。完整状态表约 290 万行；Parquet 谓词过滤可显著减少验证时内存。
    data = pd.read_parquet(
        cfg.STATE_OUTPUT,
        columns=required,
        filters=[("split", "==", args.split)],
    )
    data["t"] = pd.to_datetime(data["t"])
    data = data.sort_values(["group_id", "j"], kind="mergesort")
    group_ids = data["group_id"].drop_duplicates().to_numpy(dtype=np.int64)
    if args.max_groups > 0:
        group_ids = group_ids[: args.max_groups]
        data = data.loc[data["group_id"].isin(group_ids)]

    device = device_for_evaluation()
    actor = network_module.Actor(checkpoint["numeric_dim"], checkpoint["region_count"]).to(device)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    engine = reward_module.CounterfactualReward(cfg.EVAL_NUM_SCENARIOS)
    lambda_passenger = float(checkpoint["lambda_passenger"])
    lambda_driver = float(checkpoint["lambda_driver"])

    outputs: list[pd.DataFrame] = []
    strategies = ["SAC"]
    if args.mode in {"simple_baselines", "all"}:
        strategies.extend(
            ["历史价格_无补贴", "固定比例价格_固定补贴", "历史价格_最低可行补贴"]
        )
    if args.mode == "all":
        strategies.append("二维网格坐标优化")

    for number, group_id in enumerate(group_ids, start=1):
        rows = data.loc[data["group_id"] == group_id]
        numeric = ((rows[cfg.STATE_NUMERIC_COLUMNS].to_numpy(dtype=np.float32) - mean) / std)
        with torch.no_grad():
            sac_actions = actor.deterministic(
                torch.as_tensor(numeric, device=device),
                torch.as_tensor(rows["i_index"].to_numpy(dtype=np.int64), device=device),
                torch.as_tensor(rows["j_index"].to_numpy(dtype=np.int64), device=device),
            ).cpu().numpy()

        actions_by_strategy: dict[str, tuple[np.ndarray, str, float]] = {
            "SAC": (sac_actions, "actor", 0.0)
        }
        if args.mode in {"simple_baselines", "all"}:
            actions_by_strategy.update({
                "历史价格_无补贴": (
                np.column_stack([normalized_price(engine, 1.0, len(rows)), np.full(len(rows), -1.0)]),
                "force_zero",
                0.0,
                ),
                "固定比例价格_固定补贴": (
                np.column_stack(
                    [
                        normalized_price(engine, cfg.FIXED_PRICE_RATIO, len(rows)),
                        np.full(len(rows), -1.0),
                    ]
                ),
                "fixed",
                cfg.FIXED_SUBSIDY_RATIO,
                ),
                "历史价格_最低可行补贴": (
                np.column_stack([normalized_price(engine, 1.0, len(rows)), np.full(len(rows), -1.0)]),
                "minimum",
                0.0,
                ),
            })
        if args.mode == "all":
            grid_actions = coordinate_grid_actions(
                rows, engine, lambda_passenger, lambda_driver
            )
            actions_by_strategy["二维网格坐标优化"] = (grid_actions, "actor", 0.0)

        for strategy, (actions, subsidy_policy, fixed_ratio) in actions_by_strategy.items():
            outcome = engine.evaluate_group(
                rows,
                actions.astype(np.float32),
                subsidy_policy=subsidy_policy,
                fixed_subsidy_ratio=fixed_ratio,
            )
            outcome["price_difference"] = outcome["p_ijt"] - outcome["p_ref_ijt"]
            outcome["price_change_rate"] = (
                outcome["price_difference"] / outcome["p_ref_ijt"].clip(lower=1e-8)
            )
            outcome["price_change_percent"] = 100.0 * outcome["price_change_rate"]
            outcome["subsidy_ratio_to_historical_price"] = (
                outcome["b_ijt"] / outcome["p_ref_ijt"].clip(lower=1e-8)
            )
            outcome["strategy"] = strategy
            outputs.append(outcome)

        if number % 1_000 == 0 or number == len(group_ids):
            print(f"已评估 {number:,}/{len(group_ids):,} 个 {args.split} 组")

    output = pd.concat(outputs, ignore_index=True)
    metrics = {
        "experiment": {
            "checkpoint": str(args.checkpoint),
            "sac_gamma": float(checkpoint["sac_gamma"]),
            "split": args.split,
            "mode": args.mode,
            "groups": int(len(group_ids)),
        },
        "strategies": {
            strategy: summarize(output.loc[output["strategy"] == strategy])
            for strategy in strategies
        },
    }
    policy_output = (
        cfg.SAC_OUTPUT_DIR / f"{args.split}_policy_{args.mode}_{args.output_tag}.parquet"
    )
    metrics_output = (
        cfg.LOG_OUTPUT_DIR / f"evaluation_metrics_{args.split}_{args.mode}_{args.output_tag}.json"
    )
    policy_output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(policy_output, index=False, compression="snappy")
    metrics_output.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"策略明细：{policy_output}")
    print(f"指标汇总：{metrics_output}")


if __name__ == "__main__":
    main()
