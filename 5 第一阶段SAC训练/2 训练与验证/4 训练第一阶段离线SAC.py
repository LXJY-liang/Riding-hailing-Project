# -*- coding: utf-8 -*-
"""在动作无关的历史下一状态上训练第一阶段分组SAC。"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


HERE = Path(__file__).resolve().parent


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


cfg = load(HERE.parent / "1 训练前的准备" / "1 第一阶段SAC配置.py", "stage1_config")
env_module = load(HERE / "2 定义外生状态环境.py", "stage1_environment")
net = load(HERE / "1 定义离线SAC网络.py", "stage1_networks")
dataset_module = load(HERE / "3 定义固定历史转移数据集.py", "stage1_fixed_dataset")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def contexts(group: dict) -> list[str]:
    return [
        f"date={group['date']}, slot={group['slot']}, i={int(group['i'][r])}, j={int(group['j'][r])}"
        for r in range(len(group["j"]))
    ]


def choose_actions(actor, groups, device):
    lengths = [len(group["j"]) for group in groups]
    state = torch.as_tensor(np.concatenate([g["state"] for g in groups]), dtype=torch.float32, device=device)
    i = torch.as_tensor(np.concatenate([g["i"] for g in groups]), dtype=torch.long, device=device)
    j = torch.as_tensor(np.concatenate([g["j"] for g in groups]), dtype=torch.long, device=device)
    p_ref = torch.as_tensor(np.concatenate([g["p_ref"] for g in groups]), dtype=torch.float32, device=device)
    rows = [item for group in groups for item in contexts(group)]
    with torch.no_grad():
        action, _ = actor.sample(state, i, j, p_ref, rows)
    return list(np.split(action.cpu().numpy().astype(np.float32), np.cumsum(lengths)[:-1]))


def tensorize(group: dict, device: torch.device) -> dict:
    result = {
        "state": torch.as_tensor(group["state"], dtype=torch.float32, device=device),
        "i": torch.as_tensor(group["i"], dtype=torch.long, device=device),
        "j": torch.as_tensor(group["j"], dtype=torch.long, device=device),
        "row_context": contexts(group),
    }
    for name, dtype in (
        ("p_ref", torch.float32), ("pe_q", torch.float32),
        ("tau_steps", torch.float32), ("vi_index", torch.long),
        ("vj_index", torch.long), ("wait_q", torch.float32),
        ("vj_scenario_q", torch.float32), ("cost", torch.float32),
        ("pi", torch.float32), ("base_demand", torch.float32),
    ):
        result[name] = torch.as_tensor(group[name], dtype=dtype, device=device)
    result["supply"] = torch.tensor(float(group["supply"]), dtype=torch.float32, device=device)
    result["demand_realized"] = torch.tensor(float(group["demand_realized"]), dtype=torch.float32, device=device)
    return result


def weighted_value(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.clamp_min(0.0)
    return values.mean() if float(weights.sum()) <= 0 else (values * weights / weights.sum()).sum()


def policy_reward(group, action, lambdas):
    """在固定历史状态上计算当前策略即时结果。

    乘客和司机机会约束均使用逐OD非负违反度 [g]_+。已满足约束的
    OD违反度为0，不能用负残差抵消其他OD的正违反。结果只用于
    计算当前策略回报，不写回固定历史数据集。
    """

    context, details = net.region_action_context(group, action)
    demand_mask = group["base_demand"] > 0
    driver_mask = demand_mask & (group["supply"] > 0)
    g_passenger_raw = torch.where(
        demand_mask,
        1.0 - cfg.PASSENGER_ALPHA - details["passenger"],
        torch.zeros_like(details["passenger"]),
    )
    g_driver_raw = torch.where(
        driver_mask,
        1.0 - cfg.DRIVER_BETA - details["driver_final"],
        torch.zeros_like(details["driver_final"]),
    )
    g_passenger = torch.relu(g_passenger_raw)
    g_driver = torch.relu(g_driver_raw)
    profit = (
        cfg.PLATFORM_RATE * details["price"] - details["subsidy"]
    ) * details["completed"]
    reward = (
        profit
        - lambdas["passenger"] * g_passenger
        - lambdas["driver"] * g_driver
    ) * cfg.REWARD_SCALE
    return context, reward, details


def update_sac(models, optimizers, samples, device, lambdas):
    if not samples:
        raise ValueError("固定历史转移批次不能为空。")
    alpha = models["log_alpha"].exp().detach()
    critic_sum = torch.zeros((), device=device)
    od_count = 0
    current_groups = [tensorize(sample["current"], device) for sample in samples]
    next_groups = {
        index: tensorize(sample["next"], device)
        for index, sample in enumerate(samples)
        if not sample["done"] and sample["next"] is not None
    }
    for index, (sample, current) in enumerate(zip(samples, current_groups)):
        raw = sample["current"]
        with torch.no_grad():
            # Critic动作来自当前策略，但动作、奖励和结果不写回固定历史数据。
            action, _ = models["actor"].sample(
                current["state"], current["i"], current["j"],
                current["p_ref"], current["row_context"],
            )
            context, reward, _ = policy_reward(current, action, lambdas)
            if sample["done"] or sample["next"] is None:
                target = reward
            else:
                nxt = next_groups[index]
                next_action, next_log_prob = models["actor"].sample(
                    nxt["state"], nxt["i"], nxt["j"], nxt["p_ref"], nxt["row_context"]
                )
                next_context, _ = net.region_action_context(nxt, next_action)
                next_q = torch.minimum(
                    models["target_q1"](nxt["state"], nxt["i"], nxt["j"], next_action, next_context),
                    models["target_q2"](nxt["state"], nxt["i"], nxt["j"], next_action, next_context),
                ) - alpha * next_log_prob.squeeze(-1)
                continuation = weighted_value(next_q, nxt["base_demand"])
                target = reward + (cfg.SAC_GAMMA ** sample["gap_steps"]) * continuation
        q1 = models["q1"](current["state"], current["i"], current["j"], action, context)
        q2 = models["q2"](current["state"], current["i"], current["j"], action, context)
        critic_sum = critic_sum + F.smooth_l1_loss(
            q1, target, reduction="sum", beta=cfg.CRITIC_HUBER_DELTA
        ) + F.smooth_l1_loss(q2, target, reduction="sum", beta=cfg.CRITIC_HUBER_DELTA)
        od_count += len(raw["j"])
    critic_loss = critic_sum / max(od_count, 1)
    optimizers["critic"].zero_grad(set_to_none=True)
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(models["q1"].parameters()) + list(models["q2"].parameters()), cfg.CRITIC_GRAD_CLIP
    )
    optimizers["critic"].step()

    critic_parameters = list(models["q1"].parameters()) + list(models["q2"].parameters())
    for parameter in critic_parameters:
        parameter.requires_grad_(False)
    actor_sum = torch.zeros((), device=device)
    alpha_sum = torch.zeros((), device=device)
    for group in current_groups:
        action, log_prob = models["actor"].sample(
            group["state"], group["i"], group["j"], group["p_ref"], group["row_context"]
        )
        context, _ = net.region_action_context(group, action)
        q = torch.minimum(
            models["q1"](group["state"], group["i"], group["j"], action, context),
            models["q2"](group["state"], group["i"], group["j"], action, context),
        )
        actor_sum = actor_sum + (alpha * log_prob.squeeze(-1) - q).sum()
        alpha_sum = alpha_sum - (
            models["log_alpha"] * (log_prob.detach() + cfg.TARGET_ENTROPY)
        ).sum()
    actor_loss = actor_sum / max(od_count, 1)
    optimizers["actor"].zero_grad(set_to_none=True)
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(models["actor"].parameters(), cfg.ACTOR_GRAD_CLIP)
    optimizers["actor"].step()
    for parameter in critic_parameters:
        parameter.requires_grad_(True)

    alpha_loss = alpha_sum / max(od_count, 1)
    optimizers["alpha"].zero_grad(set_to_none=True)
    alpha_loss.backward()
    optimizers["alpha"].step()
    net.soft_update(models["target_q1"], models["q1"], cfg.TARGET_TAU)
    net.soft_update(models["target_q2"], models["q2"], cfg.TARGET_TAU)
    return {
        "critic_loss": float(critic_loss.item()),
        "actor_loss": float(actor_loss.item()),
        "alpha": float(models["log_alpha"].exp().item()),
    }


def update_lambdas(lambdas, passenger_values, driver_values):
    """用当前四小时窗口内所有有效OD的平均非负违反度更新全局乘子。

    这里再执行一次 maximum(., 0)作为防御性检查，防止调用方误传
    有符号残差后重新出现正负抵消。
    """
    for name, values in (("passenger", passenger_values), ("driver", driver_values)):
        if values:
            violations = np.maximum(np.concatenate(values), 0.0)
            mean = float(violations.mean())
            lambdas[name] = float(np.clip(lambdas[name] + cfg.LAMBDA_LR * mean, 0.0, cfg.LAMBDA_MAX))
    passenger_values.clear()
    driver_values.clear()


def episode_files(split: str) -> list[Path]:
    result = []
    for path in sorted(cfg.EPISODE_DIR.glob("episode_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            if str(data["split"]) == split:
                result.append(path)
    return result


def first_group(files, seed, available_dates):
    for path in files:
        episode = env_module.OfflineSACEpisode(path, seed, available_dates)
        while episode.slot < cfg.STEPS_PER_DAY:
            groups = episode.build_groups()
            if groups:
                return episode, groups[0]
            episode.step([], [])
    raise RuntimeError("训练episode中没有可用OD组。")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default=cfg.DEFAULT_RUN_NAME)
    parser.add_argument("--epochs", type=int, default=cfg.TRAIN_EPOCHS)
    parser.add_argument("--seed", type=int, default=cfg.RANDOM_SEED)
    args = parser.parse_args()
    cfg.ensure_directories()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    files = episode_files("train")
    if not files:
        raise FileNotFoundError("没有训练集episode。")
    checkpoint_dir = cfg.MODEL_DIR / args.run_name
    csv_path = cfg.LOG_DIR / f"training_{args.run_name}.csv"
    if checkpoint_dir.exists() or csv_path.exists():
        raise FileExistsError(f"实验名{args.run_name!r}已有结果，请更换实验名。")
    checkpoint_dir.mkdir(parents=True)

    available_dates = {path.stem.removeprefix("episode_") for path in files}
    prototype_episode, prototype = first_group(files, args.seed, available_dates)
    state_dim = prototype["state"].shape[1]
    region_count = prototype_episode.region_count
    actor = net.Actor(state_dim, region_count).to(device)
    q1 = net.Critic(state_dim, region_count).to(device)
    q2 = net.Critic(state_dim, region_count).to(device)
    models = {
        "actor": actor, "q1": q1, "q2": q2,
        "target_q1": copy.deepcopy(q1).eval(), "target_q2": copy.deepcopy(q2).eval(),
        "log_alpha": torch.tensor(np.log(cfg.INITIAL_ALPHA), dtype=torch.float32, device=device, requires_grad=True),
    }
    optimizers = {
        "actor": torch.optim.Adam(actor.parameters(), lr=cfg.ACTOR_LR),
        "critic": torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=cfg.CRITIC_LR),
        "alpha": torch.optim.Adam([models["log_alpha"]], lr=cfg.ENTROPY_LR),
    }
    fixed_dataset = dataset_module.FixedHistoricalTransitionDataset(
        files, env_module.OfflineSACEpisode, args.seed
    )
    dataset_audit = fixed_dataset.audit()
    print(
        "固定历史转移数据集："
        f"{dataset_audit['transitions']:,}组，{dataset_audit['episode_days']}天，"
        f"跨日转移={dataset_audit['cross_day_transitions']:,}，"
        f"终止转移={dataset_audit['terminal_transitions']:,}"
    )
    lambdas = {"passenger": cfg.LAMBDA_PASSENGER_INIT, "driver": cfg.LAMBDA_DRIVER_INIT}
    global_step = 0
    gradient_step = 0
    fields = [
        "epoch", "global_step", "profit", "potential_demand", "passenger_accepted_demand",
        "bilateral_accepted_demand", "completed_orders", "transaction_rate",
        "supply_fulfillment_rate", "critic_loss", "actor_loss", "alpha",
        "passenger_acceptance_mean", "passenger_violation_rate", "driver_single_acceptance_mean",
        "passenger_violation_od_rate", "driver_final_acceptance_mean", "driver_violation_rate",
        "driver_violation_od_rate", "passenger_violation_mean", "passenger_violation_max",
        "driver_violation_mean", "driver_violation_max", "hard_infeasible_od_rate", "price_ratio_mean",
        "subsidy_ratio_mean", "total_subsidy", "lambda_passenger", "lambda_driver",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            print(f"开始 epoch={epoch}/{args.epochs}", flush=True)
            totals = {name: 0.0 for name in (
                "profit", "potential", "passenger_accepted", "bilateral", "completed", "supply",
                "passenger_acceptance", "driver_single", "driver_final", "passenger_violation",
                "driver_violation", "demand_weight", "price_ratio", "subsidy_ratio", "od_weight", "subsidy",
                "passenger_active_od", "driver_active_od", "passenger_violation_od",
                "driver_violation_od", "passenger_violation_amount",
                "driver_violation_amount", "hard_infeasible_od",
            )}
            totals["passenger_violation_max"] = 0.0
            totals["driver_violation_max"] = 0.0
            losses, actor_losses = [], []
            passenger_window: list[np.ndarray] = []
            driver_window: list[np.ndarray] = []
            # 历史场景种子不随epoch变化，因此每轮读取完全相同的外生历史转移。
            for file_index, path in enumerate(files):
                print(
                    f"epoch={epoch}/{args.epochs}, "
                    f"day={file_index + 1}/{len(files)}, date={path.stem.removeprefix('episode_')}",
                    flush=True,
                )
                episode = env_module.OfflineSACEpisode(
                    path, args.seed + file_index, available_dates
                )
                while episode.slot < cfg.STEPS_PER_DAY:
                    groups = episode.build_groups()
                    if not groups:
                        episode.step([], [])
                        global_step += 1
                        continue
                    actions = choose_actions(actor, groups, device)
                    results = episode.step(groups, actions)
                    for result in results:
                        demand = np.maximum(result["base_demand"], 0.0)
                        passenger_accepted = demand * result["passenger_acceptance"]
                        totals["profit"] += float(result["profit"].sum())
                        totals["potential"] += float(demand.sum())
                        totals["passenger_accepted"] += float(passenger_accepted.sum())
                        totals["bilateral"] += float(result["bilateral_accepted_demand"].sum())
                        totals["completed"] += float(result["completed"].sum())
                        totals["supply"] += float(result["supply"])
                        totals["demand_weight"] += float(demand.sum())
                        totals["passenger_acceptance"] += float(np.sum(result["passenger_acceptance"] * demand))
                        totals["driver_single"] += float(np.sum(result["driver_single"] * demand))
                        totals["driver_final"] += float(np.sum(result["driver_final"] * demand))
                        totals["passenger_violation"] += float(np.sum((result["g_passenger"] > 0) * demand))
                        totals["driver_violation"] += float(np.sum((result["g_driver"] > 0) * demand))
                        passenger_active = result["passenger_constraint_active"]
                        driver_active = result["driver_constraint_active"]
                        totals["passenger_active_od"] += int(passenger_active.sum())
                        totals["driver_active_od"] += int(driver_active.sum())
                        totals["passenger_violation_od"] += int(
                            np.sum((result["g_passenger"] > 0) & passenger_active)
                        )
                        totals["driver_violation_od"] += int(
                            np.sum((result["g_driver"] > 0) & driver_active)
                        )
                        passenger_violations = result["g_passenger"][passenger_active]
                        driver_violations = result["g_driver"][driver_active]
                        totals["passenger_violation_amount"] += float(
                            passenger_violations.sum()
                        )
                        totals["driver_violation_amount"] += float(
                            driver_violations.sum()
                        )
                        if passenger_violations.size:
                            totals["passenger_violation_max"] = max(
                                totals["passenger_violation_max"],
                                float(passenger_violations.max()),
                            )
                        if driver_violations.size:
                            totals["driver_violation_max"] = max(
                                totals["driver_violation_max"],
                                float(driver_violations.max()),
                            )
                        totals["hard_infeasible_od"] += int(result["hard_infeasible"].sum())
                        active = demand > 0
                        totals["price_ratio"] += float(np.sum(result["price"][active] / result["p_ref"][active]))
                        totals["subsidy_ratio"] += float(np.sum(result["subsidy"][active] / result["p_ref"][active]))
                        totals["od_weight"] += int(active.sum())
                        totals["subsidy"] += float(np.sum(result["subsidy"] * result["completed"]))
                        # 乘子与Actor回报使用同一口径：逐OD非负违反度。
                        passenger_window.append(passenger_violations)
                        driver_window.append(driver_violations)
                    global_step += 1
                    if global_step % cfg.LAMBDA_UPDATE_INTERVAL_STEPS == 0:
                        update_lambdas(lambdas, passenger_window, driver_window)
                    for _ in range(cfg.GRADIENT_UPDATES_PER_TIME_STEP):
                        # 只从训练开始前已固定的历史索引中抽样；没有任何经验写入。
                        batch = fixed_dataset.sample(cfg.GROUP_BATCH_SIZE)
                        stats = update_sac(models, optimizers, batch, device, lambdas)
                        gradient_step += 1
                        losses.append(stats["critic_loss"])
                        actor_losses.append(stats["actor_loss"])
            if passenger_window or driver_window:
                update_lambdas(lambdas, passenger_window, driver_window)
            demand_total = max(totals["demand_weight"], 1e-9)
            od_total = max(totals["od_weight"], 1.0)
            row = {
                "epoch": epoch, "global_step": global_step, "profit": totals["profit"],
                "potential_demand": totals["potential"],
                "passenger_accepted_demand": totals["passenger_accepted"],
                "bilateral_accepted_demand": totals["bilateral"],
                "completed_orders": totals["completed"],
                "transaction_rate": totals["completed"] / max(totals["potential"], 1e-9),
                "supply_fulfillment_rate": totals["completed"] / max(totals["bilateral"], 1e-9),
                "critic_loss": float(np.mean(losses)) if losses else np.nan,
                "actor_loss": float(np.mean(actor_losses)) if actor_losses else np.nan,
                "alpha": float(models["log_alpha"].exp().item()),
                "passenger_acceptance_mean": totals["passenger_acceptance"] / demand_total,
                "passenger_violation_rate": totals["passenger_violation"] / demand_total,
                "passenger_violation_od_rate": totals["passenger_violation_od"] / max(totals["passenger_active_od"], 1),
                "driver_single_acceptance_mean": totals["driver_single"] / demand_total,
                "driver_final_acceptance_mean": totals["driver_final"] / demand_total,
                "driver_violation_rate": totals["driver_violation"] / demand_total,
                "driver_violation_od_rate": totals["driver_violation_od"] / max(totals["driver_active_od"], 1),
                "passenger_violation_mean": totals["passenger_violation_amount"] / max(totals["passenger_active_od"], 1),
                "passenger_violation_max": totals["passenger_violation_max"],
                "driver_violation_mean": totals["driver_violation_amount"] / max(totals["driver_active_od"], 1),
                "driver_violation_max": totals["driver_violation_max"],
                "hard_infeasible_od_rate": totals["hard_infeasible_od"] / max(totals["passenger_active_od"], 1),
                "price_ratio_mean": totals["price_ratio"] / od_total,
                "subsidy_ratio_mean": totals["subsidy_ratio"] / od_total,
                "total_subsidy": totals["subsidy"],
                "lambda_passenger": lambdas["passenger"], "lambda_driver": lambdas["driver"],
            }
            writer.writerow(row)
            stream.flush()
            checkpoint = {
                "epoch": epoch, "global_step": global_step, "gradient_step": gradient_step,
                "run_name": args.run_name,
                "state_dim": state_dim, "region_count": region_count,
                "actor": actor.state_dict(), "q1": q1.state_dict(), "q2": q2.state_dict(),
                "target_q1": models["target_q1"].state_dict(), "target_q2": models["target_q2"].state_dict(),
                "log_alpha": models["log_alpha"].detach().cpu(), "lambdas": dict(lambdas),
                "optimizer_actor": optimizers["actor"].state_dict(),
                "optimizer_critic": optimizers["critic"].state_dict(),
                "optimizer_alpha": optimizers["alpha"].state_dict(),
                "config": {
                    "exogenous_next_state": True, "vit_frozen": True,
                    "fixed_historical_transition_dataset": True,
                    "actor_experience_writeback": False,
                    "fixed_historical_transition_count": len(fixed_dataset),
                    "cross_day_transition_count": dataset_audit["cross_day_transitions"],
                    "split_boundary_terminal": True,
                    "gradient_updates_per_time_step": cfg.GRADIENT_UPDATES_PER_TIME_STEP,
                    "nonnegative_constraint_aggregation": True,
                    "lambda_update_interval_steps": cfg.LAMBDA_UPDATE_INTERVAL_STEPS,
                    "demand_completion_rate": cfg.DEMAND_COMPLETION_RATE,
                    "supply_quantile": cfg.SUPPLY_QUANTILE,
                    "supply_utilization": cfg.SUPPLY_UTILIZATION,
                },
            }
            epoch_path = checkpoint_dir / f"epoch_{epoch:03d}_offline_sac.pt"
            torch.save(checkpoint, epoch_path)
            shutil.copy2(epoch_path, checkpoint_dir / "latest_offline_sac.pt")
            print(
                f"完成 epoch={epoch}/{args.epochs}, "
                f"profit={row['profit']:.2f}, completed={row['completed_orders']:.2f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
