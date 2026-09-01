# -*- coding: utf-8 -*-
"""任务 5：在冻结历史状态上收集反事实动作，并训练离线 SAC 策略。

这里的“环境交互”不是线上交互：所有 s 和 s' 都来自历史表，当前动作只改变当前奖励。
反事实动作包括 Actor 动作、历史动作附近扰动和全可行域随机动作。收益计算器对同一
(i,t) 的全部目的地一起计算供给缩放，然后把 OD 层样本放入回放池。
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


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
    parser = argparse.ArgumentParser(description="训练冻结历史转移的离线 SAC")
    parser.add_argument("--sac-gamma", type=float, default=cfg.SAC_GAMMA)
    parser.add_argument(
        "--run-name",
        type=str,
        default=cfg.DEFAULT_RUN_NAME,
        help="模型子目录和训练日志后缀，例如 gamma099 或 gamma000。",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="只运行 5 次收集和极小验证，用于检查数据、奖励与梯度链路。",
    )
    return parser.parse_args()


class StateStore:
    """持有状态表、训练期标准化参数和连续 group_id 的切片。"""

    def __init__(self):
        columns = list(
            dict.fromkeys(
                [
                    "state_id", "group_id", "i", "j", "t", "split", "i_index", "j_index",
                    "next_state_id", "done", "p_ref_ijt", "D_ijt_observed", "S_it_hist",
                    "p_q10_ijt", "p_q90_ijt",
                    "pe_q10_ijt", "pe_q25_ijt", "pe_q50_ijt", "pe_q75_ijt", "pe_q90_ijt",
                    "tau_q10_ijt", "tau_q25_ijt", "tau_q50_ijt", "tau_q75_ijt", "tau_q90_ijt",
                    "c_q10_ijt", "c_q25_ijt", "c_q50_ijt", "c_q75_ijt", "c_q90_ijt",
                ]
                + cfg.STATE_NUMERIC_COLUMNS
            )
        )
        self.frame = pd.read_parquet(cfg.STATE_OUTPUT, columns=columns)
        self.frame["t"] = pd.to_datetime(self.frame["t"])
        self.frame = self.frame.sort_values(["group_id", "j"], kind="mergesort").reset_index(drop=True)

        # state_id 是原始状态表的稳定标识；排序后用字典把它映射到当前 numpy 行位置。
        self.state_id_to_position = pd.Series(
            np.arange(len(self.frame), dtype=np.int64), index=self.frame["state_id"].to_numpy()
        )
        train_mask = self.frame["split"].eq("train").to_numpy()
        raw_numeric = self.frame[cfg.STATE_NUMERIC_COLUMNS].to_numpy(dtype=np.float32)
        self.mean = raw_numeric[train_mask].mean(axis=0).astype(np.float32)
        self.std = raw_numeric[train_mask].std(axis=0).astype(np.float32)
        self.std[self.std < 1e-6] = 1.0
        self.numeric = ((raw_numeric - self.mean) / self.std).astype(np.float32)
        np.savez_compressed(
            cfg.STATE_SCALER_OUTPUT,
            mean=self.mean,
            std=self.std,
            columns=np.asarray(cfg.STATE_NUMERIC_COLUMNS, dtype=object),
        )

        group_values = self.frame["group_id"].to_numpy(dtype=np.int64)
        starts = np.flatnonzero(np.r_[True, group_values[1:] != group_values[:-1]])
        ends = np.r_[starts[1:], len(group_values)]
        self.group_ids = group_values[starts]
        self.group_start = dict(zip(self.group_ids.tolist(), starts.tolist()))
        self.group_end = dict(zip(self.group_ids.tolist(), ends.tolist()))
        group_split = self.frame["split"].to_numpy()[starts]
        self.groups_by_split = {
            split: self.group_ids[group_split == split] for split in ("train", "valid", "test")
        }

    def group_rows(self, group_id: int) -> tuple[pd.DataFrame, np.ndarray]:
        start = self.group_start[int(group_id)]
        end = self.group_end[int(group_id)]
        return self.frame.iloc[start:end], np.arange(start, end, dtype=np.int64)

    def next_positions(self, rows: pd.DataFrame, current_positions: np.ndarray) -> np.ndarray:
        next_ids = rows["next_state_id"].to_numpy(dtype=np.int64)
        result = current_positions.copy()
        active = next_ids >= 0
        if active.any():
            result[active] = self.state_id_to_position.loc[next_ids[active]].to_numpy(dtype=np.int64)
        return result


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def actor_actions(
    actor: torch.nn.Module,
    store: StateStore,
    positions: np.ndarray,
    device: torch.device,
    deterministic: bool,
) -> np.ndarray:
    numeric = torch.as_tensor(store.numeric[positions], device=device)
    i_index = torch.as_tensor(
        store.frame.iloc[positions]["i_index"].to_numpy(dtype=np.int64), device=device
    )
    j_index = torch.as_tensor(
        store.frame.iloc[positions]["j_index"].to_numpy(dtype=np.int64), device=device
    )
    with torch.no_grad():
        if deterministic:
            action = actor.deterministic(numeric, i_index, j_index)
        else:
            action, _ = actor.sample(numeric, i_index, j_index)
    return action.cpu().numpy().astype(np.float32)


def sample_action_source(rng: np.random.Generator) -> str:
    names = list(cfg.ACTION_SOURCE_PROBABILITIES)
    probabilities = np.asarray([cfg.ACTION_SOURCE_PROBABILITIES[name] for name in names], dtype=float)
    probabilities /= probabilities.sum()
    return str(rng.choice(names, p=probabilities))


def collect_groups(
    store: StateStore,
    actor: torch.nn.Module,
    reward_engine: reward_module.CounterfactualReward,
    buffer: network_module.ReplayBuffer,
    rng: np.random.Generator,
    device: torch.device,
    groups_per_collection: int,
) -> dict[str, float]:
    """抽取若干完整历史组，生成一批反事实经验。"""

    sampled_groups = rng.choice(
        store.groups_by_split["train"], size=groups_per_collection, replace=True
    )
    profit_values: list[float] = []
    passenger_values: list[float] = []
    driver_values: list[float] = []

    for group_id in sampled_groups:
        rows, positions = store.group_rows(int(group_id))
        source = sample_action_source(rng)
        if source == "actor":
            actions = actor_actions(actor, store, positions, device, deterministic=False)
        elif source == "local":
            center_price = reward_engine.normalized_price_action_for_ratio(1.0)
            actions = np.column_stack(
                [
                    np.clip(rng.normal(center_price, 0.20, size=len(rows)), -1.0, 1.0),
                    rng.uniform(-1.0, 0.0, size=len(rows)),
                ]
            ).astype(np.float32)
        else:
            actions = rng.uniform(-1.0, 1.0, size=(len(rows), 2)).astype(np.float32)

        outcome = reward_engine.evaluate_group(rows, actions, subsidy_policy="actor")
        next_positions = store.next_positions(rows, positions)
        batch = {
            "state": store.numeric[positions],
            "next_state": store.numeric[next_positions],
            "i_index": rows["i_index"].to_numpy(dtype=np.int64),
            "j_index": rows["j_index"].to_numpy(dtype=np.int64),
            "next_i_index": store.frame.iloc[next_positions]["i_index"].to_numpy(dtype=np.int64),
            "next_j_index": store.frame.iloc[next_positions]["j_index"].to_numpy(dtype=np.int64),
            "action": actions,
            "profit": outcome["platform_profit"].to_numpy(dtype=np.float32)[:, None],
            "g_passenger": outcome["g_passenger"].to_numpy(dtype=np.float32)[:, None],
            "g_driver": outcome["g_driver"].to_numpy(dtype=np.float32)[:, None],
            "done": rows["done"].to_numpy(dtype=np.float32)[:, None],
        }
        buffer.add_batch(batch)
        profit_values.extend(outcome["platform_profit"].tolist())
        passenger_values.extend(outcome["g_passenger"].tolist())
        driver_values.extend(outcome["g_driver"].tolist())

    return {
        "collection_profit": float(np.mean(profit_values)),
        "collection_g_passenger": float(np.mean(passenger_values)),
        "collection_g_driver": float(np.mean(driver_values)),
    }


def validate(
    actor: torch.nn.Module,
    store: StateStore,
    engine: reward_module.CounterfactualReward,
    device: torch.device,
    lambda_passenger: float,
    lambda_driver: float,
    max_groups: int,
) -> dict[str, float]:
    groups = store.groups_by_split["valid"][:max_groups]
    totals = {"profit": 0.0, "g_passenger": 0.0, "g_driver": 0.0, "rows": 0}
    actor.eval()
    for group_id in groups:
        rows, positions = store.group_rows(int(group_id))
        actions = actor_actions(actor, store, positions, device, deterministic=True)
        outcome = engine.evaluate_group(rows, actions, subsidy_policy="actor")
        totals["profit"] += float(outcome["platform_profit"].sum())
        totals["g_passenger"] += float(outcome["g_passenger"].sum())
        totals["g_driver"] += float(outcome["g_driver"].sum())
        totals["rows"] += len(outcome)
    actor.train()
    rows = max(int(totals["rows"]), 1)
    adjusted = totals["profit"] - lambda_passenger * totals["g_passenger"] - lambda_driver * totals["g_driver"]
    return {
        "validation_profit": totals["profit"],
        "validation_adjusted_reward": adjusted,
        "validation_g_passenger": totals["g_passenger"] / rows,
        "validation_g_driver": totals["g_driver"] / rows,
    }


def main() -> None:
    run_started_at = time.perf_counter()
    args = parse_args()
    cfg.ensure_output_directories()
    if not 0.0 <= args.sac_gamma <= 1.0:
        raise ValueError("--sac-gamma 必须位于 [0,1]。")
    run_name = args.run_name.strip()
    if not run_name or any(character in run_name for character in "/\\"):
        raise ValueError("--run-name 不能为空且不能包含路径分隔符。")
    model_checkpoint = cfg.MODEL_OUTPUT_DIR / run_name / "best_offline_sac.pt"
    model_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    training_history_file = cfg.LOG_OUTPUT_DIR / f"training_history_{run_name}.csv"
    if args.smoke:
        run_name = f"{run_name}_smoke"
        model_checkpoint = cfg.MODEL_OUTPUT_DIR / run_name / "best_offline_sac.pt"
        model_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        training_history_file = cfg.LOG_OUTPUT_DIR / f"training_history_{run_name}.csv"
        total_collection_steps = 5
        learning_starts = 1
        groups_per_collection = 2
        batch_size = 32
        validation_interval = 5
        validation_groups = 5
    else:
        total_collection_steps = cfg.TOTAL_COLLECTION_STEPS
        learning_starts = cfg.LEARNING_STARTS
        groups_per_collection = cfg.GROUPS_PER_COLLECTION
        batch_size = cfg.BATCH_SIZE
        validation_interval = 5_000
        validation_groups = cfg.VALIDATION_GROUPS
    random.seed(cfg.RANDOM_SEED)
    np.random.seed(cfg.RANDOM_SEED)
    torch.manual_seed(cfg.RANDOM_SEED)
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    device = choose_device()
    print(f"训练设备：{device}")

    store = StateStore()
    region_count = int(max(store.frame["i_index"].max(), store.frame["j_index"].max())) + 1
    numeric_dim = len(cfg.STATE_NUMERIC_COLUMNS)
    actor = network_module.Actor(numeric_dim, region_count).to(device)
    critic1 = network_module.Critic(numeric_dim, region_count).to(device)
    critic2 = network_module.Critic(numeric_dim, region_count).to(device)
    target1 = copy.deepcopy(critic1).to(device).eval()
    target2 = copy.deepcopy(critic2).to(device).eval()
    for parameter in list(target1.parameters()) + list(target2.parameters()):
        parameter.requires_grad_(False)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.ACTOR_LR)
    critic_optimizer = torch.optim.Adam(
        list(critic1.parameters()) + list(critic2.parameters()), lr=cfg.CRITIC_LR
    )
    log_alpha = torch.tensor(
        np.log(cfg.INITIAL_ENTROPY_ALPHA), dtype=torch.float32, device=device, requires_grad=True
    )
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=cfg.ENTROPY_LR)
    lambda_passenger = float(cfg.LAMBDA_PASSENGER_INIT)
    lambda_driver = float(cfg.LAMBDA_DRIVER_INIT)

    buffer = network_module.ReplayBuffer(cfg.REPLAY_CAPACITY, numeric_dim, cfg.RANDOM_SEED)
    train_engine = reward_module.CounterfactualReward(cfg.TRAIN_NUM_SCENARIOS)
    history: list[dict[str, float | int]] = []
    best_validation = -np.inf

    for collection_step in range(1, total_collection_steps + 1):
        collected = collect_groups(
            store, actor, train_engine, buffer, rng, device, groups_per_collection
        )
        row: dict[str, float | int] = {"collection_step": collection_step, **collected}

        if buffer.size >= learning_starts:
            for _ in range(cfg.UPDATES_PER_COLLECTION):
                batch = buffer.sample(batch_size, device)
                reward = (
                    batch["profit"]
                    - lambda_passenger * batch["g_passenger"]
                    - lambda_driver * batch["g_driver"]
                )
                alpha = log_alpha.exp().detach()
                with torch.no_grad():
                    next_action, next_log_prob = actor.sample(
                        batch["next_state"], batch["next_i_index"], batch["next_j_index"]
                    )
                    next_q = torch.minimum(
                        target1(
                            batch["next_state"], batch["next_i_index"], batch["next_j_index"], next_action
                        ),
                        target2(
                            batch["next_state"], batch["next_i_index"], batch["next_j_index"], next_action
                        ),
                    ) - alpha * next_log_prob
                    target_q = reward + args.sac_gamma * (1.0 - batch["done"]) * next_q

                q1 = critic1(batch["state"], batch["i_index"], batch["j_index"], batch["action"])
                q2 = critic2(batch["state"], batch["i_index"], batch["j_index"], batch["action"])
                critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
                critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                critic_optimizer.step()

                sampled_action, log_prob = actor.sample(
                    batch["state"], batch["i_index"], batch["j_index"]
                )
                actor_q = torch.minimum(
                    critic1(batch["state"], batch["i_index"], batch["j_index"], sampled_action),
                    critic2(batch["state"], batch["i_index"], batch["j_index"], sampled_action),
                )
                actor_loss = (log_alpha.exp().detach() * log_prob - actor_q).mean()
                actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                actor_optimizer.step()

                alpha_loss = -(log_alpha * (log_prob.detach() + cfg.TARGET_ENTROPY)).mean()
                alpha_optimizer.zero_grad(set_to_none=True)
                alpha_loss.backward()
                alpha_optimizer.step()

                lambda_passenger = float(
                    np.clip(
                        lambda_passenger + cfg.LAMBDA_LR * batch["g_passenger"].mean().item(),
                        0.0,
                        cfg.LAMBDA_MAX,
                    )
                )
                lambda_driver = float(
                    np.clip(
                        lambda_driver + cfg.LAMBDA_LR * batch["g_driver"].mean().item(),
                        0.0,
                        cfg.LAMBDA_MAX,
                    )
                )
                network_module.soft_update(target1, critic1, cfg.TARGET_TAU)
                network_module.soft_update(target2, critic2, cfg.TARGET_TAU)

            row.update(
                {
                    "critic_loss": float(critic_loss.item()),
                    "actor_loss": float(actor_loss.item()),
                    "entropy_alpha": float(log_alpha.exp().item()),
                    "lambda_passenger": lambda_passenger,
                    "lambda_driver": lambda_driver,
                    "replay_size": int(buffer.size),
                }
            )

        if collection_step % 1_000 == 0:
            elapsed_seconds = time.perf_counter() - run_started_at
            seconds_per_step = elapsed_seconds / collection_step
            remaining_hours = (
                seconds_per_step * (total_collection_steps - collection_step) / 3600.0
            )
            print(
                f"step={collection_step:,}, replay={buffer.size:,}, "
                f"profit/OD={collected['collection_profit']:.4f}, "
                f"已用时={elapsed_seconds / 3600.0:.2f}小时, "
                f"预计剩余={remaining_hours:.2f}小时"
            )

        if collection_step % validation_interval == 0 and buffer.size >= learning_starts:
            # 复用同一份 V 稠密查询数组，只临时增加场景数，避免为验证再占用约 200MB。
            original_scenarios = train_engine.num_scenarios
            train_engine.num_scenarios = cfg.EVAL_NUM_SCENARIOS
            validation = validate(
                actor,
                store,
                train_engine,
                device,
                lambda_passenger,
                lambda_driver,
                validation_groups,
            )
            train_engine.num_scenarios = original_scenarios
            row.update(validation)
            score = float(validation["validation_adjusted_reward"])
            if score > best_validation:
                best_validation = score
                torch.save(
                    {
                        "actor": actor.state_dict(),
                        "critic1": critic1.state_dict(),
                        "critic2": critic2.state_dict(),
                        "numeric_dim": numeric_dim,
                        "region_count": region_count,
                        "lambda_passenger": lambda_passenger,
                        "lambda_driver": lambda_driver,
                        "sac_gamma": float(args.sac_gamma),
                        "config": {
                            "price_ratio_min": cfg.PRICE_RATIO_MIN,
                            "price_ratio_max": cfg.PRICE_RATIO_MAX,
                            "subsidy_max_ratio": cfg.SUBSIDY_MAX_RATIO,
                            "driver_gamma": cfg.DRIVER_GAMMA,
                        },
                    },
                    model_checkpoint,
                )
                print(f"保存新的最佳模型：validation_adjusted_reward={score:.4f}")
        history.append(row)

        if collection_step % 1_000 == 0:
            pd.DataFrame(history).to_csv(training_history_file, index=False)

    pd.DataFrame(history).to_csv(training_history_file, index=False)
    if not model_checkpoint.exists():
        raise RuntimeError("训练结束但没有生成模型检查点，请检查训练步数与 LEARNING_STARTS。")
    summary = {
        "best_validation_adjusted_reward": best_validation,
        "sac_gamma": float(args.sac_gamma),
        "model": str(model_checkpoint),
        "history": str(training_history_file),
        "elapsed_hours": (time.perf_counter() - run_started_at) / 3600.0,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
