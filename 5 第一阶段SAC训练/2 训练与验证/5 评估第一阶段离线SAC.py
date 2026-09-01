# -*- coding: utf-8 -*-
"""在固定日期和固定随机种子下评估第一阶段离线SAC检查点。"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


cfg = load(HERE.parent / "1 训练前的准备" / "1 第一阶段SAC配置.py", "stage1_config")
env_module = load(HERE / "2 定义外生状态环境.py", "stage1_environment_eval")
net = load(HERE / "1 定义离线SAC网络.py", "stage1_networks_eval")


def seed_for_day(day: str, base: int) -> int:
    return int(base + pd.Timestamp(day).strftime("%Y%m%d")) % (2**32 - 1)


def files_for_split(split: str) -> list[Path]:
    result = []
    for path in sorted(cfg.EPISODE_DIR.glob("episode_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            if str(data["split"]) == split:
                result.append(path)
    return result


def checkpoint_for_run(run_name: str, checkpoint: str | None) -> Path:
    if checkpoint:
        path = Path(checkpoint).expanduser().resolve()
    else:
        path = cfg.MODEL_DIR / run_name / "best_offline_sac.pt"
        if not path.exists():
            path = cfg.MODEL_DIR / run_name / "latest_offline_sac.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def evaluate_checkpoint(checkpoint_path, split, scenario_seed, output_dir=None, save_policy=False):
    paths = files_for_split(split)
    if not paths:
        raise FileNotFoundError(f"没有{split}集episode。")
    available_dates = {path.stem.removeprefix("episode_") for path in paths}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    actor = net.Actor(checkpoint["state_dim"], checkpoint["region_count"]).to(device)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    totals = {name: 0.0 for name in (
        "profit", "potential", "passenger_accepted", "bilateral", "completed",
        "passenger_acceptance", "driver_single", "driver_final", "passenger_violation",
        "driver_violation", "price", "reference_price", "subsidy", "active_od",
        "passenger_active_od", "driver_active_od", "passenger_violation_od",
        "driver_violation_od", "hard_infeasible_od", "binding_groups", "groups",
    )}
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        episode = env_module.OfflineSACEpisode(
            path,
            seed_for_day(path.stem.removeprefix("episode_"), scenario_seed),
            available_dates,
        )
        policy_rows = []
        while episode.slot < cfg.STEPS_PER_DAY:
            groups = episode.build_groups()
            if not groups:
                episode.step([], [])
                continue
            lengths = [len(group["j"]) for group in groups]
            state = torch.as_tensor(np.concatenate([g["state"] for g in groups]), dtype=torch.float32, device=device)
            i = torch.as_tensor(np.concatenate([g["i"] for g in groups]), dtype=torch.long, device=device)
            j = torch.as_tensor(np.concatenate([g["j"] for g in groups]), dtype=torch.long, device=device)
            p_ref = torch.as_tensor(np.concatenate([g["p_ref"] for g in groups]), dtype=torch.float32, device=device)
            with torch.inference_mode():
                action = actor.deterministic(state, i, j, p_ref).cpu().numpy()
            results = episode.step(groups, list(np.split(action, np.cumsum(lengths)[:-1])))
            timestamp = pd.Timestamp(episode.date) + pd.Timedelta(minutes=15 * (episode.slot - 1))
            for result in results:
                demand = np.maximum(result["base_demand"], 0.0)
                passenger_accepted = demand * result["passenger_acceptance"]
                active = demand > 0
                totals["profit"] += float(result["profit"].sum())
                totals["potential"] += float(demand.sum())
                totals["passenger_accepted"] += float(passenger_accepted.sum())
                totals["bilateral"] += float(result["bilateral_accepted_demand"].sum())
                totals["completed"] += float(result["completed"].sum())
                totals["passenger_acceptance"] += float(np.sum(result["passenger_acceptance"] * demand))
                totals["driver_single"] += float(np.sum(result["driver_single"] * demand))
                totals["driver_final"] += float(np.sum(result["driver_final"] * demand))
                totals["passenger_violation"] += float(np.sum((result["g_passenger"] > 0) * demand))
                totals["driver_violation"] += float(np.sum((result["g_driver"] > 0) * demand))
                totals["price"] += float(np.sum(result["price"] * result["completed"]))
                totals["reference_price"] += float(np.sum(result["p_ref"] * result["completed"]))
                totals["subsidy"] += float(np.sum(result["subsidy"] * result["completed"]))
                totals["active_od"] += int(active.sum())
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
                totals["hard_infeasible_od"] += int(result["hard_infeasible"].sum())
                totals["groups"] += 1
                totals["binding_groups"] += int(result["supply_binding"])
                if save_policy:
                    n = len(result["j"])
                    policy_rows.append(pd.DataFrame({
                        "t": np.repeat(timestamp, n),
                        "i": episode.region_ids[result["i"]], "j": episode.region_ids[result["j"]],
                        "S_it_capacity": result["supply"], "potential_demand_ijt": demand,
                        "p_ref_ijt": result["p_ref"], "p_ijt": result["price"], "b_ijt": result["subsidy"],
                        "passenger_acceptance": result["passenger_acceptance"],
                        "driver_final_acceptance": result["driver_final"],
                        "bilateral_accepted_demand": result["bilateral_accepted_demand"],
                        "completed_orders": result["completed"], "profit_ijt": result["profit"],
                    }))
        if save_policy and output_dir is not None:
            pd.concat(policy_rows, ignore_index=True).to_parquet(
                output_dir / f"policy_{episode.date}.parquet", index=False, compression="snappy"
            )
    demand = max(totals["potential"], 1e-9)
    completed = max(totals["completed"], 1e-9)
    metrics = {
        "checkpoint": str(checkpoint_path), "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "split": split, "scenario_seed": int(scenario_seed), "episode_days": len(paths),
        "platform_total_profit": totals["profit"], "potential_demand": totals["potential"],
        "passenger_accepted_demand": totals["passenger_accepted"],
        "bilateral_accepted_demand": totals["bilateral"], "completed_orders": totals["completed"],
        "transaction_rate": totals["completed"] / demand,
        "supply_fulfillment_rate": totals["completed"] / max(totals["bilateral"], 1e-9),
        "passenger_acceptance_mean": totals["passenger_acceptance"] / demand,
        "driver_single_acceptance_mean": totals["driver_single"] / demand,
        "driver_final_acceptance_mean": totals["driver_final"] / demand,
        "passenger_violation_rate": totals["passenger_violation"] / demand,
        "driver_violation_rate": totals["driver_violation"] / demand,
        "passenger_violation_od_rate": totals["passenger_violation_od"] / max(totals["passenger_active_od"], 1),
        "driver_violation_od_rate": totals["driver_violation_od"] / max(totals["driver_active_od"], 1),
        "hard_infeasible_od_rate": totals["hard_infeasible_od"] / max(totals["passenger_active_od"], 1),
        "average_price": totals["price"] / completed,
        "average_reference_price": totals["reference_price"] / completed,
        "average_subsidy": totals["subsidy"] / completed,
        "total_subsidy": totals["subsidy"],
        "supply_binding_rate": totals["binding_groups"] / max(totals["groups"], 1),
        "vit_frozen": True, "exogenous_next_state": True,
    }
    if output_dir is not None:
        (output_dir / "evaluation_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default=cfg.DEFAULT_RUN_NAME)
    parser.add_argument("--split", choices=tuple(cfg.SPLITS), default="valid")
    parser.add_argument("--checkpoint")
    parser.add_argument("--scenario-seed", type=int, default=cfg.VALIDATION_SCENARIO_SEED)
    parser.add_argument("--no-policy", action="store_true")
    args = parser.parse_args()
    checkpoint = checkpoint_for_run(args.run_name, args.checkpoint)
    output = cfg.EVALUATION_DIR / args.run_name / args.split / checkpoint.stem
    metrics = evaluate_checkpoint(
        checkpoint, args.split, args.scenario_seed, output, not args.no_policy
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
