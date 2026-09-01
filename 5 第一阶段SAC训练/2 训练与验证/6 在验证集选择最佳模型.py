# -*- coding: utf-8 -*-
"""冻结评估所有epoch，并在验证集选择约束优先、利润最高的模型。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


cfg = load(HERE.parent / "1 训练前的准备" / "1 第一阶段SAC配置.py", "stage1_config")
evaluation = load(HERE / "5 评估第一阶段离线SAC.py", "stage1_evaluation")


def key(row, tolerance):
    passenger = float(row["passenger_violation_od_rate"])
    driver = float(row["driver_violation_od_rate"])
    worst = max(passenger, driver)
    feasible = passenger <= tolerance and driver <= tolerance
    return (
        0 if feasible else 1,
        -float(row["platform_total_profit"]) if feasible else worst,
        passenger + driver,
        int(row["checkpoint_epoch"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default=cfg.DEFAULT_RUN_NAME)
    parser.add_argument("--scenario-seed", type=int, default=cfg.VALIDATION_SCENARIO_SEED)
    parser.add_argument("--violation-tolerance", type=float, default=cfg.VALIDATION_VIOLATION_TOLERANCE)
    args = parser.parse_args()
    model_dir = cfg.MODEL_DIR / args.run_name
    checkpoints = sorted(model_dir.glob("epoch_*_offline_sac.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"没有epoch检查点：{model_dir}")
    rows = [
        evaluation.evaluate_checkpoint(path, "valid", args.scenario_seed, None, False)
        for path in checkpoints
    ]
    best = min(rows, key=lambda row: key(row, args.violation_tolerance))
    target = model_dir / "best_offline_sac.pt"
    shutil.copy2(best["checkpoint"], target)
    output = cfg.EVALUATION_DIR / args.run_name / "valid"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("checkpoint_epoch").to_csv(
        output / "checkpoint_selection.csv", index=False
    )
    summary = {
        "best_epoch": int(best["checkpoint_epoch"]), "best_checkpoint": str(target),
        "scenario_seed": args.scenario_seed, "best_metrics": best,
    }
    (output / "best_checkpoint.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    final = evaluation.evaluate_checkpoint(target, "valid", args.scenario_seed, output / "best", True)
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
