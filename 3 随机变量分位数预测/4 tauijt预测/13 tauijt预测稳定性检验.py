#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""比较两次或多次 tauijt 验证集预测，量化推理/训练稳定性。

用法示例：
    python "13 tauijt预测稳定性检验.py" \
      output/stability_inputs/tauijt_valid_predictions_seed42.parquet \
      output/stability_inputs/tauijt_valid_predictions_seed43.parquet \
      output/stability_inputs/tauijt_valid_predictions_seed44.parquet

若比较同一模型的重复推理，差异应为0或仅有极小浮点误差；若比较不同随机种子
训练出的模型，应重点查看各项验证指标的均值、标准差和逐样本分位数差异。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CODE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = CODE_DIR / "output" / "analysis" / "stability"
KEY_COLUMNS = ["i", "j", "t", "OD_id"]
LABEL_COLUMN = "tau_ijt_label"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="比较多次 tauijt 验证集预测的稳定性。")
    parser.add_argument(
        "prediction_files",
        nargs="+",
        type=Path,
        help="至少两个 tauijt_valid_predictions*.parquet 文件。",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="判断同一模型重复推理是否近似一致的绝对误差阈值。",
    )
    return parser.parse_args()


def get_quantile_columns(frame: pd.DataFrame) -> list[str]:
    columns = sorted(
        [column for column in frame.columns if column.startswith("q_")],
        key=lambda column: float(column.split("_", 1)[1]),
    )
    if len(columns) != 101:
        raise ValueError(f"应有101个分位数列，实际为 {len(columns)}。")
    return columns


def calculate_run_metrics(
    name: str,
    labels: np.ndarray,
    predictions: np.ndarray,
    levels: np.ndarray,
) -> dict[str, object]:
    median_index = int(np.argmin(np.abs(levels - 0.5)))
    median = predictions[:, median_index]
    errors = labels[:, None] - predictions
    pinball = np.maximum(
        levels[None, :] * errors,
        (levels[None, :] - 1.0) * errors,
    )
    calibration = np.mean(labels[:, None] <= predictions, axis=0)
    crossing = predictions[:, :-1] > predictions[:, 1:]

    def interval_coverage(lower_level: float, upper_level: float) -> float:
        lower_index = int(np.argmin(np.abs(levels - lower_level)))
        upper_index = int(np.argmin(np.abs(levels - upper_level)))
        return float(
            np.mean(
                (labels >= predictions[:, lower_index])
                & (labels <= predictions[:, upper_index])
            )
        )

    return {
        "run": name,
        "sample_count": int(len(labels)),
        "mean_pinball_loss": float(pinball.mean()),
        "median_mae": float(np.mean(np.abs(labels - median))),
        "median_rmse": float(np.sqrt(np.mean((labels - median) ** 2))),
        "median_signed_error": float(np.mean(median - labels)),
        "mean_calibration_error": float(np.mean(np.abs(calibration - levels))),
        "interval_80_coverage": interval_coverage(0.10, 0.90),
        "interval_90_coverage": interval_coverage(0.05, 0.95),
        "crossing_pair_rate": float(crossing.mean()),
        "crossing_sample_rate": float(crossing.any(axis=1).mean()),
    }


def main() -> None:
    args = parse_args()
    if len(args.prediction_files) < 2:
        raise ValueError("至少需要两个预测文件才能检验稳定性。")

    paths = [path.expanduser().resolve() for path in args.prediction_files]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("找不到预测文件：\n" + "\n".join(missing))

    baseline = pd.read_parquet(paths[0]).reset_index(drop=True)
    qcols = get_quantile_columns(baseline)
    required = set(KEY_COLUMNS + [LABEL_COLUMN] + qcols)
    missing_columns = required - set(baseline.columns)
    if missing_columns:
        raise ValueError(f"基准预测文件缺少字段：{sorted(missing_columns)}")

    levels = np.asarray([float(column.split("_", 1)[1]) for column in qcols], dtype=np.float32)
    labels = baseline[LABEL_COLUMN].to_numpy(dtype=np.float32)
    baseline_predictions = baseline[qcols].to_numpy(dtype=np.float32)
    baseline_keys = baseline[KEY_COLUMNS].astype(str)

    run_metrics = [
        calculate_run_metrics(paths[0].stem, labels, baseline_predictions, levels)
    ]
    pairwise_rows: list[dict[str, object]] = []

    for path in paths[1:]:
        current = pd.read_parquet(path).reset_index(drop=True)
        current_qcols = get_quantile_columns(current)
        if current_qcols != qcols:
            raise ValueError(f"{path.name} 的分位数列与基准文件不一致。")
        if len(current) != len(baseline):
            raise ValueError(f"{path.name} 的样本数与基准文件不一致。")
        if not current[KEY_COLUMNS].astype(str).equals(baseline_keys):
            raise ValueError(f"{path.name} 的样本顺序或 i/j/t/OD_id 与基准文件不一致。")

        current_labels = current[LABEL_COLUMN].to_numpy(dtype=np.float32)
        if not np.array_equal(current_labels, labels, equal_nan=True):
            raise ValueError(f"{path.name} 的真实标签与基准文件不一致。")

        current_predictions = current[qcols].to_numpy(dtype=np.float32)
        run_metrics.append(
            calculate_run_metrics(path.stem, labels, current_predictions, levels)
        )

        absolute_difference = np.abs(current_predictions - baseline_predictions)
        median_difference = absolute_difference[:, int(np.argmin(np.abs(levels - 0.5)))]
        pairwise_rows.append(
            {
                "baseline_run": paths[0].stem,
                "compared_run": path.stem,
                "mean_absolute_quantile_difference": float(absolute_difference.mean()),
                "p99_absolute_quantile_difference": float(np.quantile(absolute_difference, 0.99)),
                "max_absolute_quantile_difference": float(absolute_difference.max()),
                "mean_absolute_q50_difference": float(median_difference.mean()),
                "max_absolute_q50_difference": float(median_difference.max()),
                "exact_equal_value_rate": float((absolute_difference == 0).mean()),
                "all_values_within_atol": bool(np.all(absolute_difference <= args.atol)),
                "atol": float(args.atol),
            }
        )

    run_metrics_frame = pd.DataFrame(run_metrics)
    numeric_metric_columns = [
        column
        for column in run_metrics_frame.columns
        if column not in {"run", "sample_count"}
    ]
    aggregate_rows = []
    for column in numeric_metric_columns:
        values = run_metrics_frame[column].astype(float)
        aggregate_rows.append(
            {
                "metric": column,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "min": float(values.min()),
                "max": float(values.max()),
                "range": float(values.max() - values.min()),
            }
        )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_metrics_frame.to_csv(output_dir / "tauijt_stability_run_metrics.csv", index=False)
    pd.DataFrame(pairwise_rows).to_csv(
        output_dir / "tauijt_stability_pairwise_prediction_differences.csv",
        index=False,
    )
    pd.DataFrame(aggregate_rows).to_csv(
        output_dir / "tauijt_stability_metric_summary.csv",
        index=False,
    )
    summary = {
        "prediction_files": [str(path) for path in paths],
        "baseline": paths[0].stem,
        "run_count": len(paths),
        "sample_count": int(len(labels)),
        "quantile_count": len(qcols),
        "atol": float(args.atol),
        "pairwise_comparisons": pairwise_rows,
    }
    (output_dir / "tauijt_stability_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("各次运行指标：")
    print(run_metrics_frame.to_string(index=False))
    print("\n相对第一份预测的逐值差异：")
    print(pd.DataFrame(pairwise_rows).to_string(index=False))
    print(f"\n稳定性结果已保存：{output_dir}")


if __name__ == "__main__":
    main()
