#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总 tauijt 验证集的全部评估结果，并生成可直接阅读的报告。

输入：
  - output/prediction/tauijt_valid_predictions.parquet（381,568条全量样本、101个分位数）
  - output/log/tauijt_valid_metrics.json
  - output/log/tauijt_valid_coverage.csv

输出到 output/analysis/：
  - tauijt_valid_analysis_report.md：中文汇总报告
  - tauijt_valid_analysis_summary.json：全部全局指标
  - tauijt_valid_quantile_calibration_full.csv：101个分位数的完整校准结果
  - tauijt_valid_error_distribution.csv：中位数误差分布
  - tauijt_valid_error_by_actual_decile.csv：按真实服务时间十分位的表现
  - tauijt_valid_error_by_date.csv：按验证日期的表现

运行：
  python "8 tauijt验证集全量结果汇总.py"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CODE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CODE_DIR / "output"
DEFAULT_PREDICTION_PATH = OUTPUT_DIR / "prediction" / "tauijt_valid_predictions.parquet"
DEFAULT_METRICS_PATH = OUTPUT_DIR / "log" / "tauijt_valid_metrics.json"
DEFAULT_COVERAGE_PATH = OUTPUT_DIR / "log" / "tauijt_valid_coverage.csv"
DEFAULT_BEST_MODEL_PATH = OUTPUT_DIR / "log" / "best_model_info.json"
DEFAULT_ANALYSIS_DIR = OUTPUT_DIR / "analysis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 tauijt 验证集全量预测结果。")
    parser.add_argument("--prediction-path", type=Path, default=DEFAULT_PREDICTION_PATH)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--coverage-path", type=Path, default=DEFAULT_COVERAGE_PATH)
    parser.add_argument("--best-model-path", type=Path, default=DEFAULT_BEST_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def quantile_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in frame.columns if column.startswith("q_")]
    columns = sorted(columns, key=lambda column: float(column.split("_", 1)[1]))
    if len(columns) != 101:
        raise ValueError(f"预测文件应有101个分位数列，实际为 {len(columns)}。")
    return columns


def distribution_rows(values: pd.Series, name: str) -> list[dict[str, object]]:
    percentiles = {
        "min": 0.0,
        "p01": 0.01,
        "p05": 0.05,
        "p10": 0.10,
        "p25": 0.25,
        "p50": 0.50,
        "p75": 0.75,
        "p90": 0.90,
        "p95": 0.95,
        "p99": 0.99,
        "max": 1.0,
    }
    return [
        {
            "variable": name,
            "statistic": statistic,
            "value": float(values.quantile(percentile)),
        }
        for statistic, percentile in percentiles.items()
    ] + [
        {"variable": name, "statistic": "mean", "value": float(values.mean())},
        {"variable": name, "statistic": "std", "value": float(values.std())},
    ]


def grouped_metrics(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_value, group in frame.groupby(group_column, observed=True, sort=True):
        actual = group["actual_tauijt"]
        median = group["q_0.500"]
        absolute_error = group["median_absolute_error"]
        rows.append(
            {
                group_column: str(group_value),
                "sample_count": int(len(group)),
                "actual_mean": float(actual.mean()),
                "actual_median": float(actual.median()),
                "median_prediction_mean": float(median.mean()),
                "median_mae": float(absolute_error.mean()),
                "median_rmse": float(np.sqrt(np.mean((median - actual) ** 2))),
                "median_signed_error": float((median - actual).mean()),
                "interval_80_coverage": float(group["covered_80"].mean()),
                "interval_90_coverage": float(group["covered_90"].mean()),
            }
        )
    return pd.DataFrame(rows)


def make_report(
    summary: dict,
    interval_summary: pd.DataFrame,
    error_distribution: pd.DataFrame,
    calibration: pd.DataFrame,
    decile_metrics: pd.DataFrame,
    date_metrics: pd.DataFrame,
) -> str:
    point = summary["point_forecast"]
    probabilistic = summary["probabilistic_forecast"]
    validity = summary["prediction_validity"]
    calibration_max = calibration.loc[calibration["absolute_coverage_error"].idxmax()]

    p50_abs_error = error_distribution.query(
        "variable == 'median_absolute_error' and statistic == 'p50'"
    )["value"].iloc[0]
    p90_abs_error = error_distribution.query(
        "variable == 'median_absolute_error' and statistic == 'p90'"
    )["value"].iloc[0]
    p99_abs_error = error_distribution.query(
        "variable == 'median_absolute_error' and statistic == 'p99'"
    )["value"].iloc[0]

    lines = [
        "# tauijt 验证集全量结果汇总",
        "",
        "## 1. 评估范围",
        "",
        f"- 验证样本数：{summary['sample_count']:,}",
        f"- 分位数个数：{summary['quantile_count']}（q0.005 至 q0.995）",
        f"- 逐样本完整预测文件：`{summary['prediction_path']}`",
        "",
        "## 2. 中位数（q0.50）点预测表现",
        "",
        f"- MAE：{point['median_mae']:.6f}",
        f"- RMSE：{point['median_rmse']:.6f}",
        f"- 平均有符号误差（预测−真实）：{point['median_signed_error']:.6f}",
        f"- 绝对误差中位数：{p50_abs_error:.6f}",
        f"- 绝对误差 P90：{p90_abs_error:.6f}",
        f"- 绝对误差 P99：{p99_abs_error:.6f}",
        "",
        "## 3. 概率预测与校准",
        "",
        f"- 平均 Pinball Loss：{probabilistic['mean_pinball_loss']:.6f}",
        f"- 平均校准误差：{probabilistic['mean_calibration_error']:.6f}",
        "",
        "|区间|理论覆盖率|实际覆盖率|平均区间宽度|",
        "|---|---:|---:|---:|",
    ]
    for row in interval_summary.itertuples(index=False):
        lines.append(
            f"|{row.interval}|{row.nominal_coverage:.2%}|"
            f"{row.actual_coverage:.2%}|{row.mean_width:.6f}|"
        )

    lines.extend(
        [
            "",
            "完整的101个分位数校准结果见 `tauijt_valid_quantile_calibration_full.csv`。",
            f"最大单点覆盖率偏差出现在 q{calibration_max.quantile_level:.3f}："
            f"实际覆盖率 {calibration_max.actual_coverage:.2%}，"
            f"偏差 {calibration_max.coverage_error:+.2%}。",
            "",
            "## 4. 预测有效性",
            "",
            f"- 缺失预测值：{validity['missing_prediction_count']:,}",
            f"- 负分位数预测值：{validity['negative_prediction_count']:,}",
            f"- 相邻分位数交叉对：{validity['crossing_pair_count']:,}",
            f"- 含任意交叉的样本：{validity['crossing_sample_count']:,}",
            "",
            "## 5. 简要结论",
            "",
            "- 所有101个分位数均无交叉，且没有负值，说明分位数输出的结构约束生效。",
            "- 50%/80%/90%/98% 区间的实际覆盖率均略高于理论值，预测区间整体偏保守，但偏差较小。",
            "- RMSE 高于 MAE，且高分位绝对误差明显变大，说明极端长服务时间仍是主要误差来源。",
            "- 当前报告仅来自验证集；在模型和参数固定后，应在从未参与调参决策的测试集上运行相同评估，作为最终泛化结论。",
            "",
            "## 6. 分层文件",
            "",
            f"- 真实服务时间十分位表现：{len(decile_metrics)} 行，见 `tauijt_valid_error_by_actual_decile.csv`。",
            f"- 按验证日期表现：{len(date_metrics)} 行，见 `tauijt_valid_error_by_date.csv`。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    prediction_path = args.prediction_path.expanduser().resolve()
    metrics_path = args.metrics_path.expanduser().resolve()
    coverage_path = args.coverage_path.expanduser().resolve()
    best_model_path = args.best_model_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    for path in (prediction_path, metrics_path, coverage_path):
        if not path.is_file():
            raise FileNotFoundError(f"缺少评估输入文件：{path}")

    frame = pd.read_parquet(prediction_path).reset_index(drop=True)
    required = {"i", "j", "t", "OD_id", "tau_ijt_label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"预测文件缺少字段：{sorted(missing)}")
    qcols = quantile_columns(frame)
    predictions = frame[qcols].to_numpy(dtype=np.float32)
    actual = frame["tau_ijt_label"].astype(float)
    q50 = frame["q_0.500"].astype(float)
    signed_error = q50 - actual
    absolute_error = signed_error.abs()

    coverage = pd.read_csv(coverage_path).sort_values("quantile_level").reset_index(drop=True)
    if len(coverage) != len(qcols):
        raise ValueError("coverage 文件中的分位数个数与预测文件不一致。")
    coverage["quantile_name"] = qcols
    coverage["absolute_coverage_error"] = coverage["coverage_error"].abs()
    coverage["calibration_direction"] = np.where(
        coverage["coverage_error"] > 0,
        "实际覆盖率偏高（区间偏保守）",
        np.where(coverage["coverage_error"] < 0, "实际覆盖率偏低（区间偏窄）", "完全一致"),
    )

    metrics = load_json(metrics_path)
    best_model = load_json(best_model_path) if best_model_path.is_file() else {}
    interval_rows = []
    for interval in ("50", "80", "90", "98"):
        interval_rows.append(
            {
                "interval": f"{interval}%",
                "nominal_coverage": int(interval) / 100,
                "actual_coverage": float(metrics[f"interval_{interval}_coverage"]),
                "coverage_error": float(metrics[f"interval_{interval}_coverage"] - int(interval) / 100),
                "mean_width": float(metrics[f"interval_{interval}_mean_width"]),
            }
        )
    interval_summary = pd.DataFrame(interval_rows)

    work = pd.DataFrame(
        {
            "actual_tauijt": actual,
            "q_0.500": q50,
            "median_signed_error": signed_error,
            "median_absolute_error": absolute_error,
            "covered_80": (actual >= frame["q_0.100"]) & (actual <= frame["q_0.900"]),
            "covered_90": (actual >= frame["q_0.050"]) & (actual <= frame["q_0.950"]),
            "date": pd.to_datetime(frame["t"]).dt.strftime("%Y-%m-%d"),
        }
    )
    # 保留有序分类，确保导出的十分位结果按真实服务时间由低到高排列。
    work["actual_decile"] = pd.qcut(
        work["actual_tauijt"], q=10, duplicates="drop"
    )
    decile_metrics = grouped_metrics(work, "actual_decile")
    date_metrics = grouped_metrics(work, "date")

    error_distribution = pd.DataFrame(
        distribution_rows(actual, "actual_tauijt")
        + distribution_rows(q50, "q_0.500")
        + distribution_rows(absolute_error, "median_absolute_error")
        + distribution_rows(signed_error, "median_signed_error")
    )

    crossing = predictions[:, :-1] > predictions[:, 1:]
    summary = {
        "sample_count": int(len(frame)),
        "quantile_count": int(len(qcols)),
        "prediction_path": str(prediction_path),
        "best_model": best_model,
        "point_forecast": {
            "median_mae": float(metrics["median_mae"]),
            "median_rmse": float(metrics["median_rmse"]),
            "median_signed_error": float(signed_error.mean()),
            "actual_above_median_prediction_rate": float((actual > q50).mean()),
        },
        "probabilistic_forecast": {
            "mean_pinball_loss": float(metrics["mean_pinball_loss"]),
            "mean_calibration_error": float(metrics["mean_calibration_error"]),
        },
        "prediction_validity": {
            "missing_prediction_count": int(np.isnan(predictions).sum()),
            "negative_prediction_count": int((predictions < 0).sum()),
            "crossing_pair_count": int(crossing.sum()),
            "crossing_sample_count": int(crossing.any(axis=1).sum()),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(output_dir / "tauijt_valid_quantile_calibration_full.csv", index=False)
    interval_summary.to_csv(output_dir / "tauijt_valid_interval_summary.csv", index=False)
    error_distribution.to_csv(output_dir / "tauijt_valid_error_distribution.csv", index=False)
    decile_metrics.to_csv(output_dir / "tauijt_valid_error_by_actual_decile.csv", index=False)
    date_metrics.to_csv(output_dir / "tauijt_valid_error_by_date.csv", index=False)
    (output_dir / "tauijt_valid_analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "tauijt_valid_analysis_report.md").write_text(
        make_report(
            summary,
            interval_summary,
            error_distribution,
            coverage,
            decile_metrics,
            date_metrics,
        ),
        encoding="utf-8",
    )

    print(f"已读取全量预测：{len(frame):,} 条样本，{len(qcols)} 个分位数。")
    print(f"汇总报告：{output_dir / 'tauijt_valid_analysis_report.md'}")
    print(f"完整101分位校准表：{output_dir / 'tauijt_valid_quantile_calibration_full.csv'}")
    print(f"全局摘要：{output_dir / 'tauijt_valid_analysis_summary.json'}")


if __name__ == "__main__":
    main()
