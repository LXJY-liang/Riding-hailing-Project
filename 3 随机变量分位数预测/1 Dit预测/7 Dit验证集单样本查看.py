#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
随机或按条件查看验证集 Dit 预测样本。

示例：
python "7 Dit验证集单样本查看.py"
python "7 Dit验证集单样本查看.py" --include-zero
python "7 Dit验证集单样本查看.py" --region-id 17031839100
python "7 Dit验证集单样本查看.py" --region-id 17031839100 --time-contains "2024-06-25 17:30"
python "7 Dit验证集单样本查看.py" --seed 7 --n 3 --save-csv output/analysis_12pct_monotonic/sample_view.csv
python "7 Dit验证集单样本查看.py" --summary-csv output/analysis_12pct_monotonic/quantile_prediction_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CODE_DIR = Path(__file__).resolve().parent
PREDICTION_PATH = CODE_DIR / "output" / "prediction" / "dit_valid_predictions.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="查看验证集中某个区域/时段的真实 Dit 与 101 维分位数预测。"
    )
    parser.add_argument(
        "--prediction-path",
        type=Path,
        default=PREDICTION_PATH,
        help="验证集预测 parquet 路径。",
    )
    parser.add_argument(
        "--region-id",
        type=str,
        default=None,
        help="指定 region_id；不填则从全部验证集随机抽样。",
    )
    parser.add_argument(
        "--time-contains",
        type=str,
        default=None,
        help="按时间字符串模糊筛选，例如 2024-06-25 或 17:30。",
    )
    parser.add_argument(
        "--positive-only",
        action="store_true",
        default=True,
        help="只从真实 D_it_label > 0 的样本中随机抽取；默认开启。",
    )
    parser.add_argument(
        "--include-zero",
        action="store_false",
        dest="positive_only",
        help="关闭非零筛选，允许从 D_it_label = 0 的样本中抽取。",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="抽取样本数量。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=56,
        help="随机种子。",
    )
    parser.add_argument(
        "--save-csv",
        type=Path,
        default=None,
        help="可选：把抽中的样本和101维分位数保存成CSV。",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="可选：导出每个分位数在全部筛选样本上的统计摘要。",
    )
    return parser.parse_args()


def get_quantile_columns(df: pd.DataFrame) -> list[str]:
    qcols = [col for col in df.columns if col.startswith("q_")]
    return sorted(qcols, key=lambda col: float(col.split("_", maxsplit=1)[1]))


def main() -> None:
    args = parse_args()

    df = pd.read_parquet(args.prediction_path)
    qcols = get_quantile_columns(df)
    filtered = df

    if args.region_id is not None:
        filtered = filtered[
            filtered["region_id"].astype(str) == str(args.region_id)
        ]

    if args.time_contains is not None:
        filtered = filtered[
            filtered["t"].astype(str).str.contains(
                args.time_contains,
                regex=False,
                na=False,
            )
        ]

    if args.positive_only:
        filtered = filtered[filtered["D_it_label"] > 0]

    if filtered.empty:
        raise ValueError("没有找到满足筛选条件的验证集样本。")

    if args.summary_csv is not None:
        values = filtered[qcols].to_numpy(dtype=float)
        summary = pd.DataFrame(
            {
                "quantile": qcols,
                "tau": [
                    float(col.split("_", maxsplit=1)[1])
                    for col in qcols
                ],
                "min": values.min(axis=0),
                "p01": np.quantile(values, 0.01, axis=0),
                "p05": np.quantile(values, 0.05, axis=0),
                "p25": np.quantile(values, 0.25, axis=0),
                "median": np.quantile(values, 0.50, axis=0),
                "mean": values.mean(axis=0),
                "p75": np.quantile(values, 0.75, axis=0),
                "p95": np.quantile(values, 0.95, axis=0),
                "p99": np.quantile(values, 0.99, axis=0),
                "max": values.max(axis=0),
                "std": values.std(axis=0),
            }
        )
        summary_path = args.summary_csv
        if not summary_path.is_absolute():
            summary_path = CODE_DIR / summary_path
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)
        print(f"分位数预测摘要已保存：{summary_path}")

    n = min(args.n, len(filtered))
    sample = filtered.sample(n=n, random_state=args.seed)

    print(f"预测文件：{args.prediction_path}")
    print(f"筛选后样本数：{len(filtered)}")
    print(f"本次展示样本数：{n}")
    print("=" * 100)

    for sample_index, (_, row) in enumerate(sample.iterrows(), start=1):
        values = row[qcols].astype(float)
        crossing_count = int((values.to_numpy()[1:] < values.to_numpy()[:-1]).sum())
        negative_count = int((values.to_numpy() < 0).sum())

        print(f"样本 {sample_index}")
        print(f"region_id = {row['region_id']}")
        print(f"t = {row['t']}")
        print(f"D_it_label = {row['D_it_label']}")
        print(f"q0.50 = {row['q_0.500']:.6f}")
        print(f"q0.90 = {row['q_0.900']:.6f}")
        print(f"q0.99 = {row['q_0.990']:.6f}")
        print(f"相邻分位数交叉次数 = {crossing_count}")
        print(f"负预测分位数数量 = {negative_count}")
        print("-" * 100)

        quantile_table = pd.DataFrame(
            {
                "quantile": qcols,
                "tau": [float(col.split("_", maxsplit=1)[1]) for col in qcols],
                "prediction": values.to_numpy(),
            }
        )
        print(quantile_table.to_string(index=False))
        print("=" * 100)

    if args.save_csv is not None:
        save_path = args.save_csv
        if not save_path.is_absolute():
            save_path = CODE_DIR / save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        sample[["region_id", "t", "D_it_label", *qcols]].to_csv(
            save_path,
            index=False,
        )
        print(f"已保存：{save_path}")


if __name__ == "__main__":
    main()
