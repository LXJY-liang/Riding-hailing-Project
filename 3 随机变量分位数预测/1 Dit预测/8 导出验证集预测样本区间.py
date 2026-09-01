#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按行号区间导出验证集 Dit 预测样本，方便在 PyCharm 或表格软件中查看。
默认只导出关键分位数，而不是全部101维分位数。

默认导出第100条到第2100条。这里的行号按人类习惯从1开始计数，
因此 --start 100 --end 2100 会导出第100、101、...、2100条。

示例：
python "8 导出验证集预测样本区间.py"
python "8 导出验证集预测样本区间.py" --start 500 --end 1500
python "8 导出验证集预测样本区间.py" --start 100 --end 2100 --positive-only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CODE_DIR = Path(__file__).resolve().parent
PREDICTION_PATH = CODE_DIR / "output" / "prediction" / "dit_valid_predictions.parquet"
KEY_QUANTILE_COLUMNS = [
    "q_0.010",
    "q_0.100",
    "q_0.200",
    "q_0.300",
    "q_0.400",
    "q_0.500",
    "q_0.600",
    "q_0.700",
    "q_0.800",
    "q_0.900",
    "q_0.995",
]
DEFAULT_OUTPUT_PATH = (
    CODE_DIR
    / "output"
    / "analysis_12pct_monotonic"
    / "dit_valid_predictions_rows_100_2100_key_quantiles.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按行号区间导出验证集 Dit 预测样本。"
    )
    parser.add_argument(
        "--prediction-path",
        type=Path,
        default=PREDICTION_PATH,
        help="验证集预测 parquet 文件路径。",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=36230,
        help="起始行号，从1开始计数，包含该行。",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=37230,
        help="结束行号，从1开始计数，包含该行。",
    )
    parser.add_argument(
        "--positive-only",
        action="store_true",
        help="先筛选真实 D_it_label > 0，再按行号区间导出。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="导出的 CSV 文件路径。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.start < 1:
        raise ValueError("--start 必须大于等于 1。")
    if args.end < args.start:
        raise ValueError("--end 必须大于等于 --start。")

    df = pd.read_parquet(args.prediction_path)

    if args.positive_only:
        df = df[df["D_it_label"] > 0].reset_index(drop=True)

    start_index = args.start - 1
    end_index = args.end
    subset = df.iloc[start_index:end_index].copy()

    if subset.empty:
        raise ValueError(
            f"导出结果为空。筛选后总样本数={len(df)}，"
            f"请求行号区间={args.start}-{args.end}。"
        )

    subset.insert(0, "source_row_number_1_based", range(args.start, args.start + len(subset)))
    missing_quantiles = [
        col for col in KEY_QUANTILE_COLUMNS
        if col not in subset.columns
    ]
    if missing_quantiles:
        raise KeyError(f"预测文件缺少关键分位数列：{missing_quantiles}")

    output_columns = [
        "source_row_number_1_based",
        "region_id",
        "t",
        "D_it_label",
        *KEY_QUANTILE_COLUMNS,
    ]
    subset = subset[output_columns]

    output_path = args.output
    if not output_path.is_absolute():
        output_path = CODE_DIR / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(output_path, index=False)

    print(f"预测文件：{args.prediction_path}")
    print(f"筛选后总样本数：{len(df)}")
    print(f"导出行号区间：{args.start}-{args.start + len(subset) - 1}")
    print(f"导出样本数：{len(subset)}")
    print(f"真实 D_it_label > 0 样本数：{int((subset['D_it_label'] > 0).sum())}")
    print(f"已保存：{output_path}")


if __name__ == "__main__":
    main()
