#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""随机查看Dit原始特征及其实际输入Qwen的完整Prompt。"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pandas as pd


CODE_DIR = Path(__file__).resolve().parent


def load_module(file_name: str, module_name: str):
    file_path = CODE_DIR / file_name
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"无法加载模块：{file_path}")
    spec.loader.exec_module(module)
    return module


config = load_module("1 Dit配置.py", "dit_prompt_view_config")
data_module = load_module("2 Dit数据处理.py", "dit_prompt_view_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="随机抽取Dit样本，先显示结构化特征，再显示实际Prompt。"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=2,
        help="随机抽取的样本数，默认2条。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，默认42。",
    )
    parser.add_argument(
        "--split",
        choices=("all", "train", "valid", "test"),
        default="train",
        help="抽样范围，默认从训练时间段抽取。",
    )
    parser.add_argument(
        "--positive-only",
        action="store_true",
        help="只从真实D_it_label大于0的样本中抽取。",
    )
    parser.add_argument(
        "--feature-path",
        type=Path,
        default=config.feature_data_path,
        help="完整Dit特征Parquet路径。",
    )
    return parser.parse_args()


def filter_split(data: pd.DataFrame, split: str) -> pd.DataFrame:
    if split == "all":
        return data

    start = pd.Timestamp(getattr(config, f"{split}_start"))
    end = pd.Timestamp(getattr(config, f"{split}_end"))
    return data[data[config.time_column].between(start, end, inclusive="both")]


def format_value(value) -> str:
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main() -> None:
    args = parse_args()
    if args.n < 1:
        raise ValueError("--n必须大于等于1。")
    if not args.feature_path.exists():
        raise FileNotFoundError(f"找不到Dit特征文件：{args.feature_path}")

    required_columns = list(config.all_columns)
    data = pd.read_parquet(args.feature_path, columns=required_columns)
    data[config.time_column] = pd.to_datetime(data[config.time_column])
    data[config.region_column] = data[config.region_column].astype(str)
    data = filter_split(data, args.split)

    if args.positive_only:
        data = data[data[config.label_column] > 0]
    if data.empty:
        raise ValueError("筛选后没有可抽取的样本。")

    sample_count = min(args.n, len(data))
    samples = data.sample(n=sample_count, random_state=args.seed).copy()

    print(f"特征文件：{args.feature_path}")
    print(f"抽样范围：{args.split}")
    print(f"筛选后样本数：{len(data)}")
    print(f"随机种子：{args.seed}")
    print(f"本次抽取样本数：{sample_count}")

    print("\n" + "=" * 110)
    print("第一部分：随机样本的结构化特征")
    print("=" * 110)
    for display_index, (_, row) in enumerate(samples.iterrows(), start=1):
        print(f"\n样本 {display_index}")
        print("-" * 110)
        for column in required_columns:
            print(f"{column:<42} = {format_value(row[column])}")

    print("\n" + "=" * 110)
    print("第二部分：同一批样本实际输入Qwen的完整Prompt")
    print("=" * 110)
    for display_index, (_, row) in enumerate(samples.iterrows(), start=1):
        prompt = data_module.build_text(row.to_dict())
        print(f"\n样本 {display_index} Prompt")
        print("-" * 110)
        print(prompt)
        print("-" * 110)
        print(f"Prompt字符数：{len(prompt)}")


if __name__ == "__main__":
    main()
