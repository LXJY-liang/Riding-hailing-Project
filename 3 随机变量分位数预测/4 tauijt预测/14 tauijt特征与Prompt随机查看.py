#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""随机查看 tauijt 结构化特征及其实际输入 Qwen 的 Prompt。"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


CODE_DIR = Path(__file__).resolve().parent


def load_module(file_name: str, module_name: str):
    """从当前目录加载带空格和中文文件名的 Python 模块。"""

    file_path = CODE_DIR / file_name
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块：{file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


config = load_module("1 tauijt配置.py", "tauijt_prompt_view_config")
data_module = load_module("2 tauijt数据处理.py", "tauijt_prompt_view_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="随机抽取 tauijt 样本，依次显示结构化特征和 Prompt。"
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
        help="抽样范围，默认训练时间段。",
    )
    parser.add_argument(
        "--feature-path",
        type=Path,
        default=config.feature_data_path,
        help="tauijt 特征 parquet 路径。",
    )
    return parser.parse_args()


def filter_split(data: pd.DataFrame, split: str) -> pd.DataFrame:
    """按配置中的日期范围筛选数据。"""

    if split == "all":
        return data
    start = pd.Timestamp(getattr(config, f"{split}_start"))
    end = pd.Timestamp(getattr(config, f"{split}_end"))
    return data[
        data[config.time_column].between(start, end, inclusive="both")
    ]


def sample_rows(
    feature_path: Path,
    split: str,
    sample_count: int,
    seed: int,
) -> pd.DataFrame:
    """随机读取 row group，避免将完整特征文件载入内存。"""

    parquet_file = pq.ParquetFile(feature_path)
    rng = np.random.default_rng(seed)
    samples = []

    for row_group in rng.permutation(parquet_file.num_row_groups):
        data = parquet_file.read_row_group(
            int(row_group),
            columns=config.all_columns,
        ).to_pandas()
        data[config.time_column] = pd.to_datetime(data[config.time_column])
        data = filter_split(data, split)
        data = data[data[config.label_column].notna()]
        if data.empty:
            continue

        needed = sample_count - sum(len(frame) for frame in samples)
        take = min(needed, len(data))
        positions = rng.choice(len(data), size=take, replace=False)
        samples.append(data.iloc[positions].copy())
        if take == needed:
            break

    if not samples:
        raise ValueError("没有找到标签有效且符合日期范围的 tauijt 样本。")

    result = pd.concat(samples, ignore_index=True)
    if len(result) < sample_count:
        raise ValueError(
            f"只能找到{len(result)}条符合条件的样本，少于要求的{sample_count}条。"
        )
    return result


def format_value(value) -> str:
    """统一终端输出格式。"""

    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if pd.isna(value):
        return "NaN"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    return str(value)


def main() -> None:
    args = parse_args()
    if args.n < 1:
        raise ValueError("--n必须大于等于1。")

    feature_path = args.feature_path.expanduser().resolve()
    if not feature_path.is_file():
        raise FileNotFoundError(f"找不到 tauijt 特征文件：{feature_path}")

    samples = sample_rows(
        feature_path=feature_path,
        split=args.split,
        sample_count=args.n,
        seed=args.seed,
    )

    print(f"特征文件：{feature_path}")
    print(f"抽样范围：{args.split}")
    print(f"随机种子：{args.seed}")
    print(f"本次抽取样本数：{len(samples)}")

    print("\n" + "=" * 110)
    print("第一部分：随机样本的结构化特征")
    print("=" * 110)
    for display_index, (_, row) in enumerate(samples.iterrows(), start=1):
        print(f"\n样本 {display_index}")
        print("-" * 110)
        for column in config.all_columns:
            print(f"{column:<42} = {format_value(row[column])}")

    print("\n" + "=" * 110)
    print("第二部分：同一批样本实际输入 Qwen 的完整 Prompt")
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
