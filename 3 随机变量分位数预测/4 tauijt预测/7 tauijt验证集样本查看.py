#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""查看 tauijt 验证集的随机样本与指定样本。

示例：
    # 固定随机种子，随机展示两个样本，并导出完整的101个分位数
    python "7 tauijt验证集样本查看.py"

    # 随机展示两个样本，同时查看第 93921 行（从0开始计数）
    python "7 tauijt验证集样本查看.py" --row-index 93921

    # 通过订单的 i、j、时间精确指定一个样本
    python "7 tauijt验证集样本查看.py" \\
      --i 17031081402 --j 17031330100 --time "2024-06-26 17:30:00"

    # 仅按 OD_id 和时间指定一个样本，并在终端逐个显示全部101个分位数
    python "7 tauijt验证集样本查看.py" \\
      --od-id "17031081402_17031330100" --time "2024-06-26 17:30:00" \\
      --show-all-quantiles
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CODE_DIR = Path(__file__).resolve().parent
DEFAULT_PREDICTION_PATH = (
    CODE_DIR / "output" / "prediction" / "tauijt_valid_predictions.parquet"
)
DEFAULT_OUTPUT_DIR = CODE_DIR / "output" / "analysis"
IDENTIFIER_COLUMNS = ["i", "j", "t", "OD_id"]
LABEL_COLUMN = "tau_ijt_label"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "从 tauijt 验证集预测结果中随机查看两个样本，"
            "并可额外查看一个指定样本。"
        )
    )
    parser.add_argument(
        "--prediction-path",
        type=Path,
        default=DEFAULT_PREDICTION_PATH,
        help="验证集预测 parquet 的路径。",
    )
    parser.add_argument(
        "--random-n",
        type=int,
        default=2,
        help="随机展示的样本数量；默认 2。设为 0 可关闭随机抽样。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子；默认 42，方便复现同一批随机样本。",
    )
    parser.add_argument(
        "--row-index",
        type=int,
        default=None,
        help="指定样本的行位置（从 0 开始）；不能与 i/j/time/od-id 联用。",
    )
    parser.add_argument("--i", type=str, default=None, help="指定起点网格 i。")
    parser.add_argument("--j", type=str, default=None, help="指定终点网格 j。")
    parser.add_argument(
        "--time",
        type=str,
        default=None,
        help='指定时间 t，例如 "2024-06-26 17:30:00"。',
    )
    parser.add_argument("--od-id", type=str, default=None, help="指定 OD_id。")
    parser.add_argument(
        "--show-all-quantiles",
        action="store_true",
        help="在终端逐行输出每个样本的全部101个分位数。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="保存样本 CSV 的目录。",
    )
    return parser.parse_args()


def get_quantile_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in frame.columns if column.startswith("q_")]
    if len(columns) != 101:
        raise ValueError(f"应有101个分位数列，实际找到 {len(columns)} 个。")
    return sorted(columns, key=lambda column: float(column.split("_", 1)[1]))


def select_specified_sample(
    frame: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame | None:
    """按行位置，或按 i/j/t/OD_id 联合条件精确选出一个样本。"""

    filters_supplied = any(
        value is not None for value in (args.i, args.j, args.time, args.od_id)
    )
    if args.row_index is not None and filters_supplied:
        raise ValueError("--row-index 不能与 --i/--j/--time/--od-id 联用。")

    if args.row_index is not None:
        if not 0 <= args.row_index < len(frame):
            raise IndexError(
                f"--row-index 必须在 0 到 {len(frame) - 1} 之间，"
                f"当前为 {args.row_index}。"
            )
        return frame.iloc[[args.row_index]].copy()

    if not filters_supplied:
        return None

    mask = pd.Series(True, index=frame.index)
    for column, value in (
        ("i", args.i),
        ("j", args.j),
        ("t", args.time),
        ("OD_id", args.od_id),
    ):
        if value is not None:
            mask &= frame[column].astype(str).eq(str(value))

    selected = frame.loc[mask].copy()
    if selected.empty:
        raise ValueError("没有找到符合指定条件的样本，请核对 i、j、t、OD_id。")
    if len(selected) > 1:
        positions = frame.index.get_indexer(selected.index)[:10].tolist()
        raise ValueError(
            f"指定条件匹配到 {len(selected)} 条样本，不足以唯一定位。"
            f"请补充条件，或改用 --row-index；前10个行位置：{positions}。"
        )
    return selected


def build_summary(row: pd.Series, qcols: list[str], row_index: int) -> dict[str, object]:
    values = row[qcols].to_numpy(dtype=float)
    actual = float(row[LABEL_COLUMN])
    q50 = float(row["q_0.500"])

    return {
        "row_index": row_index,
        **{column: row[column] for column in IDENTIFIER_COLUMNS},
        "actual_tauijt": actual,
        "q_0.050": float(row["q_0.050"]),
        "q_0.100": float(row["q_0.100"]),
        "q_0.250": float(row["q_0.250"]),
        "q_0.500": q50,
        "q_0.750": float(row["q_0.750"]),
        "q_0.900": float(row["q_0.900"]),
        "q_0.950": float(row["q_0.950"]),
        "q_0.990": float(row["q_0.990"]),
        "median_signed_error": q50 - actual,
        "median_absolute_error": abs(q50 - actual),
        "covered_by_80pct_interval": bool(
            row["q_0.100"] <= actual <= row["q_0.900"]
        ),
        "covered_by_90pct_interval": bool(
            row["q_0.050"] <= actual <= row["q_0.950"]
        ),
        "quantile_crossing_count": int(np.sum(values[:-1] > values[1:])),
    }


def print_sample(
    title: str,
    row: pd.Series,
    qcols: list[str],
    row_index: int,
    show_all_quantiles: bool,
) -> None:
    summary = pd.Series(build_summary(row, qcols, row_index))
    print("=" * 100)
    print(title)
    print(summary.to_string())

    if show_all_quantiles:
        table = pd.DataFrame(
            {
                "quantile": [float(column.split("_", 1)[1]) for column in qcols],
                "prediction": row[qcols].to_numpy(dtype=float),
            }
        )
        print("\n全部101个分位数：")
        print(table.to_string(index=False))


def main() -> None:
    args = parse_args()
    if args.random_n < 0:
        raise ValueError("--random-n 不能小于 0。")

    prediction_path = args.prediction_path.expanduser().resolve()
    if not prediction_path.is_file():
        raise FileNotFoundError(f"找不到预测文件：{prediction_path}")

    frame = pd.read_parquet(prediction_path).reset_index(drop=True)
    missing = set(IDENTIFIER_COLUMNS + [LABEL_COLUMN]) - set(frame.columns)
    if missing:
        raise ValueError(f"预测文件缺少字段：{sorted(missing)}")
    qcols = get_quantile_columns(frame)

    specified = select_specified_sample(frame, args)
    selected_parts: list[pd.DataFrame] = []
    if args.random_n:
        # 指定样本不再参与随机抽取，保证能展示“随机N个 + 指定1个”。
        candidates = frame.drop(index=specified.index) if specified is not None else frame
        count = min(args.random_n, len(candidates))
        random_rows = candidates.sample(n=count, random_state=args.seed)
        selected_parts.append(random_rows)

    if specified is not None:
        selected_parts.append(specified)

    if not selected_parts:
        raise ValueError("请至少保留随机抽样，或提供一个指定样本条件。")

    selected = pd.concat(selected_parts).copy()
    row_positions = selected.index.to_list()
    specified_positions = set(specified.index) if specified is not None else set()

    print(f"预测文件：{prediction_path}")
    print(f"验证集样本数：{len(frame)}；分位数列数：{len(qcols)}")
    for position in row_positions:
        row = frame.loc[position]
        label = "指定样本" if position in specified_positions else "随机样本"
        print_sample(
            f"{label}（row_index={position}）",
            row,
            qcols,
            position,
            args.show_all_quantiles,
        )

    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = (CODE_DIR / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    wide = selected.copy()
    wide.insert(
        0,
        "sample_type",
        ["specified" if index in specified_positions else "random" for index in wide.index],
    )
    wide.insert(1, "row_index", wide.index)
    wide_path = output_dir / "tauijt_valid_selected_samples_wide.csv"
    wide.to_csv(wide_path, index=False)

    long = wide.melt(
        id_vars=["sample_type", "row_index", *IDENTIFIER_COLUMNS, LABEL_COLUMN],
        value_vars=qcols,
        var_name="quantile_name",
        value_name="prediction",
    )
    long.insert(
        long.columns.get_loc("quantile_name") + 1,
        "quantile_level",
        long["quantile_name"].str.split("_", n=1).str[1].astype(float),
    )
    long_path = output_dir / "tauijt_valid_selected_samples_quantiles_long.csv"
    long.to_csv(long_path, index=False)

    print("=" * 100)
    print(f"已保存宽表（含101个分位数）：{wide_path}")
    print(f"已保存长表（每样本101行）：{long_path}")


if __name__ == "__main__":
    main()
