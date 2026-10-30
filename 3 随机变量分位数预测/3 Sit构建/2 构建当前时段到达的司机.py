# -*- coding: utf-8 -*-
"""构造供给动力学中的当前时段历史成交到达量。

对每个区域 ``i`` 和决策时段 ``t``，输出

    M_arrive_it = sum_k sum_{s: s + l_{kis} = t} M_{kis}

上游完整区域面板中的 ``D_dest_it`` 已经由芝加哥内部 OD 订单按
``Trip End Timestamp`` 聚合。因而，一笔订单只要在时段 ``t`` 完成并以
区域 ``i`` 为目的地，就计入当前时段的 ``M_arrive_it``。

本脚本只负责生成确定的历史订单流，不构造 S_it，也不构造 U_it。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "1 数据清洗"
    / "3_3 区域-15min聚合（补全空值）结果.parquet"
)
OUTPUT_DIR = HERE / "output"
OUTPUT_FILE = OUTPUT_DIR / "M_arrive_it.parquet"
DIAGNOSTICS_FILE = OUTPUT_DIR / "M_arrive_it_diagnostics.json"

DATA_START = pd.Timestamp("2024-06-01 00:00:00")
DATA_END = pd.Timestamp("2024-06-30 23:45:00")
EXPECTED_REGION_COUNT = 801
EXPECTED_TIME_COUNT = 30 * 96


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="构造每个 (i,t) 在当前时段完成历史订单并到达的司机数。"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖本脚本已经生成的 parquet 和诊断 JSON。",
    )
    return parser.parse_args()


def validate_complete_panel(panel: pd.DataFrame) -> None:
    """检查输入是否为 801 区域乘以 30 天 15 分钟时段的完整面板。"""

    if panel[["region_id", "t"]].isna().any().any():
        raise ValueError("输入存在缺失的 region_id 或 t。")
    if panel[["region_id", "t"]].duplicated().any():
        raise ValueError("输入存在重复的 (region_id,t)。")
    if not panel["t"].between(DATA_START, DATA_END, inclusive="both").all():
        raise ValueError("输入包含 2024 年 6 月以外的时段。")

    region_count = panel["region_id"].nunique()
    time_count = panel["t"].nunique()
    expected_rows = EXPECTED_REGION_COUNT * EXPECTED_TIME_COUNT
    if region_count != EXPECTED_REGION_COUNT:
        raise ValueError(
            f"输入区域数为 {region_count}，预期为 {EXPECTED_REGION_COUNT}。"
        )
    if time_count != EXPECTED_TIME_COUNT:
        raise ValueError(
            f"输入时段数为 {time_count}，预期为 {EXPECTED_TIME_COUNT}。"
        )
    expected_times = pd.date_range(DATA_START, DATA_END, freq="15min")
    actual_times = pd.DatetimeIndex(panel["t"].drop_duplicates().sort_values())
    if not actual_times.equals(expected_times):
        raise ValueError("输入时段不是从 6 月 1 日到 30 日连续的 15 分钟网格。")
    if len(panel) != expected_rows:
        raise ValueError(f"输入行数为 {len(panel):,}，预期为 {expected_rows:,}。")

    counts = panel.groupby("region_id", observed=True)["t"].size()
    if not counts.eq(EXPECTED_TIME_COUNT).all():
        raise ValueError("至少一个区域没有覆盖完整的 2880 个时段。")


def main() -> None:
    args = parse_args()
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"缺少完整区域历史面板：{INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = [path for path in (OUTPUT_FILE, DIAGNOSTICS_FILE) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "输出已存在；如需重建请使用 --overwrite：\n"
            + "\n".join(str(path) for path in existing)
        )
    if args.overwrite:
        for path in existing:
            path.unlink()

    panel = pd.read_parquet(
        INPUT_FILE,
        columns=["region_id", "t", "D_dest_it"],
    )
    panel["region_id"] = panel["region_id"].astype("string")
    panel["t"] = pd.to_datetime(panel["t"], errors="raise")
    panel["D_dest_it"] = pd.to_numeric(
        panel["D_dest_it"], errors="raise"
    ).astype("int32")
    if (panel["D_dest_it"] < 0).any():
        raise ValueError("D_dest_it 不能为负数。")

    validate_complete_panel(panel)

    result = panel[["region_id", "t"]].copy()
    result["M_arrive_it"] = panel["D_dest_it"].astype("int32")
    result = result.sort_values(["t", "region_id"], kind="mergesort").reset_index(
        drop=True
    )

    input_total = int(panel["D_dest_it"].sum())
    output_total = int(result["M_arrive_it"].sum())
    if input_total != output_total:
        raise AssertionError(
            f"当前时段到达量总和不一致：实际 {output_total:,}，预期 {input_total:,}。"
        )

    result.to_parquet(OUTPUT_FILE, index=False, compression="zstd")

    diagnostics = {
        "input_file": str(INPUT_FILE),
        "output_file": str(OUTPUT_FILE),
        "rows": int(len(result)),
        "region_count": int(result["region_id"].nunique()),
        "time_count": int(result["t"].nunique()),
        "M_arrive_it_total": output_total,
        "positive_arrival_rows": int((result["M_arrive_it"] > 0).sum()),
        "time_definition": "Trip End Timestamp falls in current 15-minute slot t",
        "data_scope": "Chicago internal OD historical completed transactions",
    }
    DIAGNOSTICS_FILE.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
