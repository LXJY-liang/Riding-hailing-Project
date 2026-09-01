# -*- coding: utf-8 -*-
"""
第一步：在订单级删除非芝加哥内部 OD 订单。

只保留同时满足以下两个条件的订单：
1. 上车 Census Tract 属于研究使用的 801 个芝加哥区域；
2. 下车 Census Tract 也属于同一组 801 个区域。

因此，输出订单只包含“芝加哥内部 OD”。不允许先做区域聚合再按
单个区域筛选，因为那样会把芝加哥内出发、芝加哥外到达（或相反）的订单
错误计入内部区域状态。

输入：1_4 异常订单清洗_保留订单.parquet
输出：3_1 芝加哥内部OD订单.parquet
检查：3_1 芝加哥内部OD订单_检查.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent

ORDER_FILE = CODE_DIR / "1_4 异常订单清洗_保留订单.parquet"
CHICAGO_TRACT_FILE = (
    PROJECT_ROOT
    / "0 芝加哥census tract编号及对应的community"
    / "Census Tracts TIGER 2010.csv"
)

OUTPUT_FILE = CODE_DIR / "3_1 芝加哥内部OD订单.parquet"
DIAGNOSTICS_FILE = CODE_DIR / "3_1 芝加哥内部OD订单_检查.json"

EXPECTED_REGION_COUNT = 801
PARQUET_BATCH_SIZE = 250_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在订单级保留芝加哥内部 OD。")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖本脚本已生成的 parquet 和检查 JSON。",
    )
    return parser.parse_args()


def clean_region_id(series: pd.Series) -> pd.Series:
    """将 Census Tract 统一为 11 位数字字符串。"""
    cleaned = series.astype("string").str.strip()
    cleaned = cleaned.str.replace(r"\.0$", "", regex=True)
    cleaned = cleaned.str.extract(r"(\d+)", expand=False)
    return cleaned.str.zfill(11)


def load_chicago_regions() -> set[str]:
    """读取并检查研究使用的 801 个芝加哥 Census Tract。"""
    chicago = pd.read_csv(CHICAGO_TRACT_FILE, dtype="string", encoding="utf-8-sig")
    if "GEOID10" in chicago.columns:
        id_column = "GEOID10"
    elif "GEOID" in chicago.columns:
        id_column = "GEOID"
    else:
        id_column = chicago.columns[4]

    regions = set(clean_region_id(chicago[id_column]).dropna().astype(str))
    if len(regions) != EXPECTED_REGION_COUNT:
        raise ValueError(
            f"芝加哥区域数为 {len(regions)}，预期为 {EXPECTED_REGION_COUNT}。"
        )
    return regions


def main() -> None:
    args = parse_args()

    required = [ORDER_FILE, CHICAGO_TRACT_FILE]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少上游输入：\n" + "\n".join(missing))

    existing = [path for path in (OUTPUT_FILE, DIAGNOSTICS_FILE) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "输出已存在；如需重建请使用 --overwrite：\n"
            + "\n".join(str(path) for path in existing)
        )
    if args.overwrite:
        for path in (OUTPUT_FILE, DIAGNOSTICS_FILE):
            if path.exists():
                path.unlink()

    chicago_regions = load_chicago_regions()
    source = pq.ParquetFile(ORDER_FILE)
    required_columns = {
        "Pickup Census Tract",
        "Dropoff Census Tract",
    }
    missing_columns = sorted(required_columns - set(source.schema_arrow.names))
    if missing_columns:
        raise ValueError(f"订单文件缺少字段：{missing_columns}")

    writer: pq.ParquetWriter | None = None
    input_rows = 0
    internal_rows = 0
    pickup_outside_rows = 0
    dropoff_outside_rows = 0
    both_outside_rows = 0

    try:
        for batch_number, batch in enumerate(
            source.iter_batches(batch_size=PARQUET_BATCH_SIZE), start=1
        ):
            data = batch.to_pandas()
            input_rows += len(data)

            pickup = clean_region_id(data["Pickup Census Tract"])
            dropoff = clean_region_id(data["Dropoff Census Tract"])
            pickup_inside = pickup.isin(chicago_regions)
            dropoff_inside = dropoff.isin(chicago_regions)
            internal = pickup_inside & dropoff_inside

            pickup_outside_rows += int((~pickup_inside).sum())
            dropoff_outside_rows += int((~dropoff_inside).sum())
            both_outside_rows += int((~pickup_inside & ~dropoff_inside).sum())
            internal_rows += int(internal.sum())

            kept = data.loc[internal].copy()
            kept["Pickup Census Tract"] = pickup.loc[internal].to_numpy()
            kept["Dropoff Census Tract"] = dropoff.loc[internal].to_numpy()

            if kept.empty:
                continue

            table = pa.Table.from_pandas(kept, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    OUTPUT_FILE,
                    table.schema,
                    compression="zstd",
                )
            writer.write_table(table)

            print(
                f"batch={batch_number}, 累计输入={input_rows:,}, "
                f"累计内部OD={internal_rows:,}"
            )
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        raise ValueError("没有找到任何芝加哥内部 OD 订单。")

    diagnostics = {
        "input_rows": int(input_rows),
        "internal_od_rows": int(internal_rows),
        "removed_unique_rows": int(input_rows - internal_rows),
        "pickup_outside_rows": int(pickup_outside_rows),
        "dropoff_outside_rows": int(dropoff_outside_rows),
        "both_outside_rows": int(both_outside_rows),
        "chicago_region_count": int(len(chicago_regions)),
        "output_file": str(OUTPUT_FILE),
    }
    DIAGNOSTICS_FILE.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
