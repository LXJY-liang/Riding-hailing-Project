# -*- coding: utf-8 -*-
"""
第二步：将芝加哥内部 OD 订单聚合为稀疏的区域-15min状态。

口径：
- 出发状态使用上车区域和 Trip Start Timestamp；
- 到达状态使用下车区域和 Trip End Timestamp；
- 输入已经保证每条订单的 i,j 均属于 801 个芝加哥区域；
- 本步只保留真实出现过出发或到达订单的 (region_id,t)；
- 第三步再补全 801×96×30 完整面板。

输入：3_1 芝加哥内部OD订单.parquet
输出：3_2 区域-15min聚合（内部OD口径）结果.parquet
检查：3_2 区域-15min聚合（内部OD口径）_检查.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


CODE_DIR = Path(__file__).resolve().parent

ORDER_FILE = CODE_DIR / "3_1 芝加哥内部OD订单.parquet"
OUTPUT_FILE = CODE_DIR / "3_2 区域-15min聚合（内部OD口径）结果.parquet"
DIAGNOSTICS_FILE = CODE_DIR / "3_2 区域-15min聚合（内部OD口径）_检查.json"

DATA_START = pd.Timestamp("2024-06-01 00:00:00")
DATA_END = pd.Timestamp("2024-06-30 23:45:00")
PARQUET_BATCH_SIZE = 250_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="聚合芝加哥内部 OD 的区域-15min状态。")
    parser.add_argument("--overwrite", action="store_true", help="覆盖本脚本的旧输出。")
    return parser.parse_args()


def clean_region_id(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    cleaned = cleaned.str.replace(r"\.0$", "", regex=True)
    cleaned = cleaned.str.extract(r"(\d+)", expand=False)
    return cleaned.str.zfill(11)


def main() -> None:
    args = parse_args()
    if not ORDER_FILE.exists():
        raise FileNotFoundError(f"缺少上游内部 OD 订单：{ORDER_FILE}")

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

    source = pq.ParquetFile(ORDER_FILE)
    required = [
        "Trip Start Timestamp",
        "Trip End Timestamp",
        "Trip Seconds",
        "Trip Miles",
        "Pickup Census Tract",
        "Dropoff Census Tract",
        "Trip Total fare",
    ]
    missing = sorted(set(required) - set(source.schema_arrow.names))
    if missing:
        raise ValueError(f"内部 OD 订单缺少字段：{missing}")

    origin_parts: list[pd.DataFrame] = []
    destination_parts: list[pd.DataFrame] = []
    input_rows = 0
    departure_rows = 0
    arrival_rows = 0

    for batch_number, batch in enumerate(
        source.iter_batches(batch_size=PARQUET_BATCH_SIZE, columns=required), start=1
    ):
        data = batch.to_pandas()
        input_rows += len(data)

        data["i"] = clean_region_id(data["Pickup Census Tract"])
        data["j"] = clean_region_id(data["Dropoff Census Tract"])
        data["start_t"] = pd.to_datetime(data["Trip Start Timestamp"], errors="coerce")
        data["end_t"] = pd.to_datetime(data["Trip End Timestamp"], errors="coerce")
        for column in ("Trip Seconds", "Trip Miles", "Trip Total fare"):
            data[column] = pd.to_numeric(data[column], errors="coerce")

        valid = (
            data["i"].notna()
            & data["j"].notna()
            & data["start_t"].notna()
            & data["end_t"].notna()
            & data["Trip Seconds"].notna()
            & data["Trip Miles"].notna()
            & data["Trip Total fare"].notna()
        )
        if not valid.all():
            raise ValueError(
                f"batch {batch_number} 仍有 {int((~valid).sum()):,} 条关键字段无效的订单。"
            )

        data["tau_minutes"] = data["Trip Seconds"] / 60.0

        departure_mask = data["start_t"].between(DATA_START, DATA_END, inclusive="both")
        departure = data.loc[departure_mask]
        departure_rows += len(departure)
        if not departure.empty:
            origin_parts.append(
                departure.groupby(["i", "start_t"], as_index=False, sort=False)
                .agg(
                    D_origin_it=("Trip Total fare", "size"),
                    fare_sum=("Trip Total fare", "sum"),
                    tau_sum=("tau_minutes", "sum"),
                    miles_sum=("Trip Miles", "sum"),
                )
            )

        arrival_mask = data["end_t"].between(DATA_START, DATA_END, inclusive="both")
        arrival = data.loc[arrival_mask]
        arrival_rows += len(arrival)
        if not arrival.empty:
            destination_parts.append(
                arrival.groupby(["j", "end_t"], as_index=False, sort=False)
                .agg(D_dest_it=("Trip Total fare", "size"))
            )

        print(
            f"batch={batch_number}, 累计内部OD={input_rows:,}, "
            f"6月出发={departure_rows:,}, 6月到达={arrival_rows:,}"
        )

    if not origin_parts or not destination_parts:
        raise ValueError("内部 OD 订单没有产生区域聚合结果。")

    origin = (
        pd.concat(origin_parts, ignore_index=True)
        .groupby(["i", "start_t"], as_index=False, sort=False)
        .agg(
            D_origin_it=("D_origin_it", "sum"),
            fare_sum=("fare_sum", "sum"),
            tau_sum=("tau_sum", "sum"),
            miles_sum=("miles_sum", "sum"),
        )
        .rename(columns={"i": "region_id", "start_t": "t"})
    )
    denominator = origin["D_origin_it"].to_numpy(dtype=np.float64)
    origin["p_origin_bar_it"] = origin["fare_sum"] / denominator
    origin["tau_origin_bar_it"] = origin["tau_sum"] / denominator
    origin["d_origin_bar_it"] = origin["miles_sum"] / denominator
    origin = origin.drop(columns=["fare_sum", "tau_sum", "miles_sum"])

    destination = (
        pd.concat(destination_parts, ignore_index=True)
        .groupby(["j", "end_t"], as_index=False, sort=False)
        .agg(D_dest_it=("D_dest_it", "sum"))
        .rename(columns={"j": "region_id", "end_t": "t"})
    )

    region = origin.merge(destination, on=["region_id", "t"], how="outer", validate="one_to_one")
    region["D_origin_it"] = region["D_origin_it"].fillna(0).astype("int32")
    region["D_dest_it"] = region["D_dest_it"].fillna(0).astype("int32")
    region["net_flow_it"] = (region["D_dest_it"] - region["D_origin_it"]).astype("int32")
    for column in ("p_origin_bar_it", "tau_origin_bar_it", "d_origin_bar_it"):
        region[column] = pd.to_numeric(region[column], errors="coerce").astype("float32")

    final_columns = [
        "region_id",
        "t",
        "D_origin_it",
        "D_dest_it",
        "p_origin_bar_it",
        "tau_origin_bar_it",
        "d_origin_bar_it",
        "net_flow_it",
    ]
    region = region[final_columns].sort_values(["region_id", "t"]).reset_index(drop=True)
    if region[["region_id", "t"]].duplicated().any():
        raise AssertionError("区域聚合结果存在重复 (region_id,t)。")
    region.to_parquet(OUTPUT_FILE, index=False, compression="zstd")

    diagnostics = {
        "internal_order_rows": int(input_rows),
        "departure_rows_in_june": int(departure_rows),
        "arrival_rows_in_june": int(arrival_rows),
        "sparse_region_time_rows": int(len(region)),
        "region_count": int(region["region_id"].nunique()),
        "time_count": int(region["t"].nunique()),
        "D_origin_it_total": int(region["D_origin_it"].sum()),
        "D_dest_it_total": int(region["D_dest_it"].sum()),
        "output_file": str(OUTPUT_FILE),
    }
    DIAGNOSTICS_FILE.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
