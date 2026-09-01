# -*- coding: utf-8 -*-
"""
1_4 删除异常订单.py

作用
----
在“1_2 经纬度恢复及删除多余变量_保留订单.parquet”的基础上，继续删除
价格、服务时间、行程距离以及它们之间不一致的异常订单。

本脚本删除的异常订单
----------------------------
1. 标识与区域字段异常：
   - Trip ID 缺失或空字符串；
   - Pickup/Dropoff Census Tract 缺失，或不是 11 位数字；
   - 重复 Trip ID 只保留第一条通过其他检查的订单，删除后续重复记录。

2. 时间字段异常：
   - 上车或下车时间无法解析；
   - 下车时间早于上车时间；
   - Trip Seconds 缺失、非有限值、小于 60 秒或大于 4 小时；
   - 时间戳差与 Trip Seconds 的差异超过 30 分钟。
     注：现有上下车时间已被四舍五入到 15 分钟，因此不要求两种时长完全相等。

3. 距离与速度异常：
   - Trip Miles 缺失、非有限值、小于等于 0，或大于 100 英里；
   - 由距离和 Trip Seconds 推导的平均速度小于 0.5 mph 或大于 100 mph。

4. 价格字段异常：
   - Fare 缺失、非有限值或小于等于 0；
     这条规则会删除“Fare=0，仅有 2.00/3.25 美元 Additional Charges”的记录。
   - Additional Charges 缺失、非有限值或小于 0；
   - Trip Total fare 缺失、非有限值、小于等于 0 或大于 500 美元；
   - Trip Total fare 与 Fare + Additional Charges 的差异大于 0.01 美元。

5. 价格—距离—时间联合异常：
   - 当距离不小于 0.5 英里时，每英里价格小于 0.5 或大于 200 美元；
   - 每分钟价格小于 0.05 或大于 20 美元。

输出
----
1. 1_4 异常订单清洗_保留订单.parquet
   - 仅保留原始 10 个字段，可直接作为后续 OD/Region-15min 聚合的输入。
2. 1_4 异常订单清洗_删除订单.csv
   - 保留原始字段，并附加 deletion_reasons 和诊断字段，便于审核。
   - 使用 UTF-8 with BOM 编码，可直接用 Excel 或表格软件打开检查。
3. 1_4 异常订单清洗_处理统计.csv
   - 输出总行数、保留/删除行数和每条删除规则命中数。

安全说明
--------
- 默认不覆盖任何已有输出；如需重跑，必须显式传入 --overwrite。
- 本脚本不修改原始 1_2 parquet。
- 所有阈值都集中在“参数设置”中，正式运行前可继续根据异常样本审核调整。
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Set

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ============================================================
# 1. 路径设置
# ============================================================

CODE_DIR = Path(__file__).resolve().parent

DEFAULT_INPUT = CODE_DIR / "1_2 经纬度恢复及删除多余变量_保留订单.parquet"
DEFAULT_KEPT = CODE_DIR / "1_4 异常订单清洗_保留订单.parquet"
DEFAULT_DELETED = CODE_DIR / "1_4 异常订单清洗_删除订单.csv"
DEFAULT_SUMMARY = CODE_DIR / "1_4 异常订单清洗_处理统计.csv"


# ============================================================
# 2. 参数设置：保守的硬异常边界
# ============================================================

BATCH_SIZE = 250_000

MIN_TRIP_SECONDS = 60.0
MAX_TRIP_SECONDS = 4.0 * 60.0 * 60.0

MIN_TRIP_MILES = 0.0       # 删除 <= 0
MAX_TRIP_MILES = 100.0

MIN_SPEED_MPH = 0.5
MAX_SPEED_MPH = 100.0

MAX_TOTAL_FARE = 500.0
FARE_SUM_TOLERANCE = 0.01

# 只对不小于 0.5 英里的订单检查每英里价格，
# 避免极短行程因分母过小被误判。
MIN_MILES_FOR_UNIT_PRICE_CHECK = 0.5
MIN_FARE_PER_MILE = 0.5
MAX_FARE_PER_MILE = 200.0

MIN_FARE_PER_MINUTE = 0.05
MAX_FARE_PER_MINUTE = 20.0

# 上下车时间已四舍五入到 15 分钟；两端合计最大理论误差约 15 分钟。
# 这里使用 30 分钟容差，只删除明显矛盾记录。
MAX_TIMESTAMP_DURATION_GAP_SECONDS = 30.0 * 60.0


REQUIRED_COLUMNS = [
    "Trip ID",
    "Trip Start Timestamp",
    "Trip End Timestamp",
    "Trip Seconds",
    "Trip Miles",
    "Pickup Census Tract",
    "Dropoff Census Tract",
    "Fare",
    "Additional Charges",
    "Trip Total fare",
]

NUMERIC_COLUMNS = [
    "Trip Seconds",
    "Trip Miles",
    "Fare",
    "Additional Charges",
    "Trip Total fare",
]


REASON_DESCRIPTIONS = {
    "missing_trip_id": "Trip ID 缺失或为空",
    "duplicate_trip_id": "Trip ID 重复（删除后续通过其他检查的重复记录）",
    "invalid_pickup_tract": "上车 Census Tract 缺失或不是 11 位数字",
    "invalid_dropoff_tract": "下车 Census Tract 缺失或不是 11 位数字",
    "invalid_start_timestamp": "上车时间无法解析",
    "invalid_end_timestamp": "下车时间无法解析",
    "end_before_start": "下车时间早于上车时间",
    "missing_or_nonfinite_trip_seconds": "Trip Seconds 缺失或非有限值",
    "trip_seconds_out_of_range": "Trip Seconds 不在 [60秒, 4小时] 内",
    "missing_or_nonfinite_trip_miles": "Trip Miles 缺失或非有限值",
    "trip_miles_out_of_range": "Trip Miles 小于等于 0 或大于 100 英里",
    "missing_or_nonfinite_fare": "Fare 缺失或非有限值",
    "nonpositive_fare": "Fare 小于等于 0",
    "missing_or_nonfinite_additional_charges": "Additional Charges 缺失或非有限值",
    "negative_additional_charges": "Additional Charges 小于 0",
    "missing_or_nonfinite_total_fare": "Trip Total fare 缺失或非有限值",
    "total_fare_out_of_range": "Trip Total fare 小于等于 0 或大于 500 美元",
    "fare_components_inconsistent": "Trip Total fare 与 Fare + Additional Charges 不一致",
    "timestamp_duration_inconsistent": "时间戳差与 Trip Seconds 差异超过 30 分钟",
    "implied_speed_out_of_range": "平均速度小于 0.5 mph 或大于 100 mph",
    "fare_per_mile_out_of_range": "每英里价格小于 0.5 或大于 200 美元",
    "fare_per_minute_out_of_range": "每分钟价格小于 0.05 或大于 20 美元",
}


# ============================================================
# 3. 工具函数
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="删除价格、时间、距离和联合异常订单。")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="输入订单 parquet。")
    parser.add_argument("--kept-output", type=Path, default=DEFAULT_KEPT, help="保留订单 parquet。")
    parser.add_argument("--deleted-output", type=Path, default=DEFAULT_DELETED, help="删除订单及原因 CSV。")
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY, help="清洗统计 CSV。")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="每个分块的最大行数。")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖本脚本的三个输出文件；不会删除或覆盖输入文件。",
    )
    return parser.parse_args()


def normalize_text(series: pd.Series) -> pd.Series:
    """将字符串字段去除首尾空格，并将常见的伪缺失值转为 pd.NA。"""
    result = series.astype("string").str.strip()
    return result.mask(result.str.lower().isin(["", "nan", "none", "null", "<na>"]))


def finite_mask(series: pd.Series) -> pd.Series:
    """数值非缺失且为有限值。"""
    values = pd.to_numeric(series, errors="coerce")
    return values.notna() & np.isfinite(values)


def validate_schema(parquet_file: pq.ParquetFile) -> None:
    """在清洗前审核输入字段，避免用错数据文件。"""
    columns = set(parquet_file.schema_arrow.names)
    missing = [col for col in REQUIRED_COLUMNS if col not in columns]
    if missing:
        raise ValueError(f"输入文件缺少必要字段：{missing}")


def find_duplicate_trip_ids(input_path: Path) -> Set[str]:
    """
    只读取 Trip ID 一列，找出在整个 parquet 中重复的 ID。

    不在分块内单独去重，否则无法发现跨分块重复。
    """
    ids = pd.read_parquet(input_path, columns=["Trip ID"])["Trip ID"]
    ids = normalize_text(ids)
    duplicated = ids.notna() & ids.duplicated(keep=False)
    return set(ids.loc[duplicated].dropna().astype(str).unique())


def prepare_output_paths(paths: Iterable[Path], overwrite: bool) -> None:
    """默认拒绝覆盖；只在显式 --overwrite 时删除本脚本的旧输出。"""
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not overwrite:
            raise FileExistsError(f"输出已存在：{path}\n如确需重跑，请显式加 --overwrite。")
    if overwrite:
        for path in paths:
            if path.exists():
                path.unlink()


def normalize_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """统一分块数据类型，不对异常值做静默修复。"""
    df = df.copy()

    for col in ["Trip ID", "Pickup Census Tract", "Dropoff Census Tract"]:
        df[col] = normalize_text(df[col])

    for col in ["Trip Start Timestamp", "Trip End Timestamp"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def build_reason_masks(df: pd.DataFrame) -> tuple[Dict[str, pd.Series], pd.DataFrame]:
    """计算每条删除规则的布尔掩码和用于审核的诊断变量。"""
    index = df.index
    false = pd.Series(False, index=index)

    trip_id = normalize_text(df["Trip ID"])
    pickup = normalize_text(df["Pickup Census Tract"])
    dropoff = normalize_text(df["Dropoff Census Tract"])

    start = pd.to_datetime(df["Trip Start Timestamp"], errors="coerce")
    end = pd.to_datetime(df["Trip End Timestamp"], errors="coerce")

    seconds = pd.to_numeric(df["Trip Seconds"], errors="coerce")
    miles = pd.to_numeric(df["Trip Miles"], errors="coerce")
    fare = pd.to_numeric(df["Fare"], errors="coerce")
    additional = pd.to_numeric(df["Additional Charges"], errors="coerce")
    total = pd.to_numeric(df["Trip Total fare"], errors="coerce")

    seconds_finite = finite_mask(seconds)
    miles_finite = finite_mask(miles)
    fare_finite = finite_mask(fare)
    additional_finite = finite_mask(additional)
    total_finite = finite_mask(total)

    timestamp_duration_seconds = (end - start).dt.total_seconds()

    with np.errstate(divide="ignore", invalid="ignore"):
        implied_speed_mph = miles / (seconds / 3600.0)
        fare_per_mile = total / miles
        fare_per_minute = total / (seconds / 60.0)

    masks: Dict[str, pd.Series] = {
        "missing_trip_id": trip_id.isna(),
        # duplicate_trip_id 在其他规则完成后按全局顺序单独填充。
        "duplicate_trip_id": false.copy(),
        "invalid_pickup_tract": pickup.isna() | ~pickup.fillna("").str.fullmatch(r"\d{11}"),
        "invalid_dropoff_tract": dropoff.isna() | ~dropoff.fillna("").str.fullmatch(r"\d{11}"),
        "invalid_start_timestamp": start.isna(),
        "invalid_end_timestamp": end.isna(),
        "end_before_start": start.notna() & end.notna() & (end < start),
        "missing_or_nonfinite_trip_seconds": ~seconds_finite,
        "trip_seconds_out_of_range": (
            seconds_finite
            & ((seconds < MIN_TRIP_SECONDS) | (seconds > MAX_TRIP_SECONDS))
        ),
        "missing_or_nonfinite_trip_miles": ~miles_finite,
        "trip_miles_out_of_range": (
            miles_finite
            & ((miles <= MIN_TRIP_MILES) | (miles > MAX_TRIP_MILES))
        ),
        "missing_or_nonfinite_fare": ~fare_finite,
        "nonpositive_fare": fare_finite & (fare <= 0),
        "missing_or_nonfinite_additional_charges": ~additional_finite,
        "negative_additional_charges": additional_finite & (additional < 0),
        "missing_or_nonfinite_total_fare": ~total_finite,
        "total_fare_out_of_range": (
            total_finite & ((total <= 0) | (total > MAX_TOTAL_FARE))
        ),
        "fare_components_inconsistent": (
            fare_finite
            & additional_finite
            & total_finite
            & ((total - (fare + additional)).abs() > FARE_SUM_TOLERANCE)
        ),
        "timestamp_duration_inconsistent": (
            start.notna()
            & end.notna()
            & seconds_finite
            & (end >= start)
            & (
                (timestamp_duration_seconds - seconds).abs()
                > MAX_TIMESTAMP_DURATION_GAP_SECONDS
            )
        ),
        "implied_speed_out_of_range": (
            seconds_finite
            & miles_finite
            & (seconds > 0)
            & (miles > 0)
            & (
                (implied_speed_mph < MIN_SPEED_MPH)
                | (implied_speed_mph > MAX_SPEED_MPH)
            )
        ),
        "fare_per_mile_out_of_range": (
            total_finite
            & miles_finite
            & (total > 0)
            & (miles >= MIN_MILES_FOR_UNIT_PRICE_CHECK)
            & (
                (fare_per_mile < MIN_FARE_PER_MILE)
                | (fare_per_mile > MAX_FARE_PER_MILE)
            )
        ),
        "fare_per_minute_out_of_range": (
            total_finite
            & seconds_finite
            & (total > 0)
            & (seconds > 0)
            & (
                (fare_per_minute < MIN_FARE_PER_MINUTE)
                | (fare_per_minute > MAX_FARE_PER_MINUTE)
            )
        ),
    }

    diagnostics = pd.DataFrame(
        {
            "timestamp_duration_seconds": timestamp_duration_seconds,
            "duration_gap_seconds": (timestamp_duration_seconds - seconds).abs(),
            "implied_speed_mph": implied_speed_mph,
            "fare_per_mile": fare_per_mile,
            "fare_per_minute": fare_per_minute,
            "recomputed_total_fare": fare + additional,
        },
        index=index,
    )

    return masks, diagnostics


def combine_masks(masks: Dict[str, pd.Series], exclude: Set[str] | None = None) -> pd.Series:
    """对多个规则取逻辑或。"""
    exclude = exclude or set()
    selected = [mask for name, mask in masks.items() if name not in exclude]
    if not selected:
        raise ValueError("没有可合并的异常规则。")
    result = selected[0].copy()
    for mask in selected[1:]:
        result |= mask
    return result


def mark_later_duplicate_rows(
    df: pd.DataFrame,
    base_invalid: pd.Series,
    duplicate_ids: Set[str],
    seen_valid_duplicate_ids: Set[str],
) -> pd.Series:
    """
    对重复 Trip ID 保留第一条“通过其他检查”的记录。

    如果某个 ID 的第一条记录本身就存在价格/时间/距离异常，它不会占用
    该 ID 的“保留名额”；后续第一条合格记录仍可保留。
    """
    duplicate_mask = pd.Series(False, index=df.index)
    ids = normalize_text(df["Trip ID"])

    candidate_positions = np.flatnonzero(
        (~base_invalid).to_numpy()
        & ids.isin(duplicate_ids).to_numpy()
    )

    for position in candidate_positions:
        row_index = df.index[position]
        trip_id = str(ids.iloc[position])
        if trip_id in seen_valid_duplicate_ids:
            duplicate_mask.loc[row_index] = True
        else:
            seen_valid_duplicate_ids.add(trip_id)

    return duplicate_mask


def deletion_reason_text(masks: Dict[str, pd.Series], deleted_index: pd.Index) -> pd.Series:
    """把同一条订单同时命中的多个原因用分号合并。"""
    reason_frame = pd.DataFrame(
        {name: mask.loc[deleted_index].astype(bool) for name, mask in masks.items()},
        index=deleted_index,
    )
    values = reason_frame.to_numpy(dtype=bool)
    names = np.asarray(reason_frame.columns, dtype=object)
    text = [";".join(names[row]) for row in values]
    return pd.Series(text, index=deleted_index, dtype="string")


def table_from_source_schema(df: pd.DataFrame, source_schema: pa.Schema) -> pa.Table:
    """保留订单按输入 parquet 的原始字段和 Arrow 类型写出。"""
    table = pa.Table.from_pandas(df[source_schema.names], preserve_index=False)
    return table.cast(source_schema, safe=False)


def write_table(
    writer: pq.ParquetWriter | None,
    path: Path,
    table: pa.Table,
) -> pq.ParquetWriter | None:
    """延迟创建 ParquetWriter，避免第一块为空时无法确定 schema。"""
    if table.num_rows == 0:
        return writer
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema, compression="zstd")
    writer.write_table(table)
    return writer


# ============================================================
# 4. 主流程
# ============================================================

def main() -> None:
    args = parse_args()

    input_path = args.input.expanduser().resolve()
    kept_path = args.kept_output.expanduser().resolve()
    deleted_path = args.deleted_output.expanduser().resolve()
    summary_path = args.summary_output.expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在：{input_path}")

    if input_path in {kept_path, deleted_path, summary_path}:
        raise ValueError("输出路径不能与输入路径相同。")

    prepare_output_paths(
        [kept_path, deleted_path, summary_path],
        overwrite=args.overwrite,
    )

    parquet_file = pq.ParquetFile(input_path)
    validate_schema(parquet_file)
    source_schema = parquet_file.schema_arrow

    print("第一遍：检查全局重复 Trip ID。")
    duplicate_ids = find_duplicate_trip_ids(input_path)
    print(f"全局重复 Trip ID 种类数：{len(duplicate_ids):,}")

    kept_writer: pq.ParquetWriter | None = None
    deleted_header_written = False
    seen_valid_duplicate_ids: Set[str] = set()

    total_rows = 0
    kept_rows = 0
    deleted_rows = 0
    reason_counts: Counter[str] = Counter()
    batch_rows = []

    try:
        batches = parquet_file.iter_batches(batch_size=args.batch_size)

        for batch_id, batch in enumerate(batches, start=1):
            df = normalize_chunk(batch.to_pandas())
            total_rows += len(df)

            masks, diagnostics = build_reason_masks(df)

            # 先合并除重复以外的规则，然后在全局范围内保留第一条合格的重复 ID。
            base_invalid = combine_masks(masks, exclude={"duplicate_trip_id"})
            masks["duplicate_trip_id"] = mark_later_duplicate_rows(
                df=df,
                base_invalid=base_invalid,
                duplicate_ids=duplicate_ids,
                seen_valid_duplicate_ids=seen_valid_duplicate_ids,
            )

            delete_mask = combine_masks(masks)
            keep_mask = ~delete_mask

            kept = df.loc[keep_mask, source_schema.names].copy()
            deleted = df.loc[delete_mask, source_schema.names].copy()

            if len(deleted) > 0:
                deleted["deletion_reasons"] = deletion_reason_text(
                    masks, deleted.index
                ).values
                for col in diagnostics.columns:
                    deleted[col] = diagnostics.loc[deleted.index, col].to_numpy()

            kept_table = table_from_source_schema(kept, source_schema)
            kept_writer = write_table(kept_writer, kept_path, kept_table)

            # 删除订单以 CSV 输出，便于直接打开审核。
            # 第一个非空分块使用 utf-8-sig 写入 BOM 和表头；
            # 后续分块使用普通 utf-8 追加，避免在文件中间重复写入 BOM。
            if len(deleted) > 0:
                deleted.reset_index(drop=True).to_csv(
                    deleted_path,
                    mode="a" if deleted_header_written else "w",
                    index=False,
                    header=not deleted_header_written,
                    encoding="utf-8" if deleted_header_written else "utf-8-sig",
                )
                deleted_header_written = True

            batch_kept = int(keep_mask.sum())
            batch_deleted = int(delete_mask.sum())
            kept_rows += batch_kept
            deleted_rows += batch_deleted

            for name, mask in masks.items():
                reason_counts[name] += int(mask.sum())

            batch_rows.append(
                {
                    "batch_id": batch_id,
                    "rows": len(df),
                    "kept_rows": batch_kept,
                    "deleted_unique_rows": batch_deleted,
                }
            )

            print(
                f"batch={batch_id}, 输入={len(df):,}, "
                f"保留={batch_kept:,}, 删除={batch_deleted:,}"
            )
    finally:
        if kept_writer is not None:
            kept_writer.close()

    # 处理统计中，各原因的命中数可以相互重叠；
    # deleted_rows 才是删除的唯一订单行数。
    summary_rows = [
        {
            "metric": "input_rows",
            "count": total_rows,
            "description": "输入订单总行数",
        },
        {
            "metric": "kept_rows",
            "count": kept_rows,
            "description": "通过所有检查的订单行数",
        },
        {
            "metric": "deleted_unique_rows",
            "count": deleted_rows,
            "description": "至少命中一条规则的唯一删除行数",
        },
    ]

    for reason in REASON_DESCRIPTIONS:
        summary_rows.append(
            {
                "metric": reason,
                "count": reason_counts.get(reason, 0),
                "description": REASON_DESCRIPTIONS[reason],
            }
        )

    pd.DataFrame(summary_rows).to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n异常订单清洗完成。")
    print(f"输入行数：{total_rows:,}")
    print(f"保留行数：{kept_rows:,}")
    print(f"删除行数：{deleted_rows:,}")
    print(f"保留订单：{kept_path}")
    print(f"删除订单：{deleted_path}")
    print(f"处理统计：{summary_path}")


if __name__ == "__main__":
    main()
