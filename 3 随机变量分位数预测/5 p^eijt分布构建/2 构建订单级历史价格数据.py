# -*- coding: utf-8 -*-
"""构建滚动 p^e_ijt 估算使用的逐笔历史价格表。

该脚本只做确定性的数据清洗，不计算 KDE。每行保留一笔真实完成订单：

    price = Fare + Additional Charges

价格不包含小费。后续主程序仅在 completion_time < 目标 t 时才允许使用
该订单，因此不会将尚未完成的订单泄漏到当前估算中。行程里程和时长
同时保留，用于各层候选订单的自身路线一致性检查，以及第4层与目标
OD的距离相似检查；候选筛选不使用订单价格本身。
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def load_config():
    path = Path(__file__).resolve().parent / "1 p^eijt分布构建配置.py"
    spec = importlib.util.spec_from_file_location("pe_distribution_config", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


cfg = load_config()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建订单级滚动历史价格表。")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="仅覆盖本脚本管理的订单级历史价格文件。",
    )
    return parser.parse_args()


def clean_region_id(series: pd.Series) -> pd.Series:
    """将 Census Tract 编号统一为 11 位字符串。"""

    cleaned = series.astype("string").str.strip()
    cleaned = cleaned.str.replace(r"\.0$", "", regex=True)
    cleaned = cleaned.str.extract(r"(\d+)", expand=False)
    return cleaned.str.zfill(11)


def chicago_region_set() -> set[str]:
    """从官方 tract 表读取 801 个芝加哥区域编号。"""

    tract = pd.read_csv(cfg.TRACT_FILE, dtype="string", encoding="utf-8-sig")
    id_column = "GEOID10" if "GEOID10" in tract.columns else tract.columns[4]
    ids = clean_region_id(tract[id_column]).dropna().drop_duplicates()
    result = set(ids.astype(str))
    if len(result) != cfg.REGION_COUNT:
        raise ValueError(
            f"芝加哥 Census Tract 数量为 {len(result)}，"
            f"预期为 {cfg.REGION_COUNT}。"
        )
    return result


def main() -> None:
    args = parse_args()
    cfg.ensure_output_directories()
    valid_regions = chicago_region_set()

    source = pq.ParquetFile(cfg.RAW_ORDER_FILE)
    required_columns = [
        "Trip ID",
        "Trip Start Timestamp",
        "Trip End Timestamp",
        "Pickup Census Tract",
        "Dropoff Census Tract",
        "Fare",
        "Additional Charges",
        "Trip Miles",
        "Trip Seconds",
    ]
    missing = sorted(set(required_columns) - set(source.schema.names))
    if missing:
        raise ValueError(f"逐笔订单文件缺少字段：{missing}")

    if cfg.ORDER_HISTORY_FILE.exists():
        if args.overwrite:
            cfg.ORDER_HISTORY_FILE.unlink()
        else:
            raise FileExistsError(
                f"输出已存在：{cfg.ORDER_HISTORY_FILE}。"
                "如确认重建，请使用 --overwrite。"
            )

    writer: pq.ParquetWriter | None = None
    history_id_offset = 0
    input_rows = 0
    output_rows = 0
    removed_nonpositive = 0
    removed_outside_chicago = 0
    removed_missing_required = 0
    removed_outside_history_window = 0

    data_start = pd.Timestamp(cfg.DATA_START)
    # 比最后一个目标时刻更晚完成的订单永远不会成为 6 月目标的历史。
    latest_usable_completion = pd.Timestamp(cfg.DATA_END)
    holiday_dates = {pd.Timestamp(value).date() for value in cfg.HOLIDAY_DATES}

    for batch_number, batch in enumerate(
        source.iter_batches(
            batch_size=cfg.ORDER_BATCH_SIZE,
            columns=required_columns,
        ),
        start=1,
    ):
        data = batch.to_pandas()
        input_rows += len(data)

        # 显式统一到纳秒精度，避免不同 Parquet 时间单位在后续整数比较时
        # 产生 1000 倍尺度差。第 4 步仍会再次校验并统一。
        data["start_time"] = pd.to_datetime(
            data["Trip Start Timestamp"], errors="coerce"
        ).astype("datetime64[ns]")
        data["completion_time"] = pd.to_datetime(
            data["Trip End Timestamp"], errors="coerce"
        ).astype("datetime64[ns]")
        data["i"] = clean_region_id(data["Pickup Census Tract"])
        data["j"] = clean_region_id(data["Dropoff Census Tract"])

        fare = pd.to_numeric(data["Fare"], errors="coerce")
        additional = pd.to_numeric(data["Additional Charges"], errors="coerce")
        data["price"] = fare + additional
        data["trip_miles"] = pd.to_numeric(data["Trip Miles"], errors="coerce")
        data["trip_seconds"] = pd.to_numeric(data["Trip Seconds"], errors="coerce")

        inside_chicago = data["i"].isin(valid_regions) & data["j"].isin(valid_regions)
        removed_outside_chicago += int((~inside_chicago).sum())

        positive_price = data["price"].notna() & (data["price"] > 0)
        # 删除原因按互斥顺序计数，使终端统计可以与输入、输出
        # 行数严格对账，避免同一订单被重复计入多个删除原因。
        removed_nonpositive += int((inside_chicago & ~positive_price).sum())

        required_present = (
            data["Trip ID"].notna()
            & data["start_time"].notna()
            & data["completion_time"].notna()
        )
        removed_missing_required += int(
            (inside_chicago & positive_price & ~required_present).sum()
        )

        inside_history_window = (
            (data["start_time"] >= data_start)
            & (data["start_time"] <= latest_usable_completion)
            & (data["completion_time"] < latest_usable_completion)
        )
        removed_outside_history_window += int(
            (
                inside_chicago
                & positive_price
                & required_present
                & ~inside_history_window
            ).sum()
        )

        usable = (
            inside_chicago
            & positive_price
            & required_present
            & inside_history_window
        )
        data = data.loc[usable].copy()

        if data.empty:
            print(f"batch {batch_number}: 无可用订单")
            continue

        # 历史订单的时间状态由其出发时间定义；可用性则由完成时间控制。
        start_slot = data["start_time"].dt.floor(f"{cfg.TIME_STEP_MINUTES}min")
        data["start_slot_time"] = start_slot
        data["start_day_index"] = (
            start_slot.dt.normalize() - data_start.normalize()
        ).dt.days.astype("int16")
        data["slot_in_day"] = (
            start_slot.dt.hour * 4 + start_slot.dt.minute // cfg.TIME_STEP_MINUTES
        ).astype("int16")
        data["dow"] = start_slot.dt.dayofweek.astype("int8")
        data["is_holiday"] = start_slot.dt.date.isin(holiday_dates).astype("int8")
        data["is_weekend_or_holiday"] = (
            (data["dow"] >= 5) | (data["is_holiday"] == 1)
        ).astype("int8")

        result = data[
            [
                "Trip ID",
                "i",
                "j",
                "start_slot_time",
                "completion_time",
                "start_day_index",
                "slot_in_day",
                "dow",
                "is_holiday",
                "is_weekend_or_holiday",
                "price",
                "trip_miles",
                "trip_seconds",
            ]
        ].copy()
        result = result.rename(columns={"Trip ID": "trip_id"})
        result["history_id"] = np.arange(
            history_id_offset,
            history_id_offset + len(result),
            dtype=np.int64,
        )
        history_id_offset += len(result)

        result["trip_id"] = result["trip_id"].astype("string")
        result["i"] = result["i"].astype("string")
        result["j"] = result["j"].astype("string")
        result["price"] = result["price"].astype("float32")
        result["trip_miles"] = result["trip_miles"].astype("float32")
        result["trip_seconds"] = result["trip_seconds"].astype("float32")

        result = result[
            [
                "history_id",
                "trip_id",
                "i",
                "j",
                "start_slot_time",
                "completion_time",
                "start_day_index",
                "slot_in_day",
                "dow",
                "is_holiday",
                "is_weekend_or_holiday",
                "price",
                "trip_miles",
                "trip_seconds",
            ]
        ]

        table = pa.Table.from_pandas(result, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(
                cfg.ORDER_HISTORY_FILE,
                table.schema,
                compression=cfg.PARQUET_COMPRESSION,
            )
        writer.write_table(table)
        output_rows += len(result)
        print(f"batch {batch_number}: 累计保留 {output_rows:,} 笔历史订单")

    if writer is not None:
        writer.close()

    if output_rows == 0:
        raise ValueError("没有生成任何可用的历史价格订单。")

    print("完成")
    print("输入行数：", f"{input_rows:,}")
    print("输出行数：", f"{output_rows:,}")
    print("非芝加哥内部 OD 计数：", f"{removed_outside_chicago:,}")
    print("价格缺失或非正数计数：", f"{removed_nonpositive:,}")
    print("订单编号或时间缺失计数：", f"{removed_missing_required:,}")
    print(
        "不可能成为本月任一目标时刻历史的订单计数：",
        f"{removed_outside_history_window:,}",
    )
    classified_rows = (
        output_rows
        + removed_outside_chicago
        + removed_nonpositive
        + removed_missing_required
        + removed_outside_history_window
    )
    if classified_rows != input_rows:
        raise RuntimeError(
            f"输入输出对账失败：分类后 {classified_rows:,}，"
            f"输入 {input_rows:,}。"
        )
    print("输出：", cfg.ORDER_HISTORY_FILE)


if __name__ == "__main__":
    main()
