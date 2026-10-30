# -*- coding: utf-8 -*-
"""一次性准备滚动样本收集使用的纯数值索引。

本脚本把原先留在主计算中的下列工作全部提前完成：

1. 区域字符串清洗和 region_index 映射；
2. 历史订单按完成时间排序；
3. 同 OD 历史的 CSR 索引；
4. 第 4 层使用的“起点/全局 + 日期类型 + 时间位置”CSR 索引；
5. 工作日、普通周末、节假日三类日期标记；
6. 目标表的时间字段和区域编号派生。

其中 CSR（compressed sparse row）由一个 order 数组和一个 offsets 数组
组成。后续脚本通过整数切片直接取得候选，不再运行 pandas groupby，也
不再扫描某个起点的全部历史订单。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
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
    parser = argparse.ArgumentParser(description="准备 p^e_ijt 滚动计算数值索引。")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="仅覆盖本脚本管理的 prepared_index 文件。",
    )
    return parser.parse_args()


def clean_region_id(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    cleaned = cleaned.str.replace(r"\.0$", "", regex=True)
    cleaned = cleaned.str.extract(r"(\d+)", expand=False)
    return cleaned.str.zfill(11)


def managed_outputs() -> list[Path]:
    return [
        cfg.TARGET_PREPARED_FILE,
        cfg.HISTORY_COMPLETION_NS_FILE,
        cfg.HISTORY_I_FILE,
        cfg.HISTORY_J_FILE,
        cfg.HISTORY_DAY_FILE,
        cfg.HISTORY_SLOT_FILE,
        cfg.HISTORY_DOW_FILE,
        cfg.HISTORY_WEEKEND_HOLIDAY_FILE,
        cfg.HISTORY_DATE_TYPE_FILE,
        cfg.HISTORY_PRICE_FILE,
        cfg.HISTORY_TRIP_MILES_FILE,
        cfg.HISTORY_TRIP_SECONDS_FILE,
        cfg.OD_ORDER_FILE,
        cfg.OD_OFFSETS_FILE,
        cfg.ORIGIN_TIME_STRICT_ORDER_FILE,
        cfg.ORIGIN_TIME_STRICT_OFFSETS_FILE,
        cfg.ORIGIN_TIME_RELAXED_ORDER_FILE,
        cfg.ORIGIN_TIME_RELAXED_OFFSETS_FILE,
        cfg.ORIGIN_DATE_TYPE_TIME_ORDER_FILE,
        cfg.ORIGIN_DATE_TYPE_TIME_OFFSETS_FILE,
        cfg.GLOBAL_DATE_TYPE_TIME_ORDER_FILE,
        cfg.GLOBAL_DATE_TYPE_TIME_OFFSETS_FILE,
    ]


def build_dense_csr(keys: np.ndarray, group_count: int) -> tuple[np.ndarray, np.ndarray]:
    """把非负整数 key 转成 CSR；组内保持原完成时间顺序。"""

    keys = np.asarray(keys, dtype=np.int64)
    if len(keys) and (int(keys.min()) < 0 or int(keys.max()) >= group_count):
        raise ValueError("CSR key 超出预定范围。")
    # history 已按完成时间升序。stable 排序只按 key 重排，因此每个 key
    # 内仍保持完成时间升序，后续可直接 searchsorted(completion < t)。
    order = np.argsort(keys, kind="stable").astype(np.int32, copy=False)
    counts = np.bincount(keys, minlength=group_count).astype(np.int64, copy=False)
    offsets = np.empty(group_count + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    return order, offsets


def build_history_arrays(region_to_index: dict[str, int]) -> dict[str, int]:
    columns = [
        "i",
        "j",
        "completion_time",
        "start_day_index",
        "slot_in_day",
        "dow",
        "is_weekend_or_holiday",
        "price",
        "trip_miles",
        "trip_seconds",
    ]
    history = pd.read_parquet(cfg.ORDER_HISTORY_FILE, columns=columns)
    input_rows = len(history)
    history["i_index"] = clean_region_id(history["i"]).map(region_to_index)
    history["j_index"] = clean_region_id(history["j"]).map(region_to_index)
    # Parquet 可能恢复为 datetime64[us]，而 pandas.Timestamp.value 固定为
    # 纳秒。统一转成 datetime64[ns] 后再取整数，防止 1000 倍单位差造成
    # completion_time < t 判断把未来订单误当成历史。
    history["completion_time"] = pd.to_datetime(
        history["completion_time"], errors="coerce"
    ).astype("datetime64[ns]")
    history["price"] = pd.to_numeric(history["price"], errors="coerce")
    history["trip_miles"] = pd.to_numeric(history["trip_miles"], errors="coerce")
    history["trip_seconds"] = pd.to_numeric(history["trip_seconds"], errors="coerce")
    history = history.dropna(
        subset=["i_index", "j_index", "completion_time", "price"]
    ).copy()
    history = history.loc[history["price"] > 0].copy()
    history = history.sort_values("completion_time", kind="mergesort").reset_index(
        drop=True
    )

    i_index = history["i_index"].to_numpy(dtype=np.int16)
    j_index = history["j_index"].to_numpy(dtype=np.int16)
    completion_ns = history["completion_time"].astype("int64").to_numpy()
    day = history["start_day_index"].to_numpy(dtype=np.int16)
    slot = history["slot_in_day"].to_numpy(dtype=np.int16)
    dow = history["dow"].to_numpy(dtype=np.int8)
    weekend_holiday = history["is_weekend_or_holiday"].to_numpy(dtype=np.int8)
    data_start = pd.Timestamp(cfg.DATA_START).normalize()
    holiday_day_indices = {
        int((pd.Timestamp(value).normalize() - data_start).days)
        for value in cfg.HOLIDAY_DATES
    }
    date_type = np.where(dow >= 5, cfg.DATE_TYPE_WEEKEND, cfg.DATE_TYPE_WEEKDAY)
    if holiday_day_indices:
        date_type = np.where(
            np.isin(day, list(holiday_day_indices)),
            cfg.DATE_TYPE_HOLIDAY,
            date_type,
        )
    date_type = date_type.astype(np.int8, copy=False)
    price = history["price"].to_numpy(dtype=np.float32)
    trip_miles = history["trip_miles"].to_numpy(dtype=np.float32)
    trip_seconds = history["trip_seconds"].to_numpy(dtype=np.float32)

    np.save(cfg.HISTORY_COMPLETION_NS_FILE, completion_ns)
    np.save(cfg.HISTORY_I_FILE, i_index)
    np.save(cfg.HISTORY_J_FILE, j_index)
    np.save(cfg.HISTORY_DAY_FILE, day)
    np.save(cfg.HISTORY_SLOT_FILE, slot)
    np.save(cfg.HISTORY_DOW_FILE, dow)
    np.save(cfg.HISTORY_WEEKEND_HOLIDAY_FILE, weekend_holiday)
    np.save(cfg.HISTORY_DATE_TYPE_FILE, date_type)
    np.save(cfg.HISTORY_PRICE_FILE, price)
    np.save(cfg.HISTORY_TRIP_MILES_FILE, trip_miles)
    np.save(cfg.HISTORY_TRIP_SECONDS_FILE, trip_seconds)

    od_code = (
        i_index.astype(np.int64) * cfg.REGION_COUNT
        + j_index.astype(np.int64)
    )
    od_group_count = cfg.REGION_COUNT * cfg.REGION_COUNT
    od_order, od_offsets = build_dense_csr(od_code, od_group_count)
    np.save(cfg.OD_ORDER_FILE, od_order)
    np.save(cfg.OD_OFFSETS_FILE, od_offsets)
    del od_order, od_offsets, od_code

    strict_key = (
        (
            (
                i_index.astype(np.int64) * cfg.SLOTS_PER_DAY
                + slot.astype(np.int64)
            )
            * cfg.DOW_COUNT
            + dow.astype(np.int64)
        )
        * cfg.WEEKEND_HOLIDAY_STATE_COUNT
        + weekend_holiday.astype(np.int64)
    )
    strict_group_count = (
        cfg.REGION_COUNT
        * cfg.SLOTS_PER_DAY
        * cfg.DOW_COUNT
        * cfg.WEEKEND_HOLIDAY_STATE_COUNT
    )
    strict_order, strict_offsets = build_dense_csr(strict_key, strict_group_count)
    np.save(cfg.ORIGIN_TIME_STRICT_ORDER_FILE, strict_order)
    np.save(cfg.ORIGIN_TIME_STRICT_OFFSETS_FILE, strict_offsets)
    del strict_order, strict_offsets, strict_key

    relaxed_key = (
        (i_index.astype(np.int64) * cfg.SLOTS_PER_DAY + slot.astype(np.int64))
        * cfg.DOW_COUNT
        + dow.astype(np.int64)
    )
    relaxed_group_count = cfg.REGION_COUNT * cfg.SLOTS_PER_DAY * cfg.DOW_COUNT
    relaxed_order, relaxed_offsets = build_dense_csr(
        relaxed_key, relaxed_group_count
    )
    np.save(cfg.ORIGIN_TIME_RELAXED_ORDER_FILE, relaxed_order)
    np.save(cfg.ORIGIN_TIME_RELAXED_OFFSETS_FILE, relaxed_offsets)

    origin_date_type_time_key = (
        (
            i_index.astype(np.int64) * cfg.DATE_TYPE_COUNT
            + date_type.astype(np.int64)
        )
        * cfg.SLOTS_PER_DAY
        + slot.astype(np.int64)
    )
    origin_date_type_time_groups = (
        cfg.REGION_COUNT * cfg.DATE_TYPE_COUNT * cfg.SLOTS_PER_DAY
    )
    origin_dt_order, origin_dt_offsets = build_dense_csr(
        origin_date_type_time_key,
        origin_date_type_time_groups,
    )
    np.save(cfg.ORIGIN_DATE_TYPE_TIME_ORDER_FILE, origin_dt_order)
    np.save(cfg.ORIGIN_DATE_TYPE_TIME_OFFSETS_FILE, origin_dt_offsets)

    global_date_type_time_key = (
        date_type.astype(np.int64) * cfg.SLOTS_PER_DAY + slot.astype(np.int64)
    )
    global_date_type_time_groups = cfg.DATE_TYPE_COUNT * cfg.SLOTS_PER_DAY
    global_dt_order, global_dt_offsets = build_dense_csr(
        global_date_type_time_key,
        global_date_type_time_groups,
    )
    np.save(cfg.GLOBAL_DATE_TYPE_TIME_ORDER_FILE, global_dt_order)
    np.save(cfg.GLOBAL_DATE_TYPE_TIME_OFFSETS_FILE, global_dt_offsets)

    return {
        "history_input_rows": int(input_rows),
        "history_rows": int(len(history)),
        "od_group_count": int(od_group_count),
        "strict_origin_time_group_count": int(strict_group_count),
        "relaxed_origin_time_group_count": int(relaxed_group_count),
        "origin_date_type_time_group_count": int(origin_date_type_time_groups),
        "global_date_type_time_group_count": int(global_date_type_time_groups),
    }


def build_target_table(region_to_index: dict[str, int]) -> dict[str, int]:
    source = pq.ParquetFile(cfg.TARGET_OD_FILE)
    required = ["i", "j", "t", "D_ijt_observed"]
    missing = sorted(set(required) - set(source.schema.names))
    if missing:
        raise ValueError(f"目标 OD 文件缺少字段：{missing}")

    temporary = cfg.TARGET_PREPARED_FILE.with_suffix(".tmp.parquet")
    if temporary.exists():
        temporary.unlink()
    writer: pq.ParquetWriter | None = None
    input_rows = 0
    output_rows = 0
    invalid_rows = 0
    data_start = pd.Timestamp(cfg.DATA_START).normalize()
    holiday_dates = {pd.Timestamp(value).date() for value in cfg.HOLIDAY_DATES}

    try:
        for batch in source.iter_batches(
            batch_size=cfg.TARGET_BATCH_SIZE,
            columns=required,
        ):
            data = batch.to_pandas()
            input_rows += len(data)
            data["i"] = clean_region_id(data["i"])
            data["j"] = clean_region_id(data["j"])
            data["t"] = pd.to_datetime(data["t"], errors="coerce").astype(
                "datetime64[ns]"
            )
            data["D_ijt_observed"] = pd.to_numeric(
                data["D_ijt_observed"], errors="coerce"
            )
            data["i_index"] = data["i"].map(region_to_index)
            data["j_index"] = data["j"].map(region_to_index)
            valid = (
                data["i_index"].notna()
                & data["j_index"].notna()
                & data["t"].notna()
                & data["D_ijt_observed"].notna()
                & (data["D_ijt_observed"] > 0)
            )
            invalid_rows += int((~valid).sum())
            data = data.loc[valid].copy()
            if data.empty:
                continue

            data["target_id"] = np.arange(
                output_rows, output_rows + len(data), dtype=np.int64
            )
            data["target_ns"] = data["t"].astype("int64")
            data["target_day"] = (
                data["t"].dt.normalize() - data_start
            ).dt.days.astype("int16")
            data["target_slot"] = (
                data["t"].dt.hour * 4
                + data["t"].dt.minute // cfg.TIME_STEP_MINUTES
            ).astype("int16")
            data["target_dow"] = data["t"].dt.dayofweek.astype("int8")
            data["target_is_holiday"] = data["t"].dt.date.isin(
                holiday_dates
            ).astype("int8")
            data["target_weekend_or_holiday"] = (
                (data["target_dow"] >= 5) | (data["target_is_holiday"] == 1)
            ).astype("int8")
            data["target_date_type"] = np.where(
                data["target_is_holiday"] == 1,
                cfg.DATE_TYPE_HOLIDAY,
                np.where(
                    data["target_dow"] >= 5,
                    cfg.DATE_TYPE_WEEKEND,
                    cfg.DATE_TYPE_WEEKDAY,
                ),
            ).astype("int8")
            data["i_index"] = data["i_index"].astype("int16")
            data["j_index"] = data["j_index"].astype("int16")
            data["D_ijt_observed"] = data["D_ijt_observed"].astype("float32")

            result = data[
                [
                    "target_id",
                    "i",
                    "j",
                    "i_index",
                    "j_index",
                    "t",
                    "target_ns",
                    "target_day",
                    "target_slot",
                    "target_dow",
                    "target_is_holiday",
                    "target_weekend_or_holiday",
                    "target_date_type",
                    "D_ijt_observed",
                ]
            ]
            table = pa.Table.from_pandas(result, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary,
                    table.schema,
                    compression=cfg.PARQUET_COMPRESSION,
                )
            writer.write_table(table)
            output_rows += len(result)
    finally:
        if writer is not None:
            writer.close()

    if output_rows == 0:
        raise ValueError("目标表没有可用记录。")
    temporary.replace(cfg.TARGET_PREPARED_FILE)
    return {
        "target_input_rows": int(input_rows),
        "target_rows": int(output_rows),
        "target_invalid_rows": int(invalid_rows),
    }


def main() -> None:
    args = parse_args()
    cfg.ensure_output_directories()
    required = [
        cfg.ORDER_HISTORY_FILE,
        cfg.REGION_CENTROID_FILE,
        cfg.NORMALIZED_NEIGHBOR_FILE,
        cfg.NEIGHBOR_MATRIX_FILE,
        cfg.TARGET_OD_FILE,
        cfg.OD_ROAD_DISTANCE_FILE,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少上游输入：\n" + "\n".join(missing))

    existing = [path for path in managed_outputs() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "数值索引已存在；如确认重建，请使用 --overwrite：\n"
            + "\n".join(str(path) for path in existing)
        )
    if args.overwrite:
        for path in managed_outputs():
            if path.exists():
                path.unlink()

    centroids = pd.read_parquet(
        cfg.REGION_CENTROID_FILE,
        columns=["region_id", "region_index"],
    )
    centroids["region_id"] = clean_region_id(centroids["region_id"])
    region_to_index = dict(
        zip(centroids["region_id"].astype(str), centroids["region_index"].astype(int))
    )
    if len(region_to_index) != cfg.REGION_COUNT:
        raise ValueError(
            f"区域索引数量为 {len(region_to_index)}，预期为 {cfg.REGION_COUNT}。"
        )

    print("正在准备历史纯数值数组和 CSR 索引……")
    diagnostics = build_history_arrays(region_to_index)
    print("正在准备目标单元表……")
    diagnostics.update(build_target_table(region_to_index))
    diagnostics["index_directory"] = str(cfg.INDEX_DIR)
    cfg.INDEX_DIAGNOSTICS_FILE.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
