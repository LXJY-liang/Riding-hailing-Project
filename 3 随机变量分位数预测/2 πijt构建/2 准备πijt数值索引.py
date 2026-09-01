# -*- coding: utf-8 -*-
"""一次性准备 π_ijt 滚动构建所需的纯数值索引。

本脚本把耐时的数据整理工作全部提前完成：

1. 从完整区域面板确定 801 个区域和 2880 个目标时段；
2. 统一 Census Tract 编号和时间索引；
3. 把有订单 OD 聚合表转成 (i,t)×j 的 CSR 订单数矩阵；
4. 保存 slot、dow、holiday 等纯数值数组；
5. 记录内部 OD 加总与区域面板 D_origin_it 的口径差异。

注意：区域面板的 D_origin_it 还包含“从 801 个区域内部出发、
但目的地不在这 801 个区域内”的订单。π_ijt 的目的地集合只有
801 个内部区域，因此其分子和分母都必须使用内部 OD 订单口径。

后续正式构建脚本不再执行 pandas groupby、字符串匹配或读取
5.9 亿行的补全 OD 面板。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse


def load_config():
    path = Path(__file__).resolve().parent / "1 πijt构建配置.py"
    spec = importlib.util.spec_from_file_location("pi_build_config", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


cfg = load_config()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备 π_ijt 数值索引。")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="仅覆盖本脚本管理的 prepared_index 文件。",
    )
    return parser.parse_args()


def clean_region_id(series: pd.Series) -> pd.Series:
    """将 Census Tract 编号统一为 11 位字符串。"""

    cleaned = series.astype("string").str.strip()
    cleaned = cleaned.str.replace(r"\.0$", "", regex=True)
    cleaned = cleaned.str.extract(r"(\d+)", expand=False)
    return cleaned.str.zfill(11)


def managed_outputs() -> list[Path]:
    return [
        cfg.REGION_MAPPING_FILE,
        cfg.TIME_MAPPING_FILE,
        cfg.HISTORY_COUNT_MATRIX_FILE,
        cfg.HISTORY_ROW_TOTAL_FILE,
        cfg.TIME_SLOT_FILE,
        cfg.TIME_DOW_FILE,
        cfg.TIME_HOLIDAY_FILE,
        cfg.INDEX_DIAGNOSTICS_FILE,
    ]


def prepare_panel() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, int],
    dict[pd.Timestamp, int],
    np.ndarray,
]:
    """构造区域映射、时间映射和面板区域出发量。"""

    required = [
        "region_id",
        "t",
        "day_index",
        "slot_in_day",
        "dow",
        "is_weekend_or_holiday",
        "D_origin_it",
    ]
    panel = pd.read_parquet(cfg.REGION_PANEL_FILE, columns=required)
    panel["region_id"] = clean_region_id(panel["region_id"])
    panel["t"] = pd.to_datetime(panel["t"], errors="coerce").astype(
        "datetime64[ns]"
    )
    panel["D_origin_it"] = pd.to_numeric(
        panel["D_origin_it"], errors="coerce"
    )
    if panel[required].isna().any().any():
        missing = panel[required].isna().sum()
        raise ValueError(
            "区域面板存在缺失值：\n"
            + missing[missing > 0].to_string()
        )
    if panel[["region_id", "t"]].duplicated().any():
        raise ValueError("区域面板存在重复 (region_id,t)。")

    regions = pd.Index(panel["region_id"].unique()).sort_values()
    times = pd.DatetimeIndex(panel["t"].unique()).sort_values()
    if len(regions) != cfg.EXPECTED_REGION_COUNT:
        raise ValueError(
            f"区域数为 {len(regions)}，预期为 {cfg.EXPECTED_REGION_COUNT}。"
        )
    if len(times) != cfg.EXPECTED_TIME_COUNT:
        raise ValueError(
            f"时段数为 {len(times)}，预期为 {cfg.EXPECTED_TIME_COUNT}。"
        )

    expected_times = pd.date_range(
        cfg.DATA_START,
        cfg.DATA_END,
        freq=f"{cfg.TIME_STEP_MINUTES}min",
    )
    if not times.equals(expected_times):
        raise ValueError("区域面板的时间轴不是完整连续的 15 分钟序列。")
    expected_rows = len(regions) * len(times)
    if len(panel) != expected_rows:
        raise ValueError(
            f"区域面板行数为 {len(panel):,}，预期为 {expected_rows:,}。"
        )

    region_map = pd.DataFrame(
        {
            "region_index": np.arange(len(regions), dtype=np.int16),
            "region_id": regions.astype("string"),
        }
    )
    region_to_index = dict(
        zip(region_map["region_id"].astype(str), region_map["region_index"].astype(int))
    )
    time_to_index = {pd.Timestamp(value): index for index, value in enumerate(times)}

    holiday_dates = {pd.Timestamp(value).date() for value in cfg.HOLIDAY_DATES}
    time_map = pd.DataFrame(
        {
            "time_index": np.arange(len(times), dtype=np.int16),
            "t": times,
        }
    )
    time_map["day_index"] = (
        time_map["t"].dt.normalize() - pd.Timestamp(cfg.DATA_START).normalize()
    ).dt.days.astype("int16")
    time_map["slot_in_day"] = (
        time_map["t"].dt.hour * 4
        + time_map["t"].dt.minute // cfg.TIME_STEP_MINUTES
    ).astype("int16")
    time_map["dow"] = time_map["t"].dt.dayofweek.astype("int8")
    time_map["is_holiday"] = time_map["t"].dt.date.isin(
        holiday_dates
    ).astype("int8")
    time_map["is_weekend_or_holiday"] = (
        (time_map["dow"] >= 5) | (time_map["is_holiday"] == 1)
    ).astype("int8")

    panel["region_index"] = panel["region_id"].map(region_to_index).astype("int16")
    panel["time_index"] = panel["t"].map(time_to_index).astype("int16")
    linear_id = (
        panel["region_index"].to_numpy(dtype=np.int64) * len(times)
        + panel["time_index"].to_numpy(dtype=np.int64)
    )
    panel_origin_total = np.zeros(expected_rows, dtype=np.int64)
    panel_origin_total[linear_id] = panel["D_origin_it"].to_numpy(dtype=np.int64)

    return region_map, time_map, region_to_index, time_to_index, panel_origin_total


def build_history_count_matrix(
    region_to_index: dict[str, int],
    time_to_index: dict[pd.Timestamp, int],
) -> tuple[sparse.csr_matrix, dict[str, int]]:
    """流式读取有订单 OD，构造 (i,t)×j 稀疏订单数矩阵。"""

    source = pq.ParquetFile(cfg.ACTIVE_OD_FILE)
    required = ["i", "j", "t", "D_ijt_observed"]
    missing = sorted(set(required) - set(source.schema.names))
    if missing:
        raise ValueError(f"有订单 OD 表缺少字段：{missing}")

    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    input_rows = 0
    invalid_rows = 0

    for batch_number, batch in enumerate(
        source.iter_batches(
            batch_size=cfg.PARQUET_BATCH_SIZE,
            columns=required,
        ),
        start=1,
    ):
        data = batch.to_pandas()
        input_rows += len(data)
        data["i"] = clean_region_id(data["i"])
        data["j"] = clean_region_id(data["j"])
        data["t"] = pd.to_datetime(data["t"], errors="coerce").astype(
            "datetime64[ns]"
        )
        data["count"] = pd.to_numeric(data["D_ijt_observed"], errors="coerce")
        data["i_index"] = data["i"].map(region_to_index)
        data["j_index"] = data["j"].map(region_to_index)
        data["time_index"] = data["t"].map(time_to_index)
        valid = (
            data["i_index"].notna()
            & data["j_index"].notna()
            & data["time_index"].notna()
            & data["count"].notna()
            & (data["count"] > 0)
        )
        invalid_rows += int((~valid).sum())
        data = data.loc[valid]
        if data.empty:
            continue

        i_index = data["i_index"].to_numpy(dtype=np.int64)
        time_index = data["time_index"].to_numpy(dtype=np.int64)
        row_parts.append(i_index * cfg.EXPECTED_TIME_COUNT + time_index)
        col_parts.append(data["j_index"].to_numpy(dtype=np.int32))
        value_parts.append(data["count"].to_numpy(dtype=np.int32))
        print(
            f"batch {batch_number}: 已读取 {input_rows:,} 行，"
            f"已保留 {sum(len(x) for x in value_parts):,} 行"
        )

    if not value_parts:
        raise ValueError("有订单 OD 表没有产生任何有效历史订单。")

    matrix = sparse.coo_matrix(
        (
            np.concatenate(value_parts),
            (np.concatenate(row_parts), np.concatenate(col_parts)),
        ),
        shape=(
            cfg.EXPECTED_REGION_COUNT * cfg.EXPECTED_TIME_COUNT,
            cfg.EXPECTED_REGION_COUNT,
        ),
        dtype=np.int32,
    ).tocsr()
    matrix.sum_duplicates()
    matrix.sort_indices()
    if np.any(matrix.data <= 0):
        raise AssertionError("历史订单数矩阵出现非正数据。")

    return matrix, {
        "active_od_input_rows": int(input_rows),
        "active_od_invalid_rows": int(invalid_rows),
        "history_matrix_nnz": int(matrix.nnz),
        "history_total_orders": int(matrix.data.sum(dtype=np.int64)),
    }


def main() -> None:
    args = parse_args()
    cfg.ensure_output_directories()
    for path in (cfg.ACTIVE_OD_FILE, cfg.REGION_PANEL_FILE):
        if not path.exists():
            raise FileNotFoundError(f"缺少上游输入：{path}")

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

    print("正在准备区域和时间映射……")
    (
        region_map,
        time_map,
        region_to_index,
        time_to_index,
        panel_origin_total,
    ) = prepare_panel()

    print("正在构造 (i,t)×j 历史订单数 CSR 矩阵……")
    count_matrix, diagnostics = build_history_count_matrix(
        region_to_index,
        time_to_index,
    )
    row_total = np.asarray(count_matrix.sum(axis=1)).ravel().astype(np.int64)
    difference = panel_origin_total - row_total
    mismatch = difference != 0
    mismatch_count = int(mismatch.sum())
    mismatch_max_abs = int(np.max(np.abs(difference[mismatch]))) if mismatch_count else 0
    internal_excess = difference < 0
    internal_excess_count = int(internal_excess.sum())
    if internal_excess_count:
        internal_excess_max = int((-difference[internal_excess]).max())
        raise AssertionError(
            "内部 OD 订单数不应超过区域面板 D_origin_it："
            f"{internal_excess_count:,} 行超出，最大超出 {internal_excess_max}。"
        )

    region_map.to_csv(cfg.REGION_MAPPING_FILE, index=False)
    time_map.to_parquet(
        cfg.TIME_MAPPING_FILE,
        index=False,
        compression=cfg.PARQUET_COMPRESSION,
    )
    sparse.save_npz(cfg.HISTORY_COUNT_MATRIX_FILE, count_matrix, compressed=True)
    np.save(cfg.HISTORY_ROW_TOTAL_FILE, row_total)
    np.save(
        cfg.TIME_SLOT_FILE,
        time_map["slot_in_day"].to_numpy(dtype=np.int16),
    )
    np.save(cfg.TIME_DOW_FILE, time_map["dow"].to_numpy(dtype=np.int8))
    np.save(
        cfg.TIME_HOLIDAY_FILE,
        time_map["is_holiday"].to_numpy(dtype=np.int8),
    )

    diagnostics.update(
        {
            "region_count": int(len(region_map)),
            "time_count": int(len(time_map)),
            "target_origin_time_rows": int(len(row_total)),
            "positive_origin_time_rows": int((row_total > 0).sum()),
            "panel_total_orders": int(panel_origin_total.sum()),
            "internal_od_total_orders": int(row_total.sum()),
            "excluded_external_destination_orders": int(difference.sum()),
            "panel_origin_total_mismatch_rows": mismatch_count,
            "panel_origin_total_max_abs_difference": mismatch_max_abs,
            "internal_od_exceeds_panel_rows": internal_excess_count,
            "history_matrix_shape": [int(value) for value in count_matrix.shape],
            "index_directory": str(cfg.INDEX_DIR),
        }
    )
    cfg.INDEX_DIAGNOSTICS_FILE.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
