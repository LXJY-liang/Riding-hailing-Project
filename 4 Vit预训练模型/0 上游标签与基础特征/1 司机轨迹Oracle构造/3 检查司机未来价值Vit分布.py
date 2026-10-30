# -*- coding: utf-8 -*-
"""只读检查新版 101 维 Vit 标签的完整性、数值范围和分位数单调性。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
path = HERE / "1 轨迹Oracle构造配置.py"
spec = importlib.util.spec_from_file_location("vit_oracle_config", path)
cfg = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(cfg)


def main() -> None:
    if not cfg.VIT_OUTPUT.exists():
        raise FileNotFoundError(cfg.VIT_OUTPUT)

    parquet = pq.ParquetFile(cfg.VIT_OUTPUT)
    expected_columns = {
        "region_id", "region_index", "t", "slot_in_day", *cfg.VIT_QUANTILE_COLUMNS
    }
    missing = expected_columns.difference(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"标签缺少字段：{sorted(missing)}")

    expected_start_times = len(
        pd.date_range(cfg.LABEL_START, cfg.LABEL_END, freq=f"{cfg.TIME_STEP_MINUTES}min")
    )
    expected_regions = len(pd.read_csv(cfg.REGION_MAP_FILE, usecols=["region_id"]))
    expected_rows = expected_start_times * expected_regions
    if parquet.metadata.num_rows != expected_rows:
        raise AssertionError(
            f"行数错误：实际 {parquet.metadata.num_rows:,}，预期 {expected_rows:,}。"
        )

    row_count = 0
    nonmonotone_count = 0
    nonfinite_count = 0
    grid_error_count = 0
    rows_with_negative_values = 0
    median_sum = 0.0
    minimum = np.inf
    maximum = -np.inf
    time_min: pd.Timestamp | None = None
    time_max: pd.Timestamp | None = None
    region_ids: set[str] = set()
    median_column = "oracle_V_q_0.500"

    scan_columns = ["region_id", "region_index", "t", *cfg.VIT_QUANTILE_COLUMNS]
    for batch in parquet.iter_batches(batch_size=32768, columns=scan_columns):
        frame = batch.to_pandas()
        values = frame[cfg.VIT_QUANTILE_COLUMNS].to_numpy(np.float64)
        positions = np.arange(row_count, row_count + len(frame), dtype=np.int64)
        expected_region_index = positions % expected_regions
        expected_times = (
            np.datetime64(cfg.LABEL_START, "ns")
            + (positions // expected_regions)
            * np.timedelta64(cfg.TIME_STEP_MINUTES, "m")
        )
        grid_error_count += int(
            (frame["region_index"].to_numpy(np.int64) != expected_region_index).sum()
        )
        grid_error_count += int(
            (frame["t"].to_numpy(dtype="datetime64[ns]") != expected_times).sum()
        )
        row_count += len(frame)
        nonfinite_count += int((~np.isfinite(values)).any(axis=1).sum())
        rows_with_negative_values += int((values < -1e-6).any(axis=1).sum())
        nonmonotone_count += int((np.diff(values, axis=1) < -1e-6).any(axis=1).sum())
        median_sum += float(frame[median_column].sum())
        minimum = min(minimum, float(np.nanmin(values)))
        maximum = max(maximum, float(np.nanmax(values)))
        batch_min = pd.Timestamp(frame["t"].min())
        batch_max = pd.Timestamp(frame["t"].max())
        time_min = batch_min if time_min is None else min(time_min, batch_min)
        time_max = batch_max if time_max is None else max(time_max, batch_max)
        region_ids.update(frame["region_id"].astype(str).unique())

    print(f"行数：{row_count:,}")
    print(f"区域数：{len(region_ids):,}")
    print(f"日期范围：{time_min} 至 {time_max}")
    print(f"分位数维度：{len(cfg.VIT_QUANTILE_COLUMNS)}")
    print(f"q50 均值：{median_sum / row_count:.6f}")
    print(f"所有分位数范围：[{minimum:.6f}, {maximum:.6f}]")
    print(f"含非有限值的行：{nonfinite_count:,}")
    print(f"区域—时段网格顺序错误数：{grid_error_count:,}")
    print(
        f"含负净收益分位数的行：{rows_with_negative_values:,}"
        "（历史成交轨迹可包含负净收益，仅报告不判为失败）"
    )
    print(f"分位数不单调的行：{nonmonotone_count:,}")

    if time_min != pd.Timestamp(cfg.LABEL_START) or time_max != pd.Timestamp(cfg.LABEL_END):
        raise AssertionError("标签日期范围与配置不一致。")
    if len(region_ids) != expected_regions:
        raise AssertionError("标签区域覆盖不完整。")
    if grid_error_count or nonfinite_count or nonmonotone_count:
        raise AssertionError("新版 Vit 标签数值检查失败。")


if __name__ == "__main__":
    main()
