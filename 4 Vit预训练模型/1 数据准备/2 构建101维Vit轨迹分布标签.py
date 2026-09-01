# -*- coding: utf-8 -*-
"""将旧轨迹模拟的19维Oracle分位数单调插值为101维监督标签。"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("vit_pretrain_config", HERE / "1 Vit预训练配置.py")
cfg = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(cfg)


def interpolate_quantiles(values: np.ndarray) -> np.ndarray:
    """19维q05-q95线性插值到101维；两端使用最近可靠分位数。"""

    values = np.maximum.accumulate(np.maximum(values, 0.0), axis=1)
    result = np.empty((len(values), len(cfg.QUANTILE_LEVELS)), dtype=np.float32)
    for column, level in enumerate(cfg.QUANTILE_LEVELS):
        if level <= cfg.SOURCE_LEVELS[0]:
            result[:, column] = values[:, 0]
            continue
        if level >= cfg.SOURCE_LEVELS[-1]:
            result[:, column] = values[:, -1]
            continue
        right = int(np.searchsorted(cfg.SOURCE_LEVELS, level, side="right"))
        left = right - 1
        weight = float(
            (level - cfg.SOURCE_LEVELS[left])
            / (cfg.SOURCE_LEVELS[right] - cfg.SOURCE_LEVELS[left])
        )
        result[:, column] = values[:, left] * (1.0 - weight) + values[:, right] * weight
    return result


def build_split(split: str, batch_size: int) -> None:
    start, end = map(pd.Timestamp, cfg.SPLITS[split])
    source = pq.ParquetFile(cfg.SOURCE_VIT_DISTRIBUTION)
    writer = None
    rows = 0
    try:
        for batch in source.iter_batches(
            batch_size=batch_size,
            columns=["region_id", "region_index", "t"] + cfg.SOURCE_ORACLE_COLUMNS,
        ):
            frame = batch.to_pandas()
            frame["t"] = pd.to_datetime(frame["t"])
            frame = frame.loc[frame["t"].between(start, end)]
            if frame.empty:
                continue
            quantiles = interpolate_quantiles(
                frame[cfg.SOURCE_ORACLE_COLUMNS].to_numpy(np.float32)
            )
            output = pd.concat(
                [
                    frame[["region_id", "region_index", "t"]].reset_index(drop=True),
                    pd.DataFrame(quantiles, columns=cfg.QUANTILE_COLUMNS),
                ],
                axis=1,
            )
            table = pa.Table.from_pandas(output, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(cfg.LABEL_FILES[split], table.schema, compression="snappy")
            writer.write_table(table)
            rows += len(output)
    finally:
        if writer is not None:
            writer.close()
    if rows == 0:
        raise ValueError(f"{split}时间范围没有Vit标签。")
    print(f"{split}：已保存 {rows:,} 行101维Vit标签。")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", choices=tuple(cfg.SPLITS), default=list(cfg.SPLITS))
    parser.add_argument("--batch-size", type=int, default=50_000)
    args = parser.parse_args()
    cfg.ensure_directories()
    for split in args.splits:
        build_split(split, args.batch_size)


if __name__ == "__main__":
    main()
