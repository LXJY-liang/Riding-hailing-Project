# -*- coding: utf-8 -*-
"""把LLM预测的成交量分位数除以历史综合完成率，恢复潜在需求分布。"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent


def load_config():
    spec = importlib.util.spec_from_file_location("stage1_config", HERE / "1 第一阶段SAC配置.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


cfg = load_config()


def source_for_split(split: str) -> Path:
    exact = cfg.DIT_PREDICTION_DIR / f"dit_{split}_predictions.parquet"
    matches = [exact] if exact.exists() else sorted(
        cfg.DIT_PREDICTION_DIR.glob(f"dit_{split}_predictions*.parquet")
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"{cfg.DIT_PREDICTION_DIR}中缺少唯一的dit_{split}_predictions.parquet。"
        )
    return matches[0]


def convert_split(split: str, batch_size: int, overwrite: bool) -> dict:
    source = source_for_split(split)
    target = cfg.POTENTIAL_DEMAND_DIR / f"dit_potential_{split}_predictions.parquet"
    if target.exists() and not overwrite:
        raise FileExistsError(f"目标已存在：{target}；需要覆盖时加入--overwrite。")
    parquet = pq.ParquetFile(source)
    schema_names = parquet.schema.names
    missing = [name for name in ["region_id", "t", *cfg.QUANTILE_COLUMNS] if name not in schema_names]
    if missing:
        raise ValueError(f"{source.name}缺少字段：{missing[:5]}")

    writer = None
    row_count = 0
    try:
        for batch in parquet.iter_batches(
            batch_size=batch_size, columns=["region_id", "t", *cfg.QUANTILE_COLUMNS]
        ):
            table = pa.Table.from_batches([batch])
            arrays = [table["region_id"], table["t"]]
            previous = None
            for column in cfg.QUANTILE_COLUMNS:
                values = table[column].to_numpy(zero_copy_only=False).astype(np.float32)
                if not np.isfinite(values).all() or (values < 0).any():
                    raise ValueError(f"{source.name}的{column}含负值或NaN/inf。")
                values = values / cfg.DEMAND_COMPLETION_RATE
                if previous is not None and (values + 1e-5 < previous).any():
                    raise ValueError(f"{source.name}的分位数曲线不是单调非降。")
                previous = values
                arrays.append(pa.array(values, type=pa.float32()))
            arrays.append(
                pa.array(
                    np.full(len(batch), cfg.DEMAND_COMPLETION_RATE, dtype=np.float32),
                    type=pa.float32(),
                )
            )
            output = pa.Table.from_arrays(
                arrays,
                names=["region_id", "t", *cfg.QUANTILE_COLUMNS, "completion_rate"],
            )
            if writer is None:
                writer = pq.ParquetWriter(target, output.schema, compression="snappy")
            writer.write_table(output)
            row_count += len(batch)
    finally:
        if writer is not None:
            writer.close()
    return {
        "split": split,
        "source": str(source),
        "target": str(target),
        "rows": row_count,
        "completion_rate": cfg.DEMAND_COMPLETION_RATE,
        "scale_factor": 1.0 / cfg.DEMAND_COMPLETION_RATE,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", choices=tuple(cfg.SPLITS), default=list(cfg.SPLITS))
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0.0 < cfg.DEMAND_COMPLETION_RATE <= 1.0:
        raise ValueError("DEMAND_COMPLETION_RATE必须位于(0,1]。")
    cfg.ensure_directories()
    reports = [convert_split(split, args.batch_size, args.overwrite) for split in args.splits]
    report_file = cfg.POTENTIAL_DEMAND_DIR / "conversion_metadata.json"
    report_file.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    for report in reports:
        print(
            f"{report['split']}：{report['rows']:,}行，"
            f"完成率={report['completion_rate']:.3f}，倍率={report['scale_factor']:.4f}"
        )


if __name__ == "__main__":
    main()
