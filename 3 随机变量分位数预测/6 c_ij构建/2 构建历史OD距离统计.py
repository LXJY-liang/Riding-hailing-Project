# -*- coding: utf-8 -*-
"""统计历史 OD 订单数和里程中位数，并构造收缩校准样本。

输入：逐笔订单 Parquet、801个区域编号。
输出：
1. historical_od_distance.parquet：每个已观测OD的 N_ij 和里程中位数；
2. shrinkage_training_samples.parquet：从高样本OD中抽取的伪稀疏样本。

伪稀疏样本把高样本OD的全样本中位数视为近似真值，再重复抽取
1、3、5、10、15、19笔订单。第4个脚本将利用它们估计数据驱动的收缩参数 kappa。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def load_config():
    path = Path(__file__).resolve().parent / "1 c_ij构建配置.py"
    spec = importlib.util.spec_from_file_location("cij_config", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


cfg = load_config()


def clean_region_id(series: pd.Series) -> pd.Series:
    """将区域编号统一为11位字符串，防止CSV数字格式造成匹配失败。"""

    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.extract(r"(\d+)", expand=False)
        .str.zfill(11)
    )


def main() -> None:
    cfg.ensure_output_directories()

    # 区域集合用于剔除起点或终点不在801个研究区域内的订单。
    regions = pd.read_csv(
        cfg.TRACT_FILE,
        usecols=["GEOID10"],
        dtype="string",
        encoding="utf-8-sig",
    )["GEOID10"].str.zfill(11)
    region_set = set(regions)

    # 只读取距离统计必需的字段，避免将约606万笔订单的无关列载入内存。
    columns = [
        "Trip End Timestamp",
        "Trip Miles",
        "Trip Seconds",
        "Pickup Census Tract",
        "Dropoff Census Tract",
    ]
    orders = pd.read_parquet(cfg.ORDER_FILE, columns=columns)
    orders = orders.rename(
        columns={
            "Pickup Census Tract": "i",
            "Dropoff Census Tract": "j",
            "Trip Miles": "trip_miles",
            "Trip Seconds": "trip_seconds",
            "Trip End Timestamp": "trip_end",
        }
    )
    orders["i"] = clean_region_id(orders["i"])
    orders["j"] = clean_region_id(orders["j"])

    # 有效口径：研究区域内、验证期前已完成、正里程、正时长，
    # 且平均速度在1—80 mph之间。后续核对脚本使用完全相同的口径。
    speed_mph = orders["trip_miles"] / (orders["trip_seconds"] / 3600.0)
    valid = (
        orders["i"].isin(region_set)
        & orders["j"].isin(region_set)
        & (orders["trip_end"] < pd.Timestamp(cfg.HISTORY_END))
        & (orders["trip_miles"] > 0)
        & (orders["trip_seconds"] > 0)
        & speed_mph.between(cfg.MIN_SPEED_MPH, cfg.MAX_SPEED_MPH)
    )
    orders = orders.loc[valid, ["i", "j", "trip_miles"]].copy()

    # N_ij决定六层中的层级；中位数比均值更不受少量异常长距离影响。
    od_stats = (
        orders.groupby(["i", "j"], observed=True, sort=False)
        .agg(
            N_ij=("trip_miles", "size"),
            d_hist_median_miles=("trip_miles", "median"),
        )
        .reset_index()
    )
    od_stats["N_ij"] = od_stats["N_ij"].astype("int32")
    od_stats["d_hist_median_miles"] = od_stats[
        "d_hist_median_miles"
    ].astype("float32")
    od_stats.to_parquet(
        cfg.HISTORICAL_OD_FILE,
        index=False,
        compression=cfg.PARQUET_COMPRESSION,
    )

    # 只对 N_ij >= 50 的OD生成伪稀疏样本，以保证全样本中位数足够稳定。
    eligible = od_stats.loc[
        od_stats["N_ij"] >= cfg.PSEUDO_TRUTH_THRESHOLD,
        ["i", "j"],
    ]
    eligible_orders = orders.merge(eligible, on=["i", "j"], how="inner")
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    rows = []
    for (i, j), group in eligible_orders.groupby(["i", "j"], sort=False):
        values = group["trip_miles"].to_numpy(dtype=np.float64)
        truth = float(np.median(values))
        # 每个样本量重复抽取多次，降低单次随机抽样对 kappa 的影响。
        for sample_size in cfg.SHRINKAGE_SAMPLE_SIZES:
            for repeat in range(cfg.SHRINKAGE_REPEATS):
                sample = rng.choice(values, size=sample_size, replace=False)
                rows.append(
                    (
                        i,
                        j,
                        len(values),
                        sample_size,
                        repeat,
                        float(np.median(sample)),
                        truth,
                    )
                )

    shrinkage_samples = pd.DataFrame(
        rows,
        columns=[
            "i",
            "j",
            "N_full",
            "sample_size",
            "repeat",
            "d_sample_median_miles",
            "d_truth_median_miles",
        ],
    )
    shrinkage_samples.to_parquet(
        cfg.SHRINKAGE_SAMPLE_FILE,
        index=False,
        compression=cfg.PARQUET_COMPRESSION,
    )


if __name__ == "__main__":
    main()
