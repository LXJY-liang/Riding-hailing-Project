# -*- coding: utf-8 -*-
"""仅根据已选历史样本构建 p^e_ijt 的 101 维混合 KDE 分位数。

本脚本不读取原始订单，不做区域匹配，不重复执行分层候选收集。输入中的
sample_prices、sample_levels、实际里程相似度、两个带宽及混合权重已由第 5 个脚本
准备完成。第 1—3 层构成相同 OD KDE，第 4A、4B、4C 合并为相似 OD KDE，
先混合两个 CDF，再反演得到 101 维分位数。
每个目标先用第5步给出的基准混合权重生成KDE。若“1—2笔稀疏同OD
样本+相似OD样本”造成 q30/q50 低于结构审计阈值，则本脚本直接按
同OD行程与目标路线的一致性连续折减同OD成分质量并重算分位数。
该判断对当前样本自动执行，不读取target_id覆盖表，因而从头重建可复现。
"""

from __future__ import annotations

import os

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(variable, "1")

import argparse
import importlib.util
import json
import multiprocessing as mp
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.special import ndtr


def load_config():
    path = Path(__file__).resolve().parent / "1 p^eijt分布构建配置.py"
    spec = importlib.util.spec_from_file_location("pe_distribution_config", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


cfg = load_config()
WEIGHT_LOOKUP = np.asarray(
    [0.0, cfg.LAYER_WEIGHTS[1], cfg.LAYER_WEIGHTS[2],
     cfg.LAYER_WEIGHTS[3], cfg.LAYER_WEIGHTS[4], cfg.LAYER_WEIGHTS[5]],
    dtype=np.float64,
)


def optional_float(value: object) -> float:
    """把 Parquet 空浮点值（to_pylist 后为 None）统一转换为 NaN。"""

    return np.nan if value is None else float(value)


def kde_quantiles(
    record: tuple[list[float], list[int], list[float], float, float, float, int]
) -> np.ndarray:
    """反演一个目标的相同 OD / 相似 OD 混合 KDE 的 101 个分位点。"""

    (
        prices_raw,
        levels_raw,
        similarity_weights_raw,
        same_od_bandwidth_raw,
        similar_od_bandwidth_raw,
        same_od_mixture_weight_raw,
        available,
    ) = record
    if not available:
        return np.full(len(cfg.QUANTILE_LEVELS), np.nan, dtype=np.float32)
    prices = np.asarray(prices_raw, dtype=np.float64)
    levels = np.asarray(levels_raw, dtype=np.int8)
    similarity_weights = np.asarray(similarity_weights_raw, dtype=np.float64)
    # 当某个目标只有一个 KDE 分量时，另一个带宽在 Parquet 中为空值，
    # Arrow 的 to_pylist() 会将其读取为 None。先转成 NaN，后续只对
    # 实际存在的分量验证带宽，避免 float(None) 报错。
    same_od_bandwidth = optional_float(same_od_bandwidth_raw)
    similar_od_bandwidth = optional_float(similar_od_bandwidth_raw)
    same_od_mixture_weight = optional_float(same_od_mixture_weight_raw)
    if (
        len(prices) == 0
        or len(prices) != len(levels)
        or len(prices) != len(similarity_weights)
        or np.any((levels < 1) | (levels > 5))
        or not np.isfinite(same_od_mixture_weight)
        or not 0.0 <= same_od_mixture_weight <= 1.0
    ):
        raise ValueError("样本价格、层级或混合权重无效。")

    same_od_mask = levels <= 3
    similar_od_mask = levels >= 4
    same_od_prices = prices[same_od_mask]
    similar_od_prices = prices[similar_od_mask]
    if len(same_od_prices):
        if not np.isfinite(same_od_bandwidth) or same_od_bandwidth <= 0:
            raise ValueError("相同 OD 样本存在，但相同 OD 带宽无效。")
        same_od_weights = WEIGHT_LOOKUP[levels[same_od_mask]]
        same_od_weight_sum = float(same_od_weights.sum())
    else:
        same_od_weights = np.empty(0, dtype=np.float64)
        same_od_weight_sum = 0.0
        if same_od_mixture_weight != 0.0:
            raise ValueError("没有相同 OD 样本时，其混合权重必须为 0。")

    similar_od_mixture_weight = 1.0 - same_od_mixture_weight
    if len(similar_od_prices):
        if not np.isfinite(similar_od_bandwidth) or similar_od_bandwidth <= 0:
            raise ValueError("相似 OD 样本存在，但相似 OD 带宽无效。")
        similar_od_weights = similarity_weights[similar_od_mask]
        if np.any(
            ~np.isfinite(similar_od_weights)
            | (similar_od_weights <= 0.0)
            | (similar_od_weights > 1.0 + 1e-6)
        ):
            raise ValueError("相似 OD 里程相似度权重无效。")
        similar_od_weight_sum = float(similar_od_weights.sum())
    elif similar_od_mixture_weight != 0.0:
        raise ValueError("没有相似 OD 样本时，其混合权重必须为 0。")
    else:
        similar_od_weights = np.empty(0, dtype=np.float64)
        similar_od_weight_sum = 0.0

    component_lowers: list[float] = []
    component_uppers: list[float] = []
    if len(same_od_prices) and same_od_mixture_weight > 0.0:
        component_lowers.append(
            float(same_od_prices.min() - cfg.KDE_BOUND_SIGMAS * same_od_bandwidth)
        )
        component_uppers.append(
            float(same_od_prices.max() + cfg.KDE_BOUND_SIGMAS * same_od_bandwidth)
        )
    if len(similar_od_prices) and similar_od_mixture_weight > 0.0:
        component_lowers.append(
            float(
                similar_od_prices.min()
                - cfg.KDE_BOUND_SIGMAS * similar_od_bandwidth
            )
        )
        component_uppers.append(
            float(
                similar_od_prices.max()
                + cfg.KDE_BOUND_SIGMAS * similar_od_bandwidth
            )
        )
    if not component_lowers:
        raise ValueError("没有混合权重大于 0 的 KDE 分量。")

    lower_bound = max(0.0, min(component_lowers))
    upper_bound = max(component_uppers)
    lower = np.full(len(cfg.QUANTILE_LEVELS), lower_bound, dtype=np.float64)
    upper = np.full(len(cfg.QUANTILE_LEVELS), upper_bound, dtype=np.float64)

    # 第 k 轮后区间宽度为初始宽度 / 2**k。20 轮不是重复估算
    # 20 次 KDE，而是用 20 次 CDF 比较同时定位全部 101 个分位点。
    for _ in range(cfg.KDE_BISECTION_STEPS):
        middle = 0.5 * (lower + upper)
        cdf = np.zeros(len(cfg.QUANTILE_LEVELS), dtype=np.float64)
        if len(same_od_prices) and same_od_mixture_weight > 0.0:
            same_od_z = (
                middle[:, None] - same_od_prices[None, :]
            ) / same_od_bandwidth
            same_od_cdf = ndtr(same_od_z) @ same_od_weights / same_od_weight_sum
            cdf += same_od_mixture_weight * same_od_cdf
        if len(similar_od_prices) and similar_od_mixture_weight > 0.0:
            similar_od_z = (
                middle[:, None] - similar_od_prices[None, :]
            ) / similar_od_bandwidth
            similar_od_cdf = (
                ndtr(similar_od_z) @ similar_od_weights / similar_od_weight_sum
            )
            cdf += similar_od_mixture_weight * similar_od_cdf
        move_right = cdf < cfg.QUANTILE_LEVELS
        lower = np.where(move_right, middle, lower)
        upper = np.where(move_right, upper, middle)

    result = np.maximum.accumulate(np.maximum(0.5 * (lower + upper), 0.0))
    return result.astype(np.float32)


def kde_chunk(records: list[tuple]) -> list[np.ndarray]:
    return [adaptive_kde_result(record) for record in records]


def split_records(records: list[tuple], part_count: int) -> list[list[tuple]]:
    part_count = max(1, min(part_count, len(records)))
    bounds = np.linspace(0, len(records), part_count + 1, dtype=int)
    return [
        records[int(bounds[k]) : int(bounds[k + 1])]
        for k in range(part_count)
        if bounds[k] < bounds[k + 1]
    ]


def sample_records(table: pa.Table) -> list[tuple]:
    prices = table["sample_prices"].to_pylist()
    levels = table["sample_levels"].to_pylist()
    similarity_weights = table["sample_similarity_weights"].to_pylist()
    same_od_bandwidth = table["same_od_bandwidth_usd"].to_pylist()
    similar_od_bandwidth = table["similar_od_bandwidth_usd"].to_pylist()
    same_od_mixture_weight = table["same_od_mixture_weight"].to_pylist()
    available = table["available_flag"].to_pylist()
    trip_miles = table["sample_trip_miles"].to_pylist()
    target_trip_miles = table["target_reference_trip_miles"].to_pylist()
    same_effective_n = table["same_od_effective_sample_size"].to_pylist()
    similar_effective_n = table["similar_od_effective_sample_size"].to_pylist()
    similar_mean_similarity = table["similar_od_mean_similarity"].to_pylist()
    same_count = table["same_od_sample_count"].to_pylist()
    similar_count = table["similar_od_sample_count"].to_pylist()
    return list(
        zip(
            prices,
            levels,
            similarity_weights,
            same_od_bandwidth,
            similar_od_bandwidth,
            same_od_mixture_weight,
            available,
            trip_miles,
            target_trip_miles,
            same_effective_n,
            similar_effective_n,
            similar_mean_similarity,
            same_count,
            similar_count,
        )
    )


def route_consistency_factor(
    levels_raw: list[int],
    trip_miles_raw: list[float],
    target_trip_miles_raw: object,
) -> float:
    """计算同OD成分与目标路线距离的一致性，范围为(0,1]。"""

    levels = np.asarray(levels_raw, dtype=np.int8)
    miles = np.asarray(trip_miles_raw, dtype=np.float64)
    same = levels <= 3
    target = optional_float(target_trip_miles_raw)
    if not same.any() or not np.isfinite(target) or target <= 0.0:
        return np.nan
    raw = WEIGHT_LOOKUP[levels[same]]
    consistency = np.exp(
        -np.abs(np.log(miles[same] / target))
        / float(np.log(cfg.TRIP_MILES_RATIO_MAX))
    )
    return float(np.average(consistency, weights=raw))


def maximum_final_sample_weight(
    levels_raw: list[int],
    similarities_raw: list[float],
    same_weight: float,
) -> float:
    levels = np.asarray(levels_raw, dtype=np.int8)
    similarities = np.asarray(similarities_raw, dtype=np.float64)
    final = np.zeros(len(levels), dtype=np.float64)
    same = levels <= 3
    similar = levels >= 4
    if same.any():
        raw = WEIGHT_LOOKUP[levels[same]]
        final[same] = float(same_weight) * raw / raw.sum()
    if similar.any():
        raw = similarities[similar]
        final[similar] = (1.0 - float(same_weight)) * raw / raw.sum()
    return float(final.max()) if len(final) else np.nan


def adaptive_kde_result(record: tuple) -> tuple:
    """生成分位数，并在必要时内生修正稀疏异常同OD混合权重。"""

    base = record[:7]
    (
        prices, levels, similarity_weights, same_bandwidth, similar_bandwidth,
        original_same_weight, available,
    ) = base
    if not available:
        quantiles = kde_quantiles(base)
        return quantiles, np.nan, np.nan, np.nan, 0, np.nan, np.nan

    baseline = kde_quantiles(base)
    q30_index = int(np.argmin(np.abs(cfg.QUANTILE_LEVELS - 0.30)))
    q50_index = int(np.argmin(np.abs(cfg.QUANTILE_LEVELS - 0.50)))
    q50 = float(baseline[q50_index])
    pre_ratio = (
        float(baseline[q30_index] / q50)
        if np.isfinite(q50) and q50 > 0.0
        else np.nan
    )
    (
        trip_miles, target_trip_miles, same_effective_n,
        similar_effective_n, similar_mean_similarity, same_count, similar_count,
    ) = record[7:]
    same_count = int(same_count)
    similar_count = int(similar_count)
    corrected = 0
    factor = np.nan
    final_same_weight = float(original_same_weight)

    if (
        pre_ratio < cfg.Q30_TO_Q50_AUDIT_MIN
        and 0 < same_count <= cfg.ADAPTIVE_REWEIGHT_MAX_SAME_OD_COUNT
        and similar_count > 0
    ):
        factor = route_consistency_factor(levels, trip_miles, target_trip_miles)
        same_mass = max(float(same_effective_n), 0.0) * factor
        similar_mass = (
            cfg.SIMILAR_OD_RELIABILITY_DISCOUNT
            * float(np.clip(optional_float(similar_mean_similarity), 0.0, 1.0))
            * max(float(similar_effective_n), 0.0)
        )
        if np.isfinite(same_mass) and same_mass > 0.0 and similar_mass > 0.0:
            final_same_weight = float(same_mass / (same_mass + similar_mass))
            baseline = kde_quantiles(
                (
                    prices, levels, similarity_weights, same_bandwidth,
                    similar_bandwidth, final_same_weight, available,
                )
            )
            corrected = 1

    final_similar_weight = 1.0 - final_same_weight
    max_weight = maximum_final_sample_weight(
        levels, similarity_weights, final_same_weight
    )
    return (
        baseline,
        final_same_weight,
        final_similar_weight,
        max_weight,
        corrected,
        factor,
        pre_ratio,
    )


def write_quantile_part(
    sample_table: pa.Table,
    results: list[tuple],
    output: Path,
) -> None:
    metadata = sample_table.drop(
        [
            "sample_prices",
            "sample_levels",
            "sample_sublevels",
            "sample_trip_miles",
            "sample_similarity_weights",
        ]
    ).to_pandas()
    quantiles = np.vstack([result[0] for result in results]).astype(
        np.float32, copy=False
    )
    available = metadata["available_flag"].to_numpy(dtype=np.int8) == 1
    metadata["same_od_mixture_weight"] = np.asarray(
        [result[1] for result in results], dtype=np.float32
    )
    metadata["similar_od_mixture_weight"] = np.asarray(
        [result[2] for result in results], dtype=np.float32
    )
    metadata["max_final_sample_weight"] = np.asarray(
        [result[3] for result in results], dtype=np.float32
    )
    metadata["adaptive_reweight_flag"] = np.asarray(
        [result[4] for result in results], dtype=np.int16
    )
    metadata["same_route_consistency_factor"] = np.asarray(
        [result[5] for result in results], dtype=np.float32
    )
    metadata["pre_adjustment_q30_to_q50_ratio"] = np.asarray(
        [result[6] for result in results], dtype=np.float32
    )
    # 一次性拼接 101 列，避免逐列 insert 导致 DataFrame 高度碎片化。
    quantile_frame = pd.DataFrame(
        quantiles.astype(np.float32, copy=False),
        columns=cfg.QUANTILE_COLUMNS,
    )
    metadata = pd.concat([metadata, quantile_frame], axis=1, copy=False)
    q30 = metadata[cfg.quantile_column_name(0.30)].to_numpy(dtype=np.float64)
    q50 = metadata[cfg.quantile_column_name(0.50)].to_numpy(dtype=np.float64)
    ratio = np.divide(
        q30,
        q50,
        out=np.full(len(metadata), np.nan, dtype=np.float64),
        where=np.isfinite(q50) & (q50 > 0.0),
    )
    metadata["q30_to_q50_ratio"] = ratio.astype(np.float32)
    inverse_ratio = np.divide(
        q50,
        q30,
        out=np.full(len(metadata), np.nan, dtype=np.float64),
        where=np.isfinite(q30) & (q30 > 0.0),
    )
    metadata["q50_to_q30_ratio"] = inverse_ratio.astype(np.float32)
    metadata["q50_q30_extreme_flag"] = (
        available & np.isfinite(ratio) & (ratio < cfg.Q30_TO_Q50_AUDIT_MIN)
    ).astype(np.int16)
    table = pa.Table.from_pandas(metadata, preserve_index=False)
    pq.write_table(table, output, compression=cfg.PARQUET_COMPRESSION)

    # 将结构性异常单独落盘，方便对全部目标而非仅随机样本审计。
    flagged = metadata.loc[metadata["q50_q30_extreme_flag"] == 1]
    if not flagged.empty:
        cfg.STRUCTURAL_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        audit_output = cfg.STRUCTURAL_AUDIT_DIR / output.name
        pq.write_table(
            pa.Table.from_pandas(flagged, preserve_index=False),
            audit_output,
            compression=cfg.PARQUET_COMPRESSION,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 p^e_ijt 的 101 维 KDE 分位数。")
    parser.add_argument("--workers", type=int, default=cfg.DEFAULT_WORKERS)
    parser.add_argument("--start-part", type=int, default=1)
    parser.add_argument("--end-part", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.start_part < 1:
        raise ValueError("workers 和 start-part 必须大于等于 1。")
    if args.end_part is not None and args.end_part < args.start_part:
        raise ValueError("end-part 不得小于 start-part。")
    if args.resume and args.overwrite:
        raise ValueError("--resume 与 --overwrite 不能同时使用。")
    if not cfg.SAMPLE_DATASET.exists():
        raise FileNotFoundError(f"样本数据集不存在：{cfg.SAMPLE_DATASET}")

    sample_files = sorted(cfg.SAMPLE_DATASET.glob("part-*.parquet"))
    if not sample_files:
        raise FileNotFoundError("样本数据集没有 part-*.parquet 文件。")
    cfg.ensure_output_directories()
    if cfg.PE_DISTRIBUTION_DATASET.exists():
        if args.overwrite:
            shutil.rmtree(cfg.PE_DISTRIBUTION_DATASET)
        elif not args.resume:
            raise FileExistsError(
                f"分位数输出已存在：{cfg.PE_DISTRIBUTION_DATASET}；"
                "请使用 --resume 或 --overwrite。"
            )
    cfg.PE_DISTRIBUTION_DATASET.mkdir(parents=True, exist_ok=True)
    if args.overwrite and cfg.STRUCTURAL_AUDIT_DIR.exists():
        shutil.rmtree(cfg.STRUCTURAL_AUDIT_DIR)

    pool = mp.get_context("fork").Pool(args.workers) if args.workers > 1 else None
    processed_rows = 0
    adaptive_rows = 0
    skipped_parts = 0
    try:
        for ordinal, sample_file in enumerate(sample_files, start=1):
            if ordinal < args.start_part:
                continue
            if args.end_part is not None and ordinal > args.end_part:
                break
            output = cfg.PE_DISTRIBUTION_DATASET / sample_file.name
            if output.exists() and args.resume:
                skipped_parts += 1
                continue
            if output.exists():
                raise FileExistsError(f"分位数批次已存在：{output}")

            table = pq.read_table(sample_file)
            required = {
                "sample_prices", "sample_levels", "sample_trip_miles",
                "sample_similarity_weights", "same_od_bandwidth_usd",
                "similar_od_bandwidth_usd", "same_od_mixture_weight",
                "available_flag", "target_reference_trip_miles",
                "same_od_effective_sample_size",
                "similar_od_effective_sample_size",
                "similar_od_mean_similarity", "same_od_sample_count",
                "similar_od_sample_count",
            }
            missing = sorted(required - set(table.schema.names))
            if missing:
                raise ValueError(f"{sample_file.name} 缺少字段：{missing}")
            records = sample_records(table)
            if pool is None:
                rows = kde_chunk(records)
            else:
                chunks = split_records(records, args.workers * 2)
                rows = [row for part in pool.map(kde_chunk, chunks) for row in part]
            write_quantile_part(table, rows, output)
            processed_rows += len(table)
            adaptive_count = sum(int(row[4]) for row in rows)
            adaptive_rows += adaptive_count
            print(
                f"part {ordinal}: 写出 {len(table):,} 行；"
                f"本次累计 {processed_rows:,} 行；"
                f"内生权重修正 {adaptive_count:,} 行"
            )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    diagnostics = {
        "processed_rows_this_run": int(processed_rows),
        "adaptive_reweight_rows_this_run": int(adaptive_rows),
        "skipped_existing_parts": int(skipped_parts),
        "workers": int(args.workers),
        "quantile_count": int(len(cfg.QUANTILE_LEVELS)),
        "bisection_steps": int(cfg.KDE_BISECTION_STEPS),
        "kde_components": "same_od_layers_1_2_3 + similar_od_layer_4",
        "mixture_method": "effective_sample_size_and_trip_distance_similarity",
        "adaptive_reweight_method": (
            "q30/q50 threshold + sparse same-OD route-consistency mass"
        ),
        "external_weight_override_used": False,
        "structural_audit_rule": "q30 < 0.30 * q50",
        "structural_audit_dataset": str(cfg.STRUCTURAL_AUDIT_DIR),
        "sample_dataset": str(cfg.SAMPLE_DATASET),
        "output_dataset": str(cfg.PE_DISTRIBUTION_DATASET),
    }
    cfg.BUILD_DIAGNOSTICS_FILE.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
