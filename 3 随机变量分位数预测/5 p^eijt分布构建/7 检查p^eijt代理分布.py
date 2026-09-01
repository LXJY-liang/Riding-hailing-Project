# -*- coding: utf-8 -*-
"""流式检查样本中间件和最终 p^e_ijt 代理分布。"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds
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
    parser = argparse.ArgumentParser(description="检查 p^e_ijt 样本与分位数输出。")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="允许只检查小批量测试输出，不要求行数等于完整目标表。",
    )
    return parser.parse_args()


def check_sample_dataset() -> dict[str, int | float]:
    dataset = ds.dataset(cfg.SAMPLE_DATASET, format="parquet")
    required = {
        "target_id", "i", "j", "t", "available_flag",
        "insufficient_sample_flag", "sample_count", "same_od_sample_count",
        "similar_od_sample_count", "same_od_effective_sample_size",
        "similar_od_effective_sample_size", "same_od_bandwidth_usd",
        "similar_od_bandwidth_usd", "same_od_mixture_weight",
        "similar_od_mixture_weight", "similar_od_mean_similarity",
        "max_final_sample_weight", "max_match_level", "layer1_count",
        "layer2_count", "layer3_count", "layer4_count", "layer4a_count",
        "layer4b_count", "target_reference_trip_miles",
        "target_distance_sample_count", "distance_reference_unavailable_flag",
        "unreliable_similar_only_flag", "sample_prices", "sample_levels",
        "sample_trip_miles", "sample_similarity_weights",
    }
    missing = sorted(required - set(dataset.schema.names))
    if missing:
        raise ValueError(f"样本数据集缺少字段：{missing}")

    counters = {
        "sample_rows": 0,
        "sample_list_length_mismatch_rows": 0,
        "sample_count_above_max_rows": 0,
        "sample_count_flag_mismatch_rows": 0,
        "sample_nonpositive_or_nonfinite_price_rows": 0,
        "sample_invalid_level_rows": 0,
        "sample_layer_count_mismatch_rows": 0,
        "sample_layer4_subcount_mismatch_rows": 0,
        "sample_component_count_mismatch_rows": 0,
        "sample_component_bandwidth_invalid_rows": 0,
        "sample_component_effective_size_invalid_rows": 0,
        "sample_mixture_weight_invalid_rows": 0,
        "sample_similarity_weight_invalid_rows": 0,
        "sample_layer4_distance_invalid_rows": 0,
        "sample_max_weight_invalid_rows": 0,
    }
    columns = sorted(required - {"i", "j", "t"})
    for batch in dataset.scanner(columns=columns, batch_size=20_000).to_batches():
        data = batch.to_pandas()
        counters["sample_rows"] += len(data)
        sample_count = data["sample_count"].to_numpy(dtype=np.int32)
        available = data["available_flag"].to_numpy(dtype=np.int8) == 1
        insufficient = data["insufficient_sample_flag"].to_numpy(dtype=np.int8) == 1
        same_count = data["same_od_sample_count"].to_numpy(dtype=np.int32)
        similar_count = data["similar_od_sample_count"].to_numpy(dtype=np.int32)
        same_effective_n = data["same_od_effective_sample_size"].to_numpy(
            dtype=np.float64
        )
        similar_effective_n = data["similar_od_effective_sample_size"].to_numpy(
            dtype=np.float64
        )
        same_bandwidth = data["same_od_bandwidth_usd"].to_numpy(dtype=np.float64)
        similar_bandwidth = data["similar_od_bandwidth_usd"].to_numpy(
            dtype=np.float64
        )
        same_mix = data["same_od_mixture_weight"].to_numpy(dtype=np.float64)
        similar_mix = data["similar_od_mixture_weight"].to_numpy(dtype=np.float64)
        similar_mean = data["similar_od_mean_similarity"].to_numpy(
            dtype=np.float64
        )
        max_final_weight = data["max_final_sample_weight"].to_numpy(
            dtype=np.float64
        )

        price_lists = data["sample_prices"].tolist()
        level_lists = data["sample_levels"].tolist()
        miles_lists = data["sample_trip_miles"].tolist()
        similarity_lists = data["sample_similarity_weights"].tolist()
        price_lengths = np.fromiter((len(x) for x in price_lists), dtype=np.int32)
        level_lengths = np.fromiter((len(x) for x in level_lists), dtype=np.int32)
        miles_lengths = np.fromiter((len(x) for x in miles_lists), dtype=np.int32)
        similarity_lengths = np.fromiter(
            (len(x) for x in similarity_lists), dtype=np.int32
        )
        counters["sample_list_length_mismatch_rows"] += int(
            (
                (price_lengths != sample_count)
                | (level_lengths != sample_count)
                | (miles_lengths != sample_count)
                | (similarity_lengths != sample_count)
            ).sum()
        )
        counters["sample_count_above_max_rows"] += int((sample_count > cfg.N_MAX).sum())
        counters["sample_count_flag_mismatch_rows"] += int(
            ((sample_count < cfg.N_MIN) != insufficient).sum()
        )
        expected_same = (
            data["layer1_count"].to_numpy(dtype=np.int32)
            + data["layer2_count"].to_numpy(dtype=np.int32)
            + data["layer3_count"].to_numpy(dtype=np.int32)
        )
        expected_similar = data["layer4_count"].to_numpy(dtype=np.int32)
        counters["sample_component_count_mismatch_rows"] += int(
            ((same_count != expected_same) | (similar_count != expected_similar)).sum()
        )

        bandwidth_invalid = (
            (same_count > 0)
            & (
                ~np.isfinite(same_bandwidth)
                | (same_bandwidth < cfg.MIN_BANDWIDTH_USD - 1e-6)
            )
        ) | (
            (similar_count > 0)
            & (
                ~np.isfinite(similar_bandwidth)
                | (similar_bandwidth < cfg.MIN_BANDWIDTH_USD - 1e-6)
            )
        ) | ((same_count == 0) & np.isfinite(same_bandwidth)) | (
            (similar_count == 0) & np.isfinite(similar_bandwidth)
        )
        counters["sample_component_bandwidth_invalid_rows"] += int(
            bandwidth_invalid.sum()
        )

        effective_invalid = (
            (same_count > 0)
            & ((same_effective_n <= 0) | (same_effective_n > same_count + 1e-5))
        ) | (
            (similar_count > 0)
            & (
                (similar_effective_n <= 0)
                | (similar_effective_n > similar_count + 1e-5)
            )
        ) | ((same_count == 0) & (np.abs(same_effective_n) > 1e-6)) | (
            (similar_count == 0) & (np.abs(similar_effective_n) > 1e-6)
        )
        counters["sample_component_effective_size_invalid_rows"] += int(
            effective_invalid.sum()
        )

        expected_same_mix = np.full(len(data), np.nan, dtype=np.float64)
        expected_similar_mix = np.full(len(data), np.nan, dtype=np.float64)
        for row_index in np.flatnonzero(available):
            expected_same_mix[row_index], expected_similar_mix[row_index] = (
                cfg.component_mixture_weights(
                    same_effective_n[row_index],
                    similar_effective_n[row_index],
                    0.0 if similar_count[row_index] == 0 else similar_mean[row_index],
                    same_count[row_index],
                    similar_count[row_index],
                )
            )
        mixture_invalid = (
            available
            & (
                ~np.isfinite(same_mix)
                | ~np.isfinite(similar_mix)
                | (same_mix < -1e-6)
                | (same_mix > 1.0 + 1e-6)
                | (similar_mix < -1e-6)
                | (similar_mix > 1.0 + 1e-6)
                | (np.abs(same_mix + similar_mix - 1.0) > 1e-6)
                | (np.abs(same_mix - expected_same_mix) > 1e-6)
                | (np.abs(similar_mix - expected_similar_mix) > 1e-6)
            )
        ) | (
            (~available) & (np.isfinite(same_mix) | np.isfinite(similar_mix))
        )
        counters["sample_mixture_weight_invalid_rows"] += int(
            mixture_invalid.sum()
        )

        bad_price = 0
        bad_level = 0
        bad_similarity = 0
        bad_layer4_distance = 0
        for prices, levels, miles, similarities in zip(
            price_lists, level_lists, miles_lists, similarity_lists
        ):
            prices_array = np.asarray(prices, dtype=np.float64)
            levels_array = np.asarray(levels, dtype=np.int8)
            miles_array = np.asarray(miles, dtype=np.float64)
            similarities_array = np.asarray(similarities, dtype=np.float64)
            bad_price += int(
                len(prices_array) > 0
                and np.any(~np.isfinite(prices_array) | (prices_array <= 0))
            )
            bad_level += int(
                len(levels_array) > 0
                and np.any((levels_array < 1) | (levels_array > 4))
            )
            bad_similarity += int(
                len(similarities_array) > 0
                and np.any(
                    ~np.isfinite(similarities_array)
                    | (similarities_array <= 0.0)
                    | (similarities_array > 1.0 + 1e-6)
                )
            )
            layer4 = levels_array == 4
            bad_layer4_distance += int(
                layer4.any()
                and np.any(~np.isfinite(miles_array[layer4]) | (miles_array[layer4] <= 0))
            )
        counters["sample_nonpositive_or_nonfinite_price_rows"] += bad_price
        counters["sample_invalid_level_rows"] += bad_level
        counters["sample_similarity_weight_invalid_rows"] += bad_similarity
        counters["sample_layer4_distance_invalid_rows"] += bad_layer4_distance
        counters["sample_max_weight_invalid_rows"] += int(
            (
                available
                & (
                    ~np.isfinite(max_final_weight)
                    | (max_final_weight <= 0.0)
                    | (max_final_weight > 1.0 + 1e-6)
                )
            ).sum()
        )

        layer_sum = data[
            ["layer1_count", "layer2_count", "layer3_count", "layer4_count"]
        ].sum(axis=1).to_numpy(dtype=np.int32)
        counters["sample_layer_count_mismatch_rows"] += int(
            (layer_sum != sample_count).sum()
        )
        layer4_sum = (
            data["layer4a_count"].to_numpy(dtype=np.int32)
            + data["layer4b_count"].to_numpy(dtype=np.int32)
        )
        counters["sample_layer4_subcount_mismatch_rows"] += int(
            (layer4_sum != data["layer4_count"].to_numpy(dtype=np.int32)).sum()
        )
    return counters


def check_quantile_dataset() -> dict[str, int | float | dict[str, int]]:
    dataset = ds.dataset(cfg.PE_DISTRIBUTION_DATASET, format="parquet")
    required = {
        "target_id", "i", "j", "t", "available_flag", "sample_count",
        "same_od_sample_count", "similar_od_sample_count",
        "same_od_effective_sample_size", "similar_od_effective_sample_size",
        "same_od_bandwidth_usd", "similar_od_bandwidth_usd",
        "same_od_mixture_weight", "similar_od_mixture_weight",
        "max_match_level", "q30_to_q50_ratio", "q50_to_q30_ratio",
        "q50_q30_extreme_flag",
        *cfg.QUANTILE_COLUMNS,
    }
    missing = sorted(required - set(dataset.schema.names))
    if missing:
        raise ValueError(f"分位数数据集缺少字段：{missing}")

    counters = {
        "quantile_rows": 0,
        "available_rows": 0,
        "unavailable_rows": 0,
        "nonmonotone_rows": 0,
        "negative_quantile_rows": 0,
        "available_with_nonfinite_quantile_rows": 0,
        "unavailable_with_nonmissing_quantile_rows": 0,
        "quantile_component_effective_size_invalid_rows": 0,
        "quantile_mixture_weight_invalid_rows": 0,
        "quantile_structural_audit_flag_mismatch_rows": 0,
        "q50_q30_extreme_count": 0,
    }
    level_counts = {str(level): 0 for level in range(5)}
    columns = [
        "available_flag", "sample_count", "same_od_sample_count",
        "similar_od_sample_count", "same_od_effective_sample_size",
        "similar_od_effective_sample_size", "same_od_mixture_weight",
        "similar_od_mixture_weight", "max_match_level",
        "q30_to_q50_ratio", "q50_to_q30_ratio", "q50_q30_extreme_flag",
    ] + cfg.QUANTILE_COLUMNS
    for batch in dataset.scanner(columns=columns, batch_size=20_000).to_batches():
        data = batch.to_pandas()
        counters["quantile_rows"] += len(data)
        available = data["available_flag"].to_numpy(dtype=np.int8) == 1
        same_count = data["same_od_sample_count"].to_numpy(dtype=np.float64)
        similar_count = data["similar_od_sample_count"].to_numpy(dtype=np.float64)
        same_effective_n = data["same_od_effective_sample_size"].to_numpy(
            dtype=np.float64
        )
        similar_effective_n = data["similar_od_effective_sample_size"].to_numpy(
            dtype=np.float64
        )
        same_mix = data["same_od_mixture_weight"].to_numpy(dtype=np.float64)
        similar_mix = data["similar_od_mixture_weight"].to_numpy(dtype=np.float64)
        quantiles = data[cfg.QUANTILE_COLUMNS].to_numpy(dtype=np.float64)
        stored_ratio = data["q30_to_q50_ratio"].to_numpy(dtype=np.float64)
        stored_inverse_ratio = data["q50_to_q30_ratio"].to_numpy(
            dtype=np.float64
        )
        stored_extreme = data["q50_q30_extreme_flag"].to_numpy(dtype=np.int8)
        finite = np.isfinite(quantiles)
        counters["available_rows"] += int(available.sum())
        counters["unavailable_rows"] += int((~available).sum())
        counters["available_with_nonfinite_quantile_rows"] += int(
            np.any(~finite[available], axis=1).sum()
        )
        counters["unavailable_with_nonmissing_quantile_rows"] += int(
            np.any(finite[~available], axis=1).sum()
        )
        if available.any():
            values = quantiles[available]
            counters["nonmonotone_rows"] += int(
                np.any(np.diff(values, axis=1) < -1e-5, axis=1).sum()
            )
            counters["negative_quantile_rows"] += int(
                np.any(values < -1e-6, axis=1).sum()
            )
        q30 = data[cfg.quantile_column_name(0.30)].to_numpy(dtype=np.float64)
        q50 = data[cfg.quantile_column_name(0.50)].to_numpy(dtype=np.float64)
        expected_ratio = np.divide(
            q30,
            q50,
            out=np.full(len(data), np.nan, dtype=np.float64),
            where=np.isfinite(q50) & (q50 > 0.0),
        )
        expected_extreme = (
            available
            & np.isfinite(expected_ratio)
            & (expected_ratio < cfg.Q30_TO_Q50_AUDIT_MIN)
        )
        ratio_mismatch = available & (
            ~np.isfinite(stored_ratio)
            | ~np.isclose(
                stored_ratio,
                expected_ratio,
                rtol=1e-6,
                atol=1e-7,
            )
        )
        expected_inverse_ratio = np.divide(
            q50,
            q30,
            out=np.full(len(data), np.nan, dtype=np.float64),
            where=np.isfinite(q30) & (q30 > 0.0),
        )
        inverse_ratio_mismatch = available & (
            ~np.isfinite(stored_inverse_ratio)
            | ~np.isclose(
                stored_inverse_ratio,
                expected_inverse_ratio,
                rtol=1e-6,
                atol=1e-5,
            )
        )
        counters["quantile_structural_audit_flag_mismatch_rows"] += int(
            (
                ratio_mismatch
                | inverse_ratio_mismatch
                | (stored_extreme != expected_extreme.astype(np.int8))
            ).sum()
        )
        counters["q50_q30_extreme_count"] += int(expected_extreme.sum())
        effective_invalid = (
            (same_count > 0)
            & ((same_effective_n <= 0) | (same_effective_n > same_count + 1e-5))
        ) | (
            (similar_count > 0)
            & (
                (similar_effective_n <= 0)
                | (similar_effective_n > similar_count + 1e-5)
            )
        )
        counters["quantile_component_effective_size_invalid_rows"] += int(
            effective_invalid.sum()
        )
        counters["quantile_mixture_weight_invalid_rows"] += int(
            (
                available
                & (
                    ~np.isfinite(same_mix)
                    | ~np.isfinite(similar_mix)
                    | (np.abs(same_mix + similar_mix - 1.0) > 1e-6)
                )
            ).sum()
        )
        values, counts = np.unique(
            data["max_match_level"].to_numpy(dtype=np.int8), return_counts=True
        )
        for value, count in zip(values, counts):
            key = str(int(value))
            level_counts[key] = level_counts.get(key, 0) + int(count)
    counters["max_match_level_counts"] = level_counts
    return counters


def main() -> None:
    args = parse_args()
    cfg.ensure_output_directories()
    for path in (cfg.TARGET_PREPARED_FILE, cfg.SAMPLE_DATASET, cfg.PE_DISTRIBUTION_DATASET):
        if not path.exists():
            raise FileNotFoundError(f"检查所需输出不存在：{path}")

    target_rows = pq.ParquetFile(cfg.TARGET_PREPARED_FILE).metadata.num_rows
    sample_parts = {path.name for path in cfg.SAMPLE_DATASET.glob("part-*.parquet")}
    quantile_parts = {
        path.name for path in cfg.PE_DISTRIBUTION_DATASET.glob("part-*.parquet")
    }
    diagnostics: dict[str, object] = {
        "target_rows": int(target_rows),
        "missing_quantile_parts": sorted(sample_parts - quantile_parts),
        "unexpected_quantile_parts": sorted(quantile_parts - sample_parts),
    }
    diagnostics.update(check_sample_dataset())
    diagnostics.update(check_quantile_dataset())
    diagnostics["row_count_mismatch"] = int(
        not (
            diagnostics["sample_rows"]
            == diagnostics["quantile_rows"]
            == target_rows
        )
    )
    diagnostics["quantile_count"] = len(cfg.QUANTILE_COLUMNS)

    cfg.CHECK_DIAGNOSTICS_FILE.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))

    fatal_names = [
        name for name in diagnostics
        if name.endswith("_rows") and name not in {
            "target_rows", "sample_rows", "quantile_rows", "available_rows",
            "unavailable_rows",
        }
    ]
    failures = {
        name: diagnostics[name]
        for name in fatal_names
        if isinstance(diagnostics[name], (int, float)) and diagnostics[name] > 0
    }
    if diagnostics["missing_quantile_parts"]:
        failures["missing_quantile_parts"] = diagnostics["missing_quantile_parts"]
    if diagnostics["unexpected_quantile_parts"]:
        failures["unexpected_quantile_parts"] = diagnostics["unexpected_quantile_parts"]
    if diagnostics["row_count_mismatch"] and not args.allow_partial:
        failures["row_count_mismatch"] = 1
    if failures:
        raise AssertionError(f"p^e_ijt 代理分布检查失败：{failures}")

if __name__ == "__main__":
    main()
