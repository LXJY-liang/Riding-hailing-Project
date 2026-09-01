# -*- coding: utf-8 -*-
"""流式检查 π_ijt 分区域稀疏概率矩阵和样本诊断。

主要检查：
1. 每个输出分片的形状、索引范围和数值有效性；
2. available_flag=1 的每行概率和等于 1；
3. available_flag=0 的行为空分布；
4. 四层订单数之和与总数一致；
5. 前三层达到 N_MIN 时绝不使用第四层；
6. 第四层未达 N_STOP 时必须是候选已耗尽；
7. 停止后允许超过 N_STOP，并检查超出量记录，从而确认
   没有截断最后一个完整历史时段。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
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
    parser = argparse.ArgumentParser(description="检查 π_ijt 历史分布。")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="允许检查部分区域的小规模试跑输出。",
    )
    return parser.parse_args()


def expected_part_indices() -> set[int]:
    return set(range(cfg.EXPECTED_REGION_COUNT))


def existing_part_indices(directory: Path, suffix: str) -> set[int]:
    result: set[int] = set()
    for path in directory.glob(f"origin-*{suffix}"):
        stem = path.name.removesuffix(suffix)
        try:
            result.add(int(stem.split("-")[-1]))
        except ValueError as error:
            raise ValueError(f"无法解析输出分片名：{path.name}") from error
    return result


def check_one_origin(origin_index: int) -> dict[str, int | float]:
    pi_file = cfg.pi_part_file(origin_index)
    diagnostic_file = cfg.diagnostic_part_file(origin_index)
    pi = sparse.load_npz(pi_file).tocsr()
    diagnostics = pd.read_parquet(diagnostic_file)

    expected_shape = (cfg.EXPECTED_TIME_COUNT, cfg.EXPECTED_REGION_COUNT)
    shape_mismatch = int(pi.shape != expected_shape)
    diagnostic_row_mismatch = int(len(diagnostics) != cfg.EXPECTED_TIME_COUNT)
    if shape_mismatch or diagnostic_row_mismatch:
        return {
            "shape_mismatch": shape_mismatch,
            "diagnostic_row_mismatch": diagnostic_row_mismatch,
            "rows": int(pi.shape[0]),
            "nnz": int(pi.nnz),
        }

    required = {
        "row_id",
        "origin_index",
        "time_index",
        "layer1_order_count",
        "layer2_order_count",
        "layer3_order_count",
        "layer4_order_count",
        "layer4_period_count",
        "total_history_order_count",
        "weighted_history_order_count",
        "available_flag",
        "insufficient_sample_flag",
        "layer4_used_flag",
        "layer4_candidates_exhausted_flag",
        "layer4_stop_overshoot_orders",
        "max_match_level",
        "pi_nnz",
        "probability_sum",
    }
    missing = sorted(required - set(diagnostics.columns))
    if missing:
        raise ValueError(f"{diagnostic_file.name} 缺少字段：{missing}")

    diagnostics = diagnostics.sort_values("time_index").reset_index(drop=True)
    expected_time = np.arange(cfg.EXPECTED_TIME_COUNT, dtype=np.int16)
    expected_row_id = (
        origin_index * cfg.EXPECTED_TIME_COUNT
        + np.arange(cfg.EXPECTED_TIME_COUNT, dtype=np.int64)
    )
    identity_mismatch = int(
        np.any(diagnostics["origin_index"].to_numpy(dtype=np.int32) != origin_index)
        or np.any(diagnostics["time_index"].to_numpy(dtype=np.int16) != expected_time)
        or np.any(diagnostics["row_id"].to_numpy(dtype=np.int64) != expected_row_id)
    )

    data = pi.data.astype(np.float64, copy=False)
    invalid_probability_count = int(
        np.sum(~np.isfinite(data) | (data < 0.0) | (data > 1.0 + 1e-6))
    )
    invalid_destination_index_count = int(
        np.sum((pi.indices < 0) | (pi.indices >= cfg.EXPECTED_REGION_COUNT))
    )
    actual_nnz = np.diff(pi.indptr).astype(np.int32)
    row_sums = np.asarray(pi.sum(axis=1)).ravel().astype(np.float64)

    available = diagnostics["available_flag"].to_numpy(dtype=np.int8)
    insufficient = diagnostics["insufficient_sample_flag"].to_numpy(dtype=np.int8)
    c1 = diagnostics["layer1_order_count"].to_numpy(dtype=np.int64)
    c2 = diagnostics["layer2_order_count"].to_numpy(dtype=np.int64)
    c3 = diagnostics["layer3_order_count"].to_numpy(dtype=np.int64)
    c4 = diagnostics["layer4_order_count"].to_numpy(dtype=np.int64)
    c123 = c1 + c2 + c3
    total = diagnostics["total_history_order_count"].to_numpy(dtype=np.int64)
    period4 = diagnostics["layer4_period_count"].to_numpy(dtype=np.int32)
    used4 = diagnostics["layer4_used_flag"].to_numpy(dtype=np.int8)
    exhausted4 = diagnostics[
        "layer4_candidates_exhausted_flag"
    ].to_numpy(dtype=np.int8)
    overshoot = diagnostics["layer4_stop_overshoot_orders"].to_numpy(dtype=np.int64)

    row_sum_mismatch = int(
        np.sum(
            ((available == 1) & (np.abs(row_sums - 1.0) > cfg.PROBABILITY_SUM_ATOL))
            | ((available == 0) & (np.abs(row_sums) > cfg.PROBABILITY_SUM_ATOL))
        )
    )
    diagnostic_probability_sum_mismatch = int(
        np.sum(
            np.abs(
                diagnostics["probability_sum"].to_numpy(dtype=np.float64)
                - row_sums
            )
            > cfg.PROBABILITY_SUM_ATOL
        )
    )
    nnz_mismatch = int(
        np.sum(diagnostics["pi_nnz"].to_numpy(dtype=np.int32) != actual_nnz)
    )
    layer_total_mismatch = int(np.sum(c1 + c2 + c3 + c4 != total))
    available_flag_mismatch = int(np.sum(available != (total > 0).astype(np.int8)))
    insufficient_flag_mismatch = int(
        np.sum(insufficient != ((total > 0) & (total < cfg.N_MIN)).astype(np.int8))
    )
    layer4_used_flag_mismatch = int(
        np.sum(used4 != ((c4 > 0) & (period4 > 0)).astype(np.int8))
    )
    layer4_forbidden_rows = int(np.sum((c123 >= cfg.N_MIN) & (c4 > 0)))
    layer4_period_count_mismatch = int(
        np.sum(((c4 > 0) & (period4 <= 0)) | ((c4 == 0) & (period4 != 0)))
    )
    # 如启用第四层后仍未达停止阈值，候选必须已耗尽。
    layer4_unexplained_below_stop_rows = int(
        np.sum((c123 < cfg.N_MIN) & (total < cfg.N_STOP) & (exhausted4 != 1))
    )
    layer4_false_exhausted_rows = int(
        np.sum((total >= cfg.N_STOP) & (exhausted4 != 0))
    )
    expected_overshoot = np.where(
        (c4 > 0) & (total >= cfg.N_STOP),
        total - cfg.N_STOP,
        0,
    )
    layer4_overshoot_mismatch = int(np.sum(overshoot != expected_overshoot))
    negative_count_rows = int(
        np.sum((c1 < 0) | (c2 < 0) | (c3 < 0) | (c4 < 0) | (total < 0))
    )

    return {
        "shape_mismatch": shape_mismatch,
        "diagnostic_row_mismatch": diagnostic_row_mismatch,
        "identity_mismatch": identity_mismatch,
        "rows": int(pi.shape[0]),
        "nnz": int(pi.nnz),
        "available_rows": int((available == 1).sum()),
        "insufficient_rows": int((insufficient == 1).sum()),
        "layer4_used_rows": int((used4 == 1).sum()),
        "overshoot_rows": int((overshoot > 0).sum()),
        "maximum_overshoot_orders": int(overshoot.max(initial=0)),
        "invalid_probability_count": invalid_probability_count,
        "invalid_destination_index_count": invalid_destination_index_count,
        "row_sum_mismatch": row_sum_mismatch,
        "diagnostic_probability_sum_mismatch": diagnostic_probability_sum_mismatch,
        "nnz_mismatch": nnz_mismatch,
        "layer_total_mismatch": layer_total_mismatch,
        "available_flag_mismatch": available_flag_mismatch,
        "insufficient_flag_mismatch": insufficient_flag_mismatch,
        "layer4_used_flag_mismatch": layer4_used_flag_mismatch,
        "layer4_forbidden_rows": layer4_forbidden_rows,
        "layer4_period_count_mismatch": layer4_period_count_mismatch,
        "layer4_unexplained_below_stop_rows": layer4_unexplained_below_stop_rows,
        "layer4_false_exhausted_rows": layer4_false_exhausted_rows,
        "layer4_overshoot_mismatch": layer4_overshoot_mismatch,
        "negative_count_rows": negative_count_rows,
    }


def main() -> None:
    args = parse_args()
    cfg.ensure_output_directories()
    pi_indices = existing_part_indices(cfg.PI_PARTS_DIR, ".npz")
    diagnostic_indices = existing_part_indices(cfg.DIAGNOSTIC_PARTS_DIR, ".parquet")
    paired = pi_indices & diagnostic_indices
    missing_pi_for_diagnostics = sorted(diagnostic_indices - pi_indices)
    missing_diagnostics_for_pi = sorted(pi_indices - diagnostic_indices)
    missing_origins = sorted(expected_part_indices() - paired)
    unexpected_origins = sorted(paired - expected_part_indices())

    if not paired:
        raise FileNotFoundError("没有找到可检查的 π_ijt 分布分片。")

    totals: dict[str, int | float] = {
        "checked_origin_count": int(len(paired)),
        "rows": 0,
        "nnz": 0,
        "available_rows": 0,
        "insufficient_rows": 0,
        "layer4_used_rows": 0,
        "overshoot_rows": 0,
        "maximum_overshoot_orders": 0,
    }
    failure_keys: set[str] = set()
    for ordinal, origin_index in enumerate(sorted(paired), start=1):
        result = check_one_origin(origin_index)
        for key, value in result.items():
            if key == "maximum_overshoot_orders":
                totals[key] = max(int(totals.get(key, 0)), int(value))
            else:
                totals[key] = int(totals.get(key, 0)) + int(value)
            if (
                key
                not in {
                    "rows",
                    "nnz",
                    "available_rows",
                    "insufficient_rows",
                    "layer4_used_rows",
                    "overshoot_rows",
                    "maximum_overshoot_orders",
                }
                and int(value) > 0
            ):
                failure_keys.add(key)
        if ordinal % 50 == 0 or ordinal == len(paired):
            print(f"已检查 {ordinal}/{len(paired)} 个出发区域")

    diagnostics: dict[str, object] = {
        **totals,
        "expected_origin_count": int(cfg.EXPECTED_REGION_COUNT),
        "missing_origins": missing_origins,
        "unexpected_origins": unexpected_origins,
        "missing_pi_for_diagnostics": missing_pi_for_diagnostics,
        "missing_diagnostics_for_pi": missing_diagnostics_for_pi,
        "n_min": int(cfg.N_MIN),
        "n_stop": int(cfg.N_STOP),
        "failure_keys": sorted(failure_keys),
    }
    cfg.CHECK_DIAGNOSTICS_FILE.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "format": "per-origin scipy CSR parts",
        "matrix_axes": ["time_index", "destination_index"],
        "part_shape": [cfg.EXPECTED_TIME_COUNT, cfg.EXPECTED_REGION_COUNT],
        "origin_count": len(paired),
        "region_mapping": str(cfg.REGION_MAPPING_FILE),
        "time_mapping": str(cfg.TIME_MAPPING_FILE),
        "pi_parts_directory": str(cfg.PI_PARTS_DIR),
        "diagnostic_parts_directory": str(cfg.DIAGNOSTIC_PARTS_DIR),
        "part_name_pattern": "origin-{origin_index:04d}.npz",
        "row_id_formula": "origin_index * time_count + time_index",
        "n_min": int(cfg.N_MIN),
        "n_stop": int(cfg.N_STOP),
        "layer_weights": {str(key): float(value) for key, value in cfg.LAYER_WEIGHTS.items()},
    }
    cfg.OUTPUT_MANIFEST_FILE.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))

    failures: dict[str, object] = {}
    if failure_keys:
        failures["failure_keys"] = sorted(failure_keys)
    if missing_pi_for_diagnostics:
        failures["missing_pi_for_diagnostics"] = missing_pi_for_diagnostics
    if missing_diagnostics_for_pi:
        failures["missing_diagnostics_for_pi"] = missing_diagnostics_for_pi
    if unexpected_origins:
        failures["unexpected_origins"] = unexpected_origins
    if missing_origins and not args.allow_partial:
        failures["missing_origins"] = missing_origins
    if failures:
        raise AssertionError(f"π_ijt 历史分布检查失败：{failures}")


if __name__ == "__main__":
    main()
