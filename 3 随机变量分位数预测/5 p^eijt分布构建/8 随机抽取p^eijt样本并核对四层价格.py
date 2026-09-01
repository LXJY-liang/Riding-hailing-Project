# -*- coding: utf-8 -*-
"""随机抽取 p^e_ijt 目标，重建四层价格样本并核对 KDE 结果。

默认从 available_flag=1 的完整结果中等概率抽取 3 个目标。对每个目标：

1. 使用第 5 步的同一收集器重新执行四层匹配；
2. 导出实际进入 KDE 的每笔历史价格、层级、4A/4B、权重和时间；
3. 与第 5 步保存的 sample_prices/sample_levels/带宽逐项比较；
4. 分别重建相同 OD KDE 与相似实际里程 OD KDE，按有效样本量
   和里程相似度混合 CDF；
5. 使用第 6 步同一函数重新计算 101 维分位数；
6. 与最终 peijt_quantiles 中的 101 维结果逐项比较。

注意：算法达到 N_min=50 后会停止进入更低层级。因此某些随机目标没有
第 4 层样本，并不表示缺失，而是说明它在前面的层级已经取得足够样本。
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
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    # dataclass 等运行时功能需要能从 sys.modules 找到当前模块。
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


CODE_DIR = Path(__file__).resolve().parent
cfg = load_module(CODE_DIR / "1 p^eijt分布构建配置.py", "pe_audit_config")
sample_module = load_module(
    CODE_DIR / "5 滚动收集p^eijt历史样本.py",
    "pe_audit_sample_collector",
)
kde_module = load_module(
    CODE_DIR / "6 构建p^eijt加权KDE分位数.py",
    "pe_audit_kde_builder",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="随机抽取并核对 p^e_ijt 代理分布。")
    parser.add_argument("--sample-size", type=int, default=3, help="随机目标数量。")
    parser.add_argument(
        "--seed",
        type=int,
        default=20260925,
        help="随机种子；相同种子会得到相同目标。",
    )
    parser.add_argument(
        "--require-level4",
        action="store_true",
        help="只从实际进入第 4 层的目标中抽样，便于重点检查相似 OD 回退。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖相同随机种子和样本数的旧审计输出。",
    )
    return parser.parse_args()


def validate_inputs() -> None:
    required = [
        cfg.TARGET_PREPARED_FILE,
        cfg.SAMPLE_DATASET,
        cfg.PE_DISTRIBUTION_DATASET,
        cfg.HISTORY_COMPLETION_NS_FILE,
        cfg.HISTORY_I_FILE,
        cfg.HISTORY_J_FILE,
        cfg.HISTORY_DAY_FILE,
        cfg.HISTORY_SLOT_FILE,
        cfg.HISTORY_DOW_FILE,
        cfg.HISTORY_WEEKEND_HOLIDAY_FILE,
        cfg.HISTORY_PRICE_FILE,
        cfg.HISTORY_TRIP_MILES_FILE,
        cfg.HISTORY_TRIP_SECONDS_FILE,
        cfg.OD_ORDER_FILE,
        cfg.OD_OFFSETS_FILE,
        cfg.ORIGIN_TIME_STRICT_ORDER_FILE,
        cfg.ORIGIN_TIME_STRICT_OFFSETS_FILE,
        cfg.ORIGIN_TIME_RELAXED_ORDER_FILE,
        cfg.ORIGIN_TIME_RELAXED_OFFSETS_FILE,
        cfg.NEIGHBOR_MATRIX_FILE,
        cfg.NORMALIZED_NEIGHBOR_FILE,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少审计输入：\n" + "\n".join(missing))


def random_target_ids(
    sample_size: int,
    seed: int,
    require_level4: bool,
) -> tuple[np.ndarray, int]:
    """从符合条件的完整样本表中无放回随机抽取 target_id。"""

    dataset = ds.dataset(cfg.SAMPLE_DATASET, format="parquet")
    columns = ["target_id", "available_flag", "max_match_level"]
    eligible_parts: list[np.ndarray] = []
    for batch in dataset.scanner(columns=columns, batch_size=100_000).to_batches():
        target_id = batch.column("target_id").to_numpy(zero_copy_only=False)
        available = batch.column("available_flag").to_numpy(zero_copy_only=False) == 1
        mask = available
        if require_level4:
            level = batch.column("max_match_level").to_numpy(zero_copy_only=False)
            mask &= level == 4
        if mask.any():
            eligible_parts.append(target_id[mask].astype(np.int64, copy=False))

    eligible = np.concatenate(eligible_parts) if eligible_parts else np.empty(0, np.int64)
    if sample_size < 1:
        raise ValueError("sample-size 必须大于等于 1。")
    if len(eligible) < sample_size:
        raise ValueError(
            f"符合条件的目标只有 {len(eligible)} 个，无法抽取 {sample_size} 个。"
        )
    rng = np.random.default_rng(seed)
    chosen = rng.choice(eligible, size=sample_size, replace=False)
    return chosen.astype(np.int64), int(len(eligible))


def rows_by_target_id(path: Path, target_ids: np.ndarray) -> pd.DataFrame:
    dataset = ds.dataset(path, format="parquet")
    expression = ds.field("target_id").isin(pa.array(target_ids, type=pa.int64()))
    data = dataset.to_table(filter=expression).to_pandas()
    if len(data) != len(target_ids):
        raise ValueError(
            f"{path} 返回 {len(data)} 行，但随机目标数量为 {len(target_ids)}。"
        )
    return data.set_index("target_id", drop=False)


def region_lookup() -> np.ndarray:
    centroids = pd.read_parquet(
        cfg.REGION_CENTROID_FILE,
        columns=["region_id", "region_index"],
    ).sort_values("region_index")
    if len(centroids) != cfg.REGION_COUNT:
        raise ValueError("区域质心表数量异常。")
    return centroids["region_id"].astype(str).to_numpy()


def json_price_list(prices: np.ndarray) -> str:
    return json.dumps(
        [round(float(value), 4) for value in prices],
        ensure_ascii=False,
    )


def main() -> None:
    args = parse_args()
    validate_inputs()
    cfg.ensure_output_directories()

    pool_name = "level4" if args.require_level4 else "available"
    run_dir = cfg.AUDIT_DIR / f"seed_{args.seed}_n{args.sample_size}_{pool_name}"
    if run_dir.exists():
        if args.overwrite:
            shutil.rmtree(run_dir)
        else:
            raise FileExistsError(
                f"审计输出已存在：{run_dir}；如需重建请使用 --overwrite。"
            )
    run_dir.mkdir(parents=True, exist_ok=False)

    chosen_ids, eligible_count = random_target_ids(
        args.sample_size,
        args.seed,
        args.require_level4,
    )
    targets = rows_by_target_id(cfg.TARGET_PREPARED_FILE, chosen_ids)
    saved_samples = rows_by_target_id(cfg.SAMPLE_DATASET, chosen_ids)
    saved_quantiles = rows_by_target_id(cfg.PE_DISTRIBUTION_DATASET, chosen_ids)

    collector = sample_module.RollingSampleCollector()
    index_to_region = region_lookup()
    data_start = pd.Timestamp(cfg.DATA_START).normalize()
    weight_lookup = np.asarray(
        [0.0, cfg.LAYER_WEIGHTS[1], cfg.LAYER_WEIGHTS[2],
         cfg.LAYER_WEIGHTS[3], cfg.LAYER_WEIGHTS[4]],
        dtype=np.float64,
    )

    target_rows: list[dict[str, object]] = []
    price_rows: list[dict[str, object]] = []
    quantile_rows: list[dict[str, object]] = []
    failures: list[str] = []

    for audit_number, target_id_raw in enumerate(chosen_ids, start=1):
        target_id = int(target_id_raw)
        target = targets.loc[target_id]
        saved = saved_samples.loc[target_id]
        final = saved_quantiles.loc[target_id]

        candidates = collector.select(
            int(target["i_index"]),
            int(target["j_index"]),
            int(target["target_ns"]),
            int(target["target_day"]),
            int(target["target_slot"]),
            int(target["target_dow"]),
            bool(target["target_is_holiday"]),
            int(target["target_weekend_or_holiday"]),
        )
        indices = np.asarray(candidates.indices, dtype=np.int64)
        levels = np.asarray(candidates.levels, dtype=np.int8)
        sublevels = np.asarray(candidates.sublevels, dtype="U2")
        similarities = np.asarray(candidates.similarities, dtype=np.float64)
        prices = collector.price[indices].astype(np.float32)
        trip_miles = collector.trip_miles[indices].astype(np.float32)
        same_od_mask = levels <= 3
        similar_od_mask = levels == 4
        same_od_count = int(same_od_mask.sum())
        similar_od_count = int(similar_od_mask.sum())

        stored_prices = np.asarray(saved["sample_prices"], dtype=np.float32)
        stored_levels = np.asarray(saved["sample_levels"], dtype=np.int8)
        stored_miles = np.asarray(saved["sample_trip_miles"], dtype=np.float32)
        stored_similarities = np.asarray(
            saved["sample_similarity_weights"], dtype=np.float32
        )
        prices_equal = bool(np.array_equal(prices, stored_prices))
        levels_equal = bool(np.array_equal(levels, stored_levels))
        miles_equal = bool(np.allclose(trip_miles, stored_miles, equal_nan=True))
        similarities_equal = bool(
            np.allclose(similarities, stored_similarities, atol=1e-6, rtol=1e-6)
        )
        count_equal = len(indices) == int(saved["sample_count"])
        component_count_equal = bool(
            same_od_count == int(saved["same_od_sample_count"])
            and similar_od_count == int(saved["similar_od_sample_count"])
        )

        if same_od_count:
            same_od_bandwidth, same_od_effective_n = (
                sample_module.silverman_bandwidth(
                    prices[same_od_mask].astype(np.float64),
                    weight_lookup[levels[same_od_mask]],
                )
            )
        else:
            same_od_bandwidth, same_od_effective_n = np.nan, 0.0

        if similar_od_count:
            similar_od_bandwidth, similar_od_effective_n = (
                sample_module.silverman_bandwidth(
                    prices[similar_od_mask].astype(np.float64),
                    similarities[similar_od_mask],
                )
            )
            similar_mean_similarity = float(
                np.mean(similarities[similar_od_mask])
            )
        else:
            similar_od_bandwidth, similar_od_effective_n = np.nan, 0.0
            similar_mean_similarity = np.nan

        if len(indices):
            same_od_mixture_weight, similar_od_mixture_weight = (
                cfg.component_mixture_weights(
                    same_od_effective_n,
                    similar_od_effective_n,
                    0.0 if not similar_od_count else similar_mean_similarity,
                    same_od_count,
                    similar_od_count,
                )
            )
            rebuilt_quantiles = kde_module.kde_quantiles(
                (
                    prices.tolist(),
                    levels.tolist(),
                    similarities.tolist(),
                    same_od_bandwidth,
                    similar_od_bandwidth,
                    same_od_mixture_weight,
                    1,
                )
            )
        else:
            same_od_mixture_weight = np.nan
            similar_od_mixture_weight = np.nan
            rebuilt_quantiles = np.full(
                len(cfg.QUANTILE_COLUMNS), np.nan, dtype=np.float32
            )

        stored_same_od_bandwidth = float(saved["same_od_bandwidth_usd"])
        stored_similar_od_bandwidth = float(saved["similar_od_bandwidth_usd"])
        same_od_bandwidth_equal = bool(
            (np.isnan(same_od_bandwidth) and np.isnan(stored_same_od_bandwidth))
            or np.isclose(
                same_od_bandwidth,
                stored_same_od_bandwidth,
                atol=1e-6,
                rtol=1e-6,
            )
        )
        similar_od_bandwidth_equal = bool(
            (
                np.isnan(similar_od_bandwidth)
                and np.isnan(stored_similar_od_bandwidth)
            )
            or np.isclose(
                similar_od_bandwidth,
                stored_similar_od_bandwidth,
                atol=1e-6,
                rtol=1e-6,
            )
        )
        effective_size_equal = bool(
            np.isclose(
                same_od_effective_n,
                float(saved["same_od_effective_sample_size"]),
                atol=1e-5,
                rtol=1e-6,
            )
            and np.isclose(
                similar_od_effective_n,
                float(saved["similar_od_effective_sample_size"]),
                atol=1e-5,
                rtol=1e-6,
            )
        )
        mixture_weight_equal = bool(
            np.isclose(
                same_od_mixture_weight,
                float(saved["same_od_mixture_weight"]),
                atol=1e-6,
                rtol=0.0,
                equal_nan=True,
            )
            and np.isclose(
                similar_od_mixture_weight,
                float(saved["similar_od_mixture_weight"]),
                atol=1e-6,
                rtol=0.0,
                equal_nan=True,
            )
        )
        stored_quantiles = final[cfg.QUANTILE_COLUMNS].to_numpy(dtype=np.float32)
        finite = np.isfinite(stored_quantiles) & np.isfinite(rebuilt_quantiles)
        if finite.any():
            absolute_error = np.abs(stored_quantiles - rebuilt_quantiles)
            max_quantile_error = float(absolute_error[finite].max())
            mean_quantile_error = float(absolute_error[finite].mean())
        else:
            max_quantile_error = 0.0
            mean_quantile_error = 0.0
        quantiles_equal = bool(
            np.allclose(
                stored_quantiles,
                rebuilt_quantiles,
                atol=1e-5,
                rtol=1e-6,
                equal_nan=True,
            )
        )

        all_checks_passed = all(
            [
                prices_equal,
                levels_equal,
                miles_equal,
                similarities_equal,
                count_equal,
                component_count_equal,
                same_od_bandwidth_equal,
                similar_od_bandwidth_equal,
                effective_size_equal,
                mixture_weight_equal,
                quantiles_equal,
            ]
        )
        if not all_checks_passed:
            failures.append(f"audit_sample={audit_number}, target_id={target_id}")

        target_rows.append(
            {
                "audit_sample": audit_number,
                "target_id": target_id,
                "i": str(target["i"]),
                "j": str(target["j"]),
                "t": pd.Timestamp(target["t"]),
                "D_ijt_observed": float(target["D_ijt_observed"]),
                "target_reference_trip_miles": float(
                    saved["target_reference_trip_miles"]
                ),
                "target_distance_sample_count": int(
                    saved["target_distance_sample_count"]
                ),
                "sample_count": len(indices),
                "same_od_sample_count": same_od_count,
                "similar_od_sample_count": similar_od_count,
                "same_od_effective_sample_size_rebuilt": float(
                    same_od_effective_n
                ),
                "similar_od_effective_sample_size_rebuilt": float(
                    similar_od_effective_n
                ),
                "same_od_bandwidth_usd_stored": stored_same_od_bandwidth,
                "same_od_bandwidth_usd_rebuilt": float(same_od_bandwidth),
                "similar_od_bandwidth_usd_stored": stored_similar_od_bandwidth,
                "similar_od_bandwidth_usd_rebuilt": float(similar_od_bandwidth),
                "same_od_mixture_weight": float(same_od_mixture_weight),
                "similar_od_mixture_weight": float(similar_od_mixture_weight),
                "max_match_level": int(saved["max_match_level"]),
                "layer1_count": int(np.sum(levels == 1)),
                "layer2_count": int(np.sum(levels == 2)),
                "layer3_count": int(np.sum(levels == 3)),
                "layer4_count": int(np.sum(levels == 4)),
                "layer4a_count": int(np.sum(sublevels == "4A")),
                "layer4b_count": int(np.sum(sublevels == "4B")),
                "layer1_prices": json_price_list(prices[levels == 1]),
                "layer2_prices": json_price_list(prices[levels == 2]),
                "layer3_prices": json_price_list(prices[levels == 3]),
                "layer4a_prices": json_price_list(prices[sublevels == "4A"]),
                "layer4b_prices": json_price_list(prices[sublevels == "4B"]),
                "prices_equal_to_saved": int(prices_equal),
                "levels_equal_to_saved": int(levels_equal),
                "trip_miles_equal_to_saved": int(miles_equal),
                "similarities_equal_to_saved": int(similarities_equal),
                "sample_count_equal": int(count_equal),
                "component_count_equal": int(component_count_equal),
                "same_od_bandwidth_equal": int(same_od_bandwidth_equal),
                "similar_od_bandwidth_equal": int(similar_od_bandwidth_equal),
                "effective_size_equal": int(effective_size_equal),
                "mixture_weight_equal": int(mixture_weight_equal),
                "quantiles_equal": int(quantiles_equal),
                "max_quantile_abs_error_usd": max_quantile_error,
                "mean_quantile_abs_error_usd": mean_quantile_error,
                "all_checks_passed": int(all_checks_passed),
            }
        )

        target_ns = int(target["target_ns"])
        component_raw_weights = np.where(
            same_od_mask, weight_lookup[levels], similarities
        ).astype(np.float64)
        component_normalized_weights = np.zeros(len(indices), dtype=np.float64)
        final_kde_weights = np.zeros(len(indices), dtype=np.float64)
        if same_od_count:
            normalized = (
                component_raw_weights[same_od_mask]
                / component_raw_weights[same_od_mask].sum()
            )
            component_normalized_weights[same_od_mask] = normalized
            final_kde_weights[same_od_mask] = same_od_mixture_weight * normalized
        if similar_od_count:
            normalized = (
                component_raw_weights[similar_od_mask]
                / component_raw_weights[similar_od_mask].sum()
            )
            component_normalized_weights[similar_od_mask] = normalized
            final_kde_weights[similar_od_mask] = (
                similar_od_mixture_weight * normalized
            )

        for order_number, (
            history_index,
            level,
            sublevel,
            price,
            raw_weight,
            normalized_weight,
            final_weight,
            distance_similarity,
        ) in enumerate(
            zip(
                indices,
                levels,
                sublevels,
                prices,
                component_raw_weights,
                component_normalized_weights,
                final_kde_weights,
                similarities,
            ),
            start=1,
        ):
            completion_ns = int(collector.completion_ns[history_index])
            history_day = int(collector.start_day[history_index])
            history_slot = int(collector.slot[history_index])
            history_i_index = int(collector.i_index[history_index])
            history_j_index = int(collector.j_index[history_index])
            start_slot_time = data_start + pd.Timedelta(days=history_day) + pd.Timedelta(
                minutes=history_slot * cfg.TIME_STEP_MINUTES
            )
            price_rows.append(
                {
                    "audit_sample": audit_number,
                    "target_id": target_id,
                    "target_i": str(target["i"]),
                    "target_j": str(target["j"]),
                    "target_t": pd.Timestamp(target["t"]),
                    "selected_order": order_number,
                    "history_array_index": int(history_index),
                    "history_i": str(index_to_region[history_i_index]),
                    "history_j": str(index_to_region[history_j_index]),
                    "history_start_slot_time": start_slot_time,
                    "history_completion_time": pd.to_datetime(completion_ns, unit="ns"),
                    "completion_strictly_before_target": int(completion_ns < target_ns),
                    "history_price_usd": float(price),
                    "history_trip_miles": float(collector.trip_miles[history_index]),
                    "history_trip_seconds": float(
                        collector.trip_seconds[history_index]
                    ),
                    "distance_similarity_weight": float(distance_similarity),
                    "match_level": int(level),
                    "match_sublevel": str(sublevel),
                    "kde_component": (
                        "same_od" if int(level) <= 3 else "similar_od"
                    ),
                    "component_raw_weight": float(raw_weight),
                    "component_normalized_weight": float(normalized_weight),
                    "final_kde_weight": float(final_weight),
                    "history_start_day_index": history_day,
                    "history_slot_in_day": history_slot,
                    "history_dow": int(collector.dow[history_index]),
                    "history_weekend_or_holiday": int(
                        collector.weekend_holiday[history_index]
                    ),
                    "target_reference_trip_miles": float(
                        candidates.target_reference_trip_miles
                    ),
                }
            )

        for probability, column, stored_value, rebuilt_value in zip(
            cfg.QUANTILE_LEVELS,
            cfg.QUANTILE_COLUMNS,
            stored_quantiles,
            rebuilt_quantiles,
        ):
            quantile_rows.append(
                {
                    "audit_sample": audit_number,
                    "target_id": target_id,
                    "i": str(target["i"]),
                    "j": str(target["j"]),
                    "t": pd.Timestamp(target["t"]),
                    "probability": float(probability),
                    "quantile_column": column,
                    "stored_quantile_usd": float(stored_value),
                    "rebuilt_quantile_usd": float(rebuilt_value),
                    "absolute_error_usd": float(abs(stored_value - rebuilt_value)),
                }
            )

    target_output = run_dir / "1 随机目标与四层价格汇总.csv"
    price_output = run_dir / "2 四层历史价格逐笔明细.csv"
    quantile_output = run_dir / "3 KDE的101维分位数对比.csv"
    diagnostics_output = run_dir / "4 自动核对结果.json"
    pd.DataFrame(target_rows).to_csv(target_output, index=False, encoding="utf-8-sig")
    pd.DataFrame(price_rows).to_csv(price_output, index=False, encoding="utf-8-sig")
    pd.DataFrame(quantile_rows).to_csv(
        quantile_output, index=False, encoding="utf-8-sig"
    )

    diagnostics = {
        "seed": int(args.seed),
        "sample_size": int(args.sample_size),
        "sampling_pool": pool_name,
        "eligible_target_count": eligible_count,
        "chosen_target_ids": [int(value) for value in chosen_ids],
        "all_targets_passed": len(failures) == 0,
        "failed_targets": failures,
        "target_summary_file": str(target_output),
        "price_detail_file": str(price_output),
        "quantile_comparison_file": str(quantile_output),
    }
    diagnostics_output.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    if failures:
        raise AssertionError(f"随机样本重建核对失败：{failures}")


if __name__ == "__main__":
    main()
