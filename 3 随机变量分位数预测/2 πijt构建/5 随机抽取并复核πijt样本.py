# -*- coding: utf-8 -*-
"""随机抽取已构建的 (i,t) 样本，并从订单级原始数据独立重算 π_ijt。

复核流程：
1. 在 available_flag=1 的全部 (i,t) 中等概率随机抽样；
2. 直接读取订单级原始文件，不使用第 2 步准备的稀疏计数矩阵；
3. 对每个样本重新按 H1–H4 规则分层。H4 按 h 由近到远加入，
   并始终保留一个 15 分钟时段中的全部订单；
4. 从原始订单重新计算 801 个目的地的加权概率；
5. 与已构建的 CSR 概率矩阵及诊断信息逐项对比。

默认固定随机种子，因此重复执行可得到相同样本。
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

SELECTION_FILE = cfg.RANDOM_AUDIT_DIR / "1_随机样本清单.csv"
ORDER_DETAIL_FILE = cfg.RANDOM_AUDIT_DIR / "2_使用的原始订单明细.parquet"
ORDER_DETAIL_CSV_FILE = cfg.RANDOM_AUDIT_DIR / "2_使用的原始订单明细.csv"
PERIOD_SUMMARY_FILE = cfg.RANDOM_AUDIT_DIR / "3_历史时段分层汇总.csv"
PROBABILITY_COMPARISON_FILE = cfg.RANDOM_AUDIT_DIR / "4_目的地概率逐项对比.csv"
AUDIT_SUMMARY_FILE = cfg.RANDOM_AUDIT_DIR / "5_样本复核汇总.csv"
AUDIT_REPORT_FILE = cfg.RANDOM_AUDIT_DIR / "6_复核报告.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="随机抽取并独立复核 π_ijt 样本。")
    parser.add_argument("--sample-count", type=int, default=3, help="抽样数量。")
    parser.add_argument("--seed", type=int, default=20240921, help="随机种子。")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖本脚本管理的上一次复核结果。",
    )
    return parser.parse_args()


def managed_outputs() -> list[Path]:
    return [
        SELECTION_FILE,
        ORDER_DETAIL_FILE,
        ORDER_DETAIL_CSV_FILE,
        PERIOD_SUMMARY_FILE,
        PROBABILITY_COMPARISON_FILE,
        AUDIT_SUMMARY_FILE,
        AUDIT_REPORT_FILE,
    ]


def clean_region_id(series: pd.Series) -> pd.Series:
    """将 Census Tract 编号统一为 11 位字符串。"""

    cleaned = series.astype("string").str.strip()
    cleaned = cleaned.str.replace(r"\.0$", "", regex=True)
    cleaned = cleaned.str.extract(r"(\d+)", expand=False)
    return cleaned.str.zfill(11)


def circular_slot_distance(slot: pd.Series, target_slot: int) -> pd.Series:
    difference = (slot.astype(np.int16) - int(target_slot)).abs()
    return np.minimum(difference, cfg.SLOTS_PER_DAY - difference)


def select_available_samples(
    sample_count: int,
    seed: int,
    region_map: pd.DataFrame,
    time_map: pd.DataFrame,
) -> pd.DataFrame:
    """在全部可计算 π_ijt 的 (i,t) 中等概率无放回抽样。"""

    available_by_origin: list[np.ndarray] = []
    available_counts = np.zeros(cfg.EXPECTED_REGION_COUNT, dtype=np.int64)
    for origin_index in range(cfg.EXPECTED_REGION_COUNT):
        path = cfg.diagnostic_part_file(origin_index)
        if not path.exists():
            raise FileNotFoundError(f"缺少诊断分片：{path}")
        flags = pd.read_parquet(path, columns=["available_flag"])[
            "available_flag"
        ].to_numpy(dtype=np.int8)
        if len(flags) != cfg.EXPECTED_TIME_COUNT:
            raise ValueError(f"{path.name} 行数不正确。")
        indices = np.flatnonzero(flags == 1).astype(np.int16)
        available_by_origin.append(indices)
        available_counts[origin_index] = len(indices)

    total_available = int(available_counts.sum())
    if sample_count < 1 or sample_count > total_available:
        raise ValueError(
            f"sample-count 必须位于 [1, {total_available:,}] 之间。"
        )

    rng = np.random.default_rng(seed)
    global_ranks = rng.choice(total_available, size=sample_count, replace=False)
    cumulative = np.cumsum(available_counts)
    records: list[dict[str, object]] = []
    for sample_id, global_rank in enumerate(global_ranks, start=1):
        origin_index = int(np.searchsorted(cumulative, global_rank, side="right"))
        previous = int(cumulative[origin_index - 1]) if origin_index else 0
        within_origin_rank = int(global_rank) - previous
        time_index = int(available_by_origin[origin_index][within_origin_rank])
        records.append(
            {
                "sample_id": sample_id,
                "random_seed": seed,
                "available_population_rank": int(global_rank),
                "origin_index": origin_index,
                "origin_sequence_1based": origin_index + 1,
                "i": str(region_map.iloc[origin_index]["region_id"]),
                "time_index": time_index,
                "t": pd.Timestamp(time_map.iloc[time_index]["t"]),
            }
        )
    return pd.DataFrame(records)


def load_selected_origin_orders(
    selected_origins: set[str],
    internal_regions: set[str],
    latest_target_time: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """流式读取订单级原始数据，只保留复核可能用到的历史订单。"""

    source = pq.ParquetFile(cfg.ORDER_LEVEL_FILE)
    required = [
        "Trip ID",
        "Trip Start Timestamp",
        "Trip End Timestamp",
        "Trip Seconds",
        "Trip Miles",
        "Pickup Census Tract",
        "Dropoff Census Tract",
        "Fare",
        "Additional Charges",
        "Trip Total fare",
    ]
    missing = sorted(set(required) - set(source.schema.names))
    if missing:
        raise ValueError(f"订单级文件缺少字段：{missing}")

    parts: list[pd.DataFrame] = []
    input_rows = 0
    valid_rows = 0
    for batch in source.iter_batches(
        batch_size=cfg.PARQUET_BATCH_SIZE,
        columns=required,
    ):
        data = batch.to_pandas()
        input_rows += len(data)
        data["i"] = clean_region_id(data["Pickup Census Tract"])
        data["j"] = clean_region_id(data["Dropoff Census Tract"])
        data["h"] = pd.to_datetime(
            data["Trip Start Timestamp"], errors="coerce"
        ).astype("datetime64[ns]")
        for column in ("Trip Seconds", "Trip Miles", "Trip Total fare"):
            data[column] = pd.to_numeric(data[column], errors="coerce")

        # 与上游 OD-15min 聚合脚本使用相同的有效性口径。
        valid = (
            data["h"].notna()
            & data["i"].notna()
            & data["j"].notna()
            & data["Trip Total fare"].notna()
            & data["Trip Seconds"].notna()
            & data["Trip Miles"].notna()
        )
        valid_rows += int(valid.sum())
        keep = (
            valid
            & data["i"].isin(selected_origins)
            & data["j"].isin(internal_regions)
            & (data["h"] >= pd.Timestamp(cfg.DATA_START))
            & (data["h"] < latest_target_time)
        )
        if keep.any():
            parts.append(data.loc[keep].copy())

    if not parts:
        raise ValueError("未从订单级原始数据读取到候选历史订单。")
    orders = pd.concat(parts, ignore_index=True)
    orders = orders.sort_values(["i", "h", "j", "Trip ID"]).reset_index(drop=True)
    return orders, {
        "order_level_input_rows": int(input_rows),
        "order_level_valid_rows": int(valid_rows),
        "selected_origin_internal_history_orders": int(len(orders)),
    }


def audit_one_sample(
    sample: pd.Series,
    source_orders: pd.DataFrame,
    region_map: pd.DataFrame,
    time_map: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """从订单级数据重新分层、计算概率，并与构建结果对比。"""

    sample_id = int(sample["sample_id"])
    origin_index = int(sample["origin_index"])
    time_index = int(sample["time_index"])
    target_i = str(sample["i"])
    target_t = pd.Timestamp(sample["t"])
    target_slot = int(time_map.iloc[time_index]["slot_in_day"])
    target_dow = int(time_map.iloc[time_index]["dow"])
    target_holiday = int(time_map.iloc[time_index]["is_holiday"])

    history = source_orders.loc[
        (source_orders["i"] == target_i) & (source_orders["h"] < target_t)
    ].copy()
    if history.empty:
        raise AssertionError(f"样本 {sample_id} 已标记可用，但原始订单中没有历史。")

    history["slot_h"] = (
        history["h"].dt.hour * 4 + history["h"].dt.minute // cfg.TIME_STEP_MINUTES
    ).astype(np.int16)
    history["dow_h"] = history["h"].dt.dayofweek.astype(np.int8)
    holiday_dates = {pd.Timestamp(value).date() for value in cfg.HOLIDAY_DATES}
    history["is_holiday_h"] = history["h"].dt.date.isin(holiday_dates).astype(np.int8)
    history["circular_slot_distance"] = circular_slot_distance(
        history["slot_h"], target_slot
    ).astype(np.int16)
    history["layer"] = np.int8(0)

    mask1 = (
        (history["slot_h"] == target_slot)
        & (history["dow_h"] == target_dow)
        & (history["is_holiday_h"] == target_holiday)
    )
    mask2 = (history["slot_h"] == target_slot) & ~mask1
    mask3 = history["circular_slot_distance"].between(
        1, cfg.TIME_NEIGHBOR_SLOTS
    )
    history.loc[mask1, "layer"] = 1
    history.loc[mask2, "layer"] = 2
    history.loc[mask3, "layer"] = 3

    c123 = int((history["layer"] > 0).sum())
    selected_h4_times: list[pd.Timestamp] = []
    if c123 < cfg.N_MIN:
        candidate_periods = (
            history.loc[history["layer"] == 0]
            .groupby("h", sort=False)
            .size()
            .sort_index(ascending=False)
        )
        total = c123
        for h, period_order_count in candidate_periods.items():
            selected_h4_times.append(pd.Timestamp(h))
            total += int(period_order_count)
            if total >= cfg.N_STOP:
                break
        if selected_h4_times:
            history.loc[history["h"].isin(selected_h4_times), "layer"] = 4

    used = history.loc[history["layer"] > 0].copy()
    layer_weights = {int(k): float(v) for k, v in cfg.LAYER_WEIGHTS.items()}
    used["layer_weight"] = used["layer"].map(layer_weights).astype(np.float32)
    used["weighted_order_contribution"] = used["layer_weight"]
    used["sample_id"] = sample_id
    used["target_i"] = target_i
    used["target_origin_index"] = origin_index
    used["target_time_index"] = time_index
    used["target_t"] = target_t

    region_to_index = dict(
        zip(region_map["region_id"].astype(str), region_map["region_index"].astype(int))
    )
    time_to_index = {
        pd.Timestamp(t): int(index)
        for index, t in zip(time_map["time_index"], time_map["t"])
    }
    used["destination_index"] = used["j"].map(region_to_index).astype(np.int16)
    used["history_time_index"] = used["h"].map(time_to_index).astype(np.int16)
    h4_rank = {h: rank for rank, h in enumerate(selected_h4_times, start=1)}
    used["layer4_recency_rank"] = (
        used["h"].map(h4_rank).fillna(0).astype(np.int16)
    )

    period_summary = (
        used.groupby(["layer", "h"], as_index=False, sort=True)
        .agg(
            period_order_count=("Trip ID", "size"),
            destination_count=("j", "nunique"),
            layer_weight=("layer_weight", "first"),
            weighted_period_order_count=("weighted_order_contribution", "sum"),
            layer4_recency_rank=("layer4_recency_rank", "first"),
        )
        .sort_values(["layer", "h"])
    )
    period_summary.insert(0, "sample_id", sample_id)
    period_summary.insert(1, "target_i", target_i)
    period_summary.insert(2, "target_t", target_t)
    period_summary["layer4_stop_trigger_flag"] = (
        (period_summary["layer"] == 4)
        & (
            period_summary["layer4_recency_rank"]
            == len(selected_h4_times)
        )
    ).astype(np.int8)

    region_count = cfg.EXPECTED_REGION_COUNT
    raw_by_layer = np.zeros((4, region_count), dtype=np.int64)
    for layer, group in used.groupby("layer"):
        counts = group.groupby("destination_index").size()
        raw_by_layer[int(layer) - 1, counts.index.to_numpy(dtype=np.int32)] = (
            counts.to_numpy(dtype=np.int64)
        )
    weighted_by_destination = np.zeros(region_count, dtype=np.float64)
    for layer in range(1, 5):
        weighted_by_destination += layer_weights[layer] * raw_by_layer[layer - 1]
    weighted_total = float(weighted_by_destination.sum())
    pi_recalculated = weighted_by_destination / weighted_total

    built_matrix = sparse.load_npz(cfg.pi_part_file(origin_index)).tocsr()
    pi_built = built_matrix.getrow(time_index).toarray().ravel().astype(np.float64)
    difference = pi_recalculated - pi_built
    probability_comparison = pd.DataFrame(
        {
            "sample_id": sample_id,
            "target_i": target_i,
            "target_t": target_t,
            "destination_index": region_map["region_index"].to_numpy(dtype=np.int16),
            "j": region_map["region_id"].astype(str).to_numpy(),
            "layer1_order_count": raw_by_layer[0],
            "layer2_order_count": raw_by_layer[1],
            "layer3_order_count": raw_by_layer[2],
            "layer4_order_count": raw_by_layer[3],
            "weighted_destination_order_count": weighted_by_destination,
            "pi_recalculated_from_raw_orders": pi_recalculated,
            "pi_built": pi_built,
            "difference_recalculated_minus_built": difference,
            "absolute_difference": np.abs(difference),
        }
    )

    diagnostic = pd.read_parquet(cfg.diagnostic_part_file(origin_index))
    diagnostic_row = diagnostic.loc[diagnostic["time_index"] == time_index]
    if len(diagnostic_row) != 1:
        raise AssertionError(f"样本 {sample_id} 诊断行不唯一。")
    diagnostic_row = diagnostic_row.iloc[0]

    layer_counts = raw_by_layer.sum(axis=1).astype(np.int64)
    raw_total = int(layer_counts.sum())
    available_flag = int(raw_total > 0)
    insufficient_flag = int(0 < raw_total < cfg.N_MIN)
    layer4_period_count = len(selected_h4_times)
    layer4_used_flag = int(layer4_period_count > 0)
    exhausted_flag = int(layer_counts[:3].sum() < cfg.N_MIN and raw_total < cfg.N_STOP)
    overshoot = int(
        raw_total - cfg.N_STOP
        if layer4_period_count > 0 and raw_total >= cfg.N_STOP
        else 0
    )
    max_match_level = max((index + 1 for index, x in enumerate(layer_counts) if x > 0), default=0)
    tolerance = float(cfg.PROBABILITY_SUM_ATOL)
    max_abs_difference = float(np.max(np.abs(difference)))

    exact_checks = {
        "layer1_count_match": int(layer_counts[0]) == int(diagnostic_row["layer1_order_count"]),
        "layer2_count_match": int(layer_counts[1]) == int(diagnostic_row["layer2_order_count"]),
        "layer3_count_match": int(layer_counts[2]) == int(diagnostic_row["layer3_order_count"]),
        "layer4_count_match": int(layer_counts[3]) == int(diagnostic_row["layer4_order_count"]),
        "layer4_period_count_match": layer4_period_count == int(diagnostic_row["layer4_period_count"]),
        "total_count_match": raw_total == int(diagnostic_row["total_history_order_count"]),
        "weighted_total_match": bool(
            np.isclose(
                weighted_total,
                float(diagnostic_row["weighted_history_order_count"]),
                rtol=1e-6,
                atol=1e-4,
            )
        ),
        "available_flag_match": available_flag == int(diagnostic_row["available_flag"]),
        "insufficient_flag_match": insufficient_flag == int(diagnostic_row["insufficient_sample_flag"]),
        "layer4_used_flag_match": layer4_used_flag == int(diagnostic_row["layer4_used_flag"]),
        "layer4_exhausted_flag_match": exhausted_flag == int(diagnostic_row["layer4_candidates_exhausted_flag"]),
        "overshoot_match": overshoot == int(diagnostic_row["layer4_stop_overshoot_orders"]),
        "max_match_level_match": max_match_level == int(diagnostic_row["max_match_level"]),
        "pi_nnz_match": int(np.count_nonzero(pi_recalculated)) == int(diagnostic_row["pi_nnz"]),
        "probability_vector_match": max_abs_difference <= tolerance,
    }
    summary: dict[str, object] = {
        "sample_id": sample_id,
        "origin_index": origin_index,
        "origin_sequence_1based": origin_index + 1,
        "i": target_i,
        "time_index": time_index,
        "t": target_t,
        "layer1_orders_recalculated": int(layer_counts[0]),
        "layer2_orders_recalculated": int(layer_counts[1]),
        "layer3_orders_recalculated": int(layer_counts[2]),
        "layer4_orders_recalculated": int(layer_counts[3]),
        "layer4_periods_recalculated": layer4_period_count,
        "total_orders_recalculated": raw_total,
        "weighted_orders_recalculated": weighted_total,
        "layer1_orders_built": int(diagnostic_row["layer1_order_count"]),
        "layer2_orders_built": int(diagnostic_row["layer2_order_count"]),
        "layer3_orders_built": int(diagnostic_row["layer3_order_count"]),
        "layer4_orders_built": int(diagnostic_row["layer4_order_count"]),
        "layer4_periods_built": int(diagnostic_row["layer4_period_count"]),
        "total_orders_built": int(diagnostic_row["total_history_order_count"]),
        "weighted_orders_built": float(diagnostic_row["weighted_history_order_count"]),
        "available_flag_recalculated": available_flag,
        "available_flag_built": int(diagnostic_row["available_flag"]),
        "insufficient_flag_recalculated": insufficient_flag,
        "insufficient_flag_built": int(diagnostic_row["insufficient_sample_flag"]),
        "layer4_used_flag_recalculated": layer4_used_flag,
        "layer4_used_flag_built": int(diagnostic_row["layer4_used_flag"]),
        "overshoot_orders_recalculated": overshoot,
        "overshoot_orders_built": int(diagnostic_row["layer4_stop_overshoot_orders"]),
        "pi_nnz_recalculated": int(np.count_nonzero(pi_recalculated)),
        "pi_nnz_built": int(diagnostic_row["pi_nnz"]),
        "probability_sum_recalculated": float(pi_recalculated.sum()),
        "probability_sum_built": float(pi_built.sum()),
        "maximum_absolute_probability_difference": max_abs_difference,
        "destinations_over_tolerance": int((np.abs(difference) > tolerance).sum()),
        **{key: int(value) for key, value in exact_checks.items()},
        "all_checks_passed": int(all(exact_checks.values())),
    }
    return used, period_summary, probability_comparison, summary


def main() -> None:
    args = parse_args()
    cfg.ensure_output_directories()
    if not cfg.ORDER_LEVEL_FILE.exists():
        raise FileNotFoundError(f"缺少订单级原始数据：{cfg.ORDER_LEVEL_FILE}")

    existing = [path for path in managed_outputs() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "随机复核结果已存在；如需重新抽样，请使用 --overwrite：\n"
            + "\n".join(str(path) for path in existing)
        )
    if args.overwrite:
        for path in managed_outputs():
            if path.exists():
                path.unlink()

    region_map = pd.read_csv(cfg.REGION_MAPPING_FILE, dtype={"region_id": "string"})
    region_map = region_map.sort_values("region_index").reset_index(drop=True)
    time_map = pd.read_parquet(cfg.TIME_MAPPING_FILE).sort_values("time_index").reset_index(drop=True)
    time_map["t"] = pd.to_datetime(time_map["t"])

    print("正在从全部可用 (i,t) 中随机抽样……")
    selection = select_available_samples(args.sample_count, args.seed, region_map, time_map)
    selected_origins = set(selection["i"].astype(str))
    internal_regions = set(region_map["region_id"].astype(str))
    print("正在从订单级原始数据读取相关历史订单……")
    source_orders, source_diagnostics = load_selected_origin_orders(
        selected_origins,
        internal_regions,
        pd.Timestamp(selection["t"].max()),
    )

    order_parts: list[pd.DataFrame] = []
    period_parts: list[pd.DataFrame] = []
    probability_parts: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for _, sample in selection.iterrows():
        print(
            f"正在复核样本 {int(sample['sample_id'])}/{len(selection)}："
            f"i={sample['i']}, t={pd.Timestamp(sample['t'])}"
        )
        orders, periods, comparison, summary = audit_one_sample(
            sample,
            source_orders,
            region_map,
            time_map,
        )
        order_parts.append(orders)
        period_parts.append(periods)
        probability_parts.append(comparison)
        summaries.append(summary)

    used_orders = pd.concat(order_parts, ignore_index=True)
    period_summary = pd.concat(period_parts, ignore_index=True)
    probability_comparison = pd.concat(probability_parts, ignore_index=True)
    audit_summary = pd.DataFrame(summaries)
    selection = selection.merge(
        audit_summary[
            [
                "sample_id",
                "total_orders_recalculated",
                "layer4_periods_recalculated",
                "maximum_absolute_probability_difference",
                "all_checks_passed",
            ]
        ],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )

    selection.to_csv(SELECTION_FILE, index=False, encoding="utf-8-sig")
    used_orders.to_parquet(
        ORDER_DETAIL_FILE,
        index=False,
        compression=cfg.PARQUET_COMPRESSION,
    )
    used_orders.to_csv(ORDER_DETAIL_CSV_FILE, index=False, encoding="utf-8-sig")
    period_summary.to_csv(PERIOD_SUMMARY_FILE, index=False, encoding="utf-8-sig")
    probability_comparison.to_csv(
        PROBABILITY_COMPARISON_FILE,
        index=False,
        encoding="utf-8-sig",
    )
    audit_summary.to_csv(AUDIT_SUMMARY_FILE, index=False, encoding="utf-8-sig")

    report = {
        "sample_count": int(len(selection)),
        "random_seed": int(args.seed),
        "sampling_population": "all rows with available_flag=1",
        "order_level_source": str(cfg.ORDER_LEVEL_FILE),
        **source_diagnostics,
        "contributing_order_rows_exported": int(len(used_orders)),
        "history_period_rows_exported": int(len(period_summary)),
        "destination_comparison_rows_exported": int(len(probability_comparison)),
        "passed_sample_count": int(audit_summary["all_checks_passed"].sum()),
        "failed_sample_count": int((audit_summary["all_checks_passed"] == 0).sum()),
        "maximum_absolute_probability_difference": float(
            audit_summary["maximum_absolute_probability_difference"].max()
        ),
        "probability_tolerance": float(cfg.PROBABILITY_SUM_ATOL),
        "output_directory": str(cfg.RANDOM_AUDIT_DIR),
    }
    AUDIT_REPORT_FILE.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(audit_summary.to_string(index=False))
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if not bool(audit_summary["all_checks_passed"].all()):
        failed_ids = audit_summary.loc[
            audit_summary["all_checks_passed"] == 0, "sample_id"
        ].tolist()
        raise AssertionError(f"随机样本复核失败：sample_id={failed_ids}")


if __name__ == "__main__":
    main()
