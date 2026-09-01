# -*- coding: utf-8 -*-
"""按出发区域滚动构建 π_ijt 历史目的地概率分布。

运行前必须先执行：

    2 准备πijt数值索引.py

本脚本不再读取原始 Parquet，不执行 pandas groupby，只读取已准备的
CSR 订单数矩阵和时间数组。每个出发区域独立计算并输出一个稀疏概率矩阵，
因此支持多进程和断点续跑。

分层规则：
1. H1：同 dow、同 holiday、同 slot；
2. H2：同 slot，排除 H1；
3. H3：环形 slot 距离为 1~4，与 H1/H2 天然互斥；
4. 前三层始终完整保留。仅当其原始订单数小于 N_MIN 时，
   才从其余历史时段中由近到远加入 H4；
5. H4 按完整 15 分钟时段加入。加入最后一个时段后首次达到
   N_STOP 就停止，不会为了凑整 N_STOP 而截断该时段的订单。

输出：
1. pi_distribution_parts/origin-XXXX.npz：[time, destination] CSR 概率矩阵；
2. diagnostic_parts/origin-XXXX.parquet：每个 (i,t) 的分层订单数与标记。

并行运行示例：

    python "3 滚动构建πijt历史分布.py" --workers 4 --resume
"""

from __future__ import annotations

import os

# 多进程时限制每个进程内部数学库线程数，避免过度竞争。
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

# fork 工作进程共享下列只读数据。
_COUNT_MATRIX: sparse.csr_matrix | None = None
_ROW_TOTAL: np.ndarray | None = None
_TIME_SLOT: np.ndarray | None = None
_TIME_DOW: np.ndarray | None = None
_TIME_HOLIDAY: np.ndarray | None = None
_TIME_VALUES: np.ndarray | None = None
_REGION_IDS: np.ndarray | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="滚动构建 π_ijt 历史分布。")
    parser.add_argument("--workers", type=int, default=cfg.DEFAULT_WORKERS)
    parser.add_argument(
        "--start-origin",
        type=int,
        default=1,
        help="起始区域序号（1-based，包含）。",
    )
    parser.add_argument(
        "--end-origin",
        type=int,
        default=None,
        help="结束区域序号（1-based，包含）。",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def circular_slot_distance(left: int, right: int) -> int:
    difference = abs(int(left) - int(right))
    return min(difference, cfg.SLOTS_PER_DAY - difference)


def load_prepared_data() -> None:
    """一次加载已准备的数值索引。"""

    global _COUNT_MATRIX, _ROW_TOTAL, _TIME_SLOT, _TIME_DOW
    global _TIME_HOLIDAY, _TIME_VALUES, _REGION_IDS

    required = [
        cfg.REGION_MAPPING_FILE,
        cfg.TIME_MAPPING_FILE,
        cfg.HISTORY_COUNT_MATRIX_FILE,
        cfg.HISTORY_ROW_TOTAL_FILE,
        cfg.TIME_SLOT_FILE,
        cfg.TIME_DOW_FILE,
        cfg.TIME_HOLIDAY_FILE,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "缺少第 2 步的数值索引：\n" + "\n".join(missing)
        )

    _COUNT_MATRIX = sparse.load_npz(cfg.HISTORY_COUNT_MATRIX_FILE).tocsr()
    _COUNT_MATRIX.sort_indices()
    _ROW_TOTAL = np.load(cfg.HISTORY_ROW_TOTAL_FILE, mmap_mode="r")
    _TIME_SLOT = np.load(cfg.TIME_SLOT_FILE, mmap_mode="r")
    _TIME_DOW = np.load(cfg.TIME_DOW_FILE, mmap_mode="r")
    _TIME_HOLIDAY = np.load(cfg.TIME_HOLIDAY_FILE, mmap_mode="r")

    time_map = pd.read_parquet(cfg.TIME_MAPPING_FILE, columns=["time_index", "t"])
    time_map = time_map.sort_values("time_index")
    _TIME_VALUES = pd.to_datetime(time_map["t"]).to_numpy(dtype="datetime64[ns]")
    region_map = pd.read_csv(cfg.REGION_MAPPING_FILE, dtype={"region_id": "string"})
    region_map = region_map.sort_values("region_index")
    _REGION_IDS = region_map["region_id"].astype(str).to_numpy()

    expected_shape = (
        cfg.EXPECTED_REGION_COUNT * cfg.EXPECTED_TIME_COUNT,
        cfg.EXPECTED_REGION_COUNT,
    )
    if _COUNT_MATRIX.shape != expected_shape:
        raise ValueError(
            f"历史订单数矩阵形状为 {_COUNT_MATRIX.shape}，"
            f"预期为 {expected_shape}。"
        )
    if len(_ROW_TOTAL) != expected_shape[0]:
        raise ValueError("历史订单行总量数组长度不正确。")
    if not (
        len(_TIME_SLOT)
        == len(_TIME_DOW)
        == len(_TIME_HOLIDAY)
        == len(_TIME_VALUES)
        == cfg.EXPECTED_TIME_COUNT
    ):
        raise ValueError("时间索引数组长度不一致。")
    if len(_REGION_IDS) != cfg.EXPECTED_REGION_COUNT:
        raise ValueError("区域映射长度不正确。")


def add_history_row(
    origin_counts: sparse.csr_matrix,
    history_time: int,
    slot_counts: np.ndarray,
    strict_counts: np.ndarray,
    near_counts: np.ndarray,
    slot_totals: np.ndarray,
    strict_totals: np.ndarray,
    near_totals: np.ndarray,
) -> bool:
    """把一个刚刚成为历史的完整时段加入前三层累计器。"""

    assert _TIME_SLOT is not None
    assert _TIME_DOW is not None
    assert _TIME_HOLIDAY is not None

    start = int(origin_counts.indptr[history_time])
    end = int(origin_counts.indptr[history_time + 1])
    if start == end:
        return False

    destinations = origin_counts.indices[start:end]
    values = origin_counts.data[start:end].astype(np.int64, copy=False)
    total = int(values.sum(dtype=np.int64))
    slot = int(_TIME_SLOT[history_time])
    dow = int(_TIME_DOW[history_time])
    holiday = int(_TIME_HOLIDAY[history_time])

    slot_counts[slot, destinations] += values
    slot_totals[slot] += total
    strict_counts[dow, slot, holiday, destinations] += values
    strict_totals[dow, slot, holiday] += total

    # 历史 slot=s 属于目标 slot=s±1,...,s±4 的第三层。
    for offset in range(1, cfg.TIME_NEIGHBOR_SLOTS + 1):
        for target_slot in (
            (slot - offset) % cfg.SLOTS_PER_DAY,
            (slot + offset) % cfg.SLOTS_PER_DAY,
        ):
            near_counts[target_slot, destinations] += values
            near_totals[target_slot] += total

    return True


def build_origin_distribution(
    origin_index: int,
    origin_counts: sparse.csr_matrix,
    origin_row_totals: np.ndarray,
) -> tuple[sparse.csr_matrix, pd.DataFrame, dict[str, int | float]]:
    """构建单个出发区域的全部时段 π_ijt。"""

    assert _TIME_SLOT is not None
    assert _TIME_DOW is not None
    assert _TIME_HOLIDAY is not None
    assert _TIME_VALUES is not None
    assert _REGION_IDS is not None

    time_count = cfg.EXPECTED_TIME_COUNT
    region_count = cfg.EXPECTED_REGION_COUNT

    # 一次只为一个出发区域维护累计器，避免构造全局稠密张量。
    slot_counts = np.zeros((cfg.SLOTS_PER_DAY, region_count), dtype=np.int64)
    strict_counts = np.zeros(
        (7, cfg.SLOTS_PER_DAY, 2, region_count), dtype=np.int64
    )
    near_counts = np.zeros((cfg.SLOTS_PER_DAY, region_count), dtype=np.int64)
    slot_totals = np.zeros(cfg.SLOTS_PER_DAY, dtype=np.int64)
    strict_totals = np.zeros((7, cfg.SLOTS_PER_DAY, 2), dtype=np.int64)
    near_totals = np.zeros(cfg.SLOTS_PER_DAY, dtype=np.int64)

    # 仅记录有订单的历史时段，第四层不扫描空时段。
    positive_history_times: list[int] = []

    indptr = np.zeros(time_count + 1, dtype=np.int64)
    index_parts: list[np.ndarray] = []
    probability_parts: list[np.ndarray] = []

    layer1_count = np.zeros(time_count, dtype=np.int64)
    layer2_count = np.zeros(time_count, dtype=np.int64)
    layer3_count = np.zeros(time_count, dtype=np.int64)
    layer4_count = np.zeros(time_count, dtype=np.int64)
    layer4_period_count = np.zeros(time_count, dtype=np.int16)
    total_history_count = np.zeros(time_count, dtype=np.int64)
    weighted_history_count = np.zeros(time_count, dtype=np.float64)
    available_flag = np.zeros(time_count, dtype=np.int8)
    insufficient_sample_flag = np.zeros(time_count, dtype=np.int8)
    layer4_used_flag = np.zeros(time_count, dtype=np.int8)
    layer4_candidates_exhausted_flag = np.zeros(time_count, dtype=np.int8)
    layer4_stop_overshoot_orders = np.zeros(time_count, dtype=np.int32)
    max_match_level = np.zeros(time_count, dtype=np.int8)
    pi_nnz = np.zeros(time_count, dtype=np.int16)
    probability_sum = np.zeros(time_count, dtype=np.float32)

    w1 = float(cfg.LAYER_WEIGHTS[1])
    w2 = float(cfg.LAYER_WEIGHTS[2])
    w3 = float(cfg.LAYER_WEIGHTS[3])
    w4 = float(cfg.LAYER_WEIGHTS[4])

    for target_time in range(time_count):
        # 先把 t-1 加入历史，因此所有累计器中始终只有 h<t。
        if target_time > 0:
            history_time = target_time - 1
            if add_history_row(
                origin_counts,
                history_time,
                slot_counts,
                strict_counts,
                near_counts,
                slot_totals,
                strict_totals,
                near_totals,
            ):
                positive_history_times.append(history_time)

        slot = int(_TIME_SLOT[target_time])
        dow = int(_TIME_DOW[target_time])
        holiday = int(_TIME_HOLIDAY[target_time])

        c1 = int(strict_totals[dow, slot, holiday])
        same_slot_total = int(slot_totals[slot])
        c2 = same_slot_total - c1
        c3 = int(near_totals[slot])
        if c2 < 0:
            raise AssertionError("第二层订单数出现负数。")

        layer1_count[target_time] = c1
        layer2_count[target_time] = c2
        layer3_count[target_time] = c3
        first_three_total = c1 + c2 + c3

        # H1 是同 slot 中的高权重子集；因此可用
        # w2*same_slot + (w1-w2)*H1 避免显式构造 H2 向量。
        weighted_counts = (
            w2 * slot_counts[slot]
            + (w1 - w2) * strict_counts[dow, slot, holiday]
            + w3 * near_counts[slot]
        ).astype(np.float64, copy=False)
        weighted_total = w1 * c1 + w2 * c2 + w3 * c3

        c4 = 0
        c4_periods = 0
        if first_three_total < cfg.N_MIN:
            # 第四层仅使用环形 slot 距离 > 4 的其余时段，
            # 与前三层严格互斥。
            for history_time in reversed(positive_history_times):
                history_slot = int(_TIME_SLOT[history_time])
                if circular_slot_distance(history_slot, slot) <= cfg.TIME_NEIGHBOR_SLOTS:
                    continue

                start = int(origin_counts.indptr[history_time])
                end = int(origin_counts.indptr[history_time + 1])
                destinations = origin_counts.indices[start:end]
                values = origin_counts.data[start:end].astype(np.int64, copy=False)
                period_total = int(origin_row_totals[history_time])
                if period_total <= 0 or start == end:
                    continue

                # 必须先加入该时段的全部目的地订单，然后才检查
                # N_STOP。因此这里不会截断一个历史时段。
                weighted_counts[destinations] += w4 * values
                weighted_total += w4 * period_total
                c4 += period_total
                c4_periods += 1
                if first_three_total + c4 >= cfg.N_STOP:
                    break

            if first_three_total + c4 < cfg.N_STOP:
                layer4_candidates_exhausted_flag[target_time] = 1

        layer4_count[target_time] = c4
        layer4_period_count[target_time] = c4_periods
        layer4_used_flag[target_time] = int(c4_periods > 0)
        raw_total = first_three_total + c4
        total_history_count[target_time] = raw_total
        weighted_history_count[target_time] = weighted_total
        if c4_periods > 0 and raw_total >= cfg.N_STOP:
            layer4_stop_overshoot_orders[target_time] = raw_total - cfg.N_STOP

        if raw_total > 0:
            available_flag[target_time] = 1
            insufficient_sample_flag[target_time] = int(raw_total < cfg.N_MIN)
            positive = weighted_counts > 0
            destinations = np.flatnonzero(positive).astype(np.int32, copy=False)
            if not np.isclose(
                float(weighted_counts[positive].sum()),
                weighted_total,
                rtol=0.0,
                atol=1e-8,
            ):
                raise AssertionError("加权目的地订单数与加权总订单数不一致。")
            probabilities = (
                weighted_counts[destinations] / weighted_total
            ).astype(np.float32)
            index_parts.append(destinations)
            probability_parts.append(probabilities)
            indptr[target_time + 1] = indptr[target_time] + len(destinations)
            pi_nnz[target_time] = len(destinations)
            probability_sum[target_time] = float(probabilities.sum(dtype=np.float64))
        else:
            indptr[target_time + 1] = indptr[target_time]

        if c4 > 0:
            max_match_level[target_time] = 4
        elif c3 > 0:
            max_match_level[target_time] = 3
        elif c2 > 0:
            max_match_level[target_time] = 2
        elif c1 > 0:
            max_match_level[target_time] = 1

    if index_parts:
        indices = np.concatenate(index_parts).astype(np.int32, copy=False)
        probabilities = np.concatenate(probability_parts).astype(np.float32, copy=False)
    else:
        indices = np.empty(0, dtype=np.int32)
        probabilities = np.empty(0, dtype=np.float32)

    pi_matrix = sparse.csr_matrix(
        (probabilities, indices, indptr),
        shape=(time_count, region_count),
        dtype=np.float32,
    )
    pi_matrix.sort_indices()

    diagnostics = pd.DataFrame(
        {
            "row_id": (
                origin_index * time_count
                + np.arange(time_count, dtype=np.int64)
            ),
            "origin_index": np.full(time_count, origin_index, dtype=np.int16),
            "i": np.full(time_count, _REGION_IDS[origin_index], dtype=object),
            "time_index": np.arange(time_count, dtype=np.int16),
            "t": _TIME_VALUES,
            "layer1_order_count": layer1_count,
            "layer2_order_count": layer2_count,
            "layer3_order_count": layer3_count,
            "layer4_order_count": layer4_count,
            "layer4_period_count": layer4_period_count,
            "total_history_order_count": total_history_count,
            "weighted_history_order_count": weighted_history_count.astype(np.float32),
            "available_flag": available_flag,
            "insufficient_sample_flag": insufficient_sample_flag,
            "layer4_used_flag": layer4_used_flag,
            "layer4_candidates_exhausted_flag": layer4_candidates_exhausted_flag,
            "layer4_stop_overshoot_orders": layer4_stop_overshoot_orders,
            "max_match_level": max_match_level,
            "pi_nnz": pi_nnz,
            "probability_sum": probability_sum,
        }
    )
    summary: dict[str, int | float] = {
        "origin_index": int(origin_index),
        "rows": int(time_count),
        "available_rows": int(available_flag.sum()),
        "insufficient_rows": int(insufficient_sample_flag.sum()),
        "layer4_used_rows": int(layer4_used_flag.sum()),
        "layer4_exhausted_rows": int(layer4_candidates_exhausted_flag.sum()),
        "pi_nnz": int(pi_matrix.nnz),
        "maximum_history_order_count": int(total_history_count.max()),
        "maximum_stop_overshoot_orders": int(layer4_stop_overshoot_orders.max()),
    }
    return pi_matrix, diagnostics, summary


def build_and_write_origin(origin_index: int) -> dict[str, int | float]:
    """工作进程入口：构建并原子式写出单个出发区域。"""

    assert _COUNT_MATRIX is not None
    assert _ROW_TOTAL is not None

    start = origin_index * cfg.EXPECTED_TIME_COUNT
    end = start + cfg.EXPECTED_TIME_COUNT
    origin_counts = _COUNT_MATRIX[start:end].tocsr()
    origin_totals = np.asarray(_ROW_TOTAL[start:end], dtype=np.int64)
    pi_matrix, diagnostics, summary = build_origin_distribution(
        origin_index,
        origin_counts,
        origin_totals,
    )

    pi_output = cfg.pi_part_file(origin_index)
    diagnostic_output = cfg.diagnostic_part_file(origin_index)
    pi_temporary = pi_output.with_suffix(".tmp.npz")
    diagnostic_temporary = diagnostic_output.with_suffix(".tmp.parquet")
    for temporary in (pi_temporary, diagnostic_temporary):
        if temporary.exists():
            temporary.unlink()

    sparse.save_npz(pi_temporary, pi_matrix, compressed=True)
    diagnostics.to_parquet(
        diagnostic_temporary,
        index=False,
        compression=cfg.PARQUET_COMPRESSION,
    )
    pi_temporary.replace(pi_output)
    diagnostic_temporary.replace(diagnostic_output)
    return summary


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.start_origin < 1:
        raise ValueError("workers 和 start-origin 必须大于等于 1。")
    end_origin = args.end_origin or cfg.EXPECTED_REGION_COUNT
    if end_origin < args.start_origin or end_origin > cfg.EXPECTED_REGION_COUNT:
        raise ValueError(
            f"end-origin 必须位于 [{args.start_origin}, "
            f"{cfg.EXPECTED_REGION_COUNT}] 之间。"
        )
    if args.resume and args.overwrite:
        raise ValueError("--resume 与 --overwrite 不能同时使用。")

    cfg.ensure_output_directories()
    load_prepared_data()

    requested = list(range(args.start_origin - 1, end_origin))
    pending: list[int] = []
    skipped = 0
    for origin_index in requested:
        pi_output = cfg.pi_part_file(origin_index)
        diagnostic_output = cfg.diagnostic_part_file(origin_index)
        existing_count = int(pi_output.exists()) + int(diagnostic_output.exists())
        if existing_count == 2 and args.resume:
            skipped += 1
            continue
        if existing_count and args.overwrite:
            if pi_output.exists():
                pi_output.unlink()
            if diagnostic_output.exists():
                diagnostic_output.unlink()
        elif existing_count:
            raise FileExistsError(
                f"区域 {origin_index} 输出已存在或不完整："
                f"{pi_output}, {diagnostic_output}。"
                "请使用 --resume 或 --overwrite。"
            )
        pending.append(origin_index)

    summaries: list[dict[str, int | float]] = []
    if args.workers == 1:
        for ordinal, origin_index in enumerate(pending, start=1):
            summary = build_and_write_origin(origin_index)
            summaries.append(summary)
            print(
                f"完成区域 {origin_index + 1}/{cfg.EXPECTED_REGION_COUNT}；"
                f"本次 {ordinal}/{len(pending)}；nnz={summary['pi_nnz']:,}"
            )
    elif pending:
        # 明确使用 fork：数百 MB 的 CSR 数据页只读共享，不在每个
        # 工作进程中重复加载。
        with mp.get_context("fork").Pool(processes=args.workers) as pool:
            for ordinal, summary in enumerate(
                pool.imap_unordered(build_and_write_origin, pending),
                start=1,
            ):
                summaries.append(summary)
                print(
                    f"完成区域 {int(summary['origin_index']) + 1}/"
                    f"{cfg.EXPECTED_REGION_COUNT}；本次 {ordinal}/{len(pending)}；"
                    f"nnz={int(summary['pi_nnz']):,}"
                )

    build_diagnostics = {
        "requested_origin_count": int(len(requested)),
        "processed_origin_count_this_run": int(len(summaries)),
        "skipped_origin_count": int(skipped),
        "workers": int(args.workers),
        "n_min": int(cfg.N_MIN),
        "n_stop": int(cfg.N_STOP),
        "layer_weights": {str(key): float(value) for key, value in cfg.LAYER_WEIGHTS.items()},
        "rows_written_this_run": int(sum(int(x["rows"]) for x in summaries)),
        "pi_nnz_written_this_run": int(sum(int(x["pi_nnz"]) for x in summaries)),
        "available_rows_this_run": int(
            sum(int(x["available_rows"]) for x in summaries)
        ),
        "insufficient_rows_this_run": int(
            sum(int(x["insufficient_rows"]) for x in summaries)
        ),
        "layer4_used_rows_this_run": int(
            sum(int(x["layer4_used_rows"]) for x in summaries)
        ),
        "maximum_stop_overshoot_orders_this_run": int(
            max(
                (int(x["maximum_stop_overshoot_orders"]) for x in summaries),
                default=0,
            )
        ),
        "pi_parts_directory": str(cfg.PI_PARTS_DIR),
        "diagnostic_parts_directory": str(cfg.DIAGNOSTIC_PARTS_DIR),
    }
    cfg.BUILD_DIAGNOSTICS_FILE.write_text(
        json.dumps(build_diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(build_diagnostics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
