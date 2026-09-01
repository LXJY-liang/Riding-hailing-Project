# -*- coding: utf-8 -*-
"""滚动收集每个 (i,j,t) 的历史接受价格代理样本。

本脚本只完成四层样本选择、两个分布的 Silverman 带宽及混合权重准备，
不计算 KDE 分位数。
所有字符串清洗、区域映射、历史排序和大分组索引均已由第 4 个脚本完成。

第 1—3 层组成“相同 OD”分布，保留 1.0/0.9/0.8 的层内权重；
第 4A、4B 层合并为“同时间状态+相似实际行程里程”分布。
第 4 层内部按里程相似度加权；两个 KDE 的最终混合权重由有效
样本量和相似度共同决定，并限制第 4 层的最大整体权重。

在 PyCharm 终端中使用以下命令以 4 个进程断点续算：

    "/Users/liangxue/Downloads/anaconda3/bin/python3" \
    "/Users/liangxue/Documents/8、Python文件-网约车数据/Riding hailing_Project/3 随机变量分位数预测/5 p^eijt分布构建/5 滚动收集p^eijt历史样本.py" \
    --workers 4 \
    --resume
"""

from __future__ import annotations

import os

# 多进程时每个进程内部只使用一个数学库线程，避免进程数乘以 BLAS
# 线程数造成严重过度竞争。必须在导入 numpy/scipy 前设置。
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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def load_config():
    path = Path(__file__).resolve().parent / "1 p^eijt分布构建配置.py"
    spec = importlib.util.spec_from_file_location("pe_distribution_config", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


cfg = load_config()


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    values = values[order]
    weights = weights[order]
    total = float(weights.sum())
    positions = (np.cumsum(weights) - 0.5 * weights) / total
    return np.interp(
        probabilities,
        positions,
        values,
        left=values[0],
        right=values[-1],
    )


def silverman_bandwidth(prices: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    weights = np.asarray(weights, dtype=np.float64)
    prices = np.asarray(prices, dtype=np.float64)
    weight_sum = float(weights.sum())
    square_sum = float(np.square(weights).sum())
    effective_n = weight_sum * weight_sum / square_sum if square_sum > 0 else 0.0
    if len(prices) <= 1 or effective_n <= 1.0:
        return float(cfg.MIN_BANDWIDTH_USD), float(max(effective_n, 1.0))

    mean = float(np.average(prices, weights=weights))
    variance = float(np.average(np.square(prices - mean), weights=weights))
    sigma = float(np.sqrt(max(variance, 0.0)))
    q25, q75 = weighted_quantile(
        prices,
        weights,
        np.asarray([0.25, 0.75], dtype=np.float64),
    )
    robust_scale = min(sigma, float(q75 - q25) / 1.34)
    raw = 0.9 * robust_scale * max(effective_n, 1.0) ** (-0.2)
    if not np.isfinite(raw):
        raw = 0.0
    return max(float(raw), float(cfg.MIN_BANDWIDTH_USD)), float(effective_n)


@dataclass
class CandidateSet:
    indices: list[int]
    levels: list[int]
    sublevels: list[str]
    similarities: list[float]
    selected: set[int]
    holiday_relaxed: bool = False
    target_reference_trip_miles: float = np.nan
    target_distance_sample_count: int = 0
    distance_reference_unavailable: bool = False

    @classmethod
    def empty(cls) -> "CandidateSet":
        return cls([], [], [], [], set())

    def add(
        self,
        candidates: np.ndarray,
        level: int,
        completion_ns: np.ndarray,
        sublevel: str,
        similarities: np.ndarray | None = None,
    ) -> None:
        if len(self.indices) >= cfg.N_MAX or len(candidates) == 0:
            return
        # 候选组内按完成时间从新到旧，保证超过 N_max 时保留最新记录。
        candidates = np.asarray(candidates, dtype=np.int64)
        if similarities is None:
            similarities = np.ones(len(candidates), dtype=np.float64)
        similarities = np.asarray(similarities, dtype=np.float64)
        if len(similarities) != len(candidates):
            raise ValueError("候选样本与相似度长度不一致。")
        order = np.argsort(completion_ns[candidates], kind="stable")[::-1]
        for position in order:
            raw_index = candidates[position]
            index = int(raw_index)
            if index in self.selected:
                continue
            self.selected.add(index)
            self.indices.append(index)
            self.levels.append(level)
            self.sublevels.append(sublevel)
            self.similarities.append(float(similarities[position]))
            if len(self.indices) >= cfg.N_MAX:
                break

    @property
    def enough(self) -> bool:
        return len(self.indices) >= cfg.N_MIN


class RollingSampleCollector:
    """只读内存映射索引上的滚动样本收集器。"""

    def __init__(self) -> None:
        mmap = "r"
        self.completion_ns = np.load(cfg.HISTORY_COMPLETION_NS_FILE, mmap_mode=mmap)
        self.i_index = np.load(cfg.HISTORY_I_FILE, mmap_mode=mmap)
        self.j_index = np.load(cfg.HISTORY_J_FILE, mmap_mode=mmap)
        self.start_day = np.load(cfg.HISTORY_DAY_FILE, mmap_mode=mmap)
        self.slot = np.load(cfg.HISTORY_SLOT_FILE, mmap_mode=mmap)
        self.dow = np.load(cfg.HISTORY_DOW_FILE, mmap_mode=mmap)
        self.weekend_holiday = np.load(
            cfg.HISTORY_WEEKEND_HOLIDAY_FILE, mmap_mode=mmap
        )
        self.price = np.load(cfg.HISTORY_PRICE_FILE, mmap_mode=mmap)
        self.trip_miles = np.load(cfg.HISTORY_TRIP_MILES_FILE, mmap_mode=mmap)
        self.trip_seconds = np.load(cfg.HISTORY_TRIP_SECONDS_FILE, mmap_mode=mmap)
        self.od_order = np.load(cfg.OD_ORDER_FILE, mmap_mode=mmap)
        self.od_offsets = np.load(cfg.OD_OFFSETS_FILE, mmap_mode=mmap)
        self.strict_order = np.load(
            cfg.ORIGIN_TIME_STRICT_ORDER_FILE, mmap_mode=mmap
        )
        self.strict_offsets = np.load(
            cfg.ORIGIN_TIME_STRICT_OFFSETS_FILE, mmap_mode=mmap
        )
        self.relaxed_order = np.load(
            cfg.ORIGIN_TIME_RELAXED_ORDER_FILE, mmap_mode=mmap
        )
        self.relaxed_offsets = np.load(
            cfg.ORIGIN_TIME_RELAXED_OFFSETS_FILE, mmap_mode=mmap
        )
        self.neighbor_matrix = np.load(cfg.NEIGHBOR_MATRIX_FILE, mmap_mode=mmap)

        neighbors = pd.read_parquet(
            cfg.NORMALIZED_NEIGHBOR_FILE,
            columns=["region_index", "neighbor_region_index"],
        )
        self.neighbors = {
            int(region): group["neighbor_region_index"].to_numpy(dtype=np.int16)
            for region, group in neighbors.groupby("region_index", sort=False)
        }

    def _prefix(
        self,
        order: np.ndarray,
        offsets: np.ndarray,
        key: int,
        target_ns: int,
    ) -> np.ndarray:
        start = int(offsets[key])
        end = int(offsets[key + 1])
        group = order[start:end]
        if len(group) == 0:
            return group
        usable_end = int(
            np.searchsorted(self.completion_ns[group], target_ns, side="left")
        )
        return group[:usable_end]

    def _od_prefix(self, i: int, j: int, target_ns: int) -> np.ndarray:
        key = i * cfg.REGION_COUNT + j
        return self._prefix(self.od_order, self.od_offsets, key, target_ns)

    def _strict_origin_time_prefix(
        self,
        origin: int,
        slot: int,
        dow: int,
        weekend_holiday: int,
        target_ns: int,
    ) -> np.ndarray:
        key = (
            ((origin * cfg.SLOTS_PER_DAY + slot) * cfg.DOW_COUNT + dow)
            * cfg.WEEKEND_HOLIDAY_STATE_COUNT
            + weekend_holiday
        )
        return self._prefix(self.strict_order, self.strict_offsets, key, target_ns)

    def _relaxed_origin_time_prefix(
        self,
        origin: int,
        slot: int,
        dow: int,
        target_ns: int,
    ) -> np.ndarray:
        key = (origin * cfg.SLOTS_PER_DAY + slot) * cfg.DOW_COUNT + dow
        return self._prefix(self.relaxed_order, self.relaxed_offsets, key, target_ns)

    def _valid_route_mask(self, candidates: np.ndarray) -> np.ndarray:
        """过滤无效或明显异常行程，但不使用价格做筛选。"""

        miles = self.trip_miles[candidates].astype(np.float64)
        seconds = self.trip_seconds[candidates].astype(np.float64)
        speed_mph = np.divide(
            miles * 3600.0,
            seconds,
            out=np.full(len(candidates), np.nan, dtype=np.float64),
            where=seconds > 0,
        )
        return (
            np.isfinite(miles)
            & np.isfinite(seconds)
            & (miles > 0.0)
            & (seconds >= cfg.MIN_VALID_TRIP_SECONDS)
            & (speed_mph >= cfg.MIN_VALID_SPEED_MPH)
            & (speed_mph <= cfg.MAX_VALID_SPEED_MPH)
        )

    def _target_reference_trip_miles(
        self, same_od: np.ndarray
    ) -> tuple[float, int]:
        if len(same_od) == 0:
            return np.nan, 0
        valid = same_od[self._valid_route_mask(same_od)]
        count = int(len(valid))
        if count < cfg.MIN_TARGET_DISTANCE_HISTORY:
            return np.nan, count
        return float(np.median(self.trip_miles[valid])), count

    def _distance_similarity(
        self, candidates: np.ndarray, target_trip_miles: float
    ) -> np.ndarray:
        miles = self.trip_miles[candidates].astype(np.float64)
        log_ratio = np.abs(np.log(miles / float(target_trip_miles)))
        return np.exp(-log_ratio / cfg.DISTANCE_SIMILARITY_LOG_SCALE)

    def _distance_filter_4a(
        self,
        candidates: np.ndarray,
        target_j: int,
        target_trip_miles: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(candidates) == 0:
            return candidates, np.empty(0, dtype=np.float64)
        destinations = self.j_index[candidates]
        miles = self.trip_miles[candidates].astype(np.float64)
        ratio = miles / float(target_trip_miles)
        mask = (
            (destinations != target_j)
            & self._valid_route_mask(candidates)
            & (ratio >= cfg.TRIP_MILES_RATIO_MIN)
            & (ratio <= cfg.TRIP_MILES_RATIO_MAX)
        )
        selected = candidates[mask]
        return selected, self._distance_similarity(selected, target_trip_miles)

    def _distance_filter_4b(
        self,
        candidates: np.ndarray,
        target_j: int,
        target_trip_miles: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(candidates) == 0:
            return candidates, np.empty(0, dtype=np.float64)
        destinations = self.j_index[candidates]
        miles = self.trip_miles[candidates].astype(np.float64)
        ratio = miles / float(target_trip_miles)
        mask = (
            self.neighbor_matrix[target_j, destinations]
            & self._valid_route_mask(candidates)
            & (ratio >= cfg.TRIP_MILES_RATIO_MIN)
            & (ratio <= cfg.TRIP_MILES_RATIO_MAX)
        )
        selected = candidates[mask]
        return selected, self._distance_similarity(selected, target_trip_miles)

    def _neighbor_origin_candidates(
        self,
        origins: np.ndarray,
        slot: int,
        dow: int,
        weekend_holiday: int,
        target_ns: int,
        relaxed: bool,
    ) -> np.ndarray:
        if relaxed:
            parts = [
                self._relaxed_origin_time_prefix(int(origin), slot, dow, target_ns)
                for origin in origins
            ]
        else:
            parts = [
                self._strict_origin_time_prefix(
                    int(origin), slot, dow, weekend_holiday, target_ns
                )
                for origin in origins
            ]
        parts = [part for part in parts if len(part)]
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.int32)

    def select(
        self,
        target_i: int,
        target_j: int,
        target_ns: int,
        target_day: int,
        target_slot: int,
        target_dow: int,
        target_is_holiday: bool,
        target_weekend_holiday: int,
    ) -> CandidateSet:
        selected = CandidateSet.empty()
        same_od = self._od_prefix(target_i, target_j, target_ns)

        strict_mask = (
            (self.slot[same_od] == target_slot)
            & (self.dow[same_od] == target_dow)
            & (self.weekend_holiday[same_od] == target_weekend_holiday)
        )
        layer1 = same_od[strict_mask]
        if (
            len(layer1) == 0
            and target_is_holiday
            and cfg.ALLOW_HOLIDAY_RELAXATION
        ):
            relaxed_mask = (
                (self.slot[same_od] == target_slot)
                & (self.dow[same_od] == target_dow)
            )
            layer1 = same_od[relaxed_mask]
            selected.holiday_relaxed = len(layer1) > 0
        selected.add(layer1, 1, self.completion_ns, "1")
        if selected.enough:
            return selected

        layer2 = same_od[self.slot[same_od] == target_slot]
        selected.add(layer2, 2, self.completion_ns, "2")
        if selected.enough:
            return selected

        if len(same_od):
            # 把“日期 + 日内时段”转换成连续的 15 分钟时段编号。
            # 不能只计算 24 小时循环意义下的日内时段差。例如，目标为
            # 6 月 14 日 23:30 时，前 1 天的中心时刻是 6 月 13 日 23:30，
            # 正确的 ±1 小时窗口是 6 月 13 日 22:30 至 6 月 14 日 00:30；
            # 6 月 13 日 00:00/00:30 不在该窗口内。
            history_absolute_slot = (
                self.start_day[same_od].astype(np.int32) * cfg.SLOTS_PER_DAY
                + self.slot[same_od].astype(np.int32)
            )
            target_absolute_slot = (
                int(target_day) * cfg.SLOTS_PER_DAY + int(target_slot)
            )

            # 当天部分仍只使用目标时刻之前 1 小时的记录。
            previous_hour = (
                (self.start_day[same_od] == target_day)
                & (target_absolute_slot - history_absolute_slot >= 1)
                & (
                    target_absolute_slot - history_absolute_slot
                    <= cfg.TIME_NEIGHBOR_SLOTS
                )
            )

            # 前 1 天、前 2 天、前 7 天分别以对应的绝对时间点为中心，
            # 取前后 1 小时。窗口允许自然跨越午夜，但不会在滞后日期内
            # 把 00:xx 与 23:xx 简单视为同一天的相邻时段。
            lagged_window = np.zeros(len(same_od), dtype=bool)
            for lag_days in cfg.SPECIAL_DAY_LAGS:
                lagged_center = (
                    target_absolute_slot - int(lag_days) * cfg.SLOTS_PER_DAY
                )
                lagged_window |= (
                    np.abs(history_absolute_slot - lagged_center)
                    <= cfg.TIME_NEIGHBOR_SLOTS
                )

            layer3 = same_od[previous_hour | lagged_window]
        else:
            layer3 = same_od
        selected.add(layer3, 3, self.completion_ns, "3")
        if selected.enough:
            return selected

        target_trip_miles, target_distance_count = (
            self._target_reference_trip_miles(same_od)
        )
        selected.target_reference_trip_miles = target_trip_miles
        selected.target_distance_sample_count = target_distance_count
        if not np.isfinite(target_trip_miles):
            # 实际里程参考不可靠时不强行使用质心距离回退。
            # 已收集的第 1—3 层仍然保留。
            selected.distance_reference_unavailable = True
            return selected

        strict_4a = self._strict_origin_time_prefix(
            target_i,
            target_slot,
            target_dow,
            target_weekend_holiday,
            target_ns,
        )
        layer4a, similarity4a = self._distance_filter_4a(
            strict_4a, target_j, target_trip_miles
        )
        if (
            len(layer4a) == 0
            and target_is_holiday
            and cfg.ALLOW_HOLIDAY_RELAXATION
        ):
            relaxed_4a = self._relaxed_origin_time_prefix(
                target_i, target_slot, target_dow, target_ns
            )
            layer4a, similarity4a = self._distance_filter_4a(
                relaxed_4a, target_j, target_trip_miles
            )
            selected.holiday_relaxed |= len(layer4a) > 0
        selected.add(
            layer4a, 4, self.completion_ns, "4A", similarities=similarity4a
        )
        if selected.enough:
            return selected

        origin_neighbors = self.neighbors.get(
            target_i, np.empty(0, dtype=np.int16)
        )
        strict_4b = self._neighbor_origin_candidates(
            origin_neighbors,
            target_slot,
            target_dow,
            target_weekend_holiday,
            target_ns,
            relaxed=False,
        )
        layer4b, similarity4b = self._distance_filter_4b(
            strict_4b, target_j, target_trip_miles
        )
        if (
            len(layer4b) == 0
            and target_is_holiday
            and cfg.ALLOW_HOLIDAY_RELAXATION
        ):
            relaxed_4b = self._neighbor_origin_candidates(
                origin_neighbors,
                target_slot,
                target_dow,
                target_weekend_holiday,
                target_ns,
                relaxed=True,
            )
            layer4b, similarity4b = self._distance_filter_4b(
                relaxed_4b, target_j, target_trip_miles
            )
            selected.holiday_relaxed |= len(layer4b) > 0
        selected.add(
            layer4b, 4, self.completion_ns, "4B", similarities=similarity4b
        )
        return selected

    def collect(self, record: tuple) -> dict[str, object]:
        (
            target_id,
            i,
            j,
            target_i,
            target_j,
            t,
            target_ns,
            target_day,
            target_slot,
            target_dow,
            target_is_holiday,
            target_weekend_holiday,
            demand,
        ) = record
        base = {
            "target_id": int(target_id),
            "i": str(i),
            "j": str(j),
            "t": pd.Timestamp(t),
            "D_ijt_observed": float(demand),
        }
        candidates = self.select(
            int(target_i),
            int(target_j),
            int(target_ns),
            int(target_day),
            int(target_slot),
            int(target_dow),
            bool(target_is_holiday),
            int(target_weekend_holiday),
        )
        if not candidates.indices:
            return {
                **base,
                "available_flag": 0,
                "invalid_region_flag": 0,
                "insufficient_sample_flag": 1,
                "holiday_relaxed_flag": int(candidates.holiday_relaxed),
                "distance_reference_unavailable_flag": int(
                    candidates.distance_reference_unavailable
                ),
                "unreliable_similar_only_flag": 0,
                "sample_count": 0,
                "same_od_sample_count": 0,
                "similar_od_sample_count": 0,
                "same_od_effective_sample_size": 0.0,
                "similar_od_effective_sample_size": 0.0,
                "same_od_bandwidth_usd": np.nan,
                "similar_od_bandwidth_usd": np.nan,
                "same_od_mixture_weight": np.nan,
                "similar_od_mixture_weight": np.nan,
                "similar_od_mean_similarity": np.nan,
                "max_final_sample_weight": np.nan,
                "max_match_level": 0,
                "layer1_count": 0,
                "layer2_count": 0,
                "layer3_count": 0,
                "layer4_count": 0,
                "layer4a_count": 0,
                "layer4b_count": 0,
                "target_reference_trip_miles": float(
                    candidates.target_reference_trip_miles
                ),
                "target_distance_sample_count": int(
                    candidates.target_distance_sample_count
                ),
                "sample_prices": [],
                "sample_levels": [],
                "sample_trip_miles": [],
                "sample_similarity_weights": [],
            }

        indices = np.asarray(candidates.indices, dtype=np.int64)
        levels = np.asarray(candidates.levels, dtype=np.int8)
        similarities = np.asarray(candidates.similarities, dtype=np.float64)
        prices = self.price[indices].astype(np.float64)
        sample_trip_miles = self.trip_miles[indices].astype(np.float64)
        weight_lookup = np.asarray(
            [0.0, cfg.LAYER_WEIGHTS[1], cfg.LAYER_WEIGHTS[2],
             cfg.LAYER_WEIGHTS[3], cfg.LAYER_WEIGHTS[4]],
            dtype=np.float64,
        )
        same_od_mask = levels <= 3
        similar_od_mask = levels == 4
        same_od_count = int(same_od_mask.sum())
        similar_od_count = int(similar_od_mask.sum())

        if same_od_count:
            same_od_bandwidth, same_od_effective_n = silverman_bandwidth(
                prices[same_od_mask], weight_lookup[levels[same_od_mask]]
            )
        else:
            same_od_bandwidth, same_od_effective_n = np.nan, 0.0

        if similar_od_count:
            similar_weights = similarities[similar_od_mask]
            similar_od_bandwidth, similar_od_effective_n = silverman_bandwidth(
                prices[similar_od_mask],
                similar_weights,
            )
            similar_mean_similarity = float(np.mean(similar_weights))
        else:
            similar_od_bandwidth, similar_od_effective_n = np.nan, 0.0
            similar_mean_similarity = np.nan

        same_od_mixture_weight, similar_od_mixture_weight = (
            cfg.component_mixture_weights(
                same_od_effective_n,
                similar_od_effective_n,
                0.0 if not similar_od_count else similar_mean_similarity,
                same_od_count,
                similar_od_count,
            )
        )
        unreliable_similar_only = int(
            same_od_count == 0
            and 0 < similar_od_count < cfg.SIMILAR_ONLY_MIN_SAMPLES
        )
        available = int(
            np.isfinite(same_od_mixture_weight)
            and np.isfinite(similar_od_mixture_weight)
        )

        max_final_sample_weight = np.nan
        if available:
            component_weights = np.zeros(len(indices), dtype=np.float64)
            if same_od_count:
                raw_same = weight_lookup[levels[same_od_mask]]
                component_weights[same_od_mask] = (
                    same_od_mixture_weight * raw_same / raw_same.sum()
                )
            if similar_od_count:
                raw_similar = similarities[similar_od_mask]
                component_weights[similar_od_mask] = (
                    similar_od_mixture_weight
                    * raw_similar
                    / raw_similar.sum()
                )
            max_final_sample_weight = float(component_weights.max())

        sublevels = np.asarray(candidates.sublevels, dtype="U2")
        return {
            **base,
            "available_flag": available,
            "invalid_region_flag": 0,
            "insufficient_sample_flag": int(len(indices) < cfg.N_MIN),
            "holiday_relaxed_flag": int(candidates.holiday_relaxed),
            "distance_reference_unavailable_flag": int(
                candidates.distance_reference_unavailable
            ),
            "unreliable_similar_only_flag": unreliable_similar_only,
            "sample_count": int(len(indices)),
            "same_od_sample_count": same_od_count,
            "similar_od_sample_count": similar_od_count,
            "same_od_effective_sample_size": float(same_od_effective_n),
            "similar_od_effective_sample_size": float(similar_od_effective_n),
            "same_od_bandwidth_usd": float(same_od_bandwidth),
            "similar_od_bandwidth_usd": float(similar_od_bandwidth),
            "same_od_mixture_weight": float(same_od_mixture_weight),
            "similar_od_mixture_weight": float(similar_od_mixture_weight),
            "similar_od_mean_similarity": float(similar_mean_similarity),
            "max_final_sample_weight": float(max_final_sample_weight),
            "max_match_level": int(levels.max()),
            "layer1_count": int(np.sum(levels == 1)),
            "layer2_count": int(np.sum(levels == 2)),
            "layer3_count": int(np.sum(levels == 3)),
            "layer4_count": int(np.sum(levels == 4)),
            "layer4a_count": int(np.sum(sublevels == "4A")),
            "layer4b_count": int(np.sum(sublevels == "4B")),
            "target_reference_trip_miles": float(
                candidates.target_reference_trip_miles
            ),
            "target_distance_sample_count": int(
                candidates.target_distance_sample_count
            ),
            "sample_prices": prices.astype(np.float32).tolist(),
            "sample_levels": levels.tolist(),
            "sample_trip_miles": sample_trip_miles.astype(np.float32).tolist(),
            "sample_similarity_weights": similarities.astype(np.float32).tolist(),
        }


_WORKER_COLLECTOR: RollingSampleCollector | None = None


def collect_chunk(records: list[tuple]) -> list[dict[str, object]]:
    if _WORKER_COLLECTOR is None:
        raise RuntimeError("工作进程尚未初始化样本收集器。")
    return [_WORKER_COLLECTOR.collect(record) for record in records]


def split_records(records: list[tuple], part_count: int) -> list[list[tuple]]:
    part_count = max(1, min(part_count, len(records)))
    bounds = np.linspace(0, len(records), part_count + 1, dtype=int)
    return [
        records[int(bounds[k]) : int(bounds[k + 1])]
        for k in range(part_count)
        if bounds[k] < bounds[k + 1]
    ]


def records_from_batch(batch: pa.RecordBatch) -> list[tuple]:
    columns = batch.to_pydict()
    names = batch.schema.names
    return list(zip(*(columns[name] for name in names)))


def write_sample_part(results: list[dict[str, object]], part_number: int) -> Path:
    data = pd.DataFrame(results)
    prices = pa.array(data.pop("sample_prices").tolist(), type=pa.list_(pa.float32()))
    levels = pa.array(data.pop("sample_levels").tolist(), type=pa.list_(pa.int8()))
    trip_miles = pa.array(
        data.pop("sample_trip_miles").tolist(), type=pa.list_(pa.float32())
    )
    similarity_weights = pa.array(
        data.pop("sample_similarity_weights").tolist(), type=pa.list_(pa.float32())
    )
    int_columns = [
        "available_flag", "invalid_region_flag", "insufficient_sample_flag",
        "holiday_relaxed_flag", "distance_reference_unavailable_flag",
        "unreliable_similar_only_flag", "sample_count", "same_od_sample_count",
        "similar_od_sample_count", "max_match_level",
        "layer1_count", "layer2_count", "layer3_count", "layer4_count",
        "layer4a_count", "layer4b_count", "target_distance_sample_count",
    ]
    for column in int_columns:
        data[column] = data[column].astype("int16")
    float_columns = [
        "D_ijt_observed", "same_od_effective_sample_size",
        "similar_od_effective_sample_size", "same_od_bandwidth_usd",
        "similar_od_bandwidth_usd", "same_od_mixture_weight",
        "similar_od_mixture_weight", "similar_od_mean_similarity",
        "max_final_sample_weight", "target_reference_trip_miles",
    ]
    for column in float_columns:
        data[column] = data[column].astype("float32")
    table = pa.Table.from_pandas(data, preserve_index=False)
    table = table.append_column("sample_prices", prices)
    table = table.append_column("sample_levels", levels)
    table = table.append_column("sample_trip_miles", trip_miles)
    table = table.append_column("sample_similarity_weights", similarity_weights)
    output = cfg.SAMPLE_DATASET / f"part-{part_number:05d}.parquet"
    pq.write_table(table, output, compression=cfg.PARQUET_COMPRESSION)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="滚动收集 p^e_ijt 历史代理样本。")
    parser.add_argument("--workers", type=int, default=cfg.DEFAULT_WORKERS)
    parser.add_argument("--start-batch", type=int, default=1)
    parser.add_argument("--end-batch", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_inputs() -> None:
    required = [
        cfg.TARGET_PREPARED_FILE,
        cfg.HISTORY_COMPLETION_NS_FILE, cfg.HISTORY_I_FILE, cfg.HISTORY_J_FILE,
        cfg.HISTORY_DAY_FILE, cfg.HISTORY_SLOT_FILE, cfg.HISTORY_DOW_FILE,
        cfg.HISTORY_WEEKEND_HOLIDAY_FILE, cfg.HISTORY_PRICE_FILE,
        cfg.HISTORY_TRIP_MILES_FILE, cfg.HISTORY_TRIP_SECONDS_FILE,
        cfg.OD_ORDER_FILE, cfg.OD_OFFSETS_FILE,
        cfg.ORIGIN_TIME_STRICT_ORDER_FILE, cfg.ORIGIN_TIME_STRICT_OFFSETS_FILE,
        cfg.ORIGIN_TIME_RELAXED_ORDER_FILE, cfg.ORIGIN_TIME_RELAXED_OFFSETS_FILE,
        cfg.NEIGHBOR_MATRIX_FILE,
        cfg.NORMALIZED_NEIGHBOR_FILE,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少第 3/4 步输出：\n" + "\n".join(missing))


def main() -> None:
    global _WORKER_COLLECTOR
    args = parse_args()
    if args.workers < 1 or args.start_batch < 1:
        raise ValueError("workers 和 start-batch 必须大于等于 1。")
    if args.end_batch is not None and args.end_batch < args.start_batch:
        raise ValueError("end-batch 不得小于 start-batch。")
    if args.resume and args.overwrite:
        raise ValueError("--resume 与 --overwrite 不能同时使用。")

    cfg.ensure_output_directories()
    validate_inputs()
    if cfg.SAMPLE_DATASET.exists():
        if args.overwrite:
            shutil.rmtree(cfg.SAMPLE_DATASET)
        elif not args.resume:
            raise FileExistsError(
                f"样本输出已存在：{cfg.SAMPLE_DATASET}；请使用 --resume 或 --overwrite。"
            )
    cfg.SAMPLE_DATASET.mkdir(parents=True, exist_ok=True)

    print("正在加载内存映射数值索引……")
    _WORKER_COLLECTOR = RollingSampleCollector()
    source = pq.ParquetFile(cfg.TARGET_PREPARED_FILE)
    required_columns = [
        "target_id", "i", "j", "i_index", "j_index", "t", "target_ns",
        "target_day", "target_slot", "target_dow", "target_is_holiday",
        "target_weekend_or_holiday", "D_ijt_observed",
    ]

    pool = None
    if args.workers > 1:
        pool = mp.get_context("fork").Pool(processes=args.workers)

    counters = {
        "processed_rows_this_run": 0,
        "skipped_existing_batches": 0,
        "available_rows_this_run": 0,
        "insufficient_rows_this_run": 0,
    }
    try:
        for batch_number, batch in enumerate(
            source.iter_batches(
                batch_size=cfg.TARGET_BATCH_SIZE,
                columns=required_columns,
            ),
            start=1,
        ):
            if batch_number < args.start_batch:
                continue
            if args.end_batch is not None and batch_number > args.end_batch:
                break
            output = cfg.SAMPLE_DATASET / f"part-{batch_number - 1:05d}.parquet"
            if output.exists() and args.resume:
                counters["skipped_existing_batches"] += 1
                continue
            if output.exists():
                raise FileExistsError(f"批次输出已存在：{output}")

            records = records_from_batch(batch)
            if pool is None:
                results = collect_chunk(records)
            else:
                chunks = split_records(records, args.workers * 2)
                results = [row for part in pool.map(collect_chunk, chunks) for row in part]
            write_sample_part(results, batch_number - 1)
            counters["processed_rows_this_run"] += len(results)
            counters["available_rows_this_run"] += sum(
                int(row["available_flag"]) for row in results
            )
            counters["insufficient_rows_this_run"] += sum(
                int(row["insufficient_sample_flag"]) for row in results
            )
            print(
                f"batch {batch_number}: 写出 {len(results):,} 行；"
                f"本次累计 {counters['processed_rows_this_run']:,} 行"
            )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    diagnostics = {
        **counters,
        "workers": int(args.workers),
        "sample_dataset": str(cfg.SAMPLE_DATASET),
        "n_min": int(cfg.N_MIN),
        "n_max": int(cfg.N_MAX),
        "kde_components": "same_od_layers_1_2_3 + similar_od_layer_4",
        "mixture_method": {
            "basis": "effective_sample_size_and_trip_distance_similarity",
            "similar_reliability_discount": float(
                cfg.SIMILAR_OD_RELIABILITY_DISCOUNT
            ),
            "similar_only_min_samples": int(cfg.SIMILAR_ONLY_MIN_SAMPLES),
            "similar_weight_cap_same_1_4": float(
                cfg.SIMILAR_WEIGHT_CAP_SPARSE_SAME
            ),
            "similar_weight_cap_same_5_plus": float(
                cfg.SIMILAR_WEIGHT_CAP_ESTABLISHED_SAME
            ),
        },
    }
    cfg.SAMPLE_DIAGNOSTICS_FILE.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
