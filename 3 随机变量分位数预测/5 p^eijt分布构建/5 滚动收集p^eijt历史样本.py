# -*- coding: utf-8 -*-
"""滚动收集每个 (i,j,t) 的历史接受价格代理样本。

本脚本只完成分层样本选择、两个分布的 Silverman 带宽及混合权重准备，
不计算 KDE 分位数。
所有字符串清洗、区域映射、历史排序和大分组索引均已由第 4 个脚本完成。

第 1—3 层组成“相同 OD”分布，保留 1.0/0.9/0.8 的层内权重；
第 4A、4B、4C 层以及只对不可用目标启用的第 5A—5E 规则，合并为
“相似时间状态+相似实际行程里程”分布。相似 OD 成分内部按里程
相似度加权；两个 KDE 的最终混合权重由有效
样本量和相似度共同决定，并限制第 4 层的最大整体权重。

超短距离规则从第1层开始统一生效。第5层中的5E不是在5D失败后
才运行的独立回退，而是5A—5D每次检索时并行生效的绝对距离容差
分支：只有未通过比例距离、但通过超短距离绝对容差的订单才标为5E。

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
from functools import lru_cache
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
    target_distance_reference_source: int = 0
    distance_reference_unavailable: bool = False
    self_route_rejected_count: int = 0
    target_distance_rejected_count: int = 0
    self_short_absolute_accepted_count: int = 0
    target_short_absolute_accepted_count: int = 0
    fallback_attempted: bool = False
    fallback_final_level: int = 0

    @classmethod
    def empty(cls) -> "CandidateSet":
        return cls([], [], [], [], set())

    def add(
        self,
        candidates: np.ndarray,
        level: int,
        completion_ns: np.ndarray,
        sublevel: str | np.ndarray,
        similarities: np.ndarray | None = None,
        preserve_order: bool = False,
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
        if isinstance(sublevel, str):
            sublevels = np.full(len(candidates), sublevel, dtype="U2")
        else:
            sublevels = np.asarray(sublevel, dtype="U2")
            if len(sublevels) != len(candidates):
                raise ValueError("候选样本与子层级长度不一致。")
        if preserve_order:
            order = np.arange(len(candidates), dtype=np.int64)
        else:
            order = np.argsort(completion_ns[candidates], kind="stable")[::-1]
        for position in order:
            raw_index = candidates[position]
            index = int(raw_index)
            if index in self.selected:
                continue
            self.selected.add(index)
            self.indices.append(index)
            self.levels.append(level)
            self.sublevels.append(str(sublevels[position]))
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
        self.date_type = np.load(cfg.HISTORY_DATE_TYPE_FILE, mmap_mode=mmap)
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
        self.origin_dt_order = np.load(
            cfg.ORIGIN_DATE_TYPE_TIME_ORDER_FILE, mmap_mode=mmap
        )
        self.origin_dt_offsets = np.load(
            cfg.ORIGIN_DATE_TYPE_TIME_OFFSETS_FILE, mmap_mode=mmap
        )
        self.global_dt_order = np.load(
            cfg.GLOBAL_DATE_TYPE_TIME_ORDER_FILE, mmap_mode=mmap
        )
        self.global_dt_offsets = np.load(
            cfg.GLOBAL_DATE_TYPE_TIME_OFFSETS_FILE, mmap_mode=mmap
        )
        self.neighbor_matrix = np.load(cfg.NEIGHBOR_MATRIX_FILE, mmap_mode=mmap)

        road = pd.read_parquet(
            cfg.OD_ROAD_DISTANCE_FILE,
            columns=["i_index", "j_index", "d_map_miles"],
        )
        self.road_distance = np.full(
            (cfg.REGION_COUNT, cfg.REGION_COUNT), np.nan, dtype=np.float32
        )
        self.road_distance[
            road["i_index"].to_numpy(dtype=np.int64),
            road["j_index"].to_numpy(dtype=np.int64),
        ] = road["d_map_miles"].to_numpy(dtype=np.float32)
        if np.any(~np.isfinite(self.road_distance) | (self.road_distance <= 0.0)):
            raise ValueError("静态道路距离矩阵存在缺失或非正值。")

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

    @lru_cache(maxsize=500_000)
    def _distance_reference(
        self,
        i: int,
        j: int,
        target_ns: int,
    ) -> tuple[float, int, int]:
        """返回目标时刻可用的 OD 距离基准、样本数和来源代码。

        来源代码：1=静态道路距离，2=历史中位数向道路距离收缩，
        3=样本充分的滚动历史中位数。任何历史统计都严格限制为
        completion_time < target_ns。
        """

        same_od = self._od_prefix(i, j, target_ns)
        valid = same_od[self._valid_route_mask(same_od)]
        count = int(len(valid))
        road = float(self.road_distance[i, j])
        if count >= cfg.DISTANCE_DIRECT_HISTORY_THRESHOLD:
            return float(np.median(self.trip_miles[valid])), count, 3
        if count >= cfg.MIN_DISTANCE_HISTORY_FOR_SHRINKAGE:
            historical = float(np.median(self.trip_miles[valid]))
            weight = count / (count + float(cfg.DISTANCE_SHRINKAGE_KAPPA))
            return weight * historical + (1.0 - weight) * road, count, 2
        return road, count, 1

    @staticmethod
    def _hybrid_distance_mask(
        miles: np.ndarray,
        reference: float,
        ratio_min: float,
        ratio_max: float,
        short_reference_max: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """返回联合距离判断及其中仅由绝对容差接受的记录。

        第一项为“比例条件或超短绝对条件”的最终接受掩码；第二项只
        标记比例条件失败、但超短绝对条件成功的记录，供5E和诊断计数。
        """

        miles = np.asarray(miles, dtype=np.float64)
        reference = float(reference)
        ratio = np.divide(
            miles,
            reference,
            out=np.full(len(miles), np.nan, dtype=np.float64),
            where=np.isfinite(reference) & (reference > 0.0),
        )
        relative = (
            np.isfinite(ratio)
            & (ratio >= float(ratio_min))
            & (ratio <= float(ratio_max))
        )
        absolute = (
            np.isfinite(reference)
            & (reference > 0.0)
            & (reference <= float(short_reference_max))
            & np.isfinite(miles)
            & (
                np.abs(miles - reference)
                <= float(cfg.SHORT_DISTANCE_ABSOLUTE_TOLERANCE_MILES)
            )
        )
        return relative | absolute, absolute & ~relative

    def _self_route_filter(
        self,
        candidates: np.ndarray,
        target_ns: int,
    ) -> tuple[np.ndarray, int, int]:
        """删除自身行程里程与其 OD 距离基准不一致的订单。"""

        candidates = np.asarray(candidates, dtype=np.int64)
        if len(candidates) == 0:
            return candidates, 0, 0
        physical = self._valid_route_mask(candidates)
        accepted = np.zeros(len(candidates), dtype=bool)
        absolute_only = np.zeros(len(candidates), dtype=bool)
        od_codes = (
            self.i_index[candidates].astype(np.int64) * cfg.REGION_COUNT
            + self.j_index[candidates].astype(np.int64)
        )
        for code in np.unique(
            od_codes
        ):
            i = int(code // cfg.REGION_COUNT)
            j = int(code % cfg.REGION_COUNT)
            reference, _, _ = self._distance_reference(i, j, target_ns)
            positions = od_codes == code
            distance_ok, absolute_ok = self._hybrid_distance_mask(
                self.trip_miles[candidates[positions]].astype(np.float64),
                reference,
                cfg.SELF_ROUTE_RATIO_MIN,
                cfg.SELF_ROUTE_RATIO_MAX,
                cfg.SHORT_CANDIDATE_REFERENCE_MAX_MILES,
            )
            accepted[positions] = physical[positions] & distance_ok
            absolute_only[positions] = physical[positions] & absolute_ok
        return (
            candidates[accepted],
            int((~accepted).sum()),
            int(absolute_only.sum()),
        )

    def _distance_similarity(
        self, candidates: np.ndarray, target_trip_miles: float
    ) -> np.ndarray:
        miles = self.trip_miles[candidates].astype(np.float64)
        log_ratio = np.abs(np.log(miles / float(target_trip_miles)))
        relative = np.exp(-log_ratio / cfg.DISTANCE_SIMILARITY_LOG_SCALE)
        if target_trip_miles <= cfg.SHORT_TARGET_REFERENCE_MAX_MILES:
            absolute = np.exp(
                -np.abs(miles - float(target_trip_miles))
                / float(cfg.SHORT_DISTANCE_ABSOLUTE_TOLERANCE_MILES)
            )
            return np.maximum(relative, absolute)
        return relative

    def _origin_date_type_time_prefix(
        self, origin: int, date_type: int, slot: int, target_ns: int
    ) -> np.ndarray:
        key = (origin * cfg.DATE_TYPE_COUNT + date_type) * cfg.SLOTS_PER_DAY + slot
        return self._prefix(
            self.origin_dt_order, self.origin_dt_offsets, key, target_ns
        )

    def _global_date_type_time_prefix(
        self, date_type: int, slot: int, target_ns: int
    ) -> np.ndarray:
        key = date_type * cfg.SLOTS_PER_DAY + slot
        return self._prefix(self.global_dt_order, self.global_dt_offsets, key, target_ns)

    def _time_window_candidates(
        self,
        origins: np.ndarray | None,
        date_type: int,
        target_slot: int,
        target_ns: int,
        *,
        window_slots: int | None = None,
        circular: bool = False,
        cap_raw: bool = True,
    ) -> np.ndarray:
        if window_slots is None:
            window_slots = cfg.LAYER4_TIME_NEIGHBOR_SLOTS
        if circular:
            # 第5层匹配的是不同历史日期的日内时间位置。此时23:45与
            # 00:00应视为相邻；completion_time<t 仍负责阻止未来泄露。
            slots = sorted(
                {
                    (int(target_slot) + offset) % cfg.SLOTS_PER_DAY
                    for offset in range(-int(window_slots), int(window_slots) + 1)
                }
            )
        else:
            # 第4层保留原口径，不把同一天23:xx与00:xx直接循环相连。
            low = max(0, target_slot - int(window_slots))
            high = min(cfg.SLOTS_PER_DAY - 1, target_slot + int(window_slots))
            slots = range(low, high + 1)
        parts: list[np.ndarray] = []
        if origins is None:
            for slot in slots:
                part = self._global_date_type_time_prefix(date_type, slot, target_ns)
                if len(part):
                    parts.append(part)
        else:
            for origin in origins:
                for slot in slots:
                    part = self._origin_date_type_time_prefix(
                        int(origin), date_type, slot, target_ns
                    )
                    if len(part):
                        parts.append(part)
        if not parts:
            return np.empty(0, dtype=np.int32)
        candidates = np.concatenate(parts)
        if cap_raw and len(candidates) > cfg.MAX_RAW_CANDIDATES_PER_SUBLAYER:
            order = np.argsort(self.completion_ns[candidates], kind="stable")
            candidates = candidates[order[-cfg.MAX_RAW_CANDIDATES_PER_SUBLAYER :]]
        return candidates

    def _all_date_type_time_candidates(
        self,
        target_slot: int,
        target_ns: int,
        window_slots: int,
    ) -> np.ndarray:
        """返回全部日期类型的循环日内时间候选，并去除重复订单。"""

        parts = [
            self._time_window_candidates(
                None,
                date_type,
                target_slot,
                target_ns,
                window_slots=window_slots,
                circular=True,
                cap_raw=False,
            )
            for date_type in range(cfg.DATE_TYPE_COUNT)
        ]
        parts = [part for part in parts if len(part)]
        if not parts:
            return np.empty(0, dtype=np.int32)
        candidates = np.unique(np.concatenate(parts))
        return candidates

    def _filter_similar_layer(
        self,
        candidates: np.ndarray,
        target_i: int,
        target_j: int,
        target_ns: int,
        target_trip_miles: float,
        sublevel: str,
        *,
        ratio_min: float | None = None,
        ratio_max: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int, int]:
        """执行相似OD层的空间口径、自身路线和目标距离检查。"""

        candidates = np.asarray(candidates, dtype=np.int64)
        if len(candidates) == 0:
            return (
                candidates, np.empty(0, dtype=np.float64),
                np.empty(0, dtype=bool), 0, 0, 0, 0,
            )
        origins = self.i_index[candidates]
        destinations = self.j_index[candidates]
        same_target_od = (origins == target_i) & (destinations == target_j)
        target_is_internal = target_i == target_j
        same_route_type = (origins == destinations) == target_is_internal
        if sublevel == "4A":
            spatial = (
                (origins == target_i)
                & (destinations != target_j)
                & same_route_type
            )
        elif sublevel == "4B":
            spatial = (
                self.neighbor_matrix[target_i, origins]
                & self.neighbor_matrix[target_j, destinations]
                & same_route_type
            )
        elif sublevel in {"4C", "5A", "5B", "5C", "5D"}:
            spatial = same_route_type
        else:
            raise ValueError(f"未知第4层子层级：{sublevel}")
        candidates = candidates[spatial & ~same_target_od]
        self_checked, self_rejected, self_absolute = self._self_route_filter(
            candidates, target_ns
        )
        if len(self_checked) == 0:
            return (
                self_checked, np.empty(0, dtype=np.float64),
                np.empty(0, dtype=bool), self_rejected, 0, self_absolute, 0,
            )
        if ratio_min is None:
            ratio_min = cfg.TRIP_MILES_RATIO_MIN
        if ratio_max is None:
            ratio_max = cfg.TRIP_MILES_RATIO_MAX
        target_mask, target_absolute_only = self._hybrid_distance_mask(
            self.trip_miles[self_checked].astype(np.float64),
            target_trip_miles,
            float(ratio_min),
            float(ratio_max),
            cfg.SHORT_TARGET_REFERENCE_MAX_MILES,
        )
        target_rejected = int((~target_mask).sum())
        selected = self_checked[target_mask]
        selected_absolute_only = target_absolute_only[target_mask]
        similarity = self._distance_similarity(selected, target_trip_miles)
        if sublevel in cfg.LAYER4_SUBLEVEL_RELIABILITY:
            similarity *= float(cfg.LAYER4_SUBLEVEL_RELIABILITY[sublevel])
        return (
            selected,
            similarity,
            selected_absolute_only,
            self_rejected,
            target_rejected,
            self_absolute,
            int(selected_absolute_only.sum()),
        )

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
        target_date_type: int,
    ) -> CandidateSet:
        selected = CandidateSet.empty()
        same_od = self._od_prefix(target_i, target_j, target_ns)
        target_trip_miles, target_distance_count, target_distance_source = (
            self._distance_reference(target_i, target_j, target_ns)
        )
        selected.target_reference_trip_miles = target_trip_miles
        selected.target_distance_sample_count = target_distance_count
        selected.target_distance_reference_source = target_distance_source

        strict_mask = (
            (self.slot[same_od] == target_slot)
            & (self.dow[same_od] == target_dow)
            & (self.weekend_holiday[same_od] == target_weekend_holiday)
        )
        layer1, rejected, absolute_accepted = self._self_route_filter(
            same_od[strict_mask], target_ns
        )
        selected.self_route_rejected_count += rejected
        selected.self_short_absolute_accepted_count += absolute_accepted
        if (
            len(layer1) == 0
            and target_is_holiday
            and cfg.ALLOW_HOLIDAY_RELAXATION
        ):
            relaxed_mask = (
                (self.slot[same_od] == target_slot)
                & (self.dow[same_od] == target_dow)
            )
            layer1, rejected, absolute_accepted = self._self_route_filter(
                same_od[relaxed_mask], target_ns
            )
            selected.self_route_rejected_count += rejected
            selected.self_short_absolute_accepted_count += absolute_accepted
            selected.holiday_relaxed = len(layer1) > 0
        selected.add(layer1, 1, self.completion_ns, "1")
        if selected.enough:
            return selected

        layer2, rejected, absolute_accepted = self._self_route_filter(
            same_od[self.slot[same_od] == target_slot], target_ns
        )
        selected.self_route_rejected_count += rejected
        selected.self_short_absolute_accepted_count += absolute_accepted
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
        layer3, rejected, absolute_accepted = self._self_route_filter(
            layer3, target_ns
        )
        selected.self_route_rejected_count += rejected
        selected.self_short_absolute_accepted_count += absolute_accepted
        selected.add(layer3, 3, self.completion_ns, "3")
        if selected.enough:
            return selected

        if not np.isfinite(target_trip_miles):
            selected.distance_reference_unavailable = True
            return selected

        layer4a_raw = self._time_window_candidates(
            np.asarray([target_i], dtype=np.int16),
            target_date_type,
            target_slot,
            target_ns,
        )
        (
            layer4a, similarity4a, absolute4a, self_rejected,
            target_rejected, self_absolute, target_absolute,
        ) = (
            self._filter_similar_layer(
                layer4a_raw,
                target_i,
                target_j,
                target_ns,
                target_trip_miles,
                "4A",
            )
        )
        selected.self_route_rejected_count += self_rejected
        selected.target_distance_rejected_count += target_rejected
        selected.self_short_absolute_accepted_count += self_absolute
        selected.target_short_absolute_accepted_count += target_absolute
        if (
            len(layer4a) == 0
            and target_is_holiday
            and cfg.ALLOW_HOLIDAY_RELAXATION
        ):
            relaxed_4a = self._time_window_candidates(
                np.asarray([target_i], dtype=np.int16),
                cfg.DATE_TYPE_WEEKDAY,
                target_slot,
                target_ns,
            )
            (
                layer4a, similarity4a, absolute4a, self_rejected,
                target_rejected, self_absolute, target_absolute,
            ) = (
                self._filter_similar_layer(
                    relaxed_4a,
                    target_i,
                    target_j,
                    target_ns,
                    target_trip_miles,
                    "4A",
                )
            )
            selected.self_route_rejected_count += self_rejected
            selected.target_distance_rejected_count += target_rejected
            selected.self_short_absolute_accepted_count += self_absolute
            selected.target_short_absolute_accepted_count += target_absolute
            selected.holiday_relaxed |= len(layer4a) > 0
        selected.add(
            layer4a, 4, self.completion_ns, "4A", similarities=similarity4a
        )
        if selected.enough:
            return selected

        origin_neighbors = self.neighbors.get(
            target_i, np.empty(0, dtype=np.int16)
        )
        strict_4b = self._time_window_candidates(
            origin_neighbors,
            target_date_type,
            target_slot,
            target_ns,
        )
        (
            layer4b, similarity4b, absolute4b, self_rejected,
            target_rejected, self_absolute, target_absolute,
        ) = (
            self._filter_similar_layer(
                strict_4b,
                target_i,
                target_j,
                target_ns,
                target_trip_miles,
                "4B",
            )
        )
        selected.self_route_rejected_count += self_rejected
        selected.target_distance_rejected_count += target_rejected
        selected.self_short_absolute_accepted_count += self_absolute
        selected.target_short_absolute_accepted_count += target_absolute
        if (
            len(layer4b) == 0
            and target_is_holiday
            and cfg.ALLOW_HOLIDAY_RELAXATION
        ):
            relaxed_4b = self._time_window_candidates(
                origin_neighbors,
                cfg.DATE_TYPE_WEEKDAY,
                target_slot,
                target_ns,
            )
            (
                layer4b, similarity4b, absolute4b, self_rejected,
                target_rejected, self_absolute, target_absolute,
            ) = (
                self._filter_similar_layer(
                    relaxed_4b,
                    target_i,
                    target_j,
                    target_ns,
                    target_trip_miles,
                    "4B",
                )
            )
            selected.self_route_rejected_count += self_rejected
            selected.target_distance_rejected_count += target_rejected
            selected.self_short_absolute_accepted_count += self_absolute
            selected.target_short_absolute_accepted_count += target_absolute
            selected.holiday_relaxed |= len(layer4b) > 0
        selected.add(
            layer4b, 4, self.completion_ns, "4B", similarities=similarity4b
        )
        if selected.enough:
            return selected

        # 4C 冷启动回退：相同日期类型、前后4个15分钟时段、同为区内
        # 或区间行程；不要求起终点相邻，但仍同时通过候选自身路线检查
        # 和目标距离相似检查，因此不会仅凭价格接近而纳入样本。
        raw_4c = self._time_window_candidates(
            None, target_date_type, target_slot, target_ns
        )
        (
            layer4c, similarity4c, absolute4c, self_rejected,
            target_rejected, self_absolute, target_absolute,
        ) = (
            self._filter_similar_layer(
                raw_4c,
                target_i,
                target_j,
                target_ns,
                target_trip_miles,
                "4C",
            )
        )
        selected.self_route_rejected_count += self_rejected
        selected.target_distance_rejected_count += target_rejected
        selected.self_short_absolute_accepted_count += self_absolute
        selected.target_short_absolute_accepted_count += target_absolute
        if (
            len(layer4c) == 0
            and target_is_holiday
            and cfg.ALLOW_HOLIDAY_RELAXATION
        ):
            raw_4c = self._time_window_candidates(
                None, cfg.DATE_TYPE_WEEKDAY, target_slot, target_ns
            )
            (
                layer4c, similarity4c, absolute4c, self_rejected,
                target_rejected, self_absolute, target_absolute,
            ) = (
                self._filter_similar_layer(
                    raw_4c,
                    target_i,
                    target_j,
                    target_ns,
                    target_trip_miles,
                    "4C",
                )
            )
            selected.self_route_rejected_count += self_rejected
            selected.target_distance_rejected_count += target_rejected
            selected.self_short_absolute_accepted_count += self_absolute
            selected.target_short_absolute_accepted_count += target_absolute
            selected.holiday_relaxed |= len(layer4c) > 0
        selected.add(
            layer4c, 4, self.completion_ns, "4C", similarities=similarity4c
        )

        # 第5层只服务于原本不可用的目标：没有任何同OD样本，且经过
        # 4A/4B/4C 后相似OD仍少于10笔。已经可用的目标在这里直接返回，
        # 因而第5层不会改变其样本或分布。
        same_count = sum(level <= 3 for level in selected.levels)
        similar_count = sum(level >= 4 for level in selected.levels)
        if same_count > 0 or similar_count >= cfg.SIMILAR_ONLY_MIN_SAMPLES:
            return selected

        selected.fallback_attempted = True

        def add_fallback(
            raw: np.ndarray,
            sublevel: str,
            stage: int,
            ratio_min: float = cfg.TRIP_MILES_RATIO_MIN,
            ratio_max: float = cfg.TRIP_MILES_RATIO_MAX,
        ) -> None:
            # 第5层的全局时间候选可能很多。必须先按行程类型和目标
            # 距离做廉价向量筛选，再保留最近的最多5000笔进入昂贵的
            # “候选自身OD滚动距离”检查；不能像旧4C那样在距离筛选前
            # 截断，否则稀有长距离/短距离目标可能永远检索不到候选。
            raw = np.asarray(raw, dtype=np.int64)
            if len(raw):
                origins = self.i_index[raw]
                destinations = self.j_index[raw]
                same_target_od = (origins == target_i) & (destinations == target_j)
                same_route_type = (
                    (origins == destinations) == (target_i == target_j)
                )
                distance_ok, _ = self._hybrid_distance_mask(
                    self.trip_miles[raw].astype(np.float64),
                    target_trip_miles,
                    float(ratio_min),
                    float(ratio_max),
                    cfg.SHORT_TARGET_REFERENCE_MAX_MILES,
                )
                preliminary = (
                    ~same_target_od
                    & same_route_type
                    & distance_ok
                )
                raw = raw[preliminary]
                if len(raw) > cfg.MAX_RAW_CANDIDATES_PER_SUBLAYER:
                    order = np.argsort(self.completion_ns[raw], kind="stable")
                    raw = raw[order[-cfg.MAX_RAW_CANDIDATES_PER_SUBLAYER :]]
            (
                fallback, similarity, absolute_only, self_rejected,
                target_rejected, self_absolute, target_absolute,
            ) = (
                self._filter_similar_layer(
                    raw,
                    target_i,
                    target_j,
                    target_ns,
                    target_trip_miles,
                    sublevel,
                    ratio_min=ratio_min,
                    ratio_max=ratio_max,
                )
            )
            selected.self_route_rejected_count += self_rejected
            selected.target_distance_rejected_count += target_rejected
            selected.self_short_absolute_accepted_count += self_absolute
            selected.target_short_absolute_accepted_count += target_absolute
            if len(fallback):
                slot_gap = np.abs(
                    self.slot[fallback].astype(np.int16) - int(target_slot)
                )
                slot_gap = np.minimum(slot_gap, cfg.SLOTS_PER_DAY - slot_gap)
                distance_gap = np.abs(
                    np.log(
                        self.trip_miles[fallback].astype(np.float64)
                        / float(target_trip_miles)
                    )
                )
                date_type_mismatch = (
                    self.date_type[fallback].astype(np.int16)
                    != int(target_date_type)
                ).astype(np.int8)
                # np.lexsort最后一个键优先：日期类型、时段差、距离差、
                # 完成时间（越新越优先）。
                order = np.lexsort(
                    (
                        -self.completion_ns[fallback],
                        distance_gap,
                        slot_gap,
                        date_type_mismatch,
                    )
                )
                fallback = fallback[order]
                similarity = similarity[order]
                absolute_only = absolute_only[order]
            before = len(selected.indices)
            # 5E表示在5A—5D当前检索范围内，仅由超短绝对容差接受的
            # 订单；它与各层并行判断，不是在5D失败后才补跑。
            output_sublevels = np.where(absolute_only, "5E", sublevel)
            selected.add(
                fallback,
                5,
                self.completion_ns,
                output_sublevels,
                similarities=similarity,
                preserve_order=True,
            )
            if len(selected.indices) > before:
                selected.fallback_final_level = stage

        def fallback_is_sufficient() -> bool:
            return len(selected.indices) >= cfg.SIMILAR_ONLY_MIN_SAMPLES

        # 5A：相同日期类型，前后2小时。普通距离使用[0.8,1.25]；
        # 超短目标同时执行绝对容差分支，命中的订单标为5E。
        raw_5a = self._time_window_candidates(
            None,
            target_date_type,
            target_slot,
            target_ns,
            window_slots=cfg.LAYER5_TIME_NEIGHBOR_SLOTS_SHORT,
            circular=True,
            cap_raw=False,
        )
        add_fallback(raw_5a, "5A", 1)
        if fallback_is_sufficient():
            return selected

        # 5B：全部日期类型，仍为前后2小时和严格比例/超短联合距离。
        raw_5b = self._all_date_type_time_candidates(
            target_slot,
            target_ns,
            cfg.LAYER5_TIME_NEIGHBOR_SLOTS_SHORT,
        )
        add_fallback(raw_5b, "5B", 2)
        if fallback_is_sufficient():
            return selected

        # 5C：全部日期类型，时间扩大到前后4小时，比例范围仍不放宽。
        raw_5c = self._all_date_type_time_candidates(
            target_slot,
            target_ns,
            cfg.LAYER5_TIME_NEIGHBOR_SLOTS_LONG,
        )
        add_fallback(raw_5c, "5C", 3)
        if fallback_is_sufficient():
            return selected

        # 5D：最后才把普通距离从[0.8,1.25]小幅放宽至[0.75,1.33]。
        # 超短绝对分支仍与5D并行，且仍标记为5E。
        # raw_5c 已经覆盖同一时间和日期候选，可以直接复用。
        add_fallback(
            raw_5c,
            "5D",
            4,
            ratio_min=cfg.LAYER5_DISTANCE_RATIO_MIN,
            ratio_max=cfg.LAYER5_DISTANCE_RATIO_MAX,
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
            target_date_type,
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
            int(target_date_type),
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
                "layer4c_count": 0,
                "layer5a_count": 0,
                "layer5b_count": 0,
                "layer5c_count": 0,
                "layer5d_count": 0,
                "layer5e_count": 0,
                "target_reference_trip_miles": float(
                    candidates.target_reference_trip_miles
                ),
                "target_distance_sample_count": int(
                    candidates.target_distance_sample_count
                ),
                "target_distance_reference_source": int(
                    candidates.target_distance_reference_source
                ),
                "self_route_rejected_count": int(
                    candidates.self_route_rejected_count
                ),
                "target_distance_rejected_count": int(
                    candidates.target_distance_rejected_count
                ),
                "self_short_absolute_accepted_count": int(
                    candidates.self_short_absolute_accepted_count
                ),
                "target_short_absolute_accepted_count": int(
                    candidates.target_short_absolute_accepted_count
                ),
                "sample_prices": [],
                "sample_levels": [],
                "sample_sublevels": [],
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
             cfg.LAYER_WEIGHTS[3], cfg.LAYER_WEIGHTS[4],
             cfg.LAYER_WEIGHTS[5]],
            dtype=np.float64,
        )
        same_od_mask = levels <= 3
        similar_od_mask = levels >= 4
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
            # 为保持既有Parquet结构兼容，layer4_count继续表示整个
            # “相似OD成分”的样本总数，其中包含4A—4C和5A—5E。
            "layer4_count": int(np.sum(levels >= 4)),
            "layer4a_count": int(np.sum(sublevels == "4A")),
            "layer4b_count": int(np.sum(sublevels == "4B")),
            "layer4c_count": int(np.sum(sublevels == "4C")),
            "layer5a_count": int(np.sum(sublevels == "5A")),
            "layer5b_count": int(np.sum(sublevels == "5B")),
            "layer5c_count": int(np.sum(sublevels == "5C")),
            "layer5d_count": int(np.sum(sublevels == "5D")),
            "layer5e_count": int(np.sum(sublevels == "5E")),
            "target_reference_trip_miles": float(
                candidates.target_reference_trip_miles
            ),
            "target_distance_sample_count": int(
                candidates.target_distance_sample_count
            ),
            "target_distance_reference_source": int(
                candidates.target_distance_reference_source
            ),
            "self_route_rejected_count": int(
                candidates.self_route_rejected_count
            ),
            "target_distance_rejected_count": int(
                candidates.target_distance_rejected_count
            ),
            "self_short_absolute_accepted_count": int(
                candidates.self_short_absolute_accepted_count
            ),
            "target_short_absolute_accepted_count": int(
                candidates.target_short_absolute_accepted_count
            ),
            "sample_prices": prices.astype(np.float32).tolist(),
            "sample_levels": levels.tolist(),
            "sample_sublevels": sublevels.tolist(),
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


def sample_results_to_table(results: list[dict[str, object]]) -> pa.Table:
    """把收集结果转换成稳定的Parquet表结构，供全量和定向流程复用。"""

    data = pd.DataFrame(results)
    prices = pa.array(data.pop("sample_prices").tolist(), type=pa.list_(pa.float32()))
    levels = pa.array(data.pop("sample_levels").tolist(), type=pa.list_(pa.int8()))
    sublevels = pa.array(
        data.pop("sample_sublevels").tolist(), type=pa.list_(pa.string())
    )
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
        "layer4a_count", "layer4b_count", "layer4c_count",
        "layer5a_count", "layer5b_count", "layer5c_count", "layer5d_count",
        "layer5e_count",
        "target_distance_sample_count", "target_distance_reference_source",
        "self_route_rejected_count", "target_distance_rejected_count",
        "self_short_absolute_accepted_count",
        "target_short_absolute_accepted_count",
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
    table = table.append_column("sample_sublevels", sublevels)
    table = table.append_column("sample_trip_miles", trip_miles)
    table = table.append_column("sample_similarity_weights", similarity_weights)
    return table


def write_sample_part(results: list[dict[str, object]], part_number: int) -> Path:
    table = sample_results_to_table(results)
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
        cfg.HISTORY_DATE_TYPE_FILE,
        cfg.HISTORY_TRIP_MILES_FILE, cfg.HISTORY_TRIP_SECONDS_FILE,
        cfg.OD_ORDER_FILE, cfg.OD_OFFSETS_FILE,
        cfg.ORIGIN_TIME_STRICT_ORDER_FILE, cfg.ORIGIN_TIME_STRICT_OFFSETS_FILE,
        cfg.ORIGIN_TIME_RELAXED_ORDER_FILE, cfg.ORIGIN_TIME_RELAXED_OFFSETS_FILE,
        cfg.ORIGIN_DATE_TYPE_TIME_ORDER_FILE,
        cfg.ORIGIN_DATE_TYPE_TIME_OFFSETS_FILE,
        cfg.GLOBAL_DATE_TYPE_TIME_ORDER_FILE,
        cfg.GLOBAL_DATE_TYPE_TIME_OFFSETS_FILE,
        cfg.NEIGHBOR_MATRIX_FILE,
        cfg.NORMALIZED_NEIGHBOR_FILE,
        cfg.OD_ROAD_DISTANCE_FILE,
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
        "target_weekend_or_holiday", "target_date_type", "D_ijt_observed",
    ]

    pool = None
    if args.workers > 1:
        pool = mp.get_context("fork").Pool(processes=args.workers)

    counters = {
        "processed_rows_this_run": 0,
        "skipped_existing_batches": 0,
        "available_rows_this_run": 0,
        "insufficient_rows_this_run": 0,
        "layer5e_samples_this_run": 0,
        "target_short_absolute_rows_this_run": 0,
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
            counters["layer5e_samples_this_run"] += sum(
                int(row["layer5e_count"]) for row in results
            )
            counters["target_short_absolute_rows_this_run"] += sum(
                int(row["target_short_absolute_accepted_count"] > 0)
                for row in results
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
        "kde_components": (
            "same_od_layers_1_2_3 + similar_od_layers_4A_4B_4C_5A_5B_5C_5D_5E"
        ),
        "short_distance_rule": {
            "candidate_reference_max_miles": float(
                cfg.SHORT_CANDIDATE_REFERENCE_MAX_MILES
            ),
            "target_reference_max_miles": float(
                cfg.SHORT_TARGET_REFERENCE_MAX_MILES
            ),
            "absolute_tolerance_miles": float(
                cfg.SHORT_DISTANCE_ABSOLUTE_TOLERANCE_MILES
            ),
            "layer5e_meaning": "absolute-only branch evaluated within 5A-5D",
        },
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
