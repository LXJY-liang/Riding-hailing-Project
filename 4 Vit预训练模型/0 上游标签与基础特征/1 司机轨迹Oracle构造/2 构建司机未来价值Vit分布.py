# -*- coding: utf-8 -*-
"""模拟司机未来 4 小时收益分布，直接构造 101 维监督标签。

对每个起始状态 (i,t)，脚本重复模拟 N 名代表性司机：

1. 在当前区域以 `D_origin_it / S_reference_it` 的概率获得一笔历史成交订单；
2. 按历史目的地成交量占比抽取 j，计算该订单的司机净收益，
   并在 l 个时段后到达 j；历史成交轨迹不额外模拟司机拒单；
3. 若没有订单，司机留在原区域并等待一个时段；
4. 重复到未来 16 个时段结束。

每次轨迹的随机订单分配结果和目的地可能不同，所以会得到 N 个累计收益 Z。本版直接
从 1000 个收益样本计算 q0.005、q0.01、...、q0.99、q0.995 共 101 个经验分位数，
不再先构造 19 维标签再插值。标签使用未来历史结果，只能用于监督学习，不能直接作为
SAC 在时点 t 的状态。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def load_config():
    path = Path(__file__).resolve().parent / "1 轨迹Oracle构造配置.py"
    spec = importlib.util.spec_from_file_location("vit_oracle_config", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


cfg = load_config()


@dataclass
class TransitionTable:
    """一个历史时段的稀疏分类分布，供向量化逆 CDF 抽样使用。"""

    shifted_cdf: np.ndarray
    destination: np.ndarray
    reward: np.ndarray
    duration: np.ndarray


class HistoricalDriverSimulator:
    """在固定历史订单、供给、价格和成本下模拟司机轨迹。"""

    def __init__(self, od: pd.DataFrame, supply: pd.DataFrame, region_count: int):
        self.region_count = int(region_count)
        self.rho = float(cfg.PLATFORM_COMMISSION_RATE)
        self.horizon = int(cfg.DRIVER_VALUE_HORIZON)
        self.gamma = float(cfg.DRIVER_GAMMA)

        all_times = pd.date_range(
            min(supply["t"].min(), od["t"].min()),
            max(supply["t"].max(), od["t"].max()),
            freq=f"{cfg.TIME_STEP_MINUTES}min",
        )
        self.times = pd.DatetimeIndex(all_times)
        self.time_to_index = {timestamp: index for index, timestamp in enumerate(self.times)}

        supply = supply.copy()
        supply["time_index"] = supply["t"].map(self.time_to_index).astype(np.int32)
        self.supply = np.zeros((len(self.times), self.region_count), dtype=np.float32)
        self.supply[
            supply["time_index"].to_numpy(dtype=np.int64),
            supply["region_index"].to_numpy(dtype=np.int64),
        ] = supply[cfg.SUPPLY_VALUE_COLUMN].to_numpy(dtype=np.float32)

        od = od.copy()
        od["time_index"] = od["t"].map(self.time_to_index).astype(np.int32)
        od = od.sort_values(["time_index", "i_index", "j_index"], kind="mergesort")
        self.od = od.reset_index(drop=True)
        time_index = self.od["time_index"].to_numpy(dtype=np.int32)
        self.od_i = self.od["i_index"].to_numpy(dtype=np.int32)
        self.od_j = self.od["j_index"].to_numpy(dtype=np.int32)
        self.od_demand = self.od["D_ijt_observed"].to_numpy(dtype=np.float64)
        self.od_price = self.od["p_ref_ijt"].to_numpy(dtype=np.float32)
        self.od_cost = self.od["c_bar_ijt"].to_numpy(dtype=np.float32)
        self.od_duration = self.od["arrival_step_ijt"].to_numpy(dtype=np.int16)
        counts = np.bincount(time_index, minlength=len(self.times))
        self.time_ends = np.cumsum(counts, dtype=np.int64)
        self.time_starts = self.time_ends - counts

        # 只缓存未来窗口附近的时段，避免为 2880 个时段保存重复的大数组。
        self.cache: OrderedDict[int, TransitionTable] = OrderedDict()
        self.cache_limit = self.horizon + 2

    def _build_transition(self, time_index: int) -> TransitionTable:
        """构造某时段每个区域的“未接单 + 各目的地订单”分类分布。"""

        start = int(self.time_starts[time_index])
        end = int(self.time_ends[time_index])
        current_i = self.od_i[start:end]
        current_j = self.od_j[start:end]
        current_demand = self.od_demand[start:end]
        current_price = self.od_price[start:end]
        current_cost = self.od_cost[start:end]
        current_duration = self.od_duration[start:end]

        shifted_cdf_parts: list[np.ndarray] = []
        destination_parts: list[np.ndarray] = []
        reward_parts: list[np.ndarray] = []
        duration_parts: list[np.ndarray] = []

        origins = np.arange(self.region_count, dtype=np.int32)
        left = np.searchsorted(current_i, origins, side="left")
        right = np.searchsorted(current_i, origins, side="right")

        for origin in range(self.region_count):
            s_it = float(self.supply[time_index, origin])

            local_start = int(left[origin])
            local_end = int(right[origin])
            if local_start == local_end or s_it <= 0:
                probabilities = np.asarray([1.0], dtype=np.float64)
                destinations = np.asarray([-1], dtype=np.int32)
                rewards = np.asarray([0.0], dtype=np.float32)
                durations = np.asarray([1], dtype=np.int16)
            else:
                demand = current_demand[local_start:local_end]
                total_demand = float(demand.sum())
                # S 已保证不小于历史出发成交量。clip 只处理浮点误差或上游口径差异。
                order_probability = min(total_demand / max(s_it, 1e-8), 1.0)
                historical_reward = (
                    (1.0 - self.rho) * current_price[local_start:local_end]
                    - current_cost[local_start:local_end]
                )
                edge_probabilities = order_probability * demand / max(total_demand, 1e-8)
                no_order_probability = max(1.0 - order_probability, 0.0)
                probabilities = np.concatenate(
                    [[no_order_probability], edge_probabilities]
                )
                probabilities = probabilities / probabilities.sum()
                destinations = np.concatenate(
                    [
                        np.asarray([-1], dtype=np.int32),
                        current_j[local_start:local_end],
                    ]
                )
                rewards = np.concatenate(
                    [[0.0], historical_reward]
                ).astype(np.float32)
                durations = np.concatenate(
                    [
                        np.asarray([1], dtype=np.int16),
                        current_duration[local_start:local_end],
                    ]
                )

            cdf = np.cumsum(probabilities)
            cdf[-1] = 1.0
            # 每个 origin 的 CDF 平移到互不重叠的 [2i,2i+1] 区间。这样可对不同区域
            # 的粒子一次调用 np.searchsorted，避免逐司机 Python 循环。
            shifted_cdf_parts.append(cdf + 2.0 * origin)
            destination_parts.append(destinations)
            reward_parts.append(rewards)
            duration_parts.append(durations)

        return TransitionTable(
            shifted_cdf=np.concatenate(shifted_cdf_parts),
            destination=np.concatenate(destination_parts),
            reward=np.concatenate(reward_parts),
            duration=np.concatenate(duration_parts),
        )

    def transition(self, time_index: int) -> TransitionTable:
        if time_index in self.cache:
            value = self.cache.pop(time_index)
            self.cache[time_index] = value
            return value
        value = self._build_transition(time_index)
        self.cache[time_index] = value
        while len(self.cache) > self.cache_limit:
            self.cache.popitem(last=False)
        return value

    def simulate(self, start_time_index: int, start_regions: np.ndarray, samples: int) -> np.ndarray:
        """同时模拟同一开始时段的多个区域，返回 `[区域数, samples]` 累计收益。"""

        start_regions = np.asarray(start_regions, dtype=np.int32)
        current_region = np.repeat(start_regions, samples)
        next_decision_offset = np.zeros(len(current_region), dtype=np.int16)
        returns = np.zeros(len(current_region), dtype=np.float32)

        # 每个开始时段使用独立且可复现的随机种子；分段重跑不会改变已有结果。
        rng = np.random.default_rng(cfg.VIT_RANDOM_SEED + int(start_time_index))

        for elapsed in range(self.horizon):
            active = np.flatnonzero(next_decision_offset == elapsed)
            if not len(active):
                continue
            absolute_time = start_time_index + elapsed
            if absolute_time >= len(self.times):
                break

            table = self.transition(absolute_time)
            origin = current_region[active]
            query = rng.random(len(active)) + 2.0 * origin
            outcome_index = np.searchsorted(table.shifted_cdf, query, side="left")
            destination = table.destination[outcome_index]
            reward = table.reward[outcome_index]
            duration = table.duration[outcome_index].astype(np.int16)

            returns[active] += (self.gamma ** elapsed) * reward
            accepted = destination >= 0
            if accepted.any():
                current_region[active[accepted]] = destination[accepted]
            next_decision_offset[active] = elapsed + np.maximum(duration, 1)

        return returns.reshape(len(start_regions), samples)


def construct_targets(
    od: pd.DataFrame,
    all_times: pd.DatetimeIndex,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """生成 801 个区域的完整开始状态，避免稀疏区再次丢失 V 标签。

    虽然 Actor 只对有订单 OD 输出动作，但拒单或接单后的目的地状态可能是当时无出发
    订单的区域。因此 V 必须覆盖全部区域-时段，而不能只覆盖活跃起点。
    """

    region_indices = np.arange(int(max(od["i_index"].max(), od["j_index"].max())) + 1)
    calendar = pd.DataFrame({"t": all_times})
    calendar["slot_in_day"] = calendar["t"].dt.hour * 4 + calendar["t"].dt.minute // 15

    last_full_start = all_times[-1] - pd.Timedelta(
        minutes=(cfg.DRIVER_VALUE_HORIZON - 1) * cfg.TIME_STEP_MINUTES
    )
    if end > last_full_start:
        raise ValueError(
            f"标签结束时刻 {end} 缺少完整未来窗口；当前最晚可构造 {last_full_start}。"
        )
    calendar = calendar.loc[calendar["t"].between(start, end)]
    targets = pd.MultiIndex.from_product(
        [calendar["t"].to_numpy(), region_indices], names=["t", "region_index"]
    ).to_frame(index=False)
    targets["slot_in_day"] = targets["t"].dt.hour * 4 + targets["t"].dt.minute // 15
    return targets.sort_values(["t", "region_index"], kind="mergesort").reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构造历史司机未来 4 小时的 101 维价值标签")
    parser.add_argument(
        "--split",
        choices=("train", "valid"),
        default="train",
        help="train构造6月1—24日；valid只构造6月25—27日独立验证标签。",
    )
    parser.add_argument("--samples", type=int, default=cfg.VIT_MONTE_CARLO_SAMPLES)
    parser.add_argument(
        "--smoke-start-times",
        type=int,
        default=0,
        help="仅测试前 N 个开始时段并写入 smoke 文件；0 表示正式全量运行。",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg.ensure_output_directories()
    if args.samples != cfg.VIT_MONTE_CARLO_SAMPLES:
        raise ValueError(
            f"本次正式口径固定为 {cfg.VIT_MONTE_CARLO_SAMPLES} 条轨迹，"
            "不得通过 --samples 改变。"
        )

    region_map = pd.read_csv(cfg.REGION_MAP_FILE, dtype={"region_id": "string"})
    region_to_index = dict(zip(region_map["region_id"].astype(str), region_map["region_index"]))
    region_to_id = dict(zip(region_map["region_index"], region_map["region_id"].astype(str)))
    region_count = len(region_map)

    od_columns = [
        "i", "j", "t", "i_index", "j_index", "D_ijt_observed", "p_ref_ijt",
        "c_bar_ijt", "arrival_step_ijt", "tau_q10_ijt", "tau_q25_ijt", "tau_q50_ijt",
        "tau_q75_ijt", "tau_q90_ijt",
    ]
    od = pd.read_parquet(cfg.BASE_OD_OUTPUT, columns=od_columns)
    od["t"] = pd.to_datetime(od["t"])

    supply = pd.read_parquet(
        cfg.SUPPLY_OUTPUT, columns=["region_id", "t", cfg.SUPPLY_VALUE_COLUMN]
    )
    supply["region_id"] = supply["region_id"].astype("string")
    supply["t"] = pd.to_datetime(supply["t"])
    supply["region_index"] = supply["region_id"].map(region_to_index)
    if supply["region_index"].isna().any():
        raise ValueError("S_it 中出现不在 region_mapping.csv 的区域。")
    supply["region_index"] = supply["region_index"].astype(np.int32)

    simulator = HistoricalDriverSimulator(od, supply, region_count)
    if args.split == "train":
        label_start = pd.Timestamp(cfg.LABEL_START)
        label_end = pd.Timestamp(cfg.LABEL_END)
        formal_output = cfg.VIT_OUTPUT
        formal_diagnostics = cfg.VIT_DIAGNOSTICS
    else:
        label_start = pd.Timestamp(cfg.VALID_LABEL_START)
        label_end = pd.Timestamp(cfg.VALID_LABEL_END)
        formal_output = cfg.VIT_VALID_OUTPUT
        formal_diagnostics = cfg.VIT_VALID_DIAGNOSTICS
    targets = construct_targets(od, simulator.times, label_start, label_end)
    start_times = targets["t"].drop_duplicates().sort_values()
    if args.smoke_start_times > 0:
        start_times = start_times.iloc[: args.smoke_start_times]
        targets = targets.loc[targets["t"].isin(start_times)].copy()

    if args.smoke_start_times > 0:
        output_file = formal_output.with_name(f"{formal_output.stem}_smoke.parquet")
        diagnostics_file = formal_diagnostics.with_name(
            f"{formal_diagnostics.stem}_smoke.json"
        )
    else:
        output_file = formal_output
        diagnostics_file = formal_diagnostics
    if output_file.exists() and not args.overwrite:
        raise FileExistsError(f"{output_file} 已存在；覆盖时加入 --overwrite。")
    temporary_output = output_file.with_suffix(".parquet.tmp")
    if temporary_output.exists():
        temporary_output.unlink()

    writer: pq.ParquetWriter | None = None
    day_parts: list[pd.DataFrame] = []
    current_day: pd.Timestamp | None = None
    rows = 0
    median_sum = 0.0
    median_index = int(np.argmin(np.abs(cfg.VIT_QUANTILE_LEVELS - 0.50)))
    try:
        for number, timestamp in enumerate(start_times, start=1):
            timestamp = pd.Timestamp(timestamp)
            day = timestamp.normalize()
            if current_day is None:
                current_day = day
            if day != current_day:
                table = pa.Table.from_pandas(pd.concat(day_parts, ignore_index=True), preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(temporary_output, table.schema, compression="zstd")
                writer.write_table(table)
                day_parts.clear()
                current_day = day

            current = targets.loc[targets["t"] == timestamp].copy()
            regions = current["region_index"].to_numpy(dtype=np.int32)
            time_index = simulator.time_to_index[timestamp]
            returns = simulator.simulate(time_index, regions, args.samples)
            quantiles = np.quantile(
                returns, cfg.VIT_QUANTILE_LEVELS, axis=1, method="linear"
            ).T.astype(np.float32)
            identity = current[["region_index", "t", "slot_in_day"]].copy()
            identity.insert(
                0,
                "region_id",
                identity["region_index"].map(region_to_id).astype("string"),
            )
            quantile_frame = pd.DataFrame(
                quantiles,
                columns=cfg.VIT_QUANTILE_COLUMNS,
                index=identity.index,
            )
            part = pd.concat([identity, quantile_frame], axis=1)
            day_parts.append(part)
            rows += len(part)
            median_sum += float(quantiles[:, median_index].sum(dtype=np.float64))

            if number % 96 == 0 or number == len(start_times):
                print(f"已完成 {number:,}/{len(start_times):,} 个开始时段")

        if day_parts:
            table = pa.Table.from_pandas(pd.concat(day_parts, ignore_index=True), preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary_output, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError("没有生成任何 Vit 标签。")
    temporary_output.replace(output_file)

    diagnostics = {
        "definition": "historical_realization_proxy_label",
        "split": args.split,
        "uses_future_historical_outcomes": True,
        "intended_use": "Vit supervised target only; never direct SAC state",
        "rows": int(rows),
        "regions": int(region_count),
        "start_times": int(len(start_times)),
        "label_start": str(start_times.iloc[0]),
        "label_end": str(start_times.iloc[-1]),
        "monte_carlo_samples": int(args.samples),
        "horizon_slots": int(cfg.DRIVER_VALUE_HORIZON),
        "driver_gamma": float(cfg.DRIVER_GAMMA),
        "quantile_count": int(len(cfg.VIT_QUANTILE_LEVELS)),
        "quantile_levels": [float(value) for value in cfg.VIT_QUANTILE_LEVELS],
        "quantile_method": f"numpy.linear_from_{args.samples}_returns",
        "negative_or_zero_net_order_policy": "follow_historical_completed_order_transition",
        "value_lower_bound": None,
        "supply_proxy": str(cfg.SUPPLY_OUTPUT),
        "oracle_median_mean": median_sum / max(rows, 1),
        "smoke_start_times": int(args.smoke_start_times),
    }
    diagnostics_file.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    print(f"输出：{output_file}")


if __name__ == "__main__":
    main()
