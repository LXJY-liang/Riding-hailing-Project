# -*- coding: utf-8 -*-
"""从六种估算层级中各抽取一个 c_ij，并回溯、复算全部关键细节。

抽样规则：
1. i!=j, N>=20：历史里程中位数；
2. i!=j, 1<=N<20：历史中位数向区域间道路先验收缩；
3. i!=j, N=0：区域间道路先验；
4. i=j, N>=20：历史里程中位数；
5. i=j, 1<=N<20：历史中位数向区域内空间先验收缩；
6. i=j, N=0：区域内空间先验。

脚本会生成两个核对文件：
- JSON：六个样本的最终结果、历史统计、空间先验、校准参数和复算误差；
- CSV：六个样本对应的全部有效历史订单。第3、6层本来就没有历史订单，
  因而只在JSON中记录历史样本数为0，不会在CSV中产生订单行。

固定 AUDIT_RANDOM_SEED 后，每次运行都会抽到同一组样本，便于重复核对。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_config():
    """按文件路径载入统一配置，避免中文文件名影响常规 import。"""

    path = Path(__file__).resolve().parent / "1 c_ij构建配置.py"
    spec = importlib.util.spec_from_file_location("cij_config", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


cfg = load_config()


def load_od_history(i: str, j: str) -> pd.DataFrame:
    """读取指定OD的有效历史订单，并严格复用脚本2的数据筛选口径。"""

    columns = [
        "Trip ID",
        "Trip End Timestamp",
        "Trip Miles",
        "Trip Seconds",
        "Pickup Census Tract",
        "Dropoff Census Tract",
    ]
    # Parquet分区过滤会尽量只读取目标OD，避免为每个样本载入整张订单表。
    orders = pd.read_parquet(
        cfg.ORDER_FILE,
        columns=columns,
        filters=[
            ("Pickup Census Tract", "==", i),
            ("Dropoff Census Tract", "==", j),
        ],
    )

    # 与历史OD统计完全一致：只保留截止日前、里程和时长为正、速度合理的订单。
    speed_mph = orders["Trip Miles"] / (orders["Trip Seconds"] / 3600.0)
    valid = (
        (orders["Trip End Timestamp"] < pd.Timestamp(cfg.HISTORY_END))
        & (orders["Trip Miles"] > 0)
        & (orders["Trip Seconds"] > 0)
        & speed_mph.between(cfg.MIN_SPEED_MPH, cfg.MAX_SPEED_MPH)
    )
    history = orders.loc[valid].copy()
    history["speed_mph"] = speed_mph.loc[valid]
    history = history.sort_values(
        ["Trip End Timestamp", "Trip ID"],
        kind="mergesort",
    ).reset_index(drop=True)
    history.insert(0, "history_order", np.arange(1, len(history) + 1))
    return history


def audit_one_sample(
    sampled: pd.Series,
    parameters: dict,
    routing: pd.DataFrame,
    intra: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    """复算一个样本的 c_ij，并返回JSON摘要和逐笔历史订单。"""

    i = str(sampled["i"])
    j = str(sampled["j"])
    level = int(sampled["estimation_level"])
    history = load_od_history(i, j)

    # 给合并后的历史明细增加样本标识，便于按层级和OD筛选查看。
    history.insert(0, "j", j)
    history.insert(0, "i", i)
    history.insert(0, "estimation_level", level)
    distances = history["Trip Miles"].to_numpy(dtype=np.float64)
    history_median = float(np.median(distances)) if len(distances) else None

    if i != j:
        # 区域间先验：先取质心代表节点之间的道路距离，再用 a+b*d_map 校准。
        model = parameters["interzone"]
        prior_source = "interzone_road_distance"
        raw_prior_distance = float(sampled["d_map_miles"])
        prior_distance = model["a"] + model["b"] * raw_prior_distance
        kappa = model["kappa"]
        calibration_used = model
    else:
        # 区域内先验优先使用区域内道路节点抽样距离；不可得时使用有效半径。
        # 两种原始距离均已在脚本3中统一为英里，可直接代入英里口径的校准模型。
        intra_row = intra.loc[i]
        if pd.notna(intra_row["d_intra_network_miles"]):
            model = parameters["intrazone_network"]
            prior_source = "intrazone_network_distance"
            raw_prior_distance = float(intra_row["d_intra_network_miles"])
        else:
            model = parameters["intrazone_radius"]
            prior_source = "intrazone_effective_radius"
            raw_prior_distance = float(intra_row["effective_radius_miles"])
        prior_distance = model["a"] + model["b"] * raw_prior_distance
        kappa = parameters["intrazone_kappa"]
        calibration_used = {**model, "kappa": kappa}

    # 严格按对应层级重新计算最终距离，而不是直接复制c_ij结果表中的数值。
    if level in (1, 4):
        history_weight = 1.0
        reconstructed_distance = history_median
        formula = "historical_median"
        calibration_applied = False
    elif level in (2, 5):
        history_weight = len(distances) / (len(distances) + kappa)
        reconstructed_distance = (
            history_weight * history_median
            + (1.0 - history_weight) * prior_distance
        )
        formula = (
            "history_weight * historical_median "
            "+ (1-history_weight) * calibrated_prior"
        )
        calibration_applied = True
    else:
        history_weight = 0.0
        reconstructed_distance = prior_distance
        formula = "calibrated_prior"
        calibration_applied = True

    reconstructed_cost = cfg.COST_PER_MILE * reconstructed_distance
    summary = {
        "sample": {
            "i": i,
            "j": j,
            "estimation_level": level,
            "N_ij_in_result": int(sampled["N_ij"]),
            "d_hist_median_miles_in_result": (
                None
                if pd.isna(sampled["d_hist_median_miles"])
                else float(sampled["d_hist_median_miles"])
            ),
            "d_map_miles_in_result": float(sampled["d_map_miles"]),
            "d_prior_miles_in_result": float(sampled["d_prior_miles"]),
            "shrinkage_weight_in_result": float(sampled["shrinkage_weight"]),
            "d_ij_miles_in_result": float(sampled["d_ij_miles"]),
            "c_ij_usd_in_result": float(sampled["c_ij_usd"]),
        },
        "historical_orders": {
            "count_recomputed": int(len(distances)),
            "trip_miles": distances.tolist(),
            "median_miles_recomputed": history_median,
            "detail_file": str(cfg.AUDIT_HISTORY_FILE),
        },
        "spatial_prior": {
            "source": prior_source,
            "raw_distance_miles": raw_prior_distance,
            "calibrated_distance_miles": float(prior_distance),
            "origin_routing_node": str(routing.loc[i, "routing_node_id"]),
            "destination_routing_node": str(routing.loc[j, "routing_node_id"]),
        },
        "calibration": {
            "applied_to_final_distance": calibration_applied,
            "parameters_used": calibration_used,
        },
        "reconstruction": {
            "formula": formula,
            "history_weight": float(history_weight),
            "cost_per_mile": cfg.COST_PER_MILE,
            "distance_miles_recomputed": float(reconstructed_distance),
            "cost_usd_recomputed": float(reconstructed_cost),
            "distance_difference": float(
                reconstructed_distance - sampled["d_ij_miles"]
            ),
            "cost_difference": float(reconstructed_cost - sampled["c_ij_usd"]),
        },
    }
    return summary, history


def main() -> None:
    """每层随机抽取一个样本，统一输出六个样本的核对结果。"""

    cfg.ensure_output_directories()
    cij = pd.read_parquet(cfg.CIJ_FILE)
    parameters = json.loads(cfg.CALIBRATION_FILE.read_text(encoding="utf-8"))
    routing = pd.read_parquet(cfg.REGION_ROUTING_NODE_FILE)
    routing["region_id"] = routing["region_id"].astype(str)
    routing = routing.set_index("region_id")
    intra = pd.read_parquet(cfg.INTRA_ZONE_PRIOR_FILE)
    intra["region_id"] = intra["region_id"].astype(str)
    intra = intra.set_index("region_id")

    summaries = []
    history_tables = []
    for level in range(1, 7):
        # 每层使用不同但固定的random_state，保证六次抽样可独立复现。
        candidates = cij.loc[cij["estimation_level"] == level]
        sampled = candidates.sample(
            n=1,
            random_state=cfg.AUDIT_RANDOM_SEED + level,
        ).iloc[0]
        summary, history = audit_one_sample(
            sampled,
            parameters,
            routing,
            intra,
        )
        summaries.append(summary)
        history_tables.append(history)

    # 合并后按层级、OD和订单顺序排列；第3、6层因N=0而不会产生明细行。
    combined_history = pd.concat(history_tables, ignore_index=True)
    combined_history = combined_history.sort_values(
        ["estimation_level", "i", "j", "history_order"],
        kind="mergesort",
    )
    combined_history.to_csv(
        cfg.AUDIT_HISTORY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # JSON保留六个样本各自的完整复算链条，并附上本次抽样的全局信息。
    audit = {
        "sampling_rule": "one random OD from each estimation level",
        "random_seed_base": cfg.AUDIT_RANDOM_SEED,
        "history_end": cfg.HISTORY_END,
        "all_calibration_parameters": parameters,
        "samples": summaries,
    }
    cfg.AUDIT_SUMMARY_FILE.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
