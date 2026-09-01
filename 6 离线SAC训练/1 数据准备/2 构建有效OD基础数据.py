# -*- coding: utf-8 -*-
"""任务 2：构造第一版离线 SAC 使用的有订单 OD 状态表。

本脚本只读取现有的 OD-15min 成交聚合结果，不复制或移动上游文件。每一行对应一个
确实出现过成交订单的 (i, j, t)。无订单 OD 不进入 Actor 动作集合。
"""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG_PATH = Path(__file__).resolve().parent / "1 离线SAC统一配置.py"
CONFIG_SPEC = importlib.util.spec_from_file_location("offline_sac_config", CONFIG_PATH)
cfg = importlib.util.module_from_spec(CONFIG_SPEC)
assert CONFIG_SPEC.loader is not None
CONFIG_SPEC.loader.exec_module(cfg)

ACTIVE_OD_FILE = cfg.ACTIVE_OD_FILE
BASE_OD_OUTPUT = cfg.BASE_OD_OUTPUT
DATA_DIAGNOSTICS = cfg.DATA_DIAGNOSTICS
PRICE_CV_FALLBACK = cfg.PRICE_CV_FALLBACK
PRICE_CV_FLOOR_QUANTILE = cfg.PRICE_CV_FLOOR_QUANTILE
REGION_MAP_FILE = cfg.REGION_MAP_FILE
SPLITS = cfg.SPLITS
TRAIN_END = cfg.TRAIN_END
TRAIN_START = cfg.TRAIN_START
ensure_output_directories = cfg.ensure_output_directories


PRICE_QUANTILE_PROBABILITIES = np.asarray([0.10, 0.25, 0.50, 0.75, 0.90])
NORMAL_QUANTILE_Z = np.asarray(
    [-1.2815515655, -0.6744897502, 0.0, 0.6744897502, 1.2815515655],
    dtype=np.float64,
)


def estimate_price_cv_floor(od: pd.DataFrame) -> float:
    """仅用训练期、同一格中有重复订单的样本估计价格离散程度下限。

    原始 q10 和 q90 给出稳健标准差 `(q90-q10)/(2*1.28155)`。对只有一笔订单的
    OD-时段，该值为 0，因此使用训练期可识别样本的价格变异系数分位数作回退。
    """

    train_mask = od["t"].between(pd.Timestamp(TRAIN_START), pd.Timestamp(TRAIN_END))
    repeated = od.loc[train_mask & (od["D_ijt_observed"] >= 2)].copy()
    robust_sigma = (repeated["p_q90_ijt"] - repeated["p_q10_ijt"]) / (2 * 1.2815515655)
    cv = robust_sigma / repeated["p_q50_ijt"].clip(lower=1e-6)
    cv = cv.replace([np.inf, -np.inf], np.nan).dropna()
    cv = cv[cv > 0]

    if cv.empty:
        return float(PRICE_CV_FALLBACK)

    return float(max(np.quantile(cv, PRICE_CV_FLOOR_QUANTILE), PRICE_CV_FALLBACK))


def widen_degenerate_price_quantiles(od: pd.DataFrame, cv_floor: float) -> pd.DataFrame:
    """把成交价代理转换为非退化的乘客最高支付意愿经验分布。

    该步骤不声称恢复了真实未成交乘客的保留价格。它只把已有成交价作为中心，并用
    训练期重复订单估计的最小离散程度形成第一版历史代理分布。
    """

    result = od.copy()
    p50 = result["p_q50_ijt"].to_numpy(dtype=np.float64)
    cell_sigma = (
        result["p_q90_ijt"].to_numpy(dtype=np.float64)
        - result["p_q10_ijt"].to_numpy(dtype=np.float64)
    ) / (2 * 1.2815515655)
    sigma_floor = np.maximum(p50, 0.0) * cv_floor
    sigma = np.maximum(np.nan_to_num(cell_sigma, nan=0.0), sigma_floor)

    raw_columns = ["p_q10_ijt", "p_q25_ijt", "p_q50_ijt", "p_q75_ijt", "p_q90_ijt"]
    proxy_columns = ["pe_q10_ijt", "pe_q25_ijt", "pe_q50_ijt", "pe_q75_ijt", "pe_q90_ijt"]
    raw = result[raw_columns].to_numpy(dtype=np.float64)
    floor_curve = p50[:, None] + sigma[:, None] * NORMAL_QUANTILE_Z[None, :]
    apply_floor = cell_sigma < sigma_floor
    # 对已经有足够组内离散度的单元保留原始经验分位数；只有离散度不足时才用以 p50
    # 为中心的训练期经验下限替换整条曲线，避免只扩大上尾而不扩大下尾。
    proxy = np.where(apply_floor[:, None], floor_curve, raw)
    proxy = np.maximum.accumulate(np.clip(proxy, 0.01, None), axis=1)

    for index, column in enumerate(proxy_columns):
        result[column] = proxy[:, index].astype(np.float32)

    result["pe_dispersion_floor_applied"] = apply_floor.astype(np.int8)
    return result


def main() -> None:
    ensure_output_directories()

    columns = [
        "i", "j", "t", "day_index", "slot_in_day", "dow",
        "is_weekend_or_holiday", "is_peak_morning", "is_peak_evening", "is_night",
        "D_ijt_observed", "p_bar_ijt",
        "p_q10_ijt", "p_q25_ijt", "p_q50_ijt", "p_q75_ijt", "p_q90_ijt",
        "tau_bar_ijt", "tau_q10_ijt", "tau_q25_ijt", "tau_q50_ijt",
        "tau_q75_ijt", "tau_q90_ijt",
        "trip_miles_bar_ijt", "trip_miles_q10_ijt", "trip_miles_q25_ijt",
        "trip_miles_q50_ijt", "trip_miles_q75_ijt", "trip_miles_q90_ijt",
        "temperature_2m", "relative_humidity_2m", "rain", "weather_code",
        "wind_speed_10m",
    ]
    od = pd.read_parquet(ACTIVE_OD_FILE, columns=columns)
    od["i"] = od["i"].astype("string")
    od["j"] = od["j"].astype("string")
    od["t"] = pd.to_datetime(od["t"], errors="raise")
    od = od.loc[od["D_ijt_observed"] > 0].copy()
    nonpositive_price_rows = int((od["p_bar_ijt"] <= 0).sum())
    nonpositive_price_orders = int(
        od.loc[od["p_bar_ijt"] <= 0, "D_ijt_observed"].sum()
    )
    # 价格为 0 的成交记录不能定义 p/p_ref，也不能作为乘客最高支付意愿代理。当前数据
    # 中它们占比极低，因此明确剔除并把数量写入 diagnostics，而不是静默补成正价格。
    od = od.loc[od["p_bar_ijt"] > 0].copy()

    if od[["i", "j", "t"]].duplicated().any():
        raise ValueError("有订单 OD 聚合表存在重复的 (i,j,t)。")

    cv_floor = estimate_price_cv_floor(od)
    od = widen_degenerate_price_quantiles(od, cv_floor)

    # 历史服务成本代理与此前主流程保持一致：0.67 美元/英里 + 0.27 美元/分钟。
    od["c_bar_ijt"] = (
        0.67 * od["trip_miles_bar_ijt"] + 0.27 * od["tau_bar_ijt"]
    ).astype(np.float32)
    for quantile in (10, 25, 50, 75, 90):
        od[f"c_q{quantile}_ijt"] = (
            0.67 * od[f"trip_miles_q{quantile}_ijt"]
            + 0.27 * od[f"tau_q{quantile}_ijt"]
        ).astype(np.float32)

    od["p_ref_ijt"] = od["p_bar_ijt"].astype(np.float32)
    od["b_hist_ijt"] = np.float32(0.0)
    od["arrival_step_ijt"] = np.maximum(
        np.ceil(od["tau_bar_ijt"] / 15.0).astype(np.int16), 1
    )

    # 稳定排序后生成唯一编号。group_id 表示同一个 (i,t)，组内所有 j 必须一起计算
    # 区域供给缩放项。
    od = od.sort_values(["t", "i", "j"], kind="mergesort").reset_index(drop=True)
    od["state_id"] = np.arange(len(od), dtype=np.int64)
    od["group_id"] = pd.factorize(
        pd.MultiIndex.from_frame(od[["i", "t"]]), sort=False
    )[0].astype(np.int64)

    regions = pd.Index(pd.concat([od["i"], od["j"]], ignore_index=True).unique()).sort_values()
    region_map = pd.DataFrame(
        {"region_id": regions.astype("string"), "region_index": np.arange(len(regions), dtype=np.int32)}
    )
    region_map.to_csv(REGION_MAP_FILE, index=False)
    region_to_index = dict(zip(region_map["region_id"].astype(str), region_map["region_index"]))
    od["i_index"] = od["i"].map(region_to_index).astype(np.int32)
    od["j_index"] = od["j"].map(region_to_index).astype(np.int32)

    # split 只用于后续筛选。预热期保留为 warmup，以便构造严格只用过去日期的 V 代理。
    od["split"] = "warmup"
    for name, (start, end) in SPLITS.items():
        od.loc[od["t"].between(pd.Timestamp(start), pd.Timestamp(end)), "split"] = name

    float_columns = od.select_dtypes(include=["float64"]).columns
    od[float_columns] = od[float_columns].astype(np.float32)
    od.to_parquet(BASE_OD_OUTPUT, index=False, compression="snappy")

    diagnostics = {
        "source": str(ACTIVE_OD_FILE),
        "rows": int(len(od)),
        "active_origin_time_groups": int(od["group_id"].nunique()),
        "origins": int(od["i"].nunique()),
        "destinations": int(od["j"].nunique()),
        "time_min": str(od["t"].min()),
        "time_max": str(od["t"].max()),
        "price_cv_floor": cv_floor,
        "price_floor_applied_rows": int(od["pe_dispersion_floor_applied"].sum()),
        "split_rows": {str(k): int(v) for k, v in od["split"].value_counts().items()},
        "zero_demand_rows": int((od["D_ijt_observed"] <= 0).sum()),
        "removed_nonpositive_price_rows": nonpositive_price_rows,
        "removed_nonpositive_price_orders": nonpositive_price_orders,
    }
    DATA_DIAGNOSTICS.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    print(f"输出：{BASE_OD_OUTPUT}")


if __name__ == "__main__":
    main()
