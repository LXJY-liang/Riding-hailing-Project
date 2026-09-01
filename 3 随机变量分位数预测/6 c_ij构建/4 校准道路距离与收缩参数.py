# -*- coding: utf-8 -*-
"""利用高样本 OD 校准地图距离，并估计稀疏 OD 的收缩强度。

区域间校准模型：
    d_hist_ij = a_out + b_out * d_map_ij + error
区域内校准模型（输入和输出均为英里）：
    d_hist_ii = a_in + b_in * d_intra_map_i + error
稀疏OD收缩模型：
    d_hat = [N/(N+kappa)] * d_hist + [kappa/(N+kappa)] * d_prior

本脚本只输出calibration_parameters.json。最终六层距离和成本由第5个脚本计算。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize_scalar


def load_config():
    path = Path(__file__).resolve().parent / "1 c_ij构建配置.py"
    spec = importlib.util.spec_from_file_location("cij_config", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


cfg = load_config()


def fit_robust_linear(x: np.ndarray, y: np.ndarray, n: np.ndarray) -> tuple[float, float]:
    """估计带非负约束的 y=a+b*x 稳健回归。

    x是以英里计的地图距离或空间先验，y是以英里计的历史里程中位数。
    因此重新运行本脚本后，截距a的单位为英里，斜率b为英里/英里。
    权重随N增加，但在200处封顶，避免少数超大OD完全支配结果。
    Huber损失用于减少个别特殊OD对回归系数的影响。
    """

    weights = np.sqrt(np.minimum(n.astype(np.float64), 200.0))
    slope_start = max(float(np.median(y / np.maximum(x, 1e-6))), 0.1)
    intercept_start = max(float(np.median(y - slope_start * x)), 0.0)

    # a>=0、b>0：校准后距离不应因地图距离增加而下降。
    result = least_squares(
        lambda parameters: weights * (parameters[0] + parameters[1] * x - y),
        x0=np.asarray([intercept_start, slope_start]),
        bounds=([0.0, 1e-6], [np.inf, np.inf]),
        loss="huber",
        f_scale=0.5,
    )
    return float(result.x[0]), float(result.x[1])


def fit_kappa(
    sample_size: np.ndarray,
    sample_median: np.ndarray,
    prior: np.ndarray,
    truth: np.ndarray,
) -> float:
    """选择使伪稀疏样本收缩后 MAE 最小的 kappa。

    kappa越大，小样本历史中位数所占权重越小，距离越接近地图先验。
    目标值是高样本OD的全样本中位数，不需要人为指定收缩比例。
    """

    def objective(kappa: float) -> float:
        history_weight = sample_size / (sample_size + kappa)
        estimate = history_weight * sample_median + (1.0 - history_weight) * prior
        return float(np.mean(np.abs(estimate - truth)))

    result = minimize_scalar(
        objective,
        bounds=(0.1, 100.0),
        method="bounded",
        options={"xatol": 1e-3},
    )
    return float(result.x)


def main() -> None:
    cfg.ensure_output_directories()

    # 历史OD统计提供回归因变量；道路矩阵和区域内先验提供自变量。
    od = pd.read_parquet(cfg.HISTORICAL_OD_FILE)
    routing = pd.read_parquet(cfg.REGION_ROUTING_NODE_FILE)
    intra = pd.read_parquet(cfg.INTRA_ZONE_PRIOR_FILE)
    distance_matrix = np.load(cfg.ROAD_DISTANCE_MATRIX_FILE)
    region_to_index = dict(
        zip(routing["region_id"].astype(str), routing["region_index"].astype(int))
    )

    od["i_index"] = od["i"].map(region_to_index).astype("int16")
    od["j_index"] = od["j"].map(region_to_index).astype("int16")
    od["d_map_miles"] = distance_matrix[
        od["i_index"].to_numpy(),
        od["j_index"].to_numpy(),
    ]

    # 只使用 N_ij >= 20 的区域间OD来校准质心道路最短距离。
    high_inter = od.loc[
        (od["i"] != od["j"])
        & (od["N_ij"] >= cfg.DIRECT_HISTORY_THRESHOLD)
    ]
    a_out, b_out = fit_robust_linear(
        high_inter["d_map_miles"].to_numpy(dtype=np.float64),
        high_inter["d_hist_median_miles"].to_numpy(dtype=np.float64),
        high_inter["N_ij"].to_numpy(dtype=np.float64),
    )

    # i=j时不能使用质心到自身的0距离，因此分别校准区域内路网距离
    # 和等面积半径。两个自变量均由脚本3转换为英里；后者只在某区域
    # 无法得到路网先验时作为回退。b_in_network会在此处重新拟合，
    # 不再沿用错误米单位下得到的旧系数。
    high_intra = od.loc[
        (od["i"] == od["j"])
        & (od["N_ij"] >= cfg.DIRECT_HISTORY_THRESHOLD)
    ].merge(intra, left_on="i", right_on="region_id", how="left")

    network_rows = high_intra["d_intra_network_miles"].notna()
    a_in_network, b_in_network = fit_robust_linear(
        high_intra.loc[network_rows, "d_intra_network_miles"].to_numpy(
            dtype=np.float64
        ),
        high_intra.loc[network_rows, "d_hist_median_miles"].to_numpy(
            dtype=np.float64
        ),
        high_intra.loc[network_rows, "N_ij"].to_numpy(dtype=np.float64),
    )
    a_in_radius, b_in_radius = fit_robust_linear(
        high_intra["effective_radius_miles"].to_numpy(dtype=np.float64),
        high_intra["d_hist_median_miles"].to_numpy(dtype=np.float64),
        high_intra["N_ij"].to_numpy(dtype=np.float64),
    )

    # 区域间和区域内的尺度、样本量与误差结构不同，因此分别估计 kappa。
    samples = pd.read_parquet(cfg.SHRINKAGE_SAMPLE_FILE)
    samples["i_index"] = samples["i"].map(region_to_index).astype("int16")
    samples["j_index"] = samples["j"].map(region_to_index).astype("int16")

    inter_samples = samples.loc[samples["i"] != samples["j"]].copy()
    inter_map = distance_matrix[
        inter_samples["i_index"].to_numpy(),
        inter_samples["j_index"].to_numpy(),
    ]
    inter_prior = a_out + b_out * inter_map
    kappa_out = fit_kappa(
        inter_samples["sample_size"].to_numpy(dtype=np.float64),
        inter_samples["d_sample_median_miles"].to_numpy(dtype=np.float64),
        inter_prior.astype(np.float64),
        inter_samples["d_truth_median_miles"].to_numpy(dtype=np.float64),
    )

    # 区域内伪稀疏样本优先使用校准后路网先验，无路网样本时使用半径先验。
    intra_samples = samples.loc[samples["i"] == samples["j"]].merge(
        intra,
        left_on="i",
        right_on="region_id",
        how="left",
    )
    intra_prior = np.where(
        intra_samples["d_intra_network_miles"].notna(),
        a_in_network
        + b_in_network
        * intra_samples["d_intra_network_miles"].to_numpy(dtype=np.float64),
        a_in_radius
        + b_in_radius
        * intra_samples["effective_radius_miles"].to_numpy(dtype=np.float64),
    )
    kappa_in = fit_kappa(
        intra_samples["sample_size"].to_numpy(dtype=np.float64),
        intra_samples["d_sample_median_miles"].to_numpy(dtype=np.float64),
        intra_prior,
        intra_samples["d_truth_median_miles"].to_numpy(dtype=np.float64),
    )

    parameters = {
        "history_end": cfg.HISTORY_END,
        "distance_unit": "mile",
        "direct_history_threshold": cfg.DIRECT_HISTORY_THRESHOLD,
        "interzone": {"a": a_out, "b": b_out, "kappa": kappa_out},
        "intrazone_network": {"a": a_in_network, "b": b_in_network},
        "intrazone_radius": {"a": a_in_radius, "b": b_in_radius},
        "intrazone_kappa": kappa_in,
    }
    cfg.CALIBRATION_FILE.write_text(
        json.dumps(parameters, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
