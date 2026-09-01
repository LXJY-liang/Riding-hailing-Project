# -*- coding: utf-8 -*-
"""根据六层规则生成固定 c_ij 表和801×801成本矩阵。

六层规则：
1. i!=j, N>=20：历史里程中位数；
2. i!=j, 1<=N<20：历史中位数向校准后道路先验收缩；
3. i!=j, N=0：校准后道路先验；
4. i=j, N>=20：历史里程中位数；
5. i=j, 1<=N<20：历史中位数向校准后区域内先验收缩；
6. i=j, N=0：校准后区域内先验。

本脚本不读取原始订单、不运行最短路、不重新拟合回归，因此最终计算较快。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_config():
    path = Path(__file__).resolve().parent / "1 c_ij构建配置.py"
    spec = importlib.util.spec_from_file_location("cij_config", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


cfg = load_config()


def main() -> None:
    cfg.ensure_output_directories()

    # 所有前期结果已按region_index对齐，最终阶段只做数组索引和向量运算。
    routing = pd.read_parquet(cfg.REGION_ROUTING_NODE_FILE).sort_values(
        "region_index"
    )
    historical = pd.read_parquet(cfg.HISTORICAL_OD_FILE)
    intra = pd.read_parquet(cfg.INTRA_ZONE_PRIOR_FILE).sort_values("region_index")
    road_distance = np.load(cfg.ROAD_DISTANCE_MATRIX_FILE).astype(np.float64)
    parameters = json.loads(cfg.CALIBRATION_FILE.read_text(encoding="utf-8"))

    region_ids = routing["region_id"].astype(str).to_numpy()
    region_to_index = {
        region_id: index for index, region_id in enumerate(region_ids)
    }
    size = len(region_ids)
    # 按 i 外层、j 内层展开完整801×801空间，行号为 i_index*801+j_index。
    i_index = np.repeat(np.arange(size, dtype=np.int16), size)
    j_index = np.tile(np.arange(size, dtype=np.int16), size)

    counts = np.zeros(size * size, dtype=np.int32)
    history_median = np.full(size * size, np.nan, dtype=np.float64)
    # 历史表只包含N_ij>0的OD；未匹配的组合自动保持N_ij=0。
    historical_i = historical["i"].map(region_to_index).to_numpy(dtype=np.int64)
    historical_j = historical["j"].map(region_to_index).to_numpy(dtype=np.int64)
    historical_position = historical_i * size + historical_j
    counts[historical_position] = historical["N_ij"].to_numpy(dtype=np.int32)
    history_median[historical_position] = historical[
        "d_hist_median_miles"
    ].to_numpy(dtype=np.float64)

    same_region = i_index == j_index
    map_distance = road_distance.reshape(-1).copy()

    # i!=j的地图先验为 a_out+b_out*d_map_ij。
    interzone = parameters["interzone"]
    prior = interzone["a"] + interzone["b"] * map_distance

    # 两个区域内空间变量均已由脚本3转换为英里。
    intra_network = intra["d_intra_network_miles"].to_numpy(dtype=np.float64)
    intra_radius = intra["effective_radius_miles"].to_numpy(dtype=np.float64)
    # i=j优先使用区域内路网抽样距离；路网先验不可得时使用半径回退模型。
    raw_intra = np.where(np.isfinite(intra_network), intra_network, intra_radius)
    calibrated_intra = np.where(
        np.isfinite(intra_network),
        parameters["intrazone_network"]["a"]
        + parameters["intrazone_network"]["b"] * intra_network,
        parameters["intrazone_radius"]["a"]
        + parameters["intrazone_radius"]["b"] * intra_radius,
    )
    diagonal_positions = np.arange(size, dtype=np.int64) * (size + 1)
    map_distance[diagonal_positions] = raw_intra
    prior[diagonal_positions] = calibrated_intra
    prior = np.maximum(prior, 1e-6)

    # 历史充足、历史稀疏和无历史三种样本状态共同决定六层分类。
    high_sample = counts >= cfg.DIRECT_HISTORY_THRESHOLD
    sparse_sample = (counts > 0) & ~high_sample
    shrinkage_weight = np.zeros(size * size, dtype=np.float64)
    shrinkage_weight[high_sample] = 1.0

    # lambda=N/(N+kappa)。区域间和区域内使用各自的数据驱动kappa。
    inter_sparse = sparse_sample & ~same_region
    intra_sparse = sparse_sample & same_region
    shrinkage_weight[inter_sparse] = counts[inter_sparse] / (
        counts[inter_sparse] + interzone["kappa"]
    )
    shrinkage_weight[intra_sparse] = counts[intra_sparse] / (
        counts[intra_sparse] + parameters["intrazone_kappa"]
    )

    # 高样本直接用历史中位数；稀疏样本做加权收缩；无样本直接用空间先验。
    estimated_distance = prior.copy()
    estimated_distance[high_sample] = history_median[high_sample]
    estimated_distance[sparse_sample] = (
        shrinkage_weight[sparse_sample] * history_median[sparse_sample]
        + (1.0 - shrinkage_weight[sparse_sample]) * prior[sparse_sample]
    )
    # 距离单位是英里，c_ij单位是美元。
    cost = cfg.COST_PER_MILE * estimated_distance

    # 层级编号1—3对应i!=j，4—6对应i=j。
    level = np.empty(size * size, dtype=np.int8)
    level[(~same_region) & high_sample] = 1
    level[(~same_region) & sparse_sample] = 2
    level[(~same_region) & (counts == 0)] = 3
    level[same_region & high_sample] = 4
    level[same_region & sparse_sample] = 5
    level[same_region & (counts == 0)] = 6

    result = pd.DataFrame(
        {
            "i": region_ids[i_index],
            "j": region_ids[j_index],
            "i_index": i_index,
            "j_index": j_index,
            "N_ij": counts,
            "estimation_level": level,
            "d_hist_median_miles": history_median.astype(np.float32),
            "d_map_miles": map_distance.astype(np.float32),
            "d_prior_miles": prior.astype(np.float32),
            "shrinkage_weight": shrinkage_weight.astype(np.float32),
            "d_ij_miles": estimated_distance.astype(np.float32),
            "c_ij_usd": cost.astype(np.float32),
        }
    )
    result.to_parquet(
        cfg.CIJ_FILE,
        index=False,
        compression=cfg.PARQUET_COMPRESSION,
    )
    # 矩阵行列均按region_index排序，SAC可通过[i_index, j_index]直接取成本。
    np.save(cfg.CIJ_MATRIX_FILE, cost.reshape(size, size).astype(np.float32))


if __name__ == "__main__":
    main()
