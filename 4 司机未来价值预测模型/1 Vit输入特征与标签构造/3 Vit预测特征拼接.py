# -*- coding: utf-8 -*-
"""
3 Vit预测特征拼接.py

功能：
1. 分别构造Vit的train、valid、test输入特征；
2. 合并Dit、Sit的101维分位数预测并进行压缩；
3. 根据Dit、Sit中位数预测计算司机获得订单概率a_it；
4. 从πijt预测结果提取Top-10条件目的地分布；
5. 为Top-10目的地匹配tau_ijt的q0.10、q0.50、q0.90；
6. 加入所有直接邻近区域的Dit、Sit预测状态。
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# 1. 路径与参数
# ============================================================

project_root = Path(__file__).resolve().parents[2]

code_dir = (
    project_root
    / "4 司机未来价值预测模型"
    / "1 Vit输入特征与标签构造"
)

base_feature_file = (
    code_dir
    / "2 Vit基础特征构造（结果）.parquet"
)

neighbor_file = (
    project_root
    / "1 数据清洗"
    / "4 邻近区域关系表构造结果.parquet"
)

dit_prediction_dir = (
    project_root
    / "3 随机变量分位数预测"
    / "1 Dit预测"
    / "output"
    / "prediction"
)

sit_prediction_dir = (
    project_root
    / "3 随机变量分位数预测"
    / "3 Sit预测"
    / "output"
    / "prediction"
)

pi_output_dir = (
    project_root
    / "3 随机变量分位数预测"
    / "2 πijt预测"
    / "output"
)

pi_scalar_file = (
    project_root
    / "2 模型特征构建"
    / "2 πijt特征构造（结果）"
    / "2 πijt特征构造（标量结果）.parquet"
)

destination_map_file = (
    project_root
    / "2 模型特征构建"
    / "2 πijt特征构造（结果）"
    / "2_11 πijt目的地区域映射.csv"
)

tau_prediction_dir = (
    project_root
    / "3 随机变量分位数预测"
    / "4 tauijt预测"
    / "output"
    / "prediction"
)

output_files = {
    "oof": code_dir / "Vit_train_feature.parquet",
    "valid": code_dir / "Vit_valid_feature.parquet",
    "test": code_dir / "Vit_test_feature.parquet",
}

top_k = 10
pi_chunk_size = 50_000

quantile_levels = np.asarray(
    [0.005]
    + [level / 100 for level in range(1, 100)]
    + [0.995],
    dtype=np.float32,
)

quantile_columns = [
    f"q_{level:.3f}"
    for level in quantile_levels
]

key_quantiles = [
    0.005,
    0.05,
    0.10,
    0.50,
    0.90,
    0.95,
    0.995,
]

neighbor_quantiles = [
    0.10,
    0.50,
    0.90,
]


# ============================================================
# 2. 读取随机变量预测结果
# ============================================================

def read_prediction_parts(folder, split_name):
    """读取指定数据划分的连续随机变量预测结果。"""

    files = sorted(
        folder.glob(
            f"{split_name}*predictions*.parquet"
        )
    )

    return pd.concat(
        [
            pd.read_parquet(file)
            for file in files
        ],
        ignore_index=True,
    )


# ============================================================
# 3. 压缩101维分位数
# ============================================================

def summarize_quantiles(data, id_column, prefix):
    """将101维分位数压缩为20个5%区间均值和7个关键分位数。"""

    values = data[
        quantile_columns
    ].to_numpy(dtype=np.float32)

    result = data[
        [id_column, "t"]
    ].copy()

    for bin_index in range(20):
        lower = bin_index * 0.05
        upper = (bin_index + 1) * 0.05

        mask = (
            (quantile_levels > lower)
            & (quantile_levels <= upper)
        )

        result[
            f"{prefix}_qbin_"
            f"{int(lower * 100):02d}_"
            f"{int(upper * 100):02d}_mean"
        ] = values[:, mask].mean(axis=1)

    for level in key_quantiles:
        index = int(
            np.argmin(
                np.abs(
                    quantile_levels - level
                )
            )
        )

        result[
            f"{prefix}_q_{level:.3f}"
        ] = values[:, index]

    result[id_column] = result[
        id_column
    ].astype("string")

    result["t"] = pd.to_datetime(
        result["t"]
    )

    return result


def select_quantiles(data, id_column, prefix):
    """提取邻近区域使用的q0.10、q0.50、q0.90。"""

    result = data[
        [id_column, "t"]
    ].copy()

    for level in neighbor_quantiles:
        result[
            f"{prefix}_q_{level:.2f}"
        ] = data[
            f"q_{level:.3f}"
        ].astype("float32")

    result[id_column] = result[
        id_column
    ].astype("string")

    result["t"] = pd.to_datetime(
        result["t"]
    )

    return result


# ============================================================
# 4. πijt Top-K目的地
# ============================================================

pi_scalar = pd.read_parquet(
    pi_scalar_file,
    columns=[
        "row_id",
        "i",
        "t",
    ],
)

pi_scalar["i"] = pi_scalar[
    "i"
].astype("string")

pi_scalar["t"] = pd.to_datetime(
    pi_scalar["t"]
)

pi_scalar = pi_scalar.set_index(
    "row_id"
)

destination_map = pd.read_csv(
    destination_map_file,
    dtype={"region_id": "string"},
).sort_values(
    "destination_index"
)

destination_regions = destination_map[
    "region_id"
].to_numpy()

region_to_destination_index = {
    region_id: index
    for index, region_id
    in enumerate(destination_regions)
}


def build_pi_topk(split_name):
    """构造指定数据划分的Top-10条件目的地分布。"""

    predictions = np.load(
        pi_output_dir
        / f"{split_name}_pi_predictions.npy",
        mmap_mode="r",
    )

    row_ids = np.load(
        pi_output_dir
        / f"{split_name}_prediction_row_ids.npy",
        mmap_mode="r",
    )

    parts = []

    for start in range(
        0,
        len(row_ids),
        pi_chunk_size,
    ):
        end = min(
            start + pi_chunk_size,
            len(row_ids),
        )

        chunk_predictions = np.asarray(
            predictions[start:end],
            dtype=np.float32,
        )

        chunk_row_ids = np.asarray(
            row_ids[start:end],
            dtype=np.int64,
        )

        meta = pi_scalar.loc[
            chunk_row_ids
        ].reset_index()

        top_index = np.argpartition(
            -chunk_predictions,
            kth=top_k - 1,
            axis=1,
        )[:, :top_k]

        top_probability = np.take_along_axis(
            chunk_predictions,
            top_index,
            axis=1,
        )

        order = np.argsort(
            -top_probability,
            axis=1,
        )

        top_index = np.take_along_axis(
            top_index,
            order,
            axis=1,
        )

        top_probability = np.take_along_axis(
            top_probability,
            order,
            axis=1,
        )

        current_region_index = (
            meta["i"]
            .map(region_to_destination_index)
            .to_numpy(dtype=np.int32)
        )

        same_region_probability = (
            chunk_predictions[
                np.arange(
                    len(chunk_predictions)
                ),
                current_region_index,
            ]
        )

        part = pd.DataFrame(
            {
                "region_id": meta["i"].to_numpy(),
                "t": meta["t"].to_numpy(),
                "pi_top10_mass": top_probability.sum(axis=1),
                "pi_same_region": same_region_probability,
                "pi_other_mass": np.maximum(
                    1.0 - top_probability.sum(axis=1),
                    0.0,
                ),
            }
        )

        for rank in range(top_k):
            part[
                f"dest_{rank + 1}_region_id"
            ] = destination_regions[
                top_index[:, rank]
            ]

            part[
                f"dest_{rank + 1}_pi"
            ] = top_probability[:, rank]

        parts.append(part)

    result = pd.concat(
        parts,
        ignore_index=True,
    )

    result["region_id"] = result[
        "region_id"
    ].astype("string")

    result["t"] = pd.to_datetime(
        result["t"]
    )

    return result


# ============================================================
# 5. 构造单个数据划分
# ============================================================

def build_split(split_name):
    """构造Vit的一个数据划分。"""

    dit_raw = read_prediction_parts(
        dit_prediction_dir,
        split_name,
    )

    sit_raw = read_prediction_parts(
        sit_prediction_dir,
        split_name,
    )

    tau_raw = read_prediction_parts(
        tau_prediction_dir,
        split_name,
    )

    dit_summary = summarize_quantiles(
        dit_raw,
        "region_id",
        "D",
    )

    sit_summary = summarize_quantiles(
        sit_raw,
        "region_id",
        "S",
    )

    # 使用D、S中位数预测计算司机在当前15min至少获得一笔订单的概率
    order_ratio = (
        np.maximum(
            dit_summary[
                "D_q_0.500"
            ].to_numpy(dtype=np.float32),
            0.0,
        )
        / np.maximum(
            sit_summary[
                "S_q_0.500"
            ].to_numpy(dtype=np.float32),
            1e-6,
        )
    )

    a_it = (
        1.0 - np.exp(-order_ratio)
    ).astype(np.float32)

    ait_state = dit_summary[
        ["region_id", "t"]
    ].copy()

    ait_state["a_it"] = a_it
    ait_state["one_minus_a_it"] = (
        1.0 - a_it
    ).astype(np.float32)

    dit_neighbor = select_quantiles(
        dit_raw,
        "region_id",
        "D",
    )

    sit_neighbor = select_quantiles(
        sit_raw,
        "region_id",
        "S",
    )

    neighbor_state = dit_neighbor.merge(
        sit_neighbor,
        on=["region_id", "t"],
        how="outer",
    )

    pi_topk = build_pi_topk(
        split_name
    )

    tau_small = tau_raw[
        [
            "i",
            "j",
            "t",
            "q_0.100",
            "q_0.500",
            "q_0.900",
        ]
    ].copy()

    tau_small["i"] = tau_small[
        "i"
    ].astype("string")

    tau_small["j"] = tau_small[
        "j"
    ].astype("string")

    tau_small["t"] = pd.to_datetime(
        tau_small["t"]
    )

    for rank in range(
        1,
        top_k + 1,
    ):
        lookup = pi_topk[
            [
                "region_id",
                "t",
                f"dest_{rank}_region_id",
            ]
        ].rename(
            columns={
                "region_id": "i",
                f"dest_{rank}_region_id": "j",
            }
        )

        lookup = lookup.merge(
            tau_small,
            on=["i", "j", "t"],
            how="left",
        )

        pi_topk[
            f"dest_{rank}_tau_q0.10"
        ] = lookup[
            "q_0.100"
        ].to_numpy()

        pi_topk[
            f"dest_{rank}_tau_q0.50"
        ] = lookup[
            "q_0.500"
        ].to_numpy()

        pi_topk[
            f"dest_{rank}_tau_q0.90"
        ] = lookup[
            "q_0.900"
        ].to_numpy()

    result = pd.read_parquet(
        base_feature_file
    )

    result["region_id"] = result[
        "region_id"
    ].astype("string")

    result["t"] = pd.to_datetime(
        result["t"]
    )

    result = result.merge(
        dit_summary,
        on=["region_id", "t"],
        how="inner",
    )

    result = result.merge(
        sit_summary,
        on=["region_id", "t"],
        how="left",
    )

    result = result.merge(
        ait_state,
        on=["region_id", "t"],
        how="left",
    )

    result = result.merge(
        pi_topk,
        on=["region_id", "t"],
        how="left",
    )

    neighbor = pd.read_parquet(
        neighbor_file,
        columns=[
            "region_id",
            "neighbor_region_id",
            "weight",
        ],
    )

    neighbor["region_id"] = neighbor[
        "region_id"
    ].astype("string")

    neighbor["neighbor_region_id"] = neighbor[
        "neighbor_region_id"
    ].astype("string")

    neighbor = neighbor.sort_values(
        [
            "region_id",
            "weight",
            "neighbor_region_id",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    )

    neighbor["neighbor_rank"] = (
        neighbor.groupby(
            "region_id"
        ).cumcount()
        + 1
    )

    max_neighbors = int(
        neighbor[
            "neighbor_rank"
        ].max()
    )

    for rank in range(
        1,
        max_neighbors + 1,
    ):
        mapping = neighbor.loc[
            neighbor[
                "neighbor_rank"
            ] == rank,
            [
                "region_id",
                "neighbor_region_id",
            ],
        ]

        result = result.merge(
            mapping,
            on="region_id",
            how="left",
        )

        result = result.rename(
            columns={
                "neighbor_region_id":
                    f"neighbor_{rank}_region_id"
            }
        )

        state = neighbor_state.rename(
            columns={
                "region_id":
                    f"neighbor_{rank}_region_id",
                "D_q_0.10":
                    f"neighbor_{rank}_D_q0.10",
                "D_q_0.50":
                    f"neighbor_{rank}_D_q0.50",
                "D_q_0.90":
                    f"neighbor_{rank}_D_q0.90",
                "S_q_0.10":
                    f"neighbor_{rank}_S_q0.10",
                "S_q_0.50":
                    f"neighbor_{rank}_S_q0.50",
                "S_q_0.90":
                    f"neighbor_{rank}_S_q0.90",
            }
        )

        result = result.merge(
            state,
            on=[
                f"neighbor_{rank}_region_id",
                "t",
            ],
            how="left",
        )

    result.to_parquet(
        output_files[split_name],
        index=False,
        compression="snappy",
    )


# ============================================================
# 6. 执行
# ============================================================

for split_name in [
    "oof",
    "valid",
    "test",
]:
    build_split(split_name)
