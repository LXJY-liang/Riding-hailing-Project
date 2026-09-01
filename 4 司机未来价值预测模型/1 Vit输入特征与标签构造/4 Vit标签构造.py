# -*- coding: utf-8 -*-
"""
4 Vit标签构造.py

功能：
1. 使用历史真实D_it和历史供给代理S_it计算a_it_hist；
2. 使用过去1小时每单司机净收益均值作为r_it的历史收益代理；
3. 构造未来4小时（16个15min时段）的折现期望收益；
4. 输出Vit预训练阶段使用的单值标签。
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. 路径与参数
# ============================================================

project_root = Path(
    "/Users/liangxue/Documents/8、Python文件-网约车数据/Riding hailing_Project"
)

code_dir = (
    project_root
    / "4 司机未来价值预测模型"
    / "1 Vit输入特征与标签构造"
)

profit_feature_file = (
    code_dir
    / "1 司机历史收益特征构造（结果）.parquet"
)

dit_feature_file = (
    project_root
    / "2 模型特征构建"
    / "1 Dit特征构建（结果）.parquet"
)

sit_feature_file = (
    project_root
    / "2 模型特征构建"
    / "3 Sit特征构造（结果）.parquet"
)

output_file = (
    code_dir
    / "4 Vit标签构造（结果）.parquet"
)

reward_column = "driver_profit_per_order_i_roll4"
gamma = 0.95
future_steps = 16

# ============================================================
# 2. 读取历史收益、真实需求和供给代理
# ============================================================

profit = pd.read_parquet(
    profit_feature_file,
    columns=[
        "region_id",
        "t",
        reward_column,
    ],
)

demand = pd.read_parquet(
    dit_feature_file,
    columns=[
        "region_id",
        "t",
        "D_it_label",
    ],
)

supply = pd.read_parquet(
    sit_feature_file,
    columns=[
        "region_id",
        "t",
        "S_it_label",
    ],
)

for data in [profit, demand, supply]:
    data["region_id"] = data["region_id"].astype("string")
    data["t"] = pd.to_datetime(data["t"])


df = profit.merge(
    demand,
    on=["region_id", "t"],
    how="inner",
)

df = df.merge(
    supply,
    on=["region_id", "t"],
    how="inner",
)

df = df.sort_values(
    ["region_id", "t"]
).reset_index(drop=True)


# ============================================================
# 3. 构造历史获得订单概率和时段期望收益
# ============================================================

order_ratio = (
    np.maximum(
        df["D_it_label"].to_numpy(dtype=np.float32),
        0.0,
    )
    / np.maximum(
        df["S_it_label"].to_numpy(dtype=np.float32),
        1e-6,
    )
)

df["a_it_hist"] = (
    1.0 - np.exp(-order_ratio)
).astype(np.float32)

df["expected_reward_proxy"] = (
    df["a_it_hist"]
    * df[reward_column]
).astype(np.float32)


# ============================================================
# 4. 构造未来4小时折现收益标签
# ============================================================

grouped = df.groupby("region_id")[
    "expected_reward_proxy"
]

future_rewards = pd.concat(
    [
        grouped.shift(-h) * (gamma ** h)
        for h in range(future_steps)
    ],
    axis=1,
)

df["Vit_label"] = future_rewards.sum(
    axis=1,
    min_count=future_steps,
)


# ============================================================
# 5. 保存有效标签
# ============================================================

result = df[
    [
        "region_id",
        "t",
        "Vit_label",
    ]
].dropna(
    subset=["Vit_label"]
)

result.to_parquet(
    output_file,
    index=False,
    compression="snappy",
)
