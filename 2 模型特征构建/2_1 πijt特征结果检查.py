# -*- coding: utf-8 -*-
"""
3 πijt特征结果查看.py

功能：
1. 随机输出5个 (i,t) 样本；
2. 输出一个指定 (i,t) 样本；
3. 同时输出标量 Parquet 特征和10个 npz 中对应 row_id 的801维向量；
4. 801维向量只显示非零目的地及比例，未显示的目的地比例均为0；
5. available=0 表示该历史分布不可用，available=1 时未显示的目的地才表示真实比例为0。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from scipy import sparse


# ============================================================
# 1. 文件路径
# ============================================================

PROJECT_DIR = Path(
    "/Users/liangxue/Documents/8、Python文件-网约车数据/"
    "Riding hailing_Project"
)

RESULT_DIR = (
    PROJECT_DIR
    / "2 模型特征构建"
    / "2 πijt特征构造（结果）"
)

SCALAR_FILE = RESULT_DIR / "2 πijt特征构造（标量结果）.parquet"
DESTINATION_MAP_FILE = RESULT_DIR / "2_11 πijt目的地区域映射.csv"

NPZ_FILES = {
    "pi_lag1": RESULT_DIR / "2_1 pi_lag1.npz",
    "pi_lag2": RESULT_DIR / "2_2 pi_lag2.npz",
    "pi_roll4_mean": RESULT_DIR / "2_3 pi_roll4_mean.npz",
    "pi_roll8_mean": RESULT_DIR / "2_4 pi_roll8_mean.npz",
    "pi_same_slot_last_day": RESULT_DIR / "2_5 pi_same_slot_last_day.npz",
    "pi_same_slot_last_week": RESULT_DIR / "2_6 pi_same_slot_last_week.npz",
    "pi_daily_share_last_day": RESULT_DIR / "2_7 pi_daily_share_last_day.npz",
    "pi_daily_share_avg_last3": RESULT_DIR / "2_8 pi_daily_share_avg_last3.npz",
    "pi_daily_share_avg_last7": RESULT_DIR / "2_9 pi_daily_share_avg_last7.npz",
    "pi_ijt_label": RESULT_DIR / "2_10 pi_ijt_label.npz",
}


# ============================================================
# 2. 指定样本
# ============================================================

TARGET_I = "17031839100"
TARGET_T = "2024-06-06 18:15:00"


# ============================================================
# 3. 随机抽取5个 (i,t) 样本
# ============================================================

pf = pq.ParquetFile(SCALAR_FILE)

row_group_ids = pd.Series(
    range(pf.num_row_groups)
).sample(
    n=min(5, pf.num_row_groups),
    random_state=42,
).tolist()

random_samples = []

for row_group_id in row_group_ids:
    df = pf.read_row_group(
        row_group_id
    ).to_pandas()

    random_samples.append(
        df.sample(
            n=1,
            random_state=42,
        )
    )

random_samples = pd.concat(
    random_samples,
    ignore_index=True,
)


# ============================================================
# 4. 查询指定 (i,t) 样本
# ============================================================

dataset = ds.dataset(
    SCALAR_FILE,
    format="parquet",
)

target_table = dataset.to_table(
    filter=(
        (ds.field("i") == str(TARGET_I))
        & (
            ds.field("t")
            == pd.Timestamp(TARGET_T).to_pydatetime()
        )
    )
)

target_sample = target_table.to_pandas()


# ============================================================
# 5. 合并需要查看的样本并读取对应801维向量
# ============================================================

samples = pd.concat(
    [
        random_samples,
        target_sample,
    ],
    ignore_index=True,
)

row_ids = samples["row_id"].to_numpy(
    dtype=np.int64
)

destination_map = pd.read_csv(
    DESTINATION_MAP_FILE,
    dtype={"region_id": str},
)

destination_regions = destination_map[
    "region_id"
].to_numpy()

vector_results = {}

for feature_name, file_path in NPZ_FILES.items():
    matrix = sparse.load_npz(
        file_path
    ).tocsr()

    vector_results[feature_name] = (
        matrix[row_ids]
        .toarray()
        .astype(np.float32)
    )

    del matrix


# ============================================================
# 6. 输出函数
# ============================================================

def print_vector(feature_name, vector):
    """只输出801维向量中的非零目的地及其比例。"""

    nonzero_index = np.flatnonzero(
        vector != 0
    )

    print(f"\n{feature_name}：")

    if len(nonzero_index) == 0:
        print("全0")
        return

    for index in nonzero_index:
        print(
            f"{destination_regions[index]} = "
            f"{vector[index]:.6f}"
        )


def print_one_sample(row, position):
    """输出一个 (i,t) 样本的全部标量特征和801维向量特征。"""

    print("\n" + "=" * 80)
    print(
        f"row_id={int(row['row_id'])}，"
        f"i={row['i']}，"
        f"t={row['t']}"
    )

    print("\n标量特征：")
    print(
        row.to_frame()
        .to_string(header=False)
    )

    for feature_name in NPZ_FILES:
        print_vector(
            feature_name,
            vector_results[
                feature_name
            ][position],
        )


# ============================================================
# 7. 输出随机5个样本
# ============================================================

print("\n随机5个样本")

for position in range(len(random_samples)):
    print_one_sample(
        samples.iloc[position],
        position,
    )


# ============================================================
# 8. 输出指定样本
# ============================================================

print("\n指定样本")

target_position = len(random_samples)

print_one_sample(
    samples.iloc[target_position],
    target_position,
)
