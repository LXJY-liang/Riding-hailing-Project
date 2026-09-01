# -*- coding: utf-8 -*-
"""
2 πijt特征构造.py

功能：
1. 读取 OD-15min 面板；
2. 将样本单位改为 (i,t)，一行对应一个出发区域和一个15min时段；
3. 构造区域 i 的时间、天气和历史总需求标量特征；
4. 构造9组801维历史目的地比例特征；比如：过去一个时间段、两个时间段、一个小时等等
5. 构造801维当前真实目的地比例标签；
6. 使用 available 字段区分“历史不可用”和“历史比例为0”。

输出：
1. 2 πijt特征构造（标量结果）.parquet
    例如：
    row_id	i	t	day_index	slot_in_day	dow	temperature_2m	D_i_lag1	D_i_lag2	D_i_roll4_mean	pi_lag1_available	pi_last_day_available	label_available
    0	17031010100	2024-06-01 00:00	1	0	5	18.2	NaN	NaN	NaN	0	0	1

2. 2_1 pi_lag1.npz
3. 2_2 pi_lag2.npz
4. 2_3 pi_roll4_mean.npz
5. 2_4 pi_roll8_mean.npz
6. 2_5 pi_same_slot_last_day.npz
7. 2_6 pi_same_slot_last_week.npz
8. 2_7 pi_daily_share_last_day.npz
9. 2_8 pi_daily_share_avg_last3.npz
10. 2_9 pi_daily_share_avg_last7.npz

11. 2_10 pi_ijt_label.npz

12. 2_11 πijt目的地区域映射.csv
    例如：
    destination_index,region_id
    0,17031010100
    1,17031010201
    2,17031010202
    3,17031010300
    4,17031010400
    ...
    800,xxxxxxxxxxx

说明：
1. 所有 .npz 的第 N 行都对应标量 Parquet 中 row_id=N 的同一个 (i,t)；
2. 所有历史特征只使用 t 之前的信息；
3. D_it_observed 和 pi_ijt_label 仅用于标签，不进入后续模型文本输入；
4. 801维向量使用 scipy CSR 稀疏矩阵保存。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy import sparse

# ============================================================
# 1. 路径设置
# ============================================================

PROJECT_DIR = Path(
    "/Users/liangxue/Documents/8、Python文件-网约车数据/"
    "Riding hailing_Project"
)

INPUT_FILE = (
    PROJECT_DIR
    / "1 数据清洗"
    / "2_3 OD-15min聚合（补全时间、不管没出现的OD）结果.parquet"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "2 模型特征构建"
    / "2 πijt特征构造（结果）"
)

SCALAR_FILE = OUTPUT_DIR / "2 πijt特征构造（标量结果）.parquet"

PI_LAG1_FILE = OUTPUT_DIR / "2_1 pi_lag1.npz"
PI_LAG2_FILE = OUTPUT_DIR / "2_2 pi_lag2.npz"
PI_ROLL4_FILE = OUTPUT_DIR / "2_3 pi_roll4_mean.npz"
PI_ROLL8_FILE = OUTPUT_DIR / "2_4 pi_roll8_mean.npz"
PI_LAST_DAY_FILE = OUTPUT_DIR / "2_5 pi_same_slot_last_day.npz"
PI_LAST_WEEK_FILE = OUTPUT_DIR / "2_6 pi_same_slot_last_week.npz"

PI_DAILY_LAST_DAY_FILE = OUTPUT_DIR / "2_7 pi_daily_share_last_day.npz"
PI_DAILY_AVG3_FILE = OUTPUT_DIR / "2_8 pi_daily_share_avg_last3.npz"
PI_DAILY_AVG7_FILE = OUTPUT_DIR / "2_9 pi_daily_share_avg_last7.npz"

PI_LABEL_FILE = OUTPUT_DIR / "2_10 pi_ijt_label.npz"
DESTINATION_MAP_FILE = OUTPUT_DIR / "2_11 πijt目的地区域映射.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. 稀疏矩阵历史函数
# ============================================================

def shift_sparse(matrix, step, group_size):
    """在每个区域内部向后移动 step 个时间位置。"""

    coo = matrix.tocoo()
    position = coo.row % group_size
    keep = position + step < group_size

    return sparse.coo_matrix(
        (
            coo.data[keep],
            (
                coo.row[keep] + step,
                coo.col[keep],
            ),
        ),
        shape=matrix.shape,
        dtype=np.float32,
    ).tocsr()


def shift_available(values, step, group_size):
    """移动历史可用性标记。"""

    values = values.reshape(-1, group_size)
    result = np.zeros_like(values, dtype=np.int8)

    if step < group_size:
        result[:, step:] = values[:, :-step]

    return result.reshape(-1)

def rolling_sparse_mean(matrix, available, window, group_size):
    """
    对过去 window 个位置的有效历史分布求均值。
    D_it=0 的历史时段不参与平均。
    """
    result = sparse.csr_matrix(
        matrix.shape,
        dtype=np.float32,
    )

    count = np.zeros(
        matrix.shape[0],
        dtype=np.int16,
    )

    for step in range(1, window + 1):
        result = result + shift_sparse(
            matrix,
            step,
            group_size,
        )

        count += shift_available(
            available,
            step,
            group_size,
        )

    inverse_count = np.zeros(
        matrix.shape[0],
        dtype=np.float32,
    )

    valid = count > 0
    inverse_count[valid] = 1.0 / count[valid]

    result = result.multiply(
        inverse_count[:, None]
    ).tocsr()

    return result, valid.astype(np.int8)


# ============================================================
# 3. 标量历史函数
# ============================================================

def shift_dense(values, step):
    """二维时间序列向后移动 step 个位置，历史不可用保留 NaN。"""

    result = np.full(
        values.shape,
        np.nan,
        dtype=np.float32,
    )

    if step < values.shape[1]:
        result[:, step:] = values[:, :-step]

    return result


def rolling_dense_mean(values, window):
    """计算当前时段之前 window 个位置的均值。"""

    cumulative = np.concatenate(
        [
            np.zeros(
                (values.shape[0], 1),
                dtype=np.float64,
            ),
            np.cumsum(
                values,
                axis=1,
                dtype=np.float64,
            ),
        ],
        axis=1,
    )

    end = np.arange(values.shape[1])
    start = np.maximum(0, end - window)

    total = cumulative[:, end] - cumulative[:, start]
    count = end - start

    result = np.full(
        values.shape,
        np.nan,
        dtype=np.float32,
    )

    valid = count > 0
    result[:, valid] = (
        total[:, valid] / count[valid]
    ).astype(np.float32)

    return result


# ============================================================
# 4. 读取区域集合
# ============================================================

pf = pq.ParquetFile(INPUT_FILE)

region_set = set()

for row_group_index in range(pf.num_row_groups):
    table = pf.read_row_group(
        row_group_index,
        columns=[
            "i",
            "j",
            "D_ijt_observed",
        ],
    )

    observed = pc.cast(
        table["D_ijt_observed"],
        pa.float64(),
    )

    table = table.filter(
        pc.greater(observed, 0)
    )

    if table.num_rows == 0:
        continue

    region_set.update(
        str(value)
        for value in table["i"].to_pylist()
    )

    region_set.update(
        str(value)
        for value in table["j"].to_pylist()
    )


REGIONS = sorted(region_set)

REGION_TO_INDEX = {
    region_id: index
    for index, region_id in enumerate(REGIONS)
}

NUM_REGIONS = len(REGIONS)


# ============================================================
# 5. 读取完整时间、天气和时间状态
# ============================================================

time_columns = [
    "t",
    "day_index",
    "slot_in_day",
    "dow",
    "is_weekend_or_holiday",
    "is_peak_morning",
    "is_peak_evening",
    "is_night",
    "temperature_2m",
    "relative_humidity_2m",
    "rain",
    "weather_code",
    "wind_speed_10m",
]

time_table = (
    pf.read_row_group(
        0,
        columns=time_columns,
    )
    .to_pandas()
    .drop_duplicates("t")
)

time_table["t"] = pd.to_datetime(
    time_table["t"]
)

time_table = (
    time_table
    .sort_values("t")
    .reset_index(drop=True)
)

TIMES = time_table["t"].to_numpy()

TIME_TO_INDEX = {
    pd.Timestamp(value): index
    for index, value in enumerate(TIMES)
}

NUM_TIMES = len(TIMES)
NUM_ROWS = NUM_REGIONS * NUM_TIMES


# ============================================================
# 6. 构造 (i,t) × j 的真实订单数稀疏矩阵
# ============================================================

row_parts = []
col_parts = []
value_parts = []

for row_group_index in range(pf.num_row_groups):
    table = pf.read_row_group(
        row_group_index,
        columns=[
            "i",
            "j",
            "t",
            "D_ijt_observed",
        ],
    )

    observed = pc.cast(
        table["D_ijt_observed"],
        pa.float64(),
    )

    table = table.filter(
        pc.greater(observed, 0)
    )

    if table.num_rows == 0:
        continue

    chunk = table.to_pandas()

    chunk["i"] = chunk["i"].astype(str)
    chunk["j"] = chunk["j"].astype(str)
    chunk["t"] = pd.to_datetime(chunk["t"])

    i_index = (
        chunk["i"]
        .map(REGION_TO_INDEX)
        .to_numpy(dtype=np.int32)
    )

    j_index = (
        chunk["j"]
        .map(REGION_TO_INDEX)
        .to_numpy(dtype=np.int32)
    )

    t_index = np.fromiter(
        (
            TIME_TO_INDEX[pd.Timestamp(value)]
            for value in chunk["t"]
        ),
        dtype=np.int32,
        count=len(chunk),
    )

    row_index = (
        i_index * NUM_TIMES
        + t_index
    )

    values = (
        pd.to_numeric(
            chunk["D_ijt_observed"],
            errors="coerce",
        )
        .to_numpy(dtype=np.float32)
    )

    row_parts.append(row_index)
    col_parts.append(j_index)
    value_parts.append(values)


count_matrix = sparse.coo_matrix(
    (
        np.concatenate(value_parts),
        (
            np.concatenate(row_parts),
            np.concatenate(col_parts),
        ),
    ),
    shape=(NUM_ROWS, NUM_REGIONS),
    dtype=np.float32,
).tocsr()

count_matrix.sum_duplicates()

del row_parts, col_parts, value_parts


# ============================================================
# 7. 当前真实目的地比例标签
# ============================================================

D_it = np.asarray(
    count_matrix.sum(axis=1)
).ravel().astype(np.float32)

pi_label_available = (
    D_it > 0
).astype(np.int8)

inverse_D = np.zeros(
    NUM_ROWS,
    dtype=np.float32,
)

inverse_D[pi_label_available == 1] = (
    1.0 / D_it[pi_label_available == 1]
)

pi_label = count_matrix.multiply(
    inverse_D[:, None]
).tocsr()

sparse.save_npz(
    PI_LABEL_FILE,
    pi_label,
    compressed=True,
)


# ============================================================
# 8. 15min历史目的地比例
# ============================================================

pi_lag1 = shift_sparse(
    pi_label,
    1,
    NUM_TIMES,
)

pi_lag2 = shift_sparse(
    pi_label,
    2,
    NUM_TIMES,
)

pi_roll4, pi_roll4_available = (
    rolling_sparse_mean(
        pi_label,
        pi_label_available,
        4,
        NUM_TIMES,
    )
)

pi_roll8, pi_roll8_available = (
    rolling_sparse_mean(
        pi_label,
        pi_label_available,
        8,
        NUM_TIMES,
    )
)

pi_last_day = shift_sparse(
    pi_label,
    96,
    NUM_TIMES,
)

pi_last_week = shift_sparse(
    pi_label,
    672,
    NUM_TIMES,
)

pi_lag1_available = shift_available(
    pi_label_available,
    1,
    NUM_TIMES,
)

pi_lag2_available = shift_available(
    pi_label_available,
    2,
    NUM_TIMES,
)

pi_last_day_available = shift_available(
    pi_label_available,
    96,
    NUM_TIMES,
)

pi_last_week_available = shift_available(
    pi_label_available,
    672,
    NUM_TIMES,
)

sparse.save_npz(
    PI_LAG1_FILE,
    pi_lag1,
    compressed=True,
)

sparse.save_npz(
    PI_LAG2_FILE,
    pi_lag2,
    compressed=True,
)

sparse.save_npz(
    PI_ROLL4_FILE,
    pi_roll4,
    compressed=True,
)

sparse.save_npz(
    PI_ROLL8_FILE,
    pi_roll8,
    compressed=True,
)

sparse.save_npz(
    PI_LAST_DAY_FILE,
    pi_last_day,
    compressed=True,
)

sparse.save_npz(
    PI_LAST_WEEK_FILE,
    pi_last_week,
    compressed=True,
)

del pi_lag1, pi_lag2, pi_roll4, pi_roll8
del pi_last_day, pi_last_week


# ============================================================
# 9. 构造每天的目的地比例
# ============================================================

time_dates = pd.to_datetime(
    TIMES
).normalize()

unique_dates = pd.Index(
    time_dates
).unique()

DATE_TO_INDEX = {
    date: index
    for index, date in enumerate(unique_dates)
}

day_index_for_time = np.array(
    [
        DATE_TO_INDEX[date]
        for date in time_dates
    ],
    dtype=np.int32,
)

NUM_DAYS = len(unique_dates)

count_coo = count_matrix.tocoo()

origin_index = (
    count_coo.row // NUM_TIMES
).astype(np.int32)

time_index = (
    count_coo.row % NUM_TIMES
).astype(np.int32)

daily_row_index = (
    origin_index * NUM_DAYS
    + day_index_for_time[time_index]
)

daily_count = sparse.coo_matrix(
    (
        count_coo.data,
        (
            daily_row_index,
            count_coo.col,
        ),
    ),
    shape=(
        NUM_REGIONS * NUM_DAYS,
        NUM_REGIONS,
    ),
    dtype=np.float32,
).tocsr()

daily_count.sum_duplicates()

daily_D_i = np.asarray(
    daily_count.sum(axis=1)
).ravel().astype(np.float32)

daily_share_available = (
    daily_D_i > 0
).astype(np.int8)

inverse_daily_D = np.zeros(
    daily_D_i.shape[0],
    dtype=np.float32,
)

inverse_daily_D[
    daily_share_available == 1
] = (
    1.0
    / daily_D_i[daily_share_available == 1]
)

daily_share = daily_count.multiply(
    inverse_daily_D[:, None]
).tocsr()


# ============================================================
# 10. 构造前一天、过去3天和过去7天日均目的地比例
# ============================================================

daily_share_last_day = shift_sparse(
    daily_share,
    1,
    NUM_DAYS,
)

daily_share_avg3, daily_share_avg3_available = (
    rolling_sparse_mean(
        daily_share,
        daily_share_available,
        3,
        NUM_DAYS,
    )
)

daily_share_avg7, daily_share_avg7_available = (
    rolling_sparse_mean(
        daily_share,
        daily_share_available,
        7,
        NUM_DAYS,
    )
)

daily_share_last_day_available = shift_available(
    daily_share_available,
    1,
    NUM_DAYS,
)


# ============================================================
# 11. 将每天的比例扩展到对应当天全部 (i,t)
# ============================================================

origin_ids = np.repeat(
    np.arange(NUM_REGIONS, dtype=np.int32),
    NUM_TIMES,
)

time_ids = np.tile(
    np.arange(NUM_TIMES, dtype=np.int32),
    NUM_REGIONS,
)

daily_lookup = (
    origin_ids * NUM_DAYS
    + day_index_for_time[time_ids]
)

pi_daily_last_day = (
    daily_share_last_day[daily_lookup]
    .tocsr()
)

sparse.save_npz(
    PI_DAILY_LAST_DAY_FILE,
    pi_daily_last_day,
    compressed=True,
)

del pi_daily_last_day, daily_share_last_day

pi_daily_avg3 = (
    daily_share_avg3[daily_lookup]
    .tocsr()
)

sparse.save_npz(
    PI_DAILY_AVG3_FILE,
    pi_daily_avg3,
    compressed=True,
)

del pi_daily_avg3, daily_share_avg3

pi_daily_avg7 = (
    daily_share_avg7[daily_lookup]
    .tocsr()
)

sparse.save_npz(
    PI_DAILY_AVG7_FILE,
    pi_daily_avg7,
    compressed=True,
)

del pi_daily_avg7, daily_share_avg7


pi_daily_last_day_available = (
    daily_share_last_day_available[
        daily_lookup
    ]
).astype(np.int8)

pi_daily_avg3_available = (
    daily_share_avg3_available[
        daily_lookup
    ]
).astype(np.int8)

pi_daily_avg7_available = (
    daily_share_avg7_available[
        daily_lookup
    ]
).astype(np.int8)


# ============================================================
# 12. 区域 i 的历史总需求标量特征
# ============================================================

D_matrix = D_it.reshape(
    NUM_REGIONS,
    NUM_TIMES,
)

D_i_lag1 = shift_dense(
    D_matrix,
    1,
)

D_i_lag2 = shift_dense(
    D_matrix,
    2,
)

D_i_roll4_mean = rolling_dense_mean(
    D_matrix,
    4,
)

D_i_roll8_mean = rolling_dense_mean(
    D_matrix,
    8,
)

D_i_same_slot_last_day = shift_dense(
    D_matrix,
    96,
)

D_i_same_slot_last_week = shift_dense(
    D_matrix,
    672,
)


# ============================================================
# 13. 区域 i 的长期历史总需求
# ============================================================

daily_D_matrix = daily_D_i.reshape(
    NUM_REGIONS,
    NUM_DAYS,
)

D_i_total_last_day_daily = shift_dense(
    daily_D_matrix,
    1,
)

D_i_daily_avg_last3_daily = rolling_dense_mean(
    daily_D_matrix,
    3,
)

D_i_daily_avg_last7_daily = rolling_dense_mean(
    daily_D_matrix,
    7,
)

D_i_total_last_day = (
    D_i_total_last_day_daily[
        :,
        day_index_for_time,
    ]
)

D_i_daily_avg_last3 = (
    D_i_daily_avg_last3_daily[
        :,
        day_index_for_time,
    ]
)

D_i_daily_avg_last7 = (
    D_i_daily_avg_last7_daily[
        :,
        day_index_for_time,
    ]
)


# ============================================================
# 14. 保存标量特征 Parquet
# ============================================================

if SCALAR_FILE.exists():
    SCALAR_FILE.unlink()

writer = None

for region_index, region_id in enumerate(REGIONS):
    start = region_index * NUM_TIMES
    end = start + NUM_TIMES

    result = pd.DataFrame(
        {
            # 样本定位
            "row_id": np.arange(
                start,
                end,
                dtype=np.int64,
            ),
            "i": region_id,
            "t": time_table["t"].to_numpy(),

            # 时间特征
            "day_index": time_table[
                "day_index"
            ].to_numpy(),
            "slot_in_day": time_table[
                "slot_in_day"
            ].to_numpy(),
            "dow": time_table[
                "dow"
            ].to_numpy(),
            "is_weekend_or_holiday": time_table[
                "is_weekend_or_holiday"
            ].to_numpy(),
            "is_peak_morning": time_table[
                "is_peak_morning"
            ].to_numpy(),
            "is_peak_evening": time_table[
                "is_peak_evening"
            ].to_numpy(),
            "is_night": time_table[
                "is_night"
            ].to_numpy(),

            # 天气特征
            "temperature_2m": time_table[
                "temperature_2m"
            ].to_numpy(),
            "relative_humidity_2m": time_table[
                "relative_humidity_2m"
            ].to_numpy(),
            "rain": time_table[
                "rain"
            ].to_numpy(),
            "weather_code": time_table[
                "weather_code"
            ].to_numpy(),
            "wind_speed_10m": time_table[
                "wind_speed_10m"
            ].to_numpy(),

            # 区域 i 历史总需求
            "D_i_lag1": D_i_lag1[
                region_index
            ],
            "D_i_lag2": D_i_lag2[
                region_index
            ],
            "D_i_roll4_mean": D_i_roll4_mean[
                region_index
            ],
            "D_i_roll8_mean": D_i_roll8_mean[
                region_index
            ],
            "D_i_same_slot_last_day": (
                D_i_same_slot_last_day[
                    region_index
                ]
            ),
            "D_i_same_slot_last_week": (
                D_i_same_slot_last_week[
                    region_index
                ]
            ),
            "D_i_total_last_day": (
                D_i_total_last_day[
                    region_index
                ]
            ),
            "D_i_daily_avg_last3": (
                D_i_daily_avg_last3[
                    region_index
                ]
            ),
            "D_i_daily_avg_last7": (
                D_i_daily_avg_last7[
                    region_index
                ]
            ),

            # 801维历史比例是否可用
            "pi_lag1_available": (
                pi_lag1_available[
                    start:end
                ]
            ),
            "pi_lag2_available": (
                pi_lag2_available[
                    start:end
                ]
            ),
            "pi_roll4_available": (
                pi_roll4_available[
                    start:end
                ]
            ),
            "pi_roll8_available": (
                pi_roll8_available[
                    start:end
                ]
            ),
            "pi_same_slot_last_day_available": (
                pi_last_day_available[
                    start:end
                ]
            ),
            "pi_same_slot_last_week_available": (
                pi_last_week_available[
                    start:end
                ]
            ),
            "pi_daily_share_last_day_available": (
                pi_daily_last_day_available[
                    start:end
                ]
            ),
            "pi_daily_share_avg_last3_available": (
                pi_daily_avg3_available[
                    start:end
                ]
            ),
            "pi_daily_share_avg_last7_available": (
                pi_daily_avg7_available[
                    start:end
                ]
            ),

            # 当前真实标签辅助信息
            "D_it_observed": D_it[
                start:end
            ],
            "pi_label_available": (
                pi_label_available[
                    start:end
                ]
            ),
        }
    )

    table = pa.Table.from_pandas(
        result,
        preserve_index=False,
    )

    if writer is None:
        writer = pq.ParquetWriter(
            SCALAR_FILE,
            table.schema,
            compression="snappy",
        )

    writer.write_table(table)


if writer is not None:
    writer.close()


# ============================================================
# 15. 保存801个目的地区域固定顺序
# ============================================================

pd.DataFrame(
    {
        "destination_index": np.arange(
            NUM_REGIONS,
            dtype=np.int32,
        ),
        "region_id": REGIONS,
    }
).to_csv(
    DESTINATION_MAP_FILE,
    index=False,
)


# ============================================================
# 16. 完成
# ============================================================

print(f"完成：{OUTPUT_DIR}")
