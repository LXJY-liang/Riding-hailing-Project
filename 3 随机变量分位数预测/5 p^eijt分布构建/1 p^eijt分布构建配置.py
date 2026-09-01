# -*- coding: utf-8 -*-
"""p^e_ijt 历史市场接受价格代理分布的统一配置。

本目录是一套全新的滚动 KDE 构建流程，不读取旧版 p^e_ijt LLM
预测的特征、模型或预测结果。

数据口径：
1. 每笔历史价格 = Fare + Additional Charges，不包含小费；
2. 对目标 (i,j,t)，只使用 Trip End Timestamp < t 的已完成订单；
3. 主任务只对 2_2 表中真实出现过订单的 OD-15min 单元输出分布；
4. 仅使用 2024 年 6 月内当时可得的历史，月初按“有多少用多少”处理。
"""

from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# 1. 路径
# ---------------------------------------------------------------------------

# 当前文件位于：项目根目录/3 随机变量分位数预测/5 p^eijt分布构建/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = Path(__file__).resolve().parent

# 已删除价格、服务时间和距离异常，并且已限定为
# 801 个芝加哥 Census Tract 内部 OD 的逐笔订单。该文件用于提供
# 历史价格、实际里程、服务时间和订单完成时间。
RAW_ORDER_FILE = (
    PROJECT_ROOT
    / "1 数据清洗"
    / "3_1 芝加哥内部OD订单.parquet"
)

# 目标表：只使用其 i、j、t、D_ijt_observed 定义需估算的单元。
TARGET_OD_FILE = (
    PROJECT_ROOT
    / "1 数据清洗"
    / "2_2 OD-15min聚合_属于芝加哥.parquet"
)

TRACT_FILE = (
    PROJECT_ROOT
    / "0 芝加哥census tract编号及对应的community"
    / "Census Tracts TIGER 2010.csv"
)

NEIGHBOR_FILE = (
    PROJECT_ROOT
    / "1 数据清洗"
    / "4 邻近区域关系表构造结果.parquet"
)

OUTPUT_DIR = CODE_DIR / "output"
INTERMEDIATE_DIR = OUTPUT_DIR / "intermediate"
INDEX_DIR = INTERMEDIATE_DIR / "prepared_index"
DISTRIBUTION_DIR = OUTPUT_DIR / "distribution"
DIAGNOSTICS_DIR = OUTPUT_DIR / "diagnostics"
AUDIT_DIR = OUTPUT_DIR / "audit_random_samples"

ORDER_HISTORY_FILE = INTERMEDIATE_DIR / "order_price_history.parquet"
REGION_CENTROID_FILE = INTERMEDIATE_DIR / "region_centroids.parquet"
NORMALIZED_NEIGHBOR_FILE = INTERMEDIATE_DIR / "region_neighbors.parquet"
NEIGHBOR_MATRIX_FILE = INTERMEDIATE_DIR / "region_neighbor_matrix.npy"

# 第 4 个脚本一次性生成的纯数值计算索引。第 5、6 个脚本不再读取原始
# 订单、做区域编号清洗或执行 pandas groupby。
TARGET_PREPARED_FILE = INDEX_DIR / "target_units.parquet"
HISTORY_COMPLETION_NS_FILE = INDEX_DIR / "history_completion_ns.npy"
HISTORY_I_FILE = INDEX_DIR / "history_i_index.npy"
HISTORY_J_FILE = INDEX_DIR / "history_j_index.npy"
HISTORY_DAY_FILE = INDEX_DIR / "history_start_day.npy"
HISTORY_SLOT_FILE = INDEX_DIR / "history_slot.npy"
HISTORY_DOW_FILE = INDEX_DIR / "history_dow.npy"
HISTORY_WEEKEND_HOLIDAY_FILE = INDEX_DIR / "history_weekend_or_holiday.npy"
HISTORY_PRICE_FILE = INDEX_DIR / "history_price.npy"
HISTORY_TRIP_MILES_FILE = INDEX_DIR / "history_trip_miles.npy"
HISTORY_TRIP_SECONDS_FILE = INDEX_DIR / "history_trip_seconds.npy"
OD_ORDER_FILE = INDEX_DIR / "od_history_order.npy"
OD_OFFSETS_FILE = INDEX_DIR / "od_history_offsets.npy"
ORIGIN_TIME_STRICT_ORDER_FILE = INDEX_DIR / "origin_time_strict_order.npy"
ORIGIN_TIME_STRICT_OFFSETS_FILE = INDEX_DIR / "origin_time_strict_offsets.npy"
ORIGIN_TIME_RELAXED_ORDER_FILE = INDEX_DIR / "origin_time_relaxed_order.npy"
ORIGIN_TIME_RELAXED_OFFSETS_FILE = INDEX_DIR / "origin_time_relaxed_offsets.npy"

# 第 5 个脚本的中间输出：每个目标最终选中的价格、层级、两个分量
# 带宽及相同 OD / 相似 OD 的分段混合权重。
SAMPLE_DATASET = DISTRIBUTION_DIR / "peijt_selected_samples"

# 第 6 个脚本的主输出是 Parquet dataset 目录，内含多个 part-*.parquet。
PE_DISTRIBUTION_DATASET = DISTRIBUTION_DIR / "peijt_quantiles"
INDEX_DIAGNOSTICS_FILE = DIAGNOSTICS_DIR / "peijt_index_diagnostics.json"
SAMPLE_DIAGNOSTICS_FILE = DIAGNOSTICS_DIR / "peijt_sample_diagnostics.json"
BUILD_DIAGNOSTICS_FILE = DIAGNOSTICS_DIR / "peijt_build_diagnostics.json"
CHECK_DIAGNOSTICS_FILE = DIAGNOSTICS_DIR / "peijt_check_diagnostics.json"
STRUCTURAL_AUDIT_DIR = DIAGNOSTICS_DIR / "peijt_quantile_structure_audit"


# ---------------------------------------------------------------------------
# 2. 时间口径
# ---------------------------------------------------------------------------

DATA_START = "2024-06-01 00:00:00"
DATA_END = "2024-06-30 23:45:00"
TIME_STEP_MINUTES = 15
SLOTS_PER_DAY = 96
REGION_COUNT = 801
DOW_COUNT = 7
WEEKEND_HOLIDAY_STATE_COUNT = 2

# 2024 年 6 月芝加哥本地节假日。
# is_weekend_or_holiday = 周末或这些日期之一。
HOLIDAY_DATES = {"2024-06-19"}


# ---------------------------------------------------------------------------
# 3. 分层样本规则
# ---------------------------------------------------------------------------

N_MIN = 50
N_MAX = 200

# 第 1—3 层均属于“相同 OD”分布，保留原有层内权重。
# 第 4A、4B 层合并为“同时间状态+相似实际里程”分布；其样本
# 内部权重由里程相似度给出，两个分布的混合权重由有效样本量与
# 相似度共同决定。
LAYER_WEIGHTS = {
    1: 1.0,
    2: 0.9,
    3: 0.8,
    4: 1.0,
}


SIMILAR_OD_RELIABILITY_DISCOUNT = 0.25
SIMILAR_ONLY_MIN_SAMPLES = 10
SIMILAR_WEIGHT_CAP_SPARSE_SAME = 0.50
SIMILAR_WEIGHT_CAP_ESTABLISHED_SAME = 0.30


def component_mixture_weights(
    same_effective_n: float,
    similar_effective_n: float,
    similar_mean_similarity: float,
    same_od_count: int,
    similar_od_count: int,
) -> tuple[float, float]:
    """根据有效样本量和路程相似度返回 (w_same, w_similar)。

    第 4 层先按平均路程相似度折减，再乘 0.25 可靠性系数；
    当已有同 OD 样本时，相似 OD 总权重最多为 30%（同 OD
    仅 1—4 笔时最多 50%）。这使单笔第 4 层样本不可能获得
    30%—100% 的整体权重。完全没有同 OD 样本时，第 4 层
    至少要有 SIMILAR_ONLY_MIN_SAMPLES 笔才允许单独使用。
    """

    same_od_count = int(same_od_count)
    similar_od_count = int(similar_od_count)
    if similar_od_count <= 0:
        return (1.0, 0.0) if same_od_count > 0 else (np.nan, np.nan)
    if same_od_count <= 0:
        if similar_od_count < SIMILAR_ONLY_MIN_SAMPLES:
            return np.nan, np.nan
        return 0.0, 1.0

    same_mass = max(float(same_effective_n), 0.0)
    similar_mass = (
        SIMILAR_OD_RELIABILITY_DISCOUNT
        * float(np.clip(similar_mean_similarity, 0.0, 1.0))
        * max(float(similar_effective_n), 0.0)
    )
    if similar_mass <= 0.0:
        return 1.0, 0.0
    raw_similar = similar_mass / max(same_mass + similar_mass, np.finfo(float).eps)
    cap = (
        SIMILAR_WEIGHT_CAP_SPARSE_SAME
        if same_od_count <= 4
        else SIMILAR_WEIGHT_CAP_ESTABLISHED_SAME
    )
    similar_weight = min(raw_similar, cap)
    return 1.0 - similar_weight, similar_weight

# 第 3 层的相邻时段范围：4 个 15min，即 1 小时。
# 当天只取目标时刻之前 1 小时；前 1/2/7 天则以 t-lag 为绝对时间中心
# 取前后 1 小时。绝对时间窗口可以跨午夜，不使用 24 小时循环时钟距离。
TIME_NEIGHBOR_SLOTS = 4
SPECIAL_DAY_LAGS = (1, 2, 7)

# 第 4 层不再使用区域质心 OD 距离。目标 OD 的参考里程为当时
# 已完成的同 OD 有效行程里程中位数，至少 3 笔。候选订单必须
# 处于同一时间状态，且实际里程在参考里程的 [0.8, 1.25] 倍。
MIN_TARGET_DISTANCE_HISTORY = 3
TRIP_MILES_RATIO_MIN = 0.80
TRIP_MILES_RATIO_MAX = 1.25
MIN_VALID_TRIP_SECONDS = 60.0
MIN_VALID_SPEED_MPH = 1.0
MAX_VALID_SPEED_MPH = 80.0
DISTANCE_SIMILARITY_LOG_SCALE = float(np.log(TRIP_MILES_RATIO_MAX))

# 自动审计阈值：若 Q_0.30 < 0.3 * Q_0.50，则记录为结构性异常。
Q30_TO_Q50_AUDIT_MIN = 0.30
Q50_TO_Q30_AUDIT_MAX = 1.0 / Q30_TO_Q50_AUDIT_MIN

# 节假日严格匹配没有样本时，允许忽略 is_weekend_or_holiday，
# 但仍保持 slot_in_day + dow 一致，并记录 holiday_relaxed_flag=1。
ALLOW_HOLIDAY_RELAXATION = True


# ---------------------------------------------------------------------------
# 4. KDE 与分位数
# ---------------------------------------------------------------------------

# 101 维：0.005、0.01~0.99、0.995。
QUANTILE_LEVELS = np.asarray(
    [0.005] + [level / 100.0 for level in range(1, 100)] + [0.995],
    dtype=np.float64,
)


def quantile_column_name(level: float) -> str:
    """用千分整数命名，例如 0.005 -> pe_q0005，0.99 -> pe_q0990。"""

    return f"pe_q{int(round(level * 1000)):04d}"


QUANTILE_COLUMNS = [quantile_column_name(level) for level in QUANTILE_LEVELS]

# 按已确定口径，Silverman 带宽不得低于 0.5 美元。
MIN_BANDWIDTH_USD = 0.5

# 分层权重同时进入带宽计算：使用加权标准差、加权 IQR 和有效样本量。
USE_WEIGHTED_SILVERMAN = True

# 使用二分法逆转加权高斯 KDE 的 CDF。每轮将价格区间减半。
# 历史价格最大值为 578.75 美元；20 轮在本数据可能出现的最宽搜索
# 区间下仍达到约 1 美分以内的价格精度，避免原 28 轮的过度计算。
KDE_BISECTION_STEPS = 20
KDE_BOUND_SIGMAS = 8.0


# ---------------------------------------------------------------------------
# 5. I/O 与并行参数
# ---------------------------------------------------------------------------

ORDER_BATCH_SIZE = 250_000
TARGET_BATCH_SIZE = 20_000
PARQUET_COMPRESSION = "zstd"

# 默认不并行，便于本地小样本验证。服务器运行时通过 --workers 覆盖。
DEFAULT_WORKERS = 1


def ensure_output_directories() -> None:
    """只创建本任务的输出目录，不移动、不改写任何上游数据。"""

    for directory in (
        OUTPUT_DIR,
        INTERMEDIATE_DIR,
        INDEX_DIR,
        DISTRIBUTION_DIR,
        DIAGNOSTICS_DIR,
        AUDIT_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
