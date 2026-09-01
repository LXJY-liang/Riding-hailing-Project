# -*- coding: utf-8 -*-
"""π_ijt 历史目的地分布构建的统一配置。

这套流程不读取旧版 LLM+LoRA 的模型、Token 或预测结果。
π_ijt 完全根据目标时段 t 之前的历史成交订单数构建。

主要口径：
1. 前三层历史时段始终完整保留；
2. 前三层原始订单数小于 N_MIN 时才使用第四层；
3. 第四层按时间由近到远加入完整的 15 分钟历史时段；
4. N_STOP 是停止阈值，不是硬上限。最后一个时段不截断，
   因此第四层加入后的总订单数可以超过 N_STOP；
5. 最终使用层级权重和订单数共同估计目的地概率。
"""

from pathlib import Path


# ---------------------------------------------------------------------------
# 1. 路径
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = Path(__file__).resolve().parent

# 只读取真实出现过订单的芝加哥内部 OD，构造稀疏历史订单矩阵。
ACTIVE_OD_FILE = (
    PROJECT_ROOT
    / "1 数据清洗"
    / "2_2 OD-15min聚合_属于芝加哥.parquet"
)

# 已删除价格、服务时间和距离异常的订单级数据。第 5 步随机复核时
# 从这里重新计数，不使用已准备的 OD 稀疏矩阵，从而形成独立校验。
ORDER_LEVEL_FILE = (
    PROJECT_ROOT
    / "1 数据清洗"
    / "1_4 异常订单清洗_保留订单.parquet"
)

# 801 个区域×全部 15 分钟时段的完整面板，用于确定目标 (i,t)。
REGION_PANEL_FILE = (
    PROJECT_ROOT
    / "1 数据清洗"
    / "3_3 区域-15min聚合（补全空值）结果.parquet"
)

OUTPUT_DIR = CODE_DIR / "output"
INDEX_DIR = OUTPUT_DIR / "prepared_index"
PI_PARTS_DIR = OUTPUT_DIR / "pi_distribution_parts"
DIAGNOSTIC_PARTS_DIR = OUTPUT_DIR / "diagnostic_parts"
DIAGNOSTICS_DIR = OUTPUT_DIR / "diagnostics"
RANDOM_AUDIT_DIR = OUTPUT_DIR / "random_sample_audit"

REGION_MAPPING_FILE = INDEX_DIR / "region_mapping.csv"
TIME_MAPPING_FILE = INDEX_DIR / "time_mapping.parquet"
HISTORY_COUNT_MATRIX_FILE = INDEX_DIR / "history_od_count_matrix.npz"
HISTORY_ROW_TOTAL_FILE = INDEX_DIR / "history_origin_time_total.npy"
TIME_SLOT_FILE = INDEX_DIR / "time_slot.npy"
TIME_DOW_FILE = INDEX_DIR / "time_dow.npy"
TIME_HOLIDAY_FILE = INDEX_DIR / "time_is_holiday.npy"

INDEX_DIAGNOSTICS_FILE = DIAGNOSTICS_DIR / "pi_index_diagnostics.json"
BUILD_DIAGNOSTICS_FILE = DIAGNOSTICS_DIR / "pi_build_diagnostics.json"
CHECK_DIAGNOSTICS_FILE = DIAGNOSTICS_DIR / "pi_check_diagnostics.json"
OUTPUT_MANIFEST_FILE = OUTPUT_DIR / "pi_output_manifest.json"


# ---------------------------------------------------------------------------
# 2. 时间与区域口径
# ---------------------------------------------------------------------------

DATA_START = "2024-06-01 00:00:00"
DATA_END = "2024-06-30 23:45:00"
TIME_STEP_MINUTES = 15
SLOTS_PER_DAY = 96
EXPECTED_REGION_COUNT = 801
EXPECTED_TIME_COUNT = 30 * SLOTS_PER_DAY

# 2024 年 6 月芝加哥本地节假日。
HOLIDAY_DATES = {"2024-06-19"}


# ---------------------------------------------------------------------------
# 3. 分层规则
# ---------------------------------------------------------------------------

# 前三层订单数达到该值时，不再使用第四层。
N_MIN = 100

# 仅当前三层不足 N_MIN 时启用。这是停止阈值，不是硬上限。
N_STOP = 400

# 第三层环形 slot 距离：前后 4 个 15 分钟时段。
TIME_NEIGHBOR_SLOTS = 4

LAYER_WEIGHTS = {
    1: 1.00,
    2: 0.95,
    3: 0.90,
    4: 0.50,
}


# ---------------------------------------------------------------------------
# 4. I/O 与并行
# ---------------------------------------------------------------------------

PARQUET_BATCH_SIZE = 250_000
PARQUET_COMPRESSION = "zstd"
DEFAULT_WORKERS = 1
PROBABILITY_SUM_ATOL = 2e-5


def pi_part_file(origin_index: int) -> Path:
    """单个出发区域的 [time, destination] CSR 概率矩阵。"""

    return PI_PARTS_DIR / f"origin-{origin_index:04d}.npz"


def diagnostic_part_file(origin_index: int) -> Path:
    """单个出发区域的逐时段样本诊断。"""

    return DIAGNOSTIC_PARTS_DIR / f"origin-{origin_index:04d}.parquet"


def ensure_output_directories() -> None:
    """只创建本任务管理的输出目录。"""

    for directory in (
        OUTPUT_DIR,
        INDEX_DIR,
        PI_PARTS_DIR,
        DIAGNOSTIC_PARTS_DIR,
        DIAGNOSTICS_DIR,
        RANDOM_AUDIT_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
