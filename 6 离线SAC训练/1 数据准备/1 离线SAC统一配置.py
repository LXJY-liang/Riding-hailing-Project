# -*- coding: utf-8 -*-
"""任务 1：集中配置第一版离线 SAC 的数据路径、模型参数和实验参数。

所有脚本都从这里读取路径和模型参数。需要改变实验设定时只修改本文件，避免同一个
参数在多个脚本中出现不一致。路径均由项目根目录推导，不移动任何已有数据文件。
"""

from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# 1. 路径
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OFFLINE_ROOT = PROJECT_ROOT / "6 离线SAC训练"

DATA_PREP_DIR = OFFLINE_ROOT / "1 数据准备"
SUPPLY_DIR = OFFLINE_ROOT / "2 Sit重构"
VIT_DIR = OFFLINE_ROOT / "3 Vit准备"
SAC_DIR = OFFLINE_ROOT / "4 离线SAC模型"

DATA_OUTPUT_DIR = DATA_PREP_DIR / "output"
SUPPLY_OUTPUT_DIR = SUPPLY_DIR / "output"
VIT_OUTPUT_DIR = VIT_DIR / "output"
SAC_OUTPUT_DIR = SAC_DIR / "output"
MODEL_OUTPUT_DIR = SAC_OUTPUT_DIR / "model"
LOG_OUTPUT_DIR = SAC_OUTPUT_DIR / "log"

# 现有主流程数据。明确不读取“第一版测试”和“第二版测试”。
RAW_ORDER_FILE = (
    PROJECT_ROOT / "1 数据清洗" / "1_4 异常订单清洗_保留订单.parquet"
)
ACTIVE_OD_FILE = (
    PROJECT_ROOT / "1 数据清洗" / "2_2 OD-15min聚合_属于芝加哥.parquet"
)
REGION_PANEL_FILE = (
    PROJECT_ROOT / "1 数据清洗" / "3_3 区域-15min聚合（补全空值）结果.parquet"
)

BASE_OD_OUTPUT = DATA_OUTPUT_DIR / "active_od_base.parquet"
DATA_DIAGNOSTICS = DATA_OUTPUT_DIR / "data_diagnostics.json"
REGION_MAP_FILE = DATA_OUTPUT_DIR / "region_mapping.csv"

SUPPLY_OUTPUT = SUPPLY_OUTPUT_DIR / "S_it_historical.parquet"
SUPPLY_DIAGNOSTICS = SUPPLY_OUTPUT_DIR / "S_it_diagnostics.json"

VIT_OUTPUT = VIT_OUTPUT_DIR / "V_it_distribution.parquet"
VIT_DIAGNOSTICS = VIT_OUTPUT_DIR / "V_it_diagnostics.json"

STATE_OUTPUT = SAC_OUTPUT_DIR / "offline_sac_state.parquet"
STATE_SCALER_OUTPUT = MODEL_OUTPUT_DIR / "state_scaler.npz"
STATE_DIAGNOSTICS = LOG_OUTPUT_DIR / "state_diagnostics.json"
DEFAULT_RUN_NAME = "gamma099"
MODEL_CHECKPOINT = MODEL_OUTPUT_DIR / DEFAULT_RUN_NAME / "best_offline_sac.pt"
TRAINING_HISTORY = LOG_OUTPUT_DIR / f"training_history_{DEFAULT_RUN_NAME}.csv"
POLICY_OUTPUT = SAC_OUTPUT_DIR / "test_policy_and_baselines.parquet"
EVALUATION_METRICS = LOG_OUTPUT_DIR / "evaluation_metrics.json"


# ---------------------------------------------------------------------------
# 2. 时间划分
# ---------------------------------------------------------------------------

TRAIN_START = "2024-06-11 00:00:00"
TRAIN_END = "2024-06-24 23:45:00"
VALID_START = "2024-06-25 00:00:00"
VALID_END = "2024-06-27 23:45:00"
TEST_START = "2024-06-28 00:00:00"
TEST_END = "2024-06-30 23:45:00"

SPLITS = {
    "train": (TRAIN_START, TRAIN_END),
    "valid": (VALID_START, VALID_END),
    "test": (TEST_START, TEST_END),
}

TIME_STEP_MINUTES = 15


# ---------------------------------------------------------------------------
# 3. 历史代理与司机未来价值
# ---------------------------------------------------------------------------

# 历史供给递推中，上一时段未随订单离开的司机有多少比例继续留在原区域。
SUPPLY_RETAIN_RATE = 0.70

# 司机未来价值窗口：16 个 15 分钟时段，即未来 4 小时。
DRIVER_VALUE_HORIZON = 16
DRIVER_GAMMA = 0.95

# 每个 (区域, 时段) 的司机随机轨迹条数。正式运行可提高到 128 或 256；64 是第一版
# 在运行时间和分布稳定性之间的折中。
VIT_MONTE_CARLO_SAMPLES = 64
VIT_RANDOM_SEED = 20260914
VIT_QUANTILE_LEVELS = np.asarray(
    [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
     0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
    dtype=np.float32,
)
VIT_QUANTILE_COLUMNS = [f"V_q{int(q * 100):02d}" for q in VIT_QUANTILE_LEVELS]

# V_oracle 只用于形成历史标签。Actor 使用同一区域、同一日内时段在过去若干天的
# oracle 分位数均值；shift(1) 确保不使用当天未来结果。
VIT_PROXY_LOOKBACK_DAYS = 7
VIT_PROXY_MIN_HISTORY_DAYS = 3
VIT_WARMUP_END = "2024-06-10 23:45:00"


# ---------------------------------------------------------------------------
# 4. 平台、接受概率与动作范围
# ---------------------------------------------------------------------------

PLATFORM_COMMISSION_RATE = 0.25
PASSENGER_ALPHA = 0.05
DRIVER_BETA = 0.05
MAX_DISPATCH = 5

# 单笔补贴上限 b_max = SUBSIDY_MAX_RATIO * p_ref。
# 这是实验参数，不是数据事实；如最终采用其他比例，只需在此处修改。
SUBSIDY_MAX_RATIO = 0.80

# Actor 先输出 [-1, 1]，再把价格映射到以下历史参考价倍数范围。设置上限可防止 Actor
# 利用历史代理分布之外的外推误差。
PRICE_RATIO_MIN = 0.50
PRICE_RATIO_MAX = 1.50

# 大多数 OD-15min 只有一笔订单，原始 p 分位数会退化成同一个数。这里用训练期内
# 有重复订单的样本估计价格变异系数，并至少保留该下限，使接受概率不会退化成 0/1。
PRICE_CV_FALLBACK = 0.08
PRICE_CV_FLOOR_QUANTILE = 0.25


# ---------------------------------------------------------------------------
# 5. SAC 参数
# ---------------------------------------------------------------------------

# 平台 SAC 的 TD 折扣因子；它与 DRIVER_GAMMA 是两个不同对象。虽然当前动作不改变
# 历史下一状态，主实验仍保留标准 0.99，并用命令行再训练 gamma=0 作为必要消融。
SAC_GAMMA = 0.99
RANDOM_SEED = 42

ACTOR_HIDDEN_DIMS = (256, 256)
CRITIC_HIDDEN_DIMS = (256, 256)
REGION_EMBEDDING_DIM = 16
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
ENTROPY_LR = 3e-4
TARGET_TAU = 0.005
INITIAL_ENTROPY_ALPHA = 0.20
TARGET_ENTROPY = -2.0

BATCH_SIZE = 512
REPLAY_CAPACITY = 500_000
LEARNING_STARTS = 10_000
TOTAL_COLLECTION_STEPS = 20_000
UPDATES_PER_COLLECTION = 1
GROUPS_PER_COLLECTION = 16
VALIDATION_GROUPS = 2_000

# 收集反事实动作时混合 Actor、历史动作附近扰动和全可行域随机动作。
ACTION_SOURCE_PROBABILITIES = {
    "actor": 0.50,
    "local": 0.30,
    "uniform": 0.20,
}

# Lagrangian 乘子。回放池保存原始收益和违反度，训练时使用最新乘子构造奖励。
LAMBDA_PASSENGER_INIT = 0.0
LAMBDA_DRIVER_INIT = 0.0
LAMBDA_LR = 1e-3
LAMBDA_MAX = 100.0

# 反事实接受概率采用确定性的分层分位数场景。固定场景使不同候选动作可公平比较。
TRAIN_NUM_SCENARIOS = 32
EVAL_NUM_SCENARIOS = 64


# ---------------------------------------------------------------------------
# 6. 直接优化基线
# ---------------------------------------------------------------------------

FIXED_PRICE_RATIO = 1.10
FIXED_SUBSIDY_RATIO = 0.10
GRID_PRICE_RATIOS = np.linspace(0.70, 1.30, 7, dtype=np.float32)
GRID_SUBSIDY_FRACTIONS = np.linspace(0.0, 1.0, 5, dtype=np.float32)
GRID_COORDINATE_PASSES = 1


# ---------------------------------------------------------------------------
# 7. Actor 数值状态
# ---------------------------------------------------------------------------

STATE_NUMERIC_COLUMNS = [
    "log1p_D_ijt",
    "log1p_D_it",
    "log1p_S_it",
    "demand_supply_ratio",
    "od_demand_share",
    "p_ref_ijt",
    "pe_q10_ijt",
    "pe_q50_ijt",
    "pe_q90_ijt",
    "tau_bar_ijt",
    "tau_q10_ijt",
    "tau_q50_ijt",
    "tau_q90_ijt",
    "trip_miles_bar_ijt",
    "c_bar_ijt",
    "c_q10_ijt",
    "c_q50_ijt",
    "c_q90_ijt",
    "V_i_q10",
    "V_i_q50",
    "V_i_q90",
    "V_reject_q10",
    "V_reject_q50",
    "V_reject_q90",
    "V_dest_q10",
    "V_dest_q50",
    "V_dest_q90",
    "slot_sin",
    "slot_cos",
    "dow_sin",
    "dow_cos",
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


def ensure_output_directories() -> None:
    """创建本流程自己的输出目录，不修改或移动任何上游文件。"""

    for directory in (
        DATA_OUTPUT_DIR,
        SUPPLY_OUTPUT_DIR,
        VIT_OUTPUT_DIR,
        SAC_OUTPUT_DIR,
        MODEL_OUTPUT_DIR,
        LOG_OUTPUT_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
