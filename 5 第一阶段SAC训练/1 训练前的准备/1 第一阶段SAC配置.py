# -*- coding: utf-8 -*-
"""第一阶段外生历史状态离线SAC的统一配置。"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


ROOT = Path(
    os.environ.get("RIDING_HAILING_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).expanduser().resolve()
STAGE_DIR = ROOT / "5 第一阶段SAC训练"
OUTPUT_DIR = STAGE_DIR / "output"
PREPARED_DIR = OUTPUT_DIR / "prepared"
POTENTIAL_DEMAND_DIR = PREPARED_DIR / "potential_demand"
EPISODE_DIR = PREPARED_DIR / "episodes"
NEW_FROZEN_VIT_DIR = (
    ROOT / "4 Vit预训练模型" / "2 训练与评估"
    / "output" / "prediction" / "frozen_vit"
)
LEGACY_FROZEN_VIT_DIR = PREPARED_DIR / "frozen_vit"
_vit_dir_override = os.environ.get("FROZEN_VIT_DIR")
_new_vit_train_complete = all(
    (NEW_FROZEN_VIT_DIR / f"vit_quantiles_{day}.npz").exists()
    for day in np.arange(
        np.datetime64("2024-06-11", "D"),
        np.datetime64("2024-06-25", "D"),
        dtype="datetime64[D]",
    ).astype(str)
)
if _vit_dir_override:
    FROZEN_VIT_DIR = Path(_vit_dir_override).expanduser().resolve()
elif _new_vit_train_complete:
    # 新Vit的14个训练日全部齐备后，SAC才原子切换到Vit目录。
    FROZEN_VIT_DIR = NEW_FROZEN_VIT_DIR
else:
    # 新模型尚未完成时保留对现有SAC产物的兼容，不中断当前程序。
    FROZEN_VIT_DIR = LEGACY_FROZEN_VIT_DIR
MODEL_DIR = OUTPUT_DIR / "model"
LOG_DIR = OUTPUT_DIR / "log"
EVALUATION_DIR = OUTPUT_DIR / "evaluation"

OD_FILE = ROOT / "1 数据清洗" / "2_2 OD-15min聚合_属于芝加哥.parquet"
REGION_FILE = ROOT / "1 数据清洗" / "3_3 区域-15min聚合（补全空值）结果.parquet"

PI_ROOT = ROOT / "3 随机变量分位数预测" / "2 πijt构建" / "output"
PI_REGION_MAP = PI_ROOT / "prepared_index" / "region_mapping.csv"
PI_TIME_MAP = PI_ROOT / "prepared_index" / "time_mapping.parquet"
PI_PARTS_DIR = PI_ROOT / "pi_distribution_parts"

DIT_PREDICTION_DIR = Path(
    os.environ.get(
        "DIT_PREDICTION_DIR",
        ROOT / "3 随机变量分位数预测" / "1 Dit预测" / "output" / "prediction",
    )
).expanduser().resolve()
TAU_PREDICTION_DIR = Path(
    os.environ.get(
        "TAU_PREDICTION_DIR",
        ROOT / "3 随机变量分位数预测" / "4 tauijt预测" / "output" / "prediction",
    )
).expanduser().resolve()
PE_DATASET_DIR = (
    ROOT / "3 随机变量分位数预测" / "5 p^eijt分布构建"
    / "output" / "distribution" / "peijt_quantiles"
)
COST_FILE = (
    ROOT / "3 随机变量分位数预测" / "6 c_ij构建"
    / "output" / "final" / "c_ij.parquet"
)

VIT_PRETRAIN_DIR = ROOT / "4 Vit预训练模型"
VIT_MODEL_DIR = VIT_PRETRAIN_DIR / "2 训练与评估" / "output" / "model"
VIT_QUANTILE_HEAD_FILE = VIT_MODEL_DIR / "best_quantile_head.pt"
VIT_SCALER_FILE = VIT_PRETRAIN_DIR / "1 数据准备" / "output" / "scaler.json"
VIT_STATE_FEATURE_DIR = (
    VIT_PRETRAIN_DIR / "2 训练与评估" / "output" / "state_features"
)
VIT_MODEL_CODE = VIT_PRETRAIN_DIR / "2 训练与评估" / "1 定义Vit分位数模型.py"
VIT_STATE_FEATURE_DIM = 512

SUPPLY_FILE = PREPARED_DIR / "S_it_rolling_capacity.parquet"
SUPPLY_AUDIT_FILE = PREPARED_DIR / "S_it_rolling_capacity_audit.csv"
SUPPLY_QUANTILE = float(os.environ.get("STAGE1_SUPPLY_QUANTILE", "0.90"))
SUPPLY_UTILIZATION = float(os.environ.get("STAGE1_SUPPLY_UTILIZATION", "0.80"))
SUPPLY_MIN_SAMPLES = int(os.environ.get("STAGE1_SUPPLY_MIN_SAMPLES", "10"))

# Dit预测的是成交量；主实验按综合完成率0.80恢复潜在需求。
DEMAND_COMPLETION_RATE = float(
    os.environ.get("STAGE1_DEMAND_COMPLETION_RATE", "0.80")
)

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
STEPS_PER_DAY = 96
QUANTILE_LEVELS = np.asarray(
    [0.005] + [level / 100 for level in range(1, 100)] + [0.995], dtype=np.float32
)
QUANTILE_COLUMNS = [f"q_{level:.3f}" for level in QUANTILE_LEVELS]
PE_QUANTILE_COLUMNS = [f"pe_q{int(round(level * 1000)):04d}" for level in QUANTILE_LEVELS]
KEY_QUANTILE_LEVELS = np.asarray(
    [0.005, 0.05, 0.10, 0.50, 0.90, 0.95, 0.995], dtype=np.float32
)

PLATFORM_RATE = 0.25
PASSENGER_ALPHA = 0.30
DRIVER_BETA = 0.05
MAX_DISPATCH = 5
HARD_INFEASIBLE_ORDER_LOSS = True
PRICE_RATIO_MIN = 0.30
PRICE_RATIO_MAX = 1.50
POLICY_INITIAL_PRICE_RATIO = 0.40
SUBSIDY_MAX_RATIO = 0.50
DRIVER_GAMMA = 0.95
SAC_GAMMA = 0.99
DRIVER_ACCEPT_STE_TEMPERATURE = 1.0
ACCEPTANCE_SCENARIOS = 32

REGION_EMBEDDING_DIM = 24
ACTOR_HIDDEN = (384, 384)
CRITIC_HIDDEN = (384, 384)
REGION_CONTEXT_DIM = 7
ACTOR_LR = 2e-5
CRITIC_LR = 1e-4
ENTROPY_LR = 1e-4
ACTOR_GRAD_CLIP = 1.0
CRITIC_GRAD_CLIP = 10.0
CRITIC_HUBER_DELTA = 1.0
POLICY_LOG_STD_MIN = -5.0
POLICY_LOG_STD_MAX = 1.0
POLICY_INITIAL_LOG_STD = -2.0
POLICY_INITIAL_SUBSIDY_LOGIT = -4.0
TARGET_TAU = 0.005
INITIAL_ALPHA = 0.20
TARGET_ENTROPY = -2.0

TRAIN_EPOCHS = 20
GROUP_BATCH_SIZE = 32
# 全部历史转移组成固定索引；训练只随机读取，不存在容量、FIFO或经验写回。
GRADIENT_UPDATES_PER_TIME_STEP = 1

LAMBDA_PASSENGER_INIT = 6.0
LAMBDA_DRIVER_INIT = 0.0
# 每16个15分钟时段（即4小时）更新一次乘客/司机全局乘子。
# 每次使用该4小时内所有有效OD的逐OD非负违反度平均值，
# 不使用随机抽取的32个固定历史转移mini-batch更新乘子。
LAMBDA_UPDATE_INTERVAL_STEPS = 16
# 每日6次乘子更新，因此单次学习率为0.2/6，保持原定的
# 日级名义更新尺度0.2。
LAMBDA_LR = 2e-1 / (STEPS_PER_DAY // LAMBDA_UPDATE_INTERVAL_STEPS)
LAMBDA_MAX = 100.0
RANDOM_SEED = 42
REWARD_SCALE = 0.10
DEFAULT_RUN_NAME = "offline_sac_main"
VALIDATION_SCENARIO_SEED = 20240625
VALIDATION_VIOLATION_TOLERANCE = 1e-3


def ensure_directories() -> None:
    for path in (
        PREPARED_DIR, POTENTIAL_DEMAND_DIR, EPISODE_DIR, FROZEN_VIT_DIR,
        MODEL_DIR, LOG_DIR, EVALUATION_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def split_for_time(timestamp) -> str:
    value = np.datetime64(timestamp)
    for name, (start, end) in SPLITS.items():
        if np.datetime64(start) <= value <= np.datetime64(end):
            return name
    return "warmup"


def days_for_splits(splits: list[str]) -> list[np.datetime64]:
    days: set[np.datetime64] = set()
    for split in splits:
        start, end = SPLITS[split]
        first = np.datetime64(start, "D")
        last = np.datetime64(end, "D")
        days.update(np.arange(first, last + np.timedelta64(1, "D"), dtype="datetime64[D]"))
    return sorted(days)
