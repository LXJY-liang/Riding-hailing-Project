# -*- coding: utf-8 -*-
"""Vit 101维分位数预训练的路径与超参数。"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(
    os.environ.get("RIDING_HAILING_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).expanduser().resolve()
VIT_ROOT = PROJECT_ROOT / "4 Vit预训练模型"
OUTPUT_DIR = VIT_ROOT / "output"
DATA_DIR = OUTPUT_DIR / "data"
TOKENIZED_DIR = OUTPUT_DIR / "tokenized"
MODEL_DIR = OUTPUT_DIR / "model"
LOG_DIR = OUTPUT_DIR / "log"
STATE_TEXT_DIR = OUTPUT_DIR / "state_text"
STATE_FEATURE_DIR = OUTPUT_DIR / "state_features"
HISTORICAL_FEATURE_DIR = OUTPUT_DIR / "historical_features"
REGION_HISTORY_FEATURE_DIR = OUTPUT_DIR / "historical_region_features"
SAC_EPISODE_DIR = (
    PROJECT_ROOT / "5 第一阶段SAC训练" / "output" / "prepared" / "episodes"
)

QWEN_BASE_DIR = Path(
    os.environ.get("QWEN3_4B_BASE_PATH", PROJECT_ROOT / "0 基座模型_Qwen3-4B-Base")
).expanduser().resolve()

# 旧流程已用司机未来4小时轨迹模拟产生19维分位数。
SOURCE_VIT_DISTRIBUTION = (
    PROJECT_ROOT / "6 离线SAC训练" / "3 Vit准备" / "output" / "V_it_distribution.parquet"
)
LABEL_FILES = {
    split: DATA_DIR / f"vit_{split}_labels_101.parquet"
    for split in ("train", "valid", "test")
}

# Vit预训练只使用历史可见信息，不依赖Dit/tauijt预测结果。
BASE_FEATURE_FILE = (
    PROJECT_ROOT
    / "4 司机未来价值预测模型"
    / "1 Vit输入特征与标签构造"
    / "2 Vit基础特征构造（结果）.parquet"
)
REGION_HISTORY_FILE = (
    PROJECT_ROOT / "1 数据清洗" / "3_3 区域-15min聚合（补全空值）结果.parquet"
)
OD_HISTORY_FILE = PROJECT_ROOT / "1 数据清洗" / "2_2 OD-15min聚合_属于芝加哥.parquet"
NEIGHBOR_FILE = PROJECT_ROOT / "1 数据清洗" / "4 邻近区域关系表构造结果.parquet"
SUPPLY_HISTORY_FILE = (
    PROJECT_ROOT / "5 第一阶段SAC训练" / "output" / "prepared" / "S_it_baseline.parquet"
)
REGION_MAP_FILE = (
    PROJECT_ROOT
    / "3 随机变量分位数预测"
    / "2 πijt构建"
    / "output"
    / "prepared_index"
    / "region_mapping.csv"
)

PRETRAIN_FILES = {
    split: DATA_DIR / f"vit_{split}.parquet"
    for split in ("train", "valid", "test")
}
TOKENIZED_SPLIT_DIRS = {
    split: TOKENIZED_DIR / f"vit_{split}_tokenized"
    for split in ("train", "valid", "test")
}

TOKENIZER_DIR = MODEL_DIR / "tokenizer"
BEST_LORA_DIR = MODEL_DIR / "best_lora"
BEST_STATE_PROJECTION_FILE = MODEL_DIR / "best_state_projection.pt"
BEST_QUANTILE_HEAD_FILE = MODEL_DIR / "best_quantile_head.pt"
SCALER_FILE = MODEL_DIR / "scaler.json"
TRAINING_HISTORY_FILE = LOG_DIR / "training_history.csv"
BEST_MODEL_INFO_FILE = LOG_DIR / "best_model_info.json"

SPLITS = {
    "train": ("2024-06-11 00:00:00", "2024-06-24 23:45:00"),
    "valid": ("2024-06-25 00:00:00", "2024-06-27 23:45:00"),
    "test": ("2024-06-28 00:00:00", "2024-06-30 20:00:00"),
}

QUANTILE_LEVELS = np.asarray(
    [0.005] + [level / 100 for level in range(1, 100)] + [0.995],
    dtype=np.float32,
)
QUANTILE_COLUMNS = [f"V_q_{level:.3f}" for level in QUANTILE_LEVELS]
SOURCE_PERCENTILES = np.arange(5, 100, 5, dtype=np.int16)
SOURCE_LEVELS = SOURCE_PERCENTILES.astype(np.float32) / 100.0
SOURCE_ORACLE_COLUMNS = [f"oracle_V_q{percentile:02d}" for percentile in SOURCE_PERCENTILES]

# Vit与第一阶段SAC共用的需求分布摘要：20个5%区间均值和7个尾部敏感关键分位数。
DEMAND_KEY_LEVELS = (0.005, 0.05, 0.10, 0.50, 0.90, 0.95, 0.995)
DEMAND_FEATURE_DIM = 27
DEMAND_FEATURE_COLUMNS = [f"demand_feature_{index:02d}" for index in range(DEMAND_FEATURE_DIM)]

# 只对抽样状态进行LoRA预训练；所有状态的文本会另外保留，供训练后一次性导出编码。
MAX_SAMPLES = {
    "train": int(os.environ.get("VIT_TRAIN_MAX_SAMPLES", "25000")),
    "valid": int(os.environ.get("VIT_VALID_MAX_SAMPLES", "5000")),
    "test": int(os.environ.get("VIT_TEST_MAX_SAMPLES", "5000")),
}
RANDOM_SEED = 42
TOP_K_DESTINATIONS = 10
TOP_K_NEIGHBORS = 5
HISTORY_SLOT_RADIUS = 1
HISTORY_MIN_SAMPLES = 6
TIME_STEPS_PER_DAY = 96

MAX_LENGTH = 2048
TOKENIZER_BATCH_SIZE = 512
TOKENIZER_NUM_PROC = 8
PADDING_SIDE = "right"

LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_NAMES = ("q_proj", "v_proj")
LORA_LAYER_START = 32
LORA_LAYER_END = 35
STATE_FEATURE_DIM = 512
HEAD_HIDDEN_DIM = 512
HEAD_DROPOUT = 0.0

NUM_EPOCHS = 1
TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 16
LORA_LEARNING_RATE = 5e-5
HEAD_LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.03
MAX_GRAD_NORM = 1.0
NUM_WORKERS = 4
USE_GRADIENT_CHECKPOINTING = False
DEVICE = "auto"

EXPORT_BATCH_SIZE = int(os.environ.get("VIT_EXPORT_BATCH_SIZE", "16"))


def ensure_directories() -> None:
    for path in (
        DATA_DIR,
        TOKENIZED_DIR,
        MODEL_DIR,
        LOG_DIR,
        STATE_TEXT_DIR,
        STATE_FEATURE_DIR,
        HISTORICAL_FEATURE_DIR,
        REGION_HISTORY_FEATURE_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
