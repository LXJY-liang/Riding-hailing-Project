# -*- coding: utf-8 -*-
"""Vit 101维分位数预训练的唯一配置。

训练：使用2024-06-01至2024-06-24全部Oracle标签。
验证：使用2024-06-25至2024-06-27独立Oracle标签。
预测：仅导出2024-06-11至2024-06-24供离线SAC使用的Vit分布。
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(
    os.environ.get("RIDING_HAILING_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).expanduser().resolve()
VIT_ROOT = PROJECT_ROOT / "4 Vit预训练模型"
UPSTREAM_ROOT = VIT_ROOT / "0 上游标签与基础特征"
DATA_PREPARATION_ROOT = VIT_ROOT / "1 数据准备"
TRAINING_ROOT = VIT_ROOT / "2 训练与评估"
TRAJECTORY_ORACLE_ROOT = UPSTREAM_ROOT / "1 司机轨迹Oracle构造"
DRIVER_HISTORY_ROOT = UPSTREAM_ROOT / "2 历史司机收益基础特征"

UPSTREAM_OUTPUT_DIR = UPSTREAM_ROOT / "output"
TRAJECTORY_ORACLE_OUTPUT_DIR = UPSTREAM_OUTPUT_DIR / "1 司机轨迹Oracle构造"
DRIVER_HISTORY_OUTPUT_DIR = UPSTREAM_OUTPUT_DIR / "2 历史司机收益基础特征"
DATA_PREPARATION_OUTPUT_DIR = DATA_PREPARATION_ROOT / "output"
TRAINING_OUTPUT_DIR = TRAINING_ROOT / "output"

DATA_DIR = DATA_PREPARATION_OUTPUT_DIR / "data"
INPUT_FEATURE_DIR = DATA_PREPARATION_OUTPUT_DIR / "input_features"
TRAIN_PART_DIR = DATA_DIR / "train_parts"
VALID_PART_DIR = DATA_DIR / "valid_parts"
STATE_TEXT_DIR = DATA_PREPARATION_OUTPUT_DIR / "state_text"
TOKENIZED_DIR = DATA_PREPARATION_OUTPUT_DIR / "tokenized"
TOKENIZED_TRAIN_PART_DIR = TOKENIZED_DIR / "vit_train_parts"
TOKENIZED_VALID_PART_DIR = TOKENIZED_DIR / "vit_valid_parts"
VALIDATION_READY_FILE = TOKENIZED_VALID_PART_DIR / "_READY.json"
TOKENIZER_DIR = DATA_PREPARATION_OUTPUT_DIR / "tokenizer"
SCALER_FILE = DATA_PREPARATION_OUTPUT_DIR / "scaler.json"
DATA_PREPARATION_LOG_DIR = DATA_PREPARATION_OUTPUT_DIR / "log"

MODEL_DIR = TRAINING_OUTPUT_DIR / "model"
LOG_DIR = TRAINING_OUTPUT_DIR / "log"
STATE_FEATURE_DIR = TRAINING_OUTPUT_DIR / "state_features"
PREDICTION_DIR = TRAINING_OUTPUT_DIR / "prediction" / "frozen_vit"

QWEN_BASE_DIR = Path(
    os.environ.get("QWEN3_4B_BASE_PATH", PROJECT_ROOT / "0 基座模型_Qwen3-4B-Base")
).expanduser().resolve()

# 新构造的1000条轨迹、101维Oracle标签。运行中的.tmp文件不会被读取。
SOURCE_VIT_LABELS_101 = TRAJECTORY_ORACLE_OUTPUT_DIR / "V_it_labels_101.parquet"
SOURCE_VIT_LABELS_DIAGNOSTICS = (
    TRAJECTORY_ORACLE_OUTPUT_DIR / "V_it_labels_101_diagnostics.json"
)
SOURCE_VIT_LABELS_TEMP = TRAJECTORY_ORACLE_OUTPUT_DIR / "V_it_labels_101.parquet.tmp"
SOURCE_VIT_VALID_LABELS_101 = TRAJECTORY_ORACLE_OUTPUT_DIR / "V_it_valid_labels_101.parquet"
SOURCE_VIT_VALID_LABELS_DIAGNOSTICS = (
    TRAJECTORY_ORACLE_OUTPUT_DIR / "V_it_valid_labels_101_diagnostics.json"
)
SOURCE_VIT_VALID_LABELS_TEMP = (
    TRAJECTORY_ORACLE_OUTPUT_DIR / "V_it_valid_labels_101.parquet.tmp"
)

BASE_FEATURE_FILE = (
    DRIVER_HISTORY_OUTPUT_DIR / "2 Vit基础特征构造（结果）.parquet"
)
DRIVER_HISTORY_METADATA_FILE = (
    DRIVER_HISTORY_OUTPUT_DIR / "1 司机历史收益特征构造（元数据）.json"
)
REGION_HISTORY_FILE = (
    PROJECT_ROOT / "1 数据清洗" / "3_3 区域-15min聚合（补全空值）结果.parquet"
)
OD_HISTORY_FILE = PROJECT_ROOT / "1 数据清洗" / "2_2 OD-15min聚合_属于芝加哥.parquet"
NEIGHBOR_FILE = PROJECT_ROOT / "1 数据清洗" / "4 邻近区域关系表构造结果.parquet"
REGION_MAP_FILE = (
    PROJECT_ROOT / "3 随机变量分位数预测" / "2 πijt构建" / "output"
    / "prepared_index" / "region_mapping.csv"
)

# 6月11—24日使用新随机变量预测分布。
DIT_PREDICTION_FILE = (
    PROJECT_ROOT / "3 随机变量分位数预测" / "1 Dit预测" / "output"
    / "prediction" / "dit_train_predictions.parquet"
)
DIT_VALID_PREDICTION_FILE = (
    PROJECT_ROOT / "3 随机变量分位数预测" / "1 Dit预测" / "output"
    / "prediction" / "dit_valid_predictions.parquet"
)
SIT_PREDICTION_FILE = (
    PROJECT_ROOT / "3 随机变量分位数预测" / "3 Sit构建" / "output"
    / "S_it_weekday7_unconstrained_quantiles.parquet"
)
# 新Sit从6月11日开始。6月1—10日仅使用目标日之前的历史供给代理回退。
SIT_HISTORICAL_REFERENCE_FILE = (
    PROJECT_ROOT / "3 随机变量分位数预测" / "3 Sit构建" / "output"
    / "S_it_historical_reference.parquet"
)

TRAIN_START = "2024-06-01 00:00:00"
TRAIN_END = "2024-06-24 23:45:00"
VALID_START = "2024-06-25 00:00:00"
VALID_END = "2024-06-27 23:45:00"
PREDICT_START = "2024-06-11 00:00:00"
PREDICT_END = "2024-06-24 23:45:00"
DATE_RANGES = {
    "train": (TRAIN_START, TRAIN_END),
    "valid": (VALID_START, VALID_END),
    "predict": (PREDICT_START, PREDICT_END),
}

QUANTILE_LEVELS = np.asarray(
    [0.005] + [level / 100 for level in range(1, 100)] + [0.995], dtype=np.float32
)
SOURCE_LABEL_COLUMNS = [f"oracle_V_q_{level:.3f}" for level in QUANTILE_LEVELS]
QUANTILE_COLUMNS = [f"V_q_{level:.3f}" for level in QUANTILE_LEVELS]
RANDOM_VARIABLE_QUANTILE_COLUMNS = [f"q_{level:.3f}" for level in QUANTILE_LEVELS]
KEY_LEVELS = (0.005, 0.05, 0.10, 0.50, 0.90, 0.95, 0.995)
SUMMARY_FEATURE_DIM = 27
DEMAND_FEATURE_DIM = SUMMARY_FEATURE_DIM
DEMAND_FEATURE_COLUMNS = [f"demand_feature_{index:02d}" for index in range(SUMMARY_FEATURE_DIM)]
SUPPLY_FEATURE_COLUMNS = [f"supply_feature_{index:02d}" for index in range(SUMMARY_FEATURE_DIM)]

BEST_LORA_DIR = MODEL_DIR / "best_lora"
BEST_STATE_PROJECTION_FILE = MODEL_DIR / "best_state_projection.pt"
BEST_QUANTILE_HEAD_FILE = MODEL_DIR / "best_quantile_head.pt"
TRAINING_HISTORY_FILE = LOG_DIR / "training_history.csv"
BEST_MODEL_INFO_FILE = LOG_DIR / "best_model_info.json"
BATCH_BENCHMARK_FILE = LOG_DIR / "batch_size_benchmark.json"

RANDOM_SEED = 42
TOP_K_DESTINATIONS = 10
TOP_K_NEIGHBORS = 5
HISTORY_SLOT_RADIUS = 1
HISTORY_MIN_SAMPLES = 6
TIME_STEPS_PER_DAY = 96
EXPECTED_REGION_COUNT = 801
EXPECTED_TRAIN_ROWS = 24 * TIME_STEPS_PER_DAY * EXPECTED_REGION_COUNT
EXPECTED_VALID_ROWS = 3 * TIME_STEPS_PER_DAY * EXPECTED_REGION_COUNT

# 全量训练：不再存在25000/5000抽样上限。
MAX_LENGTH = 4096
TOKENIZER_BATCH_SIZE = 128
TOKENIZER_NUM_PROC = int(os.environ.get("VIT_TOKENIZER_NUM_PROC", "8"))
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
VALIDATION_CHECKS_PER_EPOCH = 5
VALIDATION_SAMPLE_DENOMINATOR = 10
VALIDATION_SAMPLE_SEED = RANDOM_SEED
EVAL_BATCH_SIZE = 8
EFFECTIVE_BATCH_SIZE = 32
BATCH_BENCHMARK_CANDIDATES = (2, 4)
BATCH_BENCHMARK_SAMPLES = 640
BATCH_BENCHMARK_WARMUP_STEPS = 20
USE_LENGTH_GROUPED_SAMPLER = True
LORA_LEARNING_RATE = 5e-5
HEAD_LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.03
MAX_GRAD_NORM = 1.0
NUM_WORKERS = 4
USE_GRADIENT_CHECKPOINTING = False
DEVICE = "auto"
EXPORT_BATCH_SIZE = int(os.environ.get("VIT_EXPORT_BATCH_SIZE", "16"))


def days(name: str):
    import pandas as pd

    start, end = DATE_RANGES[name]
    return list(pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), freq="D"))


def ensure_directories() -> None:
    for path in (
        UPSTREAM_OUTPUT_DIR, TRAJECTORY_ORACLE_OUTPUT_DIR, DRIVER_HISTORY_OUTPUT_DIR,
        DATA_PREPARATION_OUTPUT_DIR, DATA_DIR, INPUT_FEATURE_DIR, TRAIN_PART_DIR,
        VALID_PART_DIR, STATE_TEXT_DIR, TOKENIZED_DIR, TOKENIZED_TRAIN_PART_DIR,
        TOKENIZED_VALID_PART_DIR, TOKENIZER_DIR,
        DATA_PREPARATION_LOG_DIR, TRAINING_OUTPUT_DIR, MODEL_DIR, LOG_DIR,
        STATE_FEATURE_DIR, PREDICTION_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
