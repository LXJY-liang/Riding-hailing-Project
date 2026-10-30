# -*- coding: utf-8 -*-
"""司机未来4小时历史实现值代理标签的独立配置。"""

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[2]
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT.parent / "output" / "1 司机轨迹Oracle构造"

BASE_OD_OUTPUT = INPUT_DIR / "active_od_base.parquet"
REGION_MAP_FILE = INPUT_DIR / "region_mapping.csv"
SUPPLY_OUTPUT = (
    PROJECT_ROOT
    / "3 随机变量分位数预测"
    / "3 Sit构建"
    / "output"
    / "S_it_historical_reference.parquet"
)
SUPPLY_VALUE_COLUMN = "S_reference_it"

VIT_OUTPUT = OUTPUT_DIR / "V_it_labels_101.parquet"
VIT_DIAGNOSTICS = OUTPUT_DIR / "V_it_labels_101_diagnostics.json"
VIT_VALID_OUTPUT = OUTPUT_DIR / "V_it_valid_labels_101.parquet"
VIT_VALID_DIAGNOSTICS = OUTPUT_DIR / "V_it_valid_labels_101_diagnostics.json"

DATA_START = "2024-06-01 00:00:00"
DATA_END = "2024-06-30 23:45:00"
LABEL_START = "2024-06-01 00:00:00"
LABEL_END = "2024-06-24 23:45:00"
VALID_LABEL_START = "2024-06-25 00:00:00"
VALID_LABEL_END = "2024-06-27 23:45:00"

TIME_STEP_MINUTES = 15
PLATFORM_COMMISSION_RATE = 0.25
DRIVER_VALUE_HORIZON = 16
DRIVER_GAMMA = 0.95
VIT_MONTE_CARLO_SAMPLES = 1000
VIT_RANDOM_SEED = 20260914
VIT_QUANTILE_LEVELS = np.asarray(
    [0.005] + [level / 100 for level in range(1, 100)] + [0.995],
    dtype=np.float32,
)
VIT_QUANTILE_COLUMNS = [f"oracle_V_q_{level:.3f}" for level in VIT_QUANTILE_LEVELS]


def ensure_output_directories() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
