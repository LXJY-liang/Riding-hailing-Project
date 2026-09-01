import os
from pathlib import Path

import numpy as np


# =========================
# 1. 项目路径
# =========================

code_dir = Path(__file__).resolve().parent

project_root = Path(
    os.environ.get(
        "RIDING_HAILING_PROJECT_ROOT",
        code_dir.parents[1],
    )
).expanduser().resolve()

base_model_path = Path(
    os.environ.get(
        "QWEN3_4B_BASE_PATH",
        project_root / "0 基座模型_Qwen3-4B-Base",
    )
).expanduser().resolve()

feature_data_path = (
    project_root
    / "2 模型特征构建"
    / "4 tauijt特征构造（结果）.parquet"
)

output_dir = code_dir / "output"
data_output_dir = output_dir / "data"
tokenized_output_dir = output_dir / "tokenized"
model_output_dir = output_dir / "model"
checkpoint_output_dir = output_dir / "checkpoints"
prediction_output_dir = output_dir / "prediction"
log_dir = output_dir / "log"

train_data_path = data_output_dir / "tauijt_train.parquet"
valid_data_path = data_output_dir / "tauijt_valid.parquet"
test_data_path = data_output_dir / "tauijt_test.parquet"

train_tokenized_path = tokenized_output_dir / "tauijt_train_tokenized"
valid_tokenized_path = tokenized_output_dir / "tauijt_valid_tokenized"
test_tokenized_path = tokenized_output_dir / "tauijt_test_tokenized"

tokenizer_path = model_output_dir / "tokenizer"
lora_adapter_path = model_output_dir / "best_lora"
quantile_head_path = model_output_dir / "best_quantile_head.pt"
label_scaler_path = model_output_dir / "label_scaler.json"
training_history_path = log_dir / "training_history.csv"
best_model_info_path = log_dir / "best_model_info.json"


# =========================
# 2. 数据划分
# =========================

train_start = "2024-06-01 00:00:00"
train_end = "2024-06-24 23:45:00"

valid_start = "2024-06-25 00:00:00"
valid_end = "2024-06-27 23:45:00"

test_start = "2024-06-28 00:00:00"
test_end = "2024-06-30 23:45:00"


# =========================
# 3. 字段设置
# =========================

i_column = "i"
j_column = "j"
time_column = "t"
od_id_column = "OD_id"
label_column = "tau_ijt_label"

feature_columns = [
    "i",
    "j",
    "t",
    "OD_id",
    "dist_ij",
    "same_region_ij",
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
    "tau_ij_lag1",
    "tau_ij_lag2",
    "tau_ij_roll4_mean",
    "tau_ij_roll8_mean",
    "tau_ij_same_slot_last_day",
    "tau_ij_same_slot_last_week",
    "tau_ij_mean_last_day",
    "tau_ij_median_last_day_proxy",
    "tau_ij_mean_last3",
    "tau_ij_mean_last7",
    "tau_ij_count_last7",
    "od_tau_missing_flag",
    "global_tau_by_dist_slot",
    "similar_OD_tau_mean",
]

all_columns = feature_columns + [label_column]


# =========================
# 4. 分位数设置
# =========================

# 共101个分位数：0.005、0.01至0.99、0.995
quantile_levels = np.asarray(
    [0.005]
    + [level / 100 for level in range(1, 100)]
    + [0.995],
    dtype=np.float32,
)

quantile_names = [
    f"q_{level:.3f}"
    for level in quantile_levels
]

num_quantiles = len(quantile_levels)


# =========================
# 5. 文本和Tokenizer设置
# =========================

max_length = 512
padding_side = "right"
tokenizer_batch_size = 1000
tokenizer_num_proc = 8

# =========================
# 6. LoRA和分位数头设置
# =========================

lora_rank = 16
lora_alpha = 32
lora_dropout = 0.05
lora_target_modules = ["q_proj", "v_proj"]
lora_layer_start = 24
lora_layer_end = 35

quantile_head_hidden_size = 512
quantile_head_dropout = 0.10
quantile_head_initial_lower_bias = -3.0
quantile_head_initial_slope_bias = 1.0
smooth_pinball_alpha = 0.05
crossing_loss_weight = 0.0


# =========================
# 7. 训练设置
# =========================

seed = 42
train_sample_fraction = 0.12
num_epochs = 1
train_batch_size = 8
eval_batch_size = 16
gradient_accumulation_steps = 4
checkpoint_save_steps = 500
# 12%抽样时约13333个optimizer step，本轮默认只在epoch末做全量验证。
validation_steps = 20000
resume_from_checkpoint = os.environ.get("TAUIJT_RESUME_FROM_CHECKPOINT")

lora_learning_rate = 5e-5
head_learning_rate = 1e-4
weight_decay = 0.01
warmup_ratio = 0.03
max_grad_norm = 1.0

num_workers = 4
use_gradient_checkpointing = False
use_log1p = True
use_standardization = True

# auto：优先CUDA，其次MPS，最后CPU
device = "auto"
