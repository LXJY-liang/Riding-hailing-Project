# -*- coding: utf-8 -*-

########### max_length = 1024    # 需要根据实际prompt的token长度进行调整，否则会发生截断#######

"""
5 Vit配置.py

Vit预训练配置：
1. 输入3 Vit预测特征拼接生成的train/valid/test特征；
2. 使用4 Vit标签构造生成的Vit_label；
3. Qwen3-4B-Base + Vit专用LoRA + 单值价值头；
4. 预训练阶段使用监督损失，不进行TD更新。
"""

import os
from pathlib import Path


# ============================================================
# 1. 项目路径
# ============================================================

project_root = Path(
    os.environ.get(
        "RIDING_HAILING_PROJECT_ROOT",
        Path(__file__).resolve().parents[2],
    )
).expanduser().resolve()

vit_dir = project_root / "4 司机未来价值预测模型"
code_dir = vit_dir / "2 Vit预训练模型"

base_model_path = project_root / "0 基座模型_Qwen3-4B-Base"

train_feature_file = vit_dir / "Vit_train_feature.parquet"
valid_feature_file = vit_dir / "Vit_valid_feature.parquet"
test_feature_file = vit_dir / "Vit_test_feature.parquet"

label_file = vit_dir / "4 Vit标签构造（结果）.parquet"


# ============================================================
# 2. 输出路径
# ============================================================

output_dir = code_dir / "output"
data_dir = output_dir / "data"
tokenized_dir = output_dir / "tokenized"
model_dir = output_dir / "model"
prediction_dir = output_dir / "prediction"
log_dir = output_dir / "log"

train_data_file = data_dir / "vit_train.parquet"
valid_data_file = data_dir / "vit_valid.parquet"
test_data_file = data_dir / "vit_test.parquet"

train_tokenized_dir = tokenized_dir / "vit_train_tokenized"
valid_tokenized_dir = tokenized_dir / "vit_valid_tokenized"
test_tokenized_dir = tokenized_dir / "vit_test_tokenized"

tokenizer_dir = model_dir / "tokenizer"
best_lora_dir = model_dir / "best_lora"
best_value_head_file = model_dir / "best_value_head.pt"
label_scaler_file = model_dir / "label_scaler.json"

training_history_file = log_dir / "training_history.csv"
best_model_info_file = log_dir / "best_model_info.json"
test_prediction_file = prediction_dir / "vit_test_predictions.parquet"


# ============================================================
# 3. 字段设置
# ============================================================

region_column = "region_id"
time_column = "t"
label_column = "Vit_label"


# ============================================================
# 4. Tokenizer设置
# ============================================================

max_length = 1024    # 需要根据实际prompt的token长度进行调整，否则会发生截断
padding_side = "right"
tokenizer_batch_size = 512
tokenizer_num_proc = 8


# ============================================================
# 5. LoRA与单值价值头设置
# ============================================================

lora_rank = 16
lora_alpha = 32
lora_dropout = 0.05
lora_target_modules = ["q_proj", "v_proj"]

value_head_hidden_size = 512
value_head_dropout = 0.10


# ============================================================
# 6. 训练设置
# ============================================================

seed = 42
num_epochs = 3

train_batch_size = 1
eval_batch_size = 8
gradient_accumulation_steps = 32

lora_learning_rate = 5e-5
head_learning_rate = 1e-4
weight_decay = 0.01
warmup_ratio = 0.03
max_grad_norm = 1.0

num_workers = 4
use_gradient_checkpointing = True
use_standardization = True

# auto：优先CUDA，其次MPS，最后CPU
device = "auto"
