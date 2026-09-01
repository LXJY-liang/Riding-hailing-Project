# -*- coding: utf-8 -*-
"""
7 Vit的Tokenizer.py

功能：
1. 加载本地Qwen3-4B-Base Tokenizer；
2. 对train/valid/test文本进行Tokenize；
3. 不提前padding，训练时按batch动态padding；
4. 保存Tokenize结果。
"""

import importlib.util
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


# ============================================================
# 1. 加载配置
# ============================================================

module_dir = Path(__file__).resolve().parent

spec = importlib.util.spec_from_file_location(
    "vit_config",
    module_dir / "5 Vit配置.py",
)

cfg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)


# ============================================================
# 2. Tokenizer
# ============================================================

cfg.tokenizer_dir.mkdir(
    parents=True,
    exist_ok=True,
)

cfg.tokenized_dir.mkdir(
    parents=True,
    exist_ok=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    cfg.base_model_path,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = cfg.padding_side

tokenizer.save_pretrained(
    cfg.tokenizer_dir
)


# ============================================================
# 3. 读取数据
# ============================================================

dataset = load_dataset(
    "parquet",
    data_files={
        "train": str(cfg.train_data_file),
        "valid": str(cfg.valid_data_file),
        "test": str(cfg.test_data_file),
    },
)


# ============================================================
# 4. Tokenize
# ============================================================

def tokenize_batch(batch):
    """只截断，不提前padding。"""

    encoded = tokenizer(
        batch["text"],
        truncation=True,
        max_length=cfg.max_length,
        padding=False,
    )

    encoded["labels"] = batch["label_scaled"]

    return encoded


for split_name, output_dir in [
    ("train", cfg.train_tokenized_dir),
    ("valid", cfg.valid_tokenized_dir),
    ("test", cfg.test_tokenized_dir),
]:
    tokenized = dataset[split_name].map(
        tokenize_batch,
        batched=True,
        batch_size=cfg.tokenizer_batch_size,
        num_proc=cfg.tokenizer_num_proc,
        remove_columns=dataset[split_name].column_names,
    )

    tokenized.save_to_disk(
        str(output_dir)
    )
