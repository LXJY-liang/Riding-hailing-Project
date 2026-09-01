# -*- coding: utf-8 -*-
"""
10 Vit预训练评估.py

功能：
1. 重新加载验证损失最低的Vit专用LoRA和单值价值头；
2. 在测试集上预测Vit；
3. 输出MAE和RMSE；
4. 保存测试集预测结果。
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer


# ============================================================
# 1. 加载模块
# ============================================================

module_dir = Path(__file__).resolve().parent


def load_module(file_name, module_name):
    spec = importlib.util.spec_from_file_location(
        module_name,
        module_dir / file_name,
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


cfg = load_module(
    "5 Vit配置.py",
    "vit_config",
)

model_module = load_module(
    "8 Vit模型.py",
    "vit_model",
)


# ============================================================
# 2. 设备与数据
# ============================================================

def get_device():
    if cfg.device != "auto":
        return torch.device(cfg.device)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def get_model_dtype(device):
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16

        return torch.float16

    if device.type == "mps":
        return torch.float16

    return torch.float32


class VitCollator:
    """按当前batch动态padding。"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, samples):
        labels = torch.tensor(
            [
                sample["labels"]
                for sample in samples
            ],
            dtype=torch.float32,
        )

        model_inputs = [
            {
                "input_ids": sample["input_ids"],
                "attention_mask": sample["attention_mask"],
            }
            for sample in samples
        ]

        batch = self.tokenizer.pad(
            model_inputs,
            padding=True,
            return_tensors="pt",
        )

        batch["labels"] = labels

        return batch


# ============================================================
# 3. 加载最佳模型
# ============================================================

device = get_device()
model_dtype = get_model_dtype(device)

tokenizer = AutoTokenizer.from_pretrained(
    str(cfg.tokenizer_dir),
    local_files_only=True,
    trust_remote_code=True,
)

tokenizer.padding_side = cfg.padding_side

test_dataset = load_from_disk(
    str(cfg.test_tokenized_dir)
)

test_loader = DataLoader(
    test_dataset,
    batch_size=cfg.eval_batch_size,
    shuffle=False,
    collate_fn=VitCollator(tokenizer),
    num_workers=cfg.num_workers,
    pin_memory=device.type == "cuda",
)

model = model_module.VitValueModel(
    model_dtype=model_dtype,
    adapter_path=cfg.best_lora_dir,
    is_trainable=False,
)

model.value_head.load_state_dict(
    torch.load(
        cfg.best_value_head_file,
        map_location="cpu",
    )
)

model.to(device)
model.eval()


# ============================================================
# 4. 测试集预测
# ============================================================

predictions_scaled = []

with torch.inference_mode():
    for batch in tqdm(
        test_loader,
        desc="测试",
        dynamic_ncols=True,
    ):
        input_ids = batch[
            "input_ids"
        ].to(device)

        attention_mask = batch[
            "attention_mask"
        ].to(device)

        prediction = model(
            input_ids,
            attention_mask,
        )

        predictions_scaled.append(
            prediction.float()
            .cpu()
            .numpy()
        )


predictions_scaled = np.concatenate(
    predictions_scaled
)

with open(
    cfg.label_scaler_file,
    "r",
    encoding="utf-8",
) as file:
    scaler = json.load(file)

predictions = (
    predictions_scaled
    * scaler["label_std"]
    + scaler["label_mean"]
)

test_data = pd.read_parquet(
    cfg.test_data_file,
    columns=[
        cfg.region_column,
        cfg.time_column,
        cfg.label_column,
    ],
)

labels = test_data[
    cfg.label_column
].to_numpy(dtype=np.float32)

mae = float(
    np.mean(
        np.abs(
            labels - predictions
        )
    )
)

rmse = float(
    np.sqrt(
        np.mean(
            (
                labels - predictions
            ) ** 2
        )
    )
)

result = test_data.copy()

result["Vit_prediction"] = predictions

cfg.prediction_dir.mkdir(
    parents=True,
    exist_ok=True,
)

result.to_parquet(
    cfg.test_prediction_file,
    index=False,
    compression="snappy",
)

print(f"test_mae={mae:.6f}")
print(f"test_rmse={rmse:.6f}")
