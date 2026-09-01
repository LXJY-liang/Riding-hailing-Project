# -*- coding: utf-8 -*-
"""在验证集或测试集上评估预训练Vit分位数曲线。"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer


HERE = Path(__file__).resolve().parent


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


cfg = load(HERE.parent / "1 数据准备" / "1 Vit预训练配置.py", "vit_pretrain_config")
model_module = load(HERE / "1 定义Vit分位数模型.py", "vit_quantile_model")


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, samples):
        batch = self.tokenizer.pad(
            [{"input_ids": row["input_ids"], "attention_mask": row["attention_mask"]} for row in samples],
            padding=True,
            return_tensors="pt",
        )
        batch["log_supply"] = torch.tensor([row["log_supply"] for row in samples], dtype=torch.float32)
        batch["demand_summary"] = torch.tensor(
            [row["demand_summary"] for row in samples], dtype=torch.float32
        )
        batch["labels"] = torch.tensor([row["labels"] for row in samples], dtype=torch.float32)
        return batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else (
        torch.float16 if device.type == "cuda" else torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.TOKENIZER_DIR, local_files_only=True, trust_remote_code=True)
    data = load_from_disk(str(cfg.TOKENIZED_SPLIT_DIRS[args.split]))
    loader = DataLoader(
        data,
        batch_size=cfg.EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=Collator(tokenizer),
        num_workers=cfg.NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    model = model_module.VitQuantileModel(
        model_dtype=dtype,
        adapter_path=cfg.BEST_LORA_DIR,
        is_trainable=False,
    )
    model.state_projection.load_state_dict(torch.load(cfg.BEST_STATE_PROJECTION_FILE, map_location="cpu", weights_only=True))
    model.quantile_head.load_state_dict(torch.load(cfg.BEST_QUANTILE_HEAD_FILE, map_location="cpu", weights_only=True))
    model = model.to(device).eval()
    with cfg.SCALER_FILE.open("r", encoding="utf-8") as stream:
        scaler = json.load(stream)

    curve_error = 0.0
    median_error = 0.0
    count = 0
    median_index = int(np.argmin(np.abs(cfg.QUANTILE_LEVELS - 0.50)))
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"评估 {args.split}", dynamic_ncols=True):
            prediction = model(
                batch["input_ids"].to(device, non_blocking=True),
                batch["attention_mask"].to(device, non_blocking=True),
                batch["log_supply"].to(device, non_blocking=True),
                batch["demand_summary"].to(device, non_blocking=True),
            ).float().cpu()
            target = batch["labels"]
            prediction = prediction * scaler["label_std"] + scaler["label_mean"]
            target = target * scaler["label_std"] + scaler["label_mean"]
            curve_error += torch.abs(prediction - target).sum().item()
            median_error += torch.abs(prediction[:, median_index] - target[:, median_index]).sum().item()
            count += len(target)

    metrics = {
        "split": args.split,
        "rows": count,
        "mean_absolute_quantile_curve_error": curve_error / (count * len(cfg.QUANTILE_LEVELS)),
        "q50_mae": median_error / count,
    }
    output = cfg.LOG_DIR / f"evaluation_{args.split}.json"
    with output.open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
