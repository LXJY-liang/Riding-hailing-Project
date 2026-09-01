# -*- coding: utf-8 -*-
"""监督预训练Qwen3-4B、Vit专用LoRA、状态投影和101维分位数头。"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup


HERE = Path(__file__).resolve().parent


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


cfg = load(HERE.parent / "1 数据准备" / "1 Vit预训练配置.py", "vit_pretrain_config")
model_module = load(HERE / "1 定义Vit分位数模型.py", "vit_quantile_model")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_and_dtype() -> tuple[torch.device, torch.dtype]:
    if cfg.DEVICE != "auto":
        device = torch.device(cfg.DEVICE)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif device.type == "mps":
        dtype = torch.float16
    else:
        dtype = torch.float32
    return device, dtype


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, samples):
        batch = self.tokenizer.pad(
            [
                {"input_ids": sample["input_ids"], "attention_mask": sample["attention_mask"]}
                for sample in samples
            ],
            padding=True,
            return_tensors="pt",
        )
        batch["log_supply"] = torch.tensor([sample["log_supply"] for sample in samples], dtype=torch.float32)
        batch["demand_summary"] = torch.tensor(
            [sample["demand_summary"] for sample in samples], dtype=torch.float32
        )
        batch["labels"] = torch.tensor([sample["labels"] for sample in samples], dtype=torch.float32)
        return batch


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in tqdm(loader, desc="验证", dynamic_ncols=True):
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            prediction = model(
                batch["input_ids"].to(device, non_blocking=True),
                batch["attention_mask"].to(device, non_blocking=True),
                batch["log_supply"].to(device, non_blocking=True),
                batch["demand_summary"].to(device, non_blocking=True),
            )
            loss = model_module.quantile_curve_loss(prediction, labels)
        total += float(loss.item()) * len(labels)
        count += len(labels)
    return total / max(count, 1)


def save_best(model, epoch: int, valid_loss: float) -> None:
    if cfg.BEST_LORA_DIR.exists():
        shutil.rmtree(cfg.BEST_LORA_DIR)
    model.backbone.save_pretrained(cfg.BEST_LORA_DIR)
    torch.save(model.state_projection.state_dict(), cfg.BEST_STATE_PROJECTION_FILE)
    torch.save(model.quantile_head.state_dict(), cfg.BEST_QUANTILE_HEAD_FILE)
    with cfg.BEST_MODEL_INFO_FILE.open("w", encoding="utf-8") as stream:
        json.dump(
            {"best_epoch": epoch, "best_valid_loss": valid_loss},
            stream,
            ensure_ascii=False,
            indent=2,
        )


def main() -> None:
    cfg.ensure_directories()
    set_seed(cfg.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    device, model_dtype = device_and_dtype()
    amp_dtype = torch.bfloat16 if model_dtype == torch.bfloat16 else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.TOKENIZER_DIR,
        local_files_only=True,
        trust_remote_code=True,
    )
    tokenizer.padding_side = cfg.PADDING_SIDE
    collator = Collator(tokenizer)
    train_data = load_from_disk(str(cfg.TOKENIZED_SPLIT_DIRS["train"]))
    valid_data = load_from_disk(str(cfg.TOKENIZED_SPLIT_DIRS["valid"]))
    train_loader = DataLoader(
        train_data,
        batch_size=cfg.TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.NUM_WORKERS > 0,
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=cfg.EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.NUM_WORKERS > 0,
    )

    model = model_module.VitQuantileModel(model_dtype=model_dtype).to(device)
    lora_parameters = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
    head_parameters = list(model.state_projection.parameters()) + list(model.quantile_head.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_parameters, "lr": cfg.LORA_LEARNING_RATE},
            {"params": head_parameters, "lr": cfg.HEAD_LEARNING_RATE},
        ],
        weight_decay=cfg.WEIGHT_DECAY,
    )
    updates_per_epoch = math.ceil(len(train_loader) / cfg.GRADIENT_ACCUMULATION_STEPS)
    total_updates = updates_per_epoch * cfg.NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_updates * cfg.WARMUP_RATIO),
        num_training_steps=total_updates,
    )

    best_loss = float("inf")
    with cfg.TRAINING_HISTORY_FILE.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "train_loss", "valid_loss"])
        writer.writeheader()
        for epoch in range(1, cfg.NUM_EPOCHS + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            total_rows = 0
            progress = tqdm(train_loader, desc=f"训练 epoch {epoch}", dynamic_ncols=True)
            for step, batch in enumerate(progress, start=1):
                labels = batch["labels"].to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                    prediction = model(
                        batch["input_ids"].to(device, non_blocking=True),
                        batch["attention_mask"].to(device, non_blocking=True),
                        batch["log_supply"].to(device, non_blocking=True),
                        batch["demand_summary"].to(device, non_blocking=True),
                    )
                    loss = model_module.quantile_curve_loss(prediction, labels)
                    scaled_loss = loss / cfg.GRADIENT_ACCUMULATION_STEPS
                scaled_loss.backward()
                if step % cfg.GRADIENT_ACCUMULATION_STEPS == 0 or step == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.MAX_GRAD_NORM)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                total_loss += float(loss.item()) * len(labels)
                total_rows += len(labels)
                progress.set_postfix(loss=f"{loss.item():.4f}")

            train_loss = total_loss / max(total_rows, 1)
            valid_loss = evaluate(model, valid_loader, device, amp_dtype)
            writer.writerow({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss})
            stream.flush()
            print(f"epoch={epoch}, train_loss={train_loss:.6f}, valid_loss={valid_loss:.6f}")
            if valid_loss < best_loss:
                best_loss = valid_loss
                save_best(model, epoch, valid_loss)
                print("已保存新的最佳Vit分位数模型。")


if __name__ == "__main__":
    main()
