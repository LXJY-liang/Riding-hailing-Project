# -*- coding: utf-8 -*-
"""
9 Vit预训练.py

功能：
1. 加载提前Tokenize的训练集和验证集；
2. 冻结Qwen基座，只训练Vit专用LoRA和单值价值头；
3. 使用MSE损失进行监督式预训练；
4. 保存验证损失最低的LoRA和价值头。
"""

import importlib.util
import json
import math
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


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
# 2. 训练辅助函数
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    """按当前batch动态padding并整理单值标签。"""

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


def evaluate(
    model,
    data_loader,
    device,
    amp_dtype,
):
    """计算完整验证集MSE。"""

    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.inference_mode():
        for batch in tqdm(
            data_loader,
            desc="验证",
            dynamic_ncols=True,
        ):
            input_ids = batch[
                "input_ids"
            ].to(device)

            attention_mask = batch[
                "attention_mask"
            ].to(device)

            labels = batch[
                "labels"
            ].to(device)

            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=device.type == "cuda",
            ):
                predictions = model(
                    input_ids,
                    attention_mask,
                )

                loss = torch.nn.functional.mse_loss(
                    predictions,
                    labels,
                )

            batch_size = labels.size(0)

            total_loss += (
                loss.item()
                * batch_size
            )

            total_samples += batch_size

    return total_loss / total_samples


def save_best_model(
    model,
    epoch,
    valid_loss,
):
    """保存验证损失最低的LoRA和单值价值头。"""

    if cfg.best_lora_dir.exists():
        shutil.rmtree(
            cfg.best_lora_dir
        )

    model.backbone.save_pretrained(
        str(cfg.best_lora_dir)
    )

    torch.save(
        model.value_head.state_dict(),
        cfg.best_value_head_file,
    )

    with open(
        cfg.best_model_info_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "best_epoch": epoch,
                "best_valid_loss": valid_loss,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# 3. 预训练
# ============================================================

def main():
    set_seed(cfg.seed)

    device = get_device()
    model_dtype = get_model_dtype(device)

    amp_dtype = (
        torch.bfloat16
        if model_dtype == torch.bfloat16
        else torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(cfg.tokenizer_dir),
        local_files_only=True,
        trust_remote_code=True,
    )

    tokenizer.padding_side = cfg.padding_side

    train_dataset = load_from_disk(
        str(cfg.train_tokenized_dir)
    )

    valid_dataset = load_from_disk(
        str(cfg.valid_tokenized_dir)
    )

    collator = VitCollator(
        tokenizer
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = model_module.VitValueModel(
        model_dtype=model_dtype,
        adapter_path=None,
        is_trainable=True,
    ).to(device)

    lora_parameters = [
        parameter
        for parameter in model.backbone.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        [
            {
                "params": lora_parameters,
                "lr": cfg.lora_learning_rate,
            },
            {
                "params": model.value_head.parameters(),
                "lr": cfg.head_learning_rate,
            },
        ],
        weight_decay=cfg.weight_decay,
    )

    updates_per_epoch = math.ceil(
        len(train_loader)
        / cfg.gradient_accumulation_steps
    )

    total_updates = (
        updates_per_epoch
        * cfg.num_epochs
    )

    warmup_steps = int(
        total_updates
        * cfg.warmup_ratio
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )

    use_grad_scaler = (
        device.type == "cuda"
        and model_dtype == torch.float16
    )

    grad_scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_grad_scaler,
    )

    cfg.model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cfg.log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_valid_loss = float("inf")
    history = []

    for epoch in range(
        1,
        cfg.num_epochs + 1,
    ):
        model.train()
        optimizer.zero_grad(
            set_to_none=True
        )

        total_train_loss = 0.0
        total_train_samples = 0

        progress = tqdm(
            train_loader,
            desc=(
                f"预训练 Epoch "
                f"{epoch}/{cfg.num_epochs}"
            ),
            dynamic_ncols=True,
        )

        for step, batch in enumerate(
            progress,
            start=1,
        ):
            input_ids = batch[
                "input_ids"
            ].to(device)

            attention_mask = batch[
                "attention_mask"
            ].to(device)

            labels = batch[
                "labels"
            ].to(device)

            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=device.type == "cuda",
            ):
                predictions = model(
                    input_ids,
                    attention_mask,
                )

                loss = torch.nn.functional.mse_loss(
                    predictions,
                    labels,
                )

                scaled_loss = (
                    loss
                    / cfg.gradient_accumulation_steps
                )

            grad_scaler.scale(
                scaled_loss
            ).backward()

            should_update = (
                step
                % cfg.gradient_accumulation_steps
                == 0
                or step == len(train_loader)
            )

            if should_update:
                grad_scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg.max_grad_norm,
                )

                grad_scaler.step(
                    optimizer
                )

                grad_scaler.update()
                scheduler.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

            batch_size = labels.size(0)

            total_train_loss += (
                loss.item()
                * batch_size
            )

            total_train_samples += (
                batch_size
            )

            progress.set_postfix(
                loss=f"{loss.item():.6f}"
            )

        train_loss = (
            total_train_loss
            / total_train_samples
        )

        valid_loss = evaluate(
            model,
            valid_loader,
            device,
            amp_dtype,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
            }
        )

        pd.DataFrame(
            history
        ).to_csv(
            cfg.training_history_file,
            index=False,
        )

        print(
            f"Epoch {epoch}："
            f"训练损失={train_loss:.6f}，"
            f"验证损失={valid_loss:.6f}"
        )

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss

            save_best_model(
                model,
                epoch,
                valid_loss,
            )


if __name__ == "__main__":
    main()
