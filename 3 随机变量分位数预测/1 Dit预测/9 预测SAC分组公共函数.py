# -*- coding: utf-8 -*-
"""使用已经训练好的Dit模型，导出指定SAC日期区间的101维预测。"""

from __future__ import annotations

import importlib.util
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


HERE = Path(__file__).resolve().parent


def load_module(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


evaluator = load_module("6 Dit评估与测试.py", "dit_evaluator_for_sac")
cfg = evaluator.config


def model_paths() -> tuple[Path, Path, Path, Path]:
    """优先读取当前模型；不存在时读取12%样本训练的归档模型。"""

    if cfg.lora_adapter_path.exists() and cfg.quantile_head_path.exists():
        return (
            cfg.lora_adapter_path,
            cfg.quantile_head_path,
            cfg.label_scaler_path,
            cfg.tokenizer_path,
        )
    legacy = HERE / "output" / "legacy_training_results" / "model"
    tokenizer = cfg.tokenizer_path if cfg.tokenizer_path.exists() else legacy / "tokenizer"
    return (
        legacy / "best_lora",
        legacy / "best_quantile_head.pt",
        legacy / "label_scaler.json",
        tokenizer,
    )


def export_sac_split(split: str, start: str, end: str) -> Path:
    raw_path, tokenized_path = {
        "train": (cfg.train_data_path, cfg.train_tokenized_path),
        "valid": (cfg.valid_data_path, cfg.valid_tokenized_path),
        "test": (cfg.test_data_path, cfg.test_tokenized_path),
    }[split]
    raw = pd.read_parquet(
        raw_path,
        columns=[cfg.region_column, cfg.time_column, cfg.label_column],
    ).reset_index(drop=True)
    raw[cfg.time_column] = pd.to_datetime(raw[cfg.time_column])
    selected = np.flatnonzero(
        raw[cfg.time_column].between(pd.Timestamp(start), pd.Timestamp(end)).to_numpy()
    )
    raw = raw.iloc[selected].reset_index(drop=True)
    dataset = load_from_disk(str(tokenized_path)).select(selected.tolist())

    adapter, head, scaler, tokenizer_path = model_paths()
    cfg.label_scaler_path = scaler
    evaluator.model_module.config.label_scaler_path = scaler
    device = evaluator.get_device()
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), local_files_only=True, trust_remote_code=True
    )
    tokenizer.padding_side = cfg.padding_side
    prediction_batch_size = int(os.environ.get("DIT_SAC_PRED_BATCH_SIZE", "32"))
    loader = DataLoader(
        dataset,
        batch_size=prediction_batch_size,
        shuffle=False,
        collate_fn=evaluator.dit_collator(tokenizer),
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.num_workers > 0,
    )
    model = evaluator.model_module.dit_quantile_model(
        model_dtype=evaluator.get_model_dtype(device),
        adapter_path=adapter,
        is_trainable=False,
    )
    model.quantile_head.load_state_dict(evaluator.torch_load(head, map_location="cpu"))
    model = model.to(device).eval()

    batches = []
    total_batches = len(loader)
    start_time = time.time()
    with torch.inference_mode():
        for batch_number, batch in enumerate(loader, start=1):
            batches.append(
                model(
                    batch["input_ids"].to(device, non_blocking=True),
                    batch["attention_mask"].to(device, non_blocking=True),
                ).float().cpu().numpy()
            )
            if batch_number == 1 or batch_number % 20 == 0 or batch_number == total_batches:
                elapsed = time.time() - start_time
                speed = batch_number / max(elapsed, 1e-6)
                remaining_hours = (total_batches - batch_number) / max(speed, 1e-6) / 3600
                completed_rows = min(batch_number * prediction_batch_size, len(dataset))
                print(
                    f"{split}：已预测 {completed_rows:,}/{len(dataset):,} 行，"
                    f"批次 {batch_number:,}/{total_batches:,}，"
                    f"速度 {speed:.2f} 批/秒，预计剩余 {remaining_hours:.2f} 小时",
                    flush=True,
                )
    predictions = evaluator.inverse_transform(
        np.concatenate(batches, axis=0), evaluator.load_scaler()
    )
    result = pd.concat(
        [raw, pd.DataFrame(predictions, columns=cfg.quantile_names)], axis=1
    )
    cfg.prediction_output_dir.mkdir(parents=True, exist_ok=True)
    output = cfg.prediction_output_dir / f"dit_{split}_predictions.parquet"
    result.to_parquet(output, index=False, compression="snappy")
    print(f"已保存 {len(result):,} 行：{output}")
    return output
