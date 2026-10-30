# -*- coding: utf-8 -*-
"""按日Tokenize 6月1—24日全量Vit样本，不做任何抽样。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("vit_pretrain_config", HERE / "1 Vit预训练配置.py")
cfg = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(cfg)


def calculate_scaler() -> dict[str, object]:
    count = 0
    median_sum = median_square_sum = 0.0
    supply_sum = supply_square_sum = 0.0
    demand_sum = np.zeros(cfg.DEMAND_FEATURE_DIM, np.float64)
    demand_square_sum = np.zeros(cfg.DEMAND_FEATURE_DIM, np.float64)
    median_column = cfg.QUANTILE_COLUMNS[len(cfg.QUANTILE_COLUMNS) // 2]
    for day in cfg.days("train"):
        path = cfg.TRAIN_PART_DIR / f"vit_train_{day:%Y-%m-%d}.parquet"
        frame = pd.read_parquet(
            path, columns=["log_supply", *cfg.DEMAND_FEATURE_COLUMNS, median_column]
        )
        n = len(frame)
        median = frame[median_column].to_numpy(np.float64)
        supply = frame["log_supply"].to_numpy(np.float64)
        demand = frame[cfg.DEMAND_FEATURE_COLUMNS].to_numpy(np.float64)
        count += n
        median_sum += median.sum()
        median_square_sum += np.square(median).sum()
        supply_sum += supply.sum()
        supply_square_sum += np.square(supply).sum()
        demand_sum += demand.sum(axis=0)
        demand_square_sum += np.square(demand).sum(axis=0)
    if count == 0:
        raise ValueError("没有Vit全量训练样本。")

    def mean_std(total, square_total):
        mean = total / count
        variance = np.maximum(square_total / count - np.square(mean), 1e-12)
        return mean, np.sqrt(variance)

    label_mean, label_std = mean_std(median_sum, median_square_sum)
    supply_mean, supply_std = mean_std(supply_sum, supply_square_sum)
    demand_mean, demand_std = mean_std(demand_sum, demand_square_sum)
    scaler = {
        "rows": count,
        "label_mean": float(label_mean),
        "label_std": float(max(label_std, 1e-6)),
        "log_supply_mean": float(supply_mean),
        "log_supply_std": float(max(supply_std, 1e-6)),
        "demand_feature_mean": demand_mean.astype(float).tolist(),
        "demand_feature_std": np.maximum(demand_std, 1e-6).astype(float).tolist(),
    }
    cfg.SCALER_FILE.write_text(json.dumps(scaler, ensure_ascii=False, indent=2), encoding="utf-8")
    return scaler


def tokenize_day(
    day: pd.Timestamp, tokenizer, scaler: dict[str, object], split: str
) -> int:
    source_dir = cfg.TRAIN_PART_DIR if split == "train" else cfg.VALID_PART_DIR
    output_dir = (
        cfg.TOKENIZED_TRAIN_PART_DIR
        if split == "train"
        else cfg.TOKENIZED_VALID_PART_DIR
    )
    source = source_dir / f"vit_{split}_{day:%Y-%m-%d}.parquet"
    frame = pd.read_parquet(
        source,
        columns=["text", "log_supply", *cfg.DEMAND_FEATURE_COLUMNS, *cfg.QUANTILE_COLUMNS],
    )
    labels = (
        frame[cfg.QUANTILE_COLUMNS].to_numpy(np.float32) - float(scaler["label_mean"])
    ) / float(scaler["label_std"])
    demand_mean = np.asarray(scaler["demand_feature_mean"], np.float32)
    demand_std = np.asarray(scaler["demand_feature_std"], np.float32)
    demand = (frame[cfg.DEMAND_FEATURE_COLUMNS].to_numpy(np.float32) - demand_mean) / demand_std
    source_dataset = Dataset.from_dict(
        {
            "text": frame["text"].tolist(),
            "log_supply": (
                (frame["log_supply"].to_numpy(np.float32) - float(scaler["log_supply_mean"]))
                / float(scaler["log_supply_std"])
            ),
            "demand_summary": [row for row in demand],
            "labels": [row for row in labels],
        }
    )

    def encode(batch):
        encoded = tokenizer(batch["text"], truncation=False, padding=False)
        longest = max(map(len, encoded["input_ids"]), default=0)
        if longest > cfg.MAX_LENGTH:
            raise ValueError(
                f"{day.date()} Vit Prompt最长{longest} tokens，超过MAX_LENGTH={cfg.MAX_LENGTH}；拒绝静默截断。"
            )
        return encoded

    tokenized = source_dataset.map(
        encode,
        batched=True,
        batch_size=cfg.TOKENIZER_BATCH_SIZE,
        num_proc=cfg.TOKENIZER_NUM_PROC,
        remove_columns=["text"],
    )
    output = output_dir / f"vit_{split}_{day:%Y-%m-%d}"
    if output.exists():
        shutil.rmtree(output)
    tokenized.save_to_disk(str(output))
    print(f"已Tokenize {day.date()} {split}：{len(tokenized):,}行（全量）")
    return len(tokenized)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按日Tokenize Vit训练或验证样本。")
    parser.add_argument(
        "--split", choices=("train", "valid"), default="train",
        help="默认处理6月1—24日训练集；valid处理6月25—27日验证集。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg.ensure_directories()
    if args.split == "train":
        scaler = calculate_scaler()
    else:
        if not cfg.SCALER_FILE.exists():
            raise FileNotFoundError("验证集必须使用训练集scaler，但当前scaler.json不存在。")
        scaler = json.loads(cfg.SCALER_FILE.read_text(encoding="utf-8"))
        if cfg.VALIDATION_READY_FILE.exists():
            cfg.VALIDATION_READY_FILE.unlink()
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.QWEN_BASE_DIR, local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = cfg.PADDING_SIDE
    tokenizer.save_pretrained(cfg.TOKENIZER_DIR)
    rows = 0
    for day in cfg.days(args.split):
        rows += tokenize_day(day, tokenizer, scaler, args.split)
    if args.split == "valid":
        if rows != cfg.EXPECTED_VALID_ROWS:
            raise ValueError(
                f"验证集Tokenized行数应为{cfg.EXPECTED_VALID_ROWS:,}，实际{rows:,}。"
            )
        ready = {
            "split": "valid",
            "rows": rows,
            "days": [str(day.date()) for day in cfg.days("valid")],
            "sample_denominator": cfg.VALIDATION_SAMPLE_DENOMINATOR,
            "sample_seed": cfg.VALIDATION_SAMPLE_SEED,
        }
        temporary = cfg.VALIDATION_READY_FILE.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(ready, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(cfg.VALIDATION_READY_FILE)
        print(f"验证集已完整就绪：{cfg.VALIDATION_READY_FILE}")


if __name__ == "__main__":
    main()
