# -*- coding: utf-8 -*-
"""标准化Vit分布标签与动态数值，并提前Tokenize文本。"""

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


def calculate_scaler() -> dict[str, float]:
    train = pd.read_parquet(
        cfg.PRETRAIN_FILES["train"],
        columns=["log_supply"] + cfg.DEMAND_FEATURE_COLUMNS + cfg.QUANTILE_COLUMNS,
    )
    median = train[cfg.QUANTILE_COLUMNS[len(cfg.QUANTILE_COLUMNS) // 2]].to_numpy(np.float32)
    result = {
        "label_mean": float(median.mean()),
        "label_std": float(max(median.std(ddof=0), 1e-6)),
        "log_supply_mean": float(train["log_supply"].mean()),
        "log_supply_std": float(max(train["log_supply"].std(ddof=0), 1e-6)),
        "demand_feature_mean": train[cfg.DEMAND_FEATURE_COLUMNS].mean().astype(float).tolist(),
        "demand_feature_std": (
            train[cfg.DEMAND_FEATURE_COLUMNS].std(ddof=0).clip(lower=1e-6).astype(float).tolist()
        ),
    }
    with cfg.SCALER_FILE.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    return result


def tokenize_split(split: str, tokenizer, scaler: dict[str, float]) -> None:
    frame = pd.read_parquet(
        cfg.PRETRAIN_FILES[split],
        columns=["region_id", "t", "text", "log_supply"]
        + cfg.DEMAND_FEATURE_COLUMNS
        + cfg.QUANTILE_COLUMNS,
    )
    labels = (
        frame[cfg.QUANTILE_COLUMNS].to_numpy(np.float32) - scaler["label_mean"]
    ) / scaler["label_std"]
    source = Dataset.from_dict(
        {
            "text": frame["text"].tolist(),
            "log_supply": (
                (frame["log_supply"] - scaler["log_supply_mean"])
                / scaler["log_supply_std"]
            ).astype(np.float32).tolist(),
            "demand_summary": [
                row for row in (
                    (frame[cfg.DEMAND_FEATURE_COLUMNS].to_numpy(np.float32)
                    - np.asarray(scaler["demand_feature_mean"], dtype=np.float32))
                    / np.asarray(scaler["demand_feature_std"], dtype=np.float32)
                )
            ],
            "labels": [row for row in labels],
        }
    )

    def encode(batch):
        encoded = tokenizer(
            batch["text"],
            truncation=False,
            padding=False,
        )
        longest = max(map(len, encoded["input_ids"]), default=0)
        if longest > cfg.MAX_LENGTH:
            raise ValueError(
                f"Vit Prompt长度{longest}超过MAX_LENGTH={cfg.MAX_LENGTH}，拒绝截断。"
            )
        return encoded

    tokenized = source.map(
        encode,
        batched=True,
        batch_size=cfg.TOKENIZER_BATCH_SIZE,
        num_proc=cfg.TOKENIZER_NUM_PROC,
        remove_columns=["text"],
    )
    output = cfg.TOKENIZED_SPLIT_DIRS[split]
    if output.exists():
        shutil.rmtree(output)
    tokenized.save_to_disk(str(output))
    print(f"{split}：已Tokenize {len(tokenized):,} 行。")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", choices=tuple(cfg.SPLITS), default=list(cfg.SPLITS))
    args = parser.parse_args()
    cfg.ensure_directories()
    scaler = calculate_scaler()
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.QWEN_BASE_DIR,
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = cfg.PADDING_SIDE
    tokenizer.save_pretrained(cfg.TOKENIZER_DIR)
    for split in args.splits:
        tokenize_split(split, tokenizer, scaler)


if __name__ == "__main__":
    main()
