# -*- coding: utf-8 -*-
"""冻结Qwen+LoRA，导出6月11—24日每个(i,t)的512维Prompt状态编码。"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
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


class TextDataset(Dataset):
    def __init__(self, values: list[str]):
        self.values = values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg.ensure_directories()
    model_info = json.loads(cfg.BEST_MODEL_INFO_FILE.read_text(encoding="utf-8"))
    if int(model_info.get("training_rows", 0)) != cfg.EXPECTED_TRAIN_ROWS:
        raise ValueError("当前Vit模型不是6月1—24日全量训练版，拒绝导出。")
    model_module = load(HERE / "1 定义Vit分位数模型.py", "vit_quantile_model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else (
        torch.float16 if device.type == "cuda" else torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.TOKENIZER_DIR, local_files_only=True, trust_remote_code=True
    )
    model = model_module.VitQuantileModel(
        model_dtype=dtype, adapter_path=cfg.BEST_LORA_DIR, is_trainable=False
    )
    model.state_projection.load_state_dict(
        torch.load(cfg.BEST_STATE_PROJECTION_FILE, map_location="cpu", weights_only=True)
    )
    model = model.to(device).eval()

    def collate(texts):
        encoded = tokenizer(texts, truncation=False, padding=True, return_tensors="pt")
        if encoded["input_ids"].shape[1] > cfg.MAX_LENGTH:
            raise ValueError("Vit中文Prompt超过MAX_LENGTH，拒绝截断。")
        return encoded

    for day in cfg.days("predict"):
        source = cfg.STATE_TEXT_DIR / f"vit_state_{day:%Y-%m-%d}.parquet"
        output = cfg.STATE_FEATURE_DIR / f"vit_features_{day:%Y-%m-%d}.npz"
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"{output}已存在；覆盖时加--overwrite。")
        frame = pd.read_parquet(source).sort_values(["t", "region_index"])
        expected = cfg.TIME_STEPS_PER_DAY * frame["region_index"].nunique()
        if len(frame) != expected:
            raise ValueError(f"{day.date()}状态文本不完整。")
        loader = DataLoader(
            TextDataset(frame["text"].tolist()), batch_size=cfg.EXPORT_BATCH_SIZE,
            shuffle=False, collate_fn=collate, num_workers=cfg.NUM_WORKERS,
            pin_memory=device.type == "cuda",
        )
        parts = []
        with torch.inference_mode():
            for batch in tqdm(loader, desc=f"编码 {day.date()}", dynamic_ncols=True):
                encoded = model.encode(
                    batch["input_ids"].to(device, non_blocking=True),
                    batch["attention_mask"].to(device, non_blocking=True),
                )
                parts.append(encoded.cpu().to(torch.float16).numpy())
        features = np.concatenate(parts).reshape(
            cfg.TIME_STEPS_PER_DAY, frame["region_index"].nunique(), cfg.STATE_FEATURE_DIM
        )
        regions = (
            frame.loc[frame["t"] == frame["t"].min()]
            .sort_values("region_index")["region_id"].astype(str).to_numpy()
        )
        np.savez_compressed(
            output, date=np.asarray(str(day.date())), region_ids=regions,
            features=features, available=np.ones(features.shape[:2], bool),
        )
        print(f"已导出{day.date()}全部状态编码：{features.shape}")


if __name__ == "__main__":
    main()
