# -*- coding: utf-8 -*-
"""使用冻结的Vit预训练结果，导出6月11—24日全部(i,t)的101维分布。"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


cfg = load(HERE.parent / "1 数据准备" / "1 Vit预训练配置.py", "vit_pretrain_config")
model_module = load(HERE / "1 定义Vit分位数模型.py", "vit_quantile_model")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg.ensure_directories()
    model_info = json.loads(cfg.BEST_MODEL_INFO_FILE.read_text(encoding="utf-8"))
    if int(model_info.get("training_rows", 0)) != cfg.EXPECTED_TRAIN_ROWS:
        raise ValueError("当前Vit模型不是6月1—24日全量训练版，拒绝生成SAC输入。")
    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    device = torch.device(device_name)
    scaler = json.loads(cfg.SCALER_FILE.read_text(encoding="utf-8"))
    head = model_module.VitQuantileHead()
    head.load_state_dict(
        torch.load(cfg.BEST_QUANTILE_HEAD_FILE, map_location="cpu", weights_only=True)
    )
    head = head.to(device).eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)

    supply_mean = float(scaler["log_supply_mean"])
    supply_std = float(scaler["log_supply_std"])
    demand_mean = np.asarray(scaler["demand_feature_mean"], np.float32)
    demand_std = np.asarray(scaler["demand_feature_std"], np.float32)
    produced = []
    for day in cfg.days("predict"):
        state_path = cfg.STATE_TEXT_DIR / f"vit_state_{day:%Y-%m-%d}.parquet"
        feature_path = cfg.STATE_FEATURE_DIR / f"vit_features_{day:%Y-%m-%d}.npz"
        output_path = cfg.PREDICTION_DIR / f"vit_quantiles_{day:%Y-%m-%d}.npz"
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"{output_path}已存在；覆盖时加--overwrite。")
        state = pd.read_parquet(
            state_path,
            columns=["region_id", "region_index", "t", "log_supply", *cfg.DEMAND_FEATURE_COLUMNS],
        ).sort_values(["t", "region_index"])
        with np.load(feature_path, allow_pickle=False) as saved:
            region_ids = saved["region_ids"].astype(str)
            features = saved["features"].astype(np.float32)
            available = saved["available"].astype(bool)
        region_count = len(region_ids)
        expected = (cfg.TIME_STEPS_PER_DAY, region_count, cfg.STATE_FEATURE_DIM)
        if features.shape != expected:
            raise ValueError(f"{feature_path.name}维度错误：{features.shape}")
        state_regions = (
            state.loc[state["t"] == state["t"].min()]
            .sort_values("region_index")["region_id"].astype(str).to_numpy()
        )
        if not np.array_equal(region_ids, state_regions):
            raise ValueError(f"{day.date()}状态文本与512维编码的区域顺序不一致。")

        scaled_supply = (
            state["log_supply"].to_numpy(np.float32) - supply_mean
        ) / supply_std
        scaled_demand = (
            state[cfg.DEMAND_FEATURE_COLUMNS].to_numpy(np.float32) - demand_mean
        ) / demand_std
        flat_features = features.reshape(-1, cfg.STATE_FEATURE_DIM)
        result = np.empty((len(flat_features), len(cfg.QUANTILE_LEVELS)), np.float32)
        with torch.inference_mode():
            for start in range(0, len(flat_features), args.batch_size):
                stop = min(start + args.batch_size, len(flat_features))
                scaled = head(
                    torch.as_tensor(flat_features[start:stop], device=device),
                    torch.as_tensor(scaled_supply[start:stop], device=device),
                    torch.as_tensor(scaled_demand[start:stop], device=device),
                )
                values = scaled * float(scaler["label_std"]) + float(scaler["label_mean"])
                result[start:stop] = values.cpu().numpy()
        result = result.reshape(cfg.TIME_STEPS_PER_DAY, region_count, -1)
        if not np.isfinite(result).all() or (np.diff(result, axis=-1) < -1e-5).any():
            raise FloatingPointError(f"{day.date()} Vit预测存在NaN/inf或分位数交叉。")
        np.savez_compressed(
            output_path,
            date=np.asarray(str(day.date())), region_ids=region_ids,
            quantiles=result.astype(np.float16), available=available,
            frozen=np.asarray(True),
            demand_source=np.asarray("新Dit预测分布"),
            supply_source=np.asarray("新Sit随机供给分布"),
        )
        produced.append(str(output_path))
        print(f"已导出{day.date()} Vit预测：{result.shape}")

    metadata = {
        "frozen": True,
        "training_labels": "2024-06-01至2024-06-24全量Oracle标签",
        "prediction_dates": "2024-06-11至2024-06-24",
        "demand_input": "Dit预测模型101维分位数压缩摘要",
        "supply_input": "基于U分布重构的Sit随机供给分布；数值头使用q50，完整分布已写入Prompt",
        "files": produced,
    }
    (cfg.PREDICTION_DIR / "export_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
