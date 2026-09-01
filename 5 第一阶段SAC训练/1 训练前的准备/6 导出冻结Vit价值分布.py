# -*- coding: utf-8 -*-
"""用已有512维编码和预训练分位数头导出固定101维Vit价值曲线。

本脚本只做推理，不加载Qwen、不更新LoRA或分位数头。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


HERE = Path(__file__).resolve().parent


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


cfg = load(HERE / "1 第一阶段SAC配置.py", "stage1_config")
qtool = load(HERE / "2 分位数计算工具.py", "stage1_quantiles")


class FrozenVitQuantileHead(nn.Module):
    """与预训练代码完全同构，但不导入Qwen、Transformers或PEFT。"""

    def __init__(self, scaled_value_min: float):
        super().__init__()
        input_dim = cfg.VIT_STATE_FEATURE_DIM + 1 + 27
        self.feature = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, 512), nn.GELU(), nn.Dropout(0.0)
        )
        self.raw_output = nn.Linear(512, len(cfg.QUANTILE_LEVELS))
        self.register_buffer("scaled_value_min", torch.tensor(float(scaled_value_min)))
        self.register_buffer(
            "quantile_gaps",
            torch.as_tensor(
                cfg.QUANTILE_LEVELS[1:] - cfg.QUANTILE_LEVELS[:-1], dtype=torch.float32
            ).unsqueeze(0),
        )

    def forward(self, state_feature, scaled_log_supply, scaled_demand_summary):
        dynamic = torch.cat(
            [scaled_log_supply.unsqueeze(-1), scaled_demand_summary], dim=-1
        )
        raw = self.raw_output(
            self.feature(torch.cat([state_feature.float(), dynamic.float()], dim=-1))
        )
        lower = self.scaled_value_min + F.softplus(raw[:, :1])
        increments = F.softplus(raw[:, 1:]) * self.quantile_gaps
        return torch.cat([lower, lower + increments.cumsum(dim=1)], dim=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", choices=tuple(cfg.SPLITS), default=list(cfg.SPLITS))
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg.ensure_directories()

    for required in (cfg.VIT_QUANTILE_HEAD_FILE, cfg.VIT_SCALER_FILE):
        if not required.exists():
            raise FileNotFoundError(f"缺少Vit预训练结果：{required}")
    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    device = torch.device(device_name)
    scaler = json.loads(cfg.VIT_SCALER_FILE.read_text(encoding="utf-8"))
    scaled_min = -float(scaler["label_mean"]) / float(scaler["label_std"])
    head = FrozenVitQuantileHead(scaled_min)
    head.load_state_dict(
        torch.load(cfg.VIT_QUANTILE_HEAD_FILE, map_location="cpu", weights_only=True)
    )
    head.eval().to(device)
    for parameter in head.parameters():
        parameter.requires_grad_(False)

    supply_mean = float(scaler["log_supply_mean"])
    supply_std = float(scaler["log_supply_std"])
    demand_mean = np.asarray(scaler["demand_feature_mean"], dtype=np.float32)
    demand_std = np.asarray(scaler["demand_feature_std"], dtype=np.float32)
    if len(demand_mean) != 27 or len(demand_std) != 27:
        raise ValueError("Vit scaler不是27维需求摘要版本。")

    produced = []
    for day64 in cfg.days_for_splits(args.splits):
        day = str(day64)
        episode_path = cfg.EPISODE_DIR / f"episode_{day}.npz"
        feature_path = cfg.VIT_STATE_FEATURE_DIR / f"vit_features_{day}.npz"
        output_path = cfg.FROZEN_VIT_DIR / f"vit_quantiles_{day}.npz"
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"{output_path}已存在；需要覆盖时加入--overwrite。")
        if not episode_path.exists():
            raise FileNotFoundError(f"缺少episode：{episode_path}")
        if not feature_path.exists():
            raise FileNotFoundError(
                f"缺少{day}的512维Vit状态编码：{feature_path}。"
                "这说明该日期尚未完成Qwen状态编码导出。"
            )
        with np.load(episode_path, allow_pickle=False) as episode:
            region_ids = episode["region_ids"].astype(str)
            supply = episode["supply_capacity"].astype(np.float32)
            transaction_q = episode["demand_q_transaction"].astype(np.float32)
        with np.load(feature_path, allow_pickle=True) as encoded:
            encoded_regions = encoded["region_ids"].astype(str)
            features = encoded["features"].astype(np.float32)
            available = (
                encoded["available"].astype(bool)
                if "available" in encoded.files else np.ones(features.shape[:2], dtype=bool)
            )
        expected = (cfg.STEPS_PER_DAY, len(region_ids), cfg.VIT_STATE_FEATURE_DIM)
        if features.shape != expected or not np.array_equal(region_ids, encoded_regions):
            raise ValueError(f"{feature_path.name}与episode的区域顺序或维度不一致。")

        summary = np.log1p(qtool.compress_quantiles(transaction_q))
        scaled_supply = (np.log1p(np.maximum(supply, 0.0)) - supply_mean) / supply_std
        scaled_demand = (summary - demand_mean) / demand_std
        flat_feature = features.reshape(-1, cfg.VIT_STATE_FEATURE_DIM)
        flat_supply = scaled_supply.reshape(-1)
        flat_demand = scaled_demand.reshape(-1, 27)
        result = np.empty((len(flat_feature), len(cfg.QUANTILE_LEVELS)), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(flat_feature), args.batch_size):
                stop = min(start + args.batch_size, len(flat_feature))
                scaled = head(
                    torch.as_tensor(flat_feature[start:stop], dtype=torch.float32, device=device),
                    torch.as_tensor(flat_supply[start:stop], dtype=torch.float32, device=device),
                    torch.as_tensor(flat_demand[start:stop], dtype=torch.float32, device=device),
                )
                value = scaled * float(scaler["label_std"]) + float(scaler["label_mean"])
                result[start:stop] = value.clamp_min(0.0).cpu().numpy()
        result = result.reshape(cfg.STEPS_PER_DAY, len(region_ids), -1)
        if not np.isfinite(result).all() or (np.diff(result, axis=-1) < -1e-5).any():
            raise FloatingPointError(f"{day}的冻结Vit输出无效或分位数交叉。")
        np.savez_compressed(
            output_path,
            date=np.asarray(day),
            region_ids=np.asarray(region_ids, dtype=np.str_),
            quantiles=result.astype(np.float16),
            available=available,
            source_feature_file=np.asarray(str(feature_path)),
            frozen=np.asarray(True),
        )
        produced.append(str(output_path))
        print(f"已导出冻结Vit：{day}，形状={result.shape}")

    metadata = {
        "frozen": True,
        "head": str(cfg.VIT_QUANTILE_HEAD_FILE),
        "scaler": str(cfg.VIT_SCALER_FILE),
        "demand_input": "原始成交量Dit的27维log1p摘要",
        "supply_input": "滚动历史高分位固定容量",
        "files": produced,
    }
    (cfg.FROZEN_VIT_DIR / "export_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
