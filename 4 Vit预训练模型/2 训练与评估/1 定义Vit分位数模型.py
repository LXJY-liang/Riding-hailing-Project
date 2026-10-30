# -*- coding: utf-8 -*-
"""Qwen3-4B、Vit专用LoRA、512维状态投影与101维单调分位数头。"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel


HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE.parent / "1 数据准备" / "1 Vit预训练配置.py"
spec = importlib.util.spec_from_file_location("vit_pretrain_config", CONFIG_PATH)
cfg = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(cfg)


def lora_targets(backbone: nn.Module) -> list[str]:
    """只对Qwen第32—35层q_proj/v_proj训练LoRA，减少时间和显存。"""

    pattern = re.compile(r"(?:^|\.)layers\.(\d+)\.")
    targets = []
    for name, _ in backbone.named_modules():
        match = pattern.search(name)
        if (
            name.rsplit(".", 1)[-1] in cfg.LORA_TARGET_NAMES
            and match is not None
            and cfg.LORA_LAYER_START <= int(match.group(1)) <= cfg.LORA_LAYER_END
        ):
            targets.append(name)
    if not targets:
        raise ValueError("未找到Vit LoRA目标层。")
    return targets


class VitStateProjection(nn.Module):
    """把Qwen最后有效Token表示压缩为可预计算的512维状态。"""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, cfg.STATE_FEATURE_DIM),
            nn.GELU(),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden.float())


class VitQuantileHead(nn.Module):
    """输入512维状态、动态供给和27维需求分布摘要，输出单调101维分位数。"""

    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
            nn.LayerNorm(cfg.STATE_FEATURE_DIM + 1 + cfg.DEMAND_FEATURE_DIM),
            nn.Linear(cfg.STATE_FEATURE_DIM + 1 + cfg.DEMAND_FEATURE_DIM, cfg.HEAD_HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(cfg.HEAD_DROPOUT),
        )
        self.raw_output = nn.Linear(cfg.HEAD_HIDDEN_DIM, len(cfg.QUANTILE_LEVELS))
        self.register_buffer(
            "quantile_gaps",
            torch.as_tensor(cfg.QUANTILE_LEVELS[1:] - cfg.QUANTILE_LEVELS[:-1]).unsqueeze(0),
        )

    def forward(
        self,
        state_feature: torch.Tensor,
        scaled_log_supply: torch.Tensor,
        scaled_demand_summary: torch.Tensor,
    ) -> torch.Tensor:
        if scaled_demand_summary.shape[-1] != cfg.DEMAND_FEATURE_DIM:
            raise ValueError("Vit需求分布摘要维度不正确。")
        dynamic = torch.cat(
            [scaled_log_supply.unsqueeze(-1), scaled_demand_summary], dim=-1
        )
        raw = self.raw_output(self.feature(torch.cat([state_feature.float(), dynamic.float()], dim=-1)))
        # 最低分位点不设0下界；只约束后续分位点单调不下降。
        lower = raw[:, :1]
        increments = F.softplus(raw[:, 1:]) * self.quantile_gaps
        return torch.cat([lower, lower + increments.cumsum(dim=1)], dim=1)


class VitQuantileModel(nn.Module):
    def __init__(self, model_dtype, adapter_path: Path | None = None, is_trainable: bool = True):
        super().__init__()
        backbone = AutoModel.from_pretrained(
            cfg.QWEN_BASE_DIR,
            local_files_only=True,
            trust_remote_code=True,
            dtype=model_dtype,
            low_cpu_mem_usage=True,
        )
        backbone.config.use_cache = False
        if cfg.USE_GRADIENT_CHECKPOINTING and is_trainable:
            backbone.gradient_checkpointing_enable()
            backbone.enable_input_require_grads()

        if adapter_path is None:
            adapter = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=cfg.LORA_RANK,
                lora_alpha=cfg.LORA_ALPHA,
                lora_dropout=cfg.LORA_DROPOUT,
                target_modules=lora_targets(backbone),
                bias="none",
            )
            self.backbone = get_peft_model(backbone, adapter)
        else:
            self.backbone = PeftModel.from_pretrained(
                backbone,
                str(adapter_path),
                is_trainable=is_trainable,
            )

        if is_trainable:
            for parameter in self.backbone.parameters():
                if parameter.requires_grad:
                    parameter.data = parameter.data.float()

        self.state_projection = VitStateProjection(self.backbone.config.hidden_size).float()
        self.quantile_head = VitQuantileHead().float()

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        last_index = attention_mask.sum(dim=1) - 1
        batch_index = torch.arange(input_ids.size(0), device=input_ids.device)
        hidden = outputs.last_hidden_state[batch_index, last_index]
        return self.state_projection(hidden)

    def forward(self, input_ids, attention_mask, log_supply, demand_summary):
        return self.quantile_head(
            self.encode(input_ids, attention_mask),
            log_supply,
            demand_summary,
        )


def quantile_curve_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """标签已是轨迹模拟分位数曲线，逐分位点使用Smooth-L1监督。"""

    return F.smooth_l1_loss(prediction, target)
