# -*- coding: utf-8 -*-
"""
8 Vit模型.py

模型结构：
文本状态
    → Qwen3-4B-Base + Vit专用LoRA
    → 最后一个有效Token隐藏状态
    → 单值价值头
    → Vit预训练值
"""

import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
)
from transformers import AutoModel


# ============================================================
# 1. 加载配置
# ============================================================

module_dir = Path(__file__).resolve().parent

spec = importlib.util.spec_from_file_location(
    "vit_config",
    module_dir / "5 Vit配置.py",
)

cfg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)

# ============================================================
# 2. 单值价值头
# ============================================================

class ValueHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                hidden_size,
                cfg.value_head_hidden_size,
            ),
            nn.GELU(),
            nn.Dropout(
                cfg.value_head_dropout
            ),
            nn.Linear(
                cfg.value_head_hidden_size,
                1,
            ),
        )

    def forward(self, hidden):
        return self.net(hidden).squeeze(-1)


# ============================================================
# 3. Vit模型
# ============================================================

class VitValueModel(nn.Module):
    def __init__(
        self,
        model_dtype,
        adapter_path=None,
        is_trainable=True,
    ):
        super().__init__()

        backbone = AutoModel.from_pretrained(
            cfg.base_model_path,
            dtype=model_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        for parameter in backbone.parameters():
            parameter.requires_grad = False

        if adapter_path is None:
            lora_config = LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=cfg.lora_target_modules,
                bias="none",
            )

            backbone = get_peft_model(
                backbone,
                lora_config,
            )

        else:
            backbone = PeftModel.from_pretrained(
                backbone,
                adapter_path,
                is_trainable=is_trainable,
            )

        if (
            cfg.use_gradient_checkpointing
            and is_trainable
        ):
            backbone.gradient_checkpointing_enable()

            if hasattr(
                backbone,
                "enable_input_require_grads",
            ):
                backbone.enable_input_require_grads()

        self.backbone = backbone

        self.value_head = ValueHead(
            self.backbone.config.hidden_size
        )

        self.value_head.float()

    def forward(
        self,
        input_ids,
        attention_mask,
    ):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )

        last_index = (
            attention_mask.sum(dim=1) - 1
        )

        batch_index = torch.arange(
            input_ids.size(0),
            device=input_ids.device,
        )

        hidden = outputs.last_hidden_state[
            batch_index,
            last_index,
        ].float()

        return self.value_head(hidden)
