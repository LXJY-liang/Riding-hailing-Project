from pathlib import Path
import importlib.util
import json
import re
import torch
from torch import nn
from torch.nn import functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModel


# =========================
# 1. 读取配置
# =========================

def load_config():
    config_path = Path(__file__).resolve().parent / "1 tauijt配置.py"
    spec = importlib.util.spec_from_file_location("tauijt_config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


config = load_config()


# =========================
# 2. LoRA目标层与分位数头
# =========================

def get_lora_target_modules(backbone):
    """只选择最后12层Transformer中的q_proj和v_proj。"""

    target_names = set(config.lora_target_modules)
    target_modules = []
    layer_pattern = re.compile(r"(?:^|\.)layers\.(\d+)\.")

    for module_name, _ in backbone.named_modules():
        short_name = module_name.rsplit(".", 1)[-1]
        layer_match = layer_pattern.search(module_name)

        if short_name not in target_names or layer_match is None:
            continue

        layer_index = int(layer_match.group(1))
        if config.lora_layer_start <= layer_index <= config.lora_layer_end:
            target_modules.append(module_name)

    if not target_modules:
        raise ValueError(
            "未找到tauijt LoRA目标模块，请检查Qwen模型层命名是否包含"
            f"layers.{config.lora_layer_start}到"
            f"layers.{config.lora_layer_end}的q_proj/v_proj。"
        )

    return target_modules


def get_scaled_label_min():
    """计算标准化标签空间中服务时间非负对应的下界。"""

    with open(
        config.label_scaler_path,
        "r",
        encoding="utf-8",
    ) as file:
        scaler = json.load(file)

    if scaler["use_standardization"]:
        return -float(scaler["label_mean"]) / float(scaler["label_std"])

    return 0.0


class MonotonicSplineQuantileHead(nn.Module):
    """单调分段线性样条头，保证101个分位数有序且服务时间非负。"""

    def __init__(self, hidden_size):
        super().__init__()

        self.feature_projection = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, config.quantile_head_hidden_size),
            nn.GELU(),
            nn.Dropout(config.quantile_head_dropout),
        )
        self.raw_output = nn.Linear(
            config.quantile_head_hidden_size,
            config.num_quantiles,
        )
        self.register_buffer(
            "scaled_label_min",
            torch.tensor(get_scaled_label_min(), dtype=torch.float32),
        )
        self.register_buffer(
            "quantile_gaps",
            torch.as_tensor(
                config.quantile_levels[1:] - config.quantile_levels[:-1],
                dtype=torch.float32,
            ).unsqueeze(0),
        )
        self.reset_raw_output()

    def reset_raw_output(self):
        nn.init.normal_(self.raw_output.weight, mean=0.0, std=1e-3)
        with torch.no_grad():
            self.raw_output.bias[0].fill_(
                config.quantile_head_initial_lower_bias
            )
            self.raw_output.bias[1:].fill_(
                config.quantile_head_initial_slope_bias
            )

    def forward(self, pooled_state):
        raw_values = self.raw_output(
            self.feature_projection(pooled_state)
        )
        lower_quantile = (
            self.scaled_label_min + F.softplus(raw_values[:, :1])
        )
        positive_increments = (
            F.softplus(raw_values[:, 1:]) * self.quantile_gaps
        )

        return torch.cat(
            [
                lower_quantile,
                lower_quantile + positive_increments.cumsum(dim=1),
            ],
            dim=1,
        )

# =========================
# 3. 分位数预测模型
# =========================

class tauijt_quantile_model(nn.Module):
    """Qwen3-4B-Base、tauijt专用LoRA和101维分位数头。"""

    def __init__(self, model_dtype, adapter_path=None, is_trainable=True):
        super().__init__()

        backbone = AutoModel.from_pretrained(
            str(config.base_model_path),
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=model_dtype,
        )
        backbone.config.use_cache = False

        if config.use_gradient_checkpointing and is_trainable:
            backbone.gradient_checkpointing_enable()
            backbone.enable_input_require_grads()
            backbone.config.use_cache = False

        if adapter_path is None:
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=get_lora_target_modules(backbone),
                bias="none",
            )
            self.backbone = get_peft_model(backbone, lora_config)
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

        hidden_size = self.backbone.config.hidden_size

        self.quantile_head = (
            MonotonicSplineQuantileHead(hidden_size)
        ).float()

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )

        last_hidden_state = outputs.last_hidden_state
        last_token_index = attention_mask.sum(dim=1) - 1
        batch_index = torch.arange(
            last_hidden_state.size(0),
            device=last_hidden_state.device,
        )

        pooled_state = last_hidden_state[
            batch_index,
            last_token_index,
        ].float()

        return self.quantile_head(pooled_state)


# =========================
# 4. 分位数损失
# =========================

def tauijt_quantile_loss(predictions, labels):
    """平滑Pinball损失；单调性由Quantile Head结构保证。"""

    quantile_levels = torch.as_tensor(
        config.quantile_levels,
        dtype=predictions.dtype,
        device=predictions.device,
    ).unsqueeze(0)

    errors = labels.unsqueeze(1) - predictions
    alpha = config.smooth_pinball_alpha

    pinball_loss = (
        quantile_levels * errors
        + alpha * F.softplus(-errors / alpha)
    ).mean()

    crossing_loss = F.relu(
        predictions[:, :-1] - predictions[:, 1:]
    ).mean()

    total_loss = (
        pinball_loss
        + config.crossing_loss_weight * crossing_loss
    )

    return total_loss, pinball_loss, crossing_loss
