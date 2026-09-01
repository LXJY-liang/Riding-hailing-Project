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
    config_path = Path(__file__).resolve().parent / "1 Dit配置.py"
    spec = importlib.util.spec_from_file_location("dit_config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


config = load_config()
_DEMAND_BIN_TENSOR_CACHE = {}


# =========================
# 2. LoRA目标层
# =========================

def get_lora_target_modules(backbone):
    """只选择最后12层Transformer中的q_proj和v_proj。"""

    target_names = set(config.lora_target_modules)
    target_modules = []
    layer_pattern = re.compile(r"(?:^|\.)layers\.(\d+)\.")

    for module_name, module in backbone.named_modules():
        short_name = module_name.rsplit(".", 1)[-1]
        layer_match = layer_pattern.search(module_name)

        if short_name not in target_names or layer_match is None:
            continue

        layer_index = int(layer_match.group(1))
        if config.lora_layer_start <= layer_index <= config.lora_layer_end:
            target_modules.append(module_name)

    if not target_modules:
        raise ValueError(
            "未找到Dit LoRA目标模块，请检查Qwen模型层命名是否包含"
            f"layers.{config.lora_layer_start}到"
            f"layers.{config.lora_layer_end}的q_proj/v_proj。"
        )

    return target_modules


def get_scaled_label_min():
    """计算标准化标签空间中订单量非负对应的下界。"""

    with open(
        config.label_scaler_path,
        "r",
        encoding="utf-8",
    ) as file:
        scaler = json.load(file)

    nonnegative_log_value = 0.0

    if scaler["use_standardization"]:
        return (
            nonnegative_log_value - float(scaler["label_mean"])
        ) / float(scaler["label_std"])

    return nonnegative_log_value


def transform_label_value(value, scaler):
    """把真实订单量阈值转换到训练标签空间。"""

    transformed_value = float(value)

    if scaler["use_log1p"]:
        transformed_value = torch.log1p(
            torch.tensor(transformed_value)
        ).item()

    if scaler["use_standardization"]:
        transformed_value = (
            transformed_value - float(scaler["label_mean"])
        ) / float(scaler["label_std"])

    return transformed_value


def get_demand_bin_tensors(device, dtype):
    """返回训练标签空间下的Dit区间上界和对应样本权重。"""

    cache_key = (str(device), str(dtype))
    if cache_key in _DEMAND_BIN_TENSOR_CACHE:
        return _DEMAND_BIN_TENSOR_CACHE[cache_key]

    with open(
        config.label_scaler_path,
        "r",
        encoding="utf-8",
    ) as file:
        scaler = json.load(file)

    upper_bounds = []
    for upper_bound in config.demand_bin_upper_bounds:
        if upper_bound == float("inf"):
            upper_bounds.append(float("inf"))
        else:
            upper_bounds.append(
                transform_label_value(upper_bound, scaler)
            )

    tensors = (
        torch.tensor(upper_bounds, dtype=dtype, device=device),
        torch.tensor(
            config.demand_bin_weights,
            dtype=dtype,
            device=device,
        ),
    )
    _DEMAND_BIN_TENSOR_CACHE[cache_key] = tensors
    return tensors


def get_sample_demand_weights(labels):
    """根据scaled label所属真实Dit区间生成样本权重。"""

    upper_bounds, bin_weights = get_demand_bin_tensors(
        labels.device,
        labels.dtype,
    )
    sample_weights = torch.ones_like(labels)
    previous_upper = None

    for bin_index, upper_bound in enumerate(upper_bounds):
        if bin_index == 0:
            mask = labels <= (
                upper_bound + config.demand_zero_tolerance
            )
        elif torch.isinf(upper_bound):
            mask = labels > previous_upper
        else:
            mask = (
                (labels > previous_upper)
                & (labels <= upper_bound)
            )

        sample_weights = torch.where(
            mask,
            bin_weights[bin_index],
            sample_weights,
        )
        previous_upper = upper_bound

    return sample_weights


class MonotonicSplineQuantileHead(nn.Module):
    """单调分段线性样条分位数头，输出满足非负订单量约束的标签尺度预测。"""

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
            torch.tensor(
                get_scaled_label_min(),
                dtype=torch.float32,
            ),
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
            self.scaled_label_min
            + F.softplus(raw_values[:, :1])
        )
        positive_increments = (
            F.softplus(raw_values[:, 1:])
            * self.quantile_gaps
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

class dit_quantile_model(nn.Module):
    """Qwen3-4B-Base、Dit专用LoRA和101维分位数头。"""

    def __init__(self, model_dtype, adapter_path=None, is_trainable=True):
        super().__init__()

        backbone = AutoModel.from_pretrained(
            str(config.base_model_path),
            local_files_only=True,
            trust_remote_code=True,
            dtype=model_dtype,
        )
        backbone.config.use_cache = False

        if config.use_gradient_checkpointing and is_trainable:
            backbone.gradient_checkpointing_enable()
            backbone.enable_input_require_grads()
            backbone.config.use_cache = False

        if adapter_path is None:
            lora_target_modules = get_lora_target_modules(backbone)
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=lora_target_modules,
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

def dit_quantile_loss(predictions, labels):
    """加权平滑Pinball损失；单调性由Quantile Head结构保证。"""

    quantile_levels = torch.as_tensor(
        config.quantile_levels,
        dtype=predictions.dtype,
        device=predictions.device,
    ).unsqueeze(0)

    errors = labels.unsqueeze(1) - predictions
    alpha = config.smooth_pinball_alpha

    pinball_by_quantile = (
        quantile_levels * errors
        + alpha * F.softplus(-errors / alpha)
    )
    per_sample_pinball = pinball_by_quantile.mean(dim=1)
    pinball_loss = per_sample_pinball.mean()

    if config.demand_weighting_enabled:
        sample_weights = get_sample_demand_weights(labels)
        if config.demand_weight_normalize_by_batch:
            weighted_pinball_loss = (
                sample_weights * per_sample_pinball
            ).sum() / sample_weights.sum().clamp_min(1e-12)
        else:
            weighted_pinball_loss = (
                sample_weights * per_sample_pinball
            ).mean()
    else:
        weighted_pinball_loss = pinball_loss

    crossing_loss = F.relu(
        predictions[:, :-1] - predictions[:, 1:]
    ).mean()

    total_loss = weighted_pinball_loss

    return total_loss, pinball_loss, crossing_loss
