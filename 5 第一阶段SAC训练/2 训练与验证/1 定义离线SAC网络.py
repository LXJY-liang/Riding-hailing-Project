# -*- coding: utf-8 -*-
"""第一阶段外生状态离线SAC的共享Actor与OD级双Critic。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def load_config():
    path = Path(__file__).resolve().parent.parent / "1 训练前的准备" / "1 第一阶段SAC配置.py"
    spec = importlib.util.spec_from_file_location("stage1_config", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


cfg = load_config()


def mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = input_dim
    for width in hidden_dims:
        layers.extend([nn.Linear(current, width), nn.SiLU()])
        current = width
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


def require_finite_tensor(name: str, value: torch.Tensor, row_context=None) -> None:
    invalid = ~torch.isfinite(value)
    if not bool(invalid.any().item()):
        return
    position = torch.nonzero(invalid, as_tuple=False)[0].detach().cpu().tolist()
    row = int(position[0]) if position else 0
    context = row_context[row] if row_context is not None and row < len(row_context) else f"row={row}"
    raise FloatingPointError(f"{name}出现NaN/inf；{context}，索引={tuple(position)}")


class Actor(nn.Module):
    """直接参数化合法价格、补贴，不进行事后clip。"""

    def __init__(self, state_dim: int, region_count: int):
        super().__init__()
        self.i_embedding = nn.Embedding(region_count, cfg.REGION_EMBEDDING_DIM)
        self.j_embedding = nn.Embedding(region_count, cfg.REGION_EMBEDDING_DIM)
        width = state_dim + 2 * cfg.REGION_EMBEDDING_DIM
        self.backbone = mlp(width, cfg.ACTOR_HIDDEN, cfg.ACTOR_HIDDEN[-1])
        self.mean = nn.Linear(cfg.ACTOR_HIDDEN[-1], 2)
        self.log_std = nn.Linear(cfg.ACTOR_HIDDEN[-1], 2)
        nn.init.zeros_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.log_std.weight)
        nn.init.constant_(self.log_std.bias, cfg.POLICY_INITIAL_LOG_STD)
        unit = (cfg.POLICY_INITIAL_PRICE_RATIO - cfg.PRICE_RATIO_MIN) / (
            cfg.PRICE_RATIO_MAX - cfg.PRICE_RATIO_MIN
        )
        if not 0.0 < unit < 1.0:
            raise ValueError("POLICY_INITIAL_PRICE_RATIO必须严格位于上下限之间。")
        with torch.no_grad():
            self.mean.bias[0] = float(np.log(unit / (1.0 - unit)))
            self.mean.bias[1] = cfg.POLICY_INITIAL_SUBSIDY_LOGIT

    def distribution(self, state, i_index, j_index, row_context=None):
        require_finite_tensor("Actor state", state, row_context)
        hidden = self.backbone(
            torch.cat([state, self.i_embedding(i_index), self.j_embedding(j_index)], dim=-1)
        )
        mean = self.mean(hidden)
        log_std = self.log_std(hidden).clamp(cfg.POLICY_LOG_STD_MIN, cfg.POLICY_LOG_STD_MAX)
        require_finite_tensor("Actor mean", mean, row_context)
        return mean, log_std

    def _physical_action(self, raw, p_ref):
        price_unit = torch.sigmoid(raw[:, 0])
        price_ratio = cfg.PRICE_RATIO_MIN + (
            cfg.PRICE_RATIO_MAX - cfg.PRICE_RATIO_MIN
        ) * price_unit
        subsidy_unit = torch.sigmoid(raw[:, 1])
        return torch.stack(
            [p_ref * price_ratio, cfg.SUBSIDY_MAX_RATIO * p_ref * subsidy_unit], dim=-1
        )

    def sample(self, state, i_index, j_index, p_ref, row_context=None):
        mean, log_std = self.distribution(state, i_index, j_index, row_context)
        normal = torch.distributions.Normal(mean, log_std.exp())
        raw = normal.rsample()
        action = self._physical_action(raw, p_ref)
        log_det = (
            F.logsigmoid(raw[:, 0]) + F.logsigmoid(-raw[:, 0])
            + F.logsigmoid(raw[:, 1]) + F.logsigmoid(-raw[:, 1])
        )
        log_prob = (normal.log_prob(raw).sum(dim=-1) - log_det).unsqueeze(-1)
        require_finite_tensor("Actor action", action, row_context)
        require_finite_tensor("Actor normalized_log_prob", log_prob, row_context)
        return action, log_prob

    def deterministic(self, state, i_index, j_index, p_ref, row_context=None):
        mean, _ = self.distribution(state, i_index, j_index, row_context)
        return self._physical_action(mean, p_ref)


class Critic(nn.Module):
    """逐OD输出Q_ijt；同组OD共享区域联合动作上下文m_it。"""

    def __init__(self, state_dim: int, region_count: int):
        super().__init__()
        self.i_embedding = nn.Embedding(region_count, cfg.REGION_EMBEDDING_DIM)
        self.j_embedding = nn.Embedding(region_count, cfg.REGION_EMBEDDING_DIM)
        width = state_dim + 2 * cfg.REGION_EMBEDDING_DIM + 2 + cfg.REGION_CONTEXT_DIM
        self.value = mlp(width, cfg.CRITIC_HIDDEN, 1)

    def forward(self, state, i_index, j_index, action, region_context):
        if region_context.ndim == 1:
            region_context = region_context.unsqueeze(0).expand(len(state), -1)
        action_feature = torch.stack(
            [torch.log1p(action[:, 0].clamp_min(0.0)), torch.log1p(action[:, 1].clamp_min(0.0))],
            dim=-1,
        )
        row = torch.cat(
            [state, self.i_embedding(i_index), self.j_embedding(j_index), action_feature, region_context],
            dim=-1,
        )
        return self.value(row).squeeze(-1)


def empirical_cdf_torch(quantiles: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    levels = torch.as_tensor(cfg.QUANTILE_LEVELS, dtype=quantiles.dtype, device=quantiles.device)
    upper = torch.searchsorted(
        quantiles.contiguous(), points[:, None].contiguous(), right=False
    ).squeeze(1).clamp(1, quantiles.shape[1] - 1)
    lower = upper - 1
    x0 = quantiles.gather(1, lower[:, None]).squeeze(1)
    x1 = quantiles.gather(1, upper[:, None]).squeeze(1)
    fraction = ((points - x0) / (x1 - x0).clamp_min(1e-6)).clamp(0.0, 1.0)
    result = levels[lower] + fraction * (levels[upper] - levels[lower])
    result = torch.where(points < quantiles[:, 0], torch.zeros_like(result), result)
    return torch.where(points >= quantiles[:, -1], torch.ones_like(result), result)


def region_action_context(group: dict, action: torch.Tensor):
    """用同组全部新动作可微重算m_it，供Actor与Target-Critic共同使用。"""

    price, subsidy = action[:, 0], action[:, 1]
    passenger = 1.0 - empirical_cdf_torch(group["pe_q"], price)
    wait_sample = group["wait_q"].gather(1, group["vi_index"])
    vj_sample = group["vj_scenario_q"].gather(
        2, group["vj_index"].unsqueeze(-1)
    ).squeeze(-1)
    driver_reward = (1.0 - cfg.PLATFORM_RATE) * price + subsidy - group["cost"]
    accept_value = (
        driver_reward[:, None]
        + torch.pow(cfg.DRIVER_GAMMA, group["tau_steps"].float()) * vj_sample
        - cfg.DRIVER_GAMMA * wait_sample
    )
    hard = (accept_value >= 0.0).to(accept_value.dtype)
    smooth = torch.sigmoid(accept_value / cfg.DRIVER_ACCEPT_STE_TEMPERATURE)
    single = (hard + smooth - smooth.detach()).mean(dim=1)
    dispatch = min(cfg.MAX_DISPATCH, max(int(torch.floor(group["supply"]).item()), 0))
    driver_final = 1.0 - torch.pow(1.0 - single, dispatch)
    # 若把补贴提高到模型允许的上界仍无法满足司机机会约束，该OD不可成交。
    # 这是原模型的可行性规则，不是训练后的clip；Actor仍在合法动作域内端到端学习。
    max_driver_reward = (
        (1.0 - cfg.PLATFORM_RATE) * price
        + cfg.SUBSIDY_MAX_RATIO * group["p_ref"]
        - group["cost"]
    )
    max_accept_value = (
        max_driver_reward[:, None]
        + torch.pow(cfg.DRIVER_GAMMA, group["tau_steps"].float()) * vj_sample
        - cfg.DRIVER_GAMMA * wait_sample
    )
    max_single = (max_accept_value >= 0.0).to(max_accept_value.dtype).mean(dim=1)
    max_final = 1.0 - torch.pow(1.0 - max_single, dispatch)
    hard_infeasible = max_final < (1.0 - cfg.DRIVER_BETA)
    accepted = group["base_demand"] * passenger * driver_final
    if cfg.HARD_INFEASIBLE_ORDER_LOSS:
        accepted = torch.where(hard_infeasible, torch.zeros_like(accepted), accepted)
    accepted_total = accepted.sum()
    ratio = torch.where(
        accepted_total > 0,
        torch.minimum(torch.ones_like(accepted_total), group["supply"] / accepted_total.clamp_min(1e-6)),
        torch.zeros_like(accepted_total),
    )
    completed = accepted * ratio
    denominator = accepted_total.clamp_min(1e-6)
    context = torch.stack(
        [
            torch.log1p(group["supply"].clamp_min(0.0)),
            torch.log1p(group["demand_realized"].clamp_min(0.0)),
            torch.log1p(accepted_total.clamp_min(0.0)),
            torch.log1p(completed.sum().clamp_min(0.0)),
            ratio,
            torch.log1p((accepted * price).sum() / denominator),
            torch.log1p((accepted * subsidy).sum() / denominator),
        ]
    )
    return context, {
        "price": price,
        "subsidy": subsidy,
        "passenger": passenger,
        "driver_final": driver_final,
        "hard_infeasible": hard_infeasible,
        "accepted_demand": accepted,
        "completed": completed,
    }


@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for target_parameter, source_parameter in zip(target.parameters(), source.parameters()):
        target_parameter.mul_(1.0 - tau).add_(source_parameter, alpha=tau)
