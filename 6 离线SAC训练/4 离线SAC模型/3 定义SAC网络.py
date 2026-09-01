# -*- coding: utf-8 -*-
"""任务 3：定义离线 SAC 的 Actor、双 Critic、目标网络更新和内存回放池。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch
from torch import nn


def load_config():
    path = Path(__file__).resolve().parents[1] / "1 数据准备" / "1 离线SAC统一配置.py"
    spec = importlib.util.spec_from_file_location("offline_sac_config", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


cfg = load_config()


def mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = input_dim
    for width in hidden_dims:
        layers.extend([nn.Linear(current, width), nn.ReLU()])
        current = width
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """共享 Actor：对每个有订单的 (i,j,t) 输出二维连续动作。"""

    def __init__(self, numeric_dim: int, region_count: int):
        super().__init__()
        emb = cfg.REGION_EMBEDDING_DIM
        self.i_embedding = nn.Embedding(region_count, emb)
        self.j_embedding = nn.Embedding(region_count, emb)
        self.encoder = mlp(numeric_dim + 2 * emb, cfg.ACTOR_HIDDEN_DIMS, 2 * cfg.ACTOR_HIDDEN_DIMS[-1])
        self.mean = nn.Linear(2 * cfg.ACTOR_HIDDEN_DIMS[-1], 2)
        self.log_std = nn.Linear(2 * cfg.ACTOR_HIDDEN_DIMS[-1], 2)

    def forward(self, numeric: torch.Tensor, i_index: torch.Tensor, j_index: torch.Tensor):
        encoded = torch.cat(
            [numeric, self.i_embedding(i_index), self.j_embedding(j_index)], dim=-1
        )
        hidden = self.encoder(encoded)
        return self.mean(hidden), self.log_std(hidden).clamp(-5.0, 2.0)

    def sample(self, numeric: torch.Tensor, i_index: torch.Tensor, j_index: torch.Tensor):
        mean, log_std = self(numeric, i_index, j_index)
        distribution = torch.distributions.Normal(mean, log_std.exp())
        raw = distribution.rsample()
        action = torch.tanh(raw)
        log_prob = distribution.log_prob(raw) - torch.log(1.0 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)

    def deterministic(self, numeric: torch.Tensor, i_index: torch.Tensor, j_index: torch.Tensor):
        mean, _ = self(numeric, i_index, j_index)
        return torch.tanh(mean)


class Critic(nn.Module):
    def __init__(self, numeric_dim: int, region_count: int):
        super().__init__()
        emb = cfg.REGION_EMBEDDING_DIM
        self.i_embedding = nn.Embedding(region_count, emb)
        self.j_embedding = nn.Embedding(region_count, emb)
        self.network = mlp(numeric_dim + 2 * emb + 2, cfg.CRITIC_HIDDEN_DIMS, 1)

    def forward(
        self,
        numeric: torch.Tensor,
        i_index: torch.Tensor,
        j_index: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        encoded = torch.cat(
            [numeric, self.i_embedding(i_index), self.j_embedding(j_index), action], dim=-1
        )
        return self.network(encoded)


class ReplayBuffer:
    """保存反事实样本的原始平台收益与两个违反度。"""

    def __init__(self, capacity: int, numeric_dim: int, seed: int):
        self.capacity = int(capacity)
        self.numeric_dim = int(numeric_dim)
        self.rng = np.random.default_rng(seed)
        self.position = 0
        self.size = 0
        self.state = np.empty((capacity, numeric_dim), dtype=np.float32)
        self.next_state = np.empty((capacity, numeric_dim), dtype=np.float32)
        self.i_index = np.empty(capacity, dtype=np.int64)
        self.j_index = np.empty(capacity, dtype=np.int64)
        self.next_i_index = np.empty(capacity, dtype=np.int64)
        self.next_j_index = np.empty(capacity, dtype=np.int64)
        self.action = np.empty((capacity, 2), dtype=np.float32)
        self.profit = np.empty((capacity, 1), dtype=np.float32)
        self.g_passenger = np.empty((capacity, 1), dtype=np.float32)
        self.g_driver = np.empty((capacity, 1), dtype=np.float32)
        self.done = np.empty((capacity, 1), dtype=np.float32)

    def add_batch(self, batch: dict[str, np.ndarray]) -> None:
        n = len(batch["state"])
        indices = (np.arange(n) + self.position) % self.capacity
        for name in (
            "state", "next_state", "i_index", "j_index", "next_i_index", "next_j_index",
            "action", "profit", "g_passenger", "g_driver", "done",
        ):
            getattr(self, name)[indices] = batch[name]
        self.position = int((self.position + n) % self.capacity)
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        indices = self.rng.integers(0, self.size, size=batch_size)
        result = {}
        for name in (
            "state", "next_state", "action", "profit", "g_passenger", "g_driver", "done",
        ):
            result[name] = torch.as_tensor(getattr(self, name)[indices], device=device)
        for name in ("i_index", "j_index", "next_i_index", "next_j_index"):
            result[name] = torch.as_tensor(
                getattr(self, name)[indices], dtype=torch.long, device=device
            )
        return result


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_parameter, source_parameter in zip(target.parameters(), source.parameters()):
            target_parameter.mul_(1.0 - tau).add_(source_parameter, alpha=tau)
