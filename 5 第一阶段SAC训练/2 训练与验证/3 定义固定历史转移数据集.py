# -*- coding: utf-8 -*-
"""固定历史转移索引：只抽样，不写入、不淘汰、不随epoch变化。"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def require_finite(name: str, value, group: dict) -> None:
    if not np.isfinite(np.asarray(value)).all():
        raise FloatingPointError(
            f"固定历史字段{name}出现NaN/inf：date={group.get('date')}, "
            f"slot={group.get('slot')}, i={int(np.asarray(group['i'])[0])}"
        )


def freeze_group(group: dict) -> dict:
    """复制历史状态组；明确排除action、reward和constraint outcome。"""

    array_fields = (
        "state", "i", "j", "p_ref", "pe_q", "tau_steps", "vi_index",
        "vj_index", "wait_q", "vj_scenario_q", "cost", "pi", "base_demand",
    )
    for name in array_fields:
        require_finite(name, group[name], group)
    return {
        "date": str(group["date"]),
        "slot": int(group["slot"]),
        "state": np.asarray(group["state"], dtype=np.float32),
        "i": np.asarray(group["i"], dtype=np.int16),
        "j": np.asarray(group["j"], dtype=np.int16),
        "p_ref": np.asarray(group["p_ref"], dtype=np.float32),
        "pe_q": np.asarray(group["pe_q"], dtype=np.float16),
        "tau_steps": np.asarray(group["tau_steps"], dtype=np.int16),
        "vi_index": np.asarray(group["vi_index"], dtype=np.uint8),
        "vj_index": np.asarray(group["vj_index"], dtype=np.uint8),
        "wait_q": np.asarray(group["wait_q"], dtype=np.float16),
        "vj_scenario_q": np.asarray(group["vj_scenario_q"], dtype=np.float16),
        "supply": float(group["supply"]),
        "demand_realized": float(group["demand_realized"]),
        "cost": np.asarray(group["cost"], dtype=np.float32),
        "pi": np.asarray(group["pi"], dtype=np.float32),
        "base_demand": np.asarray(group["base_demand"], dtype=np.float32),
    }


class FixedHistoricalTransitionDataset:
    """磁盘episode上的固定转移数据集。

    数据集建立后共有固定数量的历史(i,t)转移。sample只读取这些索引，不存在
    add、append、FIFO、容量上限或Actor轨迹写回。
    """

    def __init__(self, episode_files: list[Path], episode_class, seed: int = 42):
        self.episode_files = [Path(path) for path in episode_files]
        self.episode_class = episode_class
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self._episodes: dict[int, object] = {}
        episode_index: list[int] = []
        current_group: list[int] = []
        next_episode_index: list[int] = []
        next_group: list[int] = []
        gap_steps: list[int] = []
        metadata = []
        for file_index, path in enumerate(self.episode_files):
            with np.load(path, allow_pickle=False) as data:
                date = str(data["date"])
                origins = data["group_origin"].astype(np.int32)
                slots = data["group_slot"].astype(np.int16)
            metadata.append((date, origins, slots))

        # 全局索引覆盖本次split内所有自然日，使23:45可以连接到次日00:00。
        # 不在episode_files中的日期不会进入索引，因此不会跨训练/验证/测试边界。
        lookup = {
            (date, int(origin), int(slot)): (file_index, int(group_id))
            for file_index, (date, origins, slots) in enumerate(metadata)
            for group_id, (origin, slot) in enumerate(zip(origins, slots))
        }
        self.available_dates = {date for date, _, _ in metadata}
        for file_index, (date, origins, slots) in enumerate(metadata):
            for group_id, (origin, slot) in enumerate(zip(origins, slots)):
                if int(slot) + 1 < 96:
                    next_date, next_slot = date, int(slot) + 1
                else:
                    next_date = str(np.datetime64(date, "D") + np.timedelta64(1, "D"))
                    next_slot = 0
                # 严格只连接历史下一15分钟。次日不属于当前split，或下一时段
                # 没有该区域OD决策组时，才把这条固定历史转移视为终止。
                following = lookup.get((next_date, int(origin), next_slot))
                episode_index.append(file_index)
                current_group.append(int(group_id))
                next_episode_index.append(following[0] if following is not None else -1)
                next_group.append(following[1] if following is not None else -1)
                gap_steps.append(1 if following is not None else 0)
        self.episode_index = np.asarray(episode_index, dtype=np.int16)
        self.current_group = np.asarray(current_group, dtype=np.int32)
        self.next_episode_index = np.asarray(next_episode_index, dtype=np.int16)
        self.next_group = np.asarray(next_group, dtype=np.int32)
        self.gap_steps = np.asarray(gap_steps, dtype=np.int16)
        if not (
            len(self.episode_index)
            == len(self.current_group)
            == len(self.next_episode_index)
            == len(self.next_group)
            == len(self.gap_steps)
        ):
            raise RuntimeError("固定历史转移索引长度不一致。")

    def __len__(self) -> int:
        return len(self.current_group)

    def _episode(self, index: int):
        if index not in self._episodes:
            self._episodes[index] = self.episode_class(
                self.episode_files[index], self.seed + index, self.available_dates
            )
        return self._episodes[index]

    def transition(self, index: int) -> dict:
        file_index = int(self.episode_index[index])
        episode = self._episode(file_index)
        current = freeze_group(episode.build_group(int(self.current_group[index])))
        following = int(self.next_group[index])
        following_episode = int(self.next_episode_index[index])
        return {
            "current": current,
            "next": (
                freeze_group(self._episode(following_episode).build_group(following))
                if following >= 0 else None
            ),
            "gap_steps": int(self.gap_steps[index]),
            "done": following < 0,
        }

    def sample(self, size: int) -> list[dict]:
        if size <= 0 or len(self) == 0:
            raise ValueError("固定历史转移数据集或抽样大小无效。")
        indices = self.rng.integers(0, len(self), size=size)
        return [self.transition(int(index)) for index in indices]

    def audit(self) -> dict:
        positive = self.gap_steps[self.gap_steps > 0]
        return {
            "transitions": len(self),
            "episode_days": len(self.episode_files),
            "terminal_transitions": int((self.next_group < 0).sum()),
            "cross_day_transitions": int(
                ((self.next_episode_index >= 0) & (self.next_episode_index != self.episode_index)).sum()
            ),
            "minimum_gap_steps": int(positive.min()) if len(positive) else 0,
            "maximum_gap_steps": int(self.gap_steps.max()) if len(self.gap_steps) else 0,
            "mutable_experience_methods": False,
        }
