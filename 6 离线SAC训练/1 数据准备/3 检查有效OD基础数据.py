# -*- coding: utf-8 -*-
"""任务 3：只读检查离线 SAC 基础 OD 表的键、动作集合和分位数。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG_PATH = Path(__file__).resolve().parent / "1 离线SAC统一配置.py"
CONFIG_SPEC = importlib.util.spec_from_file_location("offline_sac_config", CONFIG_PATH)
cfg = importlib.util.module_from_spec(CONFIG_SPEC)
assert CONFIG_SPEC.loader is not None
CONFIG_SPEC.loader.exec_module(cfg)

BASE_OD_OUTPUT = cfg.BASE_OD_OUTPUT


def main() -> None:
    data = pd.read_parquet(BASE_OD_OUTPUT)
    pe_columns = ["pe_q10_ijt", "pe_q25_ijt", "pe_q50_ijt", "pe_q75_ijt", "pe_q90_ijt"]
    pe = data[pe_columns].to_numpy(dtype=np.float64)
    duplicate_keys = int(data[["i", "j", "t"]].duplicated().sum())
    nonmonotone = int(np.any(np.diff(pe, axis=1) < -1e-6, axis=1).sum())
    degenerate = int((np.ptp(pe, axis=1) < 1e-8).sum())

    print(f"有订单 OD 行数：{len(data):,}")
    print(f"(i,j,t) 重复键：{duplicate_keys:,}")
    print(f"D<=0：{(data['D_ijt_observed'] <= 0).sum():,}")
    print(f"p_ref<=0：{(data['p_ref_ijt'] <= 0).sum():,}")
    print(f"乘客价格代理分位数不单调：{nonmonotone:,}")
    print(f"乘客价格代理退化分布：{degenerate:,}")
    print(data.groupby("split").agg(rows=("state_id", "size"), groups=("group_id", "nunique")))

    if duplicate_keys or nonmonotone or degenerate:
        raise AssertionError("基础 OD 状态检查失败。")


if __name__ == "__main__":
    main()
