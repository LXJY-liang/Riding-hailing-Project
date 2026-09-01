# -*- coding: utf-8 -*-
"""任务 2：只读检查 Vit 分位数单调性、覆盖率和代理历史长度。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


path = Path(__file__).resolve().parents[1] / "1 数据准备" / "1 离线SAC统一配置.py"
spec = importlib.util.spec_from_file_location("offline_sac_config", path)
cfg = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(cfg)


def main() -> None:
    data = pd.read_parquet(cfg.VIT_OUTPUT)
    proxy = data[cfg.VIT_QUANTILE_COLUMNS].to_numpy(dtype=np.float64)
    oracle_columns = [f"oracle_{column}" for column in cfg.VIT_QUANTILE_COLUMNS]
    oracle = data[oracle_columns].to_numpy(dtype=np.float64)

    proxy_nonmonotone = np.any(np.diff(proxy, axis=1) < -1e-6, axis=1)
    oracle_nonmonotone = np.any(np.diff(oracle, axis=1) < -1e-6, axis=1)
    print(f"行数：{len(data):,}")
    print(f"代理可用率：{data['V_proxy_available'].mean():.2%}")
    print(f"代理分位数不单调：{np.nansum(proxy_nonmonotone):,}")
    print(f"Oracle 分位数不单调：{oracle_nonmonotone.sum():,}")
    print(data[["V_q05", "V_q50", "V_q95", "oracle_V_q50"]].describe())

    if oracle_nonmonotone.any() or np.nansum(proxy_nonmonotone):
        raise AssertionError("V_it 分位数存在不单调行。")


if __name__ == "__main__":
    main()
