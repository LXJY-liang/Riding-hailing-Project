# -*- coding: utf-8 -*-
"""任务 2：只读检查重构后的 S_it，不修改任何数据。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


config_path = Path(__file__).resolve().parents[1] / "1 数据准备" / "1 离线SAC统一配置.py"
spec = importlib.util.spec_from_file_location("offline_sac_config", config_path)
cfg = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(cfg)


def main() -> None:
    data = pd.read_parquet(cfg.SUPPLY_OUTPUT)
    active = data["D_origin_it"] > 0
    violations = data.loc[data["S_it_hist"] + 1e-6 < data["D_origin_it"]]

    print(f"行数：{len(data):,}")
    print(f"区域数：{data['region_id'].nunique():,}")
    print(f"时段数：{data['t'].nunique():,}")
    print(f"有成交的区域-时段：{active.sum():,}")
    print(f"有成交但 S=0：{((data['S_it_hist'] <= 0) & active).sum():,}")
    print(f"S < 历史成交量：{len(violations):,}")
    print(f"需要 U_min 的时段占比：{(data['U_min_it'] > 0).mean():.2%}")
    print(data[["S_it_hist", "D_origin_it", "D_dest_it", "U_min_it"]].describe())

    if len(violations):
        raise AssertionError("检查失败：存在 S_it_hist < D_origin_it。")


if __name__ == "__main__":
    main()
