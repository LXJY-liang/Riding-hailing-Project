# -*- coding: utf-8 -*-
"""任务 4：用小型合成样本测试接受概率、供给缩放和反事实收益等核心公式。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_DIR = Path(__file__).resolve().parent


def load(file_name: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, MODULE_DIR / file_name)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


reward_module = load("2 计算反事实收益.py", "counterfactual_reward_test")


def synthetic_engine() -> reward_module.CounterfactualReward:
    engine = reward_module.CounterfactualReward.__new__(reward_module.CounterfactualReward)
    engine.num_scenarios = 32
    engine.v_times = pd.date_range("2024-06-10", periods=16, freq="15min")
    engine.time_to_index = {timestamp: index for index, timestamp in enumerate(engine.v_times)}
    levels = len(reward_module.cfg.VIT_QUANTILE_COLUMNS)
    base = np.linspace(5.0, 15.0, levels, dtype=np.float32)
    engine.v_values = np.empty((16, 2, levels), dtype=np.float32)
    engine.v_values[:, 0, :] = base
    engine.v_values[:, 1, :] = base + 2.0
    return engine


def synthetic_rows() -> pd.DataFrame:
    shared = {
        "group_id": 0,
        "i": "A",
        "t": pd.Timestamp("2024-06-10 00:00:00"),
        "i_index": 0,
        "D_ijt_observed": 2,
        "S_it_hist": 4.0,
        "p_ref_ijt": 20.0,
        "p_q10_ijt": 17.0,
        "p_q90_ijt": 23.0,
        "pe_q10_ijt": 17.0,
        "pe_q25_ijt": 18.5,
        "pe_q50_ijt": 20.0,
        "pe_q75_ijt": 21.5,
        "pe_q90_ijt": 23.0,
        "tau_q10_ijt": 10.0,
        "tau_q25_ijt": 12.0,
        "tau_q50_ijt": 15.0,
        "tau_q75_ijt": 16.0,
        "tau_q90_ijt": 20.0,
        "c_q10_ijt": 4.0,
        "c_q25_ijt": 4.5,
        "c_q50_ijt": 5.0,
        "c_q75_ijt": 5.5,
        "c_q90_ijt": 6.0,
    }
    return pd.DataFrame(
        [
            {**shared, "state_id": 1, "j": "B", "j_index": 1},
            {**shared, "state_id": 2, "j": "A", "j_index": 0},
        ]
    )


def main() -> None:
    engine = synthetic_engine()
    rows = synthetic_rows()
    low_price = np.full((2, 2), -1.0, dtype=np.float32)
    high_price = np.full((2, 2), 1.0, dtype=np.float32)
    low = engine.evaluate_group(rows, low_price, subsidy_policy="force_zero")
    high = engine.evaluate_group(rows, high_price, subsidy_policy="force_zero")

    assert low["A_ijt"].mean() >= high["A_ijt"].mean(), "价格升高不应提高乘客接受率"
    assert (low["b_ijt"] >= 0).all(), "补贴必须非负"
    assert (low["M_ijt"] >= 0).all(), "成交量必须非负"
    assert (low["supply_scale"].between(0, 1)).all(), "供给缩放必须位于 [0,1]"
    assert low["M_ijt"].sum() <= rows["S_it_hist"].iloc[0] + 1e-6
    print("核心公式单元测试通过。")


if __name__ == "__main__":
    main()
