# -*- coding: utf-8 -*-
"""任务 1：根据历史订单流重构第一版离线 SAC 的冻结供给 S_it。

重构规则：

    remaining_it = lambda_S * max(S_i,t-1 - O_i,t-1, 0)
    available_it = remaining_it + arrivals_it
    U_min_it     = max(O_it - available_it, 0)
    S_it_hist    = available_it + U_min_it

其中 O_it 是从区域 i 在 t 时段实际成交并出发的订单数，arrivals_it 是按真实订单结束
时间统计的到达司机数。U_min 不是观测到的上线司机，而是保证已成交订单可被历史供给
解释所需的最小外生净进入量。生成后 S_it_hist 在 SAC 训练中完全冻结。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_config():
    config_path = Path(__file__).resolve().parents[1] / "1 数据准备" / "1 离线SAC统一配置.py"
    spec = importlib.util.spec_from_file_location("offline_sac_config", config_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


cfg = load_config()


def reconstruct_one_region(group: pd.DataFrame, retain_rate: float) -> pd.DataFrame:
    """按时间递推一个区域的供给，保留便于审计的全部中间项。"""

    group = group.sort_values("t", kind="mergesort").copy()
    departures = group["D_origin_it"].to_numpy(dtype=np.float64)
    arrivals = group["D_dest_it"].to_numpy(dtype=np.float64)
    n = len(group)

    supply = np.zeros(n, dtype=np.float64)
    retained_idle = np.zeros(n, dtype=np.float64)
    available_before_entry = np.zeros(n, dtype=np.float64)
    minimum_external_entry = np.zeros(n, dtype=np.float64)

    # 数据从 6 月 1 日开始，无法观测 5 月 31 日夜间仍在线的司机。首时段采用最小可行
    # 初始化：只放入足以解释当前已成交订单的供给。后续正式训练从 6 月 11 日开始，
    # 因此前 10 天会自然成为供给递推的预热期。
    supply[0] = max(departures[0], arrivals[0])
    available_before_entry[0] = arrivals[0]
    minimum_external_entry[0] = max(supply[0] - arrivals[0], 0.0)

    for index in range(1, n):
        idle_after_previous_orders = max(supply[index - 1] - departures[index - 1], 0.0)
        retained_idle[index] = retain_rate * idle_after_previous_orders
        available_before_entry[index] = retained_idle[index] + arrivals[index]
        minimum_external_entry[index] = max(
            departures[index] - available_before_entry[index], 0.0
        )
        supply[index] = available_before_entry[index] + minimum_external_entry[index]

    group["retained_idle_it"] = retained_idle.astype(np.float32)
    group["available_before_entry_it"] = available_before_entry.astype(np.float32)
    group["U_min_it"] = minimum_external_entry.astype(np.float32)
    group["S_it_hist"] = supply.astype(np.float32)
    group["supply_slack_it"] = (supply - departures).astype(np.float32)
    return group


def main() -> None:
    cfg.ensure_output_directories()
    panel = pd.read_parquet(
        cfg.REGION_PANEL_FILE,
        columns=["region_id", "t", "D_origin_it", "D_dest_it"],
    )
    panel["region_id"] = panel["region_id"].astype("string")
    panel["t"] = pd.to_datetime(panel["t"], errors="raise")
    panel["D_origin_it"] = pd.to_numeric(panel["D_origin_it"], errors="raise").fillna(0)
    panel["D_dest_it"] = pd.to_numeric(panel["D_dest_it"], errors="raise").fillna(0)

    if panel[["region_id", "t"]].duplicated().any():
        raise ValueError("区域完整面板存在重复的 (region_id,t)。")
    if not 0.0 <= cfg.SUPPLY_RETAIN_RATE <= 1.0:
        raise ValueError("SUPPLY_RETAIN_RATE 必须位于 [0,1]。")

    pieces = [
        reconstruct_one_region(group, cfg.SUPPLY_RETAIN_RATE)
        for _, group in panel.groupby("region_id", sort=False, observed=True)
    ]
    result = pd.concat(pieces, ignore_index=True)
    result = result.sort_values(["t", "region_id"], kind="mergesort").reset_index(drop=True)

    inconsistent = result["S_it_hist"] + 1e-6 < result["D_origin_it"]
    if inconsistent.any():
        raise AssertionError("供给重构失败：仍存在 S_it_hist < 历史成交出发量。")
    if (result["S_it_hist"] < 0).any():
        raise AssertionError("供给重构失败：出现负供给。")

    result.to_parquet(cfg.SUPPLY_OUTPUT, index=False, compression="snappy")
    active = result["D_origin_it"] > 0
    diagnostics = {
        "source": str(cfg.REGION_PANEL_FILE),
        "retain_rate": float(cfg.SUPPLY_RETAIN_RATE),
        "rows": int(len(result)),
        "regions": int(result["region_id"].nunique()),
        "timestamps": int(result["t"].nunique()),
        "active_region_times": int(active.sum()),
        "active_rows_with_zero_supply": int(((result["S_it_hist"] <= 0) & active).sum()),
        "rows_requiring_minimum_external_entry": int((result["U_min_it"] > 0).sum()),
        "total_minimum_external_entry": float(result["U_min_it"].sum()),
        "mean_supply_active": float(result.loc[active, "S_it_hist"].mean()),
        "mean_supply_slack_active": float(result.loc[active, "supply_slack_it"].mean()),
    }
    cfg.SUPPLY_DIAGNOSTICS.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    print(f"输出：{cfg.SUPPLY_OUTPUT}")


if __name__ == "__main__":
    main()
