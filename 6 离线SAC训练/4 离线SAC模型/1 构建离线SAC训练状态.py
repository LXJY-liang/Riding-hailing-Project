# -*- coding: utf-8 -*-
"""任务 1：拼接 OD、冻结供给和过去信息 Vit 代理，生成离线 SAC 状态表。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_config():
    path = Path(__file__).resolve().parents[1] / "1 数据准备" / "1 离线SAC统一配置.py"
    spec = importlib.util.spec_from_file_location("offline_sac_config", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


cfg = load_config()


def select_v_summary(vit: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """选择 Actor 状态需要的三个 V 分位数，并增加指定前缀。"""

    selected = vit[["region_id", "t", "V_q10", "V_q50", "V_q90"]].copy()
    return selected.rename(
        columns={
            "region_id": f"{prefix}_region_id",
            "t": f"{prefix}_t",
            "V_q10": f"{prefix}_q10",
            "V_q50": f"{prefix}_q50",
            "V_q90": f"{prefix}_q90",
        }
    )


def main() -> None:
    cfg.ensure_output_directories()
    od = pd.read_parquet(cfg.BASE_OD_OUTPUT)
    od["t"] = pd.to_datetime(od["t"])
    od = od.loc[od["split"].isin(["train", "valid", "test"])].copy()

    supply = pd.read_parquet(
        cfg.SUPPLY_OUTPUT, columns=["region_id", "t", "D_origin_it", "S_it_hist"]
    )
    supply["region_id"] = supply["region_id"].astype("string")
    supply["t"] = pd.to_datetime(supply["t"])
    od = od.merge(
        supply.rename(columns={"region_id": "i"}), on=["i", "t"], how="left", validate="many_to_one"
    )
    od = od.rename(columns={"D_origin_it": "D_it_observed"})

    vit = pd.read_parquet(
        cfg.VIT_OUTPUT,
        columns=["region_id", "t", "V_proxy_available", "V_q10", "V_q50", "V_q90"],
    )
    vit = vit.loc[vit["V_proxy_available"] == 1].copy()
    vit["region_id"] = vit["region_id"].astype("string")
    vit["t"] = pd.to_datetime(vit["t"])

    # 当前司机状态 V(i,t)。
    current_v = select_v_summary(vit, "V_i")
    od["V_i_region_id"] = od["i"]
    od["V_i_t"] = od["t"]
    od = od.merge(current_v, on=["V_i_region_id", "V_i_t"], how="left", validate="many_to_one")

    # 拒单后等待一个时段的 V(i,t+1)。
    reject_v = select_v_summary(vit, "V_reject")
    od["V_reject_region_id"] = od["i"]
    od["V_reject_t"] = od["t"] + pd.Timedelta(minutes=cfg.TIME_STEP_MINUTES)
    od = od.merge(
        reject_v, on=["V_reject_region_id", "V_reject_t"], how="left", validate="many_to_one"
    )

    # 接单后按历史平均行程步数到达目的地的 V(j,t+l)。奖励环境对 tau 场景的不同 l 会
    # 直接查询完整 V 表；这里的三个分位数仅作为 Actor 的可见状态摘要。
    dest_v = select_v_summary(vit, "V_dest")
    od["V_dest_region_id"] = od["j"]
    od["V_dest_t"] = od["t"] + pd.to_timedelta(
        od["arrival_step_ijt"] * cfg.TIME_STEP_MINUTES, unit="min"
    )
    od = od.merge(dest_v, on=["V_dest_region_id", "V_dest_t"], how="left", validate="many_to_one")

    # tau 场景插值后可能产生 q10 与 q90 之间的任意整数旅行步数。完整 V 表覆盖所有
    # 区域，因此只需确保最长 q90 到达时刻仍位于 V_proxy 时间范围内。
    max_travel_steps = np.maximum(
        np.ceil(od["tau_q90_ijt"] / cfg.TIME_STEP_MINUTES).astype(np.int16), 1
    )
    latest_v_time = vit["t"].max()
    full_destination_support = (
        od["t"]
        + pd.to_timedelta(max_travel_steps * cfg.TIME_STEP_MINUTES, unit="min")
        <= latest_v_time
    )
    od = od.loc[full_destination_support].copy()

    required = [
        "S_it_hist", "V_i_q10", "V_i_q50", "V_i_q90",
        "V_reject_q10", "V_reject_q50", "V_reject_q90",
        "V_dest_q10", "V_dest_q50", "V_dest_q90",
    ]
    missing_before_drop = {column: int(od[column].isna().sum()) for column in required}
    od = od.dropna(subset=required).copy()

    od["log1p_D_ijt"] = np.log1p(od["D_ijt_observed"]).astype(np.float32)
    od["log1p_D_it"] = np.log1p(od["D_it_observed"]).astype(np.float32)
    od["log1p_S_it"] = np.log1p(od["S_it_hist"]).astype(np.float32)
    od["demand_supply_ratio"] = (
        od["D_it_observed"] / od["S_it_hist"].clip(lower=1e-6)
    ).astype(np.float32)
    od["od_demand_share"] = (
        od["D_ijt_observed"] / od["D_it_observed"].clip(lower=1e-6)
    ).astype(np.float32)
    od["slot_sin"] = np.sin(2 * np.pi * od["slot_in_day"] / 96.0).astype(np.float32)
    od["slot_cos"] = np.cos(2 * np.pi * od["slot_in_day"] / 96.0).astype(np.float32)
    od["dow_sin"] = np.sin(2 * np.pi * od["dow"] / 7.0).astype(np.float32)
    od["dow_cos"] = np.cos(2 * np.pi * od["dow"] / 7.0).astype(np.float32)

    if (od["S_it_hist"] + 1e-6 < od["D_ijt_observed"]).any():
        # 单个 OD 的 D 本应不大于区域总出发量，区域总出发量又不大于 S。
        raise AssertionError("拼接后的状态出现 D_ijt_observed > S_it_hist。")
    if od[cfg.STATE_NUMERIC_COLUMNS].isna().any().any():
        columns = od[cfg.STATE_NUMERIC_COLUMNS].columns[
            od[cfg.STATE_NUMERIC_COLUMNS].isna().any()
        ].tolist()
        raise ValueError(f"Actor 数值状态仍有缺失列：{columns}")

    od = od.sort_values(["t", "i", "j"], kind="mergesort").reset_index(drop=True)
    od["state_id"] = np.arange(len(od), dtype=np.int64)
    od["group_id"] = pd.factorize(pd.MultiIndex.from_frame(od[["i", "t"]]), sort=False)[0].astype(
        np.int64
    )

    # 当前实现的环境样本是完整 (i,t) 组，但回放池仍按 OD 行保存局部经验。下一时段
    # 优先连接同一 (i,j)；如果该 j 在下一时段没有订单，则连接同一出发区域下一时段
    # 成交量最大的有效目的地。这样不需要给零订单 OD 输出动作，也不会把“目的地集合
    # 改变”误判为 episode 终止。
    od_by_pair = od.sort_values(["i", "j", "t"], kind="mergesort")
    next_id = od_by_pair.groupby(["i", "j"], sort=False)["state_id"].shift(-1)
    next_t = od_by_pair.groupby(["i", "j"], sort=False)["t"].shift(-1)
    next_split = od_by_pair.groupby(["i", "j"], sort=False)["split"].shift(-1)
    exact_next = (
        (next_t == od_by_pair["t"] + pd.Timedelta(minutes=cfg.TIME_STEP_MINUTES))
        & (next_split == od_by_pair["split"])
    )

    # 为每个下一时段组准备一个代表 OD；排序确保最大 D 优先，平局时 j 最小以保证复现。
    representative = (
        od.sort_values(
            ["i", "t", "D_ijt_observed", "j"],
            ascending=[True, True, False, True],
            kind="mergesort",
        )
        .drop_duplicates(["i", "t"])
        [["i", "t", "split", "state_id"]]
        .rename(
            columns={
                "t": "fallback_next_t",
                "split": "fallback_next_split",
                "state_id": "fallback_next_state_id",
            }
        )
    )
    od_by_pair["fallback_next_t"] = od_by_pair["t"] + pd.Timedelta(
        minutes=cfg.TIME_STEP_MINUTES
    )
    od_by_pair = od_by_pair.merge(
        representative,
        on=["i", "fallback_next_t"],
        how="left",
        validate="many_to_one",
    )
    valid_fallback = od_by_pair["fallback_next_split"].eq(od_by_pair["split"])
    od_by_pair["next_state_id"] = np.where(
        exact_next,
        next_id,
        np.where(valid_fallback, od_by_pair["fallback_next_state_id"], -1),
    ).astype(np.int64)
    od_by_pair["next_state_fallback"] = ((~exact_next) & valid_fallback).astype(np.int8)
    next_map = od_by_pair.set_index("state_id")[["next_state_id", "next_state_fallback"]]
    od = od.join(next_map, on="state_id")
    od["next_state_id"] = od["next_state_id"].astype(np.int64)
    od["next_state_fallback"] = od["next_state_fallback"].astype(np.int8)
    od["done"] = (od["next_state_id"] < 0).astype(np.int8)

    # 删除仅用于合并的重复键，保留 V 时间以便审计目的地到达时刻。
    od = od.drop(columns=["V_i_region_id", "V_i_t", "V_reject_region_id", "V_dest_region_id"])
    float64_columns = od.select_dtypes(include=["float64"]).columns
    od[float64_columns] = od[float64_columns].astype(np.float32)
    od.to_parquet(cfg.STATE_OUTPUT, index=False, compression="snappy")

    diagnostics = {
        "rows": int(len(od)),
        "groups": int(od["group_id"].nunique()),
        "split_rows": {str(k): int(v) for k, v in od["split"].value_counts().items()},
        "split_groups": {
            str(k): int(v) for k, v in od.groupby("split")["group_id"].nunique().items()
        },
        "done_rate": float(od["done"].mean()),
        "next_state_fallback_rate": float(od["next_state_fallback"].mean()),
        "missing_before_drop": missing_before_drop,
        "active_rows_with_zero_supply": int(
            ((od["D_ijt_observed"] > 0) & (od["S_it_hist"] <= 0)).sum()
        ),
    }
    cfg.STATE_DIAGNOSTICS.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    print(f"输出：{cfg.STATE_OUTPUT}")


if __name__ == "__main__":
    main()
