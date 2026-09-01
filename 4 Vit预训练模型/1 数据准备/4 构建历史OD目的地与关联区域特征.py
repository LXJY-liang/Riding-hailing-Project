# -*- coding: utf-8 -*-
"""用SAC训练日之前的暖机历史，构建Top-10目的地、tau与三类关联区域特征。"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("vit_pretrain_config", HERE / "1 Vit预训练配置.py")
cfg = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(cfg)


def aggregate_prior(source: pd.DataFrame, day_type: int) -> dict[int, pd.DataFrame]:
    """按目标时段构建暖机期条件OD流量和tau先验。"""

    selected = source if day_type == -1 else source.loc[source["is_weekend_or_holiday"] == day_type]
    expanded = []
    for offset in range(-cfg.HISTORY_SLOT_RADIUS, cfg.HISTORY_SLOT_RADIUS + 1):
        part = selected.copy()
        part["target_slot"] = part["slot"] - offset
        part = part.loc[part["target_slot"].between(0, cfg.TIME_STEPS_PER_DAY - 1)]
        expanded.append(part)
    data = pd.concat(expanded, ignore_index=True)
    weight = data["D_ijt_observed"].to_numpy(np.float64)
    for column in ("tau_q10_ijt", "tau_q50_ijt", "tau_q90_ijt"):
        data[f"{column}_weighted"] = data[column].fillna(data["tau_bar_ijt"]).to_numpy(np.float64) * weight
    grouped = data.groupby(["target_slot", "i", "j"], observed=True, sort=False).agg(
        flow=("D_ijt_observed", "sum"),
        tau10_sum=("tau_q10_ijt_weighted", "sum"),
        tau50_sum=("tau_q50_ijt_weighted", "sum"),
        tau90_sum=("tau_q90_ijt_weighted", "sum"),
    ).reset_index()
    grouped = grouped.loc[grouped["flow"] > 0]
    for source_column, target_column in (
        ("tau10_sum", "tau_q10"),
        ("tau50_sum", "tau_q50"),
        ("tau90_sum", "tau_q90"),
    ):
        grouped[target_column] = grouped[source_column] / grouped["flow"]
    return {
        int(slot): frame[["i", "j", "flow", "tau_q10", "tau_q50", "tau_q90"]].copy()
        for slot, frame in grouped.groupby("target_slot", observed=True, sort=False)
    }


def slot_prior(
    specific: dict[int, pd.DataFrame],
    fallback: dict[int, pd.DataFrame],
    slot: int,
    region_ids: np.ndarray,
) -> pd.DataFrame:
    current = specific.get(slot, pd.DataFrame())
    all_types = fallback.get(slot, pd.DataFrame())
    if current.empty:
        return all_types.copy()
    present = set(current["i"].astype(str))
    missing = set(region_ids) - present
    if not missing:
        return current.copy()
    return pd.concat(
        [current, all_types.loc[all_types["i"].astype(str).isin(missing)]],
        ignore_index=True,
    )


def topk_arrays(prior: pd.DataFrame, region_ids: np.ndarray) -> dict[str, np.ndarray]:
    region_position = {region_id: index for index, region_id in enumerate(region_ids)}
    prior = prior.copy()
    prior["i"] = prior["i"].astype(str)
    prior["j"] = prior["j"].astype(str)
    total = prior.groupby("i", observed=True)["flow"].transform("sum")
    prior["pi"] = prior["flow"] / total
    prior = prior.sort_values(["i", "pi", "j"], ascending=[True, False, True], kind="mergesort")
    prior["rank"] = prior.groupby("i", observed=True).cumcount()
    top = prior.loc[prior["rank"] < cfg.TOP_K_DESTINATIONS]

    shape = (len(region_ids), cfg.TOP_K_DESTINATIONS)
    destination = np.full(shape, -1, dtype=np.int16)
    probability = np.zeros(shape, dtype=np.float32)
    tau10 = np.full(shape, np.nan, dtype=np.float32)
    tau50 = np.full(shape, np.nan, dtype=np.float32)
    tau90 = np.full(shape, np.nan, dtype=np.float32)
    origin = top["i"].map(region_position).to_numpy(np.int16)
    rank = top["rank"].to_numpy(np.int16)
    destination[origin, rank] = top["j"].map(region_position).fillna(-1).to_numpy(np.int16)
    probability[origin, rank] = top["pi"].to_numpy(np.float32)
    tau10[origin, rank] = top["tau_q10"].to_numpy(np.float32)
    tau50[origin, rank] = top["tau_q50"].to_numpy(np.float32)
    tau90[origin, rank] = top["tau_q90"].to_numpy(np.float32)

    same = prior.loc[prior["i"] == prior["j"]].set_index("i")["pi"]
    same_probability = pd.Series(region_ids).map(same).fillna(0.0).to_numpy(np.float32)
    top_mass = probability.sum(axis=1)
    available = np.asarray(pd.Series(region_ids).isin(prior["i"].unique()), dtype=np.int8)
    return {
        "destination": destination,
        "probability": probability,
        "tau10": tau10,
        "tau50": tau50,
        "tau90": tau90,
        "top_mass": top_mass,
        "other_mass": np.where(available > 0, np.maximum(1.0 - top_mass, 0.0), 0.0).astype(np.float32),
        "same_mass": same_probability,
        "available": available,
    }


def relation_features(
    prior: pd.DataFrame,
    previous_region: pd.DataFrame,
    previous_od: pd.DataFrame,
    region_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    position = {region_id: index for index, region_id in enumerate(region_ids)}
    demand = previous_region.set_index("region_id")["D_origin_it"].reindex(region_ids).fillna(0.0)
    supply = previous_region.set_index("region_id")["S_bar_it"].reindex(region_ids).fillna(0.0)

    prior = prior.copy()
    prior["i"] = prior["i"].astype(str)
    prior["j"] = prior["j"].astype(str)
    prior["out_weight"] = prior["flow"] / prior.groupby("i", observed=True)["flow"].transform("sum")
    prior["dest_demand"] = prior["j"].map(demand).fillna(0.0)
    prior["dest_supply"] = prior["j"].map(supply).fillna(0.0)
    prior["weighted_demand"] = prior["out_weight"] * prior["dest_demand"]
    prior["weighted_pressure"] = prior["out_weight"] * np.divide(
        prior["dest_demand"], np.maximum(prior["dest_supply"], 1e-6)
    )
    dest = prior.groupby("i", observed=True)["weighted_demand"].sum()
    pressure = prior.groupby("i", observed=True)["weighted_pressure"].sum()

    prior["in_weight"] = prior["flow"] / prior.groupby("j", observed=True)["flow"].transform("sum")
    actual = previous_od[["i", "j", "D_ijt_observed"]].copy()
    actual["i"] = actual["i"].astype(str)
    actual["j"] = actual["j"].astype(str)
    inbound = prior[["i", "j", "in_weight"]].merge(actual, on=["i", "j"], how="left")
    inbound["weighted_incoming"] = inbound["in_weight"] * inbound["D_ijt_observed"].fillna(0.0)
    origin = inbound.groupby("j", observed=True)["weighted_incoming"].sum()
    return (
        pd.Series(region_ids).map(dest).fillna(0.0).to_numpy(np.float32),
        pd.Series(region_ids).map(origin).fillna(0.0).to_numpy(np.float32),
        pd.Series(region_ids).map(pressure).fillna(0.0).to_numpy(np.float32),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", choices=tuple(cfg.SPLITS), default=list(cfg.SPLITS))
    args = parser.parse_args()
    cfg.ensure_directories()

    mapping = pd.read_csv(cfg.REGION_MAP_FILE, dtype={"region_id": "string"}).sort_values("region_index")
    region_ids = mapping["region_id"].astype(str).to_numpy()
    od = pd.read_parquet(
        cfg.OD_HISTORY_FILE,
        columns=[
            "i", "j", "t", "slot_in_day", "is_weekend_or_holiday", "D_ijt_observed",
            "tau_bar_ijt", "tau_q10_ijt", "tau_q50_ijt", "tau_q90_ijt",
        ],
    )
    od["i"] = od["i"].astype(str)
    od["j"] = od["j"].astype(str)
    od["t"] = pd.to_datetime(od["t"])
    od["slot"] = od["slot_in_day"].astype(np.int16)
    warmup = od.loc[od["t"] < pd.Timestamp(cfg.SPLITS["train"][0]).normalize()].copy()
    priors = {day_type: aggregate_prior(warmup, day_type) for day_type in (-1, 0, 1)}

    region = pd.read_parquet(cfg.REGION_HISTORY_FILE, columns=["region_id", "t", "D_origin_it"])
    supply = pd.read_parquet(cfg.SUPPLY_HISTORY_FILE, columns=["region_id", "t", "S_bar_it"])
    for frame in (region, supply):
        frame["region_id"] = frame["region_id"].astype(str)
        frame["t"] = pd.to_datetime(frame["t"])
    previous_region = region.merge(supply, on=["region_id", "t"], how="left", validate="one_to_one")
    region_by_time = previous_region.set_index("t").sort_index()
    od_by_time = od.set_index("t").sort_index()

    requested_days = sorted(
        {
            day
            for split in args.splits
            for day in pd.date_range(
                pd.Timestamp(cfg.SPLITS[split][0]).normalize(),
                pd.Timestamp(cfg.SPLITS[split][1]).normalize(),
                freq="D",
            )
        }
    )
    for day in requested_days:
        source_file = cfg.REGION_HISTORY_FEATURE_DIR / f"vit_region_history_{day:%Y-%m-%d}.parquet"
        state = pd.read_parquet(source_file).sort_values(["t", "region_id"], kind="mergesort")
        day_type = int(state["is_weekend_or_holiday"].iloc[0])
        slot_results = []
        for slot in range(cfg.TIME_STEPS_PER_DAY):
            timestamp = day + pd.Timedelta(minutes=15 * slot)
            prior = slot_prior(priors[day_type], priors[-1], slot, region_ids)
            arrays = topk_arrays(prior, region_ids)
            part = pd.DataFrame({"t": timestamp, "region_id": region_ids})
            part["pi_top10_mass"] = arrays["top_mass"]
            part["pi_other_mass"] = arrays["other_mass"]
            part["pi_same_region"] = arrays["same_mass"]
            part["pi_history_available"] = arrays["available"]
            for rank in range(cfg.TOP_K_DESTINATIONS):
                destination = arrays["destination"][:, rank]
                part[f"dest_{rank + 1}_region_id"] = np.where(
                    destination >= 0, region_ids[np.maximum(destination, 0)], ""
                )
                part[f"dest_{rank + 1}_pi"] = arrays["probability"][:, rank]
                part[f"dest_{rank + 1}_tau_q0.10"] = arrays["tau10"][:, rank]
                part[f"dest_{rank + 1}_tau_q0.50"] = arrays["tau50"][:, rank]
                part[f"dest_{rank + 1}_tau_q0.90"] = arrays["tau90"][:, rank]

            previous_time = timestamp - pd.Timedelta(minutes=15)
            previous_r = region_by_time.loc[[previous_time]].reset_index()
            try:
                previous_o = od_by_time.loc[[previous_time]].reset_index()
            except KeyError:
                previous_o = pd.DataFrame(columns=["i", "j", "D_ijt_observed"])
            related_dest, related_origin, related_pressure = relation_features(
                prior, previous_r, previous_o, region_ids
            )
            part["related_dest_demand_lag1"] = related_dest
            part["related_origin_inflow_lag1"] = related_origin
            part["related_pressure_lag1"] = related_pressure
            slot_results.append(part)

        opportunity = pd.concat(slot_results, ignore_index=True)
        result = state.merge(opportunity, on=["region_id", "t"], how="left", validate="one_to_one")
        output = cfg.HISTORICAL_FEATURE_DIR / f"vit_history_{day:%Y-%m-%d}.parquet"
        result.to_parquet(output, index=False, compression="snappy")
        print(f"已构建：{output.name}")


if __name__ == "__main__":
    main()
