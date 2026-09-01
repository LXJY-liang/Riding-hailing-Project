# -*- coding: utf-8 -*-
"""任务 9：深入分析验证集 SAC 定价、补贴、约束、历史支持和训练稳定性。

本脚本同时报告三种口径：
1. OD 等权：每个有订单的 (i,j,t) 权重相同；
2. 历史订单量加权：以 D_ijt_observed 为权重；
3. 模型成交量加权：以 M_ijt 为权重。

输出 JSON、CSV 和 PNG，便于后续论文制表和复核。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-sac-analysis")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODULE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = MODULE_DIR / "output"
POLICY_FILE = OUTPUT_DIR / "valid_policy_sac_only_gamma099.parquet"
STATE_FILE = OUTPUT_DIR / "offline_sac_state.parquet"
HISTORY_FILE = OUTPUT_DIR / "log" / "training_history_gamma099.csv"
ANALYSIS_DIR = OUTPUT_DIR / "analysis_gamma099"

COMMISSION_RATE = 0.25
PASSENGER_TARGET = 0.95
DRIVER_TARGET = 0.95


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    value_array = values.to_numpy(dtype=np.float64)
    weight_array = weights.to_numpy(dtype=np.float64)
    valid = np.isfinite(value_array) & np.isfinite(weight_array) & (weight_array > 0)
    if not valid.any():
        return float("nan")
    return float(np.average(value_array[valid], weights=weight_array[valid]))


def weighted_rate(mask: pd.Series, weights: pd.Series) -> float:
    return weighted_mean(mask.astype(np.float64), weights)


def quantiles(series: pd.Series) -> dict[str, float]:
    values = series.quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    return {f"q{int(level * 100):02d}": float(value) for level, value in values.items()}


def aggregate_group(group: pd.DataFrame) -> pd.Series:
    historical_orders = group["D_ijt_observed"]
    model_orders = group["M_ijt"]
    proxy_outside_rate = (
        weighted_rate(
            group["proxy_support_position"] != "inside_proxy_q10_q90", historical_orders
        )
        if "proxy_support_position" in group.columns
        else float("nan")
    )
    return pd.Series(
        {
            "od_rows": len(group),
            "historical_orders": float(historical_orders.sum()),
            "model_orders": float(model_orders.sum()),
            "completion_ratio": float(model_orders.sum() / max(historical_orders.sum(), 1e-8)),
            "historical_price_D_weighted": weighted_mean(group["p_ref_ijt"], historical_orders),
            "sac_price_D_weighted": weighted_mean(group["p_ijt"], historical_orders),
            "price_change_percent_D_weighted": weighted_mean(
                group["price_change_percent"], historical_orders
            ),
            "subsidy_D_weighted": weighted_mean(group["b_ijt"], historical_orders),
            "subsidized_od_share": float((group["b_ijt"] > 1e-8).mean()),
            "passenger_acceptance_D_weighted": weighted_mean(group["A_ijt"], historical_orders),
            "passenger_violation_rate": float((group["g_passenger"] > 0).mean()),
            "passenger_violation_rate_D_weighted": weighted_rate(
                group["g_passenger"] > 0, historical_orders
            ),
            "driver_final_acceptance_D_weighted": weighted_mean(
                group["Bbar_ijt"], historical_orders
            ),
            "driver_violation_rate": float((group["g_driver"] > 0).mean()),
            "driver_violation_rate_D_weighted": weighted_rate(
                group["g_driver"] > 0, historical_orders
            ),
            "outside_support_rate": float(group["outside_q10_q90_support"].mean()),
            "outside_support_rate_D_weighted": weighted_mean(
                group["outside_q10_q90_support"], historical_orders
            ),
            "outside_proxy_support_rate_D_weighted": proxy_outside_rate,
            "platform_profit": float(group["platform_profit"].sum()),
        }
    )


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    policy = pd.read_parquet(POLICY_FILE)
    state = pd.read_parquet(
        STATE_FILE,
        columns=[
            "state_id", "group_id", "p_q10_ijt", "p_q90_ijt",
            "pe_q10_ijt", "pe_q90_ijt",
        ],
        filters=[("split", "==", "valid")],
    )
    policy = policy.merge(
        state.drop(columns="group_id"), on="state_id", how="left", validate="one_to_one"
    )
    if policy[["p_q10_ijt", "p_q90_ijt", "pe_q10_ijt", "pe_q90_ijt"]].isna().any().any():
        raise ValueError("部分验证 OD 无法匹配历史价格支持区间。")

    policy["support_position"] = np.select(
        [policy["p_ijt"] < policy["p_q10_ijt"], policy["p_ijt"] > policy["p_q90_ijt"]],
        ["below_q10", "above_q90"],
        default="inside_q10_q90",
    )
    policy["distance_to_support_ratio"] = np.select(
        [policy["p_ijt"] < policy["p_q10_ijt"], policy["p_ijt"] > policy["p_q90_ijt"]],
        [
            (policy["p_ijt"] - policy["p_q10_ijt"]) / policy["p_ref_ijt"],
            (policy["p_ijt"] - policy["p_q90_ijt"]) / policy["p_ref_ijt"],
        ],
        default=0.0,
    )
    # 原始价格分位数在单订单 OD 上会退化为 q10=q90。代理分位数 pe_q10/pe_q90
    # 已使用训练期可识别的价格离散度下限进行扩宽，更适合作为稳健性口径。
    policy["raw_support_degenerate"] = (
        (policy["p_q90_ijt"] - policy["p_q10_ijt"]).abs() <= 1e-8
    )
    policy["proxy_support_position"] = np.select(
        [policy["p_ijt"] < policy["pe_q10_ijt"], policy["p_ijt"] > policy["pe_q90_ijt"]],
        ["below_proxy_q10", "above_proxy_q90"],
        default="inside_proxy_q10_q90",
    )
    policy["hour"] = policy["t"].dt.hour
    policy["date"] = policy["t"].dt.date.astype(str)
    policy["gross_commission"] = COMMISSION_RATE * policy["p_ijt"] * policy["M_ijt"]
    policy["subsidy_spend"] = policy["b_ijt"] * policy["M_ijt"]
    policy["margin_per_completed_order"] = COMMISSION_RATE * policy["p_ijt"] - policy["b_ijt"]
    policy["historical_commission_proxy"] = (
        COMMISSION_RATE * policy["p_ref_ijt"] * policy["D_ijt_observed"]
    )

    d_weight = policy["D_ijt_observed"]
    m_weight = policy["M_ijt"]
    subsidized = policy.loc[policy["b_ijt"] > 1e-8]
    passenger_violated = policy.loc[policy["g_passenger"] > 0]
    driver_violated = policy.loc[policy["g_driver"] > 0]
    positive_volume = policy.loc[policy["M_ijt"] > 1e-8]

    overall = {
        "data": {
            "od_rows": int(len(policy)),
            "groups": int(policy["group_id"].nunique()),
            "origins": int(policy["i"].nunique()),
            "destinations": int(policy["j"].nunique()),
            "historical_orders": float(d_weight.sum()),
            "model_completed_orders": float(m_weight.sum()),
            "completion_ratio": float(m_weight.sum() / d_weight.sum()),
        },
        "price": {
            "historical_price_od_mean": float(policy["p_ref_ijt"].mean()),
            "sac_price_od_mean": float(policy["p_ijt"].mean()),
            "historical_price_D_weighted": weighted_mean(policy["p_ref_ijt"], d_weight),
            "sac_price_D_weighted": weighted_mean(policy["p_ijt"], d_weight),
            "sac_price_M_weighted": weighted_mean(policy["p_ijt"], m_weight),
            "change_percent_od_mean": float(policy["price_change_percent"].mean()),
            "change_percent_D_weighted": weighted_mean(policy["price_change_percent"], d_weight),
            "change_percent_M_weighted": weighted_mean(policy["price_change_percent"], m_weight),
            "increase_share_od": float((policy["price_difference"] > 1e-6).mean()),
            "decrease_share_od": float((policy["price_difference"] < -1e-6).mean()),
            "ratio_near_lower_bound_share": float((policy["price_ratio"] <= 0.51).mean()),
            "ratio_near_upper_bound_share": float((policy["price_ratio"] >= 1.49).mean()),
            "price_change_percent_quantiles": quantiles(policy["price_change_percent"]),
            "price_ratio_quantiles": quantiles(policy["price_ratio"]),
        },
        "subsidy": {
            "subsidized_od_share": float((policy["b_ijt"] > 1e-8).mean()),
            "average_all_od": float(policy["b_ijt"].mean()),
            "average_D_weighted": weighted_mean(policy["b_ijt"], d_weight),
            "average_M_weighted": weighted_mean(policy["b_ijt"], m_weight),
            "conditional_average_positive": float(subsidized["b_ijt"].mean()),
            "conditional_median_positive": float(subsidized["b_ijt"].median()),
            "conditional_quantiles_positive": quantiles(subsidized["b_ijt"]),
            "total_spend": float(policy["subsidy_spend"].sum()),
            "positive_at_or_near_max_share": float(
                (subsidized["b_ijt"] >= subsidized["b_max_ijt"] - 1e-6).mean()
            ),
        },
        "acceptance_and_constraints": {
            "passenger_acceptance_od_mean": float(policy["A_ijt"].mean()),
            "passenger_acceptance_D_weighted": weighted_mean(policy["A_ijt"], d_weight),
            "passenger_violation_rate_od": float((policy["g_passenger"] > 0).mean()),
            "passenger_violation_rate_D_weighted": weighted_rate(
                policy["g_passenger"] > 0, d_weight
            ),
            "passenger_violation_mean_all": float(policy["g_passenger"].mean()),
            "passenger_violation_mean_conditional": float(passenger_violated["g_passenger"].mean()),
            "driver_final_acceptance_od_mean": float(policy["Bbar_ijt"].mean()),
            "driver_final_acceptance_D_weighted": weighted_mean(policy["Bbar_ijt"], d_weight),
            "driver_violation_rate_od": float((policy["g_driver"] > 0).mean()),
            "driver_violation_rate_D_weighted": weighted_rate(policy["g_driver"] > 0, d_weight),
            "driver_violation_mean_conditional": float(driver_violated["g_driver"].mean()),
            "hard_infeasible_rate_D_weighted": weighted_rate(policy["hard_infeasible"] > 0, d_weight),
        },
        "support": {
            "outside_share_od": float((policy["support_position"] != "inside_q10_q90").mean()),
            "outside_share_D_weighted": weighted_rate(
                policy["support_position"] != "inside_q10_q90", d_weight
            ),
            "below_q10_share_od": float((policy["support_position"] == "below_q10").mean()),
            "above_q90_share_od": float((policy["support_position"] == "above_q90").mean()),
            "mean_signed_distance_outside_ratio": float(
                policy.loc[policy["support_position"] != "inside_q10_q90", "distance_to_support_ratio"].mean()
            ),
            "raw_support_degenerate_share_od": float(policy["raw_support_degenerate"].mean()),
            "raw_support_degenerate_share_D_weighted": weighted_rate(
                policy["raw_support_degenerate"], d_weight
            ),
            "outside_proxy_share_od": float(
                (policy["proxy_support_position"] != "inside_proxy_q10_q90").mean()
            ),
            "outside_proxy_share_D_weighted": weighted_rate(
                policy["proxy_support_position"] != "inside_proxy_q10_q90", d_weight
            ),
            "below_proxy_q10_share_od": float(
                (policy["proxy_support_position"] == "below_proxy_q10").mean()
            ),
            "above_proxy_q90_share_od": float(
                (policy["proxy_support_position"] == "above_proxy_q90").mean()
            ),
        },
        "profit": {
            "gross_commission": float(policy["gross_commission"].sum()),
            "subsidy_spend": float(policy["subsidy_spend"].sum()),
            "platform_profit": float(policy["platform_profit"].sum()),
            "profit_per_completed_order": float(policy["platform_profit"].sum() / m_weight.sum()),
            "negative_margin_od_share_among_positive_volume": float(
                (positive_volume["margin_per_completed_order"] < 0).mean()
            ),
            "negative_profit_total": float(
                policy.loc[policy["platform_profit"] < 0, "platform_profit"].sum()
            ),
            "descriptive_historical_commission_proxy": float(
                policy["historical_commission_proxy"].sum()
            ),
        },
        "supply": {
            "binding_rate": float(
                ((policy["supply_scale"] < 1 - 1e-8) & (policy["potential_demand_ijt"] > 0)).mean()
            ),
            "zero_model_volume_share": float((policy["M_ijt"] <= 1e-8).mean()),
        },
        "correlations_spearman": {
            "price_change_vs_passenger_acceptance": float(
                policy["price_change_percent"].corr(policy["A_ijt"], method="spearman")
            ),
            "subsidy_vs_driver_final_acceptance": float(
                policy["b_ijt"].corr(policy["Bbar_ijt"], method="spearman")
            ),
            "price_change_vs_profit": float(
                policy["price_change_percent"].corr(policy["platform_profit"], method="spearman")
            ),
        },
    }

    price_bins = pd.cut(
        policy["price_change_percent"],
        bins=[-np.inf, -25, -15, -5, 5, 15, 25, np.inf],
        labels=["<-25%", "-25~-15%", "-15~-5%", "-5~5%", "5~15%", "15~25%", ">=25%"],
        ordered=True,
    )
    policy["price_change_bin"] = price_bins
    price_bin_summary = policy.groupby("price_change_bin", observed=True).apply(
        aggregate_group, include_groups=False
    ).reset_index()
    hourly = policy.groupby("hour", sort=True).apply(aggregate_group, include_groups=False).reset_index()
    daily = policy.groupby("date", sort=True).apply(aggregate_group, include_groups=False).reset_index()
    origins = policy.groupby("i", sort=False).apply(aggregate_group, include_groups=False).reset_index()
    origins["price_cut_rank"] = origins["price_change_percent_D_weighted"].rank(method="min")
    origins["passenger_violation_rank"] = origins["passenger_violation_rate"].rank(
        method="min", ascending=False
    )

    support_summary = (
        policy.groupby("support_position", observed=True)
        .apply(aggregate_group, include_groups=False)
        .reset_index()
    )
    proxy_support_summary = (
        policy.groupby("proxy_support_position", observed=True)
        .apply(aggregate_group, include_groups=False)
        .reset_index()
    )

    history = pd.read_csv(HISTORY_FILE)
    validation_columns = [
        "collection_step", "validation_profit", "validation_adjusted_reward",
        "validation_g_passenger", "validation_g_driver", "lambda_passenger", "lambda_driver",
    ]
    validation_history = history.loc[
        history["validation_adjusted_reward"].notna(), validation_columns
    ].copy()

    # 训练时每次只检查验证集前 2,000 个 (i,t) 组。由于拉格朗日乘子不断变化，
    # 原始 adjusted reward 不能直接跨检查点严格比较；补充用最终乘子统一重算的口径。
    validation_groups = state["group_id"].drop_duplicates().iloc[:2_000]
    validation_row_count = int(state["group_id"].isin(validation_groups).sum())
    final_lambda_passenger = float(history["lambda_passenger"].dropna().iloc[-1])
    final_lambda_driver = float(history["lambda_driver"].dropna().iloc[-1])
    validation_history["adjusted_at_final_lambdas"] = (
        validation_history["validation_profit"]
        - validation_row_count
        * (
            final_lambda_passenger * validation_history["validation_g_passenger"]
            + final_lambda_driver * validation_history["validation_g_driver"]
        )
    )
    overall["training"] = {
        "total_collection_steps": int(history["collection_step"].max()),
        "saved_best_step": int(
            validation_history.loc[
                validation_history["validation_adjusted_reward"].idxmax(), "collection_step"
            ]
        ),
        "validation_groups_per_checkpoint": 2_000,
        "validation_od_rows_per_checkpoint": validation_row_count,
        "critic_loss_median_first_1000": float(history["critic_loss"].iloc[:1_000].median()),
        "critic_loss_median_last_1000": float(history["critic_loss"].iloc[-1_000:].median()),
        "entropy_alpha_final": float(history["entropy_alpha"].dropna().iloc[-1]),
        "lambda_passenger_saved_checkpoint": float(
            validation_history.loc[
                validation_history["validation_adjusted_reward"].idxmax(), "lambda_passenger"
            ]
        ),
        "lambda_driver_saved_checkpoint": float(
            validation_history.loc[
                validation_history["validation_adjusted_reward"].idxmax(), "lambda_driver"
            ]
        ),
        "lambda_passenger_final": final_lambda_passenger,
        "lambda_driver_final": final_lambda_driver,
    }

    (ANALYSIS_DIR / "overall_analysis.json").write_text(
        json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    hourly.to_csv(ANALYSIS_DIR / "hourly_summary.csv", index=False)
    daily.to_csv(ANALYSIS_DIR / "daily_summary.csv", index=False)
    origins.to_csv(ANALYSIS_DIR / "origin_summary.csv", index=False)
    price_bin_summary.to_csv(ANALYSIS_DIR / "price_change_bin_summary.csv", index=False)
    support_summary.to_csv(ANALYSIS_DIR / "support_summary.csv", index=False)
    proxy_support_summary.to_csv(ANALYSIS_DIR / "proxy_support_summary.csv", index=False)
    validation_history.to_csv(ANALYSIS_DIR / "validation_history.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].hist(policy["price_change_percent"], bins=60, color="#3b82f6", alpha=0.85)
    axes[0, 0].axvline(0, color="black", linewidth=1)
    axes[0, 0].set(title="SAC price change distribution", xlabel="Price change (%)", ylabel="OD rows")

    axes[0, 1].hist(subsidized["b_ijt"], bins=50, color="#f59e0b", alpha=0.85)
    axes[0, 1].set(title="Positive subsidy distribution", xlabel="Subsidy ($)", ylabel="Subsidized OD rows")

    axes[1, 0].plot(hourly["hour"], hourly["price_change_percent_D_weighted"], marker="o")
    axes[1, 0].axhline(0, color="black", linewidth=1)
    axes[1, 0].set(title="D-weighted price change by hour", xlabel="Hour", ylabel="Price change (%)")
    axes[1, 0].set_xticks(range(0, 24, 2))

    support_counts = policy["support_position"].value_counts().reindex(
        ["below_q10", "inside_q10_q90", "above_q90"], fill_value=0
    )
    axes[1, 1].bar(support_counts.index, support_counts.values, color=["#ef4444", "#10b981", "#8b5cf6"])
    axes[1, 1].set(title="Price position versus historical support", ylabel="OD rows")
    axes[1, 1].tick_params(axis="x", rotation=15)

    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "policy_summary.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    axes[0].plot(history["collection_step"], history["critic_loss"], alpha=0.25, linewidth=0.5)
    rolling_critic = history["critic_loss"].rolling(300, min_periods=30).mean()
    axes[0].plot(history["collection_step"], rolling_critic, linewidth=2)
    axes[0].set(title="Critic loss (raw and rolling mean)", ylabel="Loss")

    axes[1].plot(validation_history["collection_step"], validation_history["validation_profit"], marker="o", label="Profit")
    axes[1].plot(
        validation_history["collection_step"],
        validation_history["validation_adjusted_reward"],
        marker="o",
        label="Adjusted reward",
    )
    axes[1].set(title="Validation checkpoints", xlabel="Collection step", ylabel="Value")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "training_summary.png", dpi=180)
    plt.close(fig)

    print(json.dumps(overall, ensure_ascii=False, indent=2))
    print(f"分析结果目录：{ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
