# -*- coding: utf-8 -*-
"""导出并诊断 q30 < 0.30*q50 的异常 p^e_ijt 代理分布。

本脚本只读取已构建的 p^e_ijt 样本和分位数，不修改原始订单、
不重新构建 KDE，也不修改 SAC 输入。

导出内容：
1. 196 条异常目标的基本信息、101维分位数、带宽、混合权重和诊断标记；
2. 每个目标实际进入 KDE 的每一个历史价格、匹配层级、里程和最终权重；
3. 按目标和匹配层级统计的价格分布；
4. 每个目标一行，分别列出第1至第4层实际选入的全部价格；
5. 异常成因汇总 JSON。

主要诊断规则：
- KDE 下尾泄漏：q30 低于原始构成价格最小值的50%；
- 两成分分离：同OD与相似OD价格中位数的比值不小于2；
- 带宽过大：任一KDE成分的带宽不小于该成分价格中位数；
- 样本过少：总样本数小于N_MIN，或同OD样本数不超过5条。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds


CODE_DIR = Path(__file__).resolve().parent


def load_config():
    path = CODE_DIR / "1 p^eijt分布构建配置.py"
    spec = importlib.util.spec_from_file_location("pe_extreme_audit_config", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


cfg = load_config()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="导出并诊断 q30 < 0.30*q50 的 p^e_ijt 异常分布。"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖本脚本管理的旧诊断目录。",
    )
    return parser.parse_args()


def split_name(values: pd.Series) -> pd.Series:
    time = pd.to_datetime(values)
    result = np.full(len(time), "before_train", dtype=object)
    result[(time >= "2024-06-11") & (time < "2024-06-25")] = "train"
    result[(time >= "2024-06-25") & (time < "2024-06-28")] = "valid"
    result[(time >= "2024-06-28") & (time < "2024-07-01")] = "test"
    return pd.Series(result, index=values.index, dtype="string")


def filtered_rows(path: Path, target_ids: np.ndarray) -> pd.DataFrame:
    dataset = ds.dataset(path, format="parquet")
    expression = ds.field("target_id").isin(pa.array(target_ids, type=pa.int64()))
    result = dataset.to_table(filter=expression).to_pandas()
    if result["target_id"].nunique() != len(target_ids):
        raise ValueError(
            f"{path} 只返回 {result['target_id'].nunique()} 个目标，"
            f"预期 {len(target_ids)} 个。"
        )
    return result


def safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0:
        return np.nan
    return float(numerator / denominator)


def component_statistics(prices: np.ndarray) -> dict[str, float]:
    if len(prices) == 0:
        return {
            "count": 0,
            "min": np.nan,
            "q10": np.nan,
            "q25": np.nan,
            "median": np.nan,
            "q75": np.nan,
            "q90": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "std": np.nan,
        }
    quantiles = np.quantile(prices, [0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        "count": int(len(prices)),
        "min": float(np.min(prices)),
        "q10": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q75": float(quantiles[3]),
        "q90": float(quantiles[4]),
        "max": float(np.max(prices)),
        "mean": float(np.mean(prices)),
        "std": float(np.std(prices, ddof=1)) if len(prices) > 1 else 0.0,
    }


def number_list_json(values: np.ndarray) -> str:
    """将数值列表保存成适合CSV单元格直接阅读的JSON字符串。

    价格和里程保留6位小数；空层明确记录为 []，不留空值。
    """
    return json.dumps(
        [round(float(value), 6) for value in values],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def main() -> None:
    args = parse_args()
    output_dir = cfg.DIAGNOSTICS_DIR / "peijt_extreme_q30_q50_audit"
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"诊断目录已存在：{output_dir}；需要重建时加 --overwrite。"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    required = [
        cfg.STRUCTURAL_AUDIT_DIR,
        cfg.SAMPLE_DATASET,
        cfg.PE_DISTRIBUTION_DATASET,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少诊断输入：\n" + "\n".join(missing))

    extreme = ds.dataset(cfg.STRUCTURAL_AUDIT_DIR, format="parquet").to_table().to_pandas()
    extreme = extreme.loc[extreme["q50_q30_extreme_flag"].eq(1)].copy()
    extreme = extreme.sort_values("target_id", kind="mergesort").reset_index(drop=True)
    if extreme.empty:
        raise ValueError("没有找到 q30 < 0.30*q50 的异常目标。")
    if not (
        extreme["pe_q0300"] < 0.30 * extreme["pe_q0500"]
    ).all():
        raise AssertionError("结构审计目录含有不符合异常规则的记录。")

    target_ids = extreme["target_id"].to_numpy(np.int64)
    samples = filtered_rows(cfg.SAMPLE_DATASET, target_ids).set_index("target_id")

    layer_weights = np.asarray(
        [0.0, cfg.LAYER_WEIGHTS[1], cfg.LAYER_WEIGHTS[2],
         cfg.LAYER_WEIGHTS[3], cfg.LAYER_WEIGHTS[4]],
        dtype=np.float64,
    )

    summaries: list[dict[str, object]] = []
    details: list[dict[str, object]] = []
    layer_summaries: list[dict[str, object]] = []
    layer_price_lists: list[dict[str, object]] = []

    for row in extreme.itertuples(index=False):
        target_id = int(row.target_id)
        sample = samples.loc[target_id]
        prices = np.asarray(sample["sample_prices"], dtype=np.float64)
        levels = np.asarray(sample["sample_levels"], dtype=np.int8)
        miles = np.asarray(sample["sample_trip_miles"], dtype=np.float64)
        similarities = np.asarray(
            sample["sample_similarity_weights"], dtype=np.float64
        )
        lengths = {len(prices), len(levels), len(miles), len(similarities)}
        if lengths != {int(sample["sample_count"])}:
            raise ValueError(f"target_id={target_id} 的样本列表长度不一致。")

        same_mask = levels <= 3
        similar_mask = levels == 4
        raw_weight = np.where(same_mask, layer_weights[levels], similarities)
        normalized_weight = np.zeros(len(prices), dtype=np.float64)
        final_weight = np.zeros(len(prices), dtype=np.float64)
        if same_mask.any():
            normalized_weight[same_mask] = raw_weight[same_mask] / raw_weight[same_mask].sum()
            final_weight[same_mask] = (
                float(sample["same_od_mixture_weight"])
                * normalized_weight[same_mask]
            )
        if similar_mask.any():
            normalized_weight[similar_mask] = (
                raw_weight[similar_mask] / raw_weight[similar_mask].sum()
            )
            final_weight[similar_mask] = (
                float(sample["similar_od_mixture_weight"])
                * normalized_weight[similar_mask]
            )

        all_stats = component_statistics(prices)
        same_stats = component_statistics(prices[same_mask])
        similar_stats = component_statistics(prices[similar_mask])

        component_median_ratio = np.nan
        if same_stats["count"] and similar_stats["count"]:
            component_median_ratio = max(
                safe_ratio(same_stats["median"], similar_stats["median"]),
                safe_ratio(similar_stats["median"], same_stats["median"]),
            )

        same_bandwidth = float(sample["same_od_bandwidth_usd"])
        similar_bandwidth = float(sample["similar_od_bandwidth_usd"])
        same_bandwidth_ratio = safe_ratio(same_bandwidth, same_stats["median"])
        similar_bandwidth_ratio = safe_ratio(
            similar_bandwidth, similar_stats["median"]
        )
        q30 = float(row.pe_q0300)
        q50 = float(row.pe_q0500)
        lower_tail_leakage = bool(q30 < 0.50 * all_stats["min"])
        separated_components = bool(
            np.isfinite(component_median_ratio) and component_median_ratio >= 2.0
        )
        large_bandwidth = bool(
            (np.isfinite(same_bandwidth_ratio) and same_bandwidth_ratio >= 1.0)
            or (
                np.isfinite(similar_bandwidth_ratio)
                and similar_bandwidth_ratio >= 1.0
            )
        )
        small_total_sample = bool(len(prices) < cfg.N_MIN)
        small_same_sample = bool(0 < int(same_mask.sum()) <= 5)

        reasons: list[str] = []
        if lower_tail_leakage:
            reasons.append("KDE下尾低于原始价格支持")
        if separated_components:
            reasons.append("同OD与相似OD价格中心明显分离")
        if large_bandwidth:
            reasons.append("小样本或离散价格导致KDE带宽过大")
        if small_total_sample:
            reasons.append("总历史样本少于N_MIN")
        if small_same_sample:
            reasons.append("同OD历史样本不超过5条")
        if not reasons:
            reasons.append("需逐条查看价格权重与KDE形状")

        summaries.append(
            {
                "target_id": target_id,
                "i": str(row.i),
                "j": str(row.j),
                "t": pd.Timestamp(row.t),
                "split": split_name(pd.Series([row.t])).iloc[0],
                "D_ijt_observed": float(row.D_ijt_observed),
                "sample_count": int(len(prices)),
                "same_od_sample_count": int(same_mask.sum()),
                "similar_od_sample_count": int(similar_mask.sum()),
                "max_match_level": int(row.max_match_level),
                "same_od_mixture_weight": float(sample["same_od_mixture_weight"]),
                "similar_od_mixture_weight": float(sample["similar_od_mixture_weight"]),
                "same_od_bandwidth_usd": same_bandwidth,
                "similar_od_bandwidth_usd": similar_bandwidth,
                "selected_price_min": all_stats["min"],
                "selected_price_q25": all_stats["q25"],
                "selected_price_median": all_stats["median"],
                "selected_price_q75": all_stats["q75"],
                "selected_price_max": all_stats["max"],
                "same_od_price_median": same_stats["median"],
                "similar_od_price_median": similar_stats["median"],
                "component_median_max_ratio": component_median_ratio,
                "same_bandwidth_to_median_ratio": same_bandwidth_ratio,
                "similar_bandwidth_to_median_ratio": similar_bandwidth_ratio,
                "q30": q30,
                "q50": q50,
                "q30_to_q50_ratio": float(row.q30_to_q50_ratio),
                "lower_tail_leakage_flag": int(lower_tail_leakage),
                "separated_components_flag": int(separated_components),
                "large_bandwidth_flag": int(large_bandwidth),
                "small_total_sample_flag": int(small_total_sample),
                "small_same_od_sample_flag": int(small_same_sample),
                "diagnostic_reasons": "；".join(reasons),
            }
        )

        # 用户审核用的“宽表”：每个目标只占一行，四层价格互不混合。
        # selected_order 版保留构建数据集中的原始顺序；sorted 版方便直观
        # 查看同一层的价格范围和孤立极端值。
        price_list_row: dict[str, object] = {
            "target_id": target_id,
            "i": str(row.i),
            "j": str(row.j),
            "t": pd.Timestamp(row.t),
            "split": split_name(pd.Series([row.t])).iloc[0],
            "q30": q30,
            "q50": q50,
            "sample_count": int(len(prices)),
            "all_prices_selected_order": number_list_json(prices),
        }
        for level in range(1, 5):
            mask = levels == level
            price_list_row[f"layer{level}_count"] = int(mask.sum())
            price_list_row[f"layer{level}_prices_selected_order"] = (
                number_list_json(prices[mask])
            )
            price_list_row[f"layer{level}_prices_sorted"] = number_list_json(
                np.sort(prices[mask])
            )
            price_list_row[f"layer{level}_trip_miles_selected_order"] = (
                number_list_json(miles[mask])
            )
        layer_price_lists.append(price_list_row)

        for number, (price, level, mile, similarity, raw, normalized, final) in enumerate(
            zip(
                prices,
                levels,
                miles,
                similarities,
                raw_weight,
                normalized_weight,
                final_weight,
            ),
            start=1,
        ):
            details.append(
                {
                    "target_id": target_id,
                    "i": str(row.i),
                    "j": str(row.j),
                    "t": pd.Timestamp(row.t),
                    "selected_order": number,
                    "history_price_usd": float(price),
                    "match_level": int(level),
                    "kde_component": "same_od" if int(level) <= 3 else "similar_od",
                    "history_trip_miles": float(mile),
                    "distance_similarity_weight": float(similarity),
                    "component_raw_weight": float(raw),
                    "component_normalized_weight": float(normalized),
                    "final_kde_weight": float(final),
                }
            )

        for level in sorted(np.unique(levels).tolist()):
            mask = levels == level
            stats = component_statistics(prices[mask])
            layer_summaries.append(
                {
                    "target_id": target_id,
                    "i": str(row.i),
                    "j": str(row.j),
                    "t": pd.Timestamp(row.t),
                    "match_level": int(level),
                    **{f"price_{key}": value for key, value in stats.items()},
                    "final_weight_sum": float(final_weight[mask].sum()),
                }
            )

    summary = pd.DataFrame(summaries).sort_values("target_id")
    detail = pd.DataFrame(details).sort_values(
        ["target_id", "match_level", "history_price_usd"], kind="mergesort"
    )
    layer_summary = pd.DataFrame(layer_summaries).sort_values(
        ["target_id", "match_level"], kind="mergesort"
    )
    layer_price_list = pd.DataFrame(layer_price_lists).sort_values(
        "target_id", kind="mergesort"
    )
    quantile_columns = [column for column in extreme.columns if column.startswith("pe_q")]
    quantile_output = extreme[
        ["target_id", "i", "j", "t", "D_ijt_observed", *quantile_columns,
         "q30_to_q50_ratio", "q50_to_q30_ratio", "q50_q30_extreme_flag"]
    ].copy()

    summary_file = output_dir / "1_196条异常目标诊断汇总.csv"
    detail_file = output_dir / "2_196条异常目标构成价格明细.csv"
    layer_file = output_dir / "3_196条异常目标分层价格统计.csv"
    quantile_file = output_dir / "4_196条异常目标101维分位数.csv"
    price_list_file = output_dir / "5_196条异常目标四层具体价格清单.csv"
    report_file = output_dir / "6_异常原因汇总.json"
    summary.to_csv(summary_file, index=False, encoding="utf-8-sig")
    detail.to_csv(detail_file, index=False, encoding="utf-8-sig")
    layer_summary.to_csv(layer_file, index=False, encoding="utf-8-sig")
    quantile_output.to_csv(quantile_file, index=False, encoding="utf-8-sig")
    layer_price_list.to_csv(price_list_file, index=False, encoding="utf-8-sig")

    report = {
        "rule": "q30 < 0.30 * q50",
        "target_count": int(len(summary)),
        "constituent_price_rows": int(len(detail)),
        "split_counts": {
            str(key): int(value)
            for key, value in summary["split"].value_counts().sort_index().items()
        },
        "flag_counts": {
            name: int(summary[name].sum())
            for name in [
                "lower_tail_leakage_flag",
                "separated_components_flag",
                "large_bandwidth_flag",
                "small_total_sample_flag",
                "small_same_od_sample_flag",
            ]
        },
        "sample_count_summary": {
            key: float(value)
            for key, value in summary["sample_count"].describe().items()
        },
        "q30_to_q50_ratio_summary": {
            key: float(value)
            for key, value in summary["q30_to_q50_ratio"].describe().items()
        },
        "output_files": {
            "target_summary": str(summary_file),
            "constituent_prices": str(detail_file),
            "layer_statistics": str(layer_file),
            "quantiles": str(quantile_file),
            "four_layer_price_lists": str(price_list_file),
        },
    }
    report_file.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
