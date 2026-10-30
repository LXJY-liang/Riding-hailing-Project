# -*- coding: utf-8 -*-
"""检查Vit全量训练样本、新D/S分布接入和中文Prompt。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("vit_pretrain_config", HERE / "1 Vit预训练配置.py")
cfg = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(cfg)


def main() -> None:
    if cfg.SOURCE_VIT_LABELS_TEMP.exists():
        raise RuntimeError("标签.tmp仍存在，1000条轨迹构建尚未完成。")
    diagnostics = json.loads(cfg.SOURCE_VIT_LABELS_DIAGNOSTICS.read_text(encoding="utf-8"))
    if diagnostics.get("monte_carlo_samples") != 1000:
        raise ValueError("标签诊断中monte_carlo_samples不是1000。")
    if diagnostics.get("negative_or_zero_net_order_policy") == "reject_and_wait_one_slot":
        raise ValueError("Vit标签使用了非正收益订单拒绝口径。")
    expected_per_day = cfg.TIME_STEPS_PER_DAY * cfg.EXPECTED_REGION_COUNT
    total = 0
    for day in cfg.days("train"):
        feature_file = cfg.INPUT_FEATURE_DIR / f"vit_input_features_{day:%Y-%m-%d}.parquet"
        sample_file = cfg.TRAIN_PART_DIR / f"vit_train_{day:%Y-%m-%d}.parquet"
        state_file = cfg.STATE_TEXT_DIR / f"vit_state_{day:%Y-%m-%d}.parquet"
        for path in (feature_file, sample_file, state_file):
            if not path.exists():
                raise FileNotFoundError(path)
            if pq.ParquetFile(path).metadata.num_rows != expected_per_day:
                raise ValueError(f"{path.name}行数不是{expected_per_day:,}。")
        frame = pd.read_parquet(
            sample_file,
            columns=["t", "text", "log_supply", *cfg.DEMAND_FEATURE_COLUMNS, *cfg.QUANTILE_COLUMNS],
        )
        values = frame[["log_supply", *cfg.DEMAND_FEATURE_COLUMNS, *cfg.QUANTILE_COLUMNS]].to_numpy()
        if not np.isfinite(values).all():
            raise FloatingPointError(f"{day.date()}数值输入或标签含NaN/inf。")
        if (np.diff(frame[cfg.QUANTILE_COLUMNS].to_numpy(), axis=1) < -1e-5).any():
            raise ValueError(f"{day.date()} Vit标签存在分位数交叉。")
        prompt = frame["text"].iloc[len(frame) // 2]
        for phrase in (
            "当前区域当前时段的需求分布",
            "可服务司机供给代理分布",
            "每单净收益依次为",
            "历史主要目的地与行程时间",
            "邻近区域机会",
            "并保证分位数曲线单调不下降",
        ):
            if phrase not in prompt:
                raise ValueError(f"{day.date()} Prompt缺少中文特征语义：{phrase}")
        total += len(frame)

    expected_total = expected_per_day * len(cfg.days("train"))
    if total != expected_total:
        raise ValueError(f"全量样本应为{expected_total:,}行，实际{total:,}行。")

    predict_features = pd.read_parquet(
        cfg.INPUT_FEATURE_DIR / "vit_input_features_2024-06-11.parquet",
        columns=["demand_distribution_source", "supply_distribution_source"],
    )
    if not predict_features["demand_distribution_source"].str.contains("Dit预测").all():
        raise ValueError("6月11日未全部使用新Dit预测分布。")
    if not predict_features["supply_distribution_source"].str.contains("基于U分布").all():
        raise ValueError("6月11日未全部使用新Sit预测分布。")
    print(f"Vit训练数据检查通过：{total:,}行，6月1—24日全量。")
    print("6月11—24日的D_it/S_it已接入新随机变量分布。")


if __name__ == "__main__":
    main()
