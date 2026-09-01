# -*- coding: utf-8 -*-

import pandas as pd
from pathlib import Path


file_path = Path(
    "/4 司机未来价值预测模型/1 Vit输入特征与标签构造/1 司机历史收益特征构造（结果）.parquet"
)


df = pd.read_parquet(file_path)


feature_columns = [
    "driver_profit_per_min_i_lag1",
    "driver_profit_per_order_i_lag1",

    "driver_profit_per_min_i_roll4",
    "driver_profit_per_order_i_roll4",

    "driver_profit_per_min_i_roll8",
    "driver_profit_per_order_i_roll8",

    "driver_profit_per_min_i_same_slot_last_day",
    "driver_profit_per_order_i_same_slot_last_day",

    "driver_profit_per_min_i_same_slot_last_week",
    "driver_profit_per_order_i_same_slot_last_week",
]


result = []

for col in feature_columns:

    series = df[col]

    result.append(
        {
            "feature": col,
            "mean": series.mean(),
            "max": series.max(),
            "min": series.min(),
            "std": series.std(),
            "negative_ratio": (
                (series < 0).sum()
                /
                series.notna().sum()
            ),
            "missing_ratio": series.isna().mean(),
        }
    )


summary = pd.DataFrame(result)


pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.float_format",
    lambda x: f"{x:.6f}"
)


print(summary.to_string(index=False))