from pathlib import Path

import pandas as pd
import numpy as np

code_dir = Path(__file__).resolve().parent
file_path = code_dir / "output" / "prediction" / "dit_valid_predictions.parquet"

df = pd.read_parquet(file_path)

label = df["D_it_label"]
q_cols = [c for c in df.columns if c.startswith("q_")]
pred = df[q_cols].to_numpy()

print("样本数：", len(df))
print("\n真实 Dit 分布：")
print(label.describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]))

print("\n缺失值数量：", df[q_cols].isna().sum().sum())
print("预测负值比例：", (pred < 0).mean())

q50_col = min(q_cols, key=lambda x: abs(float(x.split("_")[1]) - 0.5))
q50 = df[q50_col]

print("\nq0.50：")
print(q50.describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]))

print("\nq0.50 MAE：", np.mean(np.abs(label - q50)))
print("q0.50 RMSE：", np.sqrt(np.mean((label - q50) ** 2)))

crossing = pred[:, :-1] > pred[:, 1:]
print("\n分位数相邻对交叉率：", crossing.mean())
print("存在交叉的样本比例：", crossing.any(axis=1).mean())
print("平均每个样本交叉对数量：", crossing.sum(axis=1).mean())
print("最大单样本交叉对数量：", crossing.sum(axis=1).max())