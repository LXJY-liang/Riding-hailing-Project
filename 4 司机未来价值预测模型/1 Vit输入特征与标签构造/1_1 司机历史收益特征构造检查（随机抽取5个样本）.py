# -*- coding: utf-8 -*-

import pandas as pd
from pathlib import Path

file_path = Path(
    "/4 司机未来价值预测模型/1 Vit输入特征与标签构造/1 司机历史收益特征构造（结果）.parquet"
)


df = pd.read_parquet(file_path)


sample = df.sample(
    n=5,
    random_state=42
)


print(sample.to_string(index=False))