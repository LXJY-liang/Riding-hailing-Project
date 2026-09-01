# -*- coding: utf-8 -*-
"""
随机抽取司机未来价值基础特征样本
"""

import pandas as pd

# 文件路径
file_path = (
    "/Users/liangxue/Documents/8、Python文件-网约车数据/"
    "Riding hailing_Project/4 司机未来价值预测模型/1 Vit输入特征与标签构造/"
    "2 Vit基础特征构造（结果）.parquet"
)

# 读取数据并随机抽取5个样本
df = pd.read_parquet(file_path)

samples = df.sample(5, random_state=42)

# 终端输出
print(samples.to_string(index=False))