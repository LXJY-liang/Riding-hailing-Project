# -*- coding: utf-8 -*-

import random
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq


# ============================================================
# 1. 文件路径
# ============================================================

TAU_FEATURE_FILE = "/Users/liangxue/Documents/8、Python文件-网约车数据/Riding hailing_Project/2 模型特征构建/4 tauijt特征构造（结果）.parquet"


# ============================================================
# 2. 指定要查询的 OD-15min 样本
# ============================================================

TARGET_I = "17031839100"
TARGET_J = "17031833000"
TARGET_T = "2024-06-06 18:15:00"


# ============================================================
# 3. 设置终端显示格式
# ============================================================

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 300)
pd.set_option("display.max_colwidth", 120)


# ============================================================
# 4. 输出指定 OD-15min 样本
# ============================================================

dataset = ds.dataset(TAU_FEATURE_FILE, format="parquet")

schema = dataset.schema
t_type = schema.field("t").type

if "timestamp" in str(t_type):
    target_t_value = pd.Timestamp(TARGET_T).to_pydatetime()
else:
    target_t_value = TARGET_T

condition = (
    (ds.field("i") == TARGET_I)
    & (ds.field("j") == TARGET_J)
    & (ds.field("t") == target_t_value)
)

target_table = dataset.to_table(filter=condition)
target_sample = target_table.to_pandas()

print("=" * 120)
print("指定 OD-15min 样本")
print("=" * 120)

if len(target_sample) == 0:
    print("没有找到对应样本")
else:
    print(target_sample.to_string(index=False))


# ============================================================
# 5. 随机输出10个样本
#    为避免读取整个大文件，只随机读取一个 row group，再从中抽样
# ============================================================

pf = pq.ParquetFile(TAU_FEATURE_FILE)

row_group_idx = random.randint(0, pf.num_row_groups - 1)

random_df = pf.read_row_group(row_group_idx).to_pandas()

if len(random_df) >= 10:
    random_sample = random_df.sample(n=10, random_state=42)
else:
    random_sample = random_df.copy()

print("\n" + "=" * 120)
print("随机输出10个样本")
print("=" * 120)
print(random_sample.to_string(index=False))