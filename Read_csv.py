import pandas as pd

# 关闭列数限制（不省略）
pd.set_option('display.max_columns', None)

# 关闭列宽限制（不截断）
pd.set_option('display.max_colwidth', None)

# 关闭显示宽度限制
pd.set_option('display.width', None)

file_path = "//Users/liangxue/Documents/8、Python文件-网约车数据/Transportation_Network_Providers_-_Trips_(2023-2024)_20260307.csv"

df = pd.read_csv(file_path, nrows=20)

print(df)