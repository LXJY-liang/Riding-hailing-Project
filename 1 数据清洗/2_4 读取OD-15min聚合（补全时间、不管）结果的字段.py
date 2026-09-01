import pyarrow.parquet as pq

FILE_PATH = (
    "/Users/liangxue/Documents/8、Python文件-网约车数据/"
    "Riding hailing_Project/1 数据清洗/"
    "2_3 OD-15min聚合（补全时间、不管没出现的OD）结果.parquet"
)

# 指定读取第几行（第1行、第2行……）
target_row = 278873

# Python内部从0开始计数
target_index = target_row - 1

parquet_file = pq.ParquetFile(FILE_PATH)

# 找到目标行所在的 row group
cumulative_rows = 0

for row_group_index in range(parquet_file.num_row_groups):
    row_group_rows = parquet_file.metadata.row_group(row_group_index).num_rows

    if target_index < cumulative_rows + row_group_rows:
        # 目标行在当前 row group 中的位置
        row_index_in_group = target_index - cumulative_rows

        target_data = (
            parquet_file
            .read_row_group(row_group_index)
            .slice(row_index_in_group, 1)
            .to_pandas()
            .iloc[0]
        )

        break

    cumulative_rows += row_group_rows

# 输出第100行所有字段
for field_name, field_value in target_data.items():
    print(f"{field_name}: {field_value}")