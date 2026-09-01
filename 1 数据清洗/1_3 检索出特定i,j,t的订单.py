import pandas as pd

CSV_PATH = "1 经纬度恢复及删除多余变量_保留订单.csv"

TARGET_PICKUP = "17031081500"
TARGET_DROPOFF = "17031081403"
TARGET_TIME = pd.Timestamp("2024-06-30 23:30:00")

results = []

for chunk in pd.read_csv(
    CSV_PATH,
    dtype={
        "Pickup Census Tract": "string",
        "Dropoff Census Tract": "string"
    },
    chunksize=200000
):
    chunk["Trip Start Timestamp"] = pd.to_datetime(
        chunk["Trip Start Timestamp"],
        errors="coerce"
    )

    matched = chunk[
        (chunk["Pickup Census Tract"] == TARGET_PICKUP)
        & (chunk["Dropoff Census Tract"] == TARGET_DROPOFF)
        & (chunk["Trip Start Timestamp"] == TARGET_TIME)
    ]

    if not matched.empty:
        results.append(matched)

if results:
    result = pd.concat(results, ignore_index=True)
    print(result.to_string(index=False))
else:
    print("未找到符合条件的样本")