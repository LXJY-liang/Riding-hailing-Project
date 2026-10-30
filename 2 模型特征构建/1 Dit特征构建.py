# -*- coding: utf-8 -*-

"""
运行结果

使用“出发地 i 和目的地 j 均属于 801 个区域”的内部 OD 口径。
运行结果会在文件末尾输出，其中 D_it_label 总和应为 5,812,445。
"""


import os
import numpy as np
import pandas as pd


# ============================================================
# 1. 路径设置
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 严格的 801 区域内部 OD 口径区域-15min完整面板
REGION_PANEL_FILE = os.path.join(
    PROJECT_ROOT,
    "1 数据清洗",
    "3_3 区域-15min聚合（补全空值）结果.parquet",
)

# 内部 OD 历史，用于构造只依赖目标日之前历史的区域联系权重
OD_HISTORY_FILE = os.path.join(
    PROJECT_ROOT,
    "1 数据清洗",
    "2_2 OD-15min聚合_属于芝加哥.parquet",
)

# 邻近区域关系表
NEIGHBOR_FILE = os.path.join(
    PROJECT_ROOT,
    "1 数据清洗",
    "4 邻近区域关系表构造结果.parquet",
)

# Dit特征构建结果输出目录
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "2 模型特征构建")

# Dit特征构建结果输出文件
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "1 Dit特征构建（结果）.parquet")

os.makedirs(OUTPUT_DIR, exist_ok=True)

TIME_STEPS_PER_DAY = 96
RELATION_SLOT_RADIUS = 1


def construct_related_region_features(df, region_ids, region_to_idx):
    """构造逐日扩展、严格滞后一格的两项历史OD关联区域特征。"""

    if not os.path.exists(OD_HISTORY_FILE):
        raise FileNotFoundError(f"缺少OD历史文件：{OD_HISTORY_FILE}")

    od = pd.read_parquet(
        OD_HISTORY_FILE,
        columns=[
            "i",
            "j",
            "t",
            "slot_in_day",
            "is_weekend_or_holiday",
            "D_ijt_observed",
        ],
    )
    od["i"] = od["i"].astype(str)
    od["j"] = od["j"].astype(str)
    od["t"] = pd.to_datetime(od["t"], errors="raise")
    od["date"] = od["t"].dt.normalize()
    od["i_idx"] = od["i"].map(region_to_idx)
    od["j_idx"] = od["j"].map(region_to_idx)
    if od[["i_idx", "j_idx"]].isna().any().any():
        raise ValueError("OD历史中存在不属于801区域面板的起点或终点。")
    od["i_idx"] = od["i_idx"].astype(np.int16)
    od["j_idx"] = od["j_idx"].astype(np.int16)
    od["slot_in_day"] = od["slot_in_day"].astype(np.int16)
    od["D_ijt_observed"] = pd.to_numeric(
        od["D_ijt_observed"], errors="raise"
    ).astype(np.float32)
    if (od["D_ijt_observed"] < 0).any():
        raise ValueError("OD历史中出现负订单量。")

    timestamps = pd.DatetimeIndex(sorted(df["t"].unique()))
    n_regions = len(region_ids)
    dates = pd.DatetimeIndex(timestamps.normalize().unique())
    n_times = len(timestamps)
    demand_cube = (
        df["D_origin_it"].to_numpy(dtype=np.float32).reshape(n_times, n_regions)
    )
    related_dest = np.zeros((n_times, n_regions), dtype=np.float32)
    related_origin = np.zeros((n_times, n_regions), dtype=np.float32)

    # 分别累计工作日、周末/节假日，以及不区分日期类型的历史OD流量。
    flow_by_day_type = np.zeros(
        (2, TIME_STEPS_PER_DAY, n_regions, n_regions), dtype=np.float32
    )
    flow_all = np.zeros(
        (TIME_STEPS_PER_DAY, n_regions, n_regions), dtype=np.float32
    )
    od_by_time = od.groupby("t", sort=False, observed=True)
    od_times = set(od_by_time.groups)

    panel_day_type = (
        df.assign(date=df["t"].dt.normalize())
        .groupby("date", sort=True, observed=True)["is_weekend_or_holiday"]
        .first()
    )

    for day_index, day in enumerate(dates):
        day_type = int(panel_day_type.loc[day])
        if day_type not in (0, 1):
            raise ValueError(f"{day.date()}的日期类型不是0或1。")

        for slot in range(TIME_STEPS_PER_DAY):
            time_index = day_index * TIME_STEPS_PER_DAY + slot
            if time_index == 0:
                continue

            left = max(0, slot - RELATION_SLOT_RADIUS)
            right = min(TIME_STEPS_PER_DAY, slot + RELATION_SLOT_RADIUS + 1)
            specific_flow = flow_by_day_type[day_type, left:right].sum(axis=0)
            fallback_flow = flow_all[left:right].sum(axis=0)

            # 与Vit一致：某起点在相同日期类型下没有历史时，回退到所有日期类型。
            use_specific = specific_flow.sum(axis=1) > 0
            relation_flow = fallback_flow.copy()
            relation_flow[use_specific] = specific_flow[use_specific]

            out_total = relation_flow.sum(axis=1, keepdims=True)
            out_weight = np.divide(
                relation_flow,
                out_total,
                out=np.zeros_like(relation_flow),
                where=out_total > 0,
            )
            previous_demand = demand_cube[time_index - 1]
            related_dest[time_index] = out_weight @ previous_demand

            previous_time = timestamps[time_index - 1]
            actual_flow = np.zeros((n_regions, n_regions), dtype=np.float32)
            if previous_time in od_times:
                previous_od = od_by_time.get_group(previous_time)
                np.add.at(
                    actual_flow,
                    (
                        previous_od["i_idx"].to_numpy(dtype=np.int16),
                        previous_od["j_idx"].to_numpy(dtype=np.int16),
                    ),
                    previous_od["D_ijt_observed"].to_numpy(dtype=np.float32),
                )
            in_total = relation_flow.sum(axis=0, keepdims=True)
            in_weight = np.divide(
                relation_flow,
                in_total,
                out=np.zeros_like(relation_flow),
                where=in_total > 0,
            )
            related_origin[time_index] = (in_weight * actual_flow).sum(axis=0)

        # 当天全部96个时段计算完成后才纳入历史，防止同日未来信息泄漏。
        day_od = od.loc[od["date"] == day]
        if not day_od.empty:
            indices = (
                day_od["slot_in_day"].to_numpy(dtype=np.int16),
                day_od["i_idx"].to_numpy(dtype=np.int16),
                day_od["j_idx"].to_numpy(dtype=np.int16),
            )
            values = day_od["D_ijt_observed"].to_numpy(dtype=np.float32)
            np.add.at(flow_all, indices, values)
            np.add.at(flow_by_day_type[day_type], indices, values)

    outputs = (related_dest, related_origin)
    if any(not np.isfinite(values).all() for values in outputs):
        raise ValueError("历史OD关联区域特征中出现NaN或无穷值。")
    if any((values < 0).any() for values in outputs):
        raise ValueError("历史OD关联区域特征中出现负值。")
    if any(np.any(values[:TIME_STEPS_PER_DAY] != 0) for values in outputs):
        raise AssertionError("首日没有更早OD历史，两项关联特征应全部为0。")

    return tuple(values.reshape(-1) for values in outputs)


# ============================================================
# 2. 读取补全后的区域-15min完整面板
# ============================================================

df = pd.read_parquet(REGION_PANEL_FILE)

df["region_id"] = df["region_id"].astype("string")
df["t"] = pd.to_datetime(df["t"], errors="coerce")
df["D_origin_it"] = pd.to_numeric(df["D_origin_it"], errors="coerce").fillna(0).astype("int32")


# ============================================================
# 3. 基础检查
# ============================================================

if df[["region_id", "t"]].duplicated().sum() > 0:
    raise ValueError("区域-15min完整面板中存在重复的 region_id + t。")

region_count = df["region_id"].nunique()
time_count = df["t"].nunique()

if region_count != 801:
    raise ValueError(f"区域数量为 {region_count}，不是预期的801。")

if time_count != 2880:
    raise ValueError(f"时间段数量为 {time_count}，不是预期的2880。")

if len(df) != 801 * 2880:
    raise ValueError(f"完整面板样本数为 {len(df)}，不是预期的2306880。")

expected_internal_demand = 5_812_445
actual_internal_demand = int(df["D_origin_it"].sum())
if actual_internal_demand != expected_internal_demand:
    raise ValueError(
        f"内部OD口径 D_origin_it 总和为 {actual_internal_demand}，"
        f"不是预期的 {expected_internal_demand}。"
    )


# ============================================================
# 4. 构造标签
# ============================================================

# Dit预测任务的标签：当前区域 i 在当前时段 t 的真实出发订单数
# 注意：D_it_label 是标签，不是输入特征
df["D_it_label"] = df["D_origin_it"].astype("int32")


# ============================================================
# 5. 按 region_id 和 t 排序，为时间序列特征计算做准备
# ============================================================

df = df.sort_values(["region_id", "t"]).reset_index(drop=True)


# ============================================================
# 6. 构造同一区域历史需求特征
# ============================================================

# 上一个15min出发订单数：D_{i,t-1}^{out}
df["D_origin_i_lag1"] = (
    df.groupby("region_id")["D_origin_it"]
    .shift(1)
)

# 上两个15min出发订单数：D_{i,t-2}^{out}
df["D_origin_i_lag2"] = (
    df.groupby("region_id")["D_origin_it"]
    .shift(2)
)

# 过去4个15min，也就是过去1小时平均出发订单数
# 先shift(1)，再rolling，避免把当前时段真实D_origin_it放入输入特征
df["D_origin_i_roll4_mean"] = (
    df.groupby("region_id")["D_origin_it"]
    .transform(lambda s: s.shift(1).rolling(window=4, min_periods=1).mean())
)

# 过去8个15min，也就是过去2小时平均出发订单数
df["D_origin_i_roll8_mean"] = (
    df.groupby("region_id")["D_origin_it"]
    .transform(lambda s: s.shift(1).rolling(window=8, min_periods=1).mean())
)

# 前一天相同15min时段出发订单数：一天96个15min
df["D_origin_i_same_slot_last_day"] = (
    df.groupby("region_id")["D_origin_it"]
    .shift(96)
)

# 前一周相同15min时段出发订单数：一周672个15min
df["D_origin_i_same_slot_last_week"] = (
    df.groupby("region_id")["D_origin_it"]
    .shift(672)
)


# ============================================================
# 7. 构造区域长期活跃度特征
# ============================================================

# 提取日期，用于按天聚合
df["date"] = df["t"].dt.normalize()

# 每个区域每天的出发订单总量
daily = (
    df.groupby(["region_id", "date"], as_index=False)
    .agg(daily_D_origin_i=("D_origin_it", "sum"))
    .sort_values(["region_id", "date"])
    .reset_index(drop=True)
)

# 前一天全天出发订单总量
daily["D_origin_i_total_last_day"] = (
    daily.groupby("region_id")["daily_D_origin_i"]
    .shift(1)
)

# 过去3天日均出发订单量
# 先shift(1)，再rolling，避免把当天全天订单量放入当天样本的输入特征
daily["D_origin_i_daily_avg_last3"] = (
    daily.groupby("region_id")["daily_D_origin_i"]
    .transform(lambda s: s.shift(1).rolling(window=3, min_periods=1).mean())
)

# 过去7天日均出发订单量
daily["D_origin_i_daily_avg_last7"] = (
    daily.groupby("region_id")["daily_D_origin_i"]
    .transform(lambda s: s.shift(1).rolling(window=7, min_periods=1).mean())
)

daily_features = daily[
    [
        "region_id",
        "date",
        "D_origin_i_total_last_day",
        "D_origin_i_daily_avg_last3",
        "D_origin_i_daily_avg_last7",
    ]
].copy()

# 合并长期活跃度特征
df = df.merge(
    daily_features,
    on=["region_id", "date"],
    how="left"
)


# ============================================================
# 8. 历史特征缺失值处理
# ============================================================

# 前几个时段、第一天、第一周会自然缺失历史特征
# 这里统一填0，表示当前可观测历史中没有对应历史订单
own_history_cols = [
    "D_origin_i_lag1",
    "D_origin_i_lag2",
    "D_origin_i_roll4_mean",
    "D_origin_i_roll8_mean",
    "D_origin_i_same_slot_last_day",
    "D_origin_i_same_slot_last_week",
    "D_origin_i_total_last_day",
    "D_origin_i_daily_avg_last3",
    "D_origin_i_daily_avg_last7",
]

for col in own_history_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("float32")


# ============================================================
# 9. 读取邻近区域关系表
# ============================================================

neighbor = pd.read_parquet(NEIGHBOR_FILE)

neighbor["region_id"] = neighbor["region_id"].astype("string")
neighbor["neighbor_region_id"] = neighbor["neighbor_region_id"].astype("string")
neighbor["weight"] = pd.to_numeric(neighbor["weight"], errors="coerce").astype("float32")

# 只保留构造邻近需求特征需要的字段
neighbor = neighbor[
    [
        "region_id",
        "neighbor_region_id",
        "weight",
    ]
].copy()


# ============================================================
# 10. 检查邻近关系表中的区域是否都存在于完整面板
# ============================================================

panel_region_set = set(df["region_id"].dropna().unique())
neighbor_region_set = set(neighbor["region_id"].dropna().unique())
neighbor_neighbor_set = set(neighbor["neighbor_region_id"].dropna().unique())

if neighbor_region_set - panel_region_set:
    raise ValueError("邻近关系表中的 region_id 存在不属于完整面板的区域。")

if neighbor_neighbor_set - panel_region_set:
    raise ValueError("邻近关系表中的 neighbor_region_id 存在不属于完整面板的区域。")

weight_check = (
    neighbor.groupby("region_id")["weight"]
    .sum()
    .reset_index(name="weight_sum")
)

if not np.allclose(weight_check["weight_sum"].to_numpy(), 1.0, atol=1e-5):
    raise ValueError("邻近关系表中存在某些 region_id 的邻近权重和不等于1。")


# ============================================================
# 11. 构造邻近区域需求特征
# ============================================================

# 邻近需求特征使用邻近区域已经滞后的历史需求特征
# 不能使用邻近区域当前时段真实 D_origin_it，否则会造成信息泄露
neighbor_source_cols = [
    "D_origin_i_lag1",
    "D_origin_i_roll4_mean",
    "D_origin_i_roll8_mean",
]

neighbor_target_cols = {
    "D_origin_i_lag1": "neighbor_D_origin_lag1",
    "D_origin_i_roll4_mean": "neighbor_D_origin_roll4_mean",
    "D_origin_i_roll8_mean": "neighbor_D_origin_roll8_mean",
}

# 为了高效计算，先把完整面板排列成 t × region_id 的矩阵
df = df.sort_values(["t", "region_id"]).reset_index(drop=True)

region_ids = sorted(df["region_id"].astype(str).unique())
time_values = sorted(df["t"].dropna().unique())

n_regions = len(region_ids)
n_times = len(time_values)

# 检查排序后的完整面板是否严格等于 time × region_id 的笛卡尔积顺序
expected_region_order = np.tile(np.array(region_ids, dtype=object), n_times)
expected_time_order = np.repeat(np.array(time_values, dtype="datetime64[ns]"), n_regions)

actual_region_order = df["region_id"].astype(str).to_numpy()
actual_time_order = df["t"].to_numpy(dtype="datetime64[ns]")

if not np.array_equal(actual_region_order, expected_region_order):
    raise ValueError("完整面板的 region_id 排序不符合 time × region_id 的顺序。")

if not np.array_equal(actual_time_order, expected_time_order):
    raise ValueError("完整面板的 t 排序不符合 time × region_id 的顺序。")

# 建立 region_id 到矩阵列位置的映射
region_to_idx = {rid: idx for idx, rid in enumerate(region_ids)}

neighbor["target_idx"] = neighbor["region_id"].astype(str).map(region_to_idx)
neighbor["source_idx"] = neighbor["neighbor_region_id"].astype(str).map(region_to_idx)

if neighbor["target_idx"].isna().any() or neighbor["source_idx"].isna().any():
    raise ValueError("邻近关系表映射矩阵列位置时出现缺失。")

target_idx = neighbor["target_idx"].to_numpy(dtype=np.int32)
source_idx = neighbor["source_idx"].to_numpy(dtype=np.int32)
weights = neighbor["weight"].to_numpy(dtype=np.float32)

for source_col in neighbor_source_cols:
    target_col = neighbor_target_cols[source_col]

    # 当前特征矩阵 X：行是时间 t，列是区域 region_id
    X = df[source_col].to_numpy(dtype=np.float32).reshape(n_times, n_regions)

    # 邻近特征矩阵 Y：Y[t, i] = sum_k weight_{ik} * X[t, k]
    Y = np.zeros((n_times, n_regions), dtype=np.float32)

    for i_idx, k_idx, w in zip(target_idx, source_idx, weights):
        Y[:, i_idx] += w * X[:, k_idx]

    df[target_col] = Y.reshape(-1).astype("float32")


# ============================================================
# 12. 构造历史OD关联区域滞后特征
# ============================================================

(
    df["related_dest_demand_lag1"],
    df["related_origin_inflow_lag1"],
) = construct_related_region_features(df, region_ids, region_to_idx)


# ============================================================
# 13. 整理最终 Dit 特征表字段
# ============================================================

final_cols = [
    # 区域与时间定位
    "region_id",
    "t",

    # 时间特征
    "day_index",
    "slot_in_day",
    "dow",
    "is_weekend_or_holiday",
    "is_peak_morning",
    "is_peak_evening",
    "is_night",

    # 天气特征
    "temperature_2m",
    "relative_humidity_2m",
    "rain",
    "weather_code",
    "wind_speed_10m",

    # 区域自身历史需求特征
    "D_origin_i_lag1",
    "D_origin_i_lag2",
    "D_origin_i_roll4_mean",
    "D_origin_i_roll8_mean",
    "D_origin_i_same_slot_last_day",
    "D_origin_i_same_slot_last_week",

    # 区域长期活跃度特征
    "D_origin_i_total_last_day",
    "D_origin_i_daily_avg_last3",
    "D_origin_i_daily_avg_last7",

    # 邻近区域需求特征
    "neighbor_D_origin_lag1",
    "neighbor_D_origin_roll4_mean",
    "neighbor_D_origin_roll8_mean",

    # 历史OD关联区域特征
    "related_dest_demand_lag1",
    "related_origin_inflow_lag1",

    # 标签
    "D_it_label",
]

dit = df[final_cols].copy()


# ============================================================
# 14. 类型压缩
# ============================================================

dit["region_id"] = dit["region_id"].astype("string")

int_cols = [
    "day_index",
    "slot_in_day",
    "dow",
    "is_weekend_or_holiday",
    "is_peak_morning",
    "is_peak_evening",
    "is_night",
    "D_it_label",
]

for col in int_cols:
    dit[col] = pd.to_numeric(dit[col], errors="coerce").fillna(0).astype("int32")

float_cols = [
    "temperature_2m",
    "relative_humidity_2m",
    "rain",
    "weather_code",
    "wind_speed_10m",
    "D_origin_i_lag1",
    "D_origin_i_lag2",
    "D_origin_i_roll4_mean",
    "D_origin_i_roll8_mean",
    "D_origin_i_same_slot_last_day",
    "D_origin_i_same_slot_last_week",
    "D_origin_i_total_last_day",
    "D_origin_i_daily_avg_last3",
    "D_origin_i_daily_avg_last7",
    "neighbor_D_origin_lag1",
    "neighbor_D_origin_roll4_mean",
    "neighbor_D_origin_roll8_mean",
    "related_dest_demand_lag1",
    "related_origin_inflow_lag1",
]

for col in float_cols:
    dit[col] = pd.to_numeric(dit[col], errors="coerce").fillna(0).astype("float32")


# ============================================================
# 15. 保存 Dit 特征构建结果
# ============================================================

if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

dit.to_parquet(
    OUTPUT_FILE,
    index=False,
    engine="pyarrow",
    compression="snappy"
)


# ============================================================
# 16. 终端输出检查结果
# ============================================================

print("完成")
print("输出文件：", OUTPUT_FILE)
print("Dit特征表样本数：", len(dit))
print("区域数量：", dit["region_id"].nunique())
print("时间段数量：", dit["t"].nunique())
print("时间范围：", dit["t"].min(), "到", dit["t"].max())
print("D_it_label 总和：", int(dit["D_it_label"].sum()))
print("D_it_label 为0的样本数：", int((dit["D_it_label"] == 0).sum()))
print("D_it_label 大于0的样本数：", int((dit["D_it_label"] > 0).sum()))
print("缺失值总数：", int(dit.isna().sum().sum()))
