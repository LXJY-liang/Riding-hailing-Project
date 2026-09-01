# -*- coding: utf-8 -*-
"""
第三步：将稀疏的内部 OD 口径区域聚合结果补全为完整面板。

完整面板固定为：
    801 个芝加哥 Census Tract × 2024-06-01至06-30的 2880 个15分钟时段
    = 2,306,880 行。

补全规则：
- 未观测到出发/到达订单的 D_origin_it、D_dest_it 填 0；
- net_flow_it 统一重新计算为 D_dest_it - D_origin_it；
- 没有出发订单时，价格、服务时间和距离均值保留 NaN，不伪造为 0；
- 时间特征和天气特征使用完整面板的 t 重新构造；
- 保留 is_weekend_or_holiday 以兼容旧代码，并新增 is_weekend、
  is_holiday 和 day_type，便于后续分开工作日、普通周末、节假日。

输入：3_2 区域-15min聚合（内部OD口径）结果.parquet
输出：3_3 区域-15min聚合（补全空值）结果.parquet
检查：3_3 区域-15min聚合（补全空值）_检查.json

注意：这份 3_3 结果从源头就是内部 OD 口径，不再需要 3_5 事后修正。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent

SPARSE_REGION_FILE = CODE_DIR / "3_2 区域-15min聚合（内部OD口径）结果.parquet"
CHICAGO_TRACT_FILE = (
    PROJECT_ROOT
    / "0 芝加哥census tract编号及对应的community"
    / "Census Tracts TIGER 2010.csv"
)
WEATHER_FILE = PROJECT_ROOT / "0 天气数据获取" / "chicago_weather_2024_06_hourly.csv"

OUTPUT_FILE = CODE_DIR / "3_3 区域-15min聚合（补全空值）结果.parquet"
DIAGNOSTICS_FILE = CODE_DIR / "3_3 区域-15min聚合（补全空值）_检查.json"

DATA_START = pd.Timestamp("2024-06-01 00:00:00")
DATA_END = pd.Timestamp("2024-06-30 23:45:00")
EXPECTED_REGION_COUNT = 801
EXPECTED_TIME_COUNT = 30 * 96
HOLIDAY_DATES = {pd.Timestamp("2024-06-19").date()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="补全内部 OD 口径的区域-15min面板。")
    parser.add_argument("--overwrite", action="store_true", help="覆盖本脚本的旧输出。")
    return parser.parse_args()


def clean_region_id(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    cleaned = cleaned.str.replace(r"\.0$", "", regex=True)
    cleaned = cleaned.str.extract(r"(\d+)", expand=False)
    return cleaned.str.zfill(11)


def load_chicago_regions() -> list[str]:
    chicago = pd.read_csv(CHICAGO_TRACT_FILE, dtype="string", encoding="utf-8-sig")
    if "GEOID10" in chicago.columns:
        id_column = "GEOID10"
    elif "GEOID" in chicago.columns:
        id_column = "GEOID"
    else:
        id_column = chicago.columns[4]
    region_ids = sorted(set(clean_region_id(chicago[id_column]).dropna().astype(str)))
    if len(region_ids) != EXPECTED_REGION_COUNT:
        raise ValueError(
            f"芝加哥区域数为 {len(region_ids)}，预期为 {EXPECTED_REGION_COUNT}。"
        )
    return region_ids


def main() -> None:
    args = parse_args()
    required = [SPARSE_REGION_FILE, CHICAGO_TRACT_FILE, WEATHER_FILE]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少上游输入：\n" + "\n".join(missing))

    existing = [path for path in (OUTPUT_FILE, DIAGNOSTICS_FILE) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "输出已存在；如需重建请使用 --overwrite：\n"
            + "\n".join(str(path) for path in existing)
        )
    if args.overwrite:
        for path in (OUTPUT_FILE, DIAGNOSTICS_FILE):
            if path.exists():
                path.unlink()

    region_ids = load_chicago_regions()
    time_index = pd.date_range(DATA_START, DATA_END, freq="15min")
    if len(time_index) != EXPECTED_TIME_COUNT:
        raise AssertionError(f"时间数为 {len(time_index)}，预期为 {EXPECTED_TIME_COUNT}。")

    panel = pd.MultiIndex.from_product(
        [region_ids, time_index], names=["region_id", "t"]
    ).to_frame(index=False)

    observed = pd.read_parquet(SPARSE_REGION_FILE)
    observed["region_id"] = clean_region_id(observed["region_id"])
    observed["t"] = pd.to_datetime(observed["t"], errors="coerce")
    required_columns = [
        "region_id", "t", "D_origin_it", "D_dest_it", "p_origin_bar_it",
        "tau_origin_bar_it", "d_origin_bar_it",
    ]
    missing_columns = sorted(set(required_columns) - set(observed.columns))
    if missing_columns:
        raise ValueError(f"稀疏区域聚合缺少字段：{missing_columns}")
    if observed[["region_id", "t"]].isna().any().any():
        raise ValueError("稀疏区域聚合存在缺失的 region_id 或 t。")
    if observed[["region_id", "t"]].duplicated().any():
        raise ValueError("稀疏区域聚合存在重复 (region_id,t)。")
    if not set(observed["region_id"].astype(str)).issubset(set(region_ids)):
        raise ValueError("稀疏区域聚合仍包含非芝加哥区域。")
    if not observed["t"].between(DATA_START, DATA_END, inclusive="both").all():
        raise ValueError("稀疏区域聚合包含 2024 年6月以外的时间。")

    panel = panel.merge(
        observed[required_columns], on=["region_id", "t"], how="left", validate="one_to_one"
    )
    panel["D_origin_it"] = panel["D_origin_it"].fillna(0).astype("int32")
    panel["D_dest_it"] = panel["D_dest_it"].fillna(0).astype("int32")
    panel["net_flow_it"] = (panel["D_dest_it"] - panel["D_origin_it"]).astype("int32")
    for column in ("p_origin_bar_it", "tau_origin_bar_it", "d_origin_bar_it"):
        panel[column] = pd.to_numeric(panel[column], errors="coerce").astype("float32")

    panel["day_index"] = (panel["t"].dt.normalize() - DATA_START).dt.days + 1
    panel["slot_in_day"] = panel["t"].dt.hour * 4 + panel["t"].dt.minute // 15
    panel["dow"] = panel["t"].dt.dayofweek
    panel["is_weekend"] = panel["dow"].isin([5, 6]).astype("int8")
    panel["is_holiday"] = panel["t"].dt.date.isin(HOLIDAY_DATES).astype("int8")
    panel["is_weekend_or_holiday"] = (
        (panel["is_weekend"] == 1) | (panel["is_holiday"] == 1)
    ).astype("int8")
    panel["day_type"] = "weekday"
    panel.loc[panel["is_weekend"] == 1, "day_type"] = "weekend"
    panel.loc[panel["is_holiday"] == 1, "day_type"] = "holiday"
    panel["day_type"] = panel["day_type"].astype("string")
    panel["is_peak_morning"] = (
        (panel["t"].dt.hour >= 7) & (panel["t"].dt.hour < 9)
    ).astype("int8")
    panel["is_peak_evening"] = (
        (panel["t"].dt.hour >= 16) & (panel["t"].dt.hour < 19)
    ).astype("int8")
    panel["is_night"] = (
        (panel["t"].dt.hour >= 22) | (panel["t"].dt.hour < 6)
    ).astype("int8")

    weather = pd.read_csv(WEATHER_FILE)
    weather["time"] = pd.to_datetime(weather["time"], errors="coerce")
    weather_columns = [
        "time", "temperature_2m", "relative_humidity_2m", "rain",
        "weather_code", "wind_speed_10m",
    ]
    missing_weather = sorted(set(weather_columns) - set(weather.columns))
    if missing_weather:
        raise ValueError(f"天气文件缺少字段：{missing_weather}")
    panel["weather_hour"] = panel["t"].dt.floor("h")
    panel = panel.merge(
        weather[weather_columns], left_on="weather_hour", right_on="time",
        how="left", validate="many_to_one",
    ).drop(columns=["weather_hour", "time"])

    final_columns = [
        "region_id", "t", "day_index", "slot_in_day", "dow", "is_weekend",
        "is_holiday", "is_weekend_or_holiday", "day_type", "is_peak_morning",
        "is_peak_evening", "is_night", "D_origin_it", "D_dest_it",
        "p_origin_bar_it", "tau_origin_bar_it", "d_origin_bar_it", "net_flow_it",
        "temperature_2m", "relative_humidity_2m", "rain", "weather_code",
        "wind_speed_10m",
    ]
    panel = panel[final_columns].sort_values(["region_id", "t"]).reset_index(drop=True)

    expected_rows = EXPECTED_REGION_COUNT * EXPECTED_TIME_COUNT
    if len(panel) != expected_rows:
        raise AssertionError(f"面板行数为 {len(panel):,}，预期为 {expected_rows:,}。")
    if panel[["region_id", "t"]].duplicated().any():
        raise AssertionError("完整面板存在重复 (region_id,t)。")
    if panel.loc[panel["D_origin_it"] == 0, "p_origin_bar_it"].notna().any():
        raise AssertionError("无出发订单的状态不应存在平均价格。")

    integer_dtypes = {
        "day_index": "int16", "slot_in_day": "int16", "dow": "int8",
        "is_weekend": "int8", "is_holiday": "int8",
        "is_weekend_or_holiday": "int8", "is_peak_morning": "int8",
        "is_peak_evening": "int8", "is_night": "int8",
    }
    for column, dtype in integer_dtypes.items():
        panel[column] = panel[column].astype(dtype)
    for column in (
        "temperature_2m", "relative_humidity_2m", "rain", "weather_code", "wind_speed_10m",
    ):
        panel[column] = pd.to_numeric(panel[column], errors="coerce").astype("float32")

    panel.to_parquet(OUTPUT_FILE, index=False, compression="zstd")
    diagnostics = {
        "output_rows": int(len(panel)),
        "region_count": int(panel["region_id"].nunique()),
        "time_count": int(panel["t"].nunique()),
        "D_origin_it_total": int(panel["D_origin_it"].sum()),
        "D_dest_it_total": int(panel["D_dest_it"].sum()),
        "positive_origin_rows": int((panel["D_origin_it"] > 0).sum()),
        "positive_destination_rows": int((panel["D_dest_it"] > 0).sum()),
        "zero_origin_rows": int((panel["D_origin_it"] == 0).sum()),
        "weather_missing_rows": int(panel["temperature_2m"].isna().sum()),
        "weekday_rows": int((panel["day_type"] == "weekday").sum()),
        "weekend_rows": int((panel["day_type"] == "weekend").sum()),
        "holiday_rows": int((panel["day_type"] == "holiday").sum()),
        "output_file": str(OUTPUT_FILE),
        "data_scope": "Chicago internal OD only",
    }
    DIAGNOSTICS_FILE.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
