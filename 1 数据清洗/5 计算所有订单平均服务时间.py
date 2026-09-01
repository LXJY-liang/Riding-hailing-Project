# -*- coding: utf-8 -*-
"""计算清洗后所有订单的平均服务时间及其时长分布。"""

# 运行结果
# 订单数：6,061,035
# 平均服务时间：1,074.82 秒
# 即约：17.91 分钟
# 异常订单数：0


from pathlib import Path

import pandas as pd


# 清洗后的订单级数据与本脚本位于同一目录。
# 使用相对脚本位置拼接路径，移动整个项目后仍可正常运行。
DATA_FILE = Path(__file__).resolve().parent / "1_4 异常订单清洗_保留订单.parquet"

# 数据中的服务时间字段，原始单位为秒。
SERVICE_TIME_COLUMN = "Trip Seconds"

# 以 15 分钟为界统计订单数量；转换为秒后与原始字段保持相同单位。
TIME_THRESHOLD_MINUTES = 15
TIME_THRESHOLD_SECONDS = TIME_THRESHOLD_MINUTES * 60


def calculate_service_time_statistics(data_file: Path) -> dict[str, float | int]:
    """读取订单数据并计算平均服务时间和 15 分钟分组统计。"""
    if not data_file.exists():
        raise FileNotFoundError(f"找不到订单数据文件：{data_file}")

    # 只读取计算所需的一列，减少内存占用并提高读取速度。
    orders = pd.read_parquet(data_file, columns=[SERVICE_TIME_COLUMN])
    total_order_count = len(orders)

    # 将字段转换为数值；无法转换的内容会变成 NaN。
    service_seconds = pd.to_numeric(
        orders[SERVICE_TIME_COLUMN],
        errors="coerce",
    )

    # 缺失值、非数值以及小于等于 0 的时长不是有效服务时间，不参与平均值计算。
    valid_service_seconds = service_seconds[
        service_seconds.notna() & (service_seconds > 0)
    ]
    valid_order_count = len(valid_service_seconds)

    if valid_order_count == 0:
        raise ValueError("数据中没有可用于计算的有效服务时间。")

    # 分别统计小于、等于和大于 15 分钟的订单。
    # 单独保留“等于”组，避免这部分订单在结果中被遗漏。
    less_than_count = int((valid_service_seconds < TIME_THRESHOLD_SECONDS).sum())
    equal_to_count = int((valid_service_seconds == TIME_THRESHOLD_SECONDS).sum())
    greater_than_count = int((valid_service_seconds > TIME_THRESHOLD_SECONDS).sum())

    return {
        "total_count": total_order_count,
        "valid_count": valid_order_count,
        "average_seconds": float(valid_service_seconds.mean()),
        "less_than_count": less_than_count,
        "equal_to_count": equal_to_count,
        "greater_than_count": greater_than_count,
    }


def main() -> None:
    """执行计算并用常用时间单位输出结果。"""
    statistics = calculate_service_time_statistics(DATA_FILE)

    total_count = int(statistics["total_count"])
    valid_count = int(statistics["valid_count"])
    average_seconds = float(statistics["average_seconds"])
    excluded_count = total_count - valid_count

    less_than_count = int(statistics["less_than_count"])
    equal_to_count = int(statistics["equal_to_count"])
    greater_than_count = int(statistics["greater_than_count"])

    # 比例的分母是具有有效服务时间的订单数。
    less_than_ratio = less_than_count / valid_count
    equal_to_ratio = equal_to_count / valid_count
    greater_than_ratio = greater_than_count / valid_count

    print(f"全部订单数：{total_count:,}")
    print(f"有效订单数：{valid_count:,}")
    print(f"排除的异常订单数：{excluded_count:,}")
    print(f"平均服务时间：{average_seconds:.2f} 秒")
    print(f"平均服务时间：{average_seconds / 60:.2f} 分钟")
    print(f"平均服务时间：{average_seconds / 3600:.4f} 小时")
    print()
    print(f"服务时间小于 {TIME_THRESHOLD_MINUTES} 分钟：{less_than_count:,} 单，占比 {less_than_ratio:.2%}")
    print(f"服务时间等于 {TIME_THRESHOLD_MINUTES} 分钟：{equal_to_count:,} 单，占比 {equal_to_ratio:.2%}")
    print(f"服务时间大于 {TIME_THRESHOLD_MINUTES} 分钟：{greater_than_count:,} 单，占比 {greater_than_ratio:.2%}")


if __name__ == "__main__":
    main()
