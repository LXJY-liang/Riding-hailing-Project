# -*- coding: utf-8 -*-
"""c_ij 构建流程的统一配置。

本文件只保存路径、数据口径和数值参数，不执行数据计算。
其他脚本都从这里读取配置，以保证历史订单筛选、距离单位和输出路径一致。
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# 1. 上游输入
# ---------------------------------------------------------------------------

# 已删除价格、服务时间和距离异常的逐笔订单：
# 用于统计 N_ij、历史里程中位数和收缩校准样本。
ORDER_FILE = PROJECT_ROOT / "1 数据清洗" / "1_4 异常订单清洗_保留订单.parquet"
# 801个 Census Tract 的编号和边界几何。
TRACT_FILE = (
    PROJECT_ROOT
    / "0 芝加哥census tract编号及对应的community"
    / "Census Tracts TIGER 2010.csv"
)
# 2024-06-30历史快照的芝加哥可驾驶有向道路网。
ROAD_GRAPH_FILE = (
    PROJECT_ROOT
    / "0 芝加哥道路网络"
    / "chicago_drive_2024-06-30.graphml"
)

# ---------------------------------------------------------------------------
# 2. 中间结果和最终结果
# ---------------------------------------------------------------------------

OUTPUT_DIR = CODE_DIR / "output"
INTERMEDIATE_DIR = OUTPUT_DIR / "intermediate"
FINAL_DIR = OUTPUT_DIR / "final"
AUDIT_DIR = OUTPUT_DIR / "audit"

HISTORICAL_OD_FILE = INTERMEDIATE_DIR / "historical_od_distance.parquet"
SHRINKAGE_SAMPLE_FILE = INTERMEDIATE_DIR / "shrinkage_training_samples.parquet"
REGION_ROUTING_NODE_FILE = INTERMEDIATE_DIR / "region_routing_nodes.parquet"
ROAD_DISTANCE_MATRIX_FILE = INTERMEDIATE_DIR / "od_road_distance_miles.npy"
INTRA_ZONE_PRIOR_FILE = INTERMEDIATE_DIR / "intra_zone_spatial_priors.parquet"
CALIBRATION_FILE = INTERMEDIATE_DIR / "calibration_parameters.json"

CIJ_FILE = FINAL_DIR / "c_ij.parquet"  # 完整六层估算明细。
CIJ_MATRIX_FILE = FINAL_DIR / "c_ij_matrix.npy"  # SAC快速查表用801×801矩阵。
AUDIT_SUMMARY_FILE = AUDIT_DIR / "six_level_cij_audit_summary.json"
AUDIT_HISTORY_FILE = AUDIT_DIR / "six_level_cij_historical_distances.csv"

# ---------------------------------------------------------------------------
# 3. 历史数据口径和六层阈值
# ---------------------------------------------------------------------------

# 只使用验证期之前已完成的订单，避免验证和测试信息泄漏。
HISTORY_END = "2024-06-25 00:00:00"
REGION_COUNT = 801
# N_ij >= 20 直接使用历史中位数；1 <= N_ij < 20 执行收缩。
DIRECT_HISTORY_THRESHOLD = 20
# N_ij >= 50 的OD用作稳定真值，由它们生成1—19笔的伪稀疏样本。
PSEUDO_TRUTH_THRESHOLD = 50

# 通过平均速度排除里程或时长明显异常的订单。
MIN_SPEED_MPH = 1.0
MAX_SPEED_MPH = 80.0
# 成本单位为美元，距离单位统一为英里。
COST_PER_MILE = 0.56
METERS_PER_MILE = 1609.344
PROJECTED_CRS = "EPSG:26916"

# ---------------------------------------------------------------------------
# 4. 路网抽样、收缩抽样和存储参数
# ---------------------------------------------------------------------------

# 区域内路网距离的抽样参数。该步骤只在前期准备中运行一次。
INTRA_ORIGIN_NODE_COUNT = 3
INTRA_DESTINATION_NODE_COUNT = 12
DIJKSTRA_BATCH_SIZE = 32

SHRINKAGE_SAMPLE_SIZES = (1, 3, 5, 10, 15, 19)
SHRINKAGE_REPEATS = 5
RANDOM_SEED = 2024029
# 六层核对的随机种子；保持不变即可重复得到同一组6个样本。
AUDIT_RANDOM_SEED = 20240783
PARQUET_COMPRESSION = "zstd"


def ensure_output_directories() -> None:
    """只创建本流程管理的输出目录，不改动任何上游数据。"""

    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
