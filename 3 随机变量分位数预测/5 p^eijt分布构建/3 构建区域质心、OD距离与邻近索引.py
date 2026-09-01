# -*- coding: utf-8 -*-
"""构建 p^e_ijt 需要的区域编号与邻近索引。

输出三个中间件：
1. 801 个 Census Tract 的质心与紧凑 region_index；
2. 统一编号后的邻近区域关系表；
3. 801 x 801 布尔邻接矩阵。

注意：质心在本流程中只用于建立稳定的 region_index，不再计算或
使用“OD 质心距离”筛选第 4 层。第 4 层改用历史订单的实际 Trip Miles。
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import wkt


def load_config():
    path = Path(__file__).resolve().parent / "1 p^eijt分布构建配置.py"
    spec = importlib.util.spec_from_file_location("pe_distribution_config", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


cfg = load_config()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建区域距离与邻近关系计算索引。")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="仅覆盖本脚本管理的三个区域索引文件。",
    )
    return parser.parse_args()


def clean_region_id(series: pd.Series) -> pd.Series:
    """将 Census Tract 编号统一为 11 位字符串。"""

    cleaned = series.astype("string").str.strip()
    cleaned = cleaned.str.replace(r"\.0$", "", regex=True)
    cleaned = cleaned.str.extract(r"(\d+)", expand=False)
    return cleaned.str.zfill(11)


def main() -> None:
    args = parse_args()
    cfg.ensure_output_directories()
    outputs = [
        cfg.REGION_CENTROID_FILE,
        cfg.NORMALIZED_NEIGHBOR_FILE,
        cfg.NEIGHBOR_MATRIX_FILE,
    ]
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "以下输出已存在；如确认重建，请使用 --overwrite：\n"
            + "\n".join(existing)
        )

    tract_raw = pd.read_csv(cfg.TRACT_FILE, dtype="string", encoding="utf-8-sig")
    id_column = "GEOID10" if "GEOID10" in tract_raw.columns else tract_raw.columns[4]

    geometry_column = None
    for candidate in ("the_geom", "THE_GEOM", "geometry", "Geometry"):
        if candidate in tract_raw.columns:
            geometry_column = candidate
            break
    if geometry_column is None:
        raise ValueError("无法在 Census Tract 文件中找到几何字段。")

    tract_raw["region_id"] = clean_region_id(tract_raw[id_column])
    tract_raw["geometry"] = tract_raw[geometry_column].apply(wkt.loads)
    tract = gpd.GeoDataFrame(
        tract_raw[["region_id", "geometry"]].copy(),
        geometry="geometry",
        crs="EPSG:4326",
    )
    tract = (
        tract.dropna(subset=["region_id"])
        .drop_duplicates("region_id")
        .sort_values("region_id")
        .reset_index(drop=True)
    )
    if len(tract) != cfg.REGION_COUNT:
        raise ValueError(
            f"区域数量为 {len(tract)}，预期为 {cfg.REGION_COUNT}。"
        )

    # EPSG:26916 是适用于芝加哥的米制投影坐标系。
    projected = tract.to_crs("EPSG:26916").copy()
    projected["centroid"] = projected.geometry.centroid

    centroids = pd.DataFrame(
        {
            "region_id": projected["region_id"].astype("string"),
            "region_index": np.arange(len(projected), dtype=np.int16),
            "centroid_x_m": projected["centroid"].x.astype("float64"),
            "centroid_y_m": projected["centroid"].y.astype("float64"),
        }
    )
    centroids.to_parquet(
        cfg.REGION_CENTROID_FILE,
        index=False,
        compression=cfg.PARQUET_COMPRESSION,
    )

    neighbors = pd.read_parquet(cfg.NEIGHBOR_FILE)
    required = {"region_id", "neighbor_region_id"}
    missing = sorted(required - set(neighbors.columns))
    if missing:
        raise ValueError(f"邻近关系表缺少字段：{missing}")

    neighbors["region_id"] = clean_region_id(neighbors["region_id"])
    neighbors["neighbor_region_id"] = clean_region_id(neighbors["neighbor_region_id"])
    region_to_index = dict(
        zip(centroids["region_id"].astype(str), centroids["region_index"].astype(int))
    )
    neighbors["region_index"] = neighbors["region_id"].map(region_to_index)
    neighbors["neighbor_region_index"] = neighbors["neighbor_region_id"].map(
        region_to_index
    )
    neighbors = neighbors.dropna(
        subset=["region_index", "neighbor_region_index"]
    ).copy()
    neighbors["region_index"] = neighbors["region_index"].astype("int16")
    neighbors["neighbor_region_index"] = neighbors[
        "neighbor_region_index"
    ].astype("int16")

    keep_columns = [
        "region_id",
        "neighbor_region_id",
        "region_index",
        "neighbor_region_index",
    ]
    for optional in (
        "is_boundary_neighbor",
        "distance_km",
        "weight",
        "neighbor_fallback_flag",
    ):
        if optional in neighbors.columns:
            keep_columns.append(optional)
    neighbors = (
        neighbors[keep_columns]
        .drop_duplicates(["region_index", "neighbor_region_index"])
        .sort_values(["region_index", "neighbor_region_index"])
        .reset_index(drop=True)
    )
    neighbors.to_parquet(
        cfg.NORMALIZED_NEIGHBOR_FILE,
        index=False,
        compression=cfg.PARQUET_COMPRESSION,
    )

    # 样本收集会进行数百万次“候选终点是否为目标终点邻区”的判断。
    # 提前保存紧凑布尔矩阵，避免主计算中反复调用 np.isin。
    neighbor_matrix = np.zeros(
        (cfg.REGION_COUNT, cfg.REGION_COUNT),
        dtype=np.bool_,
    )
    neighbor_matrix[
        neighbors["region_index"].to_numpy(dtype=np.int16),
        neighbors["neighbor_region_index"].to_numpy(dtype=np.int16),
    ] = True
    np.save(cfg.NEIGHBOR_MATRIX_FILE, neighbor_matrix)

    print("完成")
    print("区域数量：", len(centroids))
    print("邻近关系数量：", len(neighbors))
    print("区域质心：", cfg.REGION_CENTROID_FILE)
    print("邻近索引：", cfg.NORMALIZED_NEIGHBOR_FILE)
    print("邻接矩阵：", cfg.NEIGHBOR_MATRIX_FILE)


if __name__ == "__main__":
    main()
