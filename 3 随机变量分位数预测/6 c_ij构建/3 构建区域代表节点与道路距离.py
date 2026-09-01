# -*- coding: utf-8 -*-
"""构建801个区域的路由节点、OD道路距离矩阵和区域内距离先验。

输入：Census Tract边界和已下载的芝加哥可驾驶有向路网。
输出：
1. region_routing_nodes.parquet：每个区域的路由代表节点；
2. od_road_distance_miles.npy：801×801有向道路最短距离矩阵；
3. intra_zone_spatial_priors.parquet：区域内路网距离和空间半径先验。

所有路网计算都在本前期脚本中完成。后续校准和最终 c_ij 计算只读取
已保存的数值结果，不再重复运行最短路。
"""

from __future__ import annotations

import importlib.util
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from scipy.sparse.csgraph import dijkstra
from shapely import wkt


def load_config():
    path = Path(__file__).resolve().parent / "1 c_ij构建配置.py"
    spec = importlib.util.spec_from_file_location("cij_config", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


cfg = load_config()


def main() -> None:
    cfg.ensure_output_directories()

    # 按11位GEOID排序后固定region_index，使Parquet和numpy矩阵使用同一行列顺序。
    tract_raw = pd.read_csv(cfg.TRACT_FILE, dtype="string", encoding="utf-8-sig")
    tract = gpd.GeoDataFrame(
        {"region_id": tract_raw["GEOID10"].str.zfill(11)},
        geometry=tract_raw["the_geom"].map(wkt.loads),
        crs="EPSG:4326",
    ).sort_values("region_id", ignore_index=True)
    tract["region_index"] = np.arange(len(tract), dtype=np.int16)
    tract = tract.to_crs(cfg.PROJECTED_CRS)
    tract["centroid"] = tract.geometry.centroid
    tract["representative_point"] = tract.geometry.representative_point()

    # 有向图中只保留最大强连通分量，保证任意两个选中节点之间都有可驶路径。
    graph = ox.io.load_graphml(cfg.ROAD_GRAPH_FILE)
    largest_strong = max(nx.strongly_connected_components(graph), key=len)
    graph = graph.subgraph(largest_strong).copy()
    graph_projected = ox.projection.project_graph(graph, to_crs=cfg.PROJECTED_CRS)

    nodes = ox.convert.graph_to_gdfs(
        graph_projected,
        nodes=True,
        edges=False,
    )
    node_index_name = nodes.index.name or "node_id"
    nodes = nodes.reset_index().rename(columns={node_index_name: "node_id"})
    nodes_by_id = nodes.set_index("node_id")

    # 通过空间连接一次性得到每个区域内部的主强连通路网节点。
    joined = gpd.sjoin(
        nodes[["node_id", "geometry"]],
        tract[["region_id", "geometry"]],
        predicate="within",
        how="inner",
    )
    candidates = {
        region_id: group["node_id"].to_numpy()
        for region_id, group in joined.groupby("region_id", sort=False)
    }

    routing_rows = []
    routing_nodes = []
    for row in tract.itertuples():
        region_nodes = candidates.get(row.region_id, np.empty(0, dtype=object))
        fallback_flag = int(len(region_nodes) == 0)
        # 通常选区域内距离几何质心最近的道路节点。若区域内没有节点，
        # 则从保证在边界内的 representative_point 出发，吸附到全图最近可达节点。
        target_point = row.centroid if not fallback_flag else row.representative_point

        if fallback_flag:
            dx = nodes.geometry.x.to_numpy() - target_point.x
            dy = nodes.geometry.y.to_numpy() - target_point.y
            node_id = nodes.iloc[int(np.argmin(dx * dx + dy * dy))]["node_id"]
        else:
            candidate_geometry = nodes_by_id.loc[region_nodes, "geometry"]
            node_id = candidate_geometry.distance(target_point).idxmin()

        node_geometry = nodes_by_id.loc[node_id, "geometry"]
        routing_nodes.append(node_id)
        routing_rows.append(
            {
                "region_id": row.region_id,
                "region_index": row.region_index,
                "routing_node_id": str(node_id),
                "routing_x_m": float(node_geometry.x),
                "routing_y_m": float(node_geometry.y),
                "snap_distance_m": float(node_geometry.distance(target_point)),
                "centroid_inside_flag": int(row.centroid.within(row.geometry)),
                "fallback_flag": fallback_flag,
            }
        )

    routing = pd.DataFrame(routing_rows)
    routing.to_parquet(
        cfg.REGION_ROUTING_NODE_FILE,
        index=False,
        compression=cfg.PARQUET_COMPRESSION,
    )

    # 区域内的质心到自身距离为0，不能作为 i=j 的先验。因此从每个区域
    # 抽取少量起终道路节点，将节点对最短距离的中位数作为区域内路网先验。
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    intra_targets = defaultdict(list)
    for row in tract.itertuples():
        region_nodes = candidates.get(row.region_id, np.empty(0, dtype=object))
        if len(region_nodes) < 2:
            continue
        origin_count = min(cfg.INTRA_ORIGIN_NODE_COUNT, len(region_nodes))
        destination_count = min(cfg.INTRA_DESTINATION_NODE_COUNT, len(region_nodes))
        origins = rng.choice(region_nodes, size=origin_count, replace=False)
        destinations = rng.choice(
            region_nodes,
            size=destination_count,
            replace=False,
        )
        for origin in origins:
            targets = destinations[destinations != origin]
            if len(targets):
                intra_targets[origin].append((row.region_index, targets))

    # MultiDiGraph中两个节点可能有多条平行边。转换为DiGraph时保留length最短的边，
    # 然后转为SciPy稀疏矩阵，以批量Dijkstra代替数千次Python循环路由。
    simple_graph = ox.convert.to_digraph(graph, weight="length")
    nodelist = list(simple_graph.nodes)
    node_position = {node_id: index for index, node_id in enumerate(nodelist)}
    sparse_graph = nx.to_scipy_sparse_array(
        simple_graph,
        nodelist=nodelist,
        weight="length",
        dtype=np.float64,
        format="csr",
    )

    route_rows_by_source = defaultdict(list)
    for region_index, node_id in enumerate(routing_nodes):
        route_rows_by_source[node_id].append(region_index)

    source_nodes = list(dict.fromkeys(routing_nodes + list(intra_targets)))
    route_target_positions = np.asarray(
        [node_position[node_id] for node_id in routing_nodes],
        dtype=np.int64,
    )
    road_distance_meters = np.empty(
        (cfg.REGION_COUNT, cfg.REGION_COUNT),
        dtype=np.float64,
    )
    intra_values = [[] for _ in range(cfg.REGION_COUNT)]

    # 每批同时处理若干源节点：代表节点距离写入801×801矩阵，
    # 区域内抽样源节点只保留到本区域抽样终点的距离。
    for start in range(0, len(source_nodes), cfg.DIJKSTRA_BATCH_SIZE):
        batch_nodes = source_nodes[start : start + cfg.DIJKSTRA_BATCH_SIZE]
        batch_positions = [node_position[node_id] for node_id in batch_nodes]
        distances = np.atleast_2d(
            dijkstra(
                sparse_graph,
                directed=True,
                indices=batch_positions,
                return_predecessors=False,
            )
        )
        for local_index, source_node in enumerate(batch_nodes):
            source_distances = distances[local_index]
            for region_index in route_rows_by_source.get(source_node, ()):
                road_distance_meters[region_index] = source_distances[
                    route_target_positions
                ]
            for region_index, target_nodes in intra_targets.get(source_node, ()):
                target_positions = [node_position[node] for node in target_nodes]
                values = source_distances[target_positions]
                intra_values[region_index].extend(
                    values[np.isfinite(values) & (values > 0)].tolist()
                )

    road_distance_miles = road_distance_meters / cfg.METERS_PER_MILE
    np.save(
        cfg.ROAD_DISTANCE_MATRIX_FILE,
        road_distance_miles.astype(np.float32),
    )

    # Dijkstra返回的道路距离和投影坐标计算的半径都是米。这里统一除以
    # METERS_PER_MILE，使区域间距离、区域内路网距离和有效半径全部以英里保存。
    # 等面积半径 sqrt(area/pi) 仅在区域内路网先验不可得时作为回退变量；
    # 两种空间变量都将在第4个脚本中使用高样本 i=i 订单另行校准。
    area_square_meters = tract.geometry.area.to_numpy(dtype=np.float64)
    intra_network_miles = np.asarray(
        [np.median(values) if values else np.nan for values in intra_values],
        dtype=np.float64,
    ) / cfg.METERS_PER_MILE
    intra_prior = pd.DataFrame(
        {
            "region_id": tract["region_id"].astype("string"),
            "region_index": tract["region_index"].astype("int16"),
            "effective_radius_miles": (
                np.sqrt(area_square_meters / np.pi) / cfg.METERS_PER_MILE
            ).astype("float32"),
            "d_intra_network_miles": intra_network_miles.astype("float32"),
        }
    )
    intra_prior.to_parquet(
        cfg.INTRA_ZONE_PRIOR_FILE,
        index=False,
        compression=cfg.PARQUET_COMPRESSION,
    )


if __name__ == "__main__":
    main()
