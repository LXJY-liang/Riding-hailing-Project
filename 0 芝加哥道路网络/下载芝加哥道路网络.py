# -*- coding: utf-8 -*-
"""下载并保存本项目的芝加哥可驾驶有向道路网络。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
from shapely import wkt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent
TRACT_FILE = (
    PROJECT_ROOT
    / "0 芝加哥census tract编号及对应的community"
    / "Census Tracts TIGER 2010.csv"
)

BUFFER_METERS = 5_000
OSM_SNAPSHOT_UTC = "2024-06-30T23:59:59Z"
PROJECTED_CRS = "EPSG:26916"
GRAPH_FILE = OUTPUT_DIR / "chicago_drive_2024-06-30.graphml"
GRAPH_GPKG_FILE = OUTPUT_DIR / "chicago_drive_2024-06-30.gpkg"
QUERY_BOUNDARY_FILE = OUTPUT_DIR / "chicago_801_tracts_buffer_5km.geojson"
METADATA_FILE = OUTPUT_DIR / "road_network_metadata.json"
CACHE_DIR = OUTPUT_DIR / "osmnx_cache"
LOG_DIR = OUTPUT_DIR / "logs"


def load_query_polygon():
    tract_raw = pd.read_csv(TRACT_FILE, dtype="string", encoding="utf-8-sig")
    geometry = tract_raw["the_geom"].map(wkt.loads)
    tract = gpd.GeoDataFrame(
        tract_raw[["GEOID10"]].copy(),
        geometry=geometry,
        crs="EPSG:4326",
    )
    if len(tract) != 801 or tract["GEOID10"].nunique() != 801:
        raise ValueError("区域数量或区域编号不是预期的801个。")
    if not tract.geometry.is_valid.all():
        raise ValueError("区域边界存在无效几何。")

    study_area = tract.to_crs(PROJECTED_CRS).geometry.union_all()
    query_area = study_area.buffer(BUFFER_METERS)
    query_gdf = gpd.GeoDataFrame(
        {
            "name": ["Chicago 801 tracts + 5 km buffer"],
            "buffer_m": [BUFFER_METERS],
        },
        geometry=[query_area],
        crs=PROJECTED_CRS,
    ).to_crs("EPSG:4326")
    query_gdf.to_file(QUERY_BOUNDARY_FILE, driver="GeoJSON")
    return query_gdf.geometry.iloc[0]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    query_polygon = load_query_polygon()

    ox.settings.use_cache = True
    ox.settings.cache_folder = CACHE_DIR
    ox.settings.log_console = True
    ox.settings.log_file = True
    ox.settings.logs_folder = LOG_DIR
    ox.settings.requests_timeout = 600
    ox.settings.overpass_settings = (
        '[out:json][timeout:{timeout}]'
        f'[date:"{OSM_SNAPSHOT_UTC}"]'
        "{maxsize}"
    )

    graph = ox.graph.graph_from_polygon(
        query_polygon,
        network_type="drive",
        simplify=True,
        retain_all=False,
        truncate_by_edge=True,
    )

    ox.io.save_graphml(graph, GRAPH_FILE)
    ox.io.save_graph_geopackage(graph, GRAPH_GPKG_FILE, directed=True)

    weak_components = nx.number_weakly_connected_components(graph)
    strong_components = nx.number_strongly_connected_components(graph)
    largest_strong = max((len(c) for c in nx.strongly_connected_components(graph)), default=0)
    edge_lengths = [
        float(data["length"])
        for _, _, _, data in graph.edges(keys=True, data=True)
        if data.get("length") is not None
    ]
    metadata = {
        "source": "OpenStreetMap via OSMnx and Overpass API",
        "overpass_url": ox.settings.overpass_url,
        "osm_snapshot_utc": OSM_SNAPSHOT_UTC,
        "download_completed_utc": datetime.now(timezone.utc).isoformat(),
        "study_area": "union of the project's 801 Chicago Census Tracts",
        "buffer_meters": BUFFER_METERS,
        "network_type": "drive",
        "directed": True,
        "simplified": True,
        "edge_weight_for_future_routing": "length (meters)",
        "osmnx_version": ox.__version__,
        "networkx_version": nx.__version__,
        "nodes": int(graph.number_of_nodes()),
        "edges": int(graph.number_of_edges()),
        "weakly_connected_components": int(weak_components),
        "strongly_connected_components": int(strong_components),
        "largest_strong_component_nodes": int(largest_strong),
        "edges_with_length": int(len(edge_lengths)),
        "total_directed_edge_length_km": float(sum(edge_lengths) / 1000),
        "graphml_file": str(GRAPH_FILE),
        "geopackage_file": str(GRAPH_GPKG_FILE),
        "query_boundary_file": str(QUERY_BOUNDARY_FILE),
    }
    METADATA_FILE.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
