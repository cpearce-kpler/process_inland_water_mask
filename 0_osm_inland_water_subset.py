# -*- coding: utf-8 -*-
"""
Create a lightweight high-resolution inland-water subset using vessel evidence.

Decision rule
-------------
A source inland-water feature is retained when either:

1. it intersects the stopped-vessel-position (SVP) layer; or
2. it intersects the vessel-focused Global Sea Waters (GSW) layer and also
   overlaps at least one positive pixel in the configured Boolean vessel-
   activity MBTiles dataset; or
3. it passes the GSW test, fails the Boolean-activity test, and contains at
   least the configured number of raw 2025 AIS points read from GeoParquet
   files.

Operationally:

    SVP hit                              -> KEEP immediately
    no SVP + no GSW hit                  -> DISCARD
    no SVP + GSW + Boolean activity      -> KEEP
    no SVP + GSW + no Boolean activity   -> RAW AIS fallback
    raw AIS points >= threshold          -> KEEP
    raw AIS points < threshold           -> DISCARD

The large source GeoPackage is never loaded in full. It is read through its
GeoPackage RTree in adaptive projected-coordinate tiles. Each source feature is
owned by exactly one tile using its representative point, preventing duplicate
output when a polygon crosses tile boundaries.

Only retained geometries are written. Rejected geometries are counted but are
not saved. Tile checkpoints and a JSON manifest make the process resumable.

Required packages
-----------------
geopandas, numpy, pandas, Pillow, pyarrow, pyogrio, pyproj, rasterio, shapely
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Iterator, Sequence
import gc
import hashlib
import json
import math
import os
import sqlite3
import sys
import time
import traceback

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
try:
    from PIL import Image, UnidentifiedImageError
except ImportError:
    Image = None
    UnidentifiedImageError = Exception
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None
    pq = None
import pyproj
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.windows import Window, transform as window_transform
import shapely
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box, mapping


# =============================================================================
# USER SETTINGS
# =============================================================================

# High-resolution source inland waters.
HIGH_RES_GPKG = Path(
    r"C:\Users\Craig Pearce\Desktop\osm\global_orig_land_water_masks_gpkg"
    r"\inland_water_split.gpkg"
)
HIGH_RES_LAYER = "inland_water_split"

# Strong positive evidence. The first spatial layer is selected when None.
SVP_GPKG = Path(
    r"C:\Users\Craig Pearce\Desktop\svp_2025_dp100_sp0_res0p001.gpkg"
)
SVP_LAYER: str | None = None

# Vessel-focused low-resolution water layer. The first spatial layer is selected
# when None. The optional buffer is applied after reprojection to the source CRS.
GSW_GPKG = Path(r"C:\Users\Craig Pearce\Desktop\global_sea_waters.gpkg")
GSW_LAYER: str | None = None
GSW_BUFFER_M = 0.0

# Boolean 2025 vessel-activity raster tiles. The MBTiles database is expected
# to contain raster image tiles at zoom 9. Missing tiles count as no activity.
BOOLEAN_ACTIVITY_MBTILES = Path(
    r"C:\Users\Craig Pearce\Desktop\Data_sets\ais_files\heatmap"
    r"\ais_heatmap_tiles_all_boolean_activity_z9_2025"
    r"\ais_all_vessels_boolean_activity_z9.mbtiles"
)
BOOLEAN_ACTIVITY_ZOOM = 9

# Standard MBTiles stores tile_row using TMS numbering. Leave None to read the
# optional metadata 'scheme' value and otherwise default to the MBTiles standard
# of TMS. Set explicitly to "xyz" only for a non-standard XYZ-row database.
MBTILES_SCHEME: str | None = None

# Supported activity encodings:
#   auto                       infer from sampled raster tiles
#   alpha                      alpha > threshold means activity
#   nonzero                    any non-zero RGB/grayscale value means activity
#   different_from_background any pixel different from BACKGROUND_RGBA is active
#   stored_tile                every pixel in every stored tile is active
# Auto mode is recommended. It prefers transparency, then non-zero grayscale/
# black-background data, then the most common sampled RGBA colour as background.
MBTILES_ACTIVITY_MODE = "auto"
MBTILES_ACTIVITY_THRESHOLD = 0
MBTILES_BACKGROUND_RGBA: tuple[int, int, int, int] | None = None
MBTILES_ALL_TOUCHED = True
MBTILES_ENCODING_SAMPLE_TILES = 16
MBTILES_TILE_CACHE_MAX_MB = 256.0

# Raw 2025 AIS GeoParquet fallback. This stage is used only for high-resolution
# water polygons that pass GSW but fail the Boolean-activity MBTiles test. Longitude and
# latitude column names are matched case-insensitively; common alternatives are
# tried automatically if the configured names are absent.
AIS_PARQUET_DIR = Path(
    r"C:\Users\Craig Pearce\Desktop\Data_sets\ais_files\ais_2025"
)
AIS_PARQUET_PATTERNS = ("*.parquet", "*.geoparquet", "*.pq")
AIS_LONGITUDE_COLUMN: str | None = "lon"
AIS_LATITUDE_COLUMN: str | None = "lat"
AIS_MIN_POINTS_PER_WATER = 10

# AIS is streamed once, row group by row group. Candidate polygons are indexed
# in an on-disk SQLite database and loaded into bounded cell-level STRtree caches.
AIS_PARQUET_BATCH_ROWS = 500_000
AIS_SPATIAL_CELL_SIZE_M = 100_000.0
AIS_CELL_CACHE_MAX_WKB_MB = 512.0
AIS_SKIP_ROW_GROUPS_WITHOUT_CANDIDATE_CELLS = True
AIS_RESUME_PROCESSED_ROW_GROUPS = True
REQUIRE_AIS_SCAN_COMPLETE_FOR_FINAL = True

# Optional test limits. Leave both as None for the complete AIS scan. Progress is
# committed after every row group, so an interrupted scan can resume safely.
MAX_AIS_FILES_THIS_RUN: int | None = None
MAX_AIS_ROW_GROUPS_THIS_RUN: int | None = None

# Output. Only kept source geometries are written to the final GeoPackage.
OUTPUT_DIR = Path(
    r"C:\Users\Craig Pearce\Desktop\inland_water_vessel_subset_2025_mbtiles_ais10_work"
)
TILE_CHECKPOINT_DIR = OUTPUT_DIR / "tile_checkpoints"
MANIFEST_PATH = OUTPUT_DIR / "processing_manifest.json"
REPORT_PATH = OUTPUT_DIR / "processing_report.json"
ERROR_LOG_PATH = OUTPUT_DIR / "processing_errors.txt"
FINAL_GPKG = Path(
    r"C:\Users\Craig Pearce\Desktop\inland_water_vessel_subset_2025_mbtiles_ais10.gpkg"
)
FINAL_LAYER = "inland_water_subset"
CHECKPOINT_LAYER = "kept_inland_water"
CHECKPOINT_STATUS_LAYER = "tile_status"
AIS_CANDIDATE_LAYER = "ais_fallback_candidates"
AIS_INDEX_DB = OUTPUT_DIR / "ais_candidate_index.sqlite"
AIS_SCAN_REPORT_PATH = OUTPUT_DIR / "ais_scan_report.json"

# Adaptive source tiling. Large candidate counts are recursively split before
# geometry is loaded. This requires the source GeoPackage RTree.
BASE_TILE_SIZE_M = 1_000_000.0
MIN_TILE_SIZE_M = 125_000.0
MAX_RTREE_CANDIDATES_PER_TILE = 75_000
MAX_TILE_RECURSION_LEVEL = 4

# MBTiles activity masks are decoded lazily and retained in a bounded LRU cache.
# A 256 x 256 Boolean mask occupies about 64 KiB before Python overhead.
# The configured cache is deliberately modest relative to available RAM.

# Output and recovery controls.
RESUME_COMPLETED_TILES = True
CONTINUE_ON_TILE_ERROR = True
REQUIRE_ALL_TILES_COMPLETE_FOR_FINAL = True
OVERWRITE_FINAL_GPKG = True
KEEP_TILE_CHECKPOINTS = True
FINAL_WRITE_TILE_BATCH = 25
CREATE_SPATIAL_INDEX = True
VACUUM_FINAL_GPKG = False

# Spatial-index safety. Repeated bbox reads against the large source are not
# viable without a GeoPackage RTree. SVP is also repeatedly queried by bbox.
REQUIRE_HIGH_RES_RTREE = True
REQUIRE_SVP_FAST_SPATIAL_FILTER = True

# Use normal GDAL reads/writes. This avoids Arrow-version incompatibilities on
# older GDAL installations.
USE_ARROW = False

# Testing and targeted recovery. None means use every planned source tile.
PROCESS_ONLY_TILE_IDS: set[str] | None = None
MAX_TILES_THIS_RUN: int | None = None

# Progress and cleanup.
ETA_RECENT_TILES = 10
GC_EVERY_TILES = 1

# The output intentionally contains a very small attribute schema.
WRITE_BOOLEAN_EVIDENCE_FIELDS = True

# Increment this when geometry-decision logic changes so incompatible checkpoint
# directories are not reused silently.
PROCESSING_LOGIC_VERSION = "svp_or_gsw_and_boolean_mbtiles_or_ais10_adaptive_tiles_v3"


# =============================================================================
# LOGGING AND SMALL UTILITIES
# =============================================================================


def log(message: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def format_duration(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(seconds) or seconds < 0:
        return "unknown"
    total = int(round(seconds))
    days, remainder = divmod(total, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, secs = divmod(remainder, 60)
    if days:
        return f"{days} d {hours:02d} h {minutes:02d} min"
    if hours:
        return f"{hours} h {minutes:02d} min {secs:02d} s"
    if minutes:
        return f"{minutes} min {secs:02d} s"
    return f"{secs} s"


@contextmanager
def timed(label: str) -> Iterator[None]:
    started = time.perf_counter()
    log(f"START: {label}")
    try:
        yield
    except Exception as exc:
        log(
            f"FAILED: {label} after {format_duration(time.perf_counter() - started)}: "
            f"{type(exc).__name__}: {exc}"
        )
        raise
    else:
        log(f"DONE: {label} in {format_duration(time.perf_counter() - started)}")


def stage_banner(number: int, total: int, label: str) -> None:
    log("=" * 78)
    log(f"STAGE {number}/{total}: {label}")
    log("=" * 78)


def write_json_atomic(path: Path, data: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(temp, path)


def select_spatial_layer(path: Path, requested: str | None) -> str:
    layers = pyogrio.list_layers(path)
    if requested is not None:
        names = {str(row[0]) for row in layers}
        if requested not in names:
            raise ValueError(
                f"Layer {requested!r} not found in {path}. Available: {sorted(names)}"
            )
        return requested
    for name, geometry_type in layers:
        if geometry_type is not None:
            return str(name)
    raise ValueError(f"No spatial layer found in {path}")


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def bbox_intersects(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> bool:
    return not (
        left[2] < right[0]
        or left[0] > right[2]
        or left[3] < right[1]
        or left[1] > right[3]
    )


def finite_bounds(geometry: Any) -> tuple[float, float, float, float] | None:
    if geometry is None or geometry.is_empty:
        return None
    bounds = tuple(map(float, geometry.bounds))
    if not np.isfinite(bounds).all():
        return None
    return bounds


# =============================================================================
# VECTOR LAYER METADATA AND RTREE SUPPORT
# =============================================================================


@dataclass(frozen=True)
class VectorLayerInfo:
    path: Path
    layer: str
    crs: pyproj.CRS
    geometry_type: str | None
    feature_count: int
    bounds: tuple[float, float, float, float]
    fast_spatial_filter: bool
    geometry_column: str | None
    rtree_table: str | None


def gpkg_geometry_column(path: Path, layer: str) -> str | None:
    if path.suffix.lower() != ".gpkg":
        return None
    with sqlite3.connect(path) as connection:
        row = connection.execute(
            "SELECT column_name FROM gpkg_geometry_columns WHERE table_name = ?",
            (layer,),
        ).fetchone()
    return None if row is None else str(row[0])


def gpkg_rtree_table(path: Path, layer: str, geometry_column: str | None) -> str | None:
    if geometry_column is None or path.suffix.lower() != ".gpkg":
        return None
    candidate = f"rtree_{layer}_{geometry_column}"
    with sqlite3.connect(path) as connection:
        row = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
            (candidate,),
        ).fetchone()
    return candidate if row is not None else None


def inspect_vector_layer(path: Path, requested_layer: str | None) -> VectorLayerInfo:
    if not path.exists():
        raise FileNotFoundError(path)
    layer = select_spatial_layer(path, requested_layer)
    info = pyogrio.read_info(
        path,
        layer=layer,
        force_feature_count=True,
        force_total_bounds=True,
    )
    if not info.get("crs"):
        raise ValueError(f"Layer {layer!r} in {path} has no CRS")
    bounds_raw = info.get("total_bounds")
    if bounds_raw is None:
        raise ValueError(f"Layer {layer!r} in {path} has no total bounds")
    bounds = tuple(map(float, bounds_raw))
    capabilities = info.get("capabilities") or {}
    geometry_column = gpkg_geometry_column(path, layer)
    rtree_table = gpkg_rtree_table(path, layer, geometry_column)
    return VectorLayerInfo(
        path=path,
        layer=layer,
        crs=pyproj.CRS.from_user_input(info["crs"]),
        geometry_type=(
            None if info.get("geometry_type") is None else str(info.get("geometry_type"))
        ),
        feature_count=int(info.get("features", -1)),
        bounds=bounds,
        fast_spatial_filter=bool(capabilities.get("fast_spatial_filter", False)),
        geometry_column=geometry_column,
        rtree_table=rtree_table,
    )


def rtree_candidate_count(
    layer_info: VectorLayerInfo,
    bounds: tuple[float, float, float, float],
) -> int | None:
    if layer_info.rtree_table is None:
        return None
    minx, miny, maxx, maxy = bounds
    sql = (
        f"SELECT COUNT(*) FROM {quote_identifier(layer_info.rtree_table)} "
        "WHERE maxx >= ? AND minx <= ? AND maxy >= ? AND miny <= ?"
    )
    with sqlite3.connect(layer_info.path) as connection:
        row = connection.execute(sql, (minx, maxx, miny, maxy)).fetchone()
    return 0 if row is None else int(row[0])


def log_layer_info(label: str, info: VectorLayerInfo) -> None:
    log(f"{label} path: {info.path}")
    log(f"{label} layer: {info.layer}")
    log(f"{label} CRS: {info.crs.to_string()}")
    log(f"{label} geometry type: {info.geometry_type}")
    log(f"{label} features: {info.feature_count:,}")
    log(f"{label} bounds: {info.bounds}")
    log(f"{label} fast spatial filter: {info.fast_spatial_filter}")
    log(f"{label} GeoPackage RTree: {info.rtree_table or 'not found'}")


# =============================================================================
# ADAPTIVE SOURCE TILE PLAN
# =============================================================================


@dataclass(frozen=True)
class Tile:
    tile_id: str
    minx: float
    miny: float
    maxx: float
    maxy: float
    level: int
    expected_candidates: int | None
    include_max_x: bool
    include_max_y: bool

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (self.minx, self.miny, self.maxx, self.maxy)

    @property
    def width(self) -> float:
        return self.maxx - self.minx

    @property
    def height(self) -> float:
        return self.maxy - self.miny

    @property
    def core(self) -> Polygon:
        return box(self.minx, self.miny, self.maxx, self.maxy)

    @property
    def checkpoint_path(self) -> Path:
        return TILE_CHECKPOINT_DIR / f"{self.tile_id}.gpkg"


def aligned_grid_bounds(
    bounds: tuple[float, float, float, float], tile_size: float
) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = bounds
    return (
        math.floor(minx / tile_size) * tile_size,
        math.floor(miny / tile_size) * tile_size,
        math.ceil(maxx / tile_size) * tile_size,
        math.ceil(maxy / tile_size) * tile_size,
    )


def split_tile(tile: Tile) -> list[Tile]:
    midx = (tile.minx + tile.maxx) / 2.0
    midy = (tile.miny + tile.maxy) / 2.0
    child_bounds = [
        (tile.minx, tile.miny, midx, midy),
        (midx, tile.miny, tile.maxx, midy),
        (tile.minx, midy, midx, tile.maxy),
        (midx, midy, tile.maxx, tile.maxy),
    ]
    children: list[Tile] = []
    for quadrant, (minx, miny, maxx, maxy) in enumerate(child_bounds):
        children.append(
            Tile(
                tile_id=f"{tile.tile_id}_q{quadrant}",
                minx=minx,
                miny=miny,
                maxx=maxx,
                maxy=maxy,
                level=tile.level + 1,
                expected_candidates=None,
                include_max_x=tile.include_max_x and math.isclose(maxx, tile.maxx),
                include_max_y=tile.include_max_y and math.isclose(maxy, tile.maxy),
            )
        )
    return children


def plan_adaptive_tiles(source: VectorLayerInfo) -> tuple[list[Tile], tuple[float, float, float, float]]:
    if BASE_TILE_SIZE_M <= 0 or MIN_TILE_SIZE_M <= 0:
        raise ValueError("Tile sizes must be positive")
    if MIN_TILE_SIZE_M > BASE_TILE_SIZE_M:
        raise ValueError("MIN_TILE_SIZE_M cannot exceed BASE_TILE_SIZE_M")

    grid_bounds = aligned_grid_bounds(source.bounds, BASE_TILE_SIZE_M)
    grid_minx, grid_miny, grid_maxx, grid_maxy = grid_bounds
    columns = int(round((grid_maxx - grid_minx) / BASE_TILE_SIZE_M))
    rows = int(round((grid_maxy - grid_miny) / BASE_TILE_SIZE_M))

    base_tiles: list[Tile] = []
    for row in range(rows):
        miny = grid_miny + row * BASE_TILE_SIZE_M
        maxy = min(grid_maxy, miny + BASE_TILE_SIZE_M)
        for column in range(columns):
            minx = grid_minx + column * BASE_TILE_SIZE_M
            maxx = min(grid_maxx, minx + BASE_TILE_SIZE_M)
            base_tiles.append(
                Tile(
                    tile_id=f"r{row:03d}_c{column:03d}",
                    minx=minx,
                    miny=miny,
                    maxx=maxx,
                    maxy=maxy,
                    level=0,
                    expected_candidates=None,
                    include_max_x=math.isclose(maxx, grid_maxx),
                    include_max_y=math.isclose(maxy, grid_maxy),
                )
            )

    planned: list[Tile] = []
    queue = list(base_tiles)
    while queue:
        tile = queue.pop(0)
        count = rtree_candidate_count(source, tile.bounds)
        if count == 0:
            continue
        can_split = (
            count is not None
            and count > MAX_RTREE_CANDIDATES_PER_TILE
            and tile.level < MAX_TILE_RECURSION_LEVEL
            and tile.width / 2.0 >= MIN_TILE_SIZE_M
            and tile.height / 2.0 >= MIN_TILE_SIZE_M
        )
        if can_split:
            queue[0:0] = split_tile(tile)
            continue
        planned.append(
            Tile(
                tile_id=tile.tile_id,
                minx=tile.minx,
                miny=tile.miny,
                maxx=tile.maxx,
                maxy=tile.maxy,
                level=tile.level,
                expected_candidates=count,
                include_max_x=tile.include_max_x,
                include_max_y=tile.include_max_y,
            )
        )

    planned.sort(key=lambda item: (item.miny, item.minx, item.level, item.tile_id))
    return planned, grid_bounds


def select_tiles_for_run(tiles: list[Tile]) -> list[Tile]:
    selected = list(tiles)
    if PROCESS_ONLY_TILE_IDS is not None:
        available = {tile.tile_id for tile in tiles}
        unknown = set(PROCESS_ONLY_TILE_IDS) - available
        if unknown:
            raise ValueError(f"Unknown tile IDs: {sorted(unknown)}")
        selected = [tile for tile in tiles if tile.tile_id in PROCESS_ONLY_TILE_IDS]
    if MAX_TILES_THIS_RUN is not None:
        if MAX_TILES_THIS_RUN <= 0:
            raise ValueError("MAX_TILES_THIS_RUN must be positive or None")
        selected = selected[:MAX_TILES_THIS_RUN]
    return selected


# =============================================================================
# GEOMETRY PREPARATION AND SPATIAL PREDICATES
# =============================================================================


def polygon_parts(geometry: Any) -> list[Polygon]:
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return [part for part in geometry.geoms if not part.is_empty]
    if isinstance(geometry, GeometryCollection):
        parts: list[Polygon] = []
        for child in geometry.geoms:
            parts.extend(polygon_parts(child))
        return parts
    return []


def polygonal_only(geometry: Any) -> Polygon | MultiPolygon | None:
    parts = polygon_parts(geometry)
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    candidate = MultiPolygon(parts)
    if candidate.is_valid:
        return candidate
    merged = shapely.union_all(np.asarray(parts, dtype=object))
    merged_parts = polygon_parts(merged)
    if not merged_parts:
        return None
    return merged_parts[0] if len(merged_parts) == 1 else MultiPolygon(merged_parts)


def repair_polygon_for_test(geometry: Any) -> Polygon | MultiPolygon | None:
    if geometry is None or geometry.is_empty:
        return None
    geometry = shapely.force_2d(geometry)
    if geometry.is_valid:
        return polygonal_only(geometry)
    try:
        try:
            repaired = shapely.make_valid(
                geometry,
                method="structure",
                keep_collapsed=False,
            )
        except TypeError:
            repaired = shapely.make_valid(geometry)
    except Exception:
        repaired = geometry.buffer(0)
    result = polygonal_only(repaired)
    if result is None or result.is_empty:
        return None
    if not result.is_valid:
        try:
            result = polygonal_only(result.buffer(0))
        except Exception:
            return None
    return result


def prepare_test_geometries(
    geometries: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray, int]:
    prepared: list[Any] = []
    usable_indices: list[int] = []
    repaired_count = 0
    for index, geometry in enumerate(geometries):
        if geometry is None or geometry.is_empty:
            continue
        was_invalid = not geometry.is_valid
        candidate = repair_polygon_for_test(geometry)
        if candidate is None:
            continue
        if was_invalid:
            repaired_count += 1
        prepared.append(candidate)
        usable_indices.append(index)
    return (
        np.asarray(prepared, dtype=object),
        np.asarray(usable_indices, dtype=np.int64),
        repaired_count,
    )


def owner_mask(frame: gpd.GeoDataFrame, tile: Tile) -> np.ndarray:
    if frame.empty:
        return np.asarray([], dtype=bool)
    geometry_array = np.asarray(frame.geometry.array, dtype=object)
    valid = np.asarray(
        [geometry is not None and not geometry.is_empty for geometry in geometry_array],
        dtype=bool,
    )
    result = np.zeros(len(frame), dtype=bool)
    if not np.any(valid):
        return result
    points = shapely.point_on_surface(geometry_array[valid])
    x = np.asarray(shapely.get_x(points), dtype=float)
    y = np.asarray(shapely.get_y(points), dtype=float)
    x_upper = x <= tile.maxx if tile.include_max_x else x < tile.maxx
    y_upper = y <= tile.maxy if tile.include_max_y else y < tile.maxy
    owned = (
        (x >= tile.minx)
        & x_upper
        & (y >= tile.miny)
        & y_upper
        & np.isfinite(x)
        & np.isfinite(y)
    )
    result[np.flatnonzero(valid)] = owned
    return result


def query_tree_hits(tree: shapely.STRtree | None, source_geometries: np.ndarray) -> np.ndarray:
    hits = np.zeros(len(source_geometries), dtype=bool)
    if tree is None or len(source_geometries) == 0:
        return hits
    valid_indices = np.asarray(
        [
            index
            for index, geometry in enumerate(source_geometries)
            if geometry is not None and not geometry.is_empty
        ],
        dtype=np.int64,
    )
    if len(valid_indices) == 0:
        return hits
    valid_geometries = source_geometries[valid_indices]
    pairs = np.asarray(tree.query(valid_geometries, predicate="intersects"))
    if pairs.size == 0:
        return hits
    if pairs.ndim == 1:
        # This form is returned only for a scalar query, which is not expected here,
        # but retaining the branch makes the helper robust across Shapely versions.
        hits[valid_indices[0]] = True
        return hits
    matched_local = np.unique(pairs[0])
    hits[valid_indices[matched_local]] = True
    return hits


def transform_bounds(
    bounds: tuple[float, float, float, float],
    source_crs: pyproj.CRS,
    target_crs: pyproj.CRS,
) -> tuple[float, float, float, float]:
    if source_crs == target_crs:
        return bounds
    transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)
    transformed = transformer.transform_bounds(*bounds, densify_pts=21)
    return tuple(map(float, transformed))


def read_bbox_geometries(
    layer_info: VectorLayerInfo,
    bbox_in_target_crs: tuple[float, float, float, float],
    target_crs: pyproj.CRS,
) -> np.ndarray:
    if not bbox_intersects(
        bbox_in_target_crs,
        transform_bounds(layer_info.bounds, layer_info.crs, target_crs),
    ):
        return np.asarray([], dtype=object)
    query_bbox = transform_bounds(bbox_in_target_crs, target_crs, layer_info.crs)
    frame = pyogrio.read_dataframe(
        layer_info.path,
        layer=layer_info.layer,
        columns=[],
        bbox=query_bbox,
        force_2d=True,
        use_arrow=USE_ARROW,
        on_invalid="warn",
    )
    if frame.empty:
        return np.asarray([], dtype=object)
    if frame.crs is None:
        raise ValueError(f"Layer {layer_info.layer!r} has no CRS during bbox read")
    if pyproj.CRS.from_user_input(frame.crs) != target_crs:
        frame = frame.to_crs(target_crs)
    prepared, _, _ = prepare_test_geometries(list(frame.geometry))
    return prepared


def load_gsw_reference(
    layer_info: VectorLayerInfo,
    target_crs: pyproj.CRS,
) -> tuple[np.ndarray, shapely.STRtree, dict[str, Any]]:
    with timed("Load and prepare complete GSW reference layer"):
        frame = pyogrio.read_dataframe(
            layer_info.path,
            layer=layer_info.layer,
            columns=[],
            force_2d=True,
            use_arrow=USE_ARROW,
            on_invalid="warn",
        )
        if frame.empty:
            raise RuntimeError("The GSW layer contains no geometries")
        if frame.crs is None:
            raise ValueError("The GSW layer has no CRS")
        if pyproj.CRS.from_user_input(frame.crs) != target_crs:
            frame = frame.to_crs(target_crs)
        geometries, _, repaired_count = prepare_test_geometries(list(frame.geometry))
        if len(geometries) == 0:
            raise RuntimeError("No usable polygon geometries remain in GSW")
        if GSW_BUFFER_M > 0:
            buffered = shapely.buffer(geometries, GSW_BUFFER_M)
            geometries, _, additional_repaired = prepare_test_geometries(buffered)
            repaired_count += additional_repaired
        tree = shapely.STRtree(geometries)
    report = {
        "input_features": len(frame),
        "prepared_geometries": len(geometries),
        "invalid_repaired": repaired_count,
        "buffer_m": GSW_BUFFER_M,
        "vertices": int(np.sum(shapely.get_num_coordinates(geometries), dtype=np.int64)),
    }
    del frame
    gc.collect()
    return geometries, tree, report


# =============================================================================
# BOOLEAN ACTIVITY MBTILES TESTS
# =============================================================================

WEB_MERCATOR_HALF_WORLD_M = 20_037_508.342789244
BOOLEAN_ACTIVITY_CRS = pyproj.CRS.from_epsg(3857)
_ALLOWED_MBTILES_ACTIVITY_MODES = {
    "auto",
    "alpha",
    "nonzero",
    "different_from_background",
    "stored_tile",
}
_ALLOWED_MBTILES_SCHEMES = {"tms", "xyz"}


@dataclass(frozen=True)
class ActivityTile:
    mask: np.ndarray
    bounds: tuple[float, float, float, float]
    transform: Any
    active_bounds: tuple[float, float, float, float] | None

    @property
    def cache_bytes(self) -> int:
        return int(self.mask.nbytes)


class BooleanActivityMBTiles:
    """Read Boolean raster activity from one MBTiles database lazily.

    Standard MBTiles row numbering is TMS. The public API always accepts XYZ
    y-indices because those are the natural indices derived from Web Mercator.
    """

    def __init__(self, path: Path, zoom: int) -> None:
        self.path = path
        self.zoom = int(zoom)
        self.connection: sqlite3.Connection | None = None
        self.metadata: dict[str, str] = {}
        self.scheme = "tms"
        self.image_format: str | None = None
        self.tile_width: int | None = None
        self.tile_height: int | None = None
        self.activity_mode = MBTILES_ACTIVITY_MODE.lower()
        self.background_rgba: tuple[int, int, int, int] | None = (
            None
            if MBTILES_BACKGROUND_RGBA is None
            else tuple(int(value) for value in MBTILES_BACKGROUND_RGBA)
        )
        self.tile_count = 0
        self.available_zooms: list[int] = []
        self.cache: OrderedDict[tuple[int, int], ActivityTile | None] = OrderedDict()
        self.cache_bytes = 0
        self.cache_limit_bytes = int(MBTILES_TILE_CACHE_MAX_MB * 1024 * 1024)
        self.db_tile_reads = 0
        self.decoded_tiles = 0
        self.cache_hits = 0
        self.missing_or_inactive_tiles = 0

    def __enter__(self) -> "BooleanActivityMBTiles":
        uri = f"file:{self.path.resolve().as_posix()}?mode=ro"
        self.connection = sqlite3.connect(uri, uri=True)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA query_only = ON")
        self.connection.execute("PRAGMA temp_store = MEMORY")
        self._inspect()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback_obj: Any) -> None:
        if self.connection is not None:
            self.connection.close()
        self.connection = None
        self.cache.clear()
        self.cache_bytes = 0

    def _table_names(self) -> set[str]:
        assert self.connection is not None
        return {
            str(row[0])
            for row in self.connection.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            )
        }

    def _inspect(self) -> None:
        assert self.connection is not None
        tables = self._table_names()
        if "tiles" not in tables:
            raise ValueError(f"MBTiles database has no tiles table: {self.path}")

        if "metadata" in tables:
            self.metadata = {
                str(row[0]).lower(): str(row[1])
                for row in self.connection.execute("SELECT name, value FROM metadata")
            }
        else:
            self.metadata = {}

        self.available_zooms = [
            int(row[0])
            for row in self.connection.execute(
                "SELECT DISTINCT zoom_level FROM tiles ORDER BY zoom_level"
            )
        ]
        if self.zoom not in self.available_zooms:
            raise ValueError(
                f"Configured Boolean activity zoom {self.zoom} is absent from "
                f"{self.path}. Available zooms: {self.available_zooms}"
            )

        self.tile_count = int(
            self.connection.execute(
                "SELECT COUNT(*) FROM tiles WHERE zoom_level = ?", (self.zoom,)
            ).fetchone()[0]
        )
        if self.tile_count == 0:
            raise RuntimeError(
                f"The MBTiles database contains no tiles at zoom {self.zoom}"
            )

        configured_scheme = (
            None if MBTILES_SCHEME is None else MBTILES_SCHEME.strip().lower()
        )
        metadata_scheme = self.metadata.get("scheme", "").strip().lower()
        self.scheme = configured_scheme or (
            metadata_scheme if metadata_scheme in _ALLOWED_MBTILES_SCHEMES else "tms"
        )
        if self.scheme not in _ALLOWED_MBTILES_SCHEMES:
            raise ValueError(f"Unsupported MBTiles scheme: {self.scheme!r}")

        metadata_format = self.metadata.get("format", "").strip().lower()
        if metadata_format in {"pbf", "mvt", "vector"}:
            raise ValueError(
                "The supplied MBTiles appears to contain vector tiles. This "
                "workflow requires raster PNG/JPEG/WebP Boolean activity tiles."
            )

        samples = self._sample_images(MBTILES_ENCODING_SAMPLE_TILES)
        if not samples:
            raise RuntimeError("No decodable raster tile was found in the MBTiles")
        widths = {image.width for image in samples}
        heights = {image.height for image in samples}
        if len(widths) != 1 or len(heights) != 1:
            raise ValueError(
                f"Sampled MBTiles images have inconsistent dimensions: "
                f"widths={sorted(widths)}, heights={sorted(heights)}"
            )
        self.tile_width = widths.pop()
        self.tile_height = heights.pop()
        self.image_format = samples[0].format or metadata_format or None

        if self.activity_mode not in _ALLOWED_MBTILES_ACTIVITY_MODES:
            raise ValueError(
                f"MBTILES_ACTIVITY_MODE must be one of "
                f"{sorted(_ALLOWED_MBTILES_ACTIVITY_MODES)}"
            )
        if self.activity_mode == "auto":
            self.activity_mode, inferred_background = self._infer_activity_encoding(
                samples
            )
            if self.background_rgba is None:
                self.background_rgba = inferred_background
        elif self.activity_mode == "different_from_background":
            if self.background_rgba is None:
                _, inferred_background = self._infer_activity_encoding(samples)
                self.background_rgba = inferred_background
                if self.background_rgba is None:
                    raise RuntimeError(
                        "Could not infer the background colour. Set "
                        "MBTILES_BACKGROUND_RGBA explicitly."
                    )

        for image in samples:
            image.close()

    def _sample_images(self, limit: int) -> list[Any]:
        """Decode tiles sampled across the complete zoom-level table.

        Sampling only the first few tiles can be misleading when those tiles are
        blank. Evenly spaced offsets make automatic activity-encoding detection
        much more likely to see both background and active pixels.
        """
        assert self.connection is not None
        images: list[Any] = []
        sample_count = min(max(1, int(limit)), self.tile_count)
        offsets = np.unique(
            np.linspace(0, self.tile_count - 1, sample_count, dtype=np.int64)
        )
        query = (
            "SELECT tile_data FROM tiles WHERE zoom_level = ? "
            "ORDER BY tile_column, tile_row LIMIT 1 OFFSET ?"
        )
        for offset in offsets:
            row = self.connection.execute(
                query,
                (self.zoom, int(offset)),
            ).fetchone()
            if row is None:
                continue
            try:
                image = Image.open(BytesIO(bytes(row[0])))
                image.load()
            except (UnidentifiedImageError, OSError, ValueError):
                continue
            images.append(image)
        return images

    @staticmethod
    def _infer_activity_encoding(
        images: Sequence[Any],
    ) -> tuple[str, tuple[int, int, int, int] | None]:
        alpha_zero = False
        alpha_positive = False
        native_grayscale = True
        colours: Counter[tuple[int, int, int, int]] = Counter()

        for image in images:
            native_grayscale &= image.mode in {"1", "L", "I", "I;16", "F"}
            rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
            alpha = rgba[:, :, 3]
            alpha_zero |= bool(np.any(alpha == 0))
            alpha_positive |= bool(np.any(alpha > 0))
            pixels = rgba.reshape(-1, 4)
            unique, counts = np.unique(pixels, axis=0, return_counts=True)
            for colour, count in zip(unique, counts, strict=True):
                colours[tuple(int(value) for value in colour)] += int(count)

        if alpha_zero and alpha_positive:
            return "alpha", None
        if native_grayscale:
            return "nonzero", None
        if len(colours) > 1:
            black = (0, 0, 0, 255)
            if black in colours:
                return "nonzero", None
            return "different_from_background", colours.most_common(1)[0][0]
        if len(colours) == 1:
            only_colour = next(iter(colours))
            if only_colour in {(0, 0, 0, 0), (0, 0, 0, 255)}:
                return "nonzero", None
            raise RuntimeError(
                "Boolean activity tile encoding is ambiguous because the sampled "
                f"tiles contain only one opaque colour {only_colour}. Set "
                "MBTILES_ACTIVITY_MODE and, where needed, "
                "MBTILES_BACKGROUND_RGBA explicitly."
            )
        raise RuntimeError("Could not infer the Boolean activity tile encoding")

    def describe(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "zoom": self.zoom,
            "available_zooms": self.available_zooms,
            "scheme": self.scheme,
            "format": self.image_format,
            "tile_count": self.tile_count,
            "tile_width": self.tile_width,
            "tile_height": self.tile_height,
            "activity_mode": self.activity_mode,
            "activity_threshold": MBTILES_ACTIVITY_THRESHOLD,
            "background_rgba": (
                None if self.background_rgba is None else list(self.background_rgba)
            ),
            "all_touched": MBTILES_ALL_TOUCHED,
            "cache_limit_mb": MBTILES_TILE_CACHE_MAX_MB,
            "metadata": {
                key: value
                for key, value in self.metadata.items()
                if key in {
                    "name", "description", "format", "bounds", "center",
                    "minzoom", "maxzoom", "scheme", "type", "version",
                }
            },
        }

    def stats_snapshot(self) -> dict[str, int]:
        return {
            "db_tile_reads": self.db_tile_reads,
            "decoded_tiles": self.decoded_tiles,
            "cache_hits": self.cache_hits,
            "missing_or_inactive_tiles": self.missing_or_inactive_tiles,
        }

    def _database_row(self, y_xyz: int) -> int:
        if self.scheme == "xyz":
            return int(y_xyz)
        return (1 << self.zoom) - 1 - int(y_xyz)

    def _decode_activity_mask(self, image: Any) -> np.ndarray:
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        alpha = rgba[:, :, 3]
        if self.activity_mode == "alpha":
            active = alpha > MBTILES_ACTIVITY_THRESHOLD
        elif self.activity_mode == "nonzero":
            active = (alpha > 0) & np.any(
                rgba[:, :, :3] > MBTILES_ACTIVITY_THRESHOLD,
                axis=2,
            )
        elif self.activity_mode == "different_from_background":
            if self.background_rgba is None:
                raise RuntimeError("No MBTiles background colour is configured")
            background = np.asarray(self.background_rgba, dtype=np.uint8)
            active = np.any(rgba != background, axis=2)
        elif self.activity_mode == "stored_tile":
            active = np.ones(rgba.shape[:2], dtype=bool)
        else:
            raise RuntimeError(f"Unsupported activity mode: {self.activity_mode}")
        return np.asarray(active, dtype=bool)

    @staticmethod
    def _cache_value_bytes(value: ActivityTile | None) -> int:
        # Count a small amount for a cached missing/inactive tile so a worldwide
        # run cannot accumulate an unbounded number of None entries.
        return 64 if value is None else value.cache_bytes

    def _cache_put(self, key: tuple[int, int], value: ActivityTile | None) -> None:
        if key in self.cache:
            old = self.cache.pop(key)
            self.cache_bytes -= self._cache_value_bytes(old)
        self.cache[key] = value
        self.cache_bytes += self._cache_value_bytes(value)
        while self.cache and self.cache_bytes > self.cache_limit_bytes:
            _, removed = self.cache.popitem(last=False)
            self.cache_bytes -= self._cache_value_bytes(removed)

    def get_tile(self, x: int, y_xyz: int) -> ActivityTile | None:
        key = (int(x), int(y_xyz))
        if key in self.cache:
            self.cache_hits += 1
            value = self.cache.pop(key)
            self.cache[key] = value
            return value

        assert self.connection is not None
        row = self.connection.execute(
            "SELECT tile_data FROM tiles WHERE zoom_level = ? "
            "AND tile_column = ? AND tile_row = ?",
            (self.zoom, int(x), self._database_row(y_xyz)),
        ).fetchone()
        self.db_tile_reads += 1
        if row is None:
            self.missing_or_inactive_tiles += 1
            self._cache_put(key, None)
            return None

        try:
            image = Image.open(BytesIO(bytes(row[0])))
            image.load()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise RuntimeError(
                f"Could not decode MBTiles raster tile z={self.zoom}, x={x}, "
                f"y_xyz={y_xyz}: {type(exc).__name__}: {exc}"
            ) from exc
        try:
            mask = self._decode_activity_mask(image)
            self.decoded_tiles += 1
        finally:
            image.close()

        if mask.size == 0 or not np.any(mask):
            self.missing_or_inactive_tiles += 1
            self._cache_put(key, None)
            return None

        bounds = web_mercator_tile_bounds(self.zoom, x, y_xyz)
        height, width = mask.shape
        transform = transform_from_bounds(*bounds, width=width, height=height)
        rows, cols = np.nonzero(mask)
        pixel_width = (bounds[2] - bounds[0]) / width
        pixel_height = (bounds[3] - bounds[1]) / height
        active_bounds = (
            bounds[0] + int(cols.min()) * pixel_width,
            bounds[3] - (int(rows.max()) + 1) * pixel_height,
            bounds[0] + (int(cols.max()) + 1) * pixel_width,
            bounds[3] - int(rows.min()) * pixel_height,
        )
        tile = ActivityTile(
            mask=mask,
            bounds=bounds,
            transform=transform,
            active_bounds=active_bounds,
        )
        self._cache_put(key, tile)
        return tile


def web_mercator_tile_bounds(
    zoom: int,
    x: int,
    y_xyz: int,
) -> tuple[float, float, float, float]:
    count = 1 << int(zoom)
    span = (WEB_MERCATOR_HALF_WORLD_M * 2.0) / count
    minx = -WEB_MERCATOR_HALF_WORLD_M + int(x) * span
    maxx = minx + span
    maxy = WEB_MERCATOR_HALF_WORLD_M - int(y_xyz) * span
    miny = maxy - span
    return (minx, miny, maxx, maxy)


def web_mercator_tiles_for_bounds(
    bounds: tuple[float, float, float, float],
    zoom: int,
) -> Iterator[tuple[int, int]]:
    minx, miny, maxx, maxy = bounds
    world = WEB_MERCATOR_HALF_WORLD_M
    if maxx < -world or minx > world or maxy < -world or miny > world:
        return
    minx = max(-world, min(world, minx))
    maxx = max(-world, min(world, maxx))
    miny = max(-world, min(world, miny))
    maxy = max(-world, min(world, maxy))
    count = 1 << int(zoom)
    span = (world * 2.0) / count
    epsilon = span * 1e-12
    min_col = int(math.floor((minx + world) / span))
    max_col = int(math.floor((maxx + world - epsilon) / span))
    min_row = int(math.floor((world - maxy) / span))
    max_row = int(math.floor((world - miny - epsilon) / span))
    min_col = max(0, min(count - 1, min_col))
    max_col = max(0, min(count - 1, max_col))
    min_row = max(0, min(count - 1, min_row))
    max_row = max(0, min(count - 1, max_row))
    for y_xyz in range(min_row, max_row + 1):
        for x in range(min_col, max_col + 1):
            yield x, y_xyz


def activity_tile_overlaps_geometry(tile: ActivityTile, geometry: Any) -> bool:
    """Test only the small pixel window where geometry and activity may overlap."""
    if tile.active_bounds is None:
        return False
    geometry_bounds = tuple(map(float, geometry.bounds))
    if not bbox_intersects(geometry_bounds, tile.active_bounds):
        return False

    minx = max(geometry_bounds[0], tile.active_bounds[0], tile.bounds[0])
    miny = max(geometry_bounds[1], tile.active_bounds[1], tile.bounds[1])
    maxx = min(geometry_bounds[2], tile.active_bounds[2], tile.bounds[2])
    maxy = min(geometry_bounds[3], tile.active_bounds[3], tile.bounds[3])
    if maxx < minx or maxy < miny:
        return False

    height, width = tile.mask.shape
    pixel_width = (tile.bounds[2] - tile.bounds[0]) / width
    pixel_height = (tile.bounds[3] - tile.bounds[1]) / height

    # Include a one-pixel margin so all_touched behaviour is retained at exact
    # pixel boundaries despite floating-point rounding.
    col0 = max(0, int(math.floor((minx - tile.bounds[0]) / pixel_width)) - 1)
    col1 = min(
        width,
        int(math.ceil((maxx - tile.bounds[0]) / pixel_width)) + 1,
    )
    row0 = max(0, int(math.floor((tile.bounds[3] - maxy) / pixel_height)) - 1)
    row1 = min(
        height,
        int(math.ceil((tile.bounds[3] - miny) / pixel_height)) + 1,
    )
    if col1 <= col0 or row1 <= row0:
        return False

    local_activity = tile.mask[row0:row1, col0:col1]
    if local_activity.size == 0 or not np.any(local_activity):
        return False

    window = Window(col0, row0, col1 - col0, row1 - row0)
    local_geometry_mask = geometry_mask(
        [mapping(geometry)],
        out_shape=local_activity.shape,
        transform=window_transform(window, tile.transform),
        invert=True,
        all_touched=MBTILES_ALL_TOUCHED,
    )
    return bool(np.any(local_activity & local_geometry_mask))


def boolean_activity_hits(
    reader: BooleanActivityMBTiles,
    geometries: np.ndarray,
    geometry_crs: pyproj.CRS,
) -> tuple[np.ndarray, dict[str, int]]:
    hits = np.zeros(len(geometries), dtype=bool)
    before = reader.stats_snapshot()
    candidate_tile_tests = 0
    if len(geometries) == 0:
        return hits, {
            "db_tile_reads": 0,
            "decoded_tiles": 0,
            "cache_hits": 0,
            "missing_or_inactive_tiles": 0,
            "candidate_tile_tests": 0,
        }

    if geometry_crs == BOOLEAN_ACTIVITY_CRS:
        activity_geometries = np.asarray(geometries, dtype=object)
    else:
        activity_geometries = np.asarray(
            gpd.GeoSeries(list(geometries), crs=geometry_crs)
            .to_crs(BOOLEAN_ACTIVITY_CRS)
            .array,
            dtype=object,
        )

    for index, geometry in enumerate(activity_geometries):
        if geometry is None or geometry.is_empty:
            continue
        bounds = tuple(map(float, geometry.bounds))
        for x, y_xyz in web_mercator_tiles_for_bounds(bounds, reader.zoom):
            candidate_tile_tests += 1
            tile = reader.get_tile(x, y_xyz)
            if tile is None:
                continue
            if activity_tile_overlaps_geometry(tile, geometry):
                hits[index] = True
                break

    after = reader.stats_snapshot()
    stats = {key: int(after[key] - before[key]) for key in before}
    stats["candidate_tile_tests"] = candidate_tile_tests
    return hits, stats


# =============================================================================
# SOURCE AND CHECKPOINT I/O
# =============================================================================


def read_source_tile(source: VectorLayerInfo, tile: Tile) -> gpd.GeoDataFrame:
    frame = pyogrio.read_dataframe(
        source.path,
        layer=source.layer,
        columns=[],
        bbox=tile.bounds,
        force_2d=True,
        fid_as_index=True,
        use_arrow=USE_ARROW,
        on_invalid="warn",
    )
    if frame.crs is None:
        raise ValueError(f"Source tile {tile.tile_id} was read without a CRS")
    read_crs = pyproj.CRS.from_user_input(frame.crs)
    if read_crs != source.crs:
        raise ValueError(
            f"Unexpected source CRS during tile read: {read_crs} != {source.crs}"
        )
    return frame


def write_frame(
    frame: gpd.GeoDataFrame,
    path: Path,
    layer: str,
    *,
    append: bool,
) -> None:
    if frame.empty:
        return
    options = {"SPATIAL_INDEX": "YES" if CREATE_SPATIAL_INDEX else "NO"}
    pyogrio.write_dataframe(
        frame,
        path,
        layer=layer,
        driver="GPKG",
        append=append,
        promote_to_multi=True,
        layer_options=options,
        use_arrow=USE_ARROW,
    )


def tile_status_frame(tile: Tile, report: dict[str, Any], crs: pyproj.CRS) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "tile_id": [tile.tile_id],
            "level": [tile.level],
            "expected_candidates": [tile.expected_candidates],
            "source_read": [report.get("source_read", 0)],
            "source_owned": [report.get("source_owned", 0)],
            "kept_total": [report.get("kept_total", 0)],
            "kept_svp": [report.get("kept_svp", 0)],
            "kept_gsw_boolean_activity": [report.get("kept_gsw_boolean_activity", 0)],
            "ais_fallback_candidates": [report.get("ais_fallback_candidates", 0)],
            "elapsed_seconds": [report.get("elapsed_seconds", 0.0)],
            "geometry": [tile.core],
        },
        geometry="geometry",
        crs=crs,
    )


def write_tile_checkpoint(
    tile: Tile,
    kept: gpd.GeoDataFrame,
    ais_candidates: gpd.GeoDataFrame,
    status: gpd.GeoDataFrame,
) -> None:
    final_path = tile.checkpoint_path
    temp_path = final_path.with_name(final_path.stem + ".tmp.gpkg")
    if temp_path.exists():
        temp_path.unlink()
    first_layer = True
    try:
        if not kept.empty:
            write_frame(kept, temp_path, CHECKPOINT_LAYER, append=False)
            first_layer = False
        if not ais_candidates.empty:
            write_frame(
                ais_candidates,
                temp_path,
                AIS_CANDIDATE_LAYER,
                append=not first_layer,
            )
            first_layer = False
        write_frame(status, temp_path, CHECKPOINT_STATUS_LAYER, append=not first_layer)
        if not temp_path.exists():
            raise RuntimeError(f"No checkpoint file was created for {tile.tile_id}")
        os.replace(temp_path, final_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def available_layers(path: Path) -> set[str]:
    return {str(row[0]) for row in pyogrio.list_layers(path)}


# =============================================================================
# TILE FILTERING LOGIC
# =============================================================================


def process_tile(
    tile: Tile,
    source: VectorLayerInfo,
    svp: VectorLayerInfo,
    gsw_tree: shapely.STRtree,
    activity_reader: BooleanActivityMBTiles,
) -> dict[str, Any]:
    started = time.perf_counter()
    log(
        f"{tile.tile_id}: bounds={tile.bounds}; level={tile.level}; "
        f"RTree candidates={tile.expected_candidates if tile.expected_candidates is not None else 'unknown'}"
    )

    with timed(f"{tile.tile_id}: read source bbox"):
        frame = read_source_tile(source, tile)
    source_read = len(frame)

    if frame.empty:
        owned_frame = frame
    else:
        owned_frame = frame.loc[owner_mask(frame, tile)].copy()
    del frame

    source_owned = len(owned_frame)
    report: dict[str, Any] = {
        "tile": asdict(tile),
        "status": "complete",
        "source_read": source_read,
        "source_owned": source_owned,
        "source_empty_or_unusable": 0,
        "source_invalid_repaired_for_tests": 0,
        "svp_reference_features_read": 0,
        "gsw_candidates": 0,
        "boolean_activity_candidates": 0,
        "mbtiles_db_tile_reads": 0,
        "mbtiles_decoded_tiles": 0,
        "mbtiles_cache_hits": 0,
        "mbtiles_missing_or_inactive_tiles": 0,
        "mbtiles_candidate_tile_tests": 0,
        "kept_svp": 0,
        "rejected_no_gsw": 0,
        "kept_gsw_boolean_activity": 0,
        "ais_fallback_candidates": 0,
        "kept_total": 0,
    }

    def empty_kept_frame() -> gpd.GeoDataFrame:
        data: dict[str, Any] = {
            "source_fid": pd.Series(dtype="int64"),
            "evidence": pd.Series(dtype="object"),
            "geometry": gpd.GeoSeries([], crs=source.crs),
        }
        if WRITE_BOOLEAN_EVIDENCE_FIELDS:
            data.update(
                {
                    "svp_hit": pd.Series(dtype="bool"),
                    "gsw_hit": pd.Series(dtype="bool"),
                    "boolean_activity_hit": pd.Series(dtype="bool"),
                    "ais_tested": pd.Series(dtype="bool"),
                    "ais_hit": pd.Series(dtype="bool"),
                    "ais_point_count": pd.Series(dtype="int32"),
                }
            )
        return gpd.GeoDataFrame(data, geometry="geometry", crs=source.crs)

    def empty_ais_candidate_frame() -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            {
                "source_fid": pd.Series(dtype="int64"),
                "geometry": gpd.GeoSeries([], crs=source.crs),
            },
            geometry="geometry",
            crs=source.crs,
        )

    kept_frame = empty_kept_frame()
    ais_candidate_frame = empty_ais_candidate_frame()

    if not owned_frame.empty:
        original_geometries = np.asarray(owned_frame.geometry.array, dtype=object)
        test_geometries, usable_indices, repaired_count = prepare_test_geometries(
            original_geometries
        )
        report["source_invalid_repaired_for_tests"] = repaired_count
        report["source_empty_or_unusable"] = source_owned - len(usable_indices)

        if len(usable_indices):
            usable_original = original_geometries[usable_indices]
            usable_fids = np.asarray(owned_frame.index, dtype=np.int64)[usable_indices]
            test_bounds = tuple(map(float, shapely.total_bounds(test_geometries)))

            with timed(f"{tile.tile_id}: read local SVP reference and test overlaps"):
                svp_geometries = read_bbox_geometries(svp, test_bounds, source.crs)
                report["svp_reference_features_read"] = len(svp_geometries)
                svp_tree = (
                    shapely.STRtree(svp_geometries) if len(svp_geometries) else None
                )
                svp_hits = query_tree_hits(svp_tree, test_geometries)
            report["kept_svp"] = int(np.count_nonzero(svp_hits))

            non_svp_indices = np.flatnonzero(~svp_hits)
            gsw_hits = np.zeros(len(test_geometries), dtype=bool)
            if len(non_svp_indices):
                with timed(f"{tile.tile_id}: test GSW overlaps"):
                    local_gsw = query_tree_hits(
                        gsw_tree,
                        test_geometries[non_svp_indices],
                    )
                    gsw_hits[non_svp_indices] = local_gsw
            report["gsw_candidates"] = int(np.count_nonzero((~svp_hits) & gsw_hits))
            report["rejected_no_gsw"] = int(
                np.count_nonzero((~svp_hits) & (~gsw_hits))
            )

            activity_hits_all = np.zeros(len(test_geometries), dtype=bool)
            activity_indices = np.flatnonzero((~svp_hits) & gsw_hits)
            report["boolean_activity_candidates"] = len(activity_indices)
            if len(activity_indices):
                with timed(
                    f"{tile.tile_id}: test Boolean activity MBTiles overlap"
                ):
                    local_activity_hits, activity_stats = boolean_activity_hits(
                        activity_reader,
                        test_geometries[activity_indices],
                        source.crs,
                    )
                    activity_hits_all[activity_indices] = local_activity_hits
                    report["mbtiles_db_tile_reads"] = activity_stats["db_tile_reads"]
                    report["mbtiles_decoded_tiles"] = activity_stats["decoded_tiles"]
                    report["mbtiles_cache_hits"] = activity_stats["cache_hits"]
                    report["mbtiles_missing_or_inactive_tiles"] = activity_stats[
                        "missing_or_inactive_tiles"
                    ]
                    report["mbtiles_candidate_tile_tests"] = activity_stats[
                        "candidate_tile_tests"
                    ]

            immediate_keep_mask = svp_hits | (
                (~svp_hits) & gsw_hits & activity_hits_all
            )
            ais_candidate_mask = (~svp_hits) & gsw_hits & (~activity_hits_all)

            report["kept_gsw_boolean_activity"] = int(
                np.count_nonzero((~svp_hits) & gsw_hits & activity_hits_all)
            )
            report["ais_fallback_candidates"] = int(
                np.count_nonzero(ais_candidate_mask)
            )
            report["kept_total"] = int(np.count_nonzero(immediate_keep_mask))

            kept_indices = np.flatnonzero(immediate_keep_mask)
            if len(kept_indices):
                evidence = np.where(
                    svp_hits[kept_indices],
                    "SVP",
                    "GSW+BOOLEAN_ACTIVITY_2025",
                )
                data: dict[str, Any] = {
                    "source_fid": usable_fids[kept_indices],
                    "evidence": evidence,
                    "geometry": gpd.GeoSeries(
                        usable_original[kept_indices],
                        crs=source.crs,
                    ),
                }
                if WRITE_BOOLEAN_EVIDENCE_FIELDS:
                    data["svp_hit"] = svp_hits[kept_indices]
                    data["gsw_hit"] = gsw_hits[kept_indices]
                    data["boolean_activity_hit"] = activity_hits_all[kept_indices]
                    data["ais_tested"] = np.zeros(len(kept_indices), dtype=bool)
                    data["ais_hit"] = np.zeros(len(kept_indices), dtype=bool)
                    data["ais_point_count"] = np.zeros(
                        len(kept_indices), dtype=np.int32
                    )
                kept_frame = gpd.GeoDataFrame(
                    data,
                    geometry="geometry",
                    crs=source.crs,
                )

            candidate_indices = np.flatnonzero(ais_candidate_mask)
            if len(candidate_indices):
                ais_candidate_frame = gpd.GeoDataFrame(
                    {
                        "source_fid": usable_fids[candidate_indices],
                        "geometry": gpd.GeoSeries(
                            usable_original[candidate_indices],
                            crs=source.crs,
                        ),
                    },
                    geometry="geometry",
                    crs=source.crs,
                )

            del (
                svp_geometries,
                svp_tree,
                svp_hits,
                gsw_hits,
                activity_hits_all,
                test_geometries,
                original_geometries,
                usable_original,
            )

    elapsed = time.perf_counter() - started
    report["elapsed_seconds"] = elapsed
    report["completed"] = datetime.now().isoformat(timespec="seconds")

    status_frame = tile_status_frame(tile, report, source.crs)
    with timed(f"{tile.tile_id}: write atomic tile checkpoint"):
        write_tile_checkpoint(
            tile,
            kept_frame,
            ais_candidate_frame,
            status_frame,
        )
    report["checkpoint_path"] = str(tile.checkpoint_path)
    report["checkpoint_size_mb"] = tile.checkpoint_path.stat().st_size / (1024**2)

    del owned_frame, kept_frame, ais_candidate_frame, status_frame
    return report

# =============================================================================
# RAW AIS GEOPARQUET FALLBACK
# =============================================================================

WEB_MERCATOR_HALF_WORLD_M = 20_037_508.342789244
WEB_MERCATOR_MAX_LATITUDE = 85.0511287798066
AIS_TEST_CRS = pyproj.CRS.from_epsg(3857)


@dataclass(frozen=True)
class AISGridSpec:
    origin_x: float
    origin_y: float
    cell_size: float
    columns: int
    rows: int

    @property
    def cell_count(self) -> int:
        return self.columns * self.rows


def ais_grid_spec() -> AISGridSpec:
    span = WEB_MERCATOR_HALF_WORLD_M * 2.0
    count = int(math.ceil(span / AIS_SPATIAL_CELL_SIZE_M))
    return AISGridSpec(
        origin_x=-WEB_MERCATOR_HALF_WORLD_M,
        origin_y=-WEB_MERCATOR_HALF_WORLD_M,
        cell_size=AIS_SPATIAL_CELL_SIZE_M,
        columns=count,
        rows=count,
    )


def discover_ais_parquet_files() -> list[Path]:
    files: set[Path] = set()
    for pattern in AIS_PARQUET_PATTERNS:
        files.update(path for path in AIS_PARQUET_DIR.rglob(pattern) if path.is_file())
    result = sorted(files, key=lambda path: str(path).lower())
    if not result:
        raise RuntimeError(
            f"No AIS Parquet files found below {AIS_PARQUET_DIR} using "
            f"patterns {AIS_PARQUET_PATTERNS}"
        )
    return result


def ais_dataset_signature(files: Sequence[Path]) -> dict[str, Any]:
    digest = hashlib.sha256()
    total_size = 0
    newest_mtime_ns = 0
    for path in files:
        stat = path.stat()
        relative = str(path.relative_to(AIS_PARQUET_DIR)).replace("\\", "/")
        digest.update(relative.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        total_size += stat.st_size
        newest_mtime_ns = max(newest_mtime_ns, stat.st_mtime_ns)
    return {
        "directory": str(AIS_PARQUET_DIR.resolve()),
        "file_count": len(files),
        "total_size_bytes": total_size,
        "newest_mtime_ns": newest_mtime_ns,
        "digest": digest.hexdigest(),
    }


def candidate_checkpoint_signature(
    tiles: Sequence[Tile], manifest: dict[str, Any]
) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    total_size = 0
    for tile in tiles:
        if not tile.checkpoint_path.exists():
            continue
        if manifest.get("tiles", {}).get(tile.tile_id, {}).get("status") != "complete":
            continue
        stat = tile.checkpoint_path.stat()
        digest.update(tile.tile_id.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        count += 1
        total_size += stat.st_size
    return {
        "tile_checkpoint_count": count,
        "tile_checkpoint_size_bytes": total_size,
        "digest": digest.hexdigest(),
        "tile_run_signature": manifest.get("run_signature"),
    }


def resolve_column_name(
    names: Sequence[str],
    configured: str | None,
    alternatives: Sequence[str],
    label: str,
) -> str:
    by_lower = {name.lower(): name for name in names}
    candidates = ([configured] if configured else []) + list(alternatives)
    for candidate in candidates:
        match = by_lower.get(candidate.lower())
        if match is not None:
            return match
    raise ValueError(
        f"Could not find AIS {label} column. Available={list(names)}; "
        f"configured={configured!r}"
    )


def resolve_ais_columns(names: Sequence[str]) -> tuple[str, str]:
    longitude = resolve_column_name(
        names, AIS_LONGITUDE_COLUMN, ("lon", "longitude", "lng", "long"), "longitude"
    )
    latitude = resolve_column_name(
        names, AIS_LATITUDE_COLUMN, ("lat", "latitude"), "latitude"
    )
    if longitude == latitude:
        raise ValueError("AIS longitude and latitude resolved to the same column")
    return longitude, latitude


def open_ais_database() -> sqlite3.Connection:
    connection = sqlite3.connect(AIS_INDEX_DB, timeout=120.0)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA cache_size=-262144")
    return connection


def create_ais_database_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS candidates (
            candidate_id INTEGER PRIMARY KEY,
            tile_id TEXT NOT NULL,
            source_fid INTEGER NOT NULL UNIQUE,
            test_wkb BLOB NOT NULL,
            ais_count INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS cell_members (
            cell_id INTEGER NOT NULL,
            candidate_id INTEGER NOT NULL,
            PRIMARY KEY (cell_id, candidate_id)
        );
        CREATE INDEX IF NOT EXISTS idx_cell_members_cell ON cell_members(cell_id);
        CREATE INDEX IF NOT EXISTS idx_candidates_tile ON candidates(tile_id);
        CREATE TABLE IF NOT EXISTS processed_row_groups (
            file_path TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            file_mtime_ns INTEGER NOT NULL,
            row_group INTEGER NOT NULL,
            processed_at TEXT NOT NULL,
            PRIMARY KEY (file_path, file_size, file_mtime_ns, row_group)
        );
        """
    )
    connection.commit()


def metadata_get(connection: sqlite3.Connection, key: str) -> Any:
    row = connection.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
    return None if row is None else json.loads(str(row[0]))


def metadata_set(connection: sqlite3.Connection, key: str, value: Any) -> None:
    connection.execute(
        "INSERT INTO metadata(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, json.dumps(value, sort_keys=True)),
    )


def cells_for_bounds(
    bounds: tuple[float, float, float, float], grid: AISGridSpec
) -> list[int]:
    minx, miny, maxx, maxy = bounds
    world_max_x = grid.origin_x + grid.columns * grid.cell_size
    world_max_y = grid.origin_y + grid.rows * grid.cell_size
    minx = max(grid.origin_x, minx)
    miny = max(grid.origin_y, miny)
    maxx = min(world_max_x, maxx)
    maxy = min(world_max_y, maxy)
    if maxx < minx or maxy < miny:
        return []
    min_col = max(0, min(grid.columns - 1, int(math.floor((minx - grid.origin_x) / grid.cell_size))))
    max_col = max(0, min(grid.columns - 1, int(math.floor((maxx - grid.origin_x) / grid.cell_size))))
    min_row = max(0, min(grid.rows - 1, int(math.floor((miny - grid.origin_y) / grid.cell_size))))
    max_row = max(0, min(grid.rows - 1, int(math.floor((maxy - grid.origin_y) / grid.cell_size))))
    return [
        row * grid.columns + col
        for row in range(min_row, max_row + 1)
        for col in range(min_col, max_col + 1)
    ]


def build_or_reuse_ais_candidate_index(
    tiles: Sequence[Tile],
    manifest: dict[str, Any],
    source_crs: pyproj.CRS,
    ais_files_signature: dict[str, Any],
) -> tuple[sqlite3.Connection, dict[str, Any]]:
    checkpoint_signature = candidate_checkpoint_signature(tiles, manifest)
    signature_payload_value = {
        "processing_logic_version": PROCESSING_LOGIC_VERSION,
        "checkpoint_signature": checkpoint_signature,
        "ais_dataset_signature": ais_files_signature,
        "minimum_points": AIS_MIN_POINTS_PER_WATER,
        "cell_size_m": AIS_SPATIAL_CELL_SIZE_M,
        "longitude_column": AIS_LONGITUDE_COLUMN,
        "latitude_column": AIS_LATITUDE_COLUMN,
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload_value, sort_keys=True).encode("utf-8")
    ).hexdigest()

    if AIS_INDEX_DB.exists():
        connection = open_ais_database()
        create_ais_database_schema(connection)
        if (
            metadata_get(connection, "index_signature") == signature
            and bool(metadata_get(connection, "build_complete"))
        ):
            candidate_count = int(connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0])
            memberships = int(connection.execute("SELECT COUNT(*) FROM cell_members").fetchone()[0])
            return connection, {
                "reused": True,
                "candidate_count": candidate_count,
                "cell_memberships": memberships,
                "signature": signature,
            }
        connection.close()
        AIS_INDEX_DB.unlink()
        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(AIS_INDEX_DB) + suffix)
            if sidecar.exists():
                sidecar.unlink()

    connection = open_ais_database()
    create_ais_database_schema(connection)
    metadata_set(connection, "index_signature", signature)
    metadata_set(connection, "signature_payload", signature_payload_value)
    metadata_set(connection, "build_complete", False)
    connection.commit()

    grid = ais_grid_spec()
    candidate_count = 0
    memberships = 0
    removed = 0
    complete_tiles = [
        tile for tile in tiles
        if tile.checkpoint_path.exists()
        and manifest.get("tiles", {}).get(tile.tile_id, {}).get("status") == "complete"
    ]
    for number, tile in enumerate(complete_tiles, start=1):
        if AIS_CANDIDATE_LAYER not in available_layers(tile.checkpoint_path):
            continue
        frame = pyogrio.read_dataframe(
            tile.checkpoint_path,
            layer=AIS_CANDIDATE_LAYER,
            force_2d=True,
            use_arrow=USE_ARROW,
        )
        if frame.empty:
            continue
        if frame.crs is None:
            raise ValueError(f"AIS candidate layer in {tile.tile_id} has no CRS")
        if pyproj.CRS.from_user_input(frame.crs) != source_crs:
            raise ValueError(f"AIS candidate layer CRS mismatch in {tile.tile_id}")
        test_frame = frame if source_crs == AIS_TEST_CRS else frame.to_crs(AIS_TEST_CRS)
        connection.execute("BEGIN")
        try:
            for source_fid, geometry in zip(
                frame["source_fid"].to_numpy(dtype=np.int64),
                test_frame.geometry,
                strict=True,
            ):
                test_geometry = repair_polygon_for_test(geometry)
                if test_geometry is None:
                    removed += 1
                    continue
                wkb = bytes(shapely.to_wkb(test_geometry))
                cursor = connection.execute(
                    "INSERT INTO candidates(tile_id, source_fid, test_wkb, ais_count) "
                    "VALUES(?, ?, ?, 0)",
                    (tile.tile_id, int(source_fid), sqlite3.Binary(wkb)),
                )
                candidate_id = int(cursor.lastrowid)
                cells = cells_for_bounds(tuple(map(float, test_geometry.bounds)), grid)
                connection.executemany(
                    "INSERT OR IGNORE INTO cell_members(cell_id, candidate_id) VALUES(?, ?)",
                    [(cell_id, candidate_id) for cell_id in cells],
                )
                candidate_count += 1
                memberships += len(cells)
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        del frame, test_frame
        if number == 1 or number % 100 == 0 or number == len(complete_tiles):
            log(
                f"AIS candidate-index progress: {number:,}/{len(complete_tiles):,}; "
                f"candidates={candidate_count:,}; memberships={memberships:,}"
            )

    metadata_set(connection, "build_complete", True)
    metadata_set(connection, "candidate_count", candidate_count)
    metadata_set(connection, "cell_memberships", memberships)
    metadata_set(connection, "invalid_candidates_removed", removed)
    metadata_set(connection, "ais_scan_complete", False)
    connection.commit()
    return connection, {
        "reused": False,
        "candidate_count": candidate_count,
        "cell_memberships": memberships,
        "invalid_candidates_removed": removed,
        "signature": signature,
    }


@dataclass
class AISCellIndex:
    candidate_ids: np.ndarray
    tree: shapely.STRtree
    estimated_wkb_bytes: int


class AISCellIndexCache:
    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection
        self.maximum_bytes = int(AIS_CELL_CACHE_MAX_WKB_MB * 1024 * 1024)
        self.current_bytes = 0
        self.cache: OrderedDict[int, AISCellIndex | None] = OrderedDict()

    def get(self, cell_id: int) -> AISCellIndex | None:
        if cell_id in self.cache:
            value = self.cache.pop(cell_id)
            self.cache[cell_id] = value
            return value
        rows = self.connection.execute(
            "SELECT c.candidate_id, c.test_wkb FROM cell_members AS m "
            "JOIN candidates AS c ON c.candidate_id = m.candidate_id "
            "WHERE m.cell_id = ? AND c.ais_count < ?",
            (int(cell_id), AIS_MIN_POINTS_PER_WATER),
        ).fetchall()
        if not rows:
            self.cache[cell_id] = None
            return None
        candidate_ids = np.asarray([int(row[0]) for row in rows], dtype=np.int64)
        blobs = np.asarray([bytes(row[1]) for row in rows], dtype=object)
        geometries = np.asarray(shapely.from_wkb(blobs), dtype=object)
        estimated = int(sum(len(blob) for blob in blobs))
        entry = AISCellIndex(candidate_ids, shapely.STRtree(geometries), estimated)
        self.cache[cell_id] = entry
        self.current_bytes += estimated
        while self.maximum_bytes > 0 and self.current_bytes > self.maximum_bytes and len(self.cache) > 1:
            _, evicted = self.cache.popitem(last=False)
            if evicted is not None:
                self.current_bytes -= evicted.estimated_wkb_bytes
        return entry

def arrow_column_to_float64(batch: pa.RecordBatch, name: str) -> np.ndarray:
    index = batch.schema.get_field_index(name)
    if index < 0:
        raise KeyError(name)
    column = batch.column(index)
    try:
        values = column.to_numpy(zero_copy_only=False)
    except Exception:
        values = np.asarray(column.to_pylist(), dtype=np.float64)
    if np.ma.isMaskedArray(values):
        values = values.filled(np.nan)
    return np.asarray(values, dtype=np.float64)


def lonlat_to_web_mercator(
    longitude: np.ndarray, latitude: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    valid = (
        np.isfinite(longitude)
        & np.isfinite(latitude)
        & (longitude >= -180.0)
        & (longitude <= 180.0)
        & (latitude >= -WEB_MERCATOR_MAX_LATITUDE)
        & (latitude <= WEB_MERCATOR_MAX_LATITUDE)
    )
    if not np.any(valid):
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    lon = longitude[valid]
    lat = latitude[valid]
    radius = WEB_MERCATOR_HALF_WORLD_M / math.pi
    x = radius * np.deg2rad(lon)
    y = radius * np.log(np.tan(math.pi / 4.0 + np.deg2rad(lat) / 2.0))
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def point_cell_ids(x: np.ndarray, y: np.ndarray, grid: AISGridSpec) -> np.ndarray:
    columns = np.floor((x - grid.origin_x) / grid.cell_size).astype(np.int64)
    rows = np.floor((y - grid.origin_y) / grid.cell_size).astype(np.int64)
    np.clip(columns, 0, grid.columns - 1, out=columns)
    np.clip(rows, 0, grid.rows - 1, out=rows)
    return rows * grid.columns + columns


def row_group_lonlat_bounds(
    parquet_file: pq.ParquetFile,
    row_group: int,
    longitude_column: str,
    latitude_column: str,
) -> tuple[float, float, float, float] | None:
    lon_index = parquet_file.schema_arrow.get_field_index(longitude_column)
    lat_index = parquet_file.schema_arrow.get_field_index(latitude_column)
    if lon_index < 0 or lat_index < 0:
        return None
    metadata = parquet_file.metadata.row_group(row_group)
    lon_stats = metadata.column(lon_index).statistics
    lat_stats = metadata.column(lat_index).statistics
    if lon_stats is None or lat_stats is None:
        return None
    if not lon_stats.has_min_max or not lat_stats.has_min_max:
        return None
    try:
        return (
            float(lon_stats.min),
            float(lat_stats.min),
            float(lon_stats.max),
            float(lat_stats.max),
        )
    except Exception:
        return None


def candidate_cells_intersect_lonlat_bounds(
    bounds: tuple[float, float, float, float],
    grid: AISGridSpec,
    candidate_cell_lookup: np.ndarray,
) -> bool:
    min_lon, min_lat, max_lon, max_lat = bounds
    min_lon = max(-180.0, min_lon)
    max_lon = min(180.0, max_lon)
    min_lat = max(-WEB_MERCATOR_MAX_LATITUDE, min_lat)
    max_lat = min(WEB_MERCATOR_MAX_LATITUDE, max_lat)
    if max_lon < min_lon or max_lat < min_lat:
        return False
    x, y = lonlat_to_web_mercator(
        np.asarray([min_lon, max_lon], dtype=float),
        np.asarray([min_lat, max_lat], dtype=float),
    )
    if len(x) != 2:
        return False
    cells = cells_for_bounds((float(x[0]), float(y[0]), float(x[1]), float(y[1])), grid)
    return bool(cells and np.any(candidate_cell_lookup[np.asarray(cells, dtype=np.int64)]))


def processed_row_group_exists(
    connection: sqlite3.Connection, path: Path, row_group: int
) -> bool:
    stat = path.stat()
    row = connection.execute(
        "SELECT 1 FROM processed_row_groups WHERE file_path = ? AND file_size = ? "
        "AND file_mtime_ns = ? AND row_group = ?",
        (str(path.resolve()), stat.st_size, stat.st_mtime_ns, int(row_group)),
    ).fetchone()
    return row is not None


def apply_row_group_results(
    connection: sqlite3.Connection,
    path: Path,
    row_group: int,
    increments: dict[int, int],
    counts: np.ndarray,
) -> None:
    stat = path.stat()
    updates: list[tuple[int, int]] = []
    for candidate_id, increment in increments.items():
        old = int(counts[candidate_id])
        if old >= AIS_MIN_POINTS_PER_WATER or increment <= 0:
            continue
        new = min(AIS_MIN_POINTS_PER_WATER, old + int(increment))
        if new != old:
            updates.append((new, int(candidate_id)))
    connection.execute("BEGIN")
    try:
        if updates:
            connection.executemany(
                "UPDATE candidates SET ais_count = ? WHERE candidate_id = ?",
                updates,
            )
        connection.execute(
            "INSERT OR REPLACE INTO processed_row_groups "
            "(file_path, file_size, file_mtime_ns, row_group, processed_at) "
            "VALUES(?, ?, ?, ?, ?)",
            (
                str(path.resolve()),
                stat.st_size,
                stat.st_mtime_ns,
                int(row_group),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    for new, candidate_id in updates:
        counts[candidate_id] = new


def process_ais_record_batch(
    batch: pa.RecordBatch,
    longitude_column: str,
    latitude_column: str,
    grid: AISGridSpec,
    candidate_cell_lookup: np.ndarray,
    cache: AISCellIndexCache,
    counts: np.ndarray,
    pending: dict[int, int],
) -> dict[str, int]:
    longitude = arrow_column_to_float64(batch, longitude_column)
    latitude = arrow_column_to_float64(batch, latitude_column)
    rows_read = len(longitude)
    x, y = lonlat_to_web_mercator(longitude, latitude)
    valid_points = len(x)
    if valid_points == 0:
        return {"rows_read": rows_read, "valid_points": 0, "candidate_cell_points": 0, "exact_points_tested": 0, "matched_pairs": 0}

    cell_ids = point_cell_ids(x, y, grid)
    relevant = candidate_cell_lookup[cell_ids]
    if not np.any(relevant):
        return {"rows_read": rows_read, "valid_points": valid_points, "candidate_cell_points": 0, "exact_points_tested": 0, "matched_pairs": 0}

    x = x[relevant]
    y = y[relevant]
    cell_ids = cell_ids[relevant]
    order = np.argsort(cell_ids, kind="mergesort")
    x, y, cell_ids = x[order], y[order], cell_ids[order]
    unique_cells, starts = np.unique(cell_ids, return_index=True)
    stops = np.append(starts[1:], len(cell_ids))
    exact_points_tested = 0
    matched_pairs = 0

    for cell_id, start, stop in zip(unique_cells, starts, stops, strict=True):
        entry = cache.get(int(cell_id))
        if entry is None:
            continue
        active = np.fromiter(
            (
                int(counts[candidate_id]) + int(pending.get(int(candidate_id), 0))
                < AIS_MIN_POINTS_PER_WATER
                for candidate_id in entry.candidate_ids
            ),
            dtype=bool,
            count=len(entry.candidate_ids),
        )
        if not np.any(active):
            continue
        points = shapely.points(x[start:stop], y[start:stop])
        exact_points_tested += len(points)
        pairs = np.asarray(entry.tree.query(points, predicate="intersects"))
        if pairs.size == 0:
            continue
        tree_indices = np.asarray([int(pairs[0])], dtype=np.int64) if pairs.ndim == 1 else np.asarray(pairs[1], dtype=np.int64)
        matched_ids = entry.candidate_ids[tree_indices]
        unique_ids, pair_counts = np.unique(matched_ids, return_counts=True)
        for candidate_id, pair_count in zip(unique_ids, pair_counts, strict=True):
            candidate_id = int(candidate_id)
            remaining = AIS_MIN_POINTS_PER_WATER - (
                int(counts[candidate_id]) + int(pending.get(candidate_id, 0))
            )
            if remaining <= 0:
                continue
            increment = min(int(pair_count), remaining)
            pending[candidate_id] = int(pending.get(candidate_id, 0)) + increment
            matched_pairs += increment

    return {
        "rows_read": rows_read,
        "valid_points": valid_points,
        "candidate_cell_points": len(cell_ids),
        "exact_points_tested": exact_points_tested,
        "matched_pairs": matched_pairs,
    }


def reset_ais_scan(connection: sqlite3.Connection) -> None:
    connection.execute("UPDATE candidates SET ais_count = 0")
    connection.execute("DELETE FROM processed_row_groups")
    metadata_set(connection, "ais_scan_complete", False)
    connection.commit()


def scan_ais_geoparquet(
    connection: sqlite3.Connection,
    files: Sequence[Path],
    index_report: dict[str, Any],
) -> dict[str, Any]:
    candidate_count = int(connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0])
    if candidate_count == 0:
        metadata_set(connection, "ais_scan_complete", True)
        connection.commit()
        result = {
            "complete": True,
            "candidate_count": 0,
            "accepted_candidates": 0,
            "rejected_candidates": 0,
            "files_discovered": len(files),
            "files_processed_this_run": 0,
            "row_groups_processed_this_run": 0,
            "rows_read_this_run": 0,
            "index": index_report,
        }
        write_json_atomic(AIS_SCAN_REPORT_PATH, result)
        return result

    if not AIS_RESUME_PROCESSED_ROW_GROUPS:
        reset_ais_scan(connection)

    max_candidate_id = int(connection.execute("SELECT MAX(candidate_id) FROM candidates").fetchone()[0])
    counts = np.zeros(max_candidate_id + 1, dtype=np.int32)
    for candidate_id, ais_count in connection.execute("SELECT candidate_id, ais_count FROM candidates"):
        counts[int(candidate_id)] = int(ais_count)

    accepted_before_scan = int(np.count_nonzero(counts >= AIS_MIN_POINTS_PER_WATER))
    if accepted_before_scan == candidate_count:
        metadata_set(connection, "ais_scan_complete", True)
        connection.commit()
    if bool(metadata_get(connection, "ais_scan_complete")):
        accepted = int(np.count_nonzero(counts >= AIS_MIN_POINTS_PER_WATER))
        result = {
            "complete": True,
            "reused_completed_scan": True,
            "candidate_count": candidate_count,
            "accepted_candidates": accepted,
            "rejected_candidates": candidate_count - accepted,
            "files_discovered": len(files),
            "files_processed_this_run": 0,
            "row_groups_processed_this_run": 0,
            "rows_read_this_run": 0,
            "index": index_report,
        }
        write_json_atomic(AIS_SCAN_REPORT_PATH, result)
        return result

    grid = ais_grid_spec()
    candidate_cell_lookup = np.zeros(grid.cell_count, dtype=bool)
    for (cell_id,) in connection.execute("SELECT DISTINCT cell_id FROM cell_members"):
        candidate_cell_lookup[int(cell_id)] = True
    cache = AISCellIndexCache(connection)
    selected_files = list(files)
    if MAX_AIS_FILES_THIS_RUN is not None:
        selected_files = selected_files[:MAX_AIS_FILES_THIS_RUN]

    started = time.perf_counter()
    metrics = {
        "files_discovered": len(files),
        "files_processed_this_run": 0,
        "row_groups_processed_this_run": 0,
        "row_groups_skipped_from_resume": 0,
        "row_groups_skipped_by_statistics": 0,
        "rows_read_this_run": 0,
        "valid_points_this_run": 0,
        "candidate_cell_points_this_run": 0,
        "exact_points_tested_this_run": 0,
        "matched_pairs_capped_this_run": 0,
    }
    limit_reached = False
    all_candidates_accepted = False

    for file_number, path in enumerate(selected_files, start=1):
        parquet_file = pq.ParquetFile(path)
        longitude_column, latitude_column = resolve_ais_columns(parquet_file.schema_arrow.names)
        new_work = False
        log(
            f"AIS file {file_number:,}/{len(selected_files):,}: {path.name}; "
            f"row groups={parquet_file.num_row_groups:,}; "
            f"columns=({longitude_column}, {latitude_column})"
        )
        for row_group in range(parquet_file.num_row_groups):
            if MAX_AIS_ROW_GROUPS_THIS_RUN is not None and metrics["row_groups_processed_this_run"] >= MAX_AIS_ROW_GROUPS_THIS_RUN:
                limit_reached = True
                break
            if AIS_RESUME_PROCESSED_ROW_GROUPS and processed_row_group_exists(connection, path, row_group):
                metrics["row_groups_skipped_from_resume"] += 1
                continue
            if AIS_SKIP_ROW_GROUPS_WITHOUT_CANDIDATE_CELLS:
                stats_bounds = row_group_lonlat_bounds(parquet_file, row_group, longitude_column, latitude_column)
                if stats_bounds is not None and not candidate_cells_intersect_lonlat_bounds(stats_bounds, grid, candidate_cell_lookup):
                    apply_row_group_results(connection, path, row_group, {}, counts)
                    metrics["row_groups_processed_this_run"] += 1
                    metrics["row_groups_skipped_by_statistics"] += 1
                    new_work = True
                    continue

            pending: dict[int, int] = {}
            row_metrics = {"rows_read": 0, "valid_points": 0, "candidate_cell_points": 0, "exact_points_tested": 0, "matched_pairs": 0}
            for batch in parquet_file.iter_batches(
                batch_size=AIS_PARQUET_BATCH_ROWS,
                row_groups=[row_group],
                columns=[longitude_column, latitude_column],
                use_threads=True,
            ):
                batch_metrics = process_ais_record_batch(
                    batch,
                    longitude_column,
                    latitude_column,
                    grid,
                    candidate_cell_lookup,
                    cache,
                    counts,
                    pending,
                )
                for key in row_metrics:
                    row_metrics[key] += int(batch_metrics[key])
            apply_row_group_results(connection, path, row_group, pending, counts)
            metrics["row_groups_processed_this_run"] += 1
            metrics["rows_read_this_run"] += row_metrics["rows_read"]
            metrics["valid_points_this_run"] += row_metrics["valid_points"]
            metrics["candidate_cell_points_this_run"] += row_metrics["candidate_cell_points"]
            metrics["exact_points_tested_this_run"] += row_metrics["exact_points_tested"]
            metrics["matched_pairs_capped_this_run"] += row_metrics["matched_pairs"]
            new_work = True

            processed_groups = metrics["row_groups_processed_this_run"]
            if processed_groups == 1 or processed_groups % 25 == 0:
                accepted = int(np.count_nonzero(counts >= AIS_MIN_POINTS_PER_WATER))
                log(
                    f"AIS scan progress: row groups={processed_groups:,}; "
                    f"rows read={metrics['rows_read_this_run']:,}; "
                    f"accepted={accepted:,}/{candidate_count:,}; "
                    f"elapsed={format_duration(time.perf_counter() - started)}"
                )
            if int(np.count_nonzero(counts >= AIS_MIN_POINTS_PER_WATER)) == candidate_count:
                all_candidates_accepted = True
                log(
                    "Every AIS fallback candidate has reached the point threshold; "
                    "the remaining AIS files do not need to be scanned"
                )
                break
        if new_work:
            metrics["files_processed_this_run"] += 1
        if limit_reached or all_candidates_accepted:
            break

    complete = all_candidates_accepted or (
        MAX_AIS_FILES_THIS_RUN is None
        and MAX_AIS_ROW_GROUPS_THIS_RUN is None
        and not limit_reached
        and len(selected_files) == len(files)
    )
    metadata_set(connection, "ais_scan_complete", complete)
    connection.commit()
    accepted = int(np.count_nonzero(counts >= AIS_MIN_POINTS_PER_WATER))
    result = {
        "complete": complete,
        "reused_completed_scan": False,
        "candidate_count": candidate_count,
        "accepted_candidates": accepted,
        "rejected_candidates": candidate_count - accepted if complete else None,
        **metrics,
        "elapsed_seconds": time.perf_counter() - started,
        "index": index_report,
    }
    write_json_atomic(AIS_SCAN_REPORT_PATH, result)
    return result


def accepted_ais_candidates_for_tile(
    connection: sqlite3.Connection, tile_id: str
) -> dict[int, int]:
    return {
        int(source_fid): int(ais_count)
        for source_fid, ais_count in connection.execute(
            "SELECT source_fid, ais_count FROM candidates "
            "WHERE tile_id = ? AND ais_count >= ?",
            (tile_id, AIS_MIN_POINTS_PER_WATER),
        )
    }

# =============================================================================
# MANIFEST, SIGNATURE, AND FINAL CONSOLIDATION
# =============================================================================


def file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def signature_payload(
    source: VectorLayerInfo,
    svp: VectorLayerInfo,
    gsw: VectorLayerInfo,
    activity_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "processing_logic_version": PROCESSING_LOGIC_VERSION,
        "high_res": {**file_signature(source.path), "layer": source.layer},
        "svp": {**file_signature(svp.path), "layer": svp.layer},
        "gsw": {**file_signature(gsw.path), "layer": gsw.layer},
        "boolean_activity_mbtiles": {
            **file_signature(BOOLEAN_ACTIVITY_MBTILES),
            "zoom": activity_metadata["zoom"],
            "scheme": activity_metadata["scheme"],
            "format": activity_metadata["format"],
            "tile_count": activity_metadata["tile_count"],
            "tile_width": activity_metadata["tile_width"],
            "tile_height": activity_metadata["tile_height"],
            "activity_mode": activity_metadata["activity_mode"],
            "activity_threshold": activity_metadata["activity_threshold"],
            "background_rgba": activity_metadata["background_rgba"],
            "all_touched": activity_metadata["all_touched"],
        },
        "gsw_buffer_m": GSW_BUFFER_M,
        "base_tile_size_m": BASE_TILE_SIZE_M,
        "minimum_tile_size_m": MIN_TILE_SIZE_M,
        "max_rtree_candidates_per_tile": MAX_RTREE_CANDIDATES_PER_TILE,
        "max_tile_recursion_level": MAX_TILE_RECURSION_LEVEL,
    }


def payload_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        return {
            "created": datetime.now().isoformat(timespec="seconds"),
            "tiles": {},
        }
    try:
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Could not read manifest: {MANIFEST_PATH}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("tiles"), dict):
        raise ValueError(f"Invalid manifest structure: {MANIFEST_PATH}")
    return data


def append_final_batch(frames: list[gpd.GeoDataFrame], first_write: bool) -> bool:
    if not frames:
        return first_write
    combined = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True),
        geometry="geometry",
        crs=frames[0].crs,
    )
    write_frame(
        combined,
        FINAL_GPKG,
        FINAL_LAYER,
        append=not first_write,
    )
    frames.clear()
    del combined
    gc.collect()
    return False


def consolidate_checkpoints(
    tiles: list[Tile],
    manifest: dict[str, Any],
    source_crs: pyproj.CRS,
    ais_connection: sqlite3.Connection,
) -> dict[str, Any]:
    complete = [
        tile
        for tile in tiles
        if tile.checkpoint_path.exists()
        and manifest.get("tiles", {}).get(tile.tile_id, {}).get("status") == "complete"
    ]
    if REQUIRE_ALL_TILES_COMPLETE_FOR_FINAL and len(complete) != len(tiles):
        missing = [tile.tile_id for tile in tiles if tile not in complete]
        raise RuntimeError(
            f"Cannot build final output: {len(missing):,} planned tiles are incomplete. "
            f"First missing IDs: {missing[:20]}"
        )
    if not complete:
        raise RuntimeError("No completed checkpoints are available")
    if REQUIRE_AIS_SCAN_COMPLETE_FOR_FINAL and not bool(
        metadata_get(ais_connection, "ais_scan_complete")
    ):
        raise RuntimeError(
            "The raw AIS GeoParquet scan is incomplete. Resume the script until "
            "every AIS row group has been processed before building the final output."
        )

    if FINAL_GPKG.exists():
        if not OVERWRITE_FINAL_GPKG:
            raise FileExistsError(FINAL_GPKG)
        FINAL_GPKG.unlink()

    log(f"Consolidating {len(complete):,} completed tile checkpoints")
    started = time.perf_counter()
    first_write = True
    buffer: list[gpd.GeoDataFrame] = []
    output_features = 0
    ais_output_features = 0
    evidence_counts: dict[str, int] = {}

    for number, tile in enumerate(complete, start=1):
        layers = available_layers(tile.checkpoint_path)
        if CHECKPOINT_LAYER in layers:
            frame = pyogrio.read_dataframe(
                tile.checkpoint_path,
                layer=CHECKPOINT_LAYER,
                force_2d=True,
                use_arrow=USE_ARROW,
            )
            if not frame.empty:
                if pyproj.CRS.from_user_input(frame.crs) != source_crs:
                    raise ValueError(
                        f"Checkpoint {tile.tile_id} CRS does not match source CRS"
                    )
                output_features += len(frame)
                for value, count in frame["evidence"].value_counts().items():
                    evidence_counts[str(value)] = evidence_counts.get(str(value), 0) + int(count)
                buffer.append(frame)

        accepted = accepted_ais_candidates_for_tile(ais_connection, tile.tile_id)
        if accepted and AIS_CANDIDATE_LAYER in layers:
            candidate_frame = pyogrio.read_dataframe(
                tile.checkpoint_path,
                layer=AIS_CANDIDATE_LAYER,
                force_2d=True,
                use_arrow=USE_ARROW,
            )
            if not candidate_frame.empty:
                if pyproj.CRS.from_user_input(candidate_frame.crs) != source_crs:
                    raise ValueError(
                        f"AIS candidate checkpoint {tile.tile_id} CRS mismatch"
                    )
                keep = candidate_frame["source_fid"].isin(accepted)
                candidate_frame = candidate_frame.loc[keep].copy()
                if not candidate_frame.empty:
                    candidate_frame["evidence"] = "GSW+AIS_2025_10PLUS"
                    if WRITE_BOOLEAN_EVIDENCE_FIELDS:
                        candidate_frame["svp_hit"] = False
                        candidate_frame["gsw_hit"] = True
                        candidate_frame["boolean_activity_hit"] = False
                        candidate_frame["ais_tested"] = True
                        candidate_frame["ais_hit"] = True
                        candidate_frame["ais_point_count"] = np.asarray(
                            [accepted[int(fid)] for fid in candidate_frame["source_fid"]],
                            dtype=np.int32,
                        )
                    ordered = ["source_fid", "evidence"]
                    if WRITE_BOOLEAN_EVIDENCE_FIELDS:
                        ordered.extend(
                            [
                                "svp_hit",
                                "gsw_hit",
                                "boolean_activity_hit",
                                "ais_tested",
                                "ais_hit",
                                "ais_point_count",
                            ]
                        )
                    ordered.append("geometry")
                    candidate_frame = candidate_frame[ordered]
                    ais_output_features += len(candidate_frame)
                    output_features += len(candidate_frame)
                    evidence_counts["GSW+AIS_2025_10PLUS"] = (
                        evidence_counts.get("GSW+AIS_2025_10PLUS", 0)
                        + len(candidate_frame)
                    )
                    buffer.append(candidate_frame)

        if len(buffer) >= FINAL_WRITE_TILE_BATCH or number == len(complete):
            with timed(
                f"Flush final output through checkpoint {number:,}/{len(complete):,}"
            ):
                first_write = append_final_batch(buffer, first_write)

        if number == 1 or number % 100 == 0 or number == len(complete):
            log(
                f"Final consolidation progress: {number:,}/{len(complete):,}; "
                f"retained features seen={output_features:,}; AIS fallback kept="
                f"{ais_output_features:,}"
            )

    if first_write:
        raise RuntimeError("No retained inland-water features were found")

    if VACUUM_FINAL_GPKG:
        with timed("VACUUM final GeoPackage"):
            with sqlite3.connect(FINAL_GPKG) as connection:
                connection.execute("VACUUM")

    final_info = pyogrio.read_info(
        FINAL_GPKG,
        layer=FINAL_LAYER,
        force_feature_count=True,
        force_total_bounds=True,
    )
    return {
        "tile_checkpoints_consolidated": len(complete),
        "retained_features": int(final_info.get("features", output_features)),
        "retained_by_raw_ais": ais_output_features,
        "evidence_counts": evidence_counts,
        "bounds": (
            None
            if final_info.get("total_bounds") is None
            else list(map(float, final_info["total_bounds"]))
        ),
        "crs": str(final_info.get("crs")),
        "size_mb": FINAL_GPKG.stat().st_size / (1024**2),
        "elapsed_seconds": time.perf_counter() - started,
    }

# =============================================================================
# VALIDATION AND MAIN
# =============================================================================


def validate_settings() -> None:
    if Image is None:
        raise ImportError(
            "Boolean raster MBTiles decoding requires Pillow. Install it in the "
            "active environment, for example with: conda install pillow"
        )
    if pa is None or pq is None:
        raise ImportError(
            "The raw AIS fallback requires PyArrow. Install it in the active "
            "environment, for example with: conda install -c conda-forge pyarrow"
        )
    for path in (HIGH_RES_GPKG, SVP_GPKG, GSW_GPKG, BOOLEAN_ACTIVITY_MBTILES, AIS_PARQUET_DIR):
        if not path.exists():
            raise FileNotFoundError(path)
    if GSW_BUFFER_M < 0:
        raise ValueError("GSW_BUFFER_M cannot be negative")
    if BOOLEAN_ACTIVITY_ZOOM < 0:
        raise ValueError("BOOLEAN_ACTIVITY_ZOOM cannot be negative")
    if MBTILES_ACTIVITY_MODE.lower() not in _ALLOWED_MBTILES_ACTIVITY_MODES:
        raise ValueError(
            f"MBTILES_ACTIVITY_MODE must be one of "
            f"{sorted(_ALLOWED_MBTILES_ACTIVITY_MODES)}"
        )
    if (
        MBTILES_SCHEME is not None
        and MBTILES_SCHEME.lower() not in _ALLOWED_MBTILES_SCHEMES
    ):
        raise ValueError("MBTILES_SCHEME must be None, 'tms', or 'xyz'")
    if MBTILES_ENCODING_SAMPLE_TILES <= 0:
        raise ValueError("MBTILES_ENCODING_SAMPLE_TILES must be positive")
    if MBTILES_TILE_CACHE_MAX_MB < 0:
        raise ValueError("MBTILES_TILE_CACHE_MAX_MB cannot be negative")
    if BASE_TILE_SIZE_M <= 0 or MIN_TILE_SIZE_M <= 0:
        raise ValueError("Tile sizes must be positive")
    if MAX_RTREE_CANDIDATES_PER_TILE <= 0:
        raise ValueError("MAX_RTREE_CANDIDATES_PER_TILE must be positive")
    if FINAL_WRITE_TILE_BATCH <= 0:
        raise ValueError("FINAL_WRITE_TILE_BATCH must be positive")
    if AIS_MIN_POINTS_PER_WATER <= 0:
        raise ValueError("AIS_MIN_POINTS_PER_WATER must be positive")
    if AIS_PARQUET_BATCH_ROWS <= 0 or AIS_SPATIAL_CELL_SIZE_M <= 0:
        raise ValueError("AIS batch rows and spatial cell size must be positive")
    if AIS_CELL_CACHE_MAX_WKB_MB < 0:
        raise ValueError("AIS_CELL_CACHE_MAX_WKB_MB cannot be negative")
    if MAX_AIS_FILES_THIS_RUN is not None and MAX_AIS_FILES_THIS_RUN <= 0:
        raise ValueError("MAX_AIS_FILES_THIS_RUN must be positive or None")
    if MAX_AIS_ROW_GROUPS_THIS_RUN is not None and MAX_AIS_ROW_GROUPS_THIS_RUN <= 0:
        raise ValueError("MAX_AIS_ROW_GROUPS_THIS_RUN must be positive or None")



def main() -> None:
    overall_started = time.perf_counter()
    validate_settings()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TILE_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    log("Starting high-resolution inland-water vessel-activity subset")
    log(
        f"Python {sys.version.split()[0]}; GeoPandas {gpd.__version__}; "
        f"Shapely {shapely.__version__} / GEOS {shapely.geos_version_string}; "
        f"Pyogrio {pyogrio.__version__}; Rasterio {rasterio.__version__}; "
        f"Pillow {Image.__version__ if Image is not None else 'not installed'}; "
        f"PyArrow {pa.__version__ if pa is not None else 'not installed'}; Pyproj {pyproj.__version__}"
    )
    log(
        "Decision rule: SVP OR (GSW AND Boolean MBTiles activity) OR "
        f"(GSW AND no Boolean activity AND at least {AIS_MIN_POINTS_PER_WATER} raw AIS points)"
    )

    stage_banner(1, 7, "Inspect vector, Boolean MBTiles, and AIS GeoParquet inputs")
    source = inspect_vector_layer(HIGH_RES_GPKG, HIGH_RES_LAYER)
    svp = inspect_vector_layer(SVP_GPKG, SVP_LAYER)
    gsw = inspect_vector_layer(GSW_GPKG, GSW_LAYER)
    log_layer_info("High-resolution inland water", source)
    log_layer_info("SVP", svp)
    log_layer_info("GSW", gsw)

    if not source.crs.is_projected:
        raise ValueError("The high-resolution source CRS must be projected")
    source_unit = source.crs.axis_info[0].unit_name.lower()
    if source_unit not in {"metre", "meter"}:
        raise ValueError("The high-resolution source CRS must use metre units")
    if REQUIRE_HIGH_RES_RTREE and source.rtree_table is None:
        raise RuntimeError(
            "The high-resolution GeoPackage layer has no detectable RTree spatial "
            "index. Create a spatial index in QGIS before running this tiled process."
        )
    if REQUIRE_SVP_FAST_SPATIAL_FILTER and not (
        svp.fast_spatial_filter or svp.rtree_table is not None
    ):
        raise RuntimeError(
            "The SVP layer does not report fast spatial filtering and has no "
            "detectable RTree. Create a spatial index before processing."
        )

    ais_files = discover_ais_parquet_files()
    ais_files_meta = ais_dataset_signature(ais_files)
    log(
        f"AIS GeoParquet files: {ais_files_meta['file_count']:,}; total size="
        f"{ais_files_meta['total_size_bytes'] / (1024 ** 3):,.2f} GB; "
        f"directory={AIS_PARQUET_DIR}"
    )

    manifest = load_manifest()
    processed_now = skipped = failed = 0
    selected_tiles: list[Tile] = []
    all_tiles: list[Tile] = []
    complete_tiles: list[Tile] = []
    failed_tiles: list[str] = []

    with BooleanActivityMBTiles(
        BOOLEAN_ACTIVITY_MBTILES,
        BOOLEAN_ACTIVITY_ZOOM,
    ) as activity_reader:
        activity_meta = activity_reader.describe()
        log(
            f"Boolean activity MBTiles: zoom={activity_meta['zoom']}; "
            f"tiles={activity_meta['tile_count']:,}; scheme={activity_meta['scheme']}; "
            f"format={activity_meta['format']}"
        )
        log(
            f"Boolean tile dimensions: {activity_meta['tile_width']} x "
            f"{activity_meta['tile_height']}; activity mode="
            f"{activity_meta['activity_mode']}; background="
            f"{activity_meta['background_rgba']}; all_touched="
            f"{activity_meta['all_touched']}"
        )
        payload = signature_payload(source, svp, gsw, activity_meta)
        signature = payload_hash(payload)
        previous_signature = manifest.get("run_signature")
        existing_checkpoints = list(TILE_CHECKPOINT_DIR.glob("*.gpkg"))
        if previous_signature and previous_signature != signature and existing_checkpoints:
            raise RuntimeError(
                "Existing checkpoints were created with different vector/MBTiles "
                "inputs or settings. Use the new default OUTPUT_DIR or remove old "
                "checkpoints."
            )
        manifest["run_signature"] = signature
        manifest["run_signature_payload"] = payload
        manifest["updated"] = datetime.now().isoformat(timespec="seconds")
        manifest["inputs"] = {
            "high_res": asdict(source),
            "svp": asdict(svp),
            "gsw": asdict(gsw),
            "boolean_activity_mbtiles": activity_meta,
            "ais_geoparquet": ais_files_meta,
        }
        for key in ("high_res", "svp", "gsw"):
            manifest["inputs"][key]["path"] = str(manifest["inputs"][key]["path"])
            manifest["inputs"][key]["crs"] = str(manifest["inputs"][key]["crs"])
        write_json_atomic(MANIFEST_PATH, manifest)

        stage_banner(2, 7, "Build adaptive source tile plan and GSW index")
        with timed("Plan adaptive source tiles from the high-resolution RTree"):
            all_tiles, grid_bounds = plan_adaptive_tiles(source)
        selected_tiles = select_tiles_for_run(all_tiles)
        if not all_tiles:
            raise RuntimeError("The source RTree returned no non-empty processing tiles")
        log(f"Aligned processing grid bounds: {grid_bounds}")
        log(
            f"Adaptive tile plan: {len(all_tiles):,} non-empty tiles; "
            f"selected this run={len(selected_tiles):,}; maximum RTree candidates "
            f"per unsplit tile={MAX_RTREE_CANDIDATES_PER_TILE:,}"
        )
        _, gsw_tree, gsw_report = load_gsw_reference(gsw, source.crs)
        log(
            f"GSW reference prepared: {gsw_report['prepared_geometries']:,} "
            f"geometries; {gsw_report['vertices']:,} vertices; "
            f"buffer={gsw_report['buffer_m']:g} m"
        )
        manifest["tile_plan"] = {
            "count": len(all_tiles),
            "grid_bounds": list(grid_bounds),
            "selected_this_run": len(selected_tiles),
        }
        manifest["gsw_preparation"] = gsw_report
        write_json_atomic(MANIFEST_PATH, manifest)

        stage_banner(3, 7, "Process source tiles with short-circuit evidence tests")
        successful_durations: list[float] = []
        tile_phase_started = time.perf_counter()
        for number, tile in enumerate(selected_tiles, start=1):
            existing = manifest["tiles"].get(tile.tile_id, {})
            if (
                RESUME_COMPLETED_TILES
                and tile.checkpoint_path.exists()
                and existing.get("status") == "complete"
            ):
                skipped += 1
                if number == 1 or number % 100 == 0 or number == len(selected_tiles):
                    log(
                        f"Skipping complete tile {tile.tile_id}; plan progress="
                        f"{number:,}/{len(selected_tiles):,}"
                    )
                continue

            recent = successful_durations[-ETA_RECENT_TILES:]
            seconds_per_tile = sum(recent) / len(recent) if recent else None
            remaining = len(selected_tiles) - number + 1
            log("-" * 78)
            log(
                f"TILE {number:,}/{len(selected_tiles):,}: {tile.tile_id}; "
                f"elapsed={format_duration(time.perf_counter() - tile_phase_started)}; "
                f"rolling ETA={format_duration(seconds_per_tile * remaining if seconds_per_tile else None)}"
            )
            tile_started = time.perf_counter()
            try:
                tile_report = process_tile(tile, source, svp, gsw_tree, activity_reader)
            except Exception as exc:
                failed += 1
                error_text = traceback.format_exc()
                log(f"ERROR: tile {tile.tile_id} failed: {type(exc).__name__}: {exc}")
                with ERROR_LOG_PATH.open("a", encoding="utf-8") as error_file:
                    error_file.write(
                        f"\n{'=' * 78}\n{datetime.now().isoformat()} "
                        f"{tile.tile_id}\n{error_text}\n"
                    )
                manifest["tiles"][tile.tile_id] = {
                    "tile": asdict(tile),
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "failed": datetime.now().isoformat(timespec="seconds"),
                }
                write_json_atomic(MANIFEST_PATH, manifest)
                if not CONTINUE_ON_TILE_ERROR:
                    raise
                gc.collect()
                continue

            duration = time.perf_counter() - tile_started
            successful_durations.append(duration)
            processed_now += 1
            manifest["tiles"][tile.tile_id] = tile_report
            manifest["updated"] = datetime.now().isoformat(timespec="seconds")
            write_json_atomic(MANIFEST_PATH, manifest)
            recent = successful_durations[-ETA_RECENT_TILES:]
            seconds_per_tile = sum(recent) / len(recent)
            remaining = len(selected_tiles) - number
            log(
                f"TILE {tile.tile_id} COMPLETE in {format_duration(duration)}; "
                f"owned={tile_report['source_owned']:,}; immediate keep="
                f"{tile_report['kept_total']:,} (SVP={tile_report['kept_svp']:,}, "
                f"GSW+activity={tile_report['kept_gsw_boolean_activity']:,}); AIS fallback "
                f"candidates={tile_report['ais_fallback_candidates']:,}; "
                f"estimated remaining={format_duration(seconds_per_tile * remaining)}"
            )
            if GC_EVERY_TILES > 0 and processed_now % GC_EVERY_TILES == 0:
                gc.collect()

        del gsw_tree
        complete_tiles = [
            tile for tile in all_tiles
            if tile.checkpoint_path.exists()
            and manifest["tiles"].get(tile.tile_id, {}).get("status") == "complete"
        ]
        failed_tiles = [
            tile.tile_id for tile in all_tiles
            if manifest["tiles"].get(tile.tile_id, {}).get("status") == "failed"
        ]
        log(
            f"Tile stage finished: complete={len(complete_tiles):,}/{len(all_tiles):,}; "
            f"processed this run={processed_now:,}; skipped={skipped:,}; "
            f"failed this run={failed:,}"
        )

    if REQUIRE_ALL_TILES_COMPLETE_FOR_FINAL and len(complete_tiles) != len(all_tiles):
        raise RuntimeError(
            "The source-tile stage is incomplete. Resume the script before scanning "
            "the 120 GB AIS GeoParquet dataset."
        )

    ais_connection: sqlite3.Connection | None = None
    try:
        stage_banner(4, 7, "Build or reuse the raw-AIS fallback candidate index")
        with timed("Build/reuse AIS fallback SQLite spatial-cell index"):
            ais_connection, ais_index_report = build_or_reuse_ais_candidate_index(
                all_tiles,
                manifest,
                source.crs,
                ais_files_meta,
            )
        log(
            f"AIS fallback candidates: {ais_index_report['candidate_count']:,}; "
            f"cell memberships={ais_index_report['cell_memberships']:,}; "
            f"index reused={ais_index_report['reused']}"
        )

        stage_banner(5, 7, "Scan raw 2025 AIS GeoParquet and count exact point overlaps")
        with timed("Stream AIS GeoParquet row groups"):
            ais_scan_report = scan_ais_geoparquet(
                ais_connection,
                ais_files,
                ais_index_report,
            )
        log(
            f"AIS fallback accepted={ais_scan_report['accepted_candidates']:,}/"
            f"{ais_scan_report['candidate_count']:,}; complete="
            f"{ais_scan_report['complete']}"
        )
        if REQUIRE_AIS_SCAN_COMPLETE_FOR_FINAL and not ais_scan_report["complete"]:
            raise RuntimeError(
                "The AIS scan stopped before every file/row group was processed. "
                "Run the script again to resume; final consolidation is intentionally "
                "deferred."
            )

        stage_banner(6, 7, "Consolidate all retained source geometries")
        with timed("Build final lightweight inland-water subset"):
            consolidation = consolidate_checkpoints(
                all_tiles,
                manifest,
                source.crs,
                ais_connection,
            )
        log(
            f"Final subset: {consolidation['retained_features']:,} features; "
            f"size={consolidation['size_mb']:,.2f} MB; evidence="
            f"{consolidation['evidence_counts']}"
        )

        stage_banner(7, 7, "Write report and finish")
        complete_reports = [
            manifest["tiles"][tile.tile_id]
            for tile in all_tiles
            if manifest["tiles"].get(tile.tile_id, {}).get("status") == "complete"
        ]
        immediate_kept = sum(int(item.get("kept_total", 0)) for item in complete_reports)
        totals = {
            "source_features_read_with_bbox_duplicates": sum(
                int(item.get("source_read", 0)) for item in complete_reports
            ),
            "source_features_owned_and_tested": sum(
                int(item.get("source_owned", 0)) for item in complete_reports
            ),
            "kept_svp": sum(int(item.get("kept_svp", 0)) for item in complete_reports),
            "rejected_no_gsw": sum(
                int(item.get("rejected_no_gsw", 0)) for item in complete_reports
            ),
            "kept_gsw_boolean_activity": sum(
                int(item.get("kept_gsw_boolean_activity", 0)) for item in complete_reports
            ),
            "ais_fallback_candidates": sum(
                int(item.get("ais_fallback_candidates", 0)) for item in complete_reports
            ),
            "kept_raw_ais": int(ais_scan_report["accepted_candidates"]),
            "rejected_after_raw_ais": ais_scan_report.get("rejected_candidates"),
            "kept_total": immediate_kept + int(ais_scan_report["accepted_candidates"]),
            "mbtiles_db_tile_reads": sum(
                int(item.get("mbtiles_db_tile_reads", 0)) for item in complete_reports
            ),
            "mbtiles_decoded_tiles": sum(
                int(item.get("mbtiles_decoded_tiles", 0)) for item in complete_reports
            ),
            "mbtiles_cache_hits": sum(
                int(item.get("mbtiles_cache_hits", 0)) for item in complete_reports
            ),
            "mbtiles_candidate_tile_tests": sum(
                int(item.get("mbtiles_candidate_tile_tests", 0)) for item in complete_reports
            ),
        }
        report = {
            "created": datetime.now().isoformat(timespec="seconds"),
            "processing_logic_version": PROCESSING_LOGIC_VERSION,
            "decision_rule": (
                "SVP OR (GSW AND Boolean MBTiles activity) OR "
                f"(GSW AND no Boolean activity AND raw AIS points >= {AIS_MIN_POINTS_PER_WATER})"
            ),
            "inputs": manifest["inputs"],
            "settings": {
                "gsw_buffer_m": GSW_BUFFER_M,
                "boolean_activity_zoom": BOOLEAN_ACTIVITY_ZOOM,
                "mbtiles_scheme": activity_meta["scheme"],
                "mbtiles_activity_mode": activity_meta["activity_mode"],
                "mbtiles_activity_threshold": MBTILES_ACTIVITY_THRESHOLD,
                "mbtiles_background_rgba": activity_meta["background_rgba"],
                "mbtiles_all_touched": MBTILES_ALL_TOUCHED,
                "mbtiles_tile_cache_max_mb": MBTILES_TILE_CACHE_MAX_MB,
                "ais_min_points_per_water": AIS_MIN_POINTS_PER_WATER,
                "ais_longitude_column": AIS_LONGITUDE_COLUMN,
                "ais_latitude_column": AIS_LATITUDE_COLUMN,
                "ais_parquet_batch_rows": AIS_PARQUET_BATCH_ROWS,
                "ais_spatial_cell_size_m": AIS_SPATIAL_CELL_SIZE_M,
                "ais_cell_cache_max_wkb_mb": AIS_CELL_CACHE_MAX_WKB_MB,
                "base_tile_size_m": BASE_TILE_SIZE_M,
                "minimum_tile_size_m": MIN_TILE_SIZE_M,
                "max_rtree_candidates_per_tile": MAX_RTREE_CANDIDATES_PER_TILE,
                "max_tile_recursion_level": MAX_TILE_RECURSION_LEVEL,
            },
            "tile_plan_count": len(all_tiles),
            "completed_tile_count": len(complete_tiles),
            "failed_tile_ids": failed_tiles,
            "source_processing_this_run": {
                "selected": len(selected_tiles),
                "processed": processed_now,
                "skipped": skipped,
                "failed": failed,
            },
            "ais_scan": ais_scan_report,
            "totals": totals,
            "consolidation": consolidation,
            "outputs": {
                "final_gpkg": str(FINAL_GPKG),
                "final_layer": FINAL_LAYER,
                "checkpoint_directory": str(TILE_CHECKPOINT_DIR),
                "ais_candidate_index": str(AIS_INDEX_DB),
                "ais_scan_report": str(AIS_SCAN_REPORT_PATH),
                "manifest": str(MANIFEST_PATH),
                "error_log": str(ERROR_LOG_PATH) if ERROR_LOG_PATH.exists() else None,
            },
            "elapsed_seconds": time.perf_counter() - overall_started,
        }
        write_json_atomic(REPORT_PATH, report)
        log(f"Wrote report: {REPORT_PATH}")

        if not KEEP_TILE_CHECKPOINTS and not failed_tiles and ais_scan_report["complete"]:
            import shutil
            with timed("Remove tile checkpoints"):
                shutil.rmtree(TILE_CHECKPOINT_DIR)

        log(
            f"Completed in {format_duration(time.perf_counter() - overall_started)}; "
            f"final output={FINAL_GPKG}"
        )
    finally:
        if ais_connection is not None:
            ais_connection.close()


if __name__ == "__main__":
    main()
