# -*- coding: utf-8 -*-
"""
Expand an existing vessel-validated inland-water subset through the original
high-resolution inland-water polygons by iterative spatial connectivity.

A source polygon is added when it intersects, touches, or lies within the
configured metric tolerance of any already-selected source polygon. Newly added
polygons become the next frontier. Processing stops automatically when the
frontier is empty, which is equivalent to repeatedly running the overlap test
until an iteration adds zero polygons.

The implementation is a queue-based graph search rather than repeated full-layer
scans. It uses the original GeoPackage RTree to read only nearby candidates, so
unrelated inland-water polygons are never brought into later iterations.

The original source geometry is written unchanged to the final output. The
1-metre value is used only as an adjacency test; no output geometry is buffered.

Inputs configured below
-----------------------
* Original high-resolution inland water:
  inland_water_split.gpkg / inland_water_split
* Existing amalgamated vessel-validated subset:
  inland_water_vessel_subset_2025_mbtiles_ais10_merged.gpkg /
  inland_water_subset

Outputs
-------
* One GeoPackage containing seeds plus every iteratively connected source shape.
* A resumable SQLite state database and a JSON processing report.

Important
---------
The output can grow through an entire connected river/lake/canal network. That is
intentional: if seed A connects to B and B connects to C, C is retained even when
C is not directly within the tolerance of A.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence
import gc
import hashlib
import json
import math
import os
import sqlite3
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import pyproj
import shapely
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon


# =============================================================================
# USER SETTINGS
# =============================================================================

# Original high-resolution inland-water source.
SOURCE_GPKG = Path(
    r"C:\Users\Craig Pearce\Desktop\osm\original_files"
    r"\inland_water_split.gpkg"
)
SOURCE_LAYER = "inland_water_split"

# Existing vessel-validated amalgamated subset. Its source_fid values identify
# the initial seed polygons in SOURCE_GPKG.
SEED_GPKG = Path(
    r"C:\Users\Craig Pearce\Desktop\inland_water_vessel_subset_2025_mbtiles_ais10_work"
    r"\inland_water_vessel_subset_2025_mbtiles_ais10_merged.gpkg"
)
SEED_LAYER = "inland_water_subset"
SEED_SOURCE_FID_FIELD = "source_fid"
SEED_EVIDENCE_FIELD = "evidence"

# Final connected subset: one GeoPackage / one layer.
OUTPUT_GPKG = Path(
    r"C:\Users\Craig Pearce\Desktop"
    r"\inland_water_vessel_subset_2025_mbtiles_ais10_connected_1m.gpkg"
)
OUTPUT_LAYER = "inland_water_subset_connected"

# Restart state and report. Deleting this folder starts the expansion from zero.
WORK_DIR = Path(
    r"C:\Users\Craig Pearce\Desktop"
    r"\inland_water_vessel_subset_2025_mbtiles_ais10_connected_1m_work"
)
STATE_DB = WORK_DIR / "connectivity_state.sqlite"
REPORT_PATH = WORK_DIR / "connectivity_report.json"

# Adjacency is inclusive: intersect/touch OR distance <= this value.
CONNECTIVITY_TOLERANCE_M = 1.0

# Queue and I/O limits. Lower these if an individual frontier produces unusually
# heavy candidate reads; raise them cautiously when processing is stable.
FRONTIER_BATCH_SIZE = 250
SOURCE_FID_READ_BATCH = 5_000
SEED_ATTRIBUTE_READ_BATCH = 100_000
# Rows whose source_fid is NULL/non-numeric are resolved back to the original
# inland_water_split feature by comparing their unchanged seed geometry against
# nearby source geometries. This preserves those shapes as connectivity seeds.
RECOVER_MISSING_SEED_SOURCE_FIDS = True
SEED_FID_RECOVERY_BATCH = 2_000
# GeoPackage RTree bounds are stored at limited precision, so this tolerance is
# intentionally wider than the 1 m connectivity tolerance. Exact/topological
# geometry equality is still required before a source FID is accepted.
SEED_FID_RECOVERY_RTREE_BOUNDS_TOLERANCE_M = 10.0
FAIL_ON_UNRESOLVED_SEED_SOURCE_FID = True
OUTPUT_WRITE_BATCH = 10_000
SQLITE_IN_CLAUSE_BATCH = 800

# A safety valve for testing. None means continue until no reachable shapes remain.
MAX_FRONTIER_BATCHES_THIS_RUN: int | None = None
MAX_CONNECTION_DEPTH: int | None = None

# Geometry used for adjacency testing can be repaired when invalid. The geometry
# written to OUTPUT_GPKG is always reread unchanged from SOURCE_GPKG.
REPAIR_INVALID_FOR_ADJACENCY_TEST = True
FAIL_ON_MISSING_SOURCE_FID = True

# Resume / output behaviour.
RESUME_EXISTING_STATE = True
RESET_EXISTING_STATE = False
OVERWRITE_FINAL_OUTPUT = True
WRITE_FINAL_OUTPUT_WHEN_COMPLETE = True
WRITE_PARTIAL_OUTPUT = False

# GeoPackage / GDAL settings.
CREATE_SPATIAL_INDEX = True
USE_ARROW = False
VACUUM_FINAL_GPKG = False

# SQLite state settings. A negative cache_size is expressed in KiB.
STATE_SQLITE_CACHE_MB = 256
STATE_BUSY_TIMEOUT_MS = 60_000

# Progress reporting.
LOG_EVERY_FRONTIER_BATCHES = 1
PROCESSING_LOGIC_VERSION = "iterative_source_rtree_dwithin_v1"


# =============================================================================
# DATA MODELS AND BASIC HELPERS
# =============================================================================


@dataclass(frozen=True)
class SourceLayerInfo:
    path: Path
    layer: str
    crs: pyproj.CRS
    feature_count: int
    geometry_column: str
    fid_column: str
    rtree_table: str
    fast_spatial_filter: bool


@dataclass(frozen=True)
class FrontierRow:
    source_fid: int
    depth: int
    root_seed_fid: int
    root_evidence: str


@dataclass
class RunCounters:
    frontier_batches: int = 0
    frontier_features_processed: int = 0
    rtree_bbox_hits: int = 0
    unique_unselected_candidates: int = 0
    candidate_geometries_read: int = 0
    candidate_geometries_unusable: int = 0
    adjacency_matches: int = 0
    newly_selected: int = 0
    missing_frontier_fids: int = 0
    missing_candidate_fids: int = 0
    seed_rows_missing_or_invalid_fid: int = 0
    seed_rows_recovered_fid: int = 0
    seed_rows_unresolved_fid: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            key: int(value)
            for key, value in self.__dict__.items()
        }


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


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def sqlite_readonly_uri(path: Path) -> str:
    return path.resolve().as_uri() + "?mode=ro"


def chunked(values: Sequence[int] | list[int], size: int) -> Iterator[list[int]]:
    if size <= 0:
        raise ValueError("Chunk size must be positive")
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def available_spatial_layers(path: Path) -> list[str]:
    return [
        str(name)
        for name, geometry_type in pyogrio.list_layers(path)
        if geometry_type is not None
    ]


def select_layer(path: Path, requested: str | None) -> str:
    layers = available_spatial_layers(path)
    if not layers:
        raise RuntimeError(f"No spatial layers found in {path}")
    if requested is None:
        return layers[0]
    if requested not in layers:
        raise ValueError(
            f"Layer {requested!r} not found in {path}. Available: {layers}"
        )
    return requested


def remove_state_files(path: Path) -> None:
    for candidate in (
        path,
        Path(str(path) + "-wal"),
        Path(str(path) + "-shm"),
    ):
        if candidate.exists():
            candidate.unlink()


def remove_output(path: Path) -> None:
    if not path.exists():
        return
    if not OVERWRITE_FINAL_OUTPUT:
        raise FileExistsError(path)
    try:
        path.unlink()
    except PermissionError as exc:
        raise PermissionError(
            f"Could not replace {path}. Close it in QGIS or any other program "
            "and rerun the script."
        ) from exc


# =============================================================================
# SOURCE / SEED INSPECTION
# =============================================================================


def inspect_source_layer() -> SourceLayerInfo:
    if not SOURCE_GPKG.exists():
        raise FileNotFoundError(SOURCE_GPKG)

    layer = select_layer(SOURCE_GPKG, SOURCE_LAYER)
    info = pyogrio.read_info(
        SOURCE_GPKG,
        layer=layer,
        force_feature_count=True,
        force_total_bounds=True,
    )
    if not info.get("crs"):
        raise ValueError(f"Source layer {layer!r} has no CRS")
    crs = pyproj.CRS.from_user_input(info["crs"])
    if not crs.is_projected:
        raise ValueError(
            "The connectivity test is configured in metres, but the source CRS "
            f"is not projected: {crs.to_string()}"
        )
    unit_name = crs.axis_info[0].unit_name.lower() if crs.axis_info else ""
    if unit_name not in {"metre", "meter"}:
        raise ValueError(
            "The source CRS must use metre units for CONNECTIVITY_TOLERANCE_M. "
            f"Detected unit: {unit_name!r}"
        )

    capabilities = info.get("capabilities") or {}
    fast_spatial_filter = bool(capabilities.get("fast_spatial_filter", False))

    with sqlite3.connect(sqlite_readonly_uri(SOURCE_GPKG), uri=True) as connection:
        geometry_row = connection.execute(
            "SELECT column_name FROM gpkg_geometry_columns WHERE table_name = ?",
            (layer,),
        ).fetchone()
        if geometry_row is None:
            raise RuntimeError(
                f"Could not find geometry metadata for {layer!r} in {SOURCE_GPKG}"
            )
        geometry_column = str(geometry_row[0])

        table_info = connection.execute(
            f"PRAGMA table_info({quote_identifier(layer)})"
        ).fetchall()
        primary_keys = [
            (int(row[5]), str(row[1]))
            for row in table_info
            if int(row[5]) > 0
        ]
        if not primary_keys:
            raise RuntimeError(
                f"Could not identify the integer feature-id column for {layer!r}"
            )
        fid_column = sorted(primary_keys)[0][1]

        rtree_table = f"rtree_{layer}_{geometry_column}"
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
            (rtree_table,),
        ).fetchone()
        if exists is None:
            raise RuntimeError(
                f"The required GeoPackage RTree {rtree_table!r} was not found. "
                "Create a spatial index for inland_water_split in QGIS before "
                "running this iterative expansion."
            )

    result = SourceLayerInfo(
        path=SOURCE_GPKG,
        layer=layer,
        crs=crs,
        feature_count=int(info.get("features", -1)),
        geometry_column=geometry_column,
        fid_column=fid_column,
        rtree_table=rtree_table,
        fast_spatial_filter=fast_spatial_filter,
    )
    log(f"Source: {result.path}")
    log(f"Source layer: {result.layer}")
    log(f"Source CRS: {result.crs.to_string()}")
    log(f"Source features: {result.feature_count:,}")
    log(f"Source FID column: {result.fid_column}")
    log(f"Source geometry column: {result.geometry_column}")
    log(f"Source RTree: {result.rtree_table}")
    log(f"GDAL fast spatial filter reported: {result.fast_spatial_filter}")
    return result


def inspect_seed_layer(source: SourceLayerInfo) -> tuple[str, int, list[str]]:
    if not SEED_GPKG.exists():
        raise FileNotFoundError(SEED_GPKG)
    layer = select_layer(SEED_GPKG, SEED_LAYER)
    info = pyogrio.read_info(
        SEED_GPKG,
        layer=layer,
        force_feature_count=True,
        force_total_bounds=True,
    )
    fields = [str(value) for value in info.get("fields", [])]
    if SEED_SOURCE_FID_FIELD not in fields:
        raise ValueError(
            f"Seed layer {layer!r} is missing required field "
            f"{SEED_SOURCE_FID_FIELD!r}. Available fields: {fields}"
        )
    if info.get("crs"):
        seed_crs = pyproj.CRS.from_user_input(info["crs"])
        if seed_crs != source.crs:
            log(
                "WARNING: seed CRS differs from source CRS. This is acceptable "
                "because source_fid provides the authoritative link and source "
                "geometry will be reread from inland_water_split. "
                f"Seed={seed_crs.to_string()}, source={source.crs.to_string()}"
            )
    feature_count = int(info.get("features", -1))
    log(f"Seed subset: {SEED_GPKG}")
    log(f"Seed layer: {layer}")
    log(f"Seed rows: {feature_count:,}")
    return layer, feature_count, fields


# =============================================================================
# STATE DATABASE
# =============================================================================


def processing_signature(
    source: SourceLayerInfo,
    seed_layer: str,
) -> tuple[str, dict[str, Any]]:
    source_stat = source.path.stat()
    seed_stat = SEED_GPKG.stat()
    payload = {
        "logic_version": PROCESSING_LOGIC_VERSION,
        "source_path": str(source.path.resolve()),
        "source_layer": source.layer,
        "source_size": int(source_stat.st_size),
        "source_mtime_ns": int(source_stat.st_mtime_ns),
        "source_crs": source.crs.to_string(),
        "source_rtree": source.rtree_table,
        "seed_path": str(SEED_GPKG.resolve()),
        "seed_layer": seed_layer,
        "seed_size": int(seed_stat.st_size),
        "seed_mtime_ns": int(seed_stat.st_mtime_ns),
        "seed_fid_field": SEED_SOURCE_FID_FIELD,
        "seed_evidence_field": SEED_EVIDENCE_FIELD,
        "connectivity_tolerance_m": float(CONNECTIVITY_TOLERANCE_M),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest(), payload


def open_state_database(
    signature: str,
    signature_payload: dict[str, Any],
) -> sqlite3.Connection:
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    if RESET_EXISTING_STATE:
        remove_state_files(STATE_DB)
    elif STATE_DB.exists() and not RESUME_EXISTING_STATE:
        raise FileExistsError(
            f"State database already exists: {STATE_DB}. Set "
            "RESUME_EXISTING_STATE=True, RESET_EXISTING_STATE=True, or remove it."
        )

    connection = sqlite3.connect(STATE_DB, timeout=STATE_BUSY_TIMEOUT_MS / 1000)
    connection.execute(f"PRAGMA busy_timeout={STATE_BUSY_TIMEOUT_MS}")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute(
        f"PRAGMA cache_size={-int(max(1, STATE_SQLITE_CACHE_MB) * 1024)}"
    )
    connection.execute("PRAGMA foreign_keys=ON")

    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS selected (
            source_fid INTEGER PRIMARY KEY,
            depth INTEGER NOT NULL,
            parent_source_fid INTEGER,
            root_seed_fid INTEGER NOT NULL,
            root_evidence TEXT NOT NULL,
            evidence TEXT NOT NULL,
            processed INTEGER NOT NULL DEFAULT 0,
            selected_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_selected_frontier
            ON selected(processed, depth, source_fid);

        CREATE INDEX IF NOT EXISTS idx_selected_root
            ON selected(root_seed_fid);
        """
    )

    existing_signature_row = connection.execute(
        "SELECT value FROM metadata WHERE key='processing_signature'"
    ).fetchone()
    if existing_signature_row is not None:
        existing_signature = json.loads(str(existing_signature_row[0]))
        if existing_signature != signature:
            connection.close()
            raise RuntimeError(
                "Existing connectivity state was created from different inputs or "
                "settings. Set RESET_EXISTING_STATE=True or use a different WORK_DIR."
            )
    else:
        metadata_values = {
            "processing_signature": signature,
            "processing_signature_payload": signature_payload,
            "created": datetime.now().isoformat(timespec="seconds"),
            "expansion_complete": False,
            "run_counters": RunCounters().to_dict(),
        }
        connection.executemany(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            [
                (key, json.dumps(value, sort_keys=True))
                for key, value in metadata_values.items()
            ],
        )
        connection.commit()

    return connection


def metadata_get(connection: sqlite3.Connection, key: str, default: Any = None) -> Any:
    row = connection.execute(
        "SELECT value FROM metadata WHERE key = ?", (key,)
    ).fetchone()
    if row is None:
        return default
    try:
        return json.loads(str(row[0]))
    except Exception:
        return row[0]


def metadata_set(connection: sqlite3.Connection, key: str, value: Any) -> None:
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
        (key, json.dumps(value, sort_keys=True)),
    )


def load_counters(connection: sqlite3.Connection) -> RunCounters:
    raw = metadata_get(connection, "run_counters", {}) or {}
    counters = RunCounters()
    for key in counters.__dict__:
        if key in raw:
            setattr(counters, key, int(raw[key]))
    return counters


def save_counters(connection: sqlite3.Connection, counters: RunCounters) -> None:
    metadata_set(connection, "run_counters", counters.to_dict())


def _query_source_candidates_for_seed_bounds(
    source_connection: sqlite3.Connection,
    source: SourceLayerInfo,
    geometry: Any,
) -> list[int]:
    """Return source FIDs whose RTree bounds approximately match a seed geometry."""
    if geometry is None or geometry.is_empty:
        return []
    minx, miny, maxx, maxy = map(float, geometry.bounds)
    tol = float(SEED_FID_RECOVERY_RTREE_BOUNDS_TOLERANCE_M)
    sql = (
        f"SELECT id FROM {quote_identifier(source.rtree_table)} "
        "WHERE minx >= ? AND minx <= ? "
        "AND maxx >= ? AND maxx <= ? "
        "AND miny >= ? AND miny <= ? "
        "AND maxy >= ? AND maxy <= ?"
    )
    rows = source_connection.execute(
        sql,
        (
            minx - tol, minx + tol,
            maxx - tol, maxx + tol,
            miny - tol, miny + tol,
            maxy - tol, maxy + tol,
        ),
    )
    return [int(row[0]) for row in rows]


def recover_missing_seed_source_fids(
    source_connection: sqlite3.Connection,
    source: SourceLayerInfo,
    seed_geometry_frame: gpd.GeoDataFrame,
) -> tuple[dict[int, int], list[int]]:
    """Recover missing source_fid values by matching unchanged seed geometries.

    The amalgamated subset is intended to retain original inland_water_split
    geometry unchanged. Therefore a seed row with a NULL source_fid can be linked
    back to the source by RTree-bounding-box preselection followed by exact
    topological geometry equality. The geometry itself is never modified.
    """
    resolved: dict[int, int] = {}
    unresolved: list[int] = []
    if seed_geometry_frame.empty:
        return resolved, unresolved

    seed_items = [
        (int(seed_fid), geometry)
        for seed_fid, geometry in seed_geometry_frame.geometry.items()
    ]

    for batch_start in range(0, len(seed_items), SEED_FID_RECOVERY_BATCH):
        seed_batch = seed_items[batch_start : batch_start + SEED_FID_RECOVERY_BATCH]
        candidate_map: dict[int, list[int]] = {}
        all_candidate_fids: set[int] = set()

        for seed_fid, geometry in seed_batch:
            candidates = _query_source_candidates_for_seed_bounds(
                source_connection,
                source,
                geometry,
            )
            candidate_map[seed_fid] = candidates
            all_candidate_fids.update(candidates)

        source_geometries = read_source_geometries_by_fid(
            source,
            sorted(all_candidate_fids),
        )

        for seed_fid, seed_geometry in seed_batch:
            if seed_geometry is None or seed_geometry.is_empty:
                unresolved.append(seed_fid)
                continue

            matches: list[int] = []
            for source_fid in candidate_map.get(seed_fid, []):
                source_geometry = source_geometries.get(source_fid)
                if source_geometry is None or source_geometry.is_empty:
                    continue
                try:
                    equal = bool(shapely.equals(seed_geometry, source_geometry))
                except Exception:
                    equal = False
                if equal:
                    matches.append(int(source_fid))

            # Invalid-but-equivalent source/seed geometries can occasionally make
            # the direct equality predicate fail. Repair only temporary comparison
            # copies as a fallback; the source geometry written later is unchanged.
            if not matches and REPAIR_INVALID_FOR_ADJACENCY_TEST:
                seed_test = prepare_test_geometry(seed_geometry)
                if seed_test is not None:
                    for source_fid in candidate_map.get(seed_fid, []):
                        source_geometry = source_geometries.get(source_fid)
                        source_test = prepare_test_geometry(source_geometry)
                        if source_test is None:
                            continue
                        try:
                            equal = bool(shapely.equals(seed_test, source_test))
                        except Exception:
                            equal = False
                        if equal:
                            matches.append(int(source_fid))

            if matches:
                # Duplicate identical source geometries are rare. The smallest
                # source FID is deterministic and is sufficient as the seed node;
                # any duplicate source polygon remains discoverable as a neighbour.
                resolved[seed_fid] = min(matches)
            else:
                unresolved.append(seed_fid)

        del source_geometries, candidate_map, all_candidate_fids
        gc.collect()

    return resolved, unresolved


def ingest_seed_rows(
    connection: sqlite3.Connection,
    source_connection: sqlite3.Connection,
    source: SourceLayerInfo,
    seed_layer: str,
    seed_count: int,
    seed_fields: list[str],
    counters: RunCounters,
) -> tuple[int, int]:
    columns = [SEED_SOURCE_FID_FIELD]
    has_evidence = SEED_EVIDENCE_FIELD in seed_fields
    if has_evidence:
        columns.append(SEED_EVIDENCE_FIELD)

    inserted_total = 0
    rows_seen = 0
    offset = 0
    while seed_count < 0 or offset < seed_count:
        frame = pyogrio.read_dataframe(
            SEED_GPKG,
            layer=seed_layer,
            columns=columns,
            read_geometry=False,
            fid_as_index=True,
            skip_features=offset,
            max_features=SEED_ATTRIBUTE_READ_BATCH,
            use_arrow=USE_ARROW,
        )
        if frame.empty:
            break
        rows_seen += len(frame)

        raw_fids = frame[SEED_SOURCE_FID_FIELD]
        fids_numeric = pd.to_numeric(raw_fids, errors="coerce")
        numeric_values = fids_numeric.to_numpy(dtype=float, na_value=np.nan)
        finite = np.isfinite(numeric_values)
        integral = np.zeros(len(frame), dtype=bool)
        integral[finite] = np.isclose(
            numeric_values[finite],
            np.rint(numeric_values[finite]),
            rtol=0.0,
            atol=1e-9,
        )
        valid = finite & integral
        invalid = ~valid

        if has_evidence:
            evidence_values = frame[SEED_EVIDENCE_FIELD].fillna("SEED").astype(str)
        else:
            evidence_values = pd.Series(["SEED"] * len(frame), index=frame.index)

        rows_to_insert: list[tuple[Any, ...]] = []
        for position in np.flatnonzero(valid):
            fid = int(round(float(numeric_values[position])))
            evidence = str(evidence_values.iloc[position])
            rows_to_insert.append((fid, fid, evidence, evidence))

        invalid_seed_layer_fids = [int(value) for value in frame.index[invalid]]
        counters.seed_rows_missing_or_invalid_fid += len(invalid_seed_layer_fids)

        if invalid_seed_layer_fids:
            if not RECOVER_MISSING_SEED_SOURCE_FIDS:
                examples = raw_fids.iloc[np.flatnonzero(invalid)[:10]].tolist()
                raise ValueError(
                    "Seed layer contains NULL/non-numeric source_fid values and "
                    "RECOVER_MISSING_SEED_SOURCE_FIDS=False; "
                    f"examples={examples}"
                )

            log(
                f"Seed ingestion batch contains {len(invalid_seed_layer_fids):,} "
                "NULL/non-numeric source_fid values; resolving them from their "
                "unchanged geometries"
            )
            seed_geometry_frame = pyogrio.read_dataframe(
                SEED_GPKG,
                layer=seed_layer,
                columns=[],
                fids=invalid_seed_layer_fids,
                fid_as_index=True,
                force_2d=True,
                use_arrow=USE_ARROW,
                on_invalid="warn",
            )
            recovered, unresolved = recover_missing_seed_source_fids(
                source_connection,
                source,
                seed_geometry_frame,
            )
            counters.seed_rows_recovered_fid += len(recovered)
            counters.seed_rows_unresolved_fid += len(unresolved)

            evidence_by_seed_fid = {
                int(seed_fid): str(evidence_values.loc[seed_fid])
                for seed_fid in invalid_seed_layer_fids
                if seed_fid in evidence_values.index
            }
            for seed_layer_fid, recovered_source_fid in recovered.items():
                evidence = evidence_by_seed_fid.get(seed_layer_fid, "SEED")
                rows_to_insert.append(
                    (
                        int(recovered_source_fid),
                        int(recovered_source_fid),
                        str(evidence),
                        str(evidence),
                    )
                )

            log(
                f"Seed source_fid recovery: recovered={len(recovered):,}; "
                f"unresolved={len(unresolved):,}"
            )
            if unresolved and FAIL_ON_UNRESOLVED_SEED_SOURCE_FID:
                raise ValueError(
                    f"Could not recover source_fid for {len(unresolved):,} seed "
                    "geometries. Example seed-layer FIDs: "
                    f"{unresolved[:10]}. These rows have valid output geometry but "
                    "could not be matched exactly back to inland_water_split."
                )
            del seed_geometry_frame

        now = datetime.now().isoformat(timespec="seconds")
        before = connection.total_changes
        connection.executemany(
            """
            INSERT OR IGNORE INTO selected(
                source_fid, depth, parent_source_fid,
                root_seed_fid, root_evidence, evidence,
                processed, selected_at
            ) VALUES (?, 0, NULL, ?, ?, ?, 0, ?)
            """,
            [
                (
                    source_fid,
                    root_seed_fid,
                    root_evidence,
                    evidence,
                    now,
                )
                for source_fid, root_seed_fid, root_evidence, evidence in rows_to_insert
            ],
        )
        inserted_total += connection.total_changes - before
        connection.commit()
        save_counters(connection, counters)
        connection.commit()

        offset += len(frame)
        log(
            f"Seed ingestion: read={rows_seen:,}"
            + (f"/{seed_count:,}" if seed_count >= 0 else "")
            + f"; new unique seeds={inserted_total:,}"
        )
        if len(frame) < SEED_ATTRIBUTE_READ_BATCH:
            break

    total_selected = int(
        connection.execute("SELECT COUNT(*) FROM selected").fetchone()[0]
    )
    return inserted_total, total_selected


# =============================================================================
# GEOMETRY READING AND ADJACENCY TESTING
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


def prepare_test_geometry(geometry: Any) -> Polygon | MultiPolygon | None:
    if geometry is None or geometry.is_empty:
        return None
    geometry = shapely.force_2d(geometry)
    if geometry.is_valid or not REPAIR_INVALID_FOR_ADJACENCY_TEST:
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
    if result is None:
        return None
    if not result.is_valid:
        try:
            result = polygonal_only(result.buffer(0))
        except Exception:
            return None
    return result


def read_source_geometries_by_fid(
    source: SourceLayerInfo,
    fids: Sequence[int],
) -> dict[int, Any]:
    result: dict[int, Any] = {}
    if not fids:
        return result
    for fid_batch in chunked(list(fids), SOURCE_FID_READ_BATCH):
        frame = pyogrio.read_dataframe(
            source.path,
            layer=source.layer,
            columns=[],
            fids=fid_batch,
            fid_as_index=True,
            force_2d=True,
            use_arrow=USE_ARROW,
            on_invalid="warn",
        )
        for source_fid, geometry in frame.geometry.items():
            result[int(source_fid)] = geometry
        del frame
    return result


def fetch_frontier_batch(connection: sqlite3.Connection) -> list[FrontierRow]:
    where = "processed = 0"
    parameters: list[Any] = []
    if MAX_CONNECTION_DEPTH is not None:
        where += " AND depth < ?"
        parameters.append(int(MAX_CONNECTION_DEPTH))

    depth_row = connection.execute(
        f"SELECT MIN(depth) FROM selected WHERE {where}", parameters
    ).fetchone()
    if depth_row is None or depth_row[0] is None:
        return []
    depth = int(depth_row[0])

    rows = connection.execute(
        f"""
        SELECT source_fid, depth, root_seed_fid, root_evidence
        FROM selected
        WHERE {where} AND depth = ?
        ORDER BY source_fid
        LIMIT ?
        """,
        [*parameters, depth, int(FRONTIER_BATCH_SIZE)],
    ).fetchall()
    return [
        FrontierRow(
            source_fid=int(row[0]),
            depth=int(row[1]),
            root_seed_fid=int(row[2]),
            root_evidence=str(row[3]),
        )
        for row in rows
    ]


def unprocessed_count(connection: sqlite3.Connection) -> int:
    if MAX_CONNECTION_DEPTH is None:
        row = connection.execute(
            "SELECT COUNT(*) FROM selected WHERE processed = 0"
        ).fetchone()
    else:
        row = connection.execute(
            "SELECT COUNT(*) FROM selected WHERE processed = 0 AND depth < ?",
            (int(MAX_CONNECTION_DEPTH),),
        ).fetchone()
    return int(row[0])


def query_rtree_candidates(
    source_connection: sqlite3.Connection,
    source: SourceLayerInfo,
    frontier_geometries: Sequence[Any],
) -> tuple[set[int], int]:
    sql = (
        f"SELECT id FROM {quote_identifier(source.rtree_table)} "
        "WHERE maxx >= ? AND minx <= ? AND maxy >= ? AND miny <= ?"
    )
    cursor = source_connection.cursor()
    candidates: set[int] = set()
    total_hits = 0
    tolerance = float(CONNECTIVITY_TOLERANCE_M)

    for geometry in frontier_geometries:
        minx, miny, maxx, maxy = map(float, geometry.bounds)
        rows = cursor.execute(
            sql,
            (
                minx - tolerance,
                maxx + tolerance,
                miny - tolerance,
                maxy + tolerance,
            ),
        )
        for row in rows:
            total_hits += 1
            candidates.add(int(row[0]))
    return candidates, total_hits


def remove_already_selected(
    connection: sqlite3.Connection,
    candidate_fids: set[int],
) -> list[int]:
    if not candidate_fids:
        return []
    selected_found: set[int] = set()
    candidate_list = sorted(candidate_fids)
    for batch in chunked(candidate_list, SQLITE_IN_CLAUSE_BATCH):
        placeholders = ",".join("?" for _ in batch)
        rows = connection.execute(
            f"SELECT source_fid FROM selected WHERE source_fid IN ({placeholders})",
            batch,
        )
        selected_found.update(int(row[0]) for row in rows)
    return [fid for fid in candidate_list if fid not in selected_found]


def adjacency_parent_indices(
    frontier_geometries: np.ndarray,
    candidate_geometries: np.ndarray,
) -> dict[int, int]:
    """Return candidate-array index -> matching frontier-array index."""
    if len(frontier_geometries) == 0 or len(candidate_geometries) == 0:
        return {}

    tree = shapely.STRtree(frontier_geometries)
    tolerance = float(CONNECTIVITY_TOLERANCE_M)

    try:
        if tolerance == 0:
            pairs = tree.query(candidate_geometries, predicate="intersects")
        else:
            pairs = tree.query(
                candidate_geometries,
                predicate="dwithin",
                distance=tolerance,
            )
        if pairs.size == 0:
            return {}
        candidate_indices = np.asarray(pairs[0], dtype=np.int64)
        frontier_indices = np.asarray(pairs[1], dtype=np.int64)

        # Deterministic first parent: candidate index, then frontier index.
        order = np.lexsort((frontier_indices, candidate_indices))
        candidate_indices = candidate_indices[order]
        frontier_indices = frontier_indices[order]
        unique_positions = np.flatnonzero(
            np.r_[True, candidate_indices[1:] != candidate_indices[:-1]]
        )
        return {
            int(candidate_indices[position]): int(frontier_indices[position])
            for position in unique_positions
        }
    except Exception as exc:
        log(
            "WARNING: STRtree dwithin query failed; using a slower exact-distance "
            f"fallback: {type(exc).__name__}: {exc}"
        )

    matches: dict[int, int] = {}
    for candidate_index, candidate in enumerate(candidate_geometries):
        if candidate is None or candidate.is_empty:
            continue
        minx, miny, maxx, maxy = map(float, candidate.bounds)
        envelope = shapely.box(
            minx - tolerance,
            miny - tolerance,
            maxx + tolerance,
            maxy + tolerance,
        )
        nearby = np.asarray(tree.query(envelope), dtype=np.int64)
        if len(nearby) == 0:
            continue
        frontier_subset = frontier_geometries[nearby]
        if tolerance == 0:
            flags = np.asarray(
                shapely.intersects(frontier_subset, candidate), dtype=bool
            )
        else:
            distances = np.asarray(
                shapely.distance(frontier_subset, candidate), dtype=float
            )
            flags = distances <= tolerance
        hit_positions = np.flatnonzero(flags)
        if len(hit_positions):
            matches[candidate_index] = int(nearby[int(hit_positions[0])])
    return matches


def process_frontier_batch(
    state_connection: sqlite3.Connection,
    source_connection: sqlite3.Connection,
    source: SourceLayerInfo,
    frontier_rows: list[FrontierRow],
    counters: RunCounters,
) -> int:
    frontier_fids = [row.source_fid for row in frontier_rows]
    frontier_original = read_source_geometries_by_fid(source, frontier_fids)
    missing_frontier = [fid for fid in frontier_fids if fid not in frontier_original]
    if missing_frontier:
        counters.missing_frontier_fids += len(missing_frontier)
        message = (
            f"Could not read {len(missing_frontier):,} frontier source FIDs; "
            f"examples={missing_frontier[:10]}"
        )
        if FAIL_ON_MISSING_SOURCE_FID:
            raise RuntimeError(message)
        log(f"WARNING: {message}; marking them processed")

    valid_rows: list[FrontierRow] = []
    valid_geometries: list[Any] = []
    unusable_frontier: list[int] = []
    for row in frontier_rows:
        geometry = frontier_original.get(row.source_fid)
        test_geometry = prepare_test_geometry(geometry)
        if test_geometry is None:
            unusable_frontier.append(row.source_fid)
            continue
        valid_rows.append(row)
        valid_geometries.append(test_geometry)

    if unusable_frontier:
        message = (
            f"{len(unusable_frontier):,} selected frontier geometries were unusable "
            f"for adjacency testing; examples={unusable_frontier[:10]}"
        )
        if FAIL_ON_MISSING_SOURCE_FID:
            raise RuntimeError(message)
        log(f"WARNING: {message}")

    newly_selected_total = 0
    if valid_rows:
        frontier_test_array = np.asarray(valid_geometries, dtype=object)
        candidate_fid_set, bbox_hits = query_rtree_candidates(
            source_connection,
            source,
            valid_geometries,
        )
        counters.rtree_bbox_hits += bbox_hits
        candidate_fids = remove_already_selected(
            state_connection,
            candidate_fid_set,
        )
        counters.unique_unselected_candidates += len(candidate_fids)

        parent_by_fid: dict[int, FrontierRow] = {
            row.source_fid: row for row in valid_rows
        }

        for candidate_batch in chunked(candidate_fids, SOURCE_FID_READ_BATCH):
            candidate_original = read_source_geometries_by_fid(
                source,
                candidate_batch,
            )
            missing_candidates = [
                fid for fid in candidate_batch if fid not in candidate_original
            ]
            if missing_candidates:
                counters.missing_candidate_fids += len(missing_candidates)
                if FAIL_ON_MISSING_SOURCE_FID:
                    raise RuntimeError(
                        f"Could not read candidate FIDs from source: "
                        f"{missing_candidates[:10]}"
                    )

            usable_candidate_fids: list[int] = []
            usable_candidate_geometries: list[Any] = []
            for candidate_fid in candidate_batch:
                geometry = candidate_original.get(candidate_fid)
                test_geometry = prepare_test_geometry(geometry)
                if test_geometry is None:
                    counters.candidate_geometries_unusable += 1
                    continue
                usable_candidate_fids.append(candidate_fid)
                usable_candidate_geometries.append(test_geometry)

            counters.candidate_geometries_read += len(candidate_original)
            if not usable_candidate_fids:
                continue

            candidate_test_array = np.asarray(
                usable_candidate_geometries,
                dtype=object,
            )
            matches = adjacency_parent_indices(
                frontier_test_array,
                candidate_test_array,
            )
            counters.adjacency_matches += len(matches)
            if not matches:
                continue

            now = datetime.now().isoformat(timespec="seconds")
            rows_to_insert: list[tuple[Any, ...]] = []
            for candidate_index, frontier_index in matches.items():
                candidate_fid = int(usable_candidate_fids[candidate_index])
                parent_row = valid_rows[frontier_index]
                rows_to_insert.append(
                    (
                        candidate_fid,
                        int(parent_row.depth + 1),
                        int(parent_row.source_fid),
                        int(parent_row.root_seed_fid),
                        str(parent_row.root_evidence),
                        "CONNECTED_WITHIN_1M",
                        0,
                        now,
                    )
                )

            before = state_connection.total_changes
            state_connection.executemany(
                """
                INSERT OR IGNORE INTO selected(
                    source_fid, depth, parent_source_fid,
                    root_seed_fid, root_evidence, evidence,
                    processed, selected_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows_to_insert,
            )
            inserted = state_connection.total_changes - before
            newly_selected_total += inserted
            counters.newly_selected += inserted
            state_connection.commit()

            del (
                candidate_original,
                usable_candidate_geometries,
                candidate_test_array,
            )

        del frontier_test_array, candidate_fid_set, candidate_fids

    # Every frontier row has now had its neighbour query. Mark it processed even
    # when it had no candidate neighbours; that makes terminal/isolated branches
    # disappear from subsequent iterations automatically.
    state_connection.executemany(
        "UPDATE selected SET processed = 1 WHERE source_fid = ?",
        [(row.source_fid,) for row in frontier_rows],
    )
    counters.frontier_features_processed += len(frontier_rows)
    state_connection.commit()

    del frontier_original, valid_geometries
    gc.collect()
    return newly_selected_total


# =============================================================================
# FINAL OUTPUT
# =============================================================================


def write_output_frame(
    frame: gpd.GeoDataFrame,
    append: bool,
) -> None:
    if frame.empty:
        return
    pyogrio.write_dataframe(
        frame.reset_index(drop=True),
        OUTPUT_GPKG,
        layer=OUTPUT_LAYER,
        driver="GPKG",
        append=append,
        promote_to_multi=True,
        layer_options={
            "SPATIAL_INDEX": "YES" if CREATE_SPATIAL_INDEX else "NO"
        },
        use_arrow=USE_ARROW,
    )


def build_final_output(
    state_connection: sqlite3.Connection,
    source: SourceLayerInfo,
    *,
    complete: bool,
) -> dict[str, Any]:
    if not complete and not WRITE_PARTIAL_OUTPUT:
        log(
            "Expansion is incomplete; final output is not written because "
            "WRITE_PARTIAL_OUTPUT=False"
        )
        return {
            "written": False,
            "reason": "expansion_incomplete",
        }

    remove_output(OUTPUT_GPKG)
    output_started = False
    last_fid = -1
    written = 0
    started = time.perf_counter()

    while True:
        rows = state_connection.execute(
            """
            SELECT source_fid, depth, parent_source_fid,
                   root_seed_fid, root_evidence, evidence
            FROM selected
            WHERE source_fid > ?
            ORDER BY source_fid
            LIMIT ?
            """,
            (int(last_fid), int(OUTPUT_WRITE_BATCH)),
        ).fetchall()
        if not rows:
            break

        fids = [int(row[0]) for row in rows]
        geometry_by_fid = read_source_geometries_by_fid(source, fids)
        missing = [fid for fid in fids if fid not in geometry_by_fid]
        if missing:
            raise RuntimeError(
                f"Final output could not reread selected source FIDs: {missing[:10]}"
            )

        source_fids: list[int] = []
        evidences: list[str] = []
        is_seed: list[bool] = []
        depths: list[int] = []
        parents: list[int | None] = []
        root_fids: list[int] = []
        root_evidences: list[str] = []
        geometries: list[Any] = []

        for row in rows:
            source_fid = int(row[0])
            depth = int(row[1])
            source_fids.append(source_fid)
            depths.append(depth)
            parents.append(None if row[2] is None else int(row[2]))
            root_fids.append(int(row[3]))
            root_evidences.append(str(row[4]))
            evidences.append(str(row[5]))
            is_seed.append(depth == 0)
            geometries.append(geometry_by_fid[source_fid])

        frame = gpd.GeoDataFrame(
            {
                "source_fid": np.asarray(source_fids, dtype=np.int64),
                "evidence": evidences,
                "is_seed": np.asarray(is_seed, dtype=bool),
                "connection_depth": np.asarray(depths, dtype=np.int32),
                "parent_source_fid": pd.array(parents, dtype="Int64"),
                "root_seed_fid": np.asarray(root_fids, dtype=np.int64),
                "root_evidence": root_evidences,
                "geometry": gpd.GeoSeries(geometries, crs=source.crs),
            },
            geometry="geometry",
            crs=source.crs,
        )
        write_output_frame(frame, append=output_started)
        output_started = True
        written += len(frame)
        last_fid = fids[-1]

        log(
            f"Final output: wrote {written:,} selected source geometries; "
            f"elapsed={format_duration(time.perf_counter() - started)}"
        )
        del frame, geometry_by_fid, geometries
        gc.collect()

    if not output_started:
        log("WARNING: no selected geometries were available to write")
        return {
            "written": False,
            "reason": "no_selected_geometries",
        }

    if VACUUM_FINAL_GPKG:
        log("VACUUM final GeoPackage")
        with sqlite3.connect(OUTPUT_GPKG) as connection:
            connection.execute("VACUUM")

    info = pyogrio.read_info(
        OUTPUT_GPKG,
        layer=OUTPUT_LAYER,
        force_feature_count=True,
        force_total_bounds=True,
    )
    result = {
        "written": True,
        "path": str(OUTPUT_GPKG),
        "layer": OUTPUT_LAYER,
        "feature_count": int(info.get("features", written)),
        "crs": str(info.get("crs")),
        "bounds": (
            None
            if info.get("total_bounds") is None
            else list(map(float, info["total_bounds"]))
        ),
        "size_mb": OUTPUT_GPKG.stat().st_size / (1024**2),
        "elapsed_seconds": time.perf_counter() - started,
        "complete_expansion": bool(complete),
    }
    log(
        f"Final connected subset: {OUTPUT_GPKG}; "
        f"features={result['feature_count']:,}; size={result['size_mb']:,.2f} MB"
    )
    return result


# =============================================================================
# REPORTING
# =============================================================================


def selected_summary(connection: sqlite3.Connection) -> dict[str, Any]:
    total = int(connection.execute("SELECT COUNT(*) FROM selected").fetchone()[0])
    seeds = int(
        connection.execute("SELECT COUNT(*) FROM selected WHERE depth = 0").fetchone()[0]
    )
    processed = int(
        connection.execute("SELECT COUNT(*) FROM selected WHERE processed = 1").fetchone()[0]
    )
    max_depth_raw = connection.execute("SELECT MAX(depth) FROM selected").fetchone()[0]
    depth_rows = connection.execute(
        "SELECT depth, COUNT(*) FROM selected GROUP BY depth ORDER BY depth"
    ).fetchall()
    evidence_rows = connection.execute(
        "SELECT evidence, COUNT(*) FROM selected GROUP BY evidence ORDER BY evidence"
    ).fetchall()
    return {
        "selected_total": total,
        "seed_count": seeds,
        "connected_added": total - seeds,
        "processed_selected": processed,
        "unprocessed_selected": total - processed,
        "maximum_connection_depth": (
            None if max_depth_raw is None else int(max_depth_raw)
        ),
        "counts_by_depth": {str(int(depth)): int(count) for depth, count in depth_rows},
        "counts_by_evidence": {str(value): int(count) for value, count in evidence_rows},
    }


def write_report(data: dict[str, Any]) -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    temporary = REPORT_PATH.with_suffix(REPORT_PATH.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(temporary, REPORT_PATH)


# =============================================================================
# VALIDATION AND MAIN
# =============================================================================


def validate_settings() -> None:
    if CONNECTIVITY_TOLERANCE_M < 0:
        raise ValueError("CONNECTIVITY_TOLERANCE_M cannot be negative")
    if FRONTIER_BATCH_SIZE <= 0:
        raise ValueError("FRONTIER_BATCH_SIZE must be positive")
    if SOURCE_FID_READ_BATCH <= 0:
        raise ValueError("SOURCE_FID_READ_BATCH must be positive")
    if SEED_ATTRIBUTE_READ_BATCH <= 0:
        raise ValueError("SEED_ATTRIBUTE_READ_BATCH must be positive")
    if SEED_FID_RECOVERY_BATCH <= 0:
        raise ValueError("SEED_FID_RECOVERY_BATCH must be positive")
    if SEED_FID_RECOVERY_RTREE_BOUNDS_TOLERANCE_M < 0:
        raise ValueError("SEED_FID_RECOVERY_RTREE_BOUNDS_TOLERANCE_M cannot be negative")
    if OUTPUT_WRITE_BATCH <= 0:
        raise ValueError("OUTPUT_WRITE_BATCH must be positive")
    if SQLITE_IN_CLAUSE_BATCH <= 0:
        raise ValueError("SQLITE_IN_CLAUSE_BATCH must be positive")
    if MAX_FRONTIER_BATCHES_THIS_RUN is not None and MAX_FRONTIER_BATCHES_THIS_RUN <= 0:
        raise ValueError("MAX_FRONTIER_BATCHES_THIS_RUN must be positive or None")
    if MAX_CONNECTION_DEPTH is not None and MAX_CONNECTION_DEPTH < 0:
        raise ValueError("MAX_CONNECTION_DEPTH must be non-negative or None")


def main() -> None:
    overall_started = time.perf_counter()
    validate_settings()
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    log("Starting iterative inland-water connectivity expansion")
    log(
        f"Python {os.sys.version.split()[0]}; GeoPandas {gpd.__version__}; "
        f"Shapely {shapely.__version__} / GEOS {shapely.geos_version_string}; "
        f"Pyogrio {pyogrio.__version__}; Pyproj {pyproj.__version__}"
    )
    log(
        "Logic: use current retained source_fid values as seeds; repeatedly add "
        f"original source polygons within {CONNECTIVITY_TOLERANCE_M:g} m; stop "
        "when the queue is empty. Original geometries are not buffered or altered."
    )

    source = inspect_source_layer()
    seed_layer, seed_count, seed_fields = inspect_seed_layer(source)
    signature, signature_payload = processing_signature(source, seed_layer)
    state_connection = open_state_database(signature, signature_payload)
    counters = load_counters(state_connection)

    source_connection = sqlite3.connect(
        sqlite_readonly_uri(source.path),
        uri=True,
        timeout=STATE_BUSY_TIMEOUT_MS / 1000,
    )
    source_connection.execute(f"PRAGMA busy_timeout={STATE_BUSY_TIMEOUT_MS}")
    source_connection.execute("PRAGMA query_only=ON")
    source_connection.execute(
        f"PRAGMA cache_size={-int(max(1, STATE_SQLITE_CACHE_MB) * 1024)}"
    )

    try:
        new_seeds, total_after_seed = ingest_seed_rows(
            state_connection,
            source_connection,
            source,
            seed_layer,
            seed_count,
            seed_fields,
            counters,
        )
        log(
            f"Seed state ready: new seeds={new_seeds:,}; "
            f"selected total={total_after_seed:,}; "
            f"frontier remaining={unprocessed_count(state_connection):,}"
        )

        batches_this_run = 0
        run_started = time.perf_counter()
        recent_durations: list[float] = []

        while True:
            if (
                MAX_FRONTIER_BATCHES_THIS_RUN is not None
                and batches_this_run >= MAX_FRONTIER_BATCHES_THIS_RUN
            ):
                log(
                    "Reached MAX_FRONTIER_BATCHES_THIS_RUN; state was committed and "
                    "the next run will resume from the remaining frontier"
                )
                break

            frontier_rows = fetch_frontier_batch(state_connection)
            if not frontier_rows:
                break

            batch_started = time.perf_counter()
            current_depth = frontier_rows[0].depth
            selected_before = int(
                state_connection.execute("SELECT COUNT(*) FROM selected").fetchone()[0]
            )
            remaining_before = unprocessed_count(state_connection)
            log("-" * 78)
            log(
                f"Frontier batch {counters.frontier_batches + 1:,}: "
                f"depth={current_depth:,}; frontier={len(frontier_rows):,}; "
                f"selected={selected_before:,}; queue={remaining_before:,}"
            )

            added = process_frontier_batch(
                state_connection,
                source_connection,
                source,
                frontier_rows,
                counters,
            )
            counters.frontier_batches += 1
            batches_this_run += 1
            duration = time.perf_counter() - batch_started
            recent_durations.append(duration)
            recent_durations = recent_durations[-20:]

            save_counters(state_connection, counters)
            metadata_set(
                state_connection,
                "updated",
                datetime.now().isoformat(timespec="seconds"),
            )
            state_connection.commit()

            if (
                LOG_EVERY_FRONTIER_BATCHES <= 1
                or counters.frontier_batches % LOG_EVERY_FRONTIER_BATCHES == 0
            ):
                selected_now = int(
                    state_connection.execute("SELECT COUNT(*) FROM selected").fetchone()[0]
                )
                queue_now = unprocessed_count(state_connection)
                average = sum(recent_durations) / len(recent_durations)
                # Queue length can grow or shrink, so this is deliberately labelled
                # a rough queue-drain estimate rather than a guaranteed ETA.
                batches_left_estimate = math.ceil(queue_now / FRONTIER_BATCH_SIZE)
                log(
                    f"Batch complete in {format_duration(duration)}; added={added:,}; "
                    f"selected={selected_now:,}; queue={queue_now:,}; "
                    f"rough queue-drain estimate="
                    f"{format_duration(average * batches_left_estimate)}"
                )

        remaining = unprocessed_count(state_connection)
        expansion_complete = remaining == 0
        metadata_set(state_connection, "expansion_complete", expansion_complete)
        metadata_set(
            state_connection,
            "updated",
            datetime.now().isoformat(timespec="seconds"),
        )
        save_counters(state_connection, counters)
        state_connection.commit()

        summary = selected_summary(state_connection)
        log(
            f"Expansion stage: complete={expansion_complete}; "
            f"selected={summary['selected_total']:,}; "
            f"seeds={summary['seed_count']:,}; "
            f"connected added={summary['connected_added']:,}; "
            f"max depth={summary['maximum_connection_depth']}"
        )

        output_report: dict[str, Any] | None = None
        if WRITE_FINAL_OUTPUT_WHEN_COMPLETE and expansion_complete:
            output_report = build_final_output(
                state_connection,
                source,
                complete=True,
            )
        elif WRITE_PARTIAL_OUTPUT:
            output_report = build_final_output(
                state_connection,
                source,
                complete=expansion_complete,
            )
        else:
            log(
                "Final output not written during this run. Resume until the queue "
                "is empty, or enable WRITE_PARTIAL_OUTPUT."
            )

        report = {
            "created": datetime.now().isoformat(timespec="seconds"),
            "logic_version": PROCESSING_LOGIC_VERSION,
            "source": {
                "path": str(source.path),
                "layer": source.layer,
                "crs": source.crs.to_string(),
                "feature_count": source.feature_count,
                "fid_column": source.fid_column,
                "geometry_column": source.geometry_column,
                "rtree_table": source.rtree_table,
            },
            "seed": {
                "path": str(SEED_GPKG),
                "layer": seed_layer,
                "source_fid_field": SEED_SOURCE_FID_FIELD,
                "evidence_field": SEED_EVIDENCE_FIELD,
                "rows_reported": seed_count,
            },
            "settings": {
                "connectivity_tolerance_m": CONNECTIVITY_TOLERANCE_M,
                "frontier_batch_size": FRONTIER_BATCH_SIZE,
                "source_fid_read_batch": SOURCE_FID_READ_BATCH,
                "max_connection_depth": MAX_CONNECTION_DEPTH,
                "repair_invalid_for_adjacency_test": (
                    REPAIR_INVALID_FOR_ADJACENCY_TEST
                ),
            },
            "expansion_complete": expansion_complete,
            "frontier_remaining": remaining,
            "selection_summary": summary,
            "counters": counters.to_dict(),
            "output": output_report,
            "state_database": str(STATE_DB),
            "elapsed_seconds_this_run": time.perf_counter() - overall_started,
            "expansion_elapsed_seconds_this_run": time.perf_counter() - run_started,
        }
        write_report(report)
        log(f"Wrote report: {REPORT_PATH}")
        log(
            f"Completed this run in "
            f"{format_duration(time.perf_counter() - overall_started)}"
        )
    finally:
        source_connection.close()
        state_connection.close()


if __name__ == "__main__":
    main()
