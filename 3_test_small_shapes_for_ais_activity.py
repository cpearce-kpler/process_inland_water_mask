#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For each small candidate shape (e.g. the <100m^2 inland/sea shapes under
review for removal), test whether it actually contains AIS activity from a
real, recognized vessel -- without ever scanning the full ~10.04B-row AIS
dataset.

How this stays fast
--------------------
Every AIS row already has a precomputed `hierarchical_cell` (0.01/0.001
degree grid cell) sitting in the *.grid_0p01_0p001.parquet sidecars, built
earlier in this project, entirely separate from the land/water hierarchy.
Since a <100m^2 shape (usually well under 15m across) only ever overlaps a
handful of 0.001-degree (~111m) cells:

    1. Reproject the (few, tiny) candidate shapes from EPSG:8857 to WGS84
       once, and compute the small set of hierarchical_cell values each one
       overlaps.
    2. Make ONE pass over each source file's *compact* hierarchical_cell
       sidecar (not the raw AIS data) and keep only row positions whose
       cell is in that target set -- a plain integer membership test.
    3. Only for that (normally tiny) filtered subset, go back to the
       matching raw AIS parquet and fetch SHIP_ID/LAT/LON for just those
       row positions.
    4. Check each matched SHIP_ID against the vessel lookup (the same
       numpy_searchsorted-by-SHIP_ID design as vessel_lookup already uses)
       and discard points from ships that aren't recognized, real vessels.
    5. For the remaining points, do the exact point-in-polygon test
       against the (small) set of nearby candidates and count hits.

A 111m grid cell is much bigger than a <100m^2 shape, so "row is in a
matching cell" does NOT mean "row is inside the shape" -- step 5's exact
polygon test is what actually decides that, on a population that should
already be a tiny fraction of the full dataset.

"Legitimate vessel" here means: SHIP_ID is present in vessel_hot.parquet,
AND it has at least one real identifier (IMO or MMSI not the -1 sentinel).
Ships failing this are excluded from the point count entirely, so an
erroneous/spoofed SHIP_ID sitting inside a shape doesn't save it from
removal.

Assumptions worth checking against your real run before trusting results:
    - The AIS grid uses the SAME global 0.01/0.001-degree definition as the
      land/water hierarchy (parent_id = lat_idx*36000 + lon_idx,
      hierarchical_cell = parent_id*100 + child_id). Verified here by
      decoding a real hierarchical_cell value from your uploaded sidecar
      back to lat/lon and confirming it lands where the corresponding AIS
      screenshot data actually is.
    - Each sidecar's own `source_relative_path` metadata field names its
      matching raw AIS parquet file, resolved under --ais-source-dir.

Usage (PowerShell)
-------------------
    & "C:\\anaconda\\python.exe" -u "C:\\Users\\Craig Pearce\\Desktop\\test_small_shapes_for_ais_activity.py" `
        --candidates-gpkg "C:\\Users\\Craig Pearce\\Desktop\\test_geoms.gpkg" `
        --grid-assignment-dir "C:\\Users\\Craig Pearce\\Desktop\\ais_2025_grid_assignment\\sidecars" `
        --ais-source-dir "C:\\Users\\Craig Pearce\\Desktop\\Data_sets\\ais_files\\ais_2025" `
        --vessel-hot-parquet "C:\\Users\\Craig Pearce\\Desktop\\vessel_metadata_lookup\\vessel_hot.parquet" `
        --output-dir "C:\\Users\\Craig Pearce\\Desktop\\small_shape_ais_test" `
        --max-area-m2 100 `
        --min-points 10
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

import fiona
import pyproj
import shapely
import shapely.ops
from shapely.geometry import shape as shapely_shape, mapping as shapely_mapping, Point


# ---------------------------------------------------------------------------
# Prevent Windows from sleeping (display/system) for the duration of a run
# that could take hours. Restored on exit, including on error, via the
# try/finally in main(). No-op on non-Windows platforms.
# ---------------------------------------------------------------------------

_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001
_ES_DISPLAY_REQUIRED = 0x00000002


def _prevent_sleep() -> None:
    if sys.platform != "win32":
        return
    import ctypes

    ctypes.windll.kernel32.SetThreadExecutionState(
        _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED | _ES_DISPLAY_REQUIRED
    )


def _allow_sleep() -> None:
    if sys.platform != "win32":
        return
    import ctypes

    ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)


PARENT_RESOLUTION = 0.01
FINE_RESOLUTION = 0.001
PARENT_LON_CELLS = 36_000
PARENT_LAT_CELLS = 18_000
CHILDREN_PER_PARENT_SIDE = 10  # 10x10 = 100 children per parent

NULL_SENTINEL = -1


# ---------------------------------------------------------------------------
# Grid math (validated against a real hierarchical_cell value from the
# uploaded sidecar, decoding back to real AIS coordinates)
# ---------------------------------------------------------------------------


def fine_indices_from_latlon(lat: float, lon: float) -> tuple[int, int]:
    lat_c = min(lat, math.nextafter(90.0, -math.inf))
    lon_c = min(lon, math.nextafter(180.0, -math.inf))
    lat_fine = int(math.floor((lat_c + 90.0) / FINE_RESOLUTION))
    lon_fine = int(math.floor((lon_c + 180.0) / FINE_RESOLUTION))
    lat_fine = min(max(lat_fine, 0), PARENT_LAT_CELLS * 10 - 1)
    lon_fine = min(max(lon_fine, 0), PARENT_LON_CELLS * 10 - 1)
    return lat_fine, lon_fine


def hierarchical_cell_from_fine_indices(lat_fine: int, lon_fine: int) -> int:
    parent_lat = lat_fine // CHILDREN_PER_PARENT_SIDE
    parent_lon = lon_fine // CHILDREN_PER_PARENT_SIDE
    parent_id = parent_lat * PARENT_LON_CELLS + parent_lon
    child_lat = lat_fine % CHILDREN_PER_PARENT_SIDE
    child_lon = lon_fine % CHILDREN_PER_PARENT_SIDE
    child_id = child_lat * CHILDREN_PER_PARENT_SIDE + child_lon
    return parent_id * 100 + child_id


def cells_overlapping_bbox(minlat: float, minlon: float, maxlat: float, maxlon: float) -> list[int]:
    lat_fine_min, lon_fine_min = fine_indices_from_latlon(minlat, minlon)
    lat_fine_max, lon_fine_max = fine_indices_from_latlon(maxlat, maxlon)
    cells = []
    for lat_fine in range(lat_fine_min, lat_fine_max + 1):
        for lon_fine in range(lon_fine_min, lon_fine_max + 1):
            cells.append(hierarchical_cell_from_fine_indices(lat_fine, lon_fine))
    return cells


# ---------------------------------------------------------------------------
# Vessel legitimacy lookup
# ---------------------------------------------------------------------------


class VesselLookup:
    def __init__(self, vessel_hot_parquet: Path):
        table = pq.read_table(
            vessel_hot_parquet, columns=["SHIP_ID", "IMO", "MMSI", "grouped_type_code"]
        )
        order = np.argsort(table.column("SHIP_ID").to_numpy())
        self.ship_id = table.column("SHIP_ID").to_numpy()[order]
        self.imo = table.column("IMO").to_numpy()[order]
        self.mmsi = table.column("MMSI").to_numpy()[order]
        self.grouped_type_code = table.column("grouped_type_code").to_numpy()[order]

    def legitimate_mask(self, ship_ids: np.ndarray) -> np.ndarray:
        rank = np.searchsorted(self.ship_id, ship_ids)
        in_range = rank < self.ship_id.size
        found = np.zeros(ship_ids.shape, dtype=bool)
        idx = np.flatnonzero(in_range)
        if idx.size:
            found[idx] = self.ship_id[rank[idx]] == ship_ids[idx]
        legit = np.zeros(ship_ids.shape, dtype=bool)
        found_idx = np.flatnonzero(found)
        if found_idx.size:
            r = rank[found_idx]
            has_real_id = (self.imo[r] != NULL_SENTINEL) | (self.mmsi[r] != NULL_SENTINEL)
            legit[found_idx] = has_real_id
        return legit


# ---------------------------------------------------------------------------
# Candidate shapes: load, reproject to WGS84, compute target cells
# ---------------------------------------------------------------------------


REASON_REMOVE_TINY = "remove_tiny_area"
REASON_KEEP_LARGE = "keep_large_area"
REASON_KEEP_DENSITY_FAST = "keep_density_fast_path"
REASON_KEEP_EXACT = "keep_exact_ais_count"
REASON_REMOVE_EXACT = "remove_exact_ais_count"


def load_and_classify_candidates(
    candidates_gpkg: Path,
    layer: str | None,
    *,
    min_valid_area_m2: float,
    auto_keep_area_m2: float | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Any]:
    """Returns (resolved, middle_tier, src_crs).

    resolved: candidates already decided (remove_tiny / keep_large) --
    these never get a cell-overlap computation, which is what keeps this
    safe for the extreme size tail (a shape with 680k vertices and a
    bounding box spanning millions of cells never has cells enumerated at
    all).

    middle_tier: bounded-size candidates carrying their wgs84 geometry and
    overlapping cells, with no density/exact decision made yet -- that
    happens afterwards in a dedicated targeted pass (see
    run_density_prepass / run_exact_pipeline), once all middle_tier
    candidates are known and their target cells can be pooled together.
    """
    resolved: list[dict[str, Any]] = []
    middle_tier: list[dict[str, Any]] = []

    with fiona.open(str(candidates_gpkg), layer=layer) as src:
        src_crs = src.crs
        to_wgs84 = pyproj.Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
        for feature in src:
            geom = feature.get("geometry")
            if geom is None:
                continue
            native_geom = shapely_shape(geom)
            area = native_geom.area
            fid = feature.get("id")

            if area < min_valid_area_m2:
                cx, cy = native_geom.centroid.coords[0]
                lon, lat = to_wgs84.transform(cx, cy)
                resolved.append(
                    {
                        "fid": fid,
                        "native_area_m2": area,
                        "decision": "remove",
                        "reason": REASON_REMOVE_TINY,
                        "ais_point_count": None,
                        "density_grid_sum": None,
                        "wgs84_centroid": (lon, lat),
                    }
                )
                continue

            if auto_keep_area_m2 is not None and area >= auto_keep_area_m2:
                # Reproject only the centroid point, never the full huge
                # geometry -- this is the case cell enumeration would have
                # been dangerous for, so it's never attempted here.
                cx, cy = native_geom.centroid.coords[0]
                lon, lat = to_wgs84.transform(cx, cy)
                resolved.append(
                    {
                        "fid": fid,
                        "native_area_m2": area,
                        "decision": "keep",
                        "reason": REASON_KEEP_LARGE,
                        "ais_point_count": None,
                        "density_grid_sum": None,
                        "wgs84_centroid": (lon, lat),
                    }
                )
                continue

            # Middle tier: bounded size, safe to enumerate cells.
            wgs84_geom = shapely.ops.transform(
                lambda x, y, t=to_wgs84: t.transform(x, y), native_geom
            )
            minlon, minlat, maxlon, maxlat = wgs84_geom.bounds
            cells = cells_overlapping_bbox(minlat, minlon, maxlat, maxlon)
            middle_tier.append(
                {
                    "fid": fid,
                    "native_area_m2": area,
                    "wgs84_geom": wgs84_geom,
                    "cells": cells,
                }
            )

    return resolved, middle_tier, src_crs


def build_cell_index(candidates: list[dict[str, Any]]) -> tuple[np.ndarray, dict[int, list[int]]]:
    """Sorted array of every target cell, and cell -> candidate indices."""
    cell_to_candidates: dict[int, list[int]] = defaultdict(list)
    for ci, cand in enumerate(candidates):
        for cell in cand["cells"]:
            cell_to_candidates[cell].append(ci)
    all_cells = np.array(sorted(cell_to_candidates.keys()), dtype=np.uint64)
    return all_cells, dict(cell_to_candidates)


def run_density_prepass(
    middle_tier: list[dict[str, Any]],
    *,
    grid_assignment_dir: Path,
    progress_files: int,
    max_sidecar_files: int | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """One lightweight pass over the sidecars, counting occurrences of ONLY
    the cells that middle_tier candidates actually touch -- never a global
    grid. This is what replaces the standalone density-grid builder that
    ran out of memory: bounded by the (small) target-cell set pooled from
    real candidates, not by every occupied cell on Earth.

    No raw AIS refetch, no vessel-legitimacy check, no point-in-polygon
    test happens here -- just an integer membership test per row and a
    scatter-increment for matches, which is why this pass is much cheaper
    than the exact pipeline per file it touches.

    Returns (per_candidate_density_sum, prepass_stats).
    """
    all_cells, cell_to_candidates = build_cell_index(middle_tier)
    print(f"  {len(middle_tier):,} middle-tier candidates -> {all_cells.size:,} distinct target cells")

    cell_counts = np.zeros(all_cells.size, dtype=np.int64)
    sidecar_files = sorted(grid_assignment_dir.glob("*.parquet"))
    if not sidecar_files:
        raise FileNotFoundError(f"No sidecar parquet files found under {grid_assignment_dir}")
    if max_sidecar_files is not None:
        sidecar_files = sidecar_files[:max_sidecar_files]

    started = time.perf_counter()
    files_with_matches = 0
    total_matched_rows = 0

    for file_index, sidecar_path in enumerate(sidecar_files, start=1):
        table = pq.read_table(sidecar_path, columns=["hierarchical_cell"])
        col = table.column("hierarchical_cell")
        valid_mask = np.asarray(col.is_valid())
        values = col.to_numpy(zero_copy_only=False)
        values = np.where(valid_mask, values, np.uint64(2**64 - 1))

        rank = np.searchsorted(all_cells, values)
        in_range = rank < all_cells.size
        match_mask = np.zeros(values.shape, dtype=bool)
        idx_in_range = np.flatnonzero(in_range)
        if idx_in_range.size:
            match_mask[idx_in_range] = all_cells[rank[idx_in_range]] == values[idx_in_range]

        matched_positions = np.flatnonzero(match_mask)
        if matched_positions.size:
            files_with_matches += 1
            total_matched_rows += matched_positions.size
            matched_ranks = rank[matched_positions]
            np.add.at(cell_counts, matched_ranks, 1)

        if file_index % progress_files == 0 or file_index == len(sidecar_files):
            elapsed = time.perf_counter() - started
            print(
                f"  [density prepass {elapsed:7.1f}s] {file_index:,}/{len(sidecar_files):,} files  "
                f"files_with_matches={files_with_matches:,}  matched_rows={total_matched_rows:,}",
                flush=True,
            )

    per_candidate_sum = np.zeros(len(middle_tier), dtype=np.int64)
    for cell_pos, cell in enumerate(all_cells):
        count = cell_counts[cell_pos]
        if count == 0:
            continue
        for ci in cell_to_candidates[int(cell)]:
            per_candidate_sum[ci] += count

    stats = {
        "elapsed_seconds": time.perf_counter() - started,
        "sidecar_files_scanned": len(sidecar_files),
        "target_cells": int(all_cells.size),
        "files_with_matches": files_with_matches,
        "total_matched_rows": total_matched_rows,
    }
    return per_candidate_sum, stats


def run_exact_pipeline(
    needs_exact: list[dict[str, Any]],
    *,
    grid_assignment_dir: Path,
    ais_source_dir: Path,
    vessel_hot_parquet: Path,
    min_points: int,
    progress_files: int,
    max_sidecar_files: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Runs the original single-pass-over-sidecars pipeline, but only for
    the candidates the fast tiers couldn't resolve. Returns per-candidate
    decisions (same order as needs_exact) plus scan-level stats."""
    all_cells, cell_to_candidates = build_cell_index(needs_exact)
    print(f"  {len(needs_exact):,} candidates need the exact pipeline -> "
          f"{all_cells.size:,} distinct target cells")

    print("Loading vessel lookup...")
    vessel_lookup = VesselLookup(vessel_hot_parquet)

    counts = np.zeros(len(needs_exact), dtype=np.int64)
    sidecar_files = sorted(grid_assignment_dir.glob("*.parquet"))
    if not sidecar_files:
        raise FileNotFoundError(f"No sidecar parquet files found under {grid_assignment_dir}")
    if max_sidecar_files is not None:
        sidecar_files = sidecar_files[:max_sidecar_files]

    started = time.perf_counter()
    files_with_matches = 0
    total_matched_rows = 0
    total_legitimate_rows = 0

    for file_index, sidecar_path in enumerate(sidecar_files, start=1):
        pf = pq.ParquetFile(sidecar_path)
        meta = pf.metadata.metadata or {}
        source_relative_path = meta.get(b"source_relative_path")
        if source_relative_path is None:
            print(f"  WARNING: {sidecar_path.name} has no source_relative_path metadata; skipping")
            continue
        source_relative_path = source_relative_path.decode("utf-8")

        cell_table = pf.read(columns=["hierarchical_cell"])
        cell_col = cell_table.column("hierarchical_cell")
        valid_mask = np.asarray(cell_col.is_valid())
        cell_values = cell_col.to_numpy(zero_copy_only=False)
        cell_values = np.where(valid_mask, cell_values, np.uint64(2**64 - 1))

        rank = np.searchsorted(all_cells, cell_values)
        in_range = rank < all_cells.size
        match_mask = np.zeros(cell_values.shape, dtype=bool)
        idx_in_range = np.flatnonzero(in_range)
        if idx_in_range.size:
            match_mask[idx_in_range] = all_cells[rank[idx_in_range]] == cell_values[idx_in_range]

        matched_positions = np.flatnonzero(match_mask)
        if matched_positions.size == 0:
            continue

        files_with_matches += 1
        total_matched_rows += matched_positions.size

        ais_path = ais_source_dir / source_relative_path
        if not ais_path.is_file():
            print(f"  WARNING: matching AIS source file not found: {ais_path}; skipping")
            continue

        ais_table = pq.read_table(ais_path, columns=["SHIP_ID", "LAT", "LON"])
        matched = ais_table.take(matched_positions)
        ship_ids = matched.column("SHIP_ID").to_numpy(zero_copy_only=False)
        lats = matched.column("LAT").to_numpy(zero_copy_only=False)
        lons = matched.column("LON").to_numpy(zero_copy_only=False)
        matched_cells = cell_values[matched_positions]

        legit = vessel_lookup.legitimate_mask(ship_ids)
        total_legitimate_rows += int(legit.sum())
        if not np.any(legit):
            continue

        legit_idx = np.flatnonzero(legit)
        if legit_idx.size:
            candidate_point_lons: dict[int, list[float]] = defaultdict(list)
            candidate_point_lats: dict[int, list[float]] = defaultdict(list)
            for k in legit_idx:
                cell = int(matched_cells[k])
                candidate_indices = cell_to_candidates.get(cell)
                if not candidate_indices:
                    continue
                for ci in candidate_indices:
                    candidate_point_lons[ci].append(float(lons[k]))
                    candidate_point_lats[ci].append(float(lats[k]))
            for ci, lon_list in candidate_point_lons.items():
                pts = shapely.points(
                    np.asarray(lon_list, dtype=np.float64),
                    np.asarray(candidate_point_lats[ci], dtype=np.float64),
                )
                hits = shapely.intersects(pts, needs_exact[ci]["wgs84_geom"])
                counts[ci] += int(np.count_nonzero(hits))

        if file_index % progress_files == 0 or file_index == len(sidecar_files):
            elapsed = time.perf_counter() - started
            print(
                f"  [{elapsed:7.1f}s] {file_index:,}/{len(sidecar_files):,} files  "
                f"files_with_matches={files_with_matches:,}  "
                f"matched_rows={total_matched_rows:,}  legitimate_rows={total_legitimate_rows:,}",
                flush=True,
            )

    exact_results = []
    for i, cand in enumerate(needs_exact):
        keep = bool(counts[i] >= min_points)
        centroid = cand["wgs84_geom"].centroid
        exact_results.append(
            {
                "fid": cand["fid"],
                "native_area_m2": cand["native_area_m2"],
                "decision": "keep" if keep else "remove",
                "reason": REASON_KEEP_EXACT if keep else REASON_REMOVE_EXACT,
                "ais_point_count": int(counts[i]),
                "density_grid_sum": cand.get("density_grid_sum"),
                "wgs84_centroid": centroid.coords[0],
            }
        )

    scan_stats = {
        "elapsed_seconds": time.perf_counter() - started,
        "sidecar_files_scanned": len(sidecar_files),
        "files_with_matches": files_with_matches,
        "total_matched_rows": total_matched_rows,
        "total_legitimate_rows": total_legitimate_rows,
    }
    return exact_results, scan_stats


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------


def run(
    *,
    candidates_gpkg: Path,
    layer: str | None,
    grid_assignment_dir: Path,
    ais_source_dir: Path,
    vessel_hot_parquet: Path,
    output_dir: Path,
    min_valid_area_m2: float,
    auto_keep_area_m2: float | None,
    density_fast_keep_threshold: int | None,
    min_points: int,
    progress_files: int,
    max_sidecar_files: int | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading candidate shapes and classifying by size...")
    resolved, middle_tier, src_crs = load_and_classify_candidates(
        candidates_gpkg,
        layer,
        min_valid_area_m2=min_valid_area_m2,
        auto_keep_area_m2=auto_keep_area_m2,
    )
    print(
        f"  resolved by size alone: {len(resolved):,} "
        f"(remove_tiny={sum(1 for r in resolved if r['reason']==REASON_REMOVE_TINY):,}, "
        f"keep_large={sum(1 for r in resolved if r['reason']==REASON_KEEP_LARGE):,})"
    )
    print(f"  middle tier (needs testing): {len(middle_tier):,}")

    density_resolved: list[dict[str, Any]] = []
    needs_exact = middle_tier
    prepass_stats: dict[str, Any] = {}

    if middle_tier and density_fast_keep_threshold is not None:
        print("Running targeted AIS density pre-pass (sidecars only, no raw AIS refetch yet)...")
        per_candidate_sum, prepass_stats = run_density_prepass(
            middle_tier,
            grid_assignment_dir=grid_assignment_dir,
            progress_files=progress_files,
            max_sidecar_files=max_sidecar_files,
        )
        needs_exact = []
        for ci, cand in enumerate(middle_tier):
            density_sum = int(per_candidate_sum[ci])
            if density_sum >= density_fast_keep_threshold:
                centroid = cand["wgs84_geom"].centroid
                density_resolved.append(
                    {
                        "fid": cand["fid"],
                        "native_area_m2": cand["native_area_m2"],
                        "decision": "keep",
                        "reason": REASON_KEEP_DENSITY_FAST,
                        "ais_point_count": None,
                        "density_grid_sum": density_sum,
                        "wgs84_centroid": centroid.coords[0],
                    }
                )
            else:
                cand["density_grid_sum"] = density_sum
                needs_exact.append(cand)
        print(
            f"  density pre-pass resolved: {len(density_resolved):,}; "
            f"still needs exact pipeline: {len(needs_exact):,}"
        )
    elif middle_tier:
        for cand in needs_exact:
            cand["density_grid_sum"] = None

    exact_results: list[dict[str, Any]] = []
    scan_stats: dict[str, Any] = {}
    if needs_exact:
        exact_results, scan_stats = run_exact_pipeline(
            needs_exact,
            grid_assignment_dir=grid_assignment_dir,
            ais_source_dir=ais_source_dir,
            vessel_hot_parquet=vessel_hot_parquet,
            min_points=min_points,
            progress_files=progress_files,
            max_sidecar_files=max_sidecar_files,
        )

    all_results = resolved + density_resolved + exact_results
    keep_count = sum(1 for r in all_results if r["decision"] == "keep")
    reason_counts: dict[str, int] = {}
    for r in all_results:
        reason_counts[r["reason"]] = reason_counts.get(r["reason"], 0) + 1

    results_path = output_dir / "small_shape_ais_activity_results.json"
    summary = {
        "candidates_tested": len(all_results),
        "candidates_recommended_keep": keep_count,
        "candidates_recommended_remove": len(all_results) - keep_count,
        "reason_counts": reason_counts,
        "min_valid_area_m2": min_valid_area_m2,
        "auto_keep_area_m2": auto_keep_area_m2,
        "density_fast_keep_threshold": density_fast_keep_threshold,
        "min_points_threshold": min_points,
        "density_prepass_stats": prepass_stats,
        "exact_pipeline_stats": scan_stats,
        "per_candidate": [{k: v for k, v in r.items() if k != "wgs84_geom"} for r in all_results],
    }
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_candidate"}, indent=2))

    output_gpkg = output_dir / "small_shapes_with_ais_activity.gpkg"
    output_gpkg.unlink(missing_ok=True)
    schema = {
        "geometry": "Point",
        "properties": OrderedDict(
            [
                ("source_fid", "str"),
                ("native_area_m2", "float"),
                ("decision", "str"),
                ("reason", "str"),
                ("ais_point_count", "int"),
                ("density_grid_sum", "int"),
            ]
        ),
    }
    with fiona.open(str(output_gpkg), "w", driver="GPKG", crs="EPSG:4326", schema=schema) as dst:
        for i, r in enumerate(all_results):
            lon, lat = r["wgs84_centroid"]
            dst.write(
                {
                    "geometry": shapely_mapping(Point(lon, lat)),
                    "properties": {
                        "source_fid": str(r["fid"]) if r["fid"] is not None else str(i),
                        "native_area_m2": r["native_area_m2"],
                        "decision": r["decision"],
                        "reason": r["reason"],
                        "ais_point_count": (
                            r["ais_point_count"] if r["ais_point_count"] is not None else -1
                        ),
                        "density_grid_sum": (
                            r["density_grid_sum"] if r["density_grid_sum"] is not None else -1
                        ),
                    },
                }
            )
    print(f"Wrote {results_path} and {output_gpkg}")


def _area_threshold(value: str) -> float | None:
    if value.strip().lower() in ("none", "off", "disable", "disabled"):
        return None
    return float(value)


def _optional_int(value: str) -> int | None:
    if value.strip().lower() in ("none", "off", "disable", "disabled"):
        return None
    return int(value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-gpkg", required=True)
    parser.add_argument("--layer", default=None)
    parser.add_argument("--grid-assignment-dir", required=True)
    parser.add_argument("--ais-source-dir", required=True)
    parser.add_argument("--vessel-hot-parquet", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--min-valid-area-m2", type=float, default=1.0,
        help="Shapes below this are degenerate slivers, auto-removed with no testing.",
    )
    parser.add_argument(
        "--auto-keep-area-m2", type=_area_threshold, default=10_000_000.0,
        help="Shapes at/above this are auto-kept with no testing. Pass 'none' to disable.",
    )
    parser.add_argument(
        "--density-fast-keep-threshold", type=_optional_int, default=75,
        help="Middle-tier candidates with a targeted density-pass sum at/above this are kept "
        "without the exact pipeline. Pass 'none' to skip the density pre-pass entirely and "
        "send every middle-tier candidate straight to the exact pipeline.",
    )
    parser.add_argument("--min-points", type=int, default=10)
    parser.add_argument("--progress-files", type=int, default=50)
    parser.add_argument(
        "--max-sidecar-files", type=int, default=None,
        help="For a quick partial test run before scanning all sidecar files.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _prevent_sleep()
    try:
        run(
            candidates_gpkg=Path(args.candidates_gpkg),
            layer=args.layer,
            grid_assignment_dir=Path(args.grid_assignment_dir),
            ais_source_dir=Path(args.ais_source_dir),
            vessel_hot_parquet=Path(args.vessel_hot_parquet),
            output_dir=Path(args.output_dir),
            min_valid_area_m2=args.min_valid_area_m2,
            auto_keep_area_m2=args.auto_keep_area_m2,
            density_fast_keep_threshold=args.density_fast_keep_threshold,
            min_points=args.min_points,
            progress_files=args.progress_files,
            max_sidecar_files=args.max_sidecar_files,
        )
    finally:
        _allow_sleep()


if __name__ == "__main__":
    main()
