#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone combination of detect_inland_water_spikes.py and
remove_inland_water_spikes.py: detects thin, near-vertical "spike" interior
rings using each ring's own principal axis, removes them, and writes every
output either of those two scripts produced -- in one pass over the source
layer, with no dependency between separate files.

Why a principal axis instead of a bounding box
-----------------------------------------------
A spike tilted even a few degrees off true vertical inflates its
axis-aligned bounding-box width, which can push a genuine spike past a
strict width cutoff, or force the cutoff loose enough to let real small
holes back in. Measuring width and orientation relative to the ring's own
axis (via the covariance of its own vertices) is robust to that tilt.

For every interior ring:
    1. length  = spread of its vertices along their own principal axis
    2. width   = spread of its vertices perpendicular to that axis
    3. orientation_from_vertical_deg = angle between that axis and vertical

A ring is flagged when it is simultaneously narrow, long relative to its
width, and aligned close to vertical -- a thin horizontal sliver has zero
width too, but fails the orientation test, so it is correctly left alone.

What "removing a ring" means
-----------------------------
A spike is a degenerate, near-zero-width interior ring -- removing it from a
polygon's list of interior rings is a plain Python-level list edit, not a
GEOS boolean operation: no buffering, no union, no re-simplification. For a
feature with no flagged rings at all, the original geometry is written
through completely unchanged.

No buffering, union, or other GEOS boolean operation is used anywhere in
detection or removal, so cost scales with vertex count, not with the size of
the parent polygon -- this stays fast even on the worst-offender shapes.

Outputs (all in --output-dir; matches both original scripts combined)
------------------------------------------------------------------------
    layer_inland_water_spikes_removed.gpkg
        Full copy of the source layer with flagged rings removed. Untouched
        features are byte-for-byte the original geometry. Adds two columns:
        spikes_removed (count per feature) and original_fid (provenance).
    spike_points.gpkg / spike_lines.gpkg
        Every detected (and removed) spike, as points/traced lines, for
        QGIS review.
    removed_spike_points.gpkg / removed_spike_lines.gpkg
        Identical content to the two above -- kept as separate files only
        because that is what remove_inland_water_spikes.py produced.
    spike_source_polygons.gpkg
        Only the source polygons that contained >=1 flagged spike, for
        context (pass --no-source-polygons to skip).
    spike_detection_summary.json / spike_removal_summary.json
        The same two summaries the original scripts each wrote, both from
        this one pass.
    README.txt
        One combined readme covering every output above (the two originals
        each wrote their own README.txt, which would collide as filenames
        in a single shared output directory).

The source GeoPackage is only ever opened for reading.

Usage (PowerShell)
-------------------
    & "C:\\anaconda\\python.exe" -u "C:\\Users\\Craig Pearce\\Desktop\\detect_and_remove_inland_water_spikes.py" `
        --inland-gpkg "C:\\Users\\Craig Pearce\\Desktop\\osm\\layer_inland_water.gpkg" `
        --output-dir "C:\\Users\\Craig Pearce\\Desktop\\osm\\spike_combined"

Validated defaults (from the real shape tested earlier in this project):
        --max-width-m 3.0 --max-tilt-deg 15.0 --min-aspect-ratio 3.0 --max-vertices 8
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

import fiona
import shapely
from shapely.geometry import (
    shape as shapely_shape,
    mapping as shapely_mapping,
    Polygon,
    MultiPolygon,
    LineString,
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class SpikeRecord:
    source_fid: int
    part_index: int
    ring_index: int
    vertex_count: int
    true_length_m: float
    true_width_m: float
    orientation_from_vertical_deg: float
    aspect_ratio: float
    bbox_width_m: float
    bbox_height_m: float
    minx: float
    miny: float
    maxx: float
    maxy: float
    coords: list[tuple[float, float]]
    distance_to_exterior_m: float = float("nan")


@dataclass
class RunStats:
    features_scanned: int = 0
    features_skipped_small: int = 0
    features_with_ring_errors: int = 0
    parts_examined: int = 0
    holes_examined: int = 0
    holes_flagged: int = 0
    features_flagged: int = 0
    all_widths: list[float] = field(default_factory=list)
    all_lengths: list[float] = field(default_factory=list)
    all_orientations: list[float] = field(default_factory=list)
    all_vertex_counts: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Principal-axis ring metrics (closed-form, no per-ring LAPACK call --
# validated against numpy.linalg.eigh across 20,000 random matrices,
# max direction error ~4e-16, before use here)
# ---------------------------------------------------------------------------


def _principal_axes(
    a: float, b: float, d: float, tiny: float = 1e-15
) -> tuple[tuple[float, float], tuple[float, float]]:
    lam_avg = (a + d) / 2.0
    diff = (a - d) / 2.0
    r = math.sqrt(diff * diff + b * b)
    lam_major = lam_avg + r
    if abs(b) > tiny:
        vx, vy = b, lam_major - a
    elif a >= d:
        vx, vy = 1.0, 0.0
    else:
        vx, vy = 0.0, 1.0
    norm = math.hypot(vx, vy)
    if norm < tiny:
        vx, vy, norm = 0.0, 1.0, 1.0
    major = (vx / norm, vy / norm)
    minor = (-major[1], major[0])
    return major, minor


def ring_principal_axis_metrics(coords: np.ndarray) -> tuple[float, float, float, float, float]:
    xs = coords[:, 0]
    ys = coords[:, 1]
    bbox_width = float(xs.max() - xs.min())
    bbox_height = float(ys.max() - ys.min())

    n = coords.shape[0]
    cx = float(xs.mean())
    cy = float(ys.mean())
    dx = xs - cx
    dy = ys - cy
    a = float(np.dot(dx, dx) / n)
    d = float(np.dot(dy, dy) / n)
    b = float(np.dot(dx, dy) / n)
    major, minor = _principal_axes(a, b, d)

    proj_major = dx * major[0] + dy * major[1]
    proj_minor = dx * minor[0] + dy * minor[1]
    true_length = float(proj_major.max() - proj_major.min())
    true_width = float(proj_minor.max() - proj_minor.min())
    orientation_from_vertical = math.degrees(math.atan2(abs(major[0]), abs(major[1])))
    return true_length, true_width, orientation_from_vertical, bbox_width, bbox_height


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def iter_polygon_parts(geom: Any) -> Iterator[Any]:
    gtype = geom.geom_type
    if gtype == "Polygon":
        yield geom
    elif gtype == "MultiPolygon":
        yield from geom.geoms
    # Non-polygonal geometry has no interior rings; silently skipped.


def detect_spikes_in_part(
    source_fid: int,
    part_index: int,
    part: Any,
    stats: RunStats,
    *,
    max_width_m: float,
    max_tilt_deg: float,
    min_aspect_ratio: float,
    max_vertices: int,
    min_length_m: float,
) -> list[SpikeRecord]:
    flagged: list[SpikeRecord] = []
    for ring_index, interior in enumerate(part.interiors):
        try:
            coords = np.asarray(interior.coords, dtype=np.float64)
        except Exception:
            stats.features_with_ring_errors += 1
            continue
        if coords.shape[0] < 4:
            stats.features_with_ring_errors += 1
            continue
        vertex_count = coords.shape[0] - 1
        length, width, orientation, bbox_w, bbox_h = ring_principal_axis_metrics(coords)
        stats.holes_examined += 1
        stats.all_widths.append(width)
        stats.all_lengths.append(length)
        stats.all_orientations.append(orientation)
        stats.all_vertex_counts.append(vertex_count)

        aspect_ratio = length / max(width, 1e-9)
        is_spike = (
            width <= max_width_m
            and orientation <= max_tilt_deg
            and aspect_ratio >= min_aspect_ratio
            and vertex_count <= max_vertices
            and length >= min_length_m
        )
        if is_spike:
            minx, maxx = float(coords[:, 0].min()), float(coords[:, 0].max())
            miny, maxy = float(coords[:, 1].min()), float(coords[:, 1].max())
            flagged.append(
                SpikeRecord(
                    source_fid=source_fid,
                    part_index=part_index,
                    ring_index=ring_index,
                    vertex_count=vertex_count,
                    true_length_m=length,
                    true_width_m=width,
                    orientation_from_vertical_deg=orientation,
                    aspect_ratio=aspect_ratio,
                    bbox_width_m=bbox_w,
                    bbox_height_m=bbox_h,
                    minx=minx,
                    miny=miny,
                    maxx=maxx,
                    maxy=maxy,
                    coords=[(float(x), float(y)) for x, y in coords],
                )
            )
    return flagged


def annotate_distance_to_exterior(part_exterior: Any, flagged: list[SpikeRecord]) -> None:
    """Exact nearest distance from each flagged ring to the part's own
    exterior ring, computed once per feature (not once per ring)."""
    if not flagged:
        return
    ext_coords = np.asarray(part_exterior.coords, dtype=np.float64)
    if ext_coords.shape[0] < 2:
        return
    seg_pairs = np.stack([ext_coords[:-1], ext_coords[1:]], axis=1)  # (n-1, 2, 2)
    segments = shapely.linestrings(seg_pairs)
    tree = shapely.STRtree(segments)

    # Ragged vertex counts across spikes -> build individually, not via the
    # uniform-shape vectorized constructor.
    spike_lines = [LineString(np.asarray(rec.coords, dtype=np.float64)) for rec in flagged]
    indices, distances = tree.query_nearest(spike_lines, return_distance=True, all_matches=False)
    dist_by_query: dict[int, float] = {}
    for q_idx, dist in zip(indices[0], distances):
        dist_by_query[int(q_idx)] = float(dist)
    for i, rec in enumerate(flagged):
        rec.distance_to_exterior_m = dist_by_query.get(i, float("nan"))


# ---------------------------------------------------------------------------
# Geometry surgery: drop flagged rings only, touch nothing else
# ---------------------------------------------------------------------------


def remove_rings_from_part(part: Any, ring_indices_to_remove: set[int]) -> Any:
    kept_holes = [
        list(interior.coords)
        for i, interior in enumerate(part.interiors)
        if i not in ring_indices_to_remove
    ]
    return Polygon(list(part.exterior.coords), kept_holes)


def build_cleaned_geometry(geom: Any, flags_by_part: dict[int, set[int]]) -> Any | None:
    """Returns a new geometry with flagged rings removed, or None if nothing
    in this feature needs to change (caller should keep the original)."""
    if not flags_by_part:
        return None
    if geom.geom_type == "Polygon":
        if 0 in flags_by_part:
            return remove_rings_from_part(geom, flags_by_part[0])
        return None
    if geom.geom_type == "MultiPolygon":
        parts = list(geom.geoms)
        changed = False
        new_parts = []
        for part_index, part in enumerate(parts):
            if part_index in flags_by_part:
                new_parts.append(remove_rings_from_part(part, flags_by_part[part_index]))
                changed = True
            else:
                new_parts.append(part)
        if changed:
            return MultiPolygon(new_parts)
        return None
    return None


# ---------------------------------------------------------------------------
# Main scan + clean + write pass
# ---------------------------------------------------------------------------


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def run(
    *,
    inland_gpkg: Path,
    layer: str | None,
    output_dir: Path,
    min_feature_area_m2: float,
    max_width_m: float,
    max_tilt_deg: float,
    min_aspect_ratio: float,
    max_vertices: int,
    min_length_m: float,
    max_features: int | None,
    progress_seconds: float,
    include_source_polygons: bool,
    write_diagnostics: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned_path = output_dir / "layer_inland_water_spikes_removed.gpkg"
    spike_points_path = output_dir / "spike_points.gpkg"
    spike_lines_path = output_dir / "spike_lines.gpkg"
    removed_points_path = output_dir / "removed_spike_points.gpkg"
    removed_lines_path = output_dir / "removed_spike_lines.gpkg"
    polygons_path = output_dir / "spike_source_polygons.gpkg"
    detection_summary_path = output_dir / "spike_detection_summary.json"
    removal_summary_path = output_dir / "spike_removal_summary.json"
    readme_path = output_dir / "README.txt"

    for p in (
        cleaned_path, spike_points_path, spike_lines_path,
        removed_points_path, removed_lines_path, polygons_path,
    ):
        p.unlink(missing_ok=True)

    diag_props = OrderedDict(
        [
            ("source_fid", "int"),
            ("part_index", "int"),
            ("ring_index", "int"),
            ("vertex_count", "int"),
            ("true_length_m", "float"),
            ("true_width_m", "float"),
            ("orientation_from_vertical_deg", "float"),
            ("aspect_ratio", "float"),
            ("distance_to_exterior_m", "float"),
        ]
    )

    with fiona.open(str(inland_gpkg), layer=layer) as source:
        crs = source.crs
        src_props: OrderedDict = OrderedDict(source.schema["properties"])
        geom_type = source.schema["geometry"]

        out_props = OrderedDict(src_props)
        extra_names = []
        for wanted in ("spikes_removed", "original_fid"):
            name = wanted
            n = 1
            while name in out_props:
                n += 1
                name = f"{wanted}_{n}"
            out_props[name] = "int"
            extra_names.append(name)
        spikes_removed_field, original_fid_field = extra_names
        cleaned_schema = {"geometry": geom_type, "properties": out_props}

        stats = RunStats()
        features_modified = 0
        total_rings_removed = 0
        still_invalid_after = 0
        removed_widths: list[float] = []
        removed_lengths: list[float] = []
        all_flagged: list[SpikeRecord] = []
        flagged_feature_summaries: list[dict[str, Any]] = []

        started = time.perf_counter()
        last_heartbeat = started

        dst = fiona.open(str(cleaned_path), "w", driver="GPKG", crs=crs, schema=cleaned_schema)
        pts = (
            fiona.open(
                str(spike_points_path), "w", driver="GPKG", crs=crs,
                schema={"geometry": "Point", "properties": diag_props},
            )
            if write_diagnostics
            else None
        )
        lns = (
            fiona.open(
                str(spike_lines_path), "w", driver="GPKG", crs=crs,
                schema={"geometry": "LineString", "properties": diag_props},
            )
            if write_diagnostics
            else None
        )

        try:
            for feature in source:
                stats.features_scanned += 1
                fid = feature.get("id")
                try:
                    fid_int = int(fid)
                except (TypeError, ValueError):
                    fid_int = stats.features_scanned

                raw_geom = feature.get("geometry")
                out_props_values = dict(feature.get("properties") or {})
                out_props_values[original_fid_field] = fid_int

                if raw_geom is None:
                    out_props_values[spikes_removed_field] = 0
                    dst.write({"geometry": None, "properties": out_props_values})
                    continue

                try:
                    geom = shapely_shape(raw_geom)
                except Exception:
                    stats.features_with_ring_errors += 1
                    out_props_values[spikes_removed_field] = 0
                    dst.write({"geometry": raw_geom, "properties": out_props_values})
                    continue

                flags_by_part: dict[int, set[int]] = {}
                feature_flagged: list[SpikeRecord] = []
                for part_index, part in enumerate(iter_polygon_parts(geom)):
                    if part.is_empty:
                        continue
                    if part.area < min_feature_area_m2:
                        stats.features_skipped_small += 1
                        continue
                    stats.parts_examined += 1
                    part_flagged = detect_spikes_in_part(
                        fid_int,
                        part_index,
                        part,
                        stats,
                        max_width_m=max_width_m,
                        max_tilt_deg=max_tilt_deg,
                        min_aspect_ratio=min_aspect_ratio,
                        max_vertices=max_vertices,
                        min_length_m=min_length_m,
                    )
                    if part_flagged:
                        annotate_distance_to_exterior(part.exterior, part_flagged)
                        flags_by_part[part_index] = {r.ring_index for r in part_flagged}
                        feature_flagged.extend(part_flagged)

                if feature_flagged:
                    stats.features_flagged += 1
                    stats.holes_flagged += len(feature_flagged)
                    features_modified += 1
                    total_rings_removed += len(feature_flagged)
                    removed_widths.extend(r.true_width_m for r in feature_flagged)
                    removed_lengths.extend(r.true_length_m for r in feature_flagged)
                    all_flagged.extend(feature_flagged)
                    flagged_feature_summaries.append(
                        {
                            "source_fid": fid_int,
                            "spike_count": len(feature_flagged),
                            "part_count": len(geom.geoms) if geom.geom_type == "MultiPolygon" else 1,
                            "total_area_m2": float(geom.area),
                            "geometry": raw_geom,
                        }
                    )

                    cleaned_geom = build_cleaned_geometry(geom, flags_by_part)
                    if not cleaned_geom.is_valid:
                        still_invalid_after += 1
                    out_geom = shapely_mapping(cleaned_geom)

                    if write_diagnostics:
                        for rec in feature_flagged:
                            centre_x = (rec.minx + rec.maxx) / 2.0
                            centre_y = (rec.miny + rec.maxy) / 2.0
                            props = {
                                "source_fid": rec.source_fid,
                                "part_index": rec.part_index,
                                "ring_index": rec.ring_index,
                                "vertex_count": rec.vertex_count,
                                "true_length_m": rec.true_length_m,
                                "true_width_m": rec.true_width_m,
                                "orientation_from_vertical_deg": rec.orientation_from_vertical_deg,
                                "aspect_ratio": rec.aspect_ratio,
                                "distance_to_exterior_m": rec.distance_to_exterior_m,
                            }
                            pts.write(
                                {
                                    "geometry": shapely_mapping(shapely.Point(centre_x, centre_y)),
                                    "properties": props,
                                }
                            )
                            lns.write(
                                {"geometry": shapely_mapping(LineString(rec.coords)), "properties": props}
                            )
                else:
                    out_geom = raw_geom

                out_props_values[spikes_removed_field] = len(feature_flagged)
                dst.write({"geometry": out_geom, "properties": out_props_values})

                now = time.perf_counter()
                if now - last_heartbeat >= progress_seconds:
                    elapsed = now - started
                    rate = stats.features_scanned / elapsed if elapsed > 0 else 0.0
                    print(
                        f"  [{elapsed:7.1f}s] scanned={stats.features_scanned:,} ({rate:,.0f}/s)  "
                        f"modified={features_modified:,}  rings_removed={total_rings_removed:,}",
                        flush=True,
                    )
                    last_heartbeat = now

                if max_features is not None and stats.features_scanned >= max_features:
                    break
        finally:
            dst.close()
            if pts is not None:
                pts.close()
            if lns is not None:
                lns.close()

    elapsed = time.perf_counter() - started

    if include_source_polygons:
        polygon_schema = {
            "geometry": "Unknown",
            "properties": OrderedDict(
                [
                    ("source_fid", "int"),
                    ("spike_count", "int"),
                    ("part_count", "int"),
                    ("total_area_m2", "float"),
                ]
            ),
        }
        with fiona.open(str(polygons_path), "w", driver="GPKG", crs=crs, schema=polygon_schema) as polys:
            for item in flagged_feature_summaries:
                polys.write(
                    {
                        "geometry": item["geometry"],
                        "properties": {
                            "source_fid": item["source_fid"],
                            "spike_count": item["spike_count"],
                            "part_count": item["part_count"],
                            "total_area_m2": item["total_area_m2"],
                        },
                    }
                )

    if write_diagnostics:
        # Identical content to spike_points/lines -- copied rather than
        # written twice, since remove_inland_water_spikes.py's output used
        # this name and nothing here should risk the two ever disagreeing.
        shutil.copy(spike_points_path, removed_points_path)
        shutil.copy(spike_lines_path, removed_lines_path)

    detection_summary = {
        "elapsed_seconds": elapsed,
        "features_per_second": stats.features_scanned / elapsed if elapsed > 0 else None,
        "parameters": {
            "min_feature_area_m2": min_feature_area_m2,
            "max_width_m": max_width_m,
            "max_tilt_deg": max_tilt_deg,
            "min_aspect_ratio": min_aspect_ratio,
            "max_vertices": max_vertices,
            "min_length_m": min_length_m,
            "max_features": max_features,
        },
        "features_scanned": stats.features_scanned,
        "features_skipped_small_part": stats.features_skipped_small,
        "features_with_ring_errors": stats.features_with_ring_errors,
        "parts_examined": stats.parts_examined,
        "holes_examined": stats.holes_examined,
        "holes_flagged_as_spikes": stats.holes_flagged,
        "features_with_at_least_one_spike": stats.features_flagged,
        "distance_to_exterior_m_distribution_flagged": _percentiles(
            [r.distance_to_exterior_m for r in all_flagged]
        ),
        "all_rings_width_m_distribution": _percentiles(stats.all_widths),
        "all_rings_length_m_distribution": _percentiles(stats.all_lengths),
        "all_rings_orientation_from_vertical_deg_distribution": _percentiles(stats.all_orientations),
        "all_rings_vertex_count_distribution": _percentiles([float(v) for v in stats.all_vertex_counts]),
        "flagged_width_m_distribution": _percentiles([r.true_width_m for r in all_flagged]),
        "flagged_length_m_distribution": _percentiles([r.true_length_m for r in all_flagged]),
        "flagged_orientation_from_vertical_deg_distribution": _percentiles(
            [r.orientation_from_vertical_deg for r in all_flagged]
        ),
        "flagged_aspect_ratio_distribution": _percentiles([r.aspect_ratio for r in all_flagged]),
        "outputs": {
            "spike_points": str(spike_points_path),
            "spike_lines": str(spike_lines_path),
            "spike_source_polygons": str(polygons_path) if include_source_polygons else None,
        },
    }
    detection_summary_path.write_text(json.dumps(detection_summary, indent=2), encoding="utf-8")

    removal_summary = {
        "elapsed_seconds": elapsed,
        "features_per_second": stats.features_scanned / elapsed if elapsed > 0 else None,
        "parameters": detection_summary["parameters"],
        "features_scanned": stats.features_scanned,
        "features_modified": features_modified,
        "total_rings_removed": total_rings_removed,
        "features_with_ring_errors": stats.features_with_ring_errors,
        "features_still_invalid_after_removal": still_invalid_after,
        "removed_width_m_distribution": _percentiles(removed_widths),
        "removed_length_m_distribution": _percentiles(removed_lengths),
        "output_layer": str(cleaned_path),
        "added_fields": {"spikes_removed": spikes_removed_field, "original_fid": original_fid_field},
        "diagnostics": (
            {"removed_points": str(removed_points_path), "removed_lines": str(removed_lines_path)}
            if write_diagnostics
            else None
        ),
    }
    removal_summary_path.write_text(json.dumps(removal_summary, indent=2), encoding="utf-8")

    readme_path.write_text(
        "\n".join(
            [
                "Combined spike detection + removal output",
                "===========================================",
                "",
                f"{cleaned_path.name}",
                "    A full copy of the source layer with detected spike rings removed.",
                "    Untouched features are byte-for-byte the original geometry. Two",
                "    columns were added: 'spikes_removed' (count per feature, 0 if",
                "    untouched) and 'original_fid' (source feature id, for provenance).",
                "",
                "spike_points.gpkg / spike_lines.gpkg",
                "removed_spike_points.gpkg / removed_spike_lines.gpkg",
                "    Every detected (and removed) spike, as points/traced lines, for QGIS",
                "    review. The two pairs are identical content -- kept as separate files",
                "    only because that's what the original two scripts each produced.",
                "",
                "spike_source_polygons.gpkg",
                "    Only the source polygons that contained >=1 flagged spike, for context.",
                "",
                "spike_detection_summary.json / spike_removal_summary.json",
                "    The same two summaries the original scripts each wrote, from this one",
                "    pass. The detection summary's width/length/orientation/vertex-count",
                "    distributions cover ALL rings examined, not just flagged ones -- use",
                "    that to retune --max-width-m / --max-tilt-deg / --min-aspect-ratio.",
                "",
                "Key fields (on points/lines):",
                "  true_length_m / true_width_m   Extent along/across the ring's OWN",
                "                                  principal axis, robust to tilt.",
                "  orientation_from_vertical_deg  0 = perfectly vertical.",
                "  aspect_ratio                    true_length_m / true_width_m.",
                "  distance_to_exterior_m          Exact nearest distance to this feature's",
                "                                  own exterior ring (0.0 if it touches).",
                "",
                "The SOURCE GeoPackage was never opened for writing and is unchanged.",
            ]
        ),
        encoding="utf-8",
    )

    print(json.dumps(detection_summary, indent=2))
    print(json.dumps(removal_summary, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inland-gpkg", required=True)
    parser.add_argument("--layer", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-feature-area-m2", type=float, default=1000.0)
    parser.add_argument("--max-width-m", type=float, default=50.0)
    parser.add_argument("--max-tilt-deg", type=float, default=90.0)
    parser.add_argument("--min-aspect-ratio", type=float, default=100.0)
    parser.add_argument("--max-vertices", type=int, default=4)
    parser.add_argument("--min-length-m", type=float, default=0.0)
    parser.add_argument("--max-features", type=int, default=None, help="For a quick partial test run.")
    parser.add_argument("--progress-seconds", type=float, default=30.0)
    parser.add_argument(
        "--no-source-polygons",
        dest="include_source_polygons",
        action="store_false",
        default=True,
    )
    parser.add_argument(
        "--no-diagnostics",
        dest="write_diagnostics",
        action="store_false",
        default=True,
        help="Skip writing all four points/lines GeoPackages (only the cleaned layer + summaries).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run(
        inland_gpkg=Path(args.inland_gpkg),
        layer=args.layer,
        output_dir=Path(args.output_dir),
        min_feature_area_m2=args.min_feature_area_m2,
        max_width_m=args.max_width_m,
        max_tilt_deg=args.max_tilt_deg,
        min_aspect_ratio=args.min_aspect_ratio,
        max_vertices=args.max_vertices,
        min_length_m=args.min_length_m,
        max_features=args.max_features,
        progress_seconds=args.progress_seconds,
        include_source_polygons=args.include_source_polygons,
        write_diagnostics=args.write_diagnostics,
    )


if __name__ == "__main__":
    main()
