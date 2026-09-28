# process_inland_water_mask

## Scripts in order of workflow
- 0_osm_inland_water_subset.py
- 1_output_inland_water_propagation.py
- 2_detect_and_remove_inland_water_spikes.py
- 3_test_small_shapes_for_ais_activity.py

## Purpose of each script
### 0_osm_inland_water_subset.py
Retains inland-water polygons using three sequential evidence routes: SVP overlap, GSW + Boolean density map (all vessels, zoom 9) activity, or GSW + at least 10 raw AIS points. Everything else is discarded.

### 1_output_inland_water_propagation.py
The script takes the filtered inland-water output from the previous process, then expands the new output through using the original high-resolution OSM geometry wherever polygons are connected within one meter. This ensures that the final output does not have incomplete water bodies.

### 2_detect_and_remove_inland_water_spikes.py
The script detects thin near-vertical interior rings using principal-axis geometry measurements, and removes only the rings that meet all spike criteria. It preserves the original source geometry, and produces a new cleaned output and also separate visuals of the points (vertices) and lines removed to facilitate a subjective/visual assessment via QGIS.

### 3_test_small_shapes_for_ais_activity.py
This script is an extra small shape filter that applies a more targeted and speed optimised approach: it first uses shape size (<100m2) and then 0.01°/0.001° AIS aggregation grid sidecars to cheaply test for AIS grind containment, then uses exact raw-AIS point tests only for unresolved candidates. This makes it possible to assess whether tiny inland-water shapes have genuine vessel activity without scanning the entire AIS archive unnecessarily, then output a new layer with unwanted small shapes filtered out.

## Dependencies
The four scripts use a shared Python geospatial/data-processing stack comprising
GeoPandas, Shapely, Pyogrio/Fiona, PyProj, NumPy, Pandas, PyArrow, Rasterio and
Pillow. They also use Python's standard libraries for SQLite state management,
JSON reporting, file handling, batching, logging and process control.

## Inputs
The input for 0_osm_inland_water_subset.py is the original inland waters layer, which could be subjected to initial cleaning to facilitate speed optimisation e.g. removing duplicate vertices or geometry simplifying. The output inland waters layer of each script can then be used as the input layer for the next script in the sequence.

## Outputs
Each script produces a new geopackage output of the inland waters layer plus visualisations (where applicable) that help with subjective assessments.

## Logical summary for all scripts
High-resolution OSM inland water
            |
            v
[1] Vessel-evidence filtering
    SVP
     OR
    GSW + Boolean AIS activity
     OR
    GSW + raw AIS >= 10 points
            |
            v
Vessel-validated inland-water subset
            |
            v
[2] Connectivity expansion
    Add source polygons that intersect/touch
    or are within 1 m of selected polygons
    Repeat until no new polygons are found
            |
            v
Connected high-resolution inland-water subset
            |
            v
[3] Spike detection/removal
    Identify thin, near-vertical interior rings
    using width + orientation + aspect ratio + vertex count
            |
            v
Cleaned inland-water geometries
            |
            v
[4] Small-shape AIS validation
    Size-based filtering
     -> 0.01° / 0.001° grid cell selection
     -> AIS density pre-pass
     -> exact AIS + vessel legitimacy test
            |
            v
Final inland-water dataset


## Performance and optimisation

### 0_osm_inland_water_subset.py
Uses RTree/STRtree spatial indexing, adaptive tiling, progressive evidence filtering, lazy raster caching, AIS cell indexing and Parquet row-group elimination to minimise geometry and AIS processing while keeping memory bounded.

### 1_output_inland_water_propagation.py
Uses an RTree + queue-based graph search + STRtree to expand only from the current connected frontier, avoiding repeated full-layer scans and keeping the geometry workload focused on nearby candidates.

### 2_detect_and_remove_inland_water_spikes.py
Uses a single streaming pass and lightweight mathematical geometry tests, avoiding expensive GEOS operations so spike detection/removal scales mainly with polygon vertex count.

### 3_test_small_shapes_for_ais_activity.py
Uses the existing 0.01°/0.001° AIS aggregation grid as a coarse-to-fine spatial index, with a cheap sidecar density pass followed by selective raw-AIS retrieval, vessel validation and exact geometry testing.

## Restart / recovery and validation

### 0_osm_inland_water_subset.py
It checks that required inputs and dependencies exist, verifies CRS and required R-trees, validates MBTiles structure/encoding, checks AIS configuration, records tile failures, and requires all processing tiles and the AIS scan to be complete before final output.

### 1_output_inland_water_propagation.py
This validates the source and seed datasets before processing, including CRS, required FID fields, and the presence of the source GeoPackage RTree. It also uses a processing signature to prevent an existing SQLite state database from being reused with different inputs or settings. During processing it detects missing source/candidate geometries and unresolved seed FIDs. The final report also provides counts by depth, evidence type, processed/unprocessed state and total selected geometries.

### 2_detect_and_remove_inland_water_spikes.py
This script checks malformed rings, invalid tracks or unusable geometry cases, and counts whether cleaned geometries remain invalid after spike removal.
Its JSON summaries provide distributions of widths, lengths, orientations and vertex counts for both examined and flagged rings.

### 3_test_small_shapes_for_ais_activity.py
Validates that the required sidecars/AIS files exist and reports how many rows, legitimate vessels and exact AIS matches were found.
