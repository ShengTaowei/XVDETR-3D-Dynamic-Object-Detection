#!/usr/bin/env python3
"""
Generate satellite-imagery AOI shapefiles for the nuScenes dataset.

WHY THIS SCRIPT EXISTS
----------------------
nuScenes does **not** ship per-frame GPS.  Every location (``boston-seaport``,
``singapore-onenorth``, ``singapore-hollandvillage``, ``singapore-queenstown``)
uses its own local metric "global" frame whose origin sits at the
south-western corner of that location's map.  The only official geo-anchor is
the latitude/longitude of that corner, published by the dev-kit in

    nuscenes-devkit/python-sdk/nuscenes/scripts/export_poses.py
        :: REFERENCE_COORDINATES

Combined with the documented assumption that the local frame is axis aligned
(``x -> East``, ``y -> North``), that anchor is enough to project any ego pose
onto the WGS84 ellipsoid.  This script

    1. reads the raw nuScenes JSON tables directly (no dev-kit, no images,
       no GPU),
    2. projects every key frame of the requested camera to lat/lon,
    3. writes one AOI polygon per location -- optionally exploded into a
       regular grid of downloadable tiles -- as an ESRI shapefile in
       EPSG:4326, which is what Google Earth Pro expects,
    4. dumps per-frame lat/lon + heading to CSV.  That CSV is the input the
       satellite cropping step (cf. ``tools/crop_satellite_tiles.py``) needs
       to cut one geo-aligned patch per frame.

ACCURACY WARNING
----------------
The projection is only as good as the "x = East, y = North" assumption; the
community reports offsets on the order of a few tens of metres.  That is
harmless when *downloading* imagery (use ``--margin-m`` to add slack) but it
matters for pixel-accurate cropping, so after the first download always verify
with ``tools/visualize_sat_alignment.py``.  If a systematic offset shows up,
hand-tune ``REFERENCE_COORDINATES`` (or add a per-location ``(dx, dy)`` shift)
before mass-producing the crops.

Another practical note: nuScenes was collected in 2018.  In Google Earth Pro,
switch on the historical imagery slider and pick the ~2018 frame so that the
road/building layout matches the recordings.

USAGE
-----
    # one AOI polygon per location, 300 m of slack
    python tools/nuscenes_satellite_regions.py \
        --dataroot /data/sets/nuscenes --version v1.0-trainval \
        --out-dir data/nuScenes/satellite_regions

    # same, but exploded into ~600 m tiles with 50 m overlap (Google Earth Pro
    # can only export ~4800x4800 px per shot, so large areas must be tiled)
    python tools/nuscenes_satellite_regions.py \
        --dataroot /data/sets/nuscenes --version v1.0-trainval \
        --out-dir data/nuScenes/satellite_regions --tile-m 600 --overlap-m 50

Memory: the loader parses ``sample_data.json`` and ``ego_pose.json`` once; for
the full ``v1.0-trainval`` that peaks at a few GB of RAM.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Geo anchoring
# --------------------------------------------------------------------------- #
# South-western corner of every nuScenes map.  Copied verbatim from
# nuscenes-devkit python-sdk/nuscenes/scripts/export_poses.py.
REFERENCE_COORDINATES: Dict[str, Tuple[float, float]] = {
    "boston-seaport": (42.336849169438615, -71.05785369873047),
    "singapore-onenorth": (1.2882100868743724, 103.78475189208984),
    "singapore-hollandvillage": (1.2993652317780957, 103.78217697143555),
    "singapore-queenstown": (1.2782562240223188, 103.76741409301758),
}

EARTH_RADIUS_METERS = 6.378137e6
# Kept consistent with the spherical projection above, so that the metre/degree
# conversions used for buffering and tiling match derive_latlon()'s scale.
M_PER_DEG_LAT = math.pi / 180.0 * EARTH_RADIUS_METERS


def m_per_deg_lon(lat: float) -> float:
    return M_PER_DEG_LAT * math.cos(math.radians(lat))


def get_coordinate(ref_lat: float, ref_lon: float, bearing: float,
                   dist: float) -> Tuple[float, float]:
    """Destination point given a reference point, a bearing and a distance.

    Same great-circle formula as the dev-kit (movable-type latlong).

    :param ref_lat: reference latitude in degrees.
    :param ref_lon: reference longitude in degrees.
    :param bearing: clockwise angle in radians from north to the target.
    :param dist: ground distance in metres.
    """
    lat, lon = math.radians(ref_lat), math.radians(ref_lon)
    angular_distance = dist / EARTH_RADIUS_METERS

    target_lat = math.asin(
        math.sin(lat) * math.cos(angular_distance)
        + math.cos(lat) * math.sin(angular_distance) * math.cos(bearing)
    )
    target_lon = lon + math.atan2(
        math.sin(bearing) * math.sin(angular_distance) * math.cos(lat),
        math.cos(angular_distance) - math.sin(lat) * math.sin(target_lat),
    )
    return math.degrees(target_lat), math.degrees(target_lon)


def local_to_latlon(location: str, x: float, y: float) -> Tuple[float, float]:
    """nuScenes local global-frame (x East, y North) -> WGS84 (lat, lon)."""
    if location not in REFERENCE_COORDINATES:
        raise KeyError(f"no geo reference for location '{location}'")
    ref_lat, ref_lon = REFERENCE_COORDINATES[location]
    bearing = math.atan2(x, y)          # clockwise from north
    distance = math.hypot(x, y)
    return get_coordinate(ref_lat, ref_lon, bearing, distance)


def quat_yaw(rotation: Sequence[float]) -> float:
    """Yaw of a nuScenes (w, x, y, z) quaternion, radians, CCW from +x (East)."""
    w, x, y, z = rotation
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# --------------------------------------------------------------------------- #
# nuScenes table loading
# --------------------------------------------------------------------------- #
def _load_json(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _iter_json_array(path: Path, buf_size: int = 1 << 22):
    """Stream the objects of a top-level JSON array one at a time.

    ``sample_data.json`` / ``ego_pose.json`` of the full trainval split are
    1.3 GB / 0.6 GB; materialising them with ``json.load`` costs an order of
    magnitude more RAM than the few fields we actually read, so we walk the
    array with ``raw_decode`` instead.
    """
    decoder = json.JSONDecoder()
    buffer = ""
    pos = 0
    with open(path, "r", encoding="utf-8") as handle:
        handle.read(1)  # opening '['
        while True:
            chunk = handle.read(buf_size)
            if not chunk:
                break
            buffer += chunk
            while True:
                while pos < len(buffer) and buffer[pos] in " \t\r\n,":
                    pos += 1
                if pos >= len(buffer):
                    break
                try:
                    obj, end = decoder.raw_decode(buffer, pos)
                except ValueError:
                    break  # object truncated by the buffer edge: read more
                yield obj
                pos = end
            buffer = buffer[pos:]
            pos = 0


def _table_dir(dataroot: Path, version: str) -> Path:
    candidates = [dataroot / version, dataroot / "v1.0-trainval" / version, dataroot]
    for candidate in candidates:
        if (candidate / "scene.json").is_file():
            return candidate
    raise SystemExit(f"error: cannot find nuScenes tables under {dataroot} "
                     f"(looked for {version}/scene.json)")


def _split_of_scene(scene_names_by_split: Dict[str, set], name: str) -> str:
    for split, names in scene_names_by_split.items():
        if name in names:
            return split
    return ""


def _load_splits() -> Dict[str, set]:
    """Train/val scene lists, via the dev-kit when it happens to be installed."""
    try:
        from nuscenes.utils.splits import create_splits_scenes  # type: ignore
        return {k: set(v) for k, v in create_splits_scenes(verbose=False).items()}
    except Exception:
        return {}


def load_frames(dataroot: Path, version: str, cam: str) -> List[dict]:
    """Return one record per key frame of ``cam`` with its location and pose."""
    root = _table_dir(dataroot, version)

    logs = {r["token"]: r for r in _load_json(root / "log.json")}
    scenes = {r["token"]: r for r in _load_json(root / "scene.json")}
    samples = {r["token"]: r for r in _load_json(root / "sample.json")}

    # sensor channel of every calibrated_sensor record
    sensor_channel = {r["token"]: r["channel"] for r in _load_json(root / "sensor.json")}
    calib_channel = {
        r["token"]: sensor_channel.get(r["sensor_token"])
        for r in _load_json(root / "calibrated_sensor.json")
    }

    # key frames of the requested camera: sample_token -> (ego_pose_token, ts)
    wanted: Dict[str, Tuple[str, float]] = {}
    for record in _iter_json_array(root / "sample_data.json"):
        if not record.get("is_key_frame", False):
            continue
        if calib_channel.get(record["calibrated_sensor_token"]) != cam:
            continue
        sample_token = record["sample_token"]
        # Prefer any key frame; duplicates across sweeps are harmless here.
        if sample_token not in wanted:
            wanted[sample_token] = (record["ego_pose_token"], float(record["timestamp"]) / 1e6)
    print(f"  {len(wanted)} {cam} key frames")

    needed = {token for token, _ in wanted.values()}
    poses = {}
    for record in _iter_json_array(root / "ego_pose.json"):
        token = record["token"]
        if token in needed:
            poses[token] = record
            if len(poses) == len(needed):
                break

    splits = _load_splits()
    frames: List[dict] = []
    for sample_token, (pose_token, timestamp) in wanted.items():
        pose = poses.get(pose_token)
        if pose is None:
            continue
        sample = samples.get(sample_token)
        if sample is None:
            continue
        scene = scenes.get(sample["scene_token"])
        if scene is None:
            continue
        location = logs[scene["log_token"]]["location"]
        x, y = float(pose["translation"][0]), float(pose["translation"][1])
        lat, lon = local_to_latlon(location, x, y)
        yaw = quat_yaw(pose["rotation"])
        frames.append({
            "sample_token": sample_token,
            "scene_name": scene["name"],
            "split": _split_of_scene(splits, scene["name"]),
            "location": location,
            "timestamp": timestamp,
            "x": x,
            "y": y,
            "yaw_rad": yaw,
            "heading_deg": (90.0 - math.degrees(yaw)) % 360.0,
            "lat": lat,
            "lon": lon,
        })

    frames.sort(key=lambda item: (item["location"], item["scene_name"], item["timestamp"]))
    return frames


# --------------------------------------------------------------------------- #
# AOI construction
# --------------------------------------------------------------------------- #
def buffered_bbox(points: Sequence[Tuple[float, float]], margin_m: float) -> Tuple[float, float, float, float]:
    lats = [p[0] for p in points]
    lons = [p[1] for p in points]
    lat_c = (min(lats) + max(lats)) / 2.0
    d_lat = margin_m / M_PER_DEG_LAT
    d_lon = margin_m / m_per_deg_lon(lat_c)
    return min(lons) - d_lon, min(lats) - d_lat, max(lons) + d_lon, max(lats) + d_lat


def build_polygons(frames: Sequence[dict], margin_m: float,
                   tile_m: Optional[float], overlap_m: float,
                   full_map_m: float = 0.0) -> Tuple[List[dict], Dict[str, dict]]:
    """Build the AOI polygons.

    :param full_map_m: when > 0, ignore the driven trajectory and emit a square
        box of this side length anchored at the map's south-western corner.
        Useful when only a subset of the scenes (e.g. v1.0-mini) is available
        but the whole location must still be covered by the download.
    """
    by_location: Dict[str, List[dict]] = defaultdict(list)
    for frame in frames:
        by_location[frame["location"]].append(frame)
    if full_map_m > 0:
        for location in REFERENCE_COORDINATES:
            by_location.setdefault(location, [])

    polygons: List[dict] = []
    summary: Dict[str, dict] = {}

    for location, items in sorted(by_location.items()):
        if full_map_m > 0:
            # Square box anchored at the map's south-western corner, so the
            # whole location is covered even if only part of its scenes are
            # available locally.  The margin is applied on top of it.
            ref_lat, ref_lon = REFERENCE_COORDINATES[location]
            lon_min, lat_min = ref_lon, ref_lat
            lon_max = ref_lon + full_map_m / m_per_deg_lon(ref_lat)
            lat_max = ref_lat + full_map_m / M_PER_DEG_LAT
            lon_min, lat_min, lon_max, lat_max = buffered_bbox(
                [(lat_min, lon_min), (lat_max, lon_max)], margin_m)
        else:
            lon_min, lat_min, lon_max, lat_max = buffered_bbox(
                [(f["lat"], f["lon"]) for f in items], margin_m)
        lat_c = (lat_min + lat_max) / 2.0
        lon_c = (lon_min + lon_max) / 2.0
        width_m = (lon_max - lon_min) * m_per_deg_lon(lat_c)
        height_m = (lat_max - lat_min) * M_PER_DEG_LAT

        summary[location] = {
            "n_frames": len(items),
            "n_scenes": len({f["scene_name"] for f in items}),
            "lat_min": lat_min, "lat_max": lat_max,
            "lon_min": lon_min, "lon_max": lon_max,
            "width_m": width_m, "height_m": height_m,
            "reference_latlon": list(REFERENCE_COORDINATES[location]),
        }

        if not tile_m or tile_m <= 0:
            polygons.append(_polygon(location, items, lon_min, lat_min, lon_max, lat_max,
                                     width_m, height_m, "full", 0, 0))
            continue

        nx = max(1, int(math.ceil(width_m / tile_m)))
        ny = max(1, int(math.ceil(height_m / tile_m)))
        step_lon = (lon_max - lon_min) / nx
        step_lat = (lat_max - lat_min) / ny
        pad_lon = overlap_m / m_per_deg_lon(lat_c)
        pad_lat = overlap_m / M_PER_DEG_LAT
        for i in range(nx):
            for j in range(ny):
                l0 = lon_min + i * step_lon - pad_lon
                l1 = lon_min + (i + 1) * step_lon + pad_lon
                a0 = lat_min + j * step_lat - pad_lat
                a1 = lat_min + (j + 1) * step_lat + pad_lat
                polygons.append(_polygon(location, items, l0, a0, l1, a1,
                                         (l1 - l0) * m_per_deg_lon(lat_c),
                                         (a1 - a0) * M_PER_DEG_LAT,
                                         f"tile_{i}_{j}", i, j))
        summary[location]["tiling"] = {"nx": nx, "ny": ny, "n_tiles": nx * ny}

    return polygons, summary


def _polygon(location: str, items: Sequence[dict], lon_min: float, lat_min: float,
             lon_max: float, lat_max: float, width_m: float, height_m: float,
             tile: str, ti: int, tj: int) -> dict:
    inside = sum(
        1 for f in items
        if lon_min <= f["lon"] <= lon_max and lat_min <= f["lat"] <= lat_max
    )
    return {
        "geometry": [(lon_min, lat_min), (lon_max, lat_min),
                     (lon_max, lat_max), (lon_min, lat_max), (lon_min, lat_min)],
        "location": location,
        "city": location.split("-")[0],
        "tile": tile,
        "tile_i": ti,
        "tile_j": tj,
        "n_frames": inside,
        "lon_min": lon_min, "lat_min": lat_min,
        "lon_max": lon_max, "lat_max": lat_max,
        "width_m": width_m, "height_m": height_m,
    }


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #
WKT_4326 = (
    'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,'
    '298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]'
)


def write_shapefile(polygons: Sequence[dict], path: Path) -> None:
    try:
        import geopandas as gpd  # type: ignore
        from shapely.geometry import Polygon  # type: ignore
    except ImportError:
        _write_shapefile_pyshp(polygons, path)
        return

    records = [{k: v for k, v in p.items() if k != "geometry"} for p in polygons]
    gdf = gpd.GeoDataFrame(
        records,
        geometry=[Polygon(p["geometry"]) for p in polygons],
        crs="EPSG:4326",
    )
    gdf.to_file(str(path), driver="ESRI Shapefile", encoding="utf-8")


def _write_shapefile_pyshp(polygons: Sequence[dict], path: Path) -> None:
    """Fallback writer used when geopandas is unavailable."""
    try:
        import shapefile  # type: ignore  (pyshp)
    except ImportError as exc:
        raise SystemExit(
            "error: writing a shapefile needs either geopandas or pyshp "
            "(`pip install geopandas` or `pip install pyshp`)"
        ) from exc

    writer = shapefile.Writer(str(path), shapeType=shapefile.POLYGON)
    writer.field("location", "C", size=40)
    writer.field("city", "C", size=16)
    writer.field("tile", "C", size=24)
    writer.field("tile_i", "N", size=5)
    writer.field("tile_j", "N", size=5)
    writer.field("n_frames", "N", size=8)
    writer.field("lon_min", "F", size=20, decimal=10)
    writer.field("lat_min", "F", size=20, decimal=10)
    writer.field("lon_max", "F", size=20, decimal=10)
    writer.field("lat_max", "F", size=20, decimal=10)
    writer.field("width_m", "F", size=20, decimal=3)
    writer.field("height_m", "F", size=20, decimal=3)
    for p in polygons:
        writer.poly([list(p["geometry"])])
        writer.record(p["location"], p["city"], p["tile"], p["tile_i"], p["tile_j"],
                      p["n_frames"], p["lon_min"], p["lat_min"], p["lon_max"],
                      p["lat_max"], p["width_m"], p["height_m"])
    writer.close()
    with open(str(path.with_suffix(".prj")), "w", encoding="utf-8") as handle:
        handle.write(WKT_4326)


def write_geojson(polygons: Sequence[dict], path: Path) -> None:
    features = []
    for p in polygons:
        features.append({
            "type": "Feature",
            "properties": {k: v for k, v in p.items() if k != "geometry"},
            "geometry": {"type": "Polygon", "coordinates": [list(p["geometry"])]},
        })
    payload = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": features,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_frames_csv(frames: Sequence[dict], path: Path) -> None:
    columns = ["location", "scene_name", "split", "sample_token", "timestamp",
               "x", "y", "yaw_rad", "heading_deg", "lat", "lon"]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for frame in frames:
            writer.writerow({c: frame[c] for c in columns})


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataroot", type=Path, required=True,
                        help="raw nuScenes root (contains v1.0-trainval/)")
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--cam", default="CAM_FRONT",
                        help="camera whose key frames define the AOI")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("data/nuScenes/satellite_regions"))
    parser.add_argument("--margin-m", type=float, default=300.0,
                        help="slack added around the driven area (>=200 m recommended)")
    parser.add_argument("--tile-m", type=float, default=0.0,
                        help="explode each AOI into tiles of this size (0 = one polygon "
                             "per location); Google Earth Pro exports at most ~4800x4800 px")
    parser.add_argument("--overlap-m", type=float, default=50.0,
                        help="overlap between adjacent tiles")
    parser.add_argument("--combined", action="store_true",
                        help="also write every polygon into a single nuscenes_aoi.shp")
    parser.add_argument("--full-map-m", type=float, default=0.0,
                        help="ignore the driven trajectory and emit a square box of this "
                             "side length from each map's south-western corner (covers the "
                             "whole location even when only e.g. v1.0-mini is available)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames = load_frames(args.dataroot, args.version, args.cam)
    if not frames:
        raise SystemExit("error: no key frames found -- check --dataroot/--version/--cam")
    print(f"projected {len(frames)} {args.cam} key frames to WGS84")

    polygons, summary = build_polygons(frames, args.margin_m,
                                       args.tile_m if args.tile_m > 0 else None,
                                       args.overlap_m, args.full_map_m)

    # One shapefile per location, each holding exactly one AOI polygon (unless
    # --tile-m splits it into downloadable tiles).
    by_location: Dict[str, List[dict]] = defaultdict(list)
    for polygon in polygons:
        by_location[polygon["location"]].append(polygon)
    for location, items in sorted(by_location.items()):
        write_shapefile(items, args.out_dir / f"aoi_{location}.shp")
    if args.combined:
        write_shapefile(polygons, args.out_dir / "nuscenes_aoi.shp")

    write_geojson(polygons, args.out_dir / "nuscenes_aoi.geojson")
    write_frames_csv(frames, args.out_dir / "frames_latlon.csv")
    with open(args.out_dir / "aoi_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with open(args.out_dir / "location_map.json", "w", encoding="utf-8") as handle:
        json.dump({f["sample_token"]: f["location"] for f in frames}, handle, indent=2)

    print(f"\n{'location':<26}{'frames':>8}  {'lat range':^25}  {'lon range':^25}  {'size (m)':^17}")
    for location, info in summary.items():
        lat_range = f"{info['lat_min']:.6f}..{info['lat_max']:.6f}"
        lon_range = f"{info['lon_min']:.6f}..{info['lon_max']:.6f}"
        size = f"{info['width_m']:.0f} x {info['height_m']:.0f}"
        print(f"{location:<26}{info['n_frames']:>8}  {lat_range:>25}  {lon_range:>25}  {size:>17}")

    print(f"\n{len(by_location)} shapefile(s) in {args.out_dir} (EPSG:4326, 1 polygon each):")
    for location in sorted(by_location):
        print(f"  aoi_{location}.shp")
    for name in ("nuscenes_aoi.geojson", "frames_latlon.csv", "aoi_summary.json",
                 "location_map.json"):
        print(f"  {name}")


if __name__ == "__main__":
    main()
