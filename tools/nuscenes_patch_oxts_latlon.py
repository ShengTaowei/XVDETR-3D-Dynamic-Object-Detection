#!/usr/bin/env python3
"""
Fill in the missing GPS fields of the nuScenes KITTI-style oxts records.

``tools/convert_nuscenes_to_monocop.py`` writes one KITTI-style oxts record per
nuScenes frame, but it leaves the first two fields (latitude, longitude) at 0.0
because nuScenes ships no GPS.  The yaw (field 5) it writes is already correct
and already uses the KITTI convention (0 = East, counter-clockwise positive).

This tool replaces fields 0/1 with the WGS84 coordinates projected by
``tools/nuscenes_satellite_regions.py`` (``frames_latlon.csv``) and leaves every
other field untouched.  After that, ``tools/crop_satellite_tiles.py`` can
consume nuScenes exactly like KITTI -- no other change to the pose pipeline.

Usage:
    python tools/nuscenes_patch_oxts_latlon.py                 # patch in place
    python tools/nuscenes_patch_oxts_latlon.py --dry-run       # only report
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Tuple

ROOT = Path(__file__).resolve().parent.parent
NUSC_ROOT = ROOT / "data" / "nuScenes"
DEFAULT_CSV = NUSC_ROOT / "satellite_regions" / "frames_latlon.csv"
DEFAULT_MAPPING = NUSC_ROOT / "token_mapping.json"

SPLIT_DIR = {"train": "training", "val": "validation"}


def load_latlon(csv_path: Path) -> Dict[str, Tuple[float, float]]:
    out: Dict[str, Tuple[float, float]] = {}
    with open(csv_path, "r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out[row["sample_token"]] = (float(row["lat"]), float(row["lon"]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=NUSC_ROOT,
                        help="converted nuScenes root (training/ + validation/)")
    parser.add_argument("--pose-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--token-mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    latlon = load_latlon(args.pose_csv)
    with open(args.token_mapping, "r", encoding="utf-8") as handle:
        mapping = json.load(handle)

    patched = missing = skipped = 0
    for key, entry in mapping.items():
        split = entry.get("split", key.split("/")[0])
        frame_id = entry.get("id", key.split("/")[-1])
        token = entry["token"]
        coords = latlon.get(token)
        if coords is None:
            missing += 1
            continue
        path = args.root / SPLIT_DIR[split] / "oxts" / f"{frame_id}.txt"
        if not path.is_file():
            skipped += 1
            continue
        fields = path.read_text(encoding="utf-8").strip().split()
        if len(fields) < 6:
            skipped += 1
            continue
        fields[0] = f"{coords[0]:.12g}"
        fields[1] = f"{coords[1]:.12g}"
        if not args.dry_run:
            path.write_text(" ".join(fields) + "\n", encoding="utf-8")
        patched += 1

    print(f"patched {patched} oxts records (missing pose: {missing}, missing file: {skipped})"
          + ("  [dry-run, nothing written]" if args.dry_run else ""))

    # Sanity check: no record may keep the placeholder 0.0 coordinates.
    if not args.dry_run:
        zeros = 0
        for entry in list(mapping.values())[:2000]:
            split = entry.get("split")
            path = args.root / SPLIT_DIR[split] / "oxts" / f"{entry['id']}.txt"
            if path.is_file():
                values = path.read_text(encoding="utf-8").split()
                if float(values[0]) == 0.0 and float(values[1]) == 0.0:
                    zeros += 1
        print(f"post-check: {zeros} record(s) still at lat=lon=0 (sampled 2000)")


if __name__ == "__main__":
    main()
