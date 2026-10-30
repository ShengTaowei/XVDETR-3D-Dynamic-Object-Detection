"""
Satellite tile cropping for XVDETR-on-MonoCoP.

For every KITTI frame, crop a forward-facing, ego-heading-aligned satellite
patch from the large region GeoTIFF tiles downloaded from Google Earth Pro.

Physical window (ego-centric, NOT object-centric):
    - ego sits at the BOTTOM-CENTER of the crop (not the image center), because
      a forward-facing monocular camera only observes what's ahead of the car.
    - lateral half-width:  40 m to each side  (80 m total width)
    - forward distance:    40 m ahead of the ego position
    - the crop is rotated so that the ego heading (yaw) always points "up"
      (i.e. toward the top of the output image), matching the forward
      direction of the ground camera.

Geometry notes:
    - Source tiles are in EPSG:3857 (Web Mercator). This projection is
      conformal (angle-preserving) but its local scale factor is
      1/cos(latitude), i.e. "mercator meters" are larger than true ground
      meters away from the equator. We explicitly correct for this scale
      factor using each frame's own latitude, so the 40x80 m window is a
      TRUE ground distance, not a Web Mercator pseudo-distance.
    - KITTI oxts yaw convention (verified empirically against velocity
      heading: mean offset 0.89 deg, std 9.85 deg over 6167 frames with
      speed > 3 m/s): 0 = East, counter-clockwise positive, i.e. standard
      math convention atan2(north, east).
    - Output resolution is fixed to a single GLOBAL_MPP (meters/pixel),
      shared across every frame, so all crops have identical pixel size
      and are directly batchable. GLOBAL_MPP is the mean of the 18
      downloaded tiles' own native resolution (~0.131 m/px) -- i.e. we do
      NOT resample to an arbitrary round number like 0.2 m/px; we keep the
      native resolution of the downloaded imagery.

The same pipeline runs on nuScenes with ``--dataset nusc``.  nuScenes ships no
GPS, so the per-frame latitude/longitude are first projected from the ego poses
by ``tools/nuscenes_satellite_regions.py`` and written into the KITTI-style
oxts records by ``tools/nuscenes_patch_oxts_latlon.py``.  After that step the
only nuScenes-specific detail is the frame layout: ids are global and the files
are split across ``training/`` (0..28129) and ``validation/`` (28130..34148).

Usage:
    # KITTI
    python tools/crop_satellite_tiles.py --out_dir data/KITTI/sat_crop_fwd
    python tools/crop_satellite_tiles.py --forward_m 80 --out_dir data/KITTI/sat_crop_fwd80

    # nuScenes (80 m forward window, 612x612 like the KITTI fwd80 crops)
    python tools/crop_satellite_tiles.py --dataset nusc --forward_m 80 \
        --out_dir data/nuScenes/sat_crop_fwd80
"""
import os
import math
import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform_bounds, transform as warp_transform
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KITTI_ROOT = os.path.join(REPO_ROOT, 'data', 'KITTI')
NUSC_ROOT = os.path.join(REPO_ROOT, 'data', 'nuScenes')
IMG_DIR = os.path.join(REPO_ROOT, 'image')

TIF_NAMES = [
    'region0_0', 'region0_1', 'region0_2', 'region0_3', 'region0_4', 'region0_5',
    'region0_6', 'region0_7', 'region0_8', 'region0_9', 'region0_10', 'region0_11',
    'region0_12', 'region0_13', 'region_1', 'region_2', 'region_3', 'region_4',
]

HALF_WIDTH_M = 40.0   # lateral half-width (left + right = 80 m total)
FORWARD_M = 40.0      # forward distance ahead of ego
NUM_FRAMES = 7481     # KITTI

# nuScenes: frame ids are global; the first N live under training/, the rest
# under validation/ (see tools/convert_nuscenes_to_monocop.py).
NUSC_TRAIN_FRAMES = 28130
NUSC_NUM_FRAMES = 34149


def oxts_path(root, frame_id, dataset):
    if dataset == 'nusc':
        subdir = 'training' if frame_id < NUSC_TRAIN_FRAMES else 'validation'
        return os.path.join(root, subdir, 'oxts', '%06d.txt' % frame_id)
    return os.path.join(root, 'oxts', '%06d.txt' % frame_id)


def discover_tifs(img_dir, dataset):
    """Tile stems to open: an explicit list for KITTI, a glob for nuScenes."""
    if dataset == 'nusc':
        return sorted(p.stem for p in Path(img_dir).glob('*.tif'))
    return TIF_NAMES


def load_tif_info(img_dir, tif_names):
    """Open every region tile once and cache its geo-metadata."""
    info = {}
    mpps = []
    for name in tif_names:
        path = os.path.join(img_dir, f'{name}.tif')
        ds = rasterio.open(path)
        b = ds.bounds
        lon_min, lat_min, lon_max, lat_max = transform_bounds(
            ds.crs, 'EPSG:4326', b.left, b.bottom, b.right, b.top)
        lat_c = (lat_min + lat_max) / 2
        width_m = (lon_max - lon_min) * 111320 * math.cos(math.radians(lat_c))
        height_m = (lat_max - lat_min) * 110540
        mpp = (width_m / ds.width + height_m / ds.height) / 2
        mpps.append(mpp)
        info[name] = {
            'dataset': ds,
            'bounds_4326': (lon_min, lat_min, lon_max, lat_max),
            'transform': ds.transform,  # pixel -> EPSG:3857 (affine)
            'crs': ds.crs,
        }
    global_mpp = float(np.mean(mpps))
    return info, global_mpp


def load_oxts(root, frame_id, dataset):
    p = oxts_path(root, frame_id, dataset)
    vals = open(p).read().strip().split()
    return float(vals[0]), float(vals[1]), float(vals[5])  # lat, lon, yaw(rad)


def find_tile_for_point(lat, lon, tif_info):
    """Return the name of a tile whose EPSG:4326 bounds contain (lat, lon)."""
    for name, info in tif_info.items():
        lon_min, lat_min, lon_max, lat_max = info['bounds_4326']
        if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
            return name
    return None


def crop_forward_patch(frame_id, tif_info, global_mpp, out_h, out_w,
                       root=KITTI_ROOT, dataset='kitti',
                       half_width_m=HALF_WIDTH_M, forward_m_window=FORWARD_M):
    """
    Crop and rotate a forward-facing satellite patch for one frame.

    Returns a (out_h, out_w, 3) uint8 array, or None if the frame has no
    matching tile.
    """
    lat, lon, yaw = load_oxts(root, frame_id, dataset)
    tile_name = find_tile_for_point(lat, lon, tif_info)
    if tile_name is None:
        return None, None

    info = tif_info[tile_name]
    ds = info['dataset']
    crs = info['crs']

    # ego position in the tile's projected CRS (EPSG:3857)
    xs, ys = warp_transform('EPSG:4326', crs, [lon], [lat])
    ego_x, ego_y = xs[0], ys[0]

    # EPSG:3857 local scale factor: 1 true meter = scale "projected meters" at this latitude
    scale = 1.0 / math.cos(math.radians(lat))

    # Output pixel grid: row 0 = farthest forward, row out_h-1 = ego's forward edge (bottom),
    # col 0 = left edge, col out_w-1 = right edge. Ego sits at bottom-center (row=out_h, col=out_w/2).
    # For each output pixel, compute its (lateral, forward) offset in TRUE meters from ego,
    # then rotate by yaw and convert to EPSG:3857 projected-meter offset, then to tile pixel coords.
    us, vs = np.meshgrid(np.arange(out_w), np.arange(out_h))  # us: col idx, vs: row idx
    lateral_m = (us + 0.5) / out_w * (2 * half_width_m) - half_width_m      # [-half_w, +half_w]
    forward_m = forward_m_window - (vs + 0.5) / out_h * forward_m_window    # row 0 -> farthest, bottom -> ego

    # rotate (lateral, forward) by yaw so that "forward" maps to the ego heading direction
    # ego heading unit vector in EPSG:4326/local-ENU sense: (cos(yaw), sin(yaw)) = (east, north)
    de = (lateral_m * math.sin(yaw) + forward_m * math.cos(yaw)) * scale
    dn = (-lateral_m * math.cos(yaw) + forward_m * math.sin(yaw)) * scale

    world_x = ego_x + de
    world_y = ego_y + dn

    inv_transform = ~info['transform']
    cols, rows = inv_transform * (world_x, world_y)
    cols = np.round(cols).astype(np.int64)
    rows = np.round(rows).astype(np.int64)

    cols = np.clip(cols, 0, ds.width - 1)
    rows = np.clip(rows, 0, ds.height - 1)

    # read full band data once per call is too slow across 7481 frames; instead
    # read the minimal bounding window covering this frame's sample grid.
    col_min, col_max = int(cols.min()), int(cols.max())
    row_min, row_max = int(rows.min()), int(rows.max())
    window = rasterio.windows.Window(col_min, row_min, col_max - col_min + 1, row_max - row_min + 1)
    band_data = ds.read([1, 2, 3], window=window)  # (3, h, w)
    band_data = np.transpose(band_data, (1, 2, 0))  # (h, w, 3)

    local_cols = cols - col_min
    local_rows = rows - row_min
    patch = band_data[local_rows, local_cols]  # nearest-neighbor sample, shape (out_h, out_w, 3)

    return patch.astype(np.uint8), tile_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=('kitti', 'nusc'), default='kitti')
    parser.add_argument('--root', default=None,
                        help='dataset root holding the oxts records '
                             '(default: data/KITTI or data/nuScenes)')
    parser.add_argument('--tif-dir', default=None,
                        help='directory holding the downloaded GeoTIFF tiles '
                             '(default: image/ for KITTI, <root> for nuScenes)')
    parser.add_argument('--out_dir', default=None)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)
    # satellite window size (metres). Defaults reproduce the original 40x40 crop.
    # Error analysis shows ~21% of Car GTs sit beyond 40 m forward and are never
    # covered; enlarge --forward_m (e.g. 80) to extend the satellite's reach.
    parser.add_argument('--forward_m', type=float, default=FORWARD_M,
                        help='forward distance ahead of ego (metres)')
    parser.add_argument('--half_width_m', type=float, default=HALF_WIDTH_M,
                        help='lateral half-width to each side (metres)')
    parser.add_argument('--out-h', type=int, default=0,
                        help='force the crop height in pixels (0 = derive from GSD)')
    parser.add_argument('--out-w', type=int, default=0,
                        help='force the crop width in pixels (0 = derive from GSD)')
    args = parser.parse_args()

    is_nusc = args.dataset == 'nusc'
    root = args.root or (NUSC_ROOT if is_nusc else KITTI_ROOT)
    tif_dir = args.tif_dir or (root if is_nusc else IMG_DIR)
    num_frames = NUSC_NUM_FRAMES if is_nusc else NUM_FRAMES
    end = num_frames if args.end is None else args.end
    out_dir = args.out_dir or os.path.join(root, 'sat_crop_fwd')

    os.makedirs(out_dir, exist_ok=True)

    tif_info, global_mpp = load_tif_info(tif_dir, discover_tifs(tif_dir, args.dataset))
    out_h = args.out_h or int(round(args.forward_m / global_mpp))
    out_w = args.out_w or int(round(2 * args.half_width_m / global_mpp))
    # Effective sampling rate of the output grid; the physical window is defined
    # in true metres and is exact regardless of the tiles' native GSD.
    effective_mpp = args.forward_m / out_h
    print(f'{args.dataset}: {len(tif_info)} tile(s) from {tif_dir}')
    print(f'Global resolution: {global_mpp:.5f} m/px -> output size {out_h}(H) x {out_w}(W)  '
          f'(forward={args.forward_m}m, half_width={args.half_width_m}m)')

    meta = {
        'global_mpp': effective_mpp,
        'half_width_m': args.half_width_m,
        'forward_m': args.forward_m,
        'out_h': out_h,
        'out_w': out_w,
        'ego_anchor': 'bottom-center',
        'yaw_convention': '0=East, counter-clockwise positive (verified vs. velocity heading)',
    }
    with open(os.path.join(out_dir, 'crop_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    failed = []
    for i in range(args.start, end):
        patch, tile_name = crop_forward_patch(i, tif_info, global_mpp, out_h, out_w,
                                              root=root, dataset=args.dataset,
                                              half_width_m=args.half_width_m,
                                              forward_m_window=args.forward_m)
        if patch is None:
            failed.append(i)
            continue
        Image.fromarray(patch).save(os.path.join(out_dir, f'{i}.png'))
        if i % 1000 == 0:
            print(f'[{i}/{end}] tile={tile_name}')

    print(f'Done. Failed frames (no matching tile): {len(failed)} {failed[:20]}')

    for info in tif_info.values():
        info['dataset'].close()


if __name__ == '__main__':
    main()
