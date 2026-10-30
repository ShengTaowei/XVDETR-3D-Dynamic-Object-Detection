#!/usr/bin/env python3
"""
Visualize satellite crop alignment by overlaying GT object footprints.

Usage:
  python tools/visualize_sat_alignment.py --cfg config/monocop_car_satgeo.yaml --num_samples 10

Randomly samples frames from the training set and draws GT object bounding boxes
(top-down footprint) onto the satellite crop. Each frame is saved as:
  outputs/sat_alignment_check/NNNNNN_aligned.png

If projections are correct:
  - Car footprints should align with parked/moving vehicles visible in the
    satellite image.
  - The ego vehicle anchor (bottom-center) should match the road surface at
    the camera location.
  - Heading (yaw) arrows should point along the vehicle's long axis.

If projections are consistently OFF by a few meters or systematically rotated,
it indicates a coordinate-system mismatch (flip sign, axis swap, or yaw offset).
"""
import os
import sys
import argparse
import random
import json
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# add lib to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
import yaml


def draw_rotated_box(draw, cx, cy, length_pix, width_pix, yaw_rad, color, width=2):
    """Draw a rotated rectangle (vehicle top-down footprint) on a PIL ImageDraw."""
    # corners in local frame (vehicle-aligned: x=forward=length, y=lateral=width)
    half_l = length_pix / 2
    half_w = width_pix / 2
    corners_local = np.array([
        [ half_l,  half_w],  # front-right
        [ half_l, -half_w],  # front-left
        [-half_l, -half_w],  # rear-left
        [-half_l,  half_w],  # rear-right
    ])
    # rotate by yaw (satellite convention: 0=East, counter-clockwise positive)
    cos_yaw = np.cos(yaw_rad)
    sin_yaw = np.sin(yaw_rad)
    R = np.array([[cos_yaw, -sin_yaw],
                  [sin_yaw,  cos_yaw]])
    corners_rotated = corners_local @ R.T
    corners_pixel = corners_rotated + np.array([[cx, cy]])
    
    # draw edges
    for i in range(4):
        p1 = tuple(corners_pixel[i])
        p2 = tuple(corners_pixel[(i+1) % 4])
        draw.line([p1, p2], fill=color, width=width)
    
    # draw heading arrow (front center -> slightly forward)
    front_center = (corners_pixel[0] + corners_pixel[1]) / 2
    arrow_tip = front_center + np.array([cos_yaw, sin_yaw]) * length_pix * 0.3
    draw.line([tuple(front_center), tuple(arrow_tip)], fill=color, width=width+1)


def main():
    parser = argparse.ArgumentParser(description="Visualize satellite alignment")
    parser.add_argument('--cfg', type=str, required=True, help='config file path')
    parser.add_argument('--num_samples', type=int, default=10, help='number of frames to visualize')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--output_dir', type=str, default='outputs/sat_alignment_check',
                        help='where to save visualizations')
    parser.add_argument('--frames', type=str, default='',
                        help='comma-separated real KITTI frame ids to visualize (overrides random sampling)')
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # load config
    with open(args.cfg, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    # resolve relative paths (config is written relative to project root, not tools/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if 'root_dir' in cfg['dataset'] and not os.path.isabs(cfg['dataset']['root_dir']):
        cfg['dataset']['root_dir'] = os.path.join(project_root, cfg['dataset']['root_dir'])
    if 'sat_dir' in cfg['dataset'] and not os.path.isabs(cfg['dataset']['sat_dir']):
        cfg['dataset']['sat_dir'] = os.path.join(project_root, cfg['dataset']['sat_dir'])
    
    # build dataset (train split, no augmentation)
    dataset_cfg = cfg['dataset'].copy()
    dataset_cfg['use_satellite'] = True
    dataset_cfg['random_flip'] = 0.0  # no flip for clean visualization
    dataset_cfg['random_crop'] = 0.0
    dataset = KITTI_Dataset(split='train', cfg=dataset_cfg)
    
    print(f"Dataset size: {len(dataset)} frames")
    print(f"Satellite dir: {dataset.sat_dir}")
    print(f"Satellite window: forward={dataset.sat_forward_m}m, half_width={dataset.sat_half_width_m}m")
    print(f"Satellite crop size: {dataset.sat_h} x {dataset.sat_w} pixels")
    
    # load crop metadata
    meta_path = os.path.join(dataset.sat_dir, 'crop_meta.json')
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    print(f"Global meters-per-pixel: {meta['global_mpp']:.5f}")
    
    # sample indices (or use explicitly requested frames)
    if args.frames:
        want = set(int(x) for x in args.frames.split(','))
        indices = [i for i, fid in enumerate(dataset.idx_list) if int(fid) in want]
    else:
        indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # helper to load satellite image
    def load_sat_image(dataset, idx):
        frame_id_str = dataset.idx_list[idx]
        frame_id_int = int(frame_id_str)  # remove leading zeros
        sat_path = os.path.join(dataset.sat_dir, f"{frame_id_int}.png")
        if not os.path.exists(sat_path):
            return None
        sat_img = cv2.imread(sat_path)
        sat_img = cv2.cvtColor(sat_img, cv2.COLOR_BGR2RGB)
        return sat_img  # (H, W, 3) uint8
    
    for idx in indices:
        # CRITICAL: idx is the enumerate position in idx_list, NOT the real KITTI
        # frame id. get_label / get_calib index files by real frame id, so we must
        # map through idx_list first (exactly as dataset.__getitem__ does with
        # `index = int(self.idx_list[item])`). The satellite loader already does
        # this mapping internally.
        real_id = int(dataset.idx_list[idx])

        # load raw satellite image
        sat_img = load_sat_image(dataset, idx)
        if sat_img is None:
            print(f"[Frame {dataset.idx_list[idx]}] Satellite image not found, skip.")
            continue
        sat_pil = Image.fromarray(sat_img)
        draw = ImageDraw.Draw(sat_pil)
        
        # load GT objects + calib by REAL frame id
        objects = dataset.get_label(real_id)
        frame_id = dataset.idx_list[idx]
        calib = dataset.get_calib(real_id)
        
        print(f"\n[Frame {frame_id}] {len(objects)} GT objects")
        
        def project_xz(x, z):
            """old projection: treat camera-frame (x,z) directly as sat (lateral,forward)."""
            col = (x + dataset.sat_half_width_m) / (2 * dataset.sat_half_width_m) * dataset.sat_w
            row = (dataset.sat_forward_m - z) / dataset.sat_forward_m * dataset.sat_h
            return col, row

        # draw each GT object
        for i, obj in enumerate(objects):
            if obj.cls_type == 'DontCare':
                continue
            
            # ---- OLD: camera-frame pos directly as sat coordinates (red) ----
            x_cam, z_cam = obj.pos[0], obj.pos[2]
            col_old, row_old = project_xz(x_cam, z_cam)

            # ---- NEW: transform rect-camera pos -> velodyne (ego) frame first (green) ----
            # velodyne frame: x=forward, y=left, z=up. Satellite: forward=+row-up, lateral=+right.
            pos_rect = obj.pos.reshape(1, 3)
            pos_velo = calib.rect_to_lidar(pos_rect)[0]  # (x=forward, y=left, z=up)
            forward_v = pos_velo[0]
            lateral_v = -pos_velo[1]  # left(+y) -> sat lateral is right-positive, so negate
            col_new = (lateral_v + dataset.sat_half_width_m) / (2 * dataset.sat_half_width_m) * dataset.sat_w
            row_new = (dataset.sat_forward_m - forward_v) / dataset.sat_forward_m * dataset.sat_h

            # draw both: red=old, green=new  (small circles so overlap is visible)
            r = 6
            draw.ellipse([col_old-r, row_old-r, col_old+r, row_old+r], outline='red', width=3)
            draw.ellipse([col_new-r, row_new-r, col_new+r, row_new+r], outline='lime', width=3)

            print(f"  [{i}] {obj.cls_type} cam(x={x_cam:.1f},z={z_cam:.1f}) "
                  f"-> OLD sat({col_old:.0f},{row_old:.0f})  "
                  f"velo(fwd={forward_v:.1f},lat={lateral_v:.1f}) -> NEW sat({col_new:.0f},{row_new:.0f})")
        
        # ego anchor (bottom-center)
        ego_col = dataset.sat_w / 2
        ego_row = dataset.sat_h - 1
        draw.ellipse([ego_col-5, ego_row-5, ego_col+5, ego_row+5], fill='yellow', outline='black', width=2)
        
        out_path = os.path.join(args.output_dir, f'{frame_id}_cmp.png')
        sat_pil.save(out_path)
        print(f"Saved: {out_path}  (RED=old cam-frame proj, GREEN=new velodyne-frame proj)")
    
    print(f"\nDone. Compare RED vs GREEN dots: whichever lands on the ROAD / drivable")
    print(f"surface (static structure that IS shared between satellite and street view)")
    print(f"is the correct projection. Vehicles themselves differ (different capture time).")
    return


if __name__ == '__main__':
    main()
