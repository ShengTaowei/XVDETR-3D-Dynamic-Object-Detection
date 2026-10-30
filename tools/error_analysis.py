"""Offline error analysis for KITTI 3D detection (Car) -- extended.

DIAGNOSTIC ONLY: reads prediction txts + GT labels and summarises WHERE / HOW
the model fails. Never tunes hyper-parameters, never selects a checkpoint, so
no validation-set leakage into training/model-selection.

Beyond the basic distance/occlusion/truncation miss-rate tables, this version
adds the error patterns that actually drive KITTI AP down:
  * KITTI official difficulty split (Easy/Moderate/Hard) with the real gates
    (2D box height, occlusion, truncation) -- so numbers line up with AP.
  * False-positive breakdown: ghost boxes vs near-misses (localised but <IoU).
  * Orientation flips (~180 deg) -- a classic monocular failure.
  * Depth bias (signed) -- systematic near/far bias vs random error.
  * Lateral |x| coverage vs the satellite window.

Usage:
    python tools/error_analysis.py --pred <dir> --gt data/KITTI/ground_label \
        --split data/KITTI/chen_split/val.txt [--iou 0.5] [--score 0.2]
"""
import argparse
import os
import math
import numpy as np


class Obj:
    __slots__ = ('cls', 'trunc', 'occ', 'alpha', 'box2d', 'h', 'w', 'l',
                 'x', 'y', 'z', 'ry', 'score', 'box_h')

    def __init__(self, line):
        f = line.strip().split(' ')
        self.cls = f[0]
        self.trunc = float(f[1])
        self.occ = int(float(f[2]))
        self.alpha = float(f[3])
        self.box2d = np.array([float(f[4]), float(f[5]), float(f[6]), float(f[7])])
        self.box_h = self.box2d[3] - self.box2d[1]
        self.h, self.w, self.l = float(f[8]), float(f[9]), float(f[10])
        self.x, self.y, self.z = float(f[11]), float(f[12]), float(f[13])
        self.ry = float(f[14])
        self.score = float(f[15]) if len(f) > 15 else 1.0


def load_objs(path, cls='Car'):
    if not os.path.exists(path):
        return []
    with open(path) as fp:
        return [Obj(l) for l in fp if l.strip() and l.split(' ')[0] == cls]


def bev_iou_axis_aligned(a, b):
    def box(o):
        return (o.x - o.l / 2, o.z - o.w / 2, o.x + o.l / 2, o.z + o.w / 2)
    ax1, az1, ax2, az2 = box(a)
    bx1, bz1, bx2, bz2 = box(b)
    ix1, iz1 = max(ax1, bx1), max(az1, bz1)
    ix2, iz2 = min(ax2, bx2), min(az2, bz2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iz2 - iz1)
    inter = iw * ih
    ua = (ax2 - ax1) * (az2 - az1) + (bx2 - bx1) * (bz2 - bz1) - inter
    return inter / ua if ua > 0 else 0.0


def center_dist(a, b):
    return math.hypot(a.x - b.x, a.z - b.z)


def heading_err_deg(a, b):
    d = abs(a.ry - b.ry) % (2 * math.pi)
    d = min(d, 2 * math.pi - d)
    return math.degrees(d)


def dist_bucket(z):
    if z < 20:
        return '00-20m'
    if z < 40:
        return '20-40m'
    if z < 60:
        return '40-60m'
    return '60m+'


DIST_KEYS = ['00-20m', '20-40m', '40-60m', '60m+']
OCC_KEYS = {0: 'occ0(vis)', 1: 'occ1(part)', 2: 'occ2(largely)', 3: 'occ3(unk)'}


def kitti_difficulty(o):
    """Official KITTI Car difficulty. Returns 'easy'/'mod'/'hard' or None if the
    object is ignored by the benchmark (too small / too occluded / too trunc)."""
    H_EASY, H_MOD, H_HARD = 40, 25, 25
    if o.box_h >= H_EASY and o.occ <= 0 and o.trunc <= 0.15:
        return 'easy'
    if o.box_h >= H_MOD and o.occ <= 1 and o.trunc <= 0.30:
        return 'mod'
    if o.box_h >= H_HARD and o.occ <= 2 and o.trunc <= 0.50:
        return 'hard'
    return None  # ignored region (don't-care for AP)


def in_sat_window(o, forward_m=40.0, half_w=40.0):
    return (0 < o.z <= forward_m) and (abs(o.x) <= half_w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True)
    ap.add_argument('--gt', required=True)
    ap.add_argument('--split', required=True)
    ap.add_argument('--iou', type=float, default=0.5)
    ap.add_argument('--score', type=float, default=0.2)
    ap.add_argument('--sat_forward', type=float, default=40.0)
    ap.add_argument('--sat_halfw', type=float, default=40.0)
    args = ap.parse_args()

    ids = [x.strip() for x in open(args.split) if x.strip()]

    n_gt = 0
    n_det = 0
    fn_by_dist = {k: 0 for k in DIST_KEYS}
    gt_by_dist = {k: 0 for k in DIST_KEYS}
    fn_by_occ = {k: 0 for k in OCC_KEYS.values()}
    gt_by_occ = {k: 0 for k in OCC_KEYS.values()}
    fn_by_trunc = {'t<0.3': 0, 't0.3-0.6': 0, 't>0.6': 0}
    gt_by_trunc = {'t<0.3': 0, 't0.3-0.6': 0, 't>0.6': 0}
    depth_err = {k: [] for k in DIST_KEYS}
    depth_bias = {k: [] for k in DIST_KEYS}       # signed (pred - gt)
    head_err = {k: [] for k in DIST_KEYS}
    dim_err = {k: [] for k in DIST_KEYS}
    bev_off = {k: [] for k in DIST_KEYS}
    win = {'in': {'gt': 0, 'det': 0, 'derr': []}, 'out': {'gt': 0, 'det': 0, 'derr': []}}

    # KITTI difficulty counters (only non-ignored objects)
    diff_gt = {'easy': 0, 'mod': 0, 'hard': 0}
    diff_det = {'easy': 0, 'mod': 0, 'hard': 0}

    # orientation flips among matched objects
    n_flip = 0
    n_matched_for_flip = 0

    # FP breakdown
    n_fp_ghost = 0      # no GT nearby at all
    n_fp_nearmiss = 0   # a GT within 2m centre but IoU below thr (localisation)
    n_pred_total = 0
    lat_out = 0         # GT with |x| > halfw

    def trunc_key(t):
        return 't<0.3' if t < 0.3 else ('t0.3-0.6' if t <= 0.6 else 't>0.6')

    for fid in ids:
        gts = load_objs(os.path.join(args.gt, f'{int(fid):06d}.txt'))
        preds = [p for p in load_objs(os.path.join(args.pred, f'{int(fid):06d}.txt'))
                 if p.score >= args.score]
        n_pred_total += len(preds)
        pred_matched = [False] * len(preds)

        for g in gts:
            n_gt += 1
            db = dist_bucket(g.z)
            ok = OCC_KEYS.get(g.occ, 'occ3(unk)')
            tk = trunc_key(g.trunc)
            gt_by_dist[db] += 1
            gt_by_occ[ok] += 1
            gt_by_trunc[tk] += 1
            if abs(g.x) > args.sat_halfw:
                lat_out += 1
            diff = kitti_difficulty(g)
            if diff:
                diff_gt[diff] += 1
            inw = 'in' if in_sat_window(g, args.sat_forward, args.sat_halfw) else 'out'
            win[inw]['gt'] += 1

            best_iou, best_j = 0.0, -1
            for j, p in enumerate(preds):
                if pred_matched[j]:
                    continue
                iou = bev_iou_axis_aligned(g, p)
                if iou > best_iou:
                    best_iou, best_j = iou, j

            if best_iou >= args.iou and best_j >= 0:
                pred_matched[best_j] = True
                p = preds[best_j]
                n_det += 1
                win[inw]['det'] += 1
                if diff:
                    diff_det[diff] += 1
                de = abs(p.z - g.z)
                depth_err[db].append(de)
                depth_bias[db].append(p.z - g.z)
                win[inw]['derr'].append(de)
                he = heading_err_deg(p, g)
                head_err[db].append(he)
                dim_err[db].append(abs(p.h - g.h) + abs(p.w - g.w) + abs(p.l - g.l))
                bev_off[db].append(center_dist(p, g))
                n_matched_for_flip += 1
                if he > 150:  # ~180 deg flip
                    n_flip += 1
            else:
                fn_by_dist[db] += 1
                fn_by_occ[ok] += 1
                fn_by_trunc[tk] += 1

        # FP breakdown for this frame
        for j, p in enumerate(preds):
            if pred_matched[j]:
                continue
            # is there any GT whose centre is within 2 m of this pred?
            near = any(center_dist(p, g) < 2.0 for g in gts)
            if near:
                n_fp_nearmiss += 1
            else:
                n_fp_ghost += 1

    def pct(a, b):
        return f'{100.0 * a / b:5.1f}%' if b else '  n/a'

    def stat(v):
        if not v:
            return 'n/a'
        v = np.array(v)
        return f'mean={v.mean():+.2f} med={np.median(v):+.2f} p90={np.percentile(v,90):+.2f} n={len(v)}'

    n_fp = n_fp_ghost + n_fp_nearmiss
    print('=' * 74)
    print(f'Car GTs={n_gt}  matched={n_det}  recall(all)={pct(n_det, n_gt)}  '
          f'preds>{args.score}={n_pred_total}  FP={n_fp}  (axis-aligned BEV IoU>={args.iou})')
    print('NOTE: absolute recall is pessimistic (axis-aligned IoU + all GTs incl.')
    print('      ignored). Use the RELATIVE patterns, and table [6] for AP-aligned view.')
    print('=' * 74)

    print('\n[1] MISS RATE BY DISTANCE')
    for k in DIST_KEYS:
        print(f'  {k:8s} GT={gt_by_dist[k]:5d}  miss={fn_by_dist[k]:5d}  '
              f'miss-rate={pct(fn_by_dist[k], gt_by_dist[k])}')

    print('\n[2] MISS RATE BY OCCLUSION')
    for k in OCC_KEYS.values():
        print(f'  {k:14s} GT={gt_by_occ[k]:5d}  miss={fn_by_occ[k]:5d}  '
              f'miss-rate={pct(fn_by_occ[k], gt_by_occ[k])}')

    print('\n[3] MISS RATE BY TRUNCATION')
    for k in ['t<0.3', 't0.3-0.6', 't>0.6']:
        print(f'  {k:10s} GT={gt_by_trunc[k]:5d}  miss={fn_by_trunc[k]:5d}  '
              f'miss-rate={pct(fn_by_trunc[k], gt_by_trunc[k])}')

    print('\n[4] ERROR OF MATCHED DETECTIONS, BY DISTANCE  (depth_bias is signed pred-gt)')
    for k in DIST_KEYS:
        print(f'  {k:8s} |depth|   {stat(depth_err[k])}')
        print(f'  {"":8s} depth_bias{stat(depth_bias[k])}')
        print(f'  {"":8s} headDeg   {stat(head_err[k])}')
        print(f'  {"":8s} dim       {stat(dim_err[k])}')
        print(f'  {"":8s} BEVoff(m) {stat(bev_off[k])}')

    print('\n[5] SATELLITE WINDOW: IN vs OUT  '
          f'(z<={args.sat_forward}m, |x|<={args.sat_halfw}m)')
    for k in ['in', 'out']:
        w = win[k]
        print(f'  {k:3s}  GT={w["gt"]:5d}  recall={pct(w["det"], w["gt"])}  '
              f'|depth|-err {stat(w["derr"])}')
    print(f'  (GTs with |x|>{args.sat_halfw}m, laterally outside window: {lat_out})')

    print('\n[6] KITTI OFFICIAL DIFFICULTY  (ignored objects excluded; AP-aligned)')
    for k in ['easy', 'mod', 'hard']:
        print(f'  {k:5s} GT={diff_gt[k]:5d}  matched={diff_det[k]:5d}  '
              f'recall={pct(diff_det[k], diff_gt[k])}')

    print('\n[7] FALSE POSITIVES  (unmatched predictions hurt precision -> AP)')
    print(f'  total FP={n_fp}  '
          f'ghost(no GT within 2m)={n_fp_ghost} ({pct(n_fp_ghost, n_fp)})  '
          f'near-miss(GT<2m but IoU<thr)={n_fp_nearmiss} ({pct(n_fp_nearmiss, n_fp)})')

    print('\n[8] ORIENTATION FLIPS  (~180 deg) among matched')
    print(f'  flips={n_flip} / matched={n_matched_for_flip} = {pct(n_flip, n_matched_for_flip)}')


if __name__ == '__main__':
    main()
