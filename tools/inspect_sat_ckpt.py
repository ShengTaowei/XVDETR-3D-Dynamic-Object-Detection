"""
Inspect the retained geometry-guided satellite route in a trained checkpoint.
The geo gate is identity-initialized, so its learned scale indicates whether
the detector learned to use the satellite foreground prior.

Usage:
    python tools/inspect_sat_ckpt.py --ckpt outputs/monocop_car_satgeo/.../checkpoint_best.pth
"""
import argparse

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    state = ckpt.get('model_state', ckpt.get('model', ckpt))
    state = {k.replace('module.', '', 1): v for k, v in state.items()}

    def show(key, desc):
        if key not in state:
            print(f'[missing] {key}')
            return
        t = state[key].float()
        print(f'{desc:45s} {key:55s} |w|_mean={t.abs().mean():.3e}  |w|_max={t.abs().max():.3e}'
              + (f'  value={t.flatten()[:4].tolist()}' if t.numel() <= 4 else ''))

    p = 'satellite_branch.'
    subs = sorted({k[len(p):].split('.')[0] for k in state if k.startswith(p)})
    print('satellite_branch submodules in ckpt:', subs or 'NONE (satellite branch disabled)')
    enc_keys = [k for k in state if k.startswith(p + 'encoder.')]
    enc_type = 'ResNetSatEncoder' if any('.lat2.' in k for k in enc_keys) else \
        ('scratch SatelliteEncoder' if enc_keys else 'none')
    print('satellite encoder type:', enc_type)
    print('== GeoDepthObjectGate ==')
    show(p + 'geo_gate.raw_alpha', 'learned foreground gate scale (0 = identity)')
    show(p + 'loc_head.objectness.weight', 'satellite objectness head')
    show(p + 'loc_head.center_offset.weight', 'satellite centre head')
    print('epoch:', ckpt.get('epoch'), ' best_result:', ckpt.get('best_result'))


if __name__ == '__main__':
    main()
