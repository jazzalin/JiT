"""
Autoregressive rollouts: forecasts at +L, +2L, ..., +K*L (L = the training lead time, data.lead_times, e.g. 2h),
each step conditioned on the previous step's forecast as I0.

Step k (k = 1..K) is built like a training sample whose I0 time is t0 + (k-1)L and target time t0 + kL:
  - I0: the real frame at t0 for step 1, the (8-bit quantized) forecast of step k-1 afterwards
  - It_prev: the real frame 24h before the step's target (still in the past at t0 for K*L < 24h)
  - weather: w0 at the step's I0 time, wt at its target time, wt_prev 24h before its target (INCA analyses, as in
    training: the rollout assumes a perfect weather forecast)
  - reference (reference models): the clear-sky frame is selected once, strictly before t0 (the latest real
    observation), within tod_window_min of each step's target time of day; its age is counted from the step's I0
    time. The training-time selection would pick real frames between t0 and the step's I0 time.
Every step uses the same noise for every setting and for the one-step baseline.

Baselines per step (k >= 2):
  - one-step: the same model forecasting the step from the real frame at t0 + (k-1)L (the usual 2h forecast)
  - persistence: the real frame at t0

Outputs (per EMA / CFG setting, under ema<e>/cfg<c>/):
  - <camera>/<t0>/I0_t0.png, truth_+<h>h.png, rollout_+<h>h_seed<s>.png, onestep_+<h>h_seed<s>.png
  - grids/<camera>_<t0>.jpg: the real I0, then per step the truth, the rollouts and the one-step forecast
  - metrics.json (per sample and step) and summary.json (per step means); the summary is also printed
Metrics: PSNR against the truth, detail (high-frequency energy relative to the truth: < 1 blurrier, > 1 noisier),
patch-border strength (1 = no visible patch grid) and mean brightness difference to the truth.

Example:
    python rollout_cameras.py --checkpoint results/jit_native_p16_ref/checkpoint-80.pth --split val \\
        --num_cameras 3 --num_samples 4 --steps 3 --cfg 1.0 2.0
"""
import os
import json
import argparse

import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml
from PIL import Image

from denoiser import Denoiser
from photocast_dataloader import (Photocast2DDatasetZarr, get_photocast_transforms, camera_csv_path,
                                  reference_from_cfg, select_references, reference_age_value)
from sample_cameras import resolve_checkpoint, to_unit, load_sample, annotated_grid, save_image


def get_args_parser():
    parser = argparse.ArgumentParser('JiT autoregressive rollouts', add_help=False)
    parser.add_argument('--checkpoint', required=True, type=str,
                        help='Checkpoint file, or a run folder (uses checkpoint-last.pth, else the highest checkpoint-N.pth)')
    parser.add_argument('--config', default='config.yml', type=str, help='Data config')
    parser.add_argument('--output_dir', default=None, type=str, help='Default: <run folder>/rollout-<checkpoint name>')

    parser.add_argument('--cameras', nargs='*', default=None,
                        help='Camera hashes (default: --num_cameras evenly spaced cameras of --split)')
    parser.add_argument('--split', default='val', choices=['train', 'val'])
    parser.add_argument('--num_cameras', default=4, type=int)
    parser.add_argument('--num_samples', default=4, type=int, help='Rollout start times per camera')
    parser.add_argument('--num_seeds', default=2, type=int, help='Rollouts per start time')
    parser.add_argument('--steps', default=3, type=int, help='Rollout steps (3 x 2h = +2h, +4h, +6h)')
    parser.add_argument('--start_hours', nargs=2, default=[6, 14], type=int,
                        help='Rollout start times (I0 at t0) between these hours of day, inclusive, as stored in '
                             'the camera csv; the default keeps the +6h targets in daylight for most of the year')
    parser.add_argument('--start_times', nargs='*', default=None,
                        help='Explicit start times (YYYY-mm-dd_HH:MM, the I0 time) instead of random ones; '
                             'every camera uses the ones it has')
    parser.add_argument('--ema', nargs='+', default=['1'], choices=['1', '2', 'none'])
    parser.add_argument('--cfg', nargs='+', default=[1.0], type=float, help='Weather CFG scales')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--no_reference', action='store_true',
                        help='Reference models: roll out without the clear-sky reference (black, age -1)')
    parser.add_argument('--no_onestep', action='store_true', help='Skip the one-step baseline (about 2x faster)')
    parser.add_argument('--no_quantize', action='store_true',
                        help='Feed the float forecast back as I0 (default: quantized to 8 bits, like a saved image)')
    parser.add_argument('--crop', default='center', choices=['center'],
                        help='Crop models: fixed center crop (the same view at every step)')

    parser.add_argument('--sampling_method', default='heun', type=str)
    parser.add_argument('--num_sampling_steps', default=50, type=int)
    return parser


# ---------------------------------------------------------------------------------------------- metrics

def psnr(a, b):
    return float(-10 * np.log10(((a - b) ** 2).mean() + 1e-12))


def detail(x):
    """mean absolute difference of the luminance to its 3x3 box blur (fine detail)"""
    l = torch.from_numpy(np.ascontiguousarray(x.mean(-1), dtype=np.float32))[None, None]
    b = torch.nn.functional.avg_pool2d(l, 3, stride=1, padding=1, count_include_pad=False)
    return float((l - b).abs().mean())


def border(x, p):
    """mean luminance step across patch borders relative to inside patches (1 = no visible grid)"""
    l = x.mean(-1)
    dx, dy = np.abs(np.diff(l, axis=1)), np.abs(np.diff(l, axis=0))
    bx, by = (np.arange(dx.shape[1]) % p) == p - 1, (np.arange(dy.shape[0]) % p) == p - 1
    return float((dx[:, bx].mean() + dy[by].mean()) / (dx[:, ~bx].mean() + dy[~by].mean()))


def hwc(x):
    """[3, H, W] in [-1, 1] -> [H, W, 3] numpy in [0, 1]"""
    return to_unit(x).permute(1, 2, 0).cpu().numpy()


# ---------------------------------------------------------------------------------------------- data

def rollout_starts(ds, lead_min, steps, start_hours, start_times, num_samples, seed):
    """
    Dataset indices of every step of each rollout: [[idx step 1, ..., idx step K], ...]. A start t0 qualifies when
    the training triples (I0 time t0 + (k-1)L, target t0 + kL) exist for every step.
    """
    by_i0 = {t: i for i, t in enumerate(ds.times[ds.pairs[:, 1]])}
    L = np.timedelta64(lead_min, 'm')
    chains = []
    for i, t0 in enumerate(ds.times[ds.pairs[:, 1]]):
        chain = [i] + [by_i0.get(t0 + k * L) for k in range(1, steps)]
        if all(c is not None for c in chain):
            chains.append(chain)
    if start_times:
        wanted = {pd.Timestamp(t.replace('_', ' ')) for t in start_times}
        return [c for c in chains if pd.Timestamp(ds.times[ds.pairs[c[0], 1]]) in wanted]
    hours = pd.DatetimeIndex([ds.times[ds.pairs[c[0], 1]] for c in chains]).hour.to_numpy() if chains else np.array([])
    chains = [c for c, h in zip(chains, hours) if start_hours[0] <= h <= start_hours[1]]
    if not chains:
        return []
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(chains), size=min(num_samples, len(chains)), replace=False)
    return [chains[j] for j in sorted(pick)]


def load_reference_index(reference, camera):
    path = os.path.join(reference['dir'], f'{camera}.csv')
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, usecols=['time', 'path', 'valid', 'sun_elev', 'clear_score'])


def load_chain(ds, chain, seed, ref_index, reference, lead_min, use_reference):
    """
    Samples of every step of a rollout. With a reference, the clear frame is selected strictly before the rollout
    start t0 (not before each step's I0 time, as in training) by patching the dataset's per-sample selection.
    """
    if not use_reference:
        return [load_sample(ds, idx, seed) for idx in chain]
    t0 = ds.times[ds.pairs[chain[0], 1]]
    targets = ds.times[ds.pairs[chain, 2]]
    saved = [(int(ds.ref_idx[i]), float(ds.ref_age[i])) for i in chain]
    try:
        if ref_index is None:
            best, age = np.full(len(chain), -1), np.full(len(chain), np.nan)
        else:
            best, age, c_paths, _ = select_references(
                ref_index, np.repeat(t0, len(chain)), targets, max_age_days=reference.get('max_age_days', 14),
                tod_window_min=reference.get('tod_window_min', 60), quantile=reference.get('quantile', 0.9))
            assert np.array_equal(c_paths, ds.ref_paths), 'reference candidates differ from the dataset'
        for k, i in enumerate(chain):
            ds.ref_idx[i] = best[k]
            # age counted from the step's I0 time, t0 + (k-1)L, as the model saw it in training
            ds.ref_age[i] = reference_age_value(age[k] + k * lead_min / 1440.0) if best[k] >= 0 else -1.0
        return [load_sample(ds, idx, seed) for idx in chain]
    finally:
        for i, (r, a) in zip(chain, saved):
            ds.ref_idx[i], ds.ref_age[i] = r, a


# ---------------------------------------------------------------------------------------------- main

@torch.no_grad()
def main(args):
    device = torch.device('cuda')
    ckpt_path = resolve_checkpoint(args.checkpoint)
    run_dir = os.path.dirname(ckpt_path)
    ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
    print('Checkpoint:', ckpt_path)

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    lead_min = int(cfg['data']['lead_times'][0])
    assert cfg['data']['lead_times'][0] == cfg['data']['lead_times'][1], \
        'rollouts need a single training lead time (data.lead_times [L, L])'
    assert args.steps * lead_min < 24 * 60, 'It_prev (24h before each target) must stay before the rollout start'

    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    train_args = checkpoint['args']
    print('Trained for {} epochs'.format(checkpoint.get('epoch', -1) + 1))
    train_args.sampling_method = args.sampling_method
    train_args.num_sampling_steps = args.num_sampling_steps
    model = Denoiser(train_args).to(device).eval()
    img_h, img_w = model.img_size
    cylindrical = model.net.cylindrical
    assert cylindrical or img_w == img_h, 'panorama (non-cylindrical) models are not supported'
    cfg['data']['img_size'] = [img_h, img_w]
    width_multiple = cfg['data'].get('width_multiple', 32)
    native_size = (img_h, width_multiple) if cylindrical else None
    transform = get_photocast_transforms(cfg, crop='native' if cylindrical else args.crop)
    patch = model.net.patch_size

    reference = reference_from_cfg(cfg) if model.use_reference else None
    if model.use_reference:
        assert reference is not None, 'the model takes a reference frame: enable data.reference in {}'.format(args.config)
    use_reference = reference is not None and not args.no_reference

    cameras = args.cameras
    if not cameras:
        split = cfg['data'][args.split]
        idx = np.unique(np.linspace(0, len(split) - 1, min(args.num_cameras, len(split))).round().astype(int))
        cameras = [split[i] for i in idx]
    output_dir = args.output_dir or os.path.join(run_dir, 'rollout-{}'.format(ckpt_name))
    leads = [(k + 1) * lead_min / 60 for k in range(args.steps)]
    lead_names = ['+{:g}h'.format(h) for h in leads]
    print('Cameras:', cameras)
    print('Steps:', ', '.join(lead_names), '| reference:', 'yes (selected before t0)' if use_reference else 'no')
    print('Output:', output_dir)

    weather = xr.open_zarr(os.path.join(cfg['data']['weather_root_dir'], 'inca.zarr'), consolidated=True, chunks=None)

    # rollouts of every camera: list of per-step samples
    data = []  # (camera, [sample step 1, ..., step K])
    for camera in cameras:
        try:
            ds = Photocast2DDatasetZarr(csv_file=camera_csv_path(cfg, camera), image_root_dir=cfg['data']['image_root_dir'],
                                        shared_weather_ds=weather, weather_scaler_file=cfg['data']['weather_scaler_file'],
                                        viewshed_radius=cfg['data']['viewshed_radius'], img_transform=transform,
                                        lead_times=cfg['data']['lead_times'], date_ranges=cfg['data'].get('date_ranges'),
                                        native_size=native_size, reference=reference)
        except (FileNotFoundError, IndexError) as e:
            print('  skipping {}: {}'.format(camera, e))
            continue
        chains = rollout_starts(ds, lead_min, args.steps, args.start_hours, args.start_times, args.num_samples, args.seed)
        if not chains:
            print('  skipping {}: no start time with all {} steps'.format(camera, args.steps))
            continue
        ref_index = load_reference_index(reference, camera) if use_reference else None
        for chain in chains:
            steps = load_chain(ds, chain, args.seed, ref_index, reference, lead_min, use_reference)
            data.append((camera, steps))
        print('  {}: {} rollouts from {}'.format(camera, len(chains),
                                                ', '.join(s[0]['datetimes'][1] for _, s in data[-len(chains):])))
    assert data, 'nothing to roll out'

    weight_keys = {'1': 'model_ema1', '2': 'model_ema2', 'none': 'model'}
    S = args.num_seeds
    for ema in args.ema:
        model.load_state_dict(checkpoint[weight_keys[ema]])
        for cfg_scale in args.cfg:
            model.cfg_scale = cfg_scale
            setting_dir = os.path.join(output_dir, 'ema{}'.format(ema), 'cfg{}'.format(cfg_scale))
            metrics = []
            for camera, steps in data:
                t0_name = steps[0]['datetimes'][1]
                sample_dir = os.path.join(setting_dir, camera, t0_name)
                I0_real = steps[0]['tensors'][0]
                m = dict(camera=camera, t0=t0_name, targets=[s['datetimes'][2] for s in steps],
                         reference=[s.get('reference_time', '') if use_reference else '' for s in steps], steps=[])
                save_image(to_unit(I0_real).cpu(), os.path.join(sample_dir, 'I0_t0.png'))
                x_prev, rows = None, []
                if use_reference:
                    rows.append(('Iref {}'.format(steps[0].get('reference_time') or 'none'),
                                 [to_unit(steps[0]['reference'][0]).cpu()]))
                rows.append(('I0 (real)\n{}'.format(t0_name), [to_unit(I0_real).cpu()]))

                for k, smp in enumerate(steps):
                    I0, It, It_prev, w0, wt, wt_prev, am, vm = [t.unsqueeze(0).to(device) for t in smp['tensors']]
                    rep = lambda t: t.repeat_interleave(S, 0)
                    rest = [rep(t) for t in (It_prev, w0, wt, wt_prev, am, vm)]
                    if model.use_reference:
                        Iref, age = smp['reference'][0][None].to(device), smp['reference'][1][None].to(device)
                        if not use_reference:
                            Iref, age = torch.full_like(Iref, -1.0), torch.full_like(age, -1.0)
                        rest += [rep(Iref), rep(age)]

                    def forecast(cond_I0):
                        torch.manual_seed(args.seed * 1000 + k)  # same noise for the rollout and the one-step forecast
                        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                            x = model.generate(cond_I0, rest[0], *rest[1:])
                        x = x.float().clamp(-1, 1)
                        if not args.no_quantize:
                            x = torch.round((x + 1) * 127.5) / 127.5 - 1  # as if saved as an 8-bit image
                        return x

                    x_roll = forecast(rep(I0) if k == 0 else x_prev)
                    x_one = forecast(rep(I0)) if k > 0 and not args.no_onestep else None
                    x_prev = x_roll

                    truth = hwc(It[0])
                    step_m = dict(lead=lead_names[k], persistence_psnr=psnr(hwc(I0_real), truth),
                                  truth_detail=detail(truth), truth_border=border(truth, patch))
                    for name, x in (('rollout', x_roll), ('onestep', x_one if k > 0 else x_roll)):
                        if x is None:
                            continue
                        f = [hwc(x[s]) for s in range(S)]
                        step_m[name] = dict(psnr=float(np.mean([psnr(y, truth) for y in f])),
                                            psnr_seeds=[psnr(y, truth) for y in f],
                                            detail=float(np.mean([detail(y) for y in f])) / max(step_m['truth_detail'], 1e-6),
                                            border=float(np.mean([border(y, patch) for y in f])),
                                            brightness=float(np.mean([y.mean() - truth.mean() for y in f])))
                    m['steps'].append(step_m)

                    lead = lead_names[k]
                    save_image(to_unit(It[0]).cpu(), os.path.join(sample_dir, 'truth_{}.png'.format(lead)))
                    for s in range(S):
                        save_image(to_unit(x_roll[s]).cpu(), os.path.join(sample_dir, 'rollout_{}_seed{}.png'.format(lead, args.seed + s)))
                        if x_one is not None:
                            save_image(to_unit(x_one[s]).cpu(), os.path.join(sample_dir, 'onestep_{}_seed{}.png'.format(lead, args.seed + s)))
                    rows.append(('truth {}\n{}{}'.format(lead, smp['datetimes'][2],
                                                         '\nref {}'.format(smp.get('reference_time') or 'none') if use_reference else ''),
                                 [to_unit(It[0]).cpu()]))
                    for s in range(min(S, 2)):
                        rows.append(('rollout {}\nseed {}  {:.2f} dB'.format(lead, args.seed + s, step_m['rollout']['psnr_seeds'][s]),
                                     [to_unit(x_roll[s]).cpu()]))
                    if x_one is not None:
                        rows.append(('one-step {}\nseed {}  {:.2f} dB'.format(lead, args.seed, step_m['onestep']['psnr_seeds'][0]),
                                     [to_unit(x_one[0]).cpu()]))

                metrics.append(m)
                grid = annotated_grid(rows, title='{} rollout from {}, EMA {}, CFG {}'.format(camera, t0_name, ema, cfg_scale))
                os.makedirs(os.path.join(setting_dir, 'grids'), exist_ok=True)
                grid.save(os.path.join(setting_dir, 'grids', '{}_{}.jpg'.format(camera[:8], t0_name)), quality=90)
                print('ema {} cfg {} {} {}: '.format(ema, cfg_scale, camera[:8], t0_name) + ' | '.join(
                    '{} rollout {:.2f}{} persistence {:.2f}'.format(
                        s['lead'], s['rollout']['psnr'],
                        ' one-step {:.2f}'.format(s['onestep']['psnr']) if 'onestep' in s and k > 0 else '',
                        s['persistence_psnr']) for k, s in enumerate(m['steps'])), flush=True)

            # per-step means
            summary = []
            for k, lead in enumerate(lead_names):
                st = [m['steps'][k] for m in metrics]
                row = dict(lead=lead, samples=len(st), persistence_psnr=float(np.mean([s['persistence_psnr'] for s in st])),
                           truth_border=float(np.mean([s['truth_border'] for s in st])))
                for name in ('rollout', 'onestep'):
                    if all(name in s for s in st):
                        for key in ('psnr', 'detail', 'border', 'brightness'):
                            row['{}_{}'.format(name, key)] = float(np.mean([s[name][key] for s in st]))
                summary.append(row)
            os.makedirs(setting_dir, exist_ok=True)
            with open(os.path.join(setting_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=1)
            with open(os.path.join(setting_dir, 'summary.json'), 'w') as f:
                json.dump(dict(checkpoint=ckpt_path, ema=ema, cfg=cfg_scale, reference=use_reference,
                               quantized=not args.no_quantize, steps=summary), f, indent=1)

            print('== EMA {} CFG {}: {} rollouts'.format(ema, cfg_scale, len(metrics)))
            print('   {:>5} | {:>8} {:>8} {:>8} | {:>6} {:>6} | {:>6} {:>6} | {:>6}'.format(
                'lead', 'rollout', 'one-step', 'persist', 'detail', '(1step)', 'border', '(1step)', 'truth'))
            for r in summary:
                print('   {:>5} | {:8.2f} {:>8} {:8.2f} | {:6.2f} {:>6} | {:6.3f} {:>6} | {:6.3f}'.format(
                    r['lead'], r['rollout_psnr'],
                    '{:.2f}'.format(r['onestep_psnr']) if 'onestep_psnr' in r else '-', r['persistence_psnr'],
                    r['rollout_detail'], '{:.2f}'.format(r['onestep_detail']) if 'onestep_detail' in r else '-',
                    r['rollout_border'], '{:.3f}'.format(r['onestep_border']) if 'onestep_border' in r else '-',
                    r['truth_border']))


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
