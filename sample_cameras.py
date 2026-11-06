"""
Forecast It for selected cameras with a trained JiT conditioned on (I0, It_prev) and the weather (wt, w0, wt_prev).

For every camera, a fixed random set of (It_prev, I0, It) triples is forecast with --num_seeds noise seeds per EMA,
either on 256x256 crops (the training resolution) or on full panoramas. Full panoramas are sampled by tiling:
at every ODE step the model denoises overlapping 256x256 crops around the 360 deg panorama (wrapping at the seam,
each crop with its own angle_mask) and the overlapping velocity predictions are averaged (MultiDiffusion).

Outputs, per mode (crop / panorama):
  - grids/<camera>.png: per camera, the conditioning frames, the ground truth and the forecasts of every sample
  - overview_ema<e>_cfg<c>.png: ground truth and first forecast of every camera
  - ema<e>/cfg<c>/<camera>/<I0 datetime>_seed<s>.png: individual forecasts (plus <I0 datetime>_gt.png)
Every (EMA, weather CFG scale) setting uses the same noise, so settings can be compared sample by sample.
  - samples.json: camera and (It_prev, I0, It) datetimes of every sample

Example:
    python sample_cameras.py --checkpoint results/jit_forecast --split val --mode crop panorama --num_samples 4 \
        --cfg 1.0 2.0 4.0
"""
import os
import re
import glob
import json
import argparse

import numpy as np
import torch
import torchvision
import xarray as xr
import yaml
from PIL import Image, ImageDraw

from denoiser import Denoiser
from photocast_dataloader import Photocast2DDatasetZarr, get_photocast_transforms, camera_csv_path


def get_args_parser():
    parser = argparse.ArgumentParser('JiT camera forecasts', add_help=False)
    parser.add_argument('--checkpoint', required=True, type=str,
                        help='Checkpoint file, or a run folder (uses checkpoint-last.pth, else the highest checkpoint-N.pth)')
    parser.add_argument('--config', default='config.yml', type=str, help='Data config')
    parser.add_argument('--output_dir', default=None, type=str,
                        help='Default: <run folder>/samples-<checkpoint name>')

    # what to forecast
    parser.add_argument('--cameras', nargs='*', default=None,
                        help='Camera hashes (default: --num_cameras evenly spaced cameras of --split)')
    parser.add_argument('--split', default='val', choices=['train', 'val'],
                        help='Camera list of the config used when --cameras is not given')
    parser.add_argument('--num_cameras', default=4, type=int)
    parser.add_argument('--num_samples', default=4, type=int, help='(It_prev, I0, It) triples per camera')
    parser.add_argument('--num_seeds', default=2, type=int, help='Forecasts per triple and EMA')
    parser.add_argument('--mode', nargs='+', default=['crop'], choices=['crop', 'panorama'],
                        help='crop: 256x256 crops as in training; panorama: tiled sampling of the full panorama')
    parser.add_argument('--crop', default='center', choices=['center', 'random'],
                        help='Crop used in crop mode (random crops are fixed by --seed)')
    parser.add_argument('--ema', nargs='+', default=['1'], choices=['1', '2', 'none'],
                        help='Weights: ema_params1, ema_params2 or raw weights')
    parser.add_argument('--cfg', nargs='+', default=[1.0], type=float,
                        help='Weather CFG scales (1.0 = no guidance); needs a checkpoint trained with --weather_drop_prob > 0')
    parser.add_argument('--seed', default=0, type=int)

    # sampler
    parser.add_argument('--sampling_method', default='heun', type=str)
    parser.add_argument('--num_sampling_steps', default=50, type=int)
    parser.add_argument('--tile_stride', default=128, type=int,
                        help='Panorama mode: horizontal stride between 256x256 tiles (smaller = more overlap)')
    parser.add_argument('--tile_batch', default=64, type=int,
                        help='Panorama mode: tiles per forward pass')
    return parser


def resolve_checkpoint(path):
    if os.path.isfile(path):
        return path
    last = os.path.join(path, 'checkpoint-last.pth')
    if os.path.exists(last):
        return last
    numbered = glob.glob(os.path.join(path, 'checkpoint-*.pth'))
    numbered = [p for p in numbered if re.search(r'checkpoint-(\d+)\.pth$', p)]
    if not numbered:
        raise FileNotFoundError('No checkpoint found in {}'.format(path))
    return max(numbered, key=lambda p: int(re.search(r'checkpoint-(\d+)\.pth$', p).group(1)))


def to_unit(x):
    """[-1, 1] -> [0, 1]"""
    return ((x.float() + 1) / 2).clamp(0, 1)


def load_sample(dataset, index, seed):
    """Loads a dataset sample with a fixed random crop, without touching the global RNG"""
    with torch.random.fork_rng():
        torch.manual_seed(seed + index)
        return dataset[index]


def match_panorama_widths(samples, camera, tolerance=0.02):
    """
    Full panoramas of one camera can differ in width by a few pixels (source image sizes over the camera's history,
    rounding in the aspect-preserving resize), so their frames cannot be stacked. Resizes every image-sized tensor
    (frames, angle_mask, view_mask) to the most common width; weather fields are left untouched.
    """
    widths = [t.shape[-1] for s in samples for t in s['tensors'] if t.ndim == 3 and t.shape[-2] == s['tensors'][0].shape[-2]]
    target = max(set(widths), key=widths.count)
    if max(abs(w - target) for w in widths) > tolerance * target:
        print('  WARNING: {} panorama widths range from {} to {} px: the field of view may have changed over time'.format(
            camera, min(widths), max(widths)))

    def resize(t):
        if t.ndim != 3 or t.shape[-1] == target:
            return t
        return torch.nn.functional.interpolate(t[None].float(), size=(t.shape[-2], target), mode='bilinear',
                                               align_corners=False, antialias=True)[0]

    for s in samples:
        H = s['tensors'][0].shape[-2]
        s['tensors'] = tuple(resize(t) if t.ndim == 3 and t.shape[-2] == H else t for t in s['tensors'])
    return samples


@torch.no_grad()
def generate_tiled(model, I0, It_prev, w0, wt, wt_prev, angle_mask, view_mask, tile=256, stride=128, tile_batch=64):
    """
    Samples full panoramas It [B, 3, H, W] with a model trained on tile x tile crops: at every step the velocity
    is predicted on overlapping crops (wrapping around the 360 deg seam) and averaged where crops overlap.
    """
    cond, w = model.make_conditioning(I0, It_prev, w0, wt, wt_prev, angle_mask, view_mask)
    B, _, H, W = I0.shape
    assert H == tile, 'panorama height {} must equal the tile size {}'.format(H, tile)
    device = I0.device

    # column indices of every tile, wrapping at the seam
    offsets = torch.arange(0, W, stride, device=device)
    cols = (offsets[:, None] + torch.arange(tile, device=device)[None]) % W  # [n_tiles, tile]
    n_tiles = cols.shape[0]
    counts = torch.zeros(W, device=device).index_add_(0, cols.flatten(), torch.ones(cols.numel(), device=device))

    def tiles(x):
        # [B, C, H, W] -> [B * n_tiles, C, H, tile]
        return x[..., cols].permute(0, 3, 1, 2, 4).flatten(0, 1)

    cond_tiles = tiles(cond)
    w_tiles = w.repeat_interleave(n_tiles, dim=0)

    def velocity(z, t):
        z_tiles = tiles(z)
        t_tiles = t.repeat_interleave(n_tiles)
        t_tiles = t_tiles.view(-1, 1, 1, 1)
        v_tiles = torch.cat([model._forward_sample(z_tiles[i:i + tile_batch], t_tiles[i:i + tile_batch],
                                                   cond_tiles[i:i + tile_batch], w_tiles[i:i + tile_batch])
                             for i in range(0, z_tiles.shape[0], tile_batch)])
        # [B * n_tiles, C, H, tile] -> accumulate into [B, C, H, W]
        v_tiles = v_tiles.unflatten(0, (B, n_tiles)).permute(0, 2, 3, 1, 4).flatten(3)
        v = torch.zeros_like(z).index_add_(3, cols.flatten(), v_tiles.float())
        return v / counts

    z = model.noise_scale * torch.randn(B, 3, H, W, device=device)
    timesteps = torch.linspace(0.0, 1.0, model.steps + 1, device=device)

    def euler(z, t, t_next):
        return z + (t_next - t) * velocity(z, t.expand(B))

    def heun(z, t, t_next):
        v_t = velocity(z, t.expand(B))
        v_next = velocity(z + (t_next - t) * v_t, t_next.expand(B))
        return z + (t_next - t) * 0.5 * (v_t + v_next)

    stepper = {'euler': euler, 'heun': heun}[model.method]
    for i in range(model.steps - 1):
        z = stepper(z, timesteps[i], timesteps[i + 1])
    # last step euler
    return euler(z, timesteps[-2], timesteps[-1])


def annotated_grid(rows, ncols=None, title=None, col_labels=None, pad=2, margin=170, title_height=24, header_height=16):
    """
    rows: list of (label, list of [3, H, W] tensors in [0, 1]); None or missing tiles are left white.
    Rows may have different tile widths (crops vs panoramas). Returns a PIL image with the row labels in a left margin.
    """
    ncols = ncols or max(len(tiles) for _, tiles in rows)
    row_images = []
    for _, tiles in rows:
        ref = next(t for t in tiles if t is not None)
        tiles = [t if t is not None else torch.ones_like(ref) for t in tiles]
        tiles = tiles + [torch.ones_like(ref)] * (ncols - len(tiles))
        grid = torchvision.utils.make_grid(tiles[:ncols], nrow=ncols, padding=pad, pad_value=1.0)
        row_images.append(Image.fromarray((grid.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)))

    top = (title_height if title else 0) + (header_height if col_labels else 0)
    width = margin + max(r.width for r in row_images)
    canvas = Image.new('RGB', (width, top + sum(r.height for r in row_images)), 'white')
    draw = ImageDraw.Draw(canvas)
    if title:
        draw.text((4, 4), title, fill='black')
    if col_labels:
        tile_w = (row_images[0].width - pad) // ncols
        for c, label in enumerate(col_labels):
            draw.text((margin + pad + c * tile_w + 4, top - header_height + 2), label, fill='black')
    y = top
    for (label, _), img in zip(rows, row_images):
        canvas.paste(img, (margin, y))
        for k, line in enumerate(label.split('\n')):
            draw.text((4, y + img.height // 2 - 6 * len(label.split('\n')) + 12 * k), line, fill='black')
        y += img.height
    return canvas


def save_image(x, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torchvision.utils.save_image(x, path)


@torch.no_grad()
def main(args):
    device = torch.device('cuda')
    ckpt_path = resolve_checkpoint(args.checkpoint)
    run_dir = os.path.dirname(ckpt_path)
    ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
    print('Checkpoint:', ckpt_path)

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    train_args = checkpoint['args']
    print('Trained for {} epochs'.format(checkpoint.get('epoch', -1) + 1))

    # sampler settings from the command line, architecture from training
    train_args.sampling_method = args.sampling_method
    train_args.num_sampling_steps = args.num_sampling_steps
    model = Denoiser(train_args).to(device).eval()
    tile = train_args.img_size

    cameras = args.cameras
    if not cameras:
        split = cfg['data'][args.split]
        idx = np.unique(np.linspace(0, len(split) - 1, min(args.num_cameras, len(split))).round().astype(int))
        cameras = [split[i] for i in idx]
    output_dir = args.output_dir or os.path.join(run_dir, 'samples-{}'.format(ckpt_name))
    print('Cameras:', cameras)
    print('Output:', output_dir)

    shared_weather = xr.open_zarr(os.path.join(cfg['data']['weather_root_dir'], 'inca.zarr'), consolidated=True, chunks=None)
    weight_keys = {'1': 'model_ema1', '2': 'model_ema2', 'none': 'model'}
    settings = [(e, c) for e in args.ema for c in args.cfg]
    if any(c != 1.0 for c in args.cfg) and getattr(train_args, 'weather_drop_prob', 0.0) == 0:
        print('WARNING: the checkpoint was trained without weather dropout, its null weather tokens are untrained')

    for mode in args.mode:
        mode_dir = os.path.join(output_dir, mode if mode == 'panorama' else 'crop-{}'.format(args.crop))
        transform = get_photocast_transforms(cfg, crop=args.crop if mode == 'crop' else None)

        # load the conditioning and ground truth of every camera once
        data = {}  # camera -> (list of samples, batched tensors on the device)
        for camera in cameras:
            try:
                dataset = Photocast2DDatasetZarr(csv_file=camera_csv_path(cfg, camera),
                                                 image_root_dir=cfg['data']['image_root_dir'],
                                                 shared_weather_ds=shared_weather,
                                                 weather_scaler_file=cfg['data']['weather_scaler_file'],
                                                 viewshed_radius=cfg['data']['viewshed_radius'],
                                                 img_transform=transform,
                                                 lead_times=cfg['data']['lead_times'],
                                                 date_ranges=cfg['data'].get('date_ranges'))
            except (FileNotFoundError, IndexError) as e:
                print('  skipping {}: {}'.format(camera, e))
                continue
            rng = np.random.default_rng(args.seed)
            idx = rng.choice(len(dataset), size=min(args.num_samples, len(dataset)), replace=False)
            samples = [load_sample(dataset, int(i), args.seed) for i in idx]
            if mode == 'panorama':
                samples = match_panorama_widths(samples, camera)
            tensors = [torch.stack([s['tensors'][k] for s in samples]).to(device) for k in range(len(samples[0]['tensors']))]
            data[camera] = (samples, tensors)

        os.makedirs(mode_dir, exist_ok=True)
        with open(os.path.join(mode_dir, 'samples.json'), 'w') as f:
            json.dump({c: [s['datetimes'] for s in samples] for c, (samples, _) in data.items()}, f, indent=2)

        # forecasts: (camera, ema, cfg) -> [num_samples, num_seeds, 3, H, W] in [0, 1]
        forecasts = {}
        for ema, cfg_scale in settings:
            model.load_state_dict(checkpoint[weight_keys[ema]])
            model.cfg_scale = cfg_scale
            for camera, (samples, tensors) in data.items():
                I0, It, It_prev, w0, wt, wt_prev, angle_mask, view_mask = tensors
                print('{} ema {} cfg {} camera {} ({}x{})'.format(mode, ema, cfg_scale, camera, I0.shape[-2], I0.shape[-1]))
                # every sample repeated num_seeds times; same seed -> same noise for every setting
                cond = [x.repeat_interleave(args.num_seeds, dim=0) for x in (I0, It_prev, w0, wt, wt_prev, angle_mask, view_mask)]
                torch.manual_seed(args.seed)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    if mode == 'crop':
                        x = model.generate(*cond)
                    else:
                        x = generate_tiled(model, *cond, tile=tile, stride=args.tile_stride, tile_batch=args.tile_batch)
                # the model gets no training signal outside the field of view: show it black like the real frames
                x = torch.where(cond[-1] > 0, x.float(), -1.0)
                x = to_unit(x).cpu().unflatten(0, (len(samples), args.num_seeds))
                forecasts[(camera, ema, cfg_scale)] = x

                sample_dir = os.path.join(mode_dir, 'ema{}'.format(ema), 'cfg{}'.format(cfg_scale), camera)
                for s, sample in enumerate(samples):
                    name = sample['datetimes'][1]
                    save_image(to_unit(It[s]).cpu(), os.path.join(sample_dir, '{}_gt.png'.format(name)))
                    for k in range(args.num_seeds):
                        save_image(x[s, k], os.path.join(sample_dir, '{}_seed{}.png'.format(name, args.seed + k)))

        # per-camera grids
        grid_dir = os.path.join(mode_dir, 'grids')
        os.makedirs(grid_dir, exist_ok=True)
        forecast_keys = [(e, c, k) for e, c in settings for k in range(args.num_seeds)]
        forecast_labels = ['ema{} cfg{} #{}'.format(e, c, k) for e, c, k in forecast_keys]
        for camera, (samples, tensors) in data.items():
            I0, It, It_prev = (to_unit(t).cpu() for t in tensors[:3])
            title = '{}  camera {}  ({})'.format(mode, camera, ckpt_name)
            if mode == 'crop':
                # one row per sample: It_prev | I0 | It | forecasts
                rows = []
                for s, sample in enumerate(samples):
                    dt_prev, dt0, dtt = sample['datetimes']
                    rows.append(('I0 {}\nIt {}'.format(dt0, dtt),
                                 [It_prev[s], I0[s], It[s]] + [forecasts[(camera, e, c)][s, k] for e, c, k in forecast_keys]))
                grid = annotated_grid(rows, title=title, col_labels=['It_prev (-24h)', 'I0', 'It (truth)'] + forecast_labels)
            else:
                # panoramas are wide: one row per image, blocks of rows per sample separated by a blank row
                rows = []
                for s, sample in enumerate(samples):
                    dt_prev, dt0, dtt = sample['datetimes']
                    rows += [('It_prev {}'.format(dt_prev), [It_prev[s]]),
                             ('I0 {}'.format(dt0), [I0[s]]),
                             ('It {} (truth)'.format(dtt), [It[s]])]
                    rows += [('forecast {}'.format(lab), [forecasts[(camera, e, c)][s, k]])
                             for lab, (e, c, k) in zip(forecast_labels, forecast_keys)]
                    if s + 1 < len(samples):
                        rows.append(('', [torch.ones(3, 24, It.shape[-1])]))
                grid = annotated_grid(rows, ncols=1, title=title)
            grid.save(os.path.join(grid_dir, '{}.png'.format(camera)))

        # overview per setting: truth and first forecast of every camera
        for e, c in settings:
            rows = []
            for camera, (samples, tensors) in data.items():
                It = to_unit(tensors[1]).cpu()
                if mode == 'crop':
                    tiles = []
                    for s in range(len(samples)):
                        tiles += [It[s], forecasts[(camera, e, c)][s, 0]]
                    rows.append((camera[:12], tiles))
                else:
                    rows += [('{} truth'.format(camera[:12]), [It[0]]),
                             ('{} forecast'.format(camera[:12]), [forecasts[(camera, e, c)][0, 0]])]
            col_labels = [lab for s in range(args.num_samples) for lab in ('#{} truth'.format(s), '#{} forecast'.format(s))] \
                if mode == 'crop' else None
            grid = annotated_grid(rows, ncols=None if mode == 'crop' else 1, col_labels=col_labels,
                                  title='{} ema{} cfg{}  ({})'.format(mode, e, c, ckpt_name))
            grid.save(os.path.join(mode_dir, 'overview_ema{}_cfg{}.png'.format(e, c)))

    print('Done.')


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
