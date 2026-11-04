"""
Generate novel views for selected cameras with a trained class-conditional (camera id) JiT.

For every camera, one grid is saved with
  - a "real" row: the pixel-wise median of real images (approximates the static scene) followed by random real images,
  - one row per (EMA, CFG) setting, all using the same noise per column,
so structures can be compared against the real scene and the effect of CFG/EMA seen with everything else fixed.
An overview grid per (EMA, CFG) with one row per camera and the individual PNGs are saved as well.

Example:
    python sample_cameras.py --checkpoint results/jit_cam --cameras 0 12 d9be9239aa3d708b8e25601445937eed \
        --cfg 1.0 1.5 2.0 3.0 --ema 1 --num_samples 8
"""
import os
import re
import glob
import json
import argparse

import numpy as np
import torch
import torchvision
import yaml
from PIL import Image, ImageDraw

from denoiser import Denoiser
from photocast_dataloader import Photocast2DDataset, get_photocast_transforms, camera_csv_path


def get_args_parser():
    parser = argparse.ArgumentParser('JiT camera sampling', add_help=False)
    parser.add_argument('--checkpoint', required=True, type=str,
                        help='Checkpoint file, or a run folder (uses checkpoint-last.pth, else the highest checkpoint-N.pth)')
    parser.add_argument('--cameras_json', default=None, type=str,
                        help='Class id -> camera mapping (default: cameras.json next to the checkpoint)')
    parser.add_argument('--config', default='config.yml', type=str,
                        help='Data config, used to load real images for comparison')
    parser.add_argument('--output_dir', default=None, type=str,
                        help='Default: <run folder>/samples-<checkpoint name>')

    # what to generate
    parser.add_argument('--cameras', nargs='*', default=None,
                        help='Class ids or camera hashes (default: --num_cameras evenly spaced class ids)')
    parser.add_argument('--num_cameras', default=8, type=int)
    parser.add_argument('--cfg', nargs='+', default=[1.0, 1.5, 2.0, 3.0], type=float,
                        help='CFG scales, one grid row each')
    parser.add_argument('--ema', nargs='+', default=['1'], choices=['1', '2', 'none'],
                        help='Weights: ema_params1 (smoothest once training is long), ema_params2 or raw weights')
    parser.add_argument('--num_samples', default=8, type=int, help='Samples per camera and setting (grid columns)')
    parser.add_argument('--seed', default=0, type=int)

    # sampler
    parser.add_argument('--sampling_method', default='heun', type=str)
    parser.add_argument('--num_sampling_steps', default=50, type=int)
    parser.add_argument('--interval_min', default=0.0, type=float, help='CFG interval min')
    parser.add_argument('--interval_max', default=1.0, type=float, help='CFG interval max')

    # real images for comparison
    parser.add_argument('--num_real', default=None, type=int,
                        help='Random real images in the reference row (default: num_samples - 1, after the median)')
    parser.add_argument('--median_from', default=200, type=int,
                        help='Number of real images for the pixel-wise median (0 = no median)')
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


def resolve_cameras(requested, cameras, num_cameras):
    if not requested:
        return np.unique(np.linspace(0, len(cameras) - 1, min(num_cameras, len(cameras))).round().astype(int)).tolist()
    class_ids = []
    for c in requested:
        if c in cameras:
            class_ids.append(cameras.index(c))
        elif c.isdigit() and int(c) < len(cameras):
            class_ids.append(int(c))
        else:
            raise ValueError('Unknown camera "{}": not a class id < {} nor a hash in cameras.json'.format(c, len(cameras)))
    return class_ids


def to_unit(x):
    """[-1, 1] -> [0, 1]"""
    return ((x.float() + 1) / 2).clamp(0, 1)


def load_real_images(cfg, camera, num_real, median_from, seed):
    """Returns (median image or None, list of real images) in [0, 1], or (None, []) if the data is not available"""
    try:
        dataset = Photocast2DDataset(csv_file=camera_csv_path(cfg, camera), image_root_dir=cfg['data']['image_root_dir'],
                                     img_transform=get_photocast_transforms(cfg))
    except (FileNotFoundError, IndexError) as e:
        print('  no real images for {}: {}'.format(camera, e))
        return None, []

    rng = np.random.default_rng(seed)
    n = max(num_real, median_from)
    idx = rng.choice(len(dataset), size=min(n, len(dataset)), replace=False)
    try:
        images = [to_unit(dataset[int(i)]['tensors'][0]) for i in idx]
    except (FileNotFoundError, OSError) as e:
        print('  could not read real images for {}: {}'.format(camera, e))
        return None, []

    median = torch.stack(images).median(dim=0).values if median_from > 0 else None
    return median, images[:num_real]


def annotated_grid(rows, ncols, title=None, pad=2, margin=150, title_height=24):
    """
    rows: list of (label, list of [3, H, W] tensors in [0, 1]); missing tiles are left black.
    Returns a PIL image with the row labels in a left margin.
    """
    _, H, W = next(t for _, tiles in rows for t in tiles).shape
    blank = torch.zeros(3, H, W)
    row_images = []
    for _, tiles in rows:
        tiles = list(tiles) + [blank] * (ncols - len(tiles))
        grid = torchvision.utils.make_grid(tiles[:ncols], nrow=ncols, padding=pad)
        row_images.append(Image.fromarray((grid.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)))

    top = title_height if title else 0
    canvas = Image.new('RGB', (margin + row_images[0].width, top + sum(r.height for r in row_images)), 'white')
    draw = ImageDraw.Draw(canvas)
    if title:
        draw.text((4, 4), title, fill='black')
    y = top
    for (label, _), img in zip(rows, row_images):
        canvas.paste(img, (margin, y))
        draw.text((4, y + img.height // 2 - 6), label, fill='black')
        y += img.height
    return canvas


@torch.no_grad()
def main(args):
    device = torch.device('cuda')
    ckpt_path = resolve_checkpoint(args.checkpoint)
    run_dir = os.path.dirname(ckpt_path)
    ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
    print('Checkpoint:', ckpt_path)

    with open(args.cameras_json or os.path.join(run_dir, 'cameras.json')) as f:
        cameras = json.load(f)
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    train_args = checkpoint['args']
    assert train_args.class_num == len(cameras), \
        'Checkpoint has {} classes but cameras.json lists {} cameras'.format(train_args.class_num, len(cameras))
    print('Trained for {} epochs'.format(checkpoint.get('epoch', -1) + 1))

    # sampler settings from the command line, architecture from training
    train_args.sampling_method = args.sampling_method
    train_args.num_sampling_steps = args.num_sampling_steps
    train_args.interval_min, train_args.interval_max = args.interval_min, args.interval_max
    model = Denoiser(train_args).to(device).eval()

    class_ids = resolve_cameras(args.cameras, cameras, args.num_cameras)
    num_real = args.num_samples - 1 if args.num_real is None else args.num_real
    output_dir = args.output_dir or os.path.join(run_dir, 'samples-{}'.format(ckpt_name))
    os.makedirs(output_dir, exist_ok=True)
    print('Cameras (class id: hash):', {c: cameras[c] for c in class_ids})
    print('Output:', output_dir)

    weight_keys = {'1': 'model_ema1', '2': 'model_ema2', 'none': 'model'}
    settings = [(e, c) for e in args.ema for c in args.cfg]
    samples = {}  # (class id, ema, cfg) -> [num_samples, 3, H, W] in [0, 1]

    for ema in args.ema:
        model.load_state_dict(checkpoint[weight_keys[ema]])
        for cfg_scale in args.cfg:
            model.cfg_scale = cfg_scale
            for class_id in class_ids:
                print('ema {} cfg {} class {}'.format(ema, cfg_scale, class_id))
                labels = torch.full((args.num_samples,), class_id, dtype=torch.long, device=device)
                # same seed -> same initial noise for every camera, EMA and CFG setting
                torch.manual_seed(args.seed)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    images = to_unit(model.generate(labels)).cpu()
                samples[(class_id, ema, cfg_scale)] = images

                sample_dir = os.path.join(output_dir, 'ema{}'.format(ema), 'cfg{}'.format(cfg_scale),
                                          'class{:03d}_{}'.format(class_id, cameras[class_id]))
                os.makedirs(sample_dir, exist_ok=True)
                for k, img in enumerate(images):
                    torchvision.utils.save_image(img, os.path.join(sample_dir, 'seed{}_{:02d}.png'.format(args.seed, k)))

    # per-camera grid: real row + one row per setting
    grid_dir = os.path.join(output_dir, 'grids')
    os.makedirs(grid_dir, exist_ok=True)
    for class_id in class_ids:
        camera = cameras[class_id]
        median, real = load_real_images(cfg, camera, num_real, args.median_from, args.seed)
        rows = []
        if median is not None or real:
            rows.append(('real (median, random)' if median is not None else 'real', ([median] if median is not None else []) + real))
        rows += [('ema{} cfg{}'.format(e, c), list(samples[(class_id, e, c)])) for e, c in settings]
        grid = annotated_grid(rows, args.num_samples, title='class {}  camera {}  ({})'.format(class_id, camera, ckpt_name))
        grid.save(os.path.join(grid_dir, 'class{:03d}_{}.png'.format(class_id, camera)))

    # overview per setting: one row per camera
    for e, c in settings:
        rows = [('class {}'.format(k), list(samples[(k, e, c)])) for k in class_ids]
        grid = annotated_grid(rows, args.num_samples, title='ema{} cfg{}  ({})'.format(e, c, ckpt_name))
        grid.save(os.path.join(output_dir, 'overview_ema{}_cfg{}.png'.format(e, c)))

    print('Done.')


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
