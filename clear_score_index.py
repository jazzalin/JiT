"""
Per-frame clear-sky / visibility index of the Photocast cameras, at the hour mark (minute == 0).

For every hourly frame of every camera, from a 64 x 512 thumbnail (JPEG decoded at reduced size):
  - lum, lum_top:      mean brightness of the image / of its top 35% (sky region of most panoramas)
  - blue_top:          blue-sky fraction in the top 35% (blue - red > 0.08 and blue > 0.35)
  - grey_top:          overcast / fog fraction in the top 35% (saturation < 0.08 and brightness > 0.45)
  - detail, detail_low: mean local detail (absolute difference to a blurred copy) of the image / of its bottom 65%
  - grid:              local detail on an 8 x 32 tile grid (saved separately, float16)
  - sun_elev:          solar elevation (degrees) at the camera
  - inca_SU, inca_CT:  INCA sunshine and cloud cover at the camera (from the camera CSV)
Then, per camera:
  - visibility = detail_low / detail_ref, clipped to [0, 1.5], where detail_ref is the 95th percentile of
    detail_low over the camera's daytime frames within +-30 days of the frame's date (any year): ~1 when the
    landscape is as detailed as on the camera's clearest days of the same season, ~0 in thick fog
  - tile_visibility: the same per tile (grid / per-tile 95th percentile), for per-region reference selection
  - clear_score = visibility * (0.6 + 0.4 * blue_top) * (1 - 0.5 * grey_top), 0 at night (sun_elev < 3 deg)
The raw features are kept, so the score can be redefined without re-reading the images.

Outputs in --output_dir:
  <camera>.csv          one row per hourly frame (time, path and the features above)
  <camera>_grid.npy     [n_frames, 8, 32] float16 tile detail (row order of the CSV)
  <camera>_tileref.npy  [8, 32] per-tile daytime 95th percentile (tile_visibility = grid / tileref)
  summary.csv           per camera: frames, daytime frames, detail reference, clear-frame fraction

Example:
    python clear_score_index.py --config config.yml --output_dir data/images/references --workers 10
"""
import os
import sys
import time
import argparse
from multiprocessing import Pool

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageFilter

from photocast_dataloader import camera_csv_path

THUMB = (512, 64)       # (width, height) of the analysis thumbnail
GRID = (8, 32)          # tile grid (rows, columns)
TOP = int(0.35 * THUMB[1])


def solar_elevation(times_utc, lat, lon):
    """Approximate solar elevation in degrees (NOAA formulation, accurate to well under a degree)."""
    t = pd.DatetimeIndex(times_utc)
    doy = t.dayofyear.values
    hour = t.hour.values + t.minute.values / 60
    g = 2 * np.pi / 365 * (doy - 1 + (hour - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                       - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    ha = np.radians((hour * 60 + eqtime + 4 * lon) / 4 - 180)
    phi = np.radians(lat)
    cos_zen = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(ha)
    return 90 - np.degrees(np.arccos(np.clip(cos_zen, -1, 1)))


def seasonal_reference(times, detail, day, window_days=30, q=95, min_frames=50, fallback=np.nan):
    """
    95th percentile of the daytime detail within +-window_days of each frame's date (over all years), so that
    visibility compares a frame with the camera's clearest days of the same season: snow patches make winter
    scenes far more detailed than clear summer scenes.
    """
    doy = pd.DatetimeIndex(times).dayofyear.values
    day = np.asarray(day, bool)
    d_doy, d_val = doy[day], np.asarray(detail, float)[day]
    ref = {}
    for v in np.unique(doy):
        dist = np.abs(d_doy - v)
        sel = np.minimum(dist, 366 - dist) <= window_days
        ref[v] = np.nanpercentile(d_val[sel], q) if sel.sum() >= min_frames else fallback
    return np.array([ref[v] for v in doy])


def frame_features(path):
    im = Image.open(path)
    im.draft('RGB', (max(im.width // 8, THUMB[0]), max(im.height // 8, THUMB[1])))
    im = im.convert('RGB').resize(THUMB, Image.BILINEAR)
    x = np.asarray(im, np.float32) / 255
    lum = x.mean(-1)
    top = x[:TOP]
    blue = ((top[..., 2] - top[..., 0]) > 0.08) & (top[..., 2] > 0.35)
    grey = ((top.max(-1) - top.min(-1)) < 0.08) & (lum[:TOP] > 0.45)
    blur = np.asarray(Image.fromarray((lum * 255).astype(np.uint8)).filter(ImageFilter.BoxBlur(2)), np.float32) / 255
    hp = np.abs(lum - blur)
    grid = hp.reshape(GRID[0], THUMB[1] // GRID[0], GRID[1], THUMB[0] // GRID[1]).mean((1, 3))
    return (float(lum.mean()), float(lum[:TOP].mean()), float(blue.mean()), float(grey.mean()),
            float(hp.mean()), float(hp[TOP:].mean())), grid.astype(np.float16)


def index_camera(job):
    camera, cfg, out_dir, overwrite = job
    out_csv = os.path.join(out_dir, f'{camera}.csv')
    if os.path.exists(out_csv) and not overwrite:
        # already indexed: only rebuild its summary row
        try:
            out = pd.read_csv(out_csv)
            day = out['valid'] & (out['sun_elev'] > 5)
            ref = float(np.nanpercentile(out.loc[day, 'detail_low'], 95)) if day.sum() >= 20 else np.nan
            return camera, dict(camera=camera, frames=len(out), valid=int(out['valid'].sum()), daytime=int(day.sum()),
                                detail_ref_annual=ref, clear_frac=float((out['clear_score'] > 0.6).mean())), 'exists'
        except Exception as e:  # noqa: BLE001
            return camera, None, f'exists, unreadable: {e}'
    try:
        df = pd.read_csv(camera_csv_path(cfg, camera), usecols=['reference', 'path', 'minute', 'latitude', 'longitude', 'SU', 'CT'])
    except Exception as e:  # noqa: BLE001
        return camera, None, f'csv: {e}'
    df = df[df['minute'] == 0].reset_index(drop=True)
    root = cfg['data']['image_root_dir']
    feats, grids, ok = [], [], []
    for p in df['path']:
        try:
            f, g = frame_features(os.path.join(root, p))
            feats.append(f); grids.append(g); ok.append(True)
        except Exception:  # noqa: BLE001  (missing or corrupt image)
            feats.append((np.nan,) * 6); grids.append(np.full(GRID, np.nan, np.float16)); ok.append(False)
    out = pd.DataFrame(feats, columns=['lum', 'lum_top', 'blue_top', 'grey_top', 'detail', 'detail_low'])
    out.insert(0, 'time', pd.to_datetime(df['reference']))
    out.insert(1, 'path', df['path'])
    out['valid'] = ok
    out['sun_elev'] = solar_elevation(out['time'], df['latitude'].iloc[0], df['longitude'].iloc[0]).round(2)
    out['inca_SU'] = df['SU'].values
    out['inca_CT'] = df['CT'].values
    grid = np.stack(grids) if grids else np.zeros((0,) + GRID, np.float16)

    day = out['valid'] & (out['sun_elev'] > 5)
    ref = float(np.nanpercentile(out.loc[day, 'detail_low'], 95)) if day.sum() >= 20 else np.nan
    tileref = (np.nanpercentile(grid[day.values].astype(np.float32), 95, axis=0)
               if day.sum() >= 20 else np.full(GRID, np.nan, np.float32))
    out['detail_ref'] = seasonal_reference(out['time'], out['detail_low'], day, fallback=ref)
    out['visibility'] = (out['detail_low'] / out['detail_ref']).clip(0, 1.5)
    score = out['visibility'] * (0.6 + 0.4 * out['blue_top']) * (1 - 0.5 * out['grey_top'])
    out['clear_score'] = score.where(out['sun_elev'] >= 3, 0.0).fillna(0.0)

    out.round(4).to_csv(out_csv, index=False)
    np.save(os.path.join(out_dir, f'{camera}_grid.npy'), grid)
    np.save(os.path.join(out_dir, f'{camera}_tileref.npy'), tileref.astype(np.float32))
    summary = dict(camera=camera, frames=len(out), valid=int(out['valid'].sum()), daytime=int(day.sum()),
                   detail_ref_annual=ref, clear_frac=float((out['clear_score'] > 0.6).mean()))
    return camera, summary, None


def main():
    parser = argparse.ArgumentParser('Hourly per-frame clear-score index')
    parser.add_argument('--config', default='config.yml')
    parser.add_argument('--output_dir', default='data/images/references')
    parser.add_argument('--cameras', nargs='*', default=None, help='Default: all train and val cameras of the config')
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--overwrite', action='store_true', help='Recompute cameras that already have an index')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cameras = args.cameras or list(dict.fromkeys(cfg['data']['val'] + cfg['data']['train']))
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'{len(cameras)} cameras -> {args.output_dir}', flush=True)

    t0, summaries = time.time(), []
    jobs = [(c, cfg, args.output_dir, args.overwrite) for c in cameras]
    with Pool(args.workers) as pool:
        for k, (camera, summary, err) in enumerate(pool.imap_unordered(index_camera, jobs), 1):
            if summary:
                summaries.append(summary)
            print(f'[{k}/{len(cameras)}] {camera[:8]} {err or "{} frames".format(summary["frames"])} '
                  f'({time.time() - t0:.0f}s)', flush=True)
    if summaries:
        path = os.path.join(args.output_dir, 'summary.csv')
        new = pd.DataFrame(summaries)
        if os.path.exists(path):  # keep the cameras indexed by earlier runs
            old = pd.read_csv(path)
            new = pd.concat([old[~old['camera'].isin(new['camera'])], new])
        new.sort_values('camera').to_csv(path, index=False)
    print(f'done in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    sys.exit(main())
