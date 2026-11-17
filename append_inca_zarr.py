# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy<2.3",
#     "pandas",
#     "xarray>=2024.6,<2025.2",
#     "zarr>=2.17,<3",
#     "netCDF4",
#     "dask",
# ]
# ///
"""
Appends INCA analyses in netCDF files (e.g. 2026-01-01 .. 2026-03-31) to the existing inca.zarr archive, in the
archive's own format: the 15 weather fields, float32, one chunk per time step over the full grid, blosc/lz4, time
encoded as "minutes since 2024-01-01". The netCDF reader is broken in the training env (numpy 2), so run it with
uv, which builds an isolated env from the header above:

    uv run append_inca_zarr.py --nc "/path/to/inca/2026/2026*.nc"            # dry run: checks only, writes nothing
    uv run append_inca_zarr.py --nc "/path/to/inca/2026/2026*.nc" --write    # backup, then append
    uv run append_inca_zarr.py --restore data/weather/inca/inca.zarr.backup-<timestamp>   # undo an append

Dry run (always runs first):
  1. the new files have the 15 fields and exactly the archive's x / y grid;
  2. their times: only those after the archive's last time are kept (duplicates dropped, sorted), gaps against
     the 10-minute step are listed;
  3. data quality per day (on every 6th time step): NaN fraction and constant fields per variable, flagging days
     like the broken 2024-12-29;
  4. units: per-variable mean / std of the archive's last week against the first week of the new data;
  5. a rehearsal: the archive's last 3 steps are copied to a scratch store with the same encoding, the first
     new steps appended, and the result re-read and compared (times decode, values round-trip).

--write then:
  - backs up every file an append modifies (.zmetadata, .zgroup, each array's .zarray / .zattrs and the datetime
    coordinate; the data chunks of existing time steps are never rewritten, since chunks hold one step each);
  - appends month by month, re-opening and verifying the archive after each month.
Stop any job that reads the archive (training, evaluation) while it is written.
"""
import argparse
import glob
import json
import os
import shutil
import sys
import tempfile
import time

import warnings

import numpy as np
import pandas as pd
import xarray as xr

FIELDS = ['FF', 'PT', 'WG', 'CT', 'RS', 'Z0', 'TW', 'TD', 'TG', 'ZS', 'RR', 'SU', 'ZM', 'DD', 'TT']
STEP = pd.Timedelta(minutes=10)


warnings.filterwarnings('ignore', category=RuntimeWarning)


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------------------------------------ opening

def open_archive(path):
    ds = xr.open_zarr(path, consolidated=True, chunks=None)
    missing = [f for f in FIELDS if f not in ds]
    assert not missing, f'archive lacks {missing}'
    return ds


def open_new(pattern, time_dim):
    files = sorted(glob.glob(pattern))
    assert files, f'no files match {pattern}'
    log(f'{len(files)} netCDF files: {os.path.basename(files[0])} .. {os.path.basename(files[-1])}')

    def preprocess(ds):
        if time_dim != 'datetime' and time_dim in ds.dims:
            ds = ds.rename({time_dim: 'datetime'})
        missing = [f for f in FIELDS if f not in ds]
        if missing:
            raise ValueError(f'{ds.encoding.get("source", "?")}: missing fields {missing}')
        ds = ds[FIELDS]
        return ds.drop_vars([c for c in ds.coords if c not in ('datetime', 'x', 'y')])

    ds = xr.open_mfdataset(files, preprocess=preprocess, combine='by_coords', join='inner', data_vars='minimal',
                           coords='minimal', compat='override', parallel=False, engine='netcdf4', chunks={})
    return ds


# ------------------------------------------------------------------------------------------------ checks

def check_grid(arch, new):
    for c in ('x', 'y'):
        a, b = arch[c].values, new[c].values
        if a.shape != b.shape:
            raise SystemExit(f'grid mismatch on {c}: archive {a.shape} [{a[0]} .. {a[-1]}], new {b.shape} [{b[0]} .. {b[-1]}]')
        if not np.allclose(a, b, atol=1e-3):
            if np.allclose(a, b[::-1], atol=1e-3):
                raise SystemExit(f'{c} is reversed in the new files: flip it (new = new.isel({c}=slice(None, None, -1))) first')
            raise SystemExit(f'grid mismatch on {c}: max difference {np.abs(a - b).max()}')
    dims = [tuple(new[f].dims) for f in FIELDS]
    assert all(d == ('datetime', 'y', 'x') for d in dims), f'unexpected dimension order: {set(dims)}'
    log(f'grid: OK ({arch.sizes["y"]} x {arch.sizes["x"]}, same x / y as the archive)')


def select_times(arch, new):
    last = pd.Timestamp(arch.datetime.values[-1])
    t = pd.DatetimeIndex(new.datetime.values)
    log(f'archive: {arch.sizes["datetime"]} steps, {pd.Timestamp(arch.datetime.values[0])} .. {last}')
    log(f'new files: {len(t)} steps, {t.min()} .. {t.max()}')
    if t.has_duplicates:
        log(f'  {t.duplicated().sum()} duplicate times dropped')
        new = new.isel(datetime=~t.duplicated())
    new = new.sortby('datetime')
    keep = pd.DatetimeIndex(new.datetime.values) > last
    if (~keep).any():
        log(f'  {(~keep).sum()} steps at or before the archive end dropped')
    new = new.isel(datetime=np.flatnonzero(keep))
    t = pd.DatetimeIndex(new.datetime.values)
    assert len(t), 'nothing to append: every new time is already in the archive'
    if t[0] - last != STEP:
        log(f'  NOTE: {t[0] - last} between the archive end and the first new step (expected {STEP})')
    off = t[(t - t[0]) % STEP != pd.Timedelta(0)]
    if len(off):
        log(f'  WARNING: {len(off)} times off the 10-minute grid, e.g. {list(off[:3])}')
    full = pd.date_range(t[0], t[-1], freq=STEP)
    gaps = full.difference(t)
    if len(gaps):
        days = pd.Series(gaps.date).value_counts().sort_index()
        log(f'  {len(gaps)} missing 10-minute steps on {len(days)} days: ' +
            ', '.join(f'{d} ({n})' for d, n in list(days.items())[:12]) + (' ...' if len(days) > 12 else ''))
    else:
        log('  no missing steps')
    log(f'to append: {len(t)} steps, {t[0]} .. {t[-1]}')
    return new


def check_quality(arch, new, every=6, days=7):
    """
    NaN fraction and constant fields per day and variable, on every `every`-th step. NaNs are compared with the
    archive's own rate over its last `days` days (SU, for one, is NaN at night).
    """
    base = arch.isel(datetime=slice(-int(days * 24 * 6), None, every))
    base_nan = {f: float(base[f].isnull().mean()) for f in FIELDS}
    sub = new.isel(datetime=slice(None, None, every))
    nan = sub.isnull().mean(dim=('y', 'x')).compute()
    std = sub.std(dim=('y', 'x')).compute()
    day = pd.DatetimeIndex(sub.datetime.values).normalize()
    rows = []
    for f in FIELDS:
        n = pd.Series(nan[f].values, index=day).groupby(level=0).mean()
        c = pd.Series((std[f].values == 0) | ~np.isfinite(std[f].values), index=day).groupby(level=0).mean()
        rows.append(pd.DataFrame({'nan': n, 'const': c}).assign(var=f, base=base_nan[f]))
    q = pd.concat(rows)
    q.index.name = 'day'
    q = q.reset_index()
    # constant fields are normal for RR (no rain), SU at night, ... : flag only NaNs and fields constant all day
    # that usually vary (temperatures, pressure, wind)
    vary = ['FF', 'PT', 'TW', 'TD', 'TG', 'ZM', 'TT']
    bad = q[(q['nan'] > q['base'] + 0.05) | ((q['const'] > 0.99) & q['var'].isin(vary))]
    if len(bad):
        log(f'quality: {bad["day"].nunique()} days flagged (NaN rate 5 points above the archive\'s, or a usually varying field constant):')
        for d, g in bad.groupby('day'):
            log(f'  {d:%Y-%m-%d}: ' + ', '.join(f'{r.var} nan {r.nan:.0%} (archive {r.base:.0%}) const {r.const:.0%}' for r in g.itertuples()))
    else:
        log(f'quality: OK ({q["day"].nunique()} days: NaN rates as in the archive, no frozen fields)')
    return sorted(bad['day'].unique())


def check_units(arch, new, days=7):
    n_a = int(days * 24 * 6)
    a = arch.isel(datetime=slice(-n_a, None, 6))
    b = new.isel(datetime=slice(0, n_a, 6))
    log(f'units: archive last {days} days vs new first {days} days (mean / std over the grid and time)')
    flagged = []
    for f in FIELDS:
        ma, sa = float(a[f].mean()), float(a[f].std())
        mb, sb = float(b[f].mean().compute()), float(b[f].std().compute())
        ratio = (sb + 1e-6) / (sa + 1e-6)
        shift = abs(mb - ma) / (sa + 1e-6)
        flag = ratio > 3 or ratio < 1 / 3 or shift > 3
        flagged += [f] if flag else []
        log(f'  {f:3s} archive {ma:10.3f} / {sa:9.3f}   new {mb:10.3f} / {sb:9.3f}' + ('   <-- check units' if flag else ''))
    if flagged:
        log(f'  WARNING: {flagged} differ a lot between the two weeks: check the units before writing')
    return flagged


# ------------------------------------------------------------------------------------------------ writing

def prepare(new):
    new = new[FIELDS].astype('float32').chunk({'datetime': 1, 'y': -1, 'x': -1})
    for v in list(new.variables):
        new[v].encoding = {}
    return new


def append(store, new):
    new.to_zarr(store, mode='a', append_dim='datetime', consolidated=True)


def verify(store, new, n_before):
    ds = xr.open_zarr(store, consolidated=True, chunks=None)
    t_new = pd.DatetimeIndex(new.datetime.values)
    got = pd.DatetimeIndex(ds.datetime.values[n_before:n_before + len(t_new)])
    assert ds.sizes['datetime'] == n_before + len(t_new), f'{ds.sizes["datetime"]} steps, expected {n_before + len(t_new)}'
    assert got.equals(t_new), f'times do not round-trip: {got[:3]} vs {t_new[:3]}'
    assert pd.DatetimeIndex(ds.datetime.values).is_monotonic_increasing, 'times are not increasing'
    for k in sorted({0, len(t_new) // 2, len(t_new) - 1}):
        for f in ('CT', 'TT', 'SU'):
            a = ds[f].isel(datetime=n_before + k).values
            b = new[f].isel(datetime=k).values
            assert np.allclose(a, b, equal_nan=True), f'{f} at {t_new[k]} does not round-trip'
    return ds


def rehearse(arch_path, arch, new, k=12):
    """Append the first k new steps to a scratch copy of the archive's last 3 steps, written with its encoding"""
    tmp = tempfile.mkdtemp(prefix='inca_rehearsal_')
    store = os.path.join(tmp, 'inca.zarr')
    try:
        src = xr.open_zarr(arch_path, consolidated=True)
        base = src[FIELDS].isel(datetime=slice(-3, None))
        enc = {f: {k_: v for k_, v in src[f].encoding.items() if k_ in ('chunks', 'compressor', 'dtype', '_FillValue', 'filters')}
               for f in FIELDS}
        enc['datetime'] = {k_: v for k_, v in src['datetime'].encoding.items() if k_ in ('units', 'calendar', 'dtype')}
        for v in base.variables:
            base[v].encoding = {}
        base.to_zarr(store, mode='w', consolidated=True, encoding=enc)
        part = prepare(new.isel(datetime=slice(0, k)))
        append(store, part)
        ds = verify(store, part, 3)
        meta = json.load(open(os.path.join(store, '.zmetadata')))['metadata']
        units = meta['datetime/.zattrs'].get('units')
        chunks = meta['CT/.zarray']['chunks']
        comp = meta['CT/.zarray']['compressor']
        arch_meta = json.load(open(os.path.join(arch_path, '.zmetadata')))['metadata']
        assert units == arch_meta['datetime/.zattrs']['units'], f'time units changed: {units}'
        assert chunks == arch_meta['CT/.zarray']['chunks'], f'chunks changed: {chunks}'
        assert comp == arch_meta['CT/.zarray']['compressor'], f'compressor changed: {comp}'
        log(f'rehearsal: OK (appended {k} steps to a scratch copy: times decode as {pd.Timestamp(ds.datetime.values[3])} .., '
            f'units "{units}", chunks {chunks}, {comp["cname"]})')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def backup(store):
    stamp = time.strftime('%Y%m%d-%H%M%S')
    dst = f'{store.rstrip("/")}.backup-{stamp}'
    os.makedirs(dst)
    n = 0
    for name in ('.zmetadata', '.zgroup', '.zattrs'):
        if os.path.exists(os.path.join(store, name)):
            shutil.copy2(os.path.join(store, name), dst); n += 1
    for var in sorted(os.listdir(store)):
        d = os.path.join(store, var)
        if not os.path.isdir(d):
            continue
        os.makedirs(os.path.join(dst, var))
        if var == 'datetime':
            for f in os.listdir(d):
                shutil.copy2(os.path.join(d, f), os.path.join(dst, var)); n += 1
        else:
            for name in ('.zarray', '.zattrs'):
                if os.path.exists(os.path.join(d, name)):
                    shutil.copy2(os.path.join(d, name), os.path.join(dst, var)); n += 1
    length = xr.open_zarr(store, consolidated=True, chunks=None).sizes['datetime']
    json.dump({'store': os.path.abspath(store), 'datetime_length': int(length), 'created': stamp},
              open(os.path.join(dst, 'backup.json'), 'w'), indent=1)
    log(f'backup: {n} files to {dst} (archive length {length})')
    return dst


def restore(backup_dir):
    info = json.load(open(os.path.join(backup_dir, 'backup.json')))
    store, length = info['store'], info['datetime_length']
    log(f'restoring {store} to {length} time steps from {backup_dir}')
    for name in ('.zmetadata', '.zgroup', '.zattrs'):
        if os.path.exists(os.path.join(backup_dir, name)):
            shutil.copy2(os.path.join(backup_dir, name), store)
    removed = 0
    for var in sorted(os.listdir(backup_dir)):
        src = os.path.join(backup_dir, var)
        if not os.path.isdir(src):
            continue
        dst = os.path.join(store, var)
        if var == 'datetime':
            for f in os.listdir(dst):
                if not f.startswith('.') and not os.path.exists(os.path.join(src, f)):
                    os.remove(os.path.join(dst, f)); removed += 1
        for f in os.listdir(src):
            shutil.copy2(os.path.join(src, f), dst)
        if var not in ('datetime', 'x', 'y'):
            # chunk files of appended steps: "<t>.0.0" with t >= the old length
            for f in os.listdir(dst):
                if not f.startswith('.') and int(f.split('.')[0]) >= length:
                    os.remove(os.path.join(dst, f)); removed += 1
    ds = xr.open_zarr(store, consolidated=True, chunks=None)
    assert ds.sizes['datetime'] == length
    log(f'restored: {length} steps, last {pd.Timestamp(ds.datetime.values[-1])}; {removed} appended chunk files removed')


# ------------------------------------------------------------------------------------------------ main

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--zarr', default='data/weather/inca/inca.zarr')
    p.add_argument('--nc', help='Glob of the netCDF files to append (quote it)')
    p.add_argument('--time_dim', default='datetime', help='Name of the time dimension in the netCDF files')
    p.add_argument('--write', action='store_true', help='Back up, then append (default: dry run)')
    p.add_argument('--force', action='store_true', help='Write even if the quality or unit checks flag something')
    p.add_argument('--restore', metavar='BACKUP_DIR', help='Undo an append from its backup folder')
    args = p.parse_args()

    if args.restore:
        restore(args.restore)
        return
    assert args.nc, '--nc is required'

    arch = open_archive(args.zarr)
    new = open_new(args.nc, args.time_dim)
    check_grid(arch, new)
    new = select_times(arch, new)
    bad_days = check_quality(arch, new)
    bad_units = check_units(arch, new)
    rehearse(args.zarr, arch, new)

    if not args.write:
        log('\ndry run: nothing written. Re-run with --write to append.')
        return
    if (bad_days or bad_units) and not args.force:
        sys.exit('checks flagged days or units (above): fix the inputs, or re-run with --force to write anyway')

    backup(args.zarr)
    n0 = arch.sizes['datetime']
    months = pd.DatetimeIndex(new.datetime.values).to_period('M')
    for m in months.unique():
        part = prepare(new.isel(datetime=np.flatnonzero(months == m)))
        t0 = time.time()
        append(args.zarr, part)
        verify(args.zarr, part, n0)
        n0 += part.sizes['datetime']
        log(f'appended {m}: {part.sizes["datetime"]} steps in {time.time() - t0:.0f} s (archive now {n0} steps)')
    ds = xr.open_zarr(args.zarr, consolidated=True, chunks=None)
    log(f'done: {ds.sizes["datetime"]} steps, {pd.Timestamp(ds.datetime.values[0])} .. {pd.Timestamp(ds.datetime.values[-1])}')


if __name__ == '__main__':
    main()
