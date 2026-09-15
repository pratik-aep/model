#!/usr/bin/env python3
"""Re-download ERA5 wind + GLORYS currents for 2019-2026 over the FULL
circumpolar domain, matching the 2008-2018 files.

Why: the existing 2021-2026 files cover disjoint longitude sectors (wind at
90..180, currents at -180..-120), so their intersection is empty and not one
berg-row in those years can be used. Bergs are circumpolar (only 5-11% fall in
the GLORYS sector), so both variables need the full domain.

SAFETY: synthetic fallback is force-disabled. The repo's download helpers default
to allow_synthetic_fallback=True, which silently writes FABRICATED data under a
normal-looking filename on any failure. Never enable that for training data.

Usage:
    python3 scripts/download_missing_forcing.py --dry-run
    python3 scripts/download_missing_forcing.py --years 2021 2022
    python3 scripts/download_missing_forcing.py            # all missing years
"""
import argparse, glob, os, sys, time
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path("/Users/pratiksmac/Downloads/icccy/iceberg-drift")
DATA = Path("/Users/pratiksmac/Downloads/data")
sys.path.insert(0, str(ROOT / "src"))

BBOX = (-180.0, -80.0, 180.0, -50.0)          # full circumpolar, as 2008-2018
DEFAULT_YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
MIN_LON_SPAN = 300.0                          # a real circumpolar file spans ~360


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def inspect(path):
    """Return (lon_span, lat_span, ntime) for an existing file, or None."""
    try:
        d = xr.open_dataset(path)
    except Exception:
        return None
    try:
        tc = "valid_time" if "valid_time" in d.coords else "time"
        return (float(d.longitude.max() - d.longitude.min()),
                float(d.latitude.max() - d.latitude.min()),
                int(d.sizes.get(tc, 0)))
    finally:
        d.close()


def status(year):
    """Which of this year's files are missing or too narrow."""
    w = glob.glob(str(DATA / str(year) / "era5_wind_*.nc"))
    c = glob.glob(str(DATA / str(year) / "currents_GLORYS12_*.nc"))
    need_w = need_c = True
    wi = ci = None
    if w:
        wi = inspect(w[0])
        need_w = wi is None or wi[0] < MIN_LON_SPAN
    if c:
        ci = inspect(c[0])
        need_c = ci is None or ci[0] < MIN_LON_SPAN
    return need_w, need_c, wi, ci


def verify(path, label):
    """Post-download check: real coverage, plausible values, not synthetic."""
    info = inspect(path)
    if info is None:
        log(f"  !! {label}: unreadable")
        return False
    span, latspan, nt = info
    ok = span >= MIN_LON_SPAN
    d = xr.open_dataset(path)
    try:
        var = "u10" if "u10" in d.data_vars else "uo"
        tc = "valid_time" if "valid_time" in d.coords else "time"
        a = d[var].isel({tc: 0}).values
        finite = np.isfinite(a)
        sd = float(np.nanstd(a)) if finite.any() else 0.0
    finally:
        d.close()
    log(f"  {label}: lon_span={span:.1f} lat_span={latspan:.1f} nt={nt} "
        f"{var}_std={sd:.3f} {'OK' if ok and sd > 0 else 'SUSPECT'}")
    return ok and sd > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=DEFAULT_YEARS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--wind-only", action="store_true")
    ap.add_argument("--currents-only", action="store_true")
    a = ap.parse_args()

    log(f"target domain lon {BBOX[0]}..{BBOX[2]}  lat {BBOX[1]}..{BBOX[3]}")
    plan = []
    for y in a.years:
        nw, nc, wi, ci = status(y)
        if a.currents_only:
            nw = False
        if a.wind_only:
            nc = False
        log(f"{y}: wind {'NEEDED' if nw else 'ok'} {wi}  |  currents {'NEEDED' if nc else 'ok'} {ci}")
        if nw or nc:
            plan.append((y, nw, nc))

    if not plan:
        log("nothing to download.")
        return
    gb = sum(1.3 * nw + 4.3 * nc for _, nw, nc in plan)
    import shutil
    free_gb = shutil.disk_usage(DATA).free / 1024**3
    log(f"\nPLAN: {len(plan)} years, ~{gb:.0f} GB needed | {free_gb:.1f} GB free")
    if free_gb < gb + 5:
        log(f"ABORT: need ~{gb + 5:.0f} GB (incl. 5 GB headroom), only {free_gb:.1f} GB free.")
        log("Free space or run fewer years with --years.")
        return
    if a.dry_run:
        log("dry run - nothing downloaded")
        return

    # Load download.py directly by path: the package __init__ pulls in
    # models/ensemble.py, which currently has a syntax error unrelated to
    # downloading. download.py itself imports only stdlib + third-party.
    import importlib.util
    _p = ROOT / "src" / "iceberg_drift" / "data_processing" / "download.py"
    _spec = importlib.util.spec_from_file_location("_idl_download", _p)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    download_era5_wind = _mod.download_era5_wind
    download_copernicus_currents = _mod.download_copernicus_currents

    for y, nw, nc in plan:
        s, e = f"{y}-01-01", f"{y}-12-31"
        out = DATA / str(y)
        out.mkdir(parents=True, exist_ok=True)
        if nw:
            log(f"{y}: ERA5 wind ...")
            try:
                # NOTE: both download helpers append the year to output_dir
                # internally, so pass the PARENT (data/), not data/<year>.
                download_era5_wind(output_dir=str(DATA), start_date=s, end_date=e,
                                   bbox=BBOX, allow_synthetic_fallback=False)  # never synthesise
                f = glob.glob(str(out / "era5_wind_*.nc"))
                if f:
                    verify(f[0], f"{y} wind")
            except Exception as ex:
                log(f"  !! {y} wind FAILED: {ex}")
        if nc:
            log(f"{y}: GLORYS currents ...")
            try:
                download_copernicus_currents(output_dir=str(DATA), start_date=s, end_date=e,
                                             bbox=BBOX, allow_synthetic_fallback=False)
                f = glob.glob(str(out / "currents_GLORYS12_*.nc"))
                if f:
                    verify(f[0], f"{y} currents")
            except Exception as ex:
                log(f"  !! {y} currents FAILED: {ex}")

    log("\ndone. next: re-run scripts/sample_forcing_year.py for each new year,")
    log("then rebuild the table and retrain.")


if __name__ == "__main__":
    main()
