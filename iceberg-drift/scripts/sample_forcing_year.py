"""Sample ERA5 wind + GLORYS currents for one year's iceberg-drift pairs.

Reads one 2D time-slice at a time (~6 MB) instead of loading whole arrays, so
several years can run concurrently inside 16 GB. Writes a per-year partial CSV.
"""
import glob, os, sys, time
import numpy as np
import pandas as pd
import xarray as xr

YEAR = int(sys.argv[1])
RAW = "/Users/pratiksmac/Downloads/data/raw/iceberg_positions/updated7_consol"
FORC = "/Users/pratiksmac/Downloads/data"
PART = sys.argv[2]
MIN_GAP, MAX_GAP, SPEED_CAP = 1, 9, 2.0

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {YEAR} {m}", flush=True)

# ---- pairs for this year ----
recs, n_fast = [], 0
for f in sorted(glob.glob(os.path.join(RAW, "*.csv"))):
    berg = os.path.splitext(os.path.basename(f))[0]
    try:
        d = pd.read_csv(f)
    except Exception:
        continue
    if "nic_3" not in d.columns:
        continue
    d = d[(d.nic_3 == 1) & ((d.nic_1 != 0) | (d.nic_2 != 0))]
    if len(d) < 2:
        continue
    d = d.assign(dt=pd.to_datetime(d.date.astype(int).astype(str), format="%Y%j", errors="coerce"))
    d = d.dropna(subset=["dt"]).sort_values("dt").drop_duplicates("dt")
    if len(d) < 2:
        continue
    lat, lon = d.nic_1.to_numpy(float), d.nic_2.to_numpy(float)
    t, s1, s2 = d.dt.to_numpy(), d.size_1.to_numpy(float), d.size_2.to_numpy(float)
    gap = np.diff(t).astype("timedelta64[D]").astype(int)
    dlon = lon[1:] - lon[:-1]
    dlon = np.where(dlon > 180, dlon - 360, np.where(dlon < -180, dlon + 360, dlon))
    dlat = lat[1:] - lat[:-1]
    for i in range(len(gap)):
        g = gap[i]
        if g < MIN_GAP or g > MAX_GAP:
            continue
        t0, t1 = pd.Timestamp(t[i]), pd.Timestamp(t[i + 1])
        if t0.year != YEAR or t1.year != YEAR:
            continue
        secs = g * 86400.0
        u = dlon[i] * 111.32 * np.cos(np.radians(lat[i])) * 1000.0 / secs
        v = dlat[i] * 110.57 * 1000.0 / secs
        if np.hypot(u, v) > SPEED_CAP:
            n_fast += 1
            continue
        recs.append((berg, t0, lat[i], lon[i], s1[i], s2[i], g, u, v))

P = pd.DataFrame(recs, columns=["berg_id", "date", "lat", "lon", "size_1",
                                "size_2", "gap_days", "u_berg", "v_berg"])
if len(P) == 0:
    log("no pairs"); P.to_csv(PART, index=False); sys.exit(0)
log(f"pairs {len(P)} (speed-filtered {n_fast})")

wf = glob.glob(f"{FORC}/{YEAR}/era5_wind_*.nc")
cf = glob.glob(f"{FORC}/{YEAR}/currents_GLORYS12_*.nc")
if not wf or not cf:
    log("MISSING forcing"); P.assign(u_wind=np.nan, v_wind=np.nan,
                                     u_curr=np.nan, v_curr=np.nan).to_csv(PART, index=False)
    sys.exit(0)

# expand pairs into gap days
pid, days = [], []
for i, r in P.iterrows():
    for k in range(int(r.gap_days)):
        pid.append(i); days.append(r.date + pd.Timedelta(days=k))
pid = np.asarray(pid); days = pd.DatetimeIndex(days)
plat = P.lat.to_numpy()[pid]; plon = P.lon.to_numpy()[pid]

def nearest_idx(coord, vals):
    c = np.asarray(coord, float); order = np.argsort(c); cs = c[order]
    j = np.clip(np.searchsorted(cs, vals), 1, len(cs) - 1)
    j = np.where(np.abs(vals - cs[j - 1]) <= np.abs(vals - cs[j]), j - 1, j)
    return order[j]

def in_range(coord, vals, tol):
    """True where vals lie inside the file's coverage (plus one grid step)."""
    c = np.asarray(coord, float)
    return (vals >= c.min() - tol) & (vals <= c.max() + tol)

acc = np.full((len(pid), 4), np.nan)
t0 = time.time()

ws = xr.open_dataset(wf[0]); wt = "valid_time" if "valid_time" in ws.coords else "time"
plon_w = np.where(plon > 180, plon - 360, plon)
w_ok = in_range(ws.latitude.values, plat, 0.5) & in_range(ws.longitude.values, plon_w, 0.5)
wlat_i = nearest_idx(ws.latitude.values, plat)
wlon_i = nearest_idx(ws.longitude.values, plon_w)
wtimes = pd.DatetimeIndex(ws[wt].values)
want_w = nearest_idx(wtimes.asi8, (days + pd.Timedelta(hours=12)).asi8)
for k in np.unique(want_w):
    m = (want_w == k) & w_ok
    if not m.any():
        continue
    u = ws.u10.isel({wt: int(k)}).values; v = ws.v10.isel({wt: int(k)}).values
    acc[m, 0] = u[wlat_i[m], wlon_i[m]]; acc[m, 1] = v[wlat_i[m], wlon_i[m]]
ws.close()
log(f"wind done {time.time()-t0:.0f}s (in-coverage {int(w_ok.sum())}/{len(w_ok)})")

cs_ = xr.open_dataset(cf[0])
plon_c = np.where(plon > 180, plon - 360, plon)
c_ok = in_range(cs_.latitude.values, plat, 0.2) & in_range(cs_.longitude.values, plon_c, 0.2)
clat_i = nearest_idx(cs_.latitude.values, plat)
clon_i = nearest_idx(cs_.longitude.values, plon_c)
ctimes = pd.DatetimeIndex(cs_.time.values)
want_c = nearest_idx(ctimes.asi8, days.asi8)
for k in np.unique(want_c):
    m = (want_c == k) & c_ok
    if not m.any():
        continue
    uo = cs_.uo.isel(time=int(k), depth=0).values; vo = cs_.vo.isel(time=int(k), depth=0).values
    acc[m, 2] = uo[clat_i[m], clon_i[m]]; acc[m, 3] = vo[clat_i[m], clon_i[m]]
cs_.close()
log(f"currents done {time.time()-t0:.0f}s (in-coverage {int(c_ok.sum())}/{len(c_ok)})")

smp = pd.DataFrame(acc, columns=["u_wind", "v_wind", "u_curr", "v_curr"]).assign(pid=pid)
agg = smp.groupby("pid").mean()
for c in ["u_wind", "v_wind", "u_curr", "v_curr"]:
    P[c] = np.nan
P.loc[agg.index, ["u_wind", "v_wind", "u_curr", "v_curr"]] = agg.values
P.to_csv(PART, index=False)
log(f"WROTE {len(P)} rows ({P.u_curr.notna().sum()} with forcing) in {time.time()-t0:.0f}s")
