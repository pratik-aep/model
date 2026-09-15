"""Stage B final: baselines + linear + GBM (with causal lag features) on the
corrected real-observation drift table."""
import json, os, sys, time
import numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = "/Users/pratiksmac/Downloads/icccy/iceberg-drift"
CSV, TAG = sys.argv[1], sys.argv[2]
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

df = pd.read_csv(CSV, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
ts = df.date.to_numpy().astype("datetime64[s]").astype(np.int64)
q70, q85 = np.quantile(ts, [0.70, 0.85])
tr, va, te = df[ts <= q70], df[(ts > q70) & (ts <= q85)], df[ts > q85]
log(f"{TAG}: {len(df)} rows, {df.berg_id.nunique()} bergs | train {len(tr)} val {len(va)} test {len(te)}")

PHYS = ["u_wind", "v_wind", "u_curr", "v_curr"]
def F(x):
    return np.column_stack([x[PHYS].to_numpy(float), x.lat, x.lon, x.size_1, x.size_2,
                            x.date.dt.month, np.sin(np.radians(x.lat)),
                            np.hypot(x.u_wind, x.v_wind), np.hypot(x.u_curr, x.v_curr),
                            x.prev_u, x.prev_v, x.gap_days])
yte = te[["u_berg", "v_berg"]].to_numpy(float)
def score(pu, pv):
    eu, ev = pu - yte[:, 0], pv - yte[:, 1]
    k = np.hypot(eu*86400, ev*86400)/1000
    return dict(RMSE_u=float(np.sqrt((eu**2).mean())), RMSE_v=float(np.sqrt((ev**2).mean())),
                MAE_u=float(abs(eu).mean()), MAE_v=float(abs(ev).mean()),
                pos_err_24h_km_mean=float(k.mean()), pos_err_24h_km_rms=float(np.sqrt((k**2).mean())))

R = {}
R["current_only"] = score(te.u_curr.to_numpy(), te.v_curr.to_numpy())
R["two_percent_rule"] = score(te.u_curr.to_numpy()+0.02*te.u_wind.to_numpy(),
                              te.v_curr.to_numpy()+0.02*te.v_wind.to_numpy())
R["persistence"] = score(te.prev_u.to_numpy(), te.prev_v.to_numpy())

lin = LinearRegression().fit(tr[PHYS].to_numpy(float), tr[["u_berg","v_berg"]].to_numpy(float))
pl = lin.predict(te[PHYS].to_numpy(float)); R["linear"] = score(pl[:,0], pl[:,1])
coef = {t: {n: float(c) for n, c in zip(PHYS, lin.coef_[k])} for k, t in enumerate(["u_berg","v_berg"])}
coef["intercept"] = {"u_berg": float(lin.intercept_[0]), "v_berg": float(lin.intercept_[1])}

Ftr, Fva, Fte = F(tr), F(va), F(te)
gbm, pg = {}, {}
for t in ["u_berg", "v_berg"]:
    ytr, yva = tr[t].to_numpy(float), va[t].to_numpy(float)
    best, bn = np.inf, 25
    for n in [25, 50, 100, 200, 400]:
        m = HistGradientBoostingRegressor(learning_rate=0.05, max_leaf_nodes=31, max_iter=n,
                                          early_stopping=False, random_state=0).fit(Ftr, ytr)
        r = float(np.sqrt(((m.predict(Fva)-yva)**2).mean()))
        if r < best - 1e-6: best, bn = r, n
    gbm[t] = HistGradientBoostingRegressor(learning_rate=0.05, max_leaf_nodes=31, max_iter=bn,
                                           early_stopping=False, random_state=0).fit(Ftr, ytr)
    pg[t] = gbm[t].predict(Fte)
    log(f"  gbm {t}: max_iter={bn} val_RMSE={best:.5f}")
R["gbm"] = score(pg["u_berg"], pg["v_berg"])

b = R["two_percent_rule"]
for v in R.values():
    for f in ["RMSE_u","RMSE_v","pos_err_24h_km_rms"]:
        v["skill_vs_2pct_"+f] = float(1 - v[f]/b[f])
R["linear_coefficients"] = coef
R["_meta"] = dict(csv=CSV, n_rows=len(df), n_bergs=int(df.berg_id.nunique()),
                  date_min=str(df.date.min().date()), date_max=str(df.date.max().date()),
                  n_train=len(tr), n_val=len(va), n_test=len(te),
                  test_start=str(te.date.min().date()),
                  zero_frac_test=float((np.hypot(te.u_berg,te.v_berg)==0).mean()),
                  mean_gap_days=float(df.gap_days.mean()))

joblib.dump({"gbm":gbm,"linear":lin,"features":PHYS+["lat","lon","size_1","size_2","month",
             "sin_lat","wind_spd","curr_spd","prev_u","prev_v","gap_days"]},
            f"{ROOT}/models/drift_gbm.pkl" if TAG=="free_drifting" else f"{ROOT}/models/drift_gbm_{TAG}.pkl")
json.dump(R, open(f"{ROOT}/output/drift_metrics_{TAG}.json","w"), indent=2)

hdr = f"{'model':<18}{'RMSE_u':>9}{'RMSE_v':>9}{'24h_km':>9}{'skill_2%':>10}"
rows = [hdr, "-"*len(hdr)]
for k in ["current_only","two_percent_rule","persistence","linear","gbm"]:
    v = R[k]
    rows.append(f"{k:<18}{v['RMSE_u']:>9.4f}{v['RMSE_v']:>9.4f}{v['pos_err_24h_km_rms']:>9.2f}"
                f"{v['skill_vs_2pct_pos_err_24h_km_rms']:>10.3f}")
log("\n"+"\n".join(rows))
log("linear coefficients: "+json.dumps(coef))

fig, ax = plt.subplots(1,2,figsize=(11,5))
for i,(t,lab) in enumerate([("u_berg","east"),("v_berg","north")]):
    a, p = te[t].to_numpy()*86.4, pg[t]*86.4
    ax[i].scatter(a,p,s=7,alpha=.3,edgecolors="none")
    L = np.nanpercentile(np.abs(np.r_[a,p]),99.5)
    ax[i].plot([-L,L],[-L,L],"r--",lw=1); ax[i].set_xlim(-L,L); ax[i].set_ylim(-L,L)
    ax[i].set_xlabel(f"actual 24h {lab} displacement (km)"); ax[i].set_ylabel("predicted (km)")
    ax[i].set_title(f"GBM {t} — RMSE {R['gbm']['RMSE_'+t[0]]:.4f} m/s")
plt.tight_layout()
plt.savefig(f"{ROOT}/output/drift_scatter.png" if TAG=="free_drifting" else f"{ROOT}/output/drift_scatter_{TAG}.png", dpi=130)
log("saved")
