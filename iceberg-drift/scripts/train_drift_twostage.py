"""Two-stage zero-inflated drift model.

Hyperparameters (lr=0.05, max_leaf_nodes=15) selected by 5-fold blocked
time-series CV -- see output/drift_cv_tuning.json. Shallower trees won in
5/5 folds; the signal is weak enough that extra capacity only overfits.

The target is 56% exact zeros (grounded/fast-ice bergs). One regressor must fit
both regimes at once. Instead:
  stage 1: P(berg moves at all)          -- classifier
  stage 2: E[velocity | berg moves]      -- regressor on moving rows only
  combined: E[velocity] = P(move) * E[v|move]
All features are causal (known before the forecast interval). Test never touched.
"""
import json, time
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

ROOT = "/Users/pratiksmac/Downloads/icccy/iceberg-drift"
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

d = pd.read_csv(f"{ROOT}/data/drift_trainable_rich.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
ts = d.date.to_numpy().astype("datetime64[s]").astype(np.int64)
q70, q85 = np.quantile(ts, [.70, .85])
tr, va, te = d[ts <= q70], d[(ts > q70) & (ts <= q85)], d[ts > q85]
log(f"train {len(tr)} val {len(va)} test {len(te)} | test from {te.date.min().date()}")

FEATS = ["u_wind","v_wind","u_curr","v_curr","lat","lon","size_1","size_2","gap_days",
         "prev_u","prev_v","u_lag2","v_lag2","u_lag3","v_lag3",
         "spd_lag1","spd_lag2","roll_u","roll_v","roll_spd","frac_moving",
         "n_obs_sofar","track_age_d"]
def F(x):
    m = x[FEATS].copy()
    m["month_s"] = np.sin(2*np.pi*x.date.dt.month/12)
    m["month_c"] = np.cos(2*np.pi*x.date.dt.month/12)
    m["wind_spd"] = np.hypot(x.u_wind, x.v_wind)
    m["curr_spd"] = np.hypot(x.u_curr, x.v_curr)
    return m.to_numpy(float)

Ftr, Fva, Fte = F(tr), F(va), F(te)
yte = te[["u_berg","v_berg"]].to_numpy(float)
def km_rms(pu, pv):
    k = np.hypot((pu-yte[:,0])*86400, (pv-yte[:,1])*86400)/1000
    return float(np.sqrt((k**2).mean())), float(np.median(k)), float((k<=10).mean()*100)

R = {}
mu_tr, mv_tr = tr.u_berg.mean(), tr.v_berg.mean()
R["constant"] = km_rms(np.full(len(te), mu_tr), np.full(len(te), mv_tr))

# ---------- single stage (reference) ----------
one = {}
for t in ["u_berg","v_berg"]:
    best, bn = np.inf, 50
    for n in [50,100,200,400]:
        m = HistGradientBoostingRegressor(learning_rate=.05, max_leaf_nodes=15, max_iter=n,
                                          early_stopping=False, random_state=0).fit(Ftr, tr[t])
        r = np.sqrt(((m.predict(Fva)-va[t])**2).mean())
        if r < best: best, bn = r, n
    one[t] = HistGradientBoostingRegressor(learning_rate=.05, max_leaf_nodes=15, max_iter=bn,
                                           early_stopping=False, random_state=0).fit(Ftr, tr[t])
R["single_stage_gbm"] = km_rms(one["u_berg"].predict(Fte), one["v_berg"].predict(Fte))

# ---------- two stage ----------
mov_tr = (np.hypot(tr.u_berg, tr.v_berg) > 0).astype(int)
mov_va = (np.hypot(va.u_berg, va.v_berg) > 0).astype(int)
best, bn = -1, 50
for n in [50,100,200,400]:
    c = HistGradientBoostingClassifier(learning_rate=.05, max_leaf_nodes=15, max_iter=n,
                                       early_stopping=False, random_state=0).fit(Ftr, mov_tr)
    from sklearn.metrics import roc_auc_score
    a = roc_auc_score(mov_va, c.predict_proba(Fva)[:,1])
    if a > best: best, bn = a, n
clf = HistGradientBoostingClassifier(learning_rate=.05, max_leaf_nodes=15, max_iter=bn,
                                     early_stopping=False, random_state=0).fit(Ftr, mov_tr)
log(f"stage1 classifier: max_iter={bn} val AUC={best:.4f}")

mtr = tr[np.hypot(tr.u_berg, tr.v_berg) > 0]
mva = va[np.hypot(va.u_berg, va.v_berg) > 0]
Fmtr, Fmva = F(mtr), F(mva)
two = {}
for t in ["u_berg","v_berg"]:
    b2, n2 = np.inf, 50
    for n in [50,100,200,400]:
        m = HistGradientBoostingRegressor(learning_rate=.05, max_leaf_nodes=15, max_iter=n,
                                          early_stopping=False, random_state=0).fit(Fmtr, mtr[t])
        r = np.sqrt(((m.predict(Fmva)-mva[t])**2).mean())
        if r < b2: b2, n2 = r, n
    two[t] = HistGradientBoostingRegressor(learning_rate=.05, max_leaf_nodes=15, max_iter=n2,
                                           early_stopping=False, random_state=0).fit(Fmtr, mtr[t])
    log(f"stage2 {t}: max_iter={n2} val_RMSE(moving)={b2:.5f}")

p_move = clf.predict_proba(Fte)[:,1]
R["two_stage"] = km_rms(p_move*two["u_berg"].predict(Fte), p_move*two["v_berg"].predict(Fte))

joblib.dump({"clf":clf,"reg":two,"single":one,"features":FEATS+["month_s","month_c","wind_spd","curr_spd"]},
            f"{ROOT}/models/drift_twostage.pkl")
out = {k:{"pos_err_24h_km_rms":v[0],"pos_err_24h_km_median":v[1],"within_10km_pct":v[2],
          "skill_vs_constant":1-v[0]/R["constant"][0]} for k,v in R.items()}
json.dump(out, open(f"{ROOT}/output/drift_twostage_metrics.json","w"), indent=2)

h=f"{'model':<20}{'24h RMS km':>12}{'median km':>11}{'<=10km %':>10}{'vs const':>10}"
print("\n"+h); print("-"*len(h))
for k,v in R.items():
    print(f"{k:<20}{v[0]:>12.3f}{v[1]:>11.3f}{v[2]:>10.1f}{1-v[0]/R['constant'][0]:>10.3f}")
