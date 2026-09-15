"""MLP on the corrected drift table. Scaling stats from TRAIN split only."""
import json, sys, time
import numpy as np, pandas as pd, torch, torch.nn as nn

ROOT = "/Users/pratiksmac/Downloads/icccy/iceberg-drift"
CSV, TAG, EPOCHS = sys.argv[1], sys.argv[2], int(sys.argv[3])
torch.manual_seed(0); np.random.seed(0)
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

df = pd.read_csv(CSV, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
ts = df.date.to_numpy().astype("datetime64[s]").astype(np.int64)
q70, q85 = np.quantile(ts, [0.70, 0.85])
tr, va, te = df[ts <= q70], df[(ts > q70) & (ts <= q85)], df[ts > q85]

def F(x):
    return np.column_stack([x.u_wind, x.v_wind, x.u_curr, x.v_curr, x.lat, x.lon,
                            x.size_1, x.size_2, np.sin(2*np.pi*x.date.dt.month/12),
                            np.cos(2*np.pi*x.date.dt.month/12), np.sin(np.radians(x.lat)),
                            np.hypot(x.u_wind, x.v_wind), np.hypot(x.u_curr, x.v_curr),
                            x.prev_u, x.prev_v, x.gap_days]).astype(np.float32)

Xtr, Xva, Xte = F(tr), F(va), F(te)
ytr = tr[["u_berg","v_berg"]].to_numpy(np.float32)
yva = va[["u_berg","v_berg"]].to_numpy(np.float32)
yte = te[["u_berg","v_berg"]].to_numpy(np.float32)

mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8          # TRAIN ONLY
ymu, ysd = ytr.mean(0), ytr.std(0) + 1e-8        # TRAIN ONLY
Xtr, Xva, Xte = (Xtr-mu)/sd, (Xva-mu)/sd, (Xte-mu)/sd
log(f"{TAG}: train {len(Xtr)} val {len(Xva)} test {len(Xte)} | {Xtr.shape[1]} features | {EPOCHS} epochs")

net = nn.Sequential(nn.Linear(Xtr.shape[1],128), nn.ReLU(), nn.Dropout(0.1),
                    nn.Linear(128,64), nn.ReLU(), nn.Dropout(0.1),
                    nn.Linear(64,2))
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
Xt, yt = torch.tensor(Xtr), torch.tensor((ytr-ymu)/ysd)
Xv, Xs = torch.tensor(Xva), torch.tensor(Xte)

hist, best, best_state = [], np.inf, None
for ep in range(1, EPOCHS+1):
    net.train(); perm = torch.randperm(len(Xt)); tot = 0.0
    for i in range(0, len(Xt), 256):
        j = perm[i:i+256]
        opt.zero_grad(); loss = nn.functional.mse_loss(net(Xt[j]), yt[j])
        loss.backward(); opt.step(); tot += loss.item()*len(j)
    sched.step()
    net.eval()
    with torch.no_grad():
        pv = net(Xv).numpy()*ysd + ymu
    e = pv - yva
    vkm = float(np.sqrt((np.hypot(e[:,0]*86400, e[:,1]*86400)**2).mean())/1000)
    vrmse = float(np.sqrt((e**2).mean()))
    hist.append(dict(epoch=ep, train_loss=tot/len(Xt), val_rmse=vrmse, val_km_24h=vkm))
    log(f"epoch {ep:>3}/{EPOCHS}  train_loss {tot/len(Xt):.5f}  val_RMSE {vrmse:.5f}  val_24h_km {vkm:.3f}")
    if vkm < best:
        best, best_state = vkm, {k: v.clone() for k, v in net.state_dict().items()}

net.load_state_dict(best_state); net.eval()
with torch.no_grad():
    pt = net(Xs).numpy()*ysd + ymu
e = pt - yte
k = np.hypot(e[:,0]*86400, e[:,1]*86400)/1000
res = dict(RMSE_u=float(np.sqrt((e[:,0]**2).mean())), RMSE_v=float(np.sqrt((e[:,1]**2).mean())),
           MAE_u=float(abs(e[:,0]).mean()), MAE_v=float(abs(e[:,1]).mean()),
           pos_err_24h_km_mean=float(k.mean()), pos_err_24h_km_rms=float(np.sqrt((k**2).mean())),
           best_val_24h_km=best, history=hist)
torch.save({"state_dict": net.state_dict(), "mu": mu, "sd": sd, "ymu": ymu, "ysd": ysd}, f"{ROOT}/models/drift_mlp_{TAG}.pt")
json.dump(res, open(f"{ROOT}/output/drift_mlp_metrics_{TAG}.json","w"), indent=2)
log(f"TEST  RMSE_u {res['RMSE_u']:.4f}  RMSE_v {res['RMSE_v']:.4f}  24h_km {res['pos_err_24h_km_rms']:.2f}")
