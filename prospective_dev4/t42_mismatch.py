"""PHASE-4 DEVELOPMENT 4.2: why does longer training worsen closed-loop control?
For each DR-H4 snapshot (seeds 101-105 x 12 epochs), on development plants 5200-5249 (vectorised simulator):
  - validation RMSE (open-loop data, from the training log)
  - closed-loop tracking RMSE at s in {0.6, 0.8, 1.0, 1.2, 1.4}
  - on-policy one-step error: |prediction for the chosen action - observed next rate| along the model's own trajectory
  - action sensitivity d(yhat)/du at the chosen action (Hz per unit light), compared with the plant's effective gain
    estimated by the one-parameter gain RLS of t41b (60 * g * w_u_prior), used as a reference only."""
import os, sys, json, glob, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'prospective_dev'))
from vsim import VCircuit, TARGETS, CAND, rmse
SV = [.6, .8, 1., 1.2, 1.4]; DEV = list(range(5200, 5250)); EPOCHS = [300, 600, 900, 1500, 2000, 3000, 4000, 5000, 6000, 7500, 9000, 10000]
f = lambda m, X: (np.tanh(X @ m['w1'] + m['b1']) @ m['w2'] + m['b2']).reshape(X.shape[:-1])
def run(m, s, nseed):
    B = len(DEV); c = VCircuit(DEV, s, nseed=nseed); rh = np.zeros((B, 4)); ah = np.zeros((B, 4)); R = []; err = []; sens = []
    for t in TARGETS:
        base = np.concatenate([rh / 60., ah], 1)
        X = np.concatenate([np.repeat(base[:, None, :], 101, 1), np.broadcast_to(CAND, (B, 101))[..., None]], 2)
        pred = f(m, X) * 60; cost = (pred - t) ** 2 + 2 * CAND ** 2 + 2 * (CAND - ah[:, -1:]) ** 2
        i = np.argmin(cost, 1); u = CAND[i]; y = c.step(u)
        err.append(np.abs(pred[np.arange(B), i] - y))
        lo = np.clip(i - 2, 0, 100); hi = np.clip(i + 2, 0, 100)
        sens.append((pred[np.arange(B), hi] - pred[np.arange(B), lo]) / (CAND[hi] - CAND[lo]))
        R.append(y); rh = np.concatenate([rh[:, 1:], y[:, None]], 1); ah = np.concatenate([ah[:, 1:], u[:, None]], 1)
    return float(rmse(np.array(R).T).mean()), float(np.mean(err)), float(np.mean(sens))
logs = {}
for p in glob.glob(os.path.join(HERE, 'snapshots', 'log_DR_H4_*.json')): logs.update(json.load(open(p)))
out = {}
for sd in range(101, 106):
    for e in EPOCHS:
        m = dict(np.load(os.path.join(HERE, 'snapshots', f'DR_H4_{sd}_e{e}.npz')))
        rec = {'val_rmse_hz': logs[f'DR_H4_{sd}']['val_rmse_hz'][str(e)]}
        for j, s in enumerate(SV):
            r, er, se = run(m, s, nseed=9000 + j); rec[f'{s:.2f}'] = {'track_rmse': r, 'onpolicy_err_hz': er, 'dyhat_du': se}
        out[f'{sd}_{e}'] = rec
    print('seed', sd, 'done', flush=True)
    json.dump(out, open(os.path.join(HERE, 't42_results.json'), 'w'))
print('DONE')
