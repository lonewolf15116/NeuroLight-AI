"""PHASE-4 DEVELOPMENT (4.2): retrain DR-H4 (seeds 101-105) exactly as prospective_dev/retrain_chunked.py
(same data, init, Adam, lr) but save the RAW weights at fixed epochs, so closed-loop behaviour can be traced
against training length. Resumable: each invocation runs for at most --budget seconds.
No evaluation plants are touched here."""
import sys, os, json, time, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, experiment_a as A
SNAP = [300, 600, 900, 1500, 2000, 3000, 4000, 5000, 6000, 7500, 9000, 10000]
ap = argparse.ArgumentParser(); ap.add_argument('--budget', type=float, default=1e9); ap.add_argument('--seeds', default='101,102,103,104,105'); a = ap.parse_args(); T0 = time.time()
OUT = os.path.join(ROOT, 'prospective_dev4', 'snapshots'); os.makedirs(OUT, exist_ok=True)
z = np.load(os.path.join(ROOT, 'prospective_dev', 'retrain_data_cache.npz')); X, XV, y, yv = z['xh'], z['xhv'], z['y'], z['yv']
SEEDS = [int(x) for x in a.seeds.split(',')]
for s in SEEDS:
    key = f'DR_H4_{s}'
    logp = os.path.join(OUT, f'log_{key}.json'); log = json.load(open(logp)) if os.path.exists(logp) else {}
    if log.get(key, {}).get('done'): continue
    ck = os.path.join(OUT, f'ckpt_{key}.npz')
    if os.path.exists(ck):
        c = np.load(ck); ps = [c[f'p{i}'].copy() for i in range(4)]; mo = [c[f'm{i}'].copy() for i in range(4)]
        v = [c[f'v{i}'].copy() for i in range(4)]; t = int(c['t'])
    else:
        m = A.MLP(9, 64, s); ps = [m.w1, m.b1, m.w2, m.b2]; mo = [np.zeros_like(p) for p in ps]; v = [q.copy() for q in mo]; t = 0
    rec = log.setdefault(key, {'val_rmse_hz': {}, 'done': False})
    w1, b1, w2, b2 = ps
    while t < SNAP[-1] and time.time() - T0 < a.budget:
        t += 1
        hh = np.tanh(X @ w1 + b1); e = hh @ w2 + b2 - y; dd = 2 * e / len(X); dh = (dd @ w2.T) * (1 - hh * hh)
        gs = [X.T @ dh, dh.sum(0), hh.T @ dd, dd.sum(0)]
        for i, (p, g) in enumerate(zip(ps, gs)):
            mo[i] = .9 * mo[i] + .1 * g; v[i] = .999 * v[i] + .001 * g * g
            p -= A.LR * (mo[i] / (1 - .9 ** t)) / (np.sqrt(v[i] / (1 - .999 ** t)) + 1e-8)
        if t in SNAP:
            np.savez(os.path.join(OUT, f'{key}_e{t}.npz'), w1=w1, b1=b1, w2=w2, b2=b2)
            rec['val_rmse_hz'][str(t)] = float(np.sqrt(np.mean((np.tanh(XV @ w1 + b1) @ w2 + b2 - yv) ** 2)) * 60)
    np.savez(ck, **{f'p{i}': ps[i] for i in range(4)}, **{f'm{i}': mo[i] for i in range(4)}, **{f'v{i}': v[i] for i in range(4)}, t=t)
    if t >= SNAP[-1]: rec['done'] = True
    json.dump(log, open(logp, 'w'), indent=1)
    if t < SNAP[-1]: print('budget reached', key, 'epoch', t, flush=True); sys.exit(0)
    print('DONE', key, flush=True)
print('ALL DONE for seeds', SEEDS, flush=True)
