"""PHASE-4 DEVELOPMENT: train intervention models (I2: 10-input gain-feature network; I3: closed-loop-augmented DR-H4).
Same initialisation, Adam settings and learning rate as experiment_a.MLP.fit; 6000-epoch budget; raw snapshots at
900/1500/3000/6000 plus the best-validation checkpoint. Resumable per --budget seconds.
Usage: python train_intervention.py --tag i2|i3 --seeds 101,102 --budget 160"""
import sys, os, json, time, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, experiment_a as A
SNAP = [900, 1500, 3000, 6000]
ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--seeds', default='101,102,103,104,105')
ap.add_argument('--budget', type=float, default=1e9); a = ap.parse_args(); T0 = time.time()
OUT = os.path.join(ROOT, 'prospective_dev4', f'models_{a.tag}'); os.makedirs(OUT, exist_ok=True)
for s in [int(x) for x in a.seeds.split(',')]:
    key = f'{a.tag}_{s}'; logp = os.path.join(OUT, f'log_{key}.json'); log = json.load(open(logp)) if os.path.exists(logp) else {'val': {}}
    if log.get('done'): continue
    z = np.load(os.path.join(ROOT, 'prospective_dev4', 'data', 'i2.npz' if a.tag == 'i2' else f'i3_seed{s}.npz'))
    X, y, XV, yv = z['x'], z['y'], z['xv'], z['yv']; d = X.shape[1]
    ck = os.path.join(OUT, f'ckpt_{key}.npz')
    if os.path.exists(ck):
        c = np.load(ck); ps = [c[f'p{i}'].copy() for i in range(4)]; mo = [c[f'm{i}'].copy() for i in range(4)]
        v = [c[f'v{i}'].copy() for i in range(4)]; bp = [c[f'b{i}'].copy() for i in range(4)]; t = int(c['t']); bva = float(c['bva']); bt = int(c['bt'])
    else:
        m = A.MLP(d, 64, s); ps = [m.w1, m.b1, m.w2, m.b2]; mo = [np.zeros_like(p) for p in ps]; v = [q.copy() for q in mo]
        bp = [p.copy() for p in ps]; t = 0; bva = float('inf'); bt = 0
    w1, b1, w2, b2 = ps
    while t < SNAP[-1] and time.time() - T0 < a.budget:
        t += 1
        hh = np.tanh(X @ w1 + b1); e = hh @ w2 + b2 - y; dd = 2 * e / len(X); dh = (dd @ w2.T) * (1 - hh * hh)
        gs = [X.T @ dh, dh.sum(0), hh.T @ dd, dd.sum(0)]
        for i, (p, g) in enumerate(zip(ps, gs)):
            mo[i] = .9 * mo[i] + .1 * g; v[i] = .999 * v[i] + .001 * g * g
            p -= A.LR * (mo[i] / (1 - .9 ** t)) / (np.sqrt(v[i] / (1 - .999 ** t)) + 1e-8)
        va = float(np.mean((np.tanh(XV @ w1 + b1) @ w2 + b2 - yv) ** 2))
        if va < bva: bva, bt, bp = va, t, [p.copy() for p in ps]
        if t in SNAP:
            np.savez(os.path.join(OUT, f'{key}_e{t}.npz'), w1=w1, b1=b1, w2=w2, b2=b2); log['val'][str(t)] = float(np.sqrt(va) * 60)
    np.savez(ck, **{f'p{i}': ps[i] for i in range(4)}, **{f'm{i}': mo[i] for i in range(4)}, **{f'v{i}': v[i] for i in range(4)},
             **{f'b{i}': bp[i] for i in range(4)}, t=t, bva=bva, bt=bt)
    if t >= SNAP[-1]:
        np.savez(os.path.join(OUT, f'{key}_best.npz'), w1=bp[0], b1=bp[1], w2=bp[2], b2=bp[3])
        log.update(done=True, best_epoch=bt, best_val=float(np.sqrt(bva) * 60))
    json.dump(log, open(logp, 'w'), indent=1)
    if t < SNAP[-1]: print('budget reached', key, t, flush=True); sys.exit(0)
    print('DONE', key, 'best', bt, flush=True)
print('ALL DONE', a.tag, a.seeds, flush=True)
