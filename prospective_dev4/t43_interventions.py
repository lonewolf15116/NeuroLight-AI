"""PHASE-4 DEVELOPMENT 4.2 interventions. Closed-loop evaluation, vectorised simulator, 9-point sweep.
Usage: python t43_interventions.py eval      -> plants 5200-5249: H4_900, H4_10k, I2 (gain feature), I3 (closed-loop data)
       python t43_interventions.py i1select  -> I1: per seed, choose the snapshot epoch minimising mean closed-loop RMSE
                                                on CONTROL-VALIDATION plants 5250-5299, then evaluate it on 5200-5249.
Noise seeds per sensitivity match t41/t41b so ARX numbers are comparable."""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, os.path.join(ROOT, 'prospective_dev'))
from vsim import VCircuit, TARGETS, CAND, rmse, arx_z
w0 = np.load(os.path.join(ROOT, 'prospective_dev', 'arx10_prior.npz'))['w0']; ACT = [0, 5, 6, 7, 8]; mask = np.zeros(10); mask[ACT] = 1; wa = w0 * mask; wr = w0 * (1 - mask)
S9 = [.5, .6, .7, .8, .9, 1., 1.1, 1.2, 1.4]; EVAL = list(range(5200, 5250)); CVAL = list(range(5250, 5300))
f = lambda m, X: (np.tanh(X @ m['w1'] + m['b1']) @ m['w2'] + m['b2']).reshape(X.shape[:-1])
def run(m, seeds, s, nseed, gain_feature=False):
    B = len(seeds); c = VCircuit(seeds, s, nseed=nseed); rh = np.zeros((B, 4)); ah = np.zeros((B, 4)); R = []; sens = []
    g = np.ones(B); P = np.full(B, 1e3)
    for t in TARGETS:
        base = np.concatenate([rh / 60., ah], 1)
        X = np.concatenate([np.repeat(base[:, None, :], 101, 1), np.broadcast_to(CAND, (B, 101))[..., None]], 2)
        if gain_feature: X = np.concatenate([X, np.repeat(g[:, None, None], 101, 1)], 2)
        pred = f(m, X) * 60; cost = (pred - t) ** 2 + 2 * CAND ** 2 + 2 * (CAND - ah[:, -1:]) ** 2
        i = np.argmin(cost, 1); u = CAND[i]; y = c.step(u)
        lo = np.clip(i - 2, 0, 100); hi = np.clip(i + 2, 0, 100); sens.append((pred[np.arange(B), hi] - pred[np.arange(B), lo]) / (CAND[hi] - CAND[lo]))
        if gain_feature:
            z = arx_z(rh, ah, u[:, None])[:, 0]; a = z @ wa; e = y / 60. - (z @ wr + g * a)
            k = P * a / (.98 + a * P * a); g = g + k * e; P = (P - k * a * P) / .98
        R.append(y); rh = np.concatenate([rh[:, 1:], y[:, None]], 1); ah = np.concatenate([ah[:, 1:], u[:, None]], 1)
    return rmse(np.array(R).T), float(np.mean(sens))
def sweep(paths, seeds, gf=False, svals=S9, base=8000):
    res = {}
    for j, s in enumerate(svals):
        js = S9.index(s) if s in S9 else j
        per = []; sen = []
        for p in paths:
            r, se = run(dict(np.load(p)), seeds, s, base + js, gf); per.append(r); sen.append(se)
        res[f'{s:.2f}'] = {'rmse': float(np.mean(per)), 'per_model': [float(x.mean()) for x in per], 'plant': np.mean(per, 0).tolist(), 'dyhat_du': float(np.mean(sen))}
    return res
SEEDS = range(101, 106)
if sys.argv[1] == 'eval':
    sets = {'H4_900': ([os.path.join(ROOT, f'archived_results/EXPERIMENT_A_FINAL_2026-10-07/DR_H4_seed_{s}.npz') for s in SEEDS], False),
            'H4_10k': ([os.path.join(ROOT, f'prospective_dev/retrain_10000/DR_H4_seed_{s}.npz') for s in SEEDS], False)}
    for tag, gf in (('i2', True), ('i3', False)):
        for e in ('best', 'e900', 'e1500', 'e3000', 'e6000'):
            sets[f'{tag}_{e}'] = ([os.path.join(HERE, f'models_{tag}', f'{tag}_{s}_{e}.npz') for s in SEEDS], gf)
    out = {}
    for name, (paths, gf) in sets.items():
        out[name] = sweep(paths, EVAL, gf); print(name, [round(out[name][f'{s:.2f}']['rmse'], 3) for s in S9], flush=True)
        json.dump(out, open(os.path.join(HERE, 't43_eval.json'), 'w'))
else:
    EP = [300, 600, 900, 1500, 2000, 3000, 4000, 5000, 6000, 7500, 9000]   # 10000 raw excluded: seed-103 spike
    SV = [.6, .8, 1., 1.2, 1.4]; choice = {}
    for sd in SEEDS:
        scores = {}
        for e in EP:
            p = os.path.join(HERE, 'snapshots', f'DR_H4_{sd}_e{e}.npz')
            scores[e] = float(np.mean([run(dict(np.load(p)), CVAL, s, 7000 + k)[0].mean() for k, s in enumerate(SV)]))
        choice[sd] = min(scores, key=scores.get); print('seed', sd, 'chosen epoch', choice[sd], {k: round(v, 3) for k, v in scores.items()}, flush=True)
    res = sweep([os.path.join(HERE, 'snapshots', f'DR_H4_{sd}_e{choice[sd]}.npz') for sd in SEEDS], EVAL)
    json.dump({'choice': choice, 'eval': res}, open(os.path.join(HERE, 't43_i1.json'), 'w'))
    print('I1', [round(res[f'{s:.2f}']['rmse'], 3) for s in S9])
