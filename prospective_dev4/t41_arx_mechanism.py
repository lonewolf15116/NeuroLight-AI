"""PHASE-4 DEVELOPMENT 4.1: why does adaptive ARX win?  Development plants 5200-5249, vectorised simulator.
Variants (same OLS prior, same one-step action rule):
  fixed      : no update
  full       : RLS on all 10 coefficients (v3 settings: P0 = 1e9*cov0, lambda 0.98)
  gain_only  : RLS on the candidate-action coefficient w_u only, other 9 fixed at the prior
  gain_icpt  : RLS on w_u and the intercept only
Records tracking RMSE and the adapted w_u (mean over the last 60 steps) against the hidden sensitivity s."""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'prospective_dev'))
from vsim import VCircuit, TARGETS, CAND, rmse, arx_z
p = np.load(os.path.join(os.path.dirname(HERE), 'prospective_dev', 'arx10_prior.npz')); w0, cov0 = p['w0'], p['cov0']
S = [.5, .6, .7, .8, .9, 1., 1.1, 1.2, 1.4]; DEV = list(range(5200, 5250)); LAM = .98; PS = 1e9
SETS = {'fixed': [], 'full': list(range(10)), 'gain_only': [0], 'gain_icpt': [0, 9]}
def run(seeds, s, idx, nseed):
    B = len(seeds); c = VCircuit(seeds, s, nseed=nseed); rh = np.zeros((B, 4)); ah = np.zeros((B, 4)); R = []; WU = []
    W = np.tile(w0, (B, 1)); k = len(idx)
    if k: P = np.tile(PS * (cov0[np.ix_(idx, idx)] + 1e-12 * np.eye(k)), (B, 1, 1))
    for t in TARGETS:
        Z = arx_z(rh, ah, np.broadcast_to(CAND, (B, 101))); pred = np.einsum('bkj,bj->bk', Z, W) * 60
        cost = (pred - t) ** 2 + 2 * CAND ** 2 + 2 * (CAND - ah[:, -1:]) ** 2
        i = np.argmin(cost, 1); u = CAND[i]; z = Z[np.arange(B), i]; y = c.step(u)
        if k:
            e = y / 60. - np.einsum('bj,bj->b', z, W); zi = z[:, idx]
            Pz = np.einsum('bij,bj->bi', P, zi); g = Pz / (LAM + np.einsum('bj,bj->b', zi, Pz))[:, None]
            W[:, idx] += g * e[:, None]; P = (P - np.einsum('bi,bj,bjk->bik', g, zi, P)) / LAM; P = .5 * (P + P.transpose(0, 2, 1))
        R.append(y); WU.append(W[:, 0].copy())
        rh = np.concatenate([rh[:, 1:], y[:, None]], 1); ah = np.concatenate([ah[:, 1:], u[:, None]], 1)
    return rmse(np.array(R).T), np.array(WU)[-60:].mean(0)
out = {}
for name, idx in SETS.items():
    out[name] = {}
    for j, s in enumerate(S):
        r, wu = run(DEV, s, idx, nseed=8000 + j)   # same noise stream for every variant at a given s
        out[name][f'{s:.2f}'] = {'rmse': float(r.mean()), 'rmse_plants': r.tolist(), 'wu_mean': float(wu.mean()), 'wu_sd': float(wu.std())}
    print(name, [round(out[name][f'{s:.2f}']['rmse'], 3) for s in S], flush=True)
for name in ('full', 'gain_only', 'gain_icpt'):
    wu = [out[name][f'{s:.2f}']['wu_mean'] for s in S]
    print(name, 'w_u by s:', [round(x, 3) for x in wu], '| corr(w_u, s) =', round(float(np.corrcoef(wu, S)[0, 1]), 3), '| prior w_u =', round(float(w0[0]), 3))
json.dump(out, open(os.path.join(HERE, 't41_results.json'), 'w'))
