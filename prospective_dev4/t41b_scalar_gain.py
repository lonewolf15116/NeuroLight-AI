"""PHASE-4 DEVELOPMENT 4.1b: one-parameter input-gain adaptation.
Prediction = (non-action part of prior) + g * (action part of prior), action part = candidate u and 4 lagged actions.
g is estimated online by scalar RLS (lambda 0.98, P0 = 1e3); everything else stays at the OLS prior.
If this single scalar recovers most of the full-RLS benefit, the advantage is multiplicative input-gain re-estimation.
Same plants (5200-5249) and noise seeds as t41_arx_mechanism.py."""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'prospective_dev'))
from vsim import VCircuit, TARGETS, CAND, rmse, arx_z
p = np.load(os.path.join(os.path.dirname(HERE), 'prospective_dev', 'arx10_prior.npz')); w0 = p['w0']
S = [.5, .6, .7, .8, .9, 1., 1.1, 1.2, 1.4]; DEV = list(range(5200, 5250)); LAM = .98; P0 = 1e3
ACT = [0, 5, 6, 7, 8]; mask = np.zeros(10); mask[ACT] = 1; wa = w0 * mask; wr = w0 * (1 - mask)
res = {}
for j, s in enumerate(S):
    B = len(DEV); c = VCircuit(DEV, s, nseed=8000 + j); rh = np.zeros((B, 4)); ah = np.zeros((B, 4)); R = []; G = []
    g = np.ones(B); P = np.full(B, P0)
    for t in TARGETS:
        Z = arx_z(rh, ah, np.broadcast_to(CAND, (B, 101)))
        pred = (Z @ wr + g[:, None] * (Z @ wa)) * 60
        cost = (pred - t) ** 2 + 2 * CAND ** 2 + 2 * (CAND - ah[:, -1:]) ** 2
        i = np.argmin(cost, 1); u = CAND[i]; z = Z[np.arange(B), i]; y = c.step(u)
        a = z @ wa; e = y / 60. - (z @ wr + g * a)
        k = P * a / (LAM + a * P * a); g = g + k * e; P = (P - k * a * P) / LAM
        R.append(y); G.append(g.copy())
        rh = np.concatenate([rh[:, 1:], y[:, None]], 1); ah = np.concatenate([ah[:, 1:], u[:, None]], 1)
    r = rmse(np.array(R).T); gm = np.array(G)[-60:].mean(0)
    res[f'{s:.2f}'] = {'rmse': float(r.mean()), 'rmse_plants': r.tolist(), 'g_mean': float(gm.mean()), 'g_sd': float(gm.std())}
t41 = json.load(open(os.path.join(HERE, 't41_results.json')))
print('s     fixed   full    scalar-g  g_mean')
for s in S:
    k = f'{s:.2f}'; print(f"{s:4.2f}  {t41['fixed'][k]['rmse']:.3f}  {t41['full'][k]['rmse']:.3f}  {res[k]['rmse']:.3f}    {res[k]['g_mean']:.3f}")
gm = [res[f'{s:.2f}']['g_mean'] for s in S]; print('corr(g, s) =', round(float(np.corrcoef(gm, S)[0, 1]), 3))
fx = np.mean([t41['fixed'][f'{s:.2f}']['rmse'] for s in S]); fu = np.mean([t41['full'][f'{s:.2f}']['rmse'] for s in S]); sg = np.mean([res[f'{s:.2f}']['rmse'] for s in S])
print(f'mean RMSE fixed {fx:.3f} full {fu:.3f} scalar-g {sg:.3f}  -> fraction of full-RLS benefit recovered: {(fx-sg)/(fx-fu):.2f}')
json.dump(res, open(os.path.join(HERE, 't41b_results.json'), 'w'))
