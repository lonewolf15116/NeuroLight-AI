"""PHASE-4 DEVELOPMENT: build training sets for interventions I2 (gain-estimate feature) and I3 (closed-loop data).
Training plants 2000-2099 / validation 2100-2129 with the exact Experiment-A randomisation.
I2: the 9 DR-H4 inputs + g, where g is the one-parameter input-gain RLS estimate (t41b) BEFORE observing the
    next rate; open-loop data generated with the exact scalar simulator (rows must equal the Experiment-A cache).
I3: per-model closed-loop trajectories of the five 10 000-epoch DR-H4 models on the same plants (random target
    levels 10-45 Hz held 30 steps, 300 steps), vectorised simulator; appended to the open-loop data for that seed."""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'prospective_dev'))
os.chdir(ROOT)
import experiment_a as A, experiment_b_final as E
from v02_robustness import ShiftedCircuit
from vsim import VCircuit, CAND
w0 = np.load('prospective_dev/arx10_prior.npz')['w0']; ACT = [0, 5, 6, 7, 8]; mask = np.zeros(10); mask[ACT] = 1; wa = w0 * mask; wr = w0 * (1 - mask)
LAM, P0, H = .98, 1e3, 4
def params(seed):
    r = np.random.default_rng(seed + 70000)
    return r, float(r.uniform(.5, 2)), float(r.uniform(.6, 1.4)), float(r.uniform(-2, 2))
def i2_rows(seeds):
    X, Y = [], []
    for sd in seeds:
        r, nz, s, b = params(sd); c = ShiftedCircuit(sd, noise=nz, sensitivity_scale=s, bias_shift=b)
        rh = [0.] * H; ah = [0.] * H; g = 1.; P = P0
        for t in range(300):
            if t % 4 == 0: light = float(r.uniform())
            X.append(np.r_[np.asarray(rh) / 60., np.asarray(ah), light, g])
            nxt = float(c.step(light)); Y.append([nxt / 60.])
            z = E.zvec(rh, ah, light); a = z @ wa; e = nxt / 60. - (z @ wr + g * a)
            k = P * a / (LAM + a * P * a); g += k * e; P = (P - k * a * P) / LAM
            rh = (rh + [nxt])[-H:]; ah = (ah + [light])[-H:]
    return np.asarray(X), np.asarray(Y)
class VC(VCircuit):
    def __init__(self, seeds, s, noise, bias, nseed):
        super().__init__(seeds, 1.0, noise=0.5, nseed=nseed)
        self.sens *= np.asarray(s).reshape(-1, 1); self.bias += np.asarray(bias).reshape(-1, 1); self.noise = np.asarray(noise).reshape(-1, 1)
def closed_loop_rows(m, seeds, nseed):
    P = [params(sd) for sd in seeds]; B = len(seeds)
    c = VC(seeds, [p[2] for p in P], [p[1] for p in P], [p[3] for p in P], nseed)
    rt = np.random.default_rng(nseed + 1); levels = rt.uniform(10, 45, (B, 10))
    rh = np.zeros((B, 4)); ah = np.zeros((B, 4)); X = []; Y = []
    f = lambda Q: (np.tanh(Q @ m['w1'] + m['b1']) @ m['w2'] + m['b2']).reshape(Q.shape[:-1])
    for t in range(300):
        tgt = levels[:, t // 30][:, None]
        base = np.concatenate([rh / 60., ah], 1)
        Q = np.concatenate([np.repeat(base[:, None, :], 101, 1), np.broadcast_to(CAND, (B, 101))[..., None]], 2)
        cost = (f(Q) * 60 - tgt) ** 2 + 2 * CAND ** 2 + 2 * (CAND - ah[:, -1:]) ** 2
        u = CAND[np.argmin(cost, 1)]; y = c.step(u)
        X.append(np.concatenate([base, u[:, None]], 1)); Y.append(y[:, None] / 60.)
        rh = np.concatenate([rh[:, 1:], y[:, None]], 1); ah = np.concatenate([ah[:, 1:], u[:, None]], 1)
    return np.concatenate(X), np.concatenate(Y)
if __name__ == '__main__':
    out = os.path.join(HERE, 'data'); os.makedirs(out, exist_ok=True)
    cache = np.load('prospective_dev/retrain_data_cache.npz')
    X2, Y2 = i2_rows(A.TRAIN); X2v, Y2v = i2_rows(A.VAL)
    assert np.array_equal(X2[:, :9], cache['xh']) and np.array_equal(Y2, cache['y']), 'I2 rows differ from Experiment-A cache'
    print('I2 rows match Experiment-A data; g range', X2[:, 9].min().round(2), X2[:, 9].max().round(2), flush=True)
    np.savez(os.path.join(out, 'i2.npz'), x=X2, y=Y2, xv=X2v, yv=Y2v)
    for sd in range(101, 106):
        m = dict(np.load(f'prospective_dev/retrain_10000/DR_H4_seed_{sd}.npz'))
        xc, yc = closed_loop_rows(m, list(A.TRAIN), 11000 + sd); xcv, ycv = closed_loop_rows(m, list(A.VAL), 12000 + sd)
        np.savez(os.path.join(out, f'i3_seed{sd}.npz'), x=np.concatenate([cache['xh'], xc]), y=np.concatenate([cache['y'], yc]),
                 xv=np.concatenate([cache['xhv'], xcv]), yv=np.concatenate([cache['yv'], ycv]))
        print('I3 seed', sd, 'rows', len(xc) + len(cache['xh']), flush=True)
