"""NeuroLight-AI: seeded synthetic optogenetic-like system identification/control."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Circuit:
    """Heterogeneous LIF neurons; phenomenological light gate, not ChR2 kinetics."""
    def __init__(self, seed=0, noise=0.5, n=64):
        self.rng = np.random.default_rng(seed)
        self.v = self.rng.uniform(-65, -55, n)
        self.bias = self.rng.uniform(7, 12, n)
        self.sensitivity = self.rng.uniform(18, 26, n)
        self.gate = 0.
        self.rate = 0.
        self.noise = noise
        self.refractory = np.zeros(n)

    def step(self, light):
        """Apply constant [0,1] intensity for 50 ms; return filtered Hz/neuron."""
        if not 0 <= light <= 1:
            raise ValueError('light intensity must be in [0,1]')
        spikes = 0
        for _ in range(50):
            self.gate += (light-self.gate)/10.
            active = self.refractory <= 0
            self.v[active] += ((-65-self.v[active]) + self.bias[active]
                              + self.sensitivity[active]*self.gate)/20.
            self.v[active] += self.rng.normal(0, self.noise, active.sum())
            fired = active & (self.v >= -50)
            spikes += fired.sum()
            self.v[fired] = -65
            self.refractory -= 1
            self.refractory[fired] = 2
        raw = spikes/len(self.v)*20
        self.rate = .5*self.rate + .5*raw
        return self.rate


def dataset(seeds, steps=240):
    xs, ys = [], []
    for seed in seeds:
        rng = np.random.default_rng(seed+10000)
        c = Circuit(seed)
        previous = 0.
        for t in range(steps):
            if t % 4 == 0:
                light = float(rng.uniform())
            x = [c.rate/60, previous, light]
            y = c.step(light)/60
            xs.append(x); ys.append([y]); previous = light
    return np.array(xs), np.array(ys)


class Predictor:
    def __init__(self, seed=7):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, .3, (3, 32)); self.b1 = np.zeros(32)
        self.w2 = rng.normal(0, .2, (32, 1)); self.b2 = np.zeros(1)

    def predict(self, x):
        return np.tanh(x@self.w1+self.b1)@self.w2+self.b2

    def fit(self, x, y, epochs=700):
        # Full batch Adam with explicit backprop; NumPy keeps CPU setup small.
        params = [self.w1, self.b1, self.w2, self.b2]
        m = [np.zeros_like(p) for p in params]; v = [a.copy() for a in m]
        loss = []
        for t in range(1, epochs+1):
            h = np.tanh(x@self.w1+self.b1)
            err = h@self.w2+self.b2-y
            d = 2*err/len(x)
            dh = (d@self.w2.T)*(1-h*h)
            grads = [x.T@dh, dh.sum(0), h.T@d, d.sum(0)]
            for i, (p, g) in enumerate(zip(params, grads)):
                m[i] = .9*m[i]+.1*g; v[i] = .999*v[i]+.001*g*g
                p -= .008*(m[i]/(1-.9**t))/(np.sqrt(v[i]/(1-.999**t))+1e-8)
            loss.append(float(np.mean(err*err))*3600)
        return loss


def control(model, seed, noise, mode):
    c = Circuit(seed, noise)
    targets = np.repeat([12., 30., 20., 40., 15., 35.], 30)
    previous = 0.; integral = 0.; rates = []; lights = []
    for target in targets:
        if mode == 'learned':
            candidates = np.linspace(0, 1, 101)
            x = np.column_stack([np.full(101, c.rate/60), np.full(101, previous), candidates])
            predicted = model.predict(x).ravel()*60
            # One-step predictive control: tracking + intensity + slew penalty.
            costs = (predicted-target)**2 + 2*candidates**2 + 2*(candidates-previous)**2
            light = float(candidates[np.argmin(costs)])
        elif mode == 'pi':
            error = target-c.rate
            integral = float(np.clip(integral+error*.05, -10, 10))
            light = float(np.clip(.02*error+.08*integral, 0, 1))
        else:
            light = float(np.clip(target/60, 0, 1))
        rates.append(c.step(light)); lights.append(light); previous = light
    rates = np.array(rates); lights = np.array(lights)
    return targets, rates, lights, dict(rmse_hz=float(np.sqrt(np.mean((rates-targets)**2))),
                                       mean_intensity=float(lights.mean()))


def run(out):
    out.mkdir(parents=True, exist_ok=True)
    x, y = dataset(range(20)); xv, yv = dataset(range(20, 25)); xt, yt = dataset(range(25, 30))
    model = Predictor(); losses = model.fit(x, y)
    np.savez_compressed(out/'dataset.npz', train_x=x, train_y=y, val_x=xv, val_y=yv, test_x=xt, test_y=yt)
    np.savez(out/'predictor.npz', w1=model.w1, b1=model.b1, w2=model.w2, b2=model.b2)
    # Linear regression and persistence are independent predictive baselines.
    beta = np.linalg.lstsq(np.column_stack([x, np.ones(len(x))]), y, rcond=None)[0]
    metrics = {'prediction': {}, 'control': {}}
    for name, pred in [('mlp', model.predict(xt)), ('linear', np.column_stack([xt, np.ones(len(xt))])@beta),
                       ('persistence', xt[:, :1])]:
        metrics['prediction'][name] = {'test_rmse_hz': float(np.sqrt(np.mean((pred-yt)**2))*60)}
    metrics['prediction']['mlp']['validation_rmse_hz'] = float(np.sqrt(np.mean((model.predict(xv)-yv)**2))*60)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    for noise in [.5, 1.]:
        key = f'noise_{noise}'
        metrics['control'][key] = {}
        for mode in ['open_loop', 'pi', 'learned']:
            runs = [control(model, s, noise, mode) for s in range(100, 110)]
            metrics['control'][key][mode] = {k: float(np.mean([r[3][k] for r in runs])) for k in runs[0][3]}
            metrics['control'][key][mode]['rmse_std_hz'] = float(np.std([r[3]['rmse_hz'] for r in runs]))
            if noise == .5:
                target, rate, light, _ = runs[0]
                axes[0].plot(np.arange(len(rate))*.05, rate, label=mode)
                axes[1].plot(np.arange(len(light))*.05, light, label=mode)
    axes[0].plot(np.arange(len(target))*.05, target, 'k--', label='target')
    axes[0].set(ylabel='Filtered firing rate (Hz)'); axes[0].legend()
    axes[1].set(ylabel='Light intensity', xlabel='Time (s)'); axes[1].legend()
    axes[2].plot(losses); axes[2].set(xlabel='Training epoch', ylabel='Training MSE (Hz²)')
    fig.tight_layout(); fig.savefig(out/'results.png', dpi=160); plt.close(fig)
    (out/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, default=Path('results'))
    run(parser.parse_args().out)
