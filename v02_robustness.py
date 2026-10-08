"""NeuroLight-AI v0.2: frozen-model robustness evaluation.

Place beside experiment.py and the existing results/ directory.
Does NOT retrain or overwrite v0.1.
"""
from pathlib import Path
import argparse, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment import Circuit, Predictor


TARGETS = np.repeat([12., 30., 20., 40., 15., 35.], 30)


def load_model(path):
    m = Predictor()
    with np.load(path) as w:
        for name in ("w1", "b1", "w2", "b2"):
            setattr(m, name, w[name].copy())
    return m


class ShiftedCircuit(Circuit):
    def __init__(self, seed=0, noise=.5, sensitivity_scale=1.0,
                 bias_shift=0.0, n=64):
        super().__init__(seed=seed, noise=noise, n=n)
        self.sensitivity *= sensitivity_scale
        self.bias += bias_shift


def run_controller(model, seed, mode, noise=.5, sensitivity_scale=1.0,
                   bias_shift=0.0, delay_steps=0, measurement_noise=0.0,
                   kp=.02, ki=.08):
    c = ShiftedCircuit(seed, noise, sensitivity_scale, bias_shift)
    previous = 0.; integral = 0.
    rates, lights = [], []
    history = [0.0] * (delay_steps + 1)
    rng = np.random.default_rng(seed + 50000)

    for target in TARGETS:
        observed = history[0] if delay_steps else c.rate
        if measurement_noise:
            observed += rng.normal(0, measurement_noise)

        if mode == "learned":
            candidates = np.linspace(0, 1, 101)
            x = np.column_stack([
                np.full(101, observed/60),
                np.full(101, previous),
                candidates
            ])
            pred = model.predict(x).ravel()*60
            cost = (pred-target)**2 + 2*candidates**2 + 2*(candidates-previous)**2
            light = float(candidates[np.argmin(cost)])
        elif mode == "pi":
            error = target-observed
            integral = float(np.clip(integral + error*.05, -10, 10))
            light = float(np.clip(kp*error + ki*integral, 0, 1))
        elif mode == "open_loop":
            light = float(np.clip(target/60, 0, 1))
        else:
            raise ValueError(mode)

        actual = c.step(light)
        history.append(actual)
        history.pop(0)
        rates.append(actual); lights.append(light); previous = light

    rates = np.asarray(rates); lights = np.asarray(lights)
    return {
        "rmse_hz": float(np.sqrt(np.mean((rates-TARGETS)**2))),
        "mean_intensity": float(lights.mean())
    }


def tune_pi():
    # Development seeds only; evaluation seeds below are disjoint.
    seeds = range(60, 70)
    best = None
    for kp in np.linspace(.005, .05, 10):
        for ki in np.linspace(.01, .15, 10):
            rmses = [run_controller(None, s, "pi", kp=kp, ki=ki)["rmse_hz"]
                     for s in seeds]
            score = float(np.mean(rmses))
            if best is None or score < best[0]:
                best = (score, float(kp), float(ki))
    return best


def paired_summary(model, scenario, seeds, kp, ki):
    kwargs = scenario["kwargs"]
    per_seed = []
    for s in seeds:
        learned = run_controller(model, s, "learned", **kwargs)
        pi = run_controller(model, s, "pi", kp=kp, ki=ki, **kwargs)
        diff = pi["rmse_hz"] - learned["rmse_hz"]  # positive => learned wins
        per_seed.append((learned, pi, diff))

    diffs = np.array([x[2] for x in per_seed])
    # Seed-level bootstrap CI, deterministic.
    rng = np.random.default_rng(20261006)
    boots = np.array([
        rng.choice(diffs, len(diffs), replace=True).mean()
        for _ in range(5000)
    ])
    return {
        "learned_rmse_hz": float(np.mean([x[0]["rmse_hz"] for x in per_seed])),
        "pi_rmse_hz": float(np.mean([x[1]["rmse_hz"] for x in per_seed])),
        "paired_advantage_hz": float(diffs.mean()),
        "advantage_95ci_hz": [float(np.percentile(boots, 2.5)),
                              float(np.percentile(boots, 97.5))],
        "learned_win_fraction": float(np.mean(diffs > 0)),
        "learned_mean_intensity": float(np.mean([x[0]["mean_intensity"] for x in per_seed])),
        "pi_mean_intensity": float(np.mean([x[1]["mean_intensity"] for x in per_seed]))
    }


def main(out, predictor):
    out.mkdir(parents=True, exist_ok=True)
    model = load_model(predictor)
    _, kp, ki = tune_pi()

    scenarios = [
        {"name":"nominal", "kwargs":{}},
        {"name":"sensitivity_0.6x", "kwargs":{"sensitivity_scale":.6}},
        {"name":"sensitivity_0.8x", "kwargs":{"sensitivity_scale":.8}},
        {"name":"sensitivity_1.2x", "kwargs":{"sensitivity_scale":1.2}},
        {"name":"sensitivity_1.4x", "kwargs":{"sensitivity_scale":1.4}},
        {"name":"noise_1.0", "kwargs":{"noise":1.0}},
        {"name":"noise_2.0", "kwargs":{"noise":2.0}},
        {"name":"noise_4.0", "kwargs":{"noise":4.0}},
        {"name":"delay_50ms", "kwargs":{"delay_steps":1}},
        {"name":"delay_100ms", "kwargs":{"delay_steps":2}},
        {"name":"delay_250ms", "kwargs":{"delay_steps":5}},
        {"name":"sensor_noise_2hz", "kwargs":{"measurement_noise":2.0}},
        {"name":"sensor_noise_5hz", "kwargs":{"measurement_noise":5.0}},
        {"name":"bias_minus2", "kwargs":{"bias_shift":-2.0}},
        {"name":"bias_plus2", "kwargs":{"bias_shift":2.0}},
    ]

    seeds = range(200, 230)
    results = {
        "frozen_predictor": str(predictor),
        "evaluation_seeds": list(seeds),
        "tuned_pi": {"kp":kp, "ki":ki},
        "scenarios": {}
    }
    for sc in scenarios:
        print("Running", sc["name"], flush=True)
        results["scenarios"][sc["name"]] = paired_summary(
            model, sc, seeds, kp, ki)

    (out/"robustness_metrics.json").write_text(json.dumps(results, indent=2)+"\n")

    names = [s["name"] for s in scenarios]
    learned = [results["scenarios"][n]["learned_rmse_hz"] for n in names]
    pi = [results["scenarios"][n]["pi_rmse_hz"] for n in names]
    x = np.arange(len(names)); width=.38
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x-width/2, learned, width, label="Frozen learned controller")
    ax.bar(x+width/2, pi, width, label="Tuned PI")
    ax.set_ylabel("Tracking RMSE (Hz)")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=55, ha="right")
    ax.legend(); fig.tight_layout()
    fig.savefig(out/"robustness.png", dpi=170)
    plt.close(fig)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("results_v02"))
    p.add_argument("--predictor", type=Path, default=Path("results/predictor.npz"))
    a = p.parse_args()
    main(a.out, a.predictor)
