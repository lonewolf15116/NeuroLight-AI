"""NeuroLight-AI v0.6: mechanistic history ablations.

Development-only experiment.
Frozen v0.4 final seeds 1000-1049 are NEVER used.

Train:      1400-1499
Validation: 1500-1529
Diagnostic: 1600-1649

For H=4 (200 ms), test:
  full           ordered rate + action history
  joint_shuffle  same permutation on aligned rate/action pairs
  reverse        reverse aligned history
  rate_only      remove action history
  action_only    remove rate history
  rate_shuffle   shuffle rates only
  action_shuffle shuffle actions only

Also computes paired bootstrap CIs for H4 versus tuned PI across
sensitivity 0.50 to 1.50.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from v02_robustness import ShiftedCircuit, run_controller, tune_pi

H = 4
SCALE = 60.0


class MLP:
    def __init__(self, nin=2 * H + 1, hidden=64, seed=61):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, .2, (nin, hidden))
        self.b1 = np.zeros(hidden)
        self.w2 = rng.normal(0, .15, (hidden, 1))
        self.b2 = np.zeros(1)

    def predict(self, x):
        return np.tanh(x @ self.w1 + self.b1) @ self.w2 + self.b2

    def fit(self, x, y, xv, yv, epochs=900, patience=80):
        params = [self.w1, self.b1, self.w2, self.b2]
        m = [np.zeros_like(p) for p in params]
        v = [z.copy() for z in m]
        best = None
        wait = 0

        for t in range(1, epochs + 1):
            h = np.tanh(x @ self.w1 + self.b1)
            err = h @ self.w2 + self.b2 - y
            d = 2 * err / len(x)
            dh = (d @ self.w2.T) * (1 - h * h)
            grads = [x.T @ dh, dh.sum(0), h.T @ d, d.sum(0)]

            for i, (p, g) in enumerate(zip(params, grads)):
                m[i] = .9 * m[i] + .1 * g
                v[i] = .999 * v[i] + .001 * g * g
                p -= .006 * (m[i] / (1 - .9 ** t)) / (
                    np.sqrt(v[i] / (1 - .999 ** t)) + 1e-8
                )

            val = float(np.mean((self.predict(xv) - yv) ** 2))
            if best is None or val < best[0] - 1e-9:
                best = (val, [p.copy() for p in params], t)
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        for p, q in zip(params, best[1]):
            p[...] = q
        return best[2]


def dataset(seeds, steps=300):
    xs, ys = [], []
    for seed in seeds:
        rng = np.random.default_rng(seed + 70000)
        c = ShiftedCircuit(
            seed,
            noise=float(rng.uniform(.5, 2)),
            sensitivity_scale=float(rng.uniform(.6, 1.4)),
            bias_shift=float(rng.uniform(-2, 2)),
        )
        rate_hist = [0.] * H
        action_hist = [0.] * H

        for t in range(steps):
            if t % 4 == 0:
                light = float(rng.uniform())

            xs.append(np.r_[
                np.asarray(rate_hist) / SCALE,
                np.asarray(action_hist),
                light
            ])

            nxt = c.step(light)
            ys.append([nxt / SCALE])
            rate_hist = (rate_hist + [nxt])[-H:]
            action_hist = (action_hist + [light])[-H:]

    return np.asarray(xs), np.asarray(ys)


def transform(rate_hist, action_hist, mode, rng):
    r = np.asarray(rate_hist, dtype=float).copy()
    a = np.asarray(action_hist, dtype=float).copy()

    if mode == "full":
        pass
    elif mode == "joint_shuffle":
        p = rng.permutation(H)
        r, a = r[p], a[p]
    elif mode == "reverse":
        r, a = r[::-1], a[::-1]
    elif mode == "rate_only":
        a[:] = 0
    elif mode == "action_only":
        r[:] = 0
    elif mode == "rate_shuffle":
        r = r[rng.permutation(H)]
    elif mode == "action_shuffle":
        a = a[rng.permutation(H)]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return r, a


def controller(model, seed, sensitivity, mode):
    c = ShiftedCircuit(
        seed,
        noise=.5,
        sensitivity_scale=sensitivity,
        bias_shift=0,
    )
    targets = np.repeat([12., 30., 20., 40., 15., 35.], 30)
    rate_hist = [0.] * H
    action_hist = [0.] * H
    rates = []
    rng = np.random.default_rng(seed + 91000)

    for target in targets:
        rr, aa = transform(rate_hist, action_hist, mode, rng)
        candidates = np.linspace(0, 1, 101)
        base = np.r_[rr / SCALE, aa]
        x = np.column_stack([
            np.tile(base, (len(candidates), 1)),
            candidates
        ])
        predicted = model.predict(x).ravel() * SCALE

        # Keep the control regularizer tied to the true previous command.
        cost = (
            (predicted - target) ** 2
            + 2 * candidates ** 2
            + 2 * (candidates - action_hist[-1]) ** 2
        )
        light = float(candidates[np.argmin(cost)])

        actual = c.step(light)
        rates.append(actual)
        rate_hist = (rate_hist + [actual])[-H:]
        action_hist = (action_hist + [light])[-H:]

    rates = np.asarray(rates)
    return float(np.sqrt(np.mean((rates - targets) ** 2)))


def bootstrap_advantage(h4, pi, seed):
    # Positive PI-H4 difference means H4 has lower tracking error.
    diff = np.asarray(pi) - np.asarray(h4)
    rng = np.random.default_rng(seed)
    boot = np.array([
        rng.choice(diff, len(diff), replace=True).mean()
        for _ in range(5000)
    ])
    return {
        "pi_minus_h4_mean_hz": float(diff.mean()),
        "ci95_hz": [
            float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)),
        ],
        "h4_win_fraction": float(np.mean(diff > 0)),
    }


def main(out):
    out.mkdir(parents=True, exist_ok=True)

    print("Training H=4 mechanistic model", flush=True)
    x, y = dataset(range(1400, 1500))
    xv, yv = dataset(range(1500, 1530))

    model = MLP()
    best_epoch = model.fit(x, y, xv, yv)

    training = {
        "best_epoch": best_epoch,
        "train_rmse_hz": float(
            np.sqrt(np.mean((model.predict(x) - y) ** 2)) * SCALE
        ),
        "validation_rmse_hz": float(
            np.sqrt(np.mean((model.predict(xv) - yv) ** 2)) * SCALE
        ),
    }

    np.savez(
        out / "v06_h4_model.npz",
        w1=model.w1, b1=model.b1, w2=model.w2, b2=model.b2
    )

    _, kp, ki = tune_pi()
    modes = [
        "full",
        "joint_shuffle",
        "reverse",
        "rate_only",
        "action_only",
        "rate_shuffle",
        "action_shuffle",
    ]
    sensitivities = np.round(np.arange(.50, 1.5001, .05), 2)
    seeds = list(range(1600, 1650))
    results = {}

    for j, sensitivity in enumerate(sensitivities):
        key = f"{sensitivity:.2f}"
        print("Mechanistic sensitivity", key, flush=True)

        pi = np.array([
            run_controller(
                None, seed, "pi",
                kp=kp, ki=ki,
                sensitivity_scale=float(sensitivity)
            )["rmse_hz"]
            for seed in seeds
        ])

        row = {"pi_rmse_hz": float(pi.mean())}
        arrays = {}

        for mode in modes:
            arr = np.array([
                controller(model, seed, float(sensitivity), mode)
                for seed in seeds
            ])
            arrays[mode] = arr
            row[mode] = {
                "rmse_hz": float(arr.mean()),
                "delta_vs_full_hz": (
                    0.0 if mode == "full"
                    else float((arr - arrays["full"]).mean())
                ),
            }

        row["full_vs_pi"] = bootstrap_advantage(
            arrays["full"], pi, 20261006 + j
        )
        results[key] = row

    payload = {
        "protocol": {
            "purpose": "development-only mechanistic history ablation",
            "history_steps": H,
            "history_ms": 200,
            "train_seeds": "1400-1499",
            "validation_seeds": "1500-1529",
            "diagnostic_seeds": "1600-1649",
            "excluded_frozen_v04_final_seeds": "1000-1049",
            "interpretation": {
                "joint_shuffle": (
                    "preserves rate/action pairing while disrupting chronology"
                ),
                "reverse": (
                    "preserves aligned pairs and values while reversing chronology"
                ),
                "rate_only": "removes action-history information",
                "action_only": "removes rate-history information",
            },
        },
        "training": training,
        "tuned_pi": {"kp": kp, "ki": ki},
        "sensitivity": results,
    }

    (out / "v06_mechanistic.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )

    xs = sensitivities

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(
        xs,
        [results[f"{s:.2f}"]["pi_rmse_hz"] for s in xs],
        label="PI",
        linewidth=2,
    )
    for mode in ["full", "joint_shuffle", "reverse",
                 "rate_only", "action_only"]:
        ax.plot(
            xs,
            [results[f"{s:.2f}"][mode]["rmse_hz"] for s in xs],
            label=mode,
        )
    ax.set(
        xlabel="Sensitivity scale",
        ylabel="Tracking RMSE (Hz)",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "v06_mechanisms.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    advantage = np.array([
        results[f"{s:.2f}"]["full_vs_pi"]["pi_minus_h4_mean_hz"]
        for s in xs
    ])
    lo = np.array([
        results[f"{s:.2f}"]["full_vs_pi"]["ci95_hz"][0]
        for s in xs
    ])
    hi = np.array([
        results[f"{s:.2f}"]["full_vs_pi"]["ci95_hz"][1]
        for s in xs
    ])
    ax.plot(xs, advantage, label="PI - H4 RMSE")
    ax.fill_between(xs, lo, hi, alpha=.2)
    ax.axhline(0, linestyle="--")
    ax.set(
        xlabel="Sensitivity scale",
        ylabel="Advantage of H4 over PI (Hz)",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "v06_crossover_ci.png", dpi=170)
    plt.close(fig)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path, default=Path("results_v06")
    )
    main(parser.parse_args().out)
