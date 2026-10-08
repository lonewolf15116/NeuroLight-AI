"""NeuroLight-AI B-Dev-v1: Online RLS-ARX vs frozen DR-H4.

DEVELOPMENT ONLY.
Tuning seeds: 5000-5099
Validation seeds: 5100-5149
No confirmatory holdout is declared or touched.

Purpose:
Test whether linear online system identification with the same H=4 observable
history and the same 101-action one-step predictive search can account for
the frozen DR-H4 controller's history advantage.
"""
from pathlib import Path
import argparse, json
import numpy as np
from v02_robustness import ShiftedCircuit

H = 4
SCALE = 60.0
SENS = [0.60, 0.80, 1.00, 1.20]
TUNE = range(5000, 5100)
VAL = range(5100, 5150)
MODEL_SEEDS = [101, 102, 103, 104, 105]
TARGETS = np.repeat([12., 30., 20., 40., 15., 35.], 30)
CANDIDATES = np.linspace(0., 1., 101)

# Development search only. One global configuration is selected by equal-weight
# mean tuning RMSE across all four sensitivity conditions.
LAMBDA_GRID = [0.98, 0.99, 0.995, 0.999]
P0_GRID = [10.0, 100.0, 1000.0]

def control_metrics(rates, lights):
    rates = np.asarray(rates)
    lights = np.asarray(lights)
    return {
        "rmse_hz": float(np.sqrt(np.mean((rates - TARGETS)**2))),
        "mean_intensity": float(np.mean(lights)),
        "fraction_zero": float(np.mean(lights <= 1e-12)),
        "fraction_one": float(np.mean(lights >= 1 - 1e-12)),
        "mean_abs_action_change": float(np.mean(np.abs(np.diff(lights))))
    }

class FrozenH4:
    def __init__(self, path):
        z = np.load(path)
        self.w1, self.b1 = z["w1"], z["b1"]
        self.w2, self.b2 = z["w2"], z["b2"]

    def predict(self, x):
        return np.tanh(x @ self.w1 + self.b1) @ self.w2 + self.b2

def run_h4(model, seed, sensitivity):
    c = ShiftedCircuit(seed, noise=.5, sensitivity_scale=sensitivity, bias_shift=0.)
    rh = [0.] * H
    ah = [0.] * H
    rates, lights = [], []

    for target in TARGETS:
        x = np.column_stack([
            np.tile(np.asarray(rh) / SCALE, (101, 1)),
            np.tile(np.asarray(ah), (101, 1)),
            CANDIDATES
        ])
        predicted_hz = model.predict(x).ravel() * SCALE

        # Exact Experiment-A/H4 objective: prediction error is in Hz.
        cost = ((predicted_hz - target)**2
                + 2 * CANDIDATES**2
                + 2 * (CANDIDATES - ah[-1])**2)
        u = float(CANDIDATES[np.argmin(cost)])
        r = c.step(u)

        rates.append(r)
        lights.append(u)
        rh = rh[1:] + [r]
        ah = ah[1:] + [u]

    return control_metrics(rates, lights)

class RLS:
    def __init__(self, forgetting, p0):
        self.w = np.zeros(9)
        self.P = np.eye(9) * float(p0)
        self.lam = float(forgetting)

    def predict_normalized(self, z):
        return float(np.asarray(z) @ self.w)

    def update(self, z, y_normalized):
        z = np.asarray(z, dtype=float)
        Pz = self.P @ z
        denom = self.lam + z @ Pz
        k = Pz / denom
        error = float(y_normalized - z @ self.w)
        self.w = self.w + k * error
        self.P = (self.P - np.outer(k, z) @ self.P) / self.lam
        # Numerical symmetry guard only; does not alter the estimator design.
        self.P = 0.5 * (self.P + self.P.T)
        return error

def arx_z(rh, ah, candidate):
    """9-vector: current candidate + four rates + three prior actions + intercept.

    rh = [y_{t-3}, y_{t-2}, y_{t-1}, y_t]
    ah = [u_{t-4}, u_{t-3}, u_{t-2}, u_{t-1}]

    z_t = [u_t, y_t/60, y_{t-1}/60, y_{t-2}/60, y_{t-3}/60,
           u_{t-1}, u_{t-2}, u_{t-3}, 1]
    """
    return np.array([
        candidate,
        rh[-1] / SCALE, rh[-2] / SCALE, rh[-3] / SCALE, rh[-4] / SCALE,
        ah[-1], ah[-2], ah[-3],
        1.0
    ])

def run_arx(seed, sensitivity, forgetting, p0):
    c = ShiftedCircuit(seed, noise=.5, sensitivity_scale=sensitivity, bias_shift=0.)
    rls = RLS(forgetting, p0)
    rh = [0.] * H
    ah = [0.] * H
    rates, lights, errors = [], [], []

    for target in TARGETS:
        predicted_hz = np.array([
            rls.predict_normalized(arx_z(rh, ah, u)) * SCALE
            for u in CANDIDATES
        ])

        # Same objective and units as frozen H4.
        cost = ((predicted_hz - target)**2
                + 2 * CANDIDATES**2
                + 2 * (CANDIDATES - ah[-1])**2)
        u = float(CANDIDATES[np.argmin(cost)])

        # Execute selected action and then update from the observed transition.
        z = arx_z(rh, ah, u)
        r = c.step(u)
        errors.append(rls.update(z, r / SCALE))

        rates.append(r)
        lights.append(u)
        rh = rh[1:] + [r]
        ah = ah[1:] + [u]

    out = control_metrics(rates, lights)
    out["mean_abs_identification_error_normalized"] = float(np.mean(np.abs(errors)))
    out["final_parameters"] = [float(x) for x in rls.w]
    return out

def evaluate(fn, seeds, sensitivity):
    raw = [fn(seed, sensitivity) for seed in seeds]
    scalar_keys = [k for k, v in raw[0].items() if np.isscalar(v)]
    return {
        "mean": {k: float(np.mean([r[k] for r in raw])) for k in scalar_keys},
        "std": {k: float(np.std([r[k] for r in raw])) for k in scalar_keys},
        "raw": raw
    }

def main(out, h4_dir):
    out.mkdir(parents=True, exist_ok=True)

    models = []
    for seed in MODEL_SEEDS:
        path = h4_dir / f"DR_H4_seed_{seed}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Frozen H4 model not found: {path}")
        models.append((seed, FrozenH4(path)))

    result = {
        "warning": "B-DEV-v1 DEVELOPMENT ONLY. No confirmatory holdout declared or touched.",
        "tuning_seeds": "5000-5099",
        "validation_seeds": "5100-5149",
        "sensitivities": SENS,
        "h4_source": str(h4_dir),
        "rls_search": {
            "forgetting": LAMBDA_GRID,
            "p0": P0_GRID,
            "selection": "minimum equal-weight mean tuning RMSE over all four sensitivities"
        }
    }

    # Tune RLS hyperparameters ONLY on 5000-5099.
    search = []
    for order, (lam, p0) in enumerate(
            (x for lam in LAMBDA_GRID for p0 in P0_GRID for x in [(lam, p0)])):
        by_sens = []
        for sens in SENS:
            ev = evaluate(
                lambda seed, s, lam=lam, p0=p0: run_arx(seed, s, lam, p0),
                TUNE, sens
            )
            by_sens.append(ev["mean"]["rmse_hz"])
        search.append({
            "order": order,
            "forgetting": lam,
            "p0": p0,
            "rmse_by_sensitivity": by_sens,
            "mean_tuning_rmse_hz": float(np.mean(by_sens))
        })

    best = min(search, key=lambda x: (x["mean_tuning_rmse_hz"], x["order"]))
    result["rls_search_results"] = sorted(search, key=lambda x: x["mean_tuning_rmse_hz"])
    result["rls_choice"] = {
        "forgetting": best["forgetting"],
        "p0": best["p0"],
        "mean_tuning_rmse_hz": best["mean_tuning_rmse_hz"]
    }

    # Evaluate chosen ARX unchanged on validation seeds.
    result["arx_validation"] = {
        str(sens): evaluate(
            lambda seed, s: run_arx(
                seed, s, best["forgetting"], best["p0"]),
            VAL, sens
        )
        for sens in SENS
    }

    # Frozen H4: 5 model initializations x same 50 validation plants.
    result["h4_validation"] = {}
    for sens in SENS:
        model_records = []
        matrix = []
        for model_seed, model in models:
            ev = evaluate(
                lambda seed, s, model=model: run_h4(model, seed, s),
                VAL, sens
            )
            model_records.append({"model_seed": model_seed, "evaluation": ev})
            matrix.append([r["rmse_hz"] for r in ev["raw"]])

        arr = np.asarray(matrix)  # 5 models x 50 plants
        result["h4_validation"][str(sens)] = {
            "overall_mean_rmse_hz": float(arr.mean()),
            "per_model_mean_rmse_hz": [float(x) for x in arr.mean(axis=1)],
            "plant_mean_rmse_hz": [float(x) for x in arr.mean(axis=0)],
            "plant_sd_after_model_average": float(arr.mean(axis=0).std()),
            "model_records": model_records
        }

    # Same-plant development comparison. No confirmatory CI here.
    paired = {}
    for sens in SENS:
        k = str(sens)
        h4 = np.asarray(result["h4_validation"][k]["plant_mean_rmse_hz"])
        arx = np.asarray([r["rmse_hz"] for r in result["arx_validation"][k]["raw"]])
        d = arx - h4
        paired[k] = {
            "ARX_minus_H4_mean_hz": float(d.mean()),
            "ARX_minus_H4_sd_across_plants_hz": float(d.std()),
            "fraction_ARX_minus_H4_positive": float(np.mean(d > 0))
        }
    result["development_paired_difference"] = paired

    path = out / "experiment_b_dev_v1.json"
    path.write_text(json.dumps(result, indent=2) + "\n")

    print("B-DEV-v1 ONLY -- no confirmation holdout touched")
    print("RLS choice:", result["rls_choice"])
    for sens in SENS:
        k = str(sens)
        arx = result["arx_validation"][k]["mean"]["rmse_hz"]
        h4 = result["h4_validation"][k]["overall_mean_rmse_hz"]
        d = result["development_paired_difference"][k]["ARX_minus_H4_mean_hz"]
        print(f"sensitivity {sens:.2f}: ARX {arx:.4f} | H4 {h4:.4f} | ARX-H4 {d:+.4f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("results_experiment_b_dev_v1"))
    p.add_argument(
        "--h4-dir", type=Path,
        default=Path("archived_results/EXPERIMENT_A_FINAL_2026-10-07")
    )
    args = p.parse_args()
    main(args.out, args.h4_dir)
