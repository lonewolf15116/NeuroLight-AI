"""NeuroLight-AI Experiment B -- DEVELOPMENT ONLY.

Tuning seeds: 5000-5099
Validation seeds: 5100-5149
No confirmatory holdout is declared or touched by this script.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from v02_robustness import ShiftedCircuit

SENS = [0.60, 0.80, 1.00, 1.20]
TUNE = range(5000, 5100)
VAL = range(5100, 5150)
TARGETS = np.repeat([12., 30., 20., 40., 15., 35.], 30)
DT = 0.05
BASE_KP, BASE_KI = 0.02, 0.15
KP_BOUNDS = (0.005, 0.08)
KI_BOUNDS = (0.02, 0.40)
GAIN_BOUNDS = (1.0, 80.0)
GAIN_REF = 30.0
MIN_DU = 0.02
WARMUP = 8

PI_GRID = [(kp, ki) for kp in [0.01, 0.02, 0.03, 0.04, 0.06]
                    for ki in [0.05, 0.10, 0.15, 0.20, 0.30]]
WIN_GRID = [(w, r) for w in [6, 12, 24] for r in [0.01, 0.10, 1.0]]
EWLS_GRID = [(f, r) for f in [0.85, 0.93, 0.98] for r in [0.01, 0.10, 1.0]]

def calc_metrics(targets, rates, lights):
    rates = np.asarray(rates)
    lights = np.asarray(lights)
    return {
        "rmse_hz": float(np.sqrt(np.mean((rates-targets)**2))),
        "mean_intensity": float(lights.mean()),
        "fraction_zero": float(np.mean(lights <= 1e-12)),
        "fraction_one": float(np.mean(lights >= 1-1e-12)),
        "mean_abs_action_change": float(np.mean(np.abs(np.diff(lights))))
    }

def run_pi(seed, sensitivity, kp=BASE_KP, ki=BASE_KI):
    c = ShiftedCircuit(seed, noise=0.5, sensitivity_scale=sensitivity, bias_shift=0.0)
    integral = 0.0
    rates, lights = [], []
    for target in TARGETS:
        error = target - c.rate
        integral = float(np.clip(integral + error*DT, -10, 10))
        light = float(np.clip(kp*error + ki*integral, 0, 1))
        rates.append(c.step(light))
        lights.append(light)
    return calc_metrics(TARGETS, np.array(rates), np.array(lights))

class WindowedGain:
    def __init__(self, window, ridge):
        self.window, self.ridge = int(window), float(ridge)
        self.du, self.dr = [], []
    def update(self, du, dr):
        if abs(du) >= MIN_DU:
            self.du.append(float(du))
            self.dr.append(float(dr))
            self.du = self.du[-self.window:]
            self.dr = self.dr[-self.window:]
    def estimate(self):
        if len(self.du) < 3:
            return GAIN_REF
        x, y = np.asarray(self.du), np.asarray(self.dr)
        g = float((x @ y) / (x @ x + self.ridge))
        return float(np.clip(abs(g), *GAIN_BOUNDS))

class EWLSGain:
    def __init__(self, forgetting, ridge):
        self.f, self.ridge = float(forgetting), float(ridge)
        self.xx = self.xy = 0.0
        self.n = 0
    def update(self, du, dr):
        if abs(du) >= MIN_DU:
            self.xx = self.f*self.xx + du*du
            self.xy = self.f*self.xy + du*dr
            self.n += 1
    def estimate(self):
        if self.n < 3:
            return GAIN_REF
        g = float(self.xy / (self.xx + self.ridge))
        return float(np.clip(abs(g), *GAIN_BOUNDS))

def run_adaptive(seed, sensitivity, family, a, ridge):
    # sensitivity is used only to instantiate the simulated plant.
    # The controller never reads it.
    c = ShiftedCircuit(seed, noise=0.5, sensitivity_scale=sensitivity, bias_shift=0.0)
    est = WindowedGain(a, ridge) if family == "windowed_ridge" else EWLSGain(a, ridge)
    integral = 0.0
    rates, lights, gains = [], [], []
    previous_observation = c.rate

    for t, target in enumerate(TARGETS):
        observed = c.rate

        # Associate the most recent observed response with the previous action change.
        if len(lights) >= 2:
            du = lights[-1] - lights[-2]
            dr = observed - previous_observation
            est.update(du, dr)

        g = est.estimate()
        scale = GAIN_REF/g if t >= WARMUP else 1.0
        kp = float(np.clip(BASE_KP*scale, *KP_BOUNDS))
        ki = float(np.clip(BASE_KI*scale, *KI_BOUNDS))

        error = target - observed
        integral = float(np.clip(integral + error*DT, -10, 10))
        light = float(np.clip(kp*error + ki*integral, 0, 1))

        previous_observation = observed
        rates.append(c.step(light))
        lights.append(light)
        gains.append(g)

    out = calc_metrics(TARGETS, np.array(rates), np.array(lights))
    out["mean_gain_hat"] = float(np.mean(gains[WARMUP:]))
    out["final_gain_hat"] = float(gains[-1])
    return out

def evaluate(fn, seeds, sensitivity):
    records = [fn(seed, sensitivity) for seed in seeds]
    keys = records[0].keys()
    return {
        "mean": {k: float(np.mean([r[k] for r in records])) for k in keys},
        "std": {k: float(np.std([r[k] for r in records])) for k in keys},
        "raw": records
    }

def adaptive_score(family, a, ridge):
    by_sensitivity = []
    for sens in SENS:
        rec = evaluate(lambda seed, x: run_adaptive(seed, x, family, a, ridge), TUNE, sens)
        by_sensitivity.append(rec["mean"]["rmse_hz"])
    return float(np.mean(by_sensitivity)), by_sensitivity

def main(out):
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "warning": "DEVELOPMENT ONLY. No confirmatory holdout was declared or touched.",
        "tuning_seeds": "5000-5099",
        "validation_seeds": "5100-5149",
        "sensitivity": SENS
    }

    result["fixed_pi_validation"] = {
        str(s): evaluate(lambda seed, x: run_pi(seed, x), VAL, s) for s in SENS
    }

    oracle_choice, oracle_validation = {}, {}
    for sens in SENS:
        scores = []
        for index, (kp, ki) in enumerate(PI_GRID):
            rec = evaluate(lambda seed, x, kp=kp, ki=ki: run_pi(seed, x, kp, ki), TUNE, sens)
            scores.append((rec["mean"]["rmse_hz"], index, kp, ki))
        best = min(scores)
        oracle_choice[str(sens)] = {
            "kp": best[2], "ki": best[3], "tuning_rmse_hz": best[0]
        }
        oracle_validation[str(sens)] = evaluate(
            lambda seed, x, kp=best[2], ki=best[3]: run_pi(seed, x, kp, ki), VAL, sens
        )

    result["oracle_pi_choice"] = oracle_choice
    result["oracle_pi_validation"] = oracle_validation

    candidates = []
    order = 0
    for w, ridge in WIN_GRID:
        score, by = adaptive_score("windowed_ridge", w, ridge)
        candidates.append((score, order, "windowed_ridge", w, ridge, by))
        order += 1
    for forgetting, ridge in EWLS_GRID:
        score, by = adaptive_score("ewls", forgetting, ridge)
        candidates.append((score, order, "ewls", forgetting, ridge, by))
        order += 1

    best = min(candidates)
    result["adaptive_search"] = [
        {
            "mean_tuning_rmse_hz": c[0],
            "family": c[2],
            "a": c[3],
            "ridge": c[4],
            "rmse_by_sensitivity": c[5]
        }
        for c in sorted(candidates)
    ]
    result["adaptive_choice"] = {
        "family": best[2], "a": best[3], "ridge": best[4],
        "mean_tuning_rmse_hz": best[0]
    }
    result["adaptive_validation"] = {
        str(s): evaluate(
            lambda seed, x: run_adaptive(seed, x, best[2], best[3], best[4]), VAL, s
        )
        for s in SENS
    }

    (out/"experiment_b_development.json").write_text(json.dumps(result, indent=2)+"\n")
    print("DEVELOPMENT ONLY -- no confirmation holdout touched")
    print("Adaptive choice:", result["adaptive_choice"])
    for sens in SENS:
        f = result["fixed_pi_validation"][str(sens)]["mean"]["rmse_hz"]
        o = result["oracle_pi_validation"][str(sens)]["mean"]["rmse_hz"]
        a = result["adaptive_validation"][str(sens)]["mean"]["rmse_hz"]
        print(f"sensitivity {sens:.2f}: fixed PI {f:.4f} | oracle PI {o:.4f} | adaptive PI {a:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results_experiment_b_dev"))
    main(parser.parse_args().out)
