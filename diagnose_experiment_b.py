import json
import numpy as np

PATH = "results_experiment_b_final/experiment_b_final_results.json"

with open(PATH, "r") as f:
    data = json.load(f)

def summarize_scalar(records, field):
    x = np.asarray([r[field] for r in records], dtype=float)
    return {
        "mean": float(x.mean()),
        "sd": float(x.std(ddof=1)),
        "min": float(x.min()),
        "max": float(x.max()),
    }

def h4_records(condition):
    # 5 model initializations × 50 plants = 250 records.
    return [
        plant
        for model in condition["raw"]["h4"]
        for plant in model["plants"]
    ]

def summarize_condition(s):
    c = data["conditions"][f"{s:.2f}"]

    arx = c["raw"]["arx10_rls"]
    h4 = h4_records(c)
    pi = c["raw"]["fixed_pi"]

    arx_sat = np.asarray(
        [r["fraction_zero"] + r["fraction_one"] for r in arx]
    )
    h4_sat = np.asarray(
        [r["fraction_zero"] + r["fraction_one"] for r in h4]
    )
    pi_sat = np.asarray(
        [r["fraction_zero"] + r["fraction_one"] for r in pi]
    )

    arx_slew = np.asarray(
        [r["mean_abs_action_change"] for r in arx]
    )
    h4_slew = np.asarray(
        [r["mean_abs_action_change"] for r in h4]
    )
    pi_slew = np.asarray(
        [r["mean_abs_action_change"] for r in pi]
    )

    # Stored normalized residual from the RLS update:
    # e_t = y_t / 60 - z_t @ w_t
    arx_pred_mae_hz = np.asarray(
        [
            60.0 * r["mean_abs_identification_error_normalized"]
            for r in arx
        ]
    )

    W = np.asarray(
        [r["final_parameters"] for r in arx],
        dtype=float
    )

    return {
        "sensitivity": s,

        "ARX_boundary_fraction_mean": float(arx_sat.mean()),
        "ARX_boundary_fraction_sd": float(arx_sat.std(ddof=1)),

        "H4_boundary_fraction_mean": float(h4_sat.mean()),
        "H4_boundary_fraction_sd": float(h4_sat.std(ddof=1)),

        "PI_boundary_fraction_mean": float(pi_sat.mean()),
        "PI_boundary_fraction_sd": float(pi_sat.std(ddof=1)),

        "ARX_slew_mean": float(arx_slew.mean()),
        "ARX_slew_sd": float(arx_slew.std(ddof=1)),

        "H4_slew_mean": float(h4_slew.mean()),
        "H4_slew_sd": float(h4_slew.std(ddof=1)),

        "PI_slew_mean": float(pi_slew.mean()),
        "PI_slew_sd": float(pi_slew.std(ddof=1)),

        "ARX_prediction_MAE_Hz_mean":
            float(arx_pred_mae_hz.mean()),
        "ARX_prediction_MAE_Hz_sd":
            float(arx_pred_mae_hz.std(ddof=1)),

        "ARX_final_b0_mean": float(W[:, 0].mean()),
        "ARX_final_b0_sd": float(W[:, 0].std(ddof=1)),

        "ARX_final_weight_norm_mean":
            float(np.linalg.norm(W, axis=1).mean()),
        "ARX_final_weight_norm_sd":
            float(np.linalg.norm(W, axis=1).std(ddof=1)),

        "ARX_final_parameter_mean":
            W.mean(axis=0).tolist(),

        "ARX_final_parameter_sd":
            W.std(axis=0, ddof=1).tolist(),
    }

results = [summarize_condition(s) for s in [0.60, 0.80, 1.00, 1.20]]

print("\nFROZEN EXPERIMENT-B POST-HOC DIAGNOSTICS")
print("=" * 100)

header = (
    f"{'s':>5} | "
    f"{'ARX sat':>9} | {'H4 sat':>9} | {'PI sat':>9} | "
    f"{'ARX slew':>9} | {'H4 slew':>9} | {'PI slew':>9} | "
    f"{'ARX pred MAE Hz':>15} | {'final b0':>10} | {'||w||2':>10}"
)

print(header)
print("-" * len(header))

for r in results:
    print(
        f"{r['sensitivity']:5.2f} | "
        f"{r['ARX_boundary_fraction_mean']:9.4f} | "
        f"{r['H4_boundary_fraction_mean']:9.4f} | "
        f"{r['PI_boundary_fraction_mean']:9.4f} | "
        f"{r['ARX_slew_mean']:9.4f} | "
        f"{r['H4_slew_mean']:9.4f} | "
        f"{r['PI_slew_mean']:9.4f} | "
        f"{r['ARX_prediction_MAE_Hz_mean']:15.4f} | "
        f"{r['ARX_final_b0_mean']:10.4f} | "
        f"{r['ARX_final_weight_norm_mean']:10.4f}"
    )

print("\nARX FINAL PARAMETER MEANS")
print("=" * 100)

names = [
    "candidate_u",
    "rate_t",
    "rate_t-1",
    "rate_t-2",
    "rate_t-3",
    "u_t-1",
    "u_t-2",
    "u_t-3",
    "u_t-4",
    "intercept",
]

for r in results:
    print(f"\ns = {r['sensitivity']:.2f}")
    for name, mean, sd in zip(
        names,
        r["ARX_final_parameter_mean"],
        r["ARX_final_parameter_sd"],
    ):
        print(f"  {name:<12} {mean:+.6f} ± {sd:.6f}")