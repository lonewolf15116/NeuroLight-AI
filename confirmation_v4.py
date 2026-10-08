"""NeuroLight-AI prospective confirmation v4 (phase 4: mechanisms).

All parameters come from PROTOCOL_V4.json; nothing here may be tuned after freezing.
  python confirmation_v4.py --audit-only            hashes + smoke test on ONE development plant (5200); no confirmation plant
  python confirmation_v4.py --self-test             full analysis on THREE development plants (5200-5202); not interpreted
  python confirmation_v4.py [--budget S]            one-shot confirmation (requires FROZEN + matching MANIFEST_V4.json);
                                                    resumable per plant-condition; plants are SPENT after the first completed run.
Exact scalar simulator (v02_robustness.ShiftedCircuit), as in every earlier confirmation.
"""
from pathlib import Path
import argparse, json, hashlib, time
import numpy as np
from statistics import NormalDist
from v02_robustness import ShiftedCircuit
import experiment_b_final as E
import confirmation_v3 as V3

ROOT = Path(__file__).resolve().parent
H = 4; SCALE = 60.; TARGETS = V3.TARGETS; CAND = V3.CAND
MANIFEST = ROOT / "MANIFEST_V4.json"
sha256 = V3.sha256


# ---------------------------------------------------------------- controllers (exact simulator)
def run_arx_gain(seed, s, noise, w0, lam, p0, act_idx):
    """One-parameter input-gain RLS: prediction = non-action part of prior + g * action part."""
    mask = np.zeros(10); mask[act_idx] = 1; wa = w0 * mask; wr = w0 * (1 - mask)
    c = ShiftedCircuit(seed, noise=noise, sensitivity_scale=s, bias_shift=0.)
    rh = [0.] * H; ah = [0.] * H; g = 1.; P = p0; R = []; U = []; G = []
    for t in TARGETS:
        Z = np.stack([E.zvec(rh, ah, u) for u in CAND]); pred = (Z @ wr + g * (Z @ wa)) * SCALE
        cost = (pred - t) ** 2 + 2 * CAND ** 2 + 2 * (CAND - ah[-1]) ** 2
        i = int(np.argmin(cost)); u = float(CAND[i]); y = float(c.step(u))
        a = float(Z[i] @ wa); e = y / SCALE - (float(Z[i] @ wr) + g * a)
        k = P * a / (lam + a * P * a); g += k * e; P = (P - k * a * P) / lam
        R.append(y); U.append(u); G.append(g); rh = (rh + [y])[-H:]; ah = (ah + [u])[-H:]
    o = V3.metric(R, U); o.update(g_final_mean60=float(np.mean(G[-60:]))); return o


def run_net(m, seed, s, noise, gain_feature=None):
    """One-step predictive control with a DR-H4-type network; optional g input from the one-parameter gain RLS."""
    c = ShiftedCircuit(seed, noise=noise, sensitivity_scale=s, bias_shift=0.)
    rh = [0.] * H; ah = [0.] * H; R = []; U = []; sens = []
    if gain_feature:
        w0, lam, p0, act = gain_feature; mask = np.zeros(10); mask[act] = 1; wa = w0 * mask; wr = w0 * (1 - mask); g = 1.; P = p0
    for t in TARGETS:
        X = np.column_stack([np.tile(np.r_[np.asarray(rh) / SCALE, np.asarray(ah)], (101, 1)), CAND])
        if gain_feature: X = np.column_stack([X, np.full(101, g)])
        pred = V3.predict(m, X) * SCALE
        cost = (pred - t) ** 2 + 2 * CAND ** 2 + 2 * (CAND - ah[-1]) ** 2
        i = int(np.argmin(cost)); u = float(CAND[i]); y = float(c.step(u))
        lo, hi = max(i - 2, 0), min(i + 2, 100); sens.append((pred[hi] - pred[lo]) / (CAND[hi] - CAND[lo]))
        if gain_feature:
            z = E.zvec(rh, ah, u); a = float(z @ wa); e = y / SCALE - (float(z @ wr) + g * a)
            k = P * a / (lam + a * P * a); g += k * e; P = (P - k * a * P) / lam
        R.append(y); U.append(u); rh = (rh + [y])[-H:]; ah = (ah + [u])[-H:]
    o = V3.metric(R, U); o.update(dyhat_du=float(np.mean(sens))); return o


def load_models(spec):
    out = {}
    for name, files in spec.items():
        out[name] = []
        for f in files:
            p = ROOT / f["path"]
            if sha256(p) != f["sha256"]: raise RuntimeError(f"hash mismatch {p}")
            z = np.load(p); out[name].append({k: z[k] for k in ("w1", "b1", "w2", "b2")})
    return out


def ckey(s, nz, P):
    return f"s{s:.2f}" if nz == P["noise_primary"] else f"n{nz:.1f}_s{s:.2f}"


def conditions(P):
    return [{"key": ckey(s, nz, P), "s": s, "noise": nz} for nz in [P["noise_primary"]] + P["noise_robustness"] for s in P["sensitivities"]]


def run_plant(seed, cond, P, models, prior):
    s, nz = cond["s"], cond["noise"]; C = P["controllers"]; rec = {}
    gf = (prior["w0"], C["ARX_gain"]["lam"], C["ARX_gain"]["p0"], C["ARX_gain"]["act_idx"])
    rec["PI_tuned"] = V3.run_pi(seed, s, nz, **C["PI_tuned"])
    rec["PI_tuned_AW"] = V3.run_pi(seed, s, nz, **C["PI_tuned_AW"])
    rec["ARX_fixed"] = V3.run_arx(seed, s, nz, prior["w0"], prior["cov0"], adapt=False, lam=1.0, p_scale=1.0)
    rec["ARX_RLS"] = V3.run_arx(seed, s, nz, prior["w0"], prior["cov0"], adapt=True, **C["ARX_RLS"])
    rec["ARX_gain"] = run_arx_gain(seed, s, nz, prior["w0"], **C["ARX_gain"])
    for name in P["network_arms"]:
        rec[name] = [run_net(m, seed, s, nz, gf if name.startswith("H4_I2") else None) for m in models[name]]
    return rec


val = V3.plant_value


def bca_stat(stat, data, seed, nboot, alpha):
    """BCa interval for a general statistic of plant-level rows (data: n x k array)."""
    data = np.asarray(data); n = len(data); theta = float(stat(data)); rng = np.random.default_rng(seed)
    boots = np.array([stat(data[rng.integers(0, n, n)]) for _ in range(nboot)])
    nd = NormalDist(); prop = min(max(float(np.mean(boots < theta)), 1 / (2 * nboot)), 1 - 1 / (2 * nboot)); z0 = nd.inv_cdf(prop)
    jack = np.array([stat(np.delete(data, i, 0)) for i in range(n)]); jm = jack.mean()
    den = 6 * (np.sum((jm - jack) ** 2) ** 1.5); acc = float(np.sum((jm - jack) ** 3) / den) if den > 0 else 0.
    qs = [float(np.clip(nd.cdf(z0 + (z0 + z) / (1 - acc * (z0 + z))), 0, 1)) for z in (nd.inv_cdf(alpha / 2), nd.inv_cdf(1 - alpha / 2))]
    lo, hi = np.quantile(boots, qs)
    return {"estimate": theta, "ci": [float(lo), float(hi)], "ci_level": 1 - alpha, "resamples": nboot, "bootstrap_seed": seed}


def analyse(raw, P):
    A = P["analysis"]; NB = P["statistics"]["resamples"]; out = {}; k = [0]
    n = len(raw[conditions(P)[0]["key"]]); sv9 = np.array(P["sensitivities"])
    keys = lambda nz: [ckey(x, nz, P) for x in P["sensitivities"]]
    hikeys = lambda nz: [ckey(x, nz, P) for x in A["high_gain_set"]]
    def nextseed(): k[0] += 1; return 800000 + k[0]
    def diff(a, b, c, alpha, tag):
        d = np.asarray([val(r, a) - val(r, b) for r in raw[c]])
        res = E.bca_mean_ci(d, seed=nextseed(), nboot=NB, alpha=alpha)
        res.update(contrast=f"{a} - {b}", condition=c, ci=res.pop("bca95"), ci_level=1 - alpha, positive_fraction=float(np.mean(d > 0)), tag=tag)
        return res
    def mean_stat(rows, alpha):   # BCa for the mean over plants of one per-plant value
        return bca_stat(lambda d: d[:, 0].mean(), np.asarray(rows).reshape(-1, 1), nextseed(), NB, alpha)
    def q1(nz, alpha):
        C = keys(nz); rows = np.array([[np.mean([val(raw[c][i], "ARX_fixed") - val(raw[c][i], "ARX_gain") for c in C]),
                                        np.mean([val(raw[c][i], "ARX_fixed") - val(raw[c][i], "ARX_RLS") for c in C])] for i in range(n)])
        return bca_stat(lambda d: d[:, 0].mean() / d[:, 1].mean(), rows, nextseed(), NB, alpha)
    def hi_mean(fn, nz, alpha):
        return mean_stat([np.mean([fn(raw[c][i]) for c in hikeys(nz)]) for i in range(n)], alpha)
    plain = lambda r: val(r, "H4_snap_e6000") - val(r, "H4_snap_e900")
    gain = lambda r: val(r, "H4_I2_e6000") - val(r, "H4_I2_e900")
    q3f = lambda r: val(r, "H4_10k") - val(r, "H4_I2")
    q4f = lambda r: plain(r) - gain(r)
    P0 = P["noise_primary"]
    # ---------------- primary (sigma = noise_primary)
    out["Q1_gain_fraction"] = q1(P0, A["Q1"]["alpha"])
    out["Q3_gain_input_vs_10k"] = hi_mean(q3f, P0, A["Q3"]["alpha"])
    out["Q4_differential_training_length"] = hi_mean(q4f, P0, A["Q4"]["alpha"])
    # ---------------- secondary
    gsum = lambda i, nz: np.array([raw[c][i]["ARX_gain"]["g_final_mean60"] for c in keys(nz)])
    rk = lambda x: np.argsort(np.argsort(x)).astype(float)
    rho = [np.corrcoef(rk(gsum(i, P0)), rk(sv9))[0, 1] for i in range(n)]
    out["S_Q2_gain_spearman"] = mean_stat(rho, A["S_Q2"]["alpha"]); out["S_Q2_gain_spearman"]["per_plant"] = rho
    fits = np.array([np.polyfit(sv9, gsum(i, P0), 1) for i in range(n)])          # per plant: g = a*s + b
    resid = [float(np.mean(np.abs(gsum(i, P0) - np.polyval(fits[i], sv9)))) for i in range(n)]
    g_nom = [float(gsum(i, P0)[list(sv9).index(1.0)]) for i in range(n)]
    ratio_cv = [float(np.std(gsum(i, P0) / sv9) / np.mean(gsum(i, P0) / sv9)) for i in range(n)]
    out["S_calibration"] = {"slope_dg_ds": mean_stat(fits[:, 0], A["S_calibration"]["alpha_each"]),
                            "intercept": mean_stat(fits[:, 1], A["S_calibration"]["alpha_each"]),
                            "mean_abs_residual_from_linear": mean_stat(resid, A["S_calibration"]["alpha_each"]),
                            "g_at_nominal_s1_minus_1": mean_stat([x - 1 for x in g_nom], A["S_calibration"]["alpha_each"]),
                            "cv_of_g_over_s": mean_stat(ratio_cv, A["S_calibration"]["alpha_each"]),
                            "note": "g is a multiplier on the prior's action terms; g = 1 means the prior gain. Its absolute scale is defined relative to the domain-randomised prior, not to s itself."}
    out["S_Q4_components"] = {"plain_e6000_minus_e900": hi_mean(plain, P0, A["S_Q4dir"]["alpha_each"]),
                              "gaininput_e6000_minus_e900": hi_mean(gain, P0, A["S_Q4dir"]["alpha_each"])}
    out["S_Q4_opposite_directions"] = bool(out["S_Q4_components"]["plain_e6000_minus_e900"]["ci"][0] > 0 and out["S_Q4_components"]["gaininput_e6000_minus_e900"]["ci"][1] < 0)
    out["S_Q3_per_setting"] = [diff("H4_10k", "H4_I2", c, A["S_Q3_per_setting"]["alpha_each"], "Q3c") for c in hikeys(P0)]
    out["S1_I2_vs_ARX_RLS"] = [diff("H4_I2", "ARX_RLS", c, A["S1"]["alpha_each"], "S1") for c in keys(P0)]
    out["S2_I1_vs_10k"] = [diff("H4_10k", "H4_I1", c, A["S2"]["alpha_each"], "S2") for c in keys(P0)]
    out["S3_I3_vs_10k"] = [diff("H4_10k", "H4_I3", c, A["S3"]["alpha_each"], "S3") for c in keys(P0)]
    out["S6_I2_vs_PI_AW"] = [diff("H4_I2", "PI_tuned_AW", c, A["S6"]["alpha_each"], "S6") for c in keys(P0)]
    w0u = P["arx_prior"]["w_u"]
    def slope_rows(name):
        return np.array([[np.mean([q["dyhat_du"] for q in raw[c][i][name]]) for c in keys(P0)] +
                         [SCALE * w0u * raw[c][i]["ARX_gain"]["g_final_mean60"] for c in keys(P0)] for i in range(n)])
    sl = lambda d: np.polyfit(sv9, d[:, :9].mean(0), 1)[0] / np.polyfit(sv9, d[:, 9:].mean(0), 1)[0]
    out["S4_sensitivity_slope_ratio"] = {nm: bca_stat(sl, slope_rows(nm), nextseed(), NB, A["S4"]["alpha_each"]) for nm in ("H4_10k", "H4_900", "H4_I2")}
    # ---------------- robustness (sigma = 2.0): primary statistics repeated, secondary role
    for nz in P["noise_robustness"]:
        al = A["R"]["alpha_each"]; tag = f"R_sigma{nz:.1f}"
        out[tag] = {"Q1_gain_fraction": q1(nz, al), "Q3_gain_input_vs_10k": hi_mean(q3f, nz, al), "Q4_differential_training_length": hi_mean(q4f, nz, al)}
    names = ["PI_tuned", "PI_tuned_AW", "ARX_fixed", "ARX_RLS", "ARX_gain"] + P["network_arms"]
    out["S5_mean_rmse"] = {c["key"]: {nm: float(np.mean([val(r, nm) for r in raw[c["key"]]])) for nm in names} for c in conditions(P)}
    out["decisions"] = {
        "Q1_supported": bool(out["Q1_gain_fraction"]["ci"][0] >= A["Q1"]["threshold"]),
        "Q3_supported": bool(out["Q3_gain_input_vs_10k"]["ci"][0] > 0),
        "Q4_supported_differential_effect": bool(out["Q4_differential_training_length"]["ci"][0] > 0),
        "secondary": {"S_Q2_lower_bound_ge_0.90": bool(out["S_Q2_gain_spearman"]["ci"][0] >= A["S_Q2"]["threshold"]),
                      "S_Q4_opposite_directions": out["S_Q4_opposite_directions"],
                      "note": "secondary and robustness results are reported with their stated levels; they do not change primary decisions"}}
    return out


def manifest_files(P):
    files = ["confirmation_v4.py", "confirmation_v3.py", "PROTOCOL_V4.json", "v02_robustness.py", "experiment.py", "experiment_b_final.py",
             P["arx_prior"]["path"]] + [f["path"] for v in P["models"].values() for f in v]
    return {f: sha256(ROOT / f) for f in files}


def block(seeds, P, models, prior, partial, budget):
    raw = json.loads(partial.read_text()) if partial.exists() else {}; t0 = time.time(); total = len(seeds) * len(conditions(P))
    for cond in conditions(P):
        lst = raw.setdefault(cond["key"], [])
        while len(lst) < len(seeds):
            if time.time() - t0 > budget:
                partial.write_text(json.dumps(raw)); print(f"BUDGET REACHED - resumable; {sum(len(v) for v in raw.values())}/{total} plant-conditions done", flush=True); return None
            lst.append(run_plant(seeds[len(lst)], cond, P, models, prior))
        partial.write_text(json.dumps(raw)); print(cond["key"], "complete", flush=True)
    return raw


def main(mode, budget):
    P = json.loads((ROOT / "PROTOCOL_V4.json").read_text()); models = load_models(P["models"])
    pz = ROOT / P["arx_prior"]["path"]
    if sha256(pz) != P["arx_prior"]["sha256"]: raise RuntimeError("ARX prior hash mismatch")
    prior = dict(np.load(pz)); out = ROOT / "results_v4"; out.mkdir(exist_ok=True)
    print("mode:", mode, "| status:", P["status"], "| script sha256:", sha256(Path(__file__)))
    if mode == "audit":
        t0 = time.time(); r = run_plant(P["audit_smoke_plant"], conditions(P)[-1], P, models, prior)
        print("SMOKE plant", P["audit_smoke_plant"], "cond", conditions(P)[-1]["key"], round(time.time() - t0, 1), "s:", {n: round(val(r, n), 3) for n in r})
        (out / "audit_v4.json").write_text(json.dumps({"file_hashes": manifest_files(P), "smoke": r}, indent=1)); print("AUDIT ONLY - no confirmation plant instantiated."); return
    if mode == "selftest":
        raw = block(P["self_test_plants"], P, models, prior, out / "selftest_partial.json", budget)
        if raw is None: return
        A = analyse(raw, P); (out / "selftest_v4.json").write_text(json.dumps({"note": "n=3 development plants; not interpreted", "analysis": A}, indent=1))
        print("SELF-TEST COMPLETE", json.dumps(A["decisions"])); return
    if P["status"] != "FROZEN": raise SystemExit("Protocol status is not FROZEN; refusing to touch confirmation plants.")
    if not MANIFEST.exists(): raise SystemExit("MANIFEST_V4.json missing.")
    man = json.loads(MANIFEST.read_text()); now = manifest_files(P)
    bad = [f for f in set(man["files"]) | set(now) if man["files"].get(f) != now.get(f)]
    if bad: raise SystemExit(f"Files differ from frozen manifest: {bad}")
    raw = block(list(range(*P["confirmation"]["plant_seeds_range"])), P, models, prior, out / "confirmation_v4_raw_partial.json", budget)
    if raw is None: return
    res = {"warning": f"Plants {P['confirmation']['plant_seeds']} are SPENT.", "manifest": man, "protocol": P, "analysis": analyse(raw, P), "raw": raw}
    (out / "confirmation_v4_results.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res["analysis"]["decisions"], indent=1)); print("CONFIRMATION COMPLETE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); g = ap.add_mutually_exclusive_group()
    g.add_argument("--audit-only", action="store_true"); g.add_argument("--self-test", action="store_true")
    ap.add_argument("--budget", type=float, default=float("inf")); a = ap.parse_args()
    main("audit" if a.audit_only else "selftest" if a.self_test else "confirm", a.budget)
