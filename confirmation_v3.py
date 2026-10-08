"""NeuroLight-AI prospective confirmation v3 (Study phase 2).

ALL parameters are read from PROTOCOL_V3.json. Nothing in this file may be tuned.

Usage
  python confirmation_v3.py --audit-only   # hashes, prior check, smoke test on ONE development plant (5100).
                                           # Does NOT instantiate any confirmation plant.
  python confirmation_v3.py --dev-rehearsal [--budget S]   # full pipeline on development plants 5100-5149.
  python confirmation_v3.py [--budget S]   # one-shot confirmation (requires FROZEN + matching MANIFEST_V3.json),
                                           # resumable per plant-condition; plants in protocol["confirmation"]["plant_seeds"]
                                           # are SPENT after the first completed run, regardless of outcome.

Simulator: the original scalar experiment.Circuit via v02_robustness.ShiftedCircuit (exact, per-plant RNG),
not the vectorised development copy.
"""
from pathlib import Path
import argparse, json, hashlib, time
import numpy as np
from v02_robustness import ShiftedCircuit
import experiment_b_final as E

ROOT = Path(__file__).resolve().parent
H = 4; SCALE = 60.
TARGETS = np.repeat([12., 30., 20., 40., 15., 35.], 30)
CAND = np.linspace(0., 1., 101)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def metric(rates, lights):
    r = np.asarray(rates); u = np.asarray(lights)
    return {"rmse": float(np.sqrt(np.mean((r - TARGETS) ** 2))),
            "rmse_t50": float(np.sqrt(np.mean((r[50:] - TARGETS[50:]) ** 2))),
            "mean_intensity": float(u.mean()),
            "frac_bound": float(np.mean((u <= 1e-12) | (u >= 1 - 1e-12))),
            "slew": float(np.mean(np.abs(np.diff(u))))}


# ---------------------------------------------------------------- controllers
def run_pi(seed, s, noise, kp, ki, kt=0.0):
    """Positional PI with integrator clip +/-10. kt > 0 adds back-calculation anti-windup:
    I <- clip(I + 0.05 e + kt (u_sat - u_raw) / Ki, -10, 10). kt = 0 reproduces the legacy PI exactly."""
    c = ShiftedCircuit(seed, noise=noise, sensitivity_scale=s, bias_shift=0.)
    I = 0.; R = []; U = []
    for t in TARGETS:
        e = t - c.rate
        if kt == 0.0:
            I = float(np.clip(I + .05 * e, -10, 10)); u = float(np.clip(kp * e + ki * I, 0, 1))
        else:
            I = I + .05 * e; ur = kp * e + ki * I; u = float(np.clip(ur, 0, 1))
            I = float(np.clip(I + kt * (u - ur) / ki, -10, 10))
        R.append(float(c.step(u))); U.append(u)
    return metric(R, U)


class RLS:
    def __init__(self, w0, cov0, lam, p_scale, adapt):
        self.w = w0.copy(); self.P = p_scale * (cov0 + 1e-12 * np.eye(10)); self.lam = lam; self.adapt = adapt
    def update(self, z, y):
        e = float(y - z @ self.w)
        if self.adapt:
            Pz = self.P @ z; k = Pz / (self.lam + float(z @ Pz))
            self.w = self.w + k * e
            self.P = (self.P - np.outer(k, z) @ self.P) / self.lam; self.P = .5 * (self.P + self.P.T)
        return e


def run_arx(seed, s, noise, w0, cov0, lam, p_scale, adapt):
    c = ShiftedCircuit(seed, noise=noise, sensitivity_scale=s, bias_shift=0.)
    q = RLS(w0, cov0, lam, p_scale, adapt); rh = [0.] * H; ah = [0.] * H; R = []; U = []; errs = []
    for t in TARGETS:
        Z = np.stack([E.zvec(rh, ah, u) for u in CAND]); pred = Z @ q.w * SCALE
        cost = (pred - t) ** 2 + 2 * CAND ** 2 + 2 * (CAND - ah[-1]) ** 2
        i = int(np.argmin(cost)); u = float(CAND[i]); y = float(c.step(u))
        errs.append(q.update(Z[i], y / SCALE)); R.append(y); U.append(u)
        rh = (rh + [y])[-H:]; ah = (ah + [u])[-H:]
    o = metric(R, U)
    o.update(max_coef_drift=float(np.max(np.abs(q.w - w0))), pred_change_bound_hz=float(SCALE * np.sum(np.abs(q.w - w0))),
             innov_mae_hz=float(SCALE * np.mean(np.abs(errs))), finite=bool(np.all(np.isfinite(q.w))))
    return o


def predict(m, x):
    return (np.tanh(x @ m["w1"] + m["b1"]) @ m["w2"] + m["b2"]).reshape(-1)


def run_h4(m, seed, s, noise, shuffle=False):
    c = ShiftedCircuit(seed, noise=noise, sensitivity_scale=s, bias_shift=0.)
    rh = [0.] * H; ah = [0.] * H; R = []; U = []; rng = np.random.default_rng(seed + 91000)
    for t in TARGETS:
        rr = np.asarray(rh); aa = np.asarray(ah)
        if shuffle:
            p = rng.permutation(H); rr = rr[p]; aa = aa[p]
        X = np.column_stack([np.tile(np.r_[rr / SCALE, aa], (101, 1)), CAND])
        cost = (predict(m, X) * SCALE - t) ** 2 + 2 * CAND ** 2 + 2 * (CAND - ah[-1]) ** 2
        u = float(CAND[np.argmin(cost)]); y = float(c.step(u)); R.append(y); U.append(u)
        rh = (rh + [y])[-H:]; ah = (ah + [u])[-H:]
    return metric(R, U)


def run_mem(m, seed, s, noise):
    c = ShiftedCircuit(seed, noise=noise, sensitivity_scale=s, bias_shift=0.)
    prev = 0.; R = []; U = []
    for t in TARGETS:
        X = np.column_stack([np.full(101, c.rate / SCALE), np.full(101, prev), CAND])
        cost = (predict(m, X) * SCALE - t) ** 2 + 2 * CAND ** 2 + 2 * (CAND - prev) ** 2
        u = float(CAND[np.argmin(cost)]); R.append(float(c.step(u))); U.append(u); prev = u
    return metric(R, U)


# ---------------------------------------------------------------- run
def load_models(spec):
    out = {}
    for name, files in spec.items():
        out[name] = []
        for f in files:
            p = ROOT / f["path"]
            if sha256(p) != f["sha256"]: raise RuntimeError(f"hash mismatch {p}")
            z = np.load(p); out[name].append({k: z[k] for k in ("w1", "b1", "w2", "b2")})
    return out


def conditions(P):
    c = [{"key": f"s{s:.2f}", "s": s, "noise": P["noise_nominal"]} for s in P["sensitivities"]]
    c += [{"key": f"n{n:.1f}", "s": 1.0, "noise": n} for n in P["noise_conditions"]]
    return c


def run_plant(seed, cond, P, models, prior):
    s, nz = cond["s"], cond["noise"]; C = P["controllers"]; rec = {}
    rec["PI_legacy"] = run_pi(seed, s, nz, **C["PI_legacy"])
    rec["PI_tuned"] = run_pi(seed, s, nz, **C["PI_tuned"])
    rec["PI_tuned_AW"] = run_pi(seed, s, nz, **C["PI_tuned_AW"])
    if cond["key"] in C["PI_sens_ref"]:   # sensitivity-informed tuned PI reference (uses knowledge of s)
        rec["PI_sens_ref"] = run_pi(seed, s, nz, **C["PI_sens_ref"][cond["key"]])
    rec["ARX_fixed"] = run_arx(seed, s, nz, prior["w0"], prior["cov0"], adapt=False, **C["ARX_fixed"])
    rec["ARX_RLS"] = run_arx(seed, s, nz, prior["w0"], prior["cov0"], adapt=True, **C["ARX_RLS"])
    rec["DR_H4"] = [run_h4(m, seed, s, nz) for m in models["DR_H4"]]
    rec["DR_H4_900"] = [run_h4(m, seed, s, nz) for m in models["DR_H4_900"]]   # secondary arm (S5); never primary
    rec["DR_Mem141"] = [run_mem(m, seed, s, nz) for m in models["DR_Mem141"]]
    if cond["key"] in P["chronology_conditions"]:
        rec["DR_H4_jointshuffle"] = [run_h4(m, seed, s, nz, shuffle=True) for m in models["DR_H4"]]
    return rec


def plant_value(rec, name, field="rmse"):
    v = rec[name]
    return float(np.mean([q[field] for q in v])) if isinstance(v, list) else float(v[field])


def analyse(raw, P):
    """raw[cond_key] = list over plants of rec dicts. Returns all pre-specified contrasts."""
    out = {}; k = 0
    def ci(a, b, cond, alpha, tag):
        nonlocal k
        d = np.asarray([plant_value(r, a) - plant_value(r, b) for r in raw[cond]])
        k += 1
        res = E.bca_mean_ci(d, seed=700000 + k, nboot=P["statistics"]["resamples"], alpha=alpha)
        res.update(contrast=f"{a} - {b}", condition=cond, alpha=alpha, ci_level=1 - alpha, ci=res.pop("bca95"), positive_fraction=float(np.mean(d > 0)), tag=tag)
        return res
    A = P["analysis"]
    out["P1_history"] = [ci("DR_Mem141", "DR_H4", c, A["P1"]["alpha_each"], "P1") for c in A["P1"]["conditions"]]
    out["P2_linear_sufficiency"] = [ci("ARX_fixed", "DR_H4", c, A["P2"]["alpha"], "P2") for c in A["P2"]["conditions"]]
    allc = [c["key"] for c in conditions(P)]
    out["B1_PI_benchmark"] = [ci(pi, "DR_H4", c, A["B1"]["alpha_each"], "B1") for pi in ("PI_tuned", "PI_tuned_AW") for c in allc]
    out["S1_adaptation"] = [ci("ARX_RLS", "ARX_fixed", c, A["S1"]["alpha_each"], "S1") for c in allc]
    out["S3_chronology"] = [ci("DR_H4_jointshuffle", "DR_H4", c, A["S3"]["alpha_each"], "S3") for c in P["chronology_conditions"]]
    out["S4_adaptive_linear_vs_H4"] = [ci("ARX_RLS", "DR_H4", c, A["S4"]["alpha_each"], "S4") for c in allc]
    out["S5_training_length"] = [ci("DR_H4_900", "DR_H4", c, A["S5"]["alpha_each"], "S5") for c in allc]
    # S2 regime dependence: mean RMSE table and per-condition ranking (descriptive)
    names = ["PI_legacy", "PI_tuned", "PI_tuned_AW", "PI_sens_ref", "ARX_fixed", "ARX_RLS", "DR_H4", "DR_H4_900", "DR_Mem141"]
    tab = {}
    for c in allc:
        tab[c] = {n: float(np.mean([plant_value(r, n) for r in raw[c]])) for n in names if n in raw[c][0]}
        tab[c]["_ranking_excl_sens_ref"] = sorted([n for n in tab[c] if n != "PI_sens_ref"], key=lambda n: tab[c][n])
    out["S2_mean_rmse_and_ranking"] = tab
    # decisions
    P1 = out["P1_history"]; P2 = out["P2_linear_sufficiency"][0]; m = P["equivalence_margin_hz"]
    out["decisions"] = {
        "P1_supported": all(r["ci"][0] > 0 for r in P1),
        "P2_supported": bool(P2["ci"][0] >= -m and P2["ci"][1] <= m),
        "B1": "reported, no directional hypothesis",
        "S1_S5": "secondary/exploratory; reported with stated alpha, no decision rule; S5 never promoted to primary",
    }
    return out


MANIFEST = ROOT / "MANIFEST_V3.json"


def manifest_files(P):
    """Every file whose content determines the confirmation result."""
    files = ["confirmation_v3.py", "PROTOCOL_V3.json", "v02_robustness.py", "experiment.py", "experiment_b_final.py",
             P["arx_prior"]["path"]] + [f["path"] for v in P["models"].values() for f in v]
    return {f: sha256(ROOT / f) for f in files}


def run_block(seeds, P, models, prior, partial, budget):
    raw = json.loads(partial.read_text()) if partial.exists() else {}
    t0 = time.time(); total = len(seeds) * len(conditions(P))
    for cond in conditions(P):
        lst = raw.setdefault(cond["key"], [])
        while len(lst) < len(seeds):
            if time.time() - t0 > budget:
                partial.write_text(json.dumps(raw))
                print(f"BUDGET REACHED - resumable; {sum(len(v) for v in raw.values())}/{total} plant-conditions done", flush=True)
                return None
            lst.append(run_plant(seeds[len(lst)], cond, P, models, prior))
        partial.write_text(json.dumps(raw)); print(cond["key"], "complete", flush=True)
    return raw


def main(mode, budget=float("inf")):
    P = json.loads((ROOT / "PROTOCOL_V3.json").read_text())
    models = load_models(P["models"])
    pz = ROOT / P["arx_prior"]["path"]
    if sha256(pz) != P["arx_prior"]["sha256"]: raise RuntimeError("ARX prior hash mismatch")
    prior = dict(np.load(pz))
    script_hash = sha256(Path(__file__))
    print("mode:", mode, "| protocol status:", P["status"], "| script sha256:", script_hash)
    print("models:", {k: len(v) for k, v in models.items()}, "conditions:", [c["key"] for c in conditions(P)])
    outdir = ROOT / "results_v3"; outdir.mkdir(exist_ok=True)

    if mode == "audit":
        t0 = time.time(); smoke = {}
        for cond in conditions(P)[:1] + conditions(P)[-1:]:
            smoke[cond["key"]] = run_plant(P["audit_smoke_plant"], cond, P, models, prior)
        print("SMOKE (development plant", P["audit_smoke_plant"], ") seconds:", round(time.time() - t0, 1))
        for k, r in smoke.items():
            print(k, {n: round(plant_value(r, n), 3) for n in r})
        (outdir / "audit_v3.json").write_text(json.dumps({"script_sha256": script_hash, "file_hashes": manifest_files(P),
                                                          "protocol": P, "smoke": smoke}, indent=1))
        print("AUDIT ONLY - no confirmation plant instantiated."); return

    if mode == "rehearsal":   # full pipeline on DEVELOPMENT evaluation plants; allowed while DRAFT
        seeds = list(range(*P["development_seeds"]["rehearsal_range"]))
        raw = run_block(seeds, P, models, prior, outdir / "dev_rehearsal_raw_partial.json", budget)
        if raw is None: return
        res = {"note": "DEVELOPMENT REHEARSAL on plants %s; not confirmatory." % P["development_seeds"]["rehearsal"],
               "script_sha256": script_hash, "file_hashes": manifest_files(P), "analysis": analyse(raw, P), "raw": raw}
        (outdir / "dev_rehearsal_results.json").write_text(json.dumps(res, indent=1))
        print("REHEARSAL COMPLETE", json.dumps(res["analysis"]["decisions"])); return

    # ---- confirmation
    if P["status"] != "FROZEN":
        raise SystemExit("Protocol status is not FROZEN; refusing to touch confirmation plants.")
    if not MANIFEST.exists():
        raise SystemExit("MANIFEST_V3.json missing; run freeze_v3.py first.")
    man = json.loads(MANIFEST.read_text()); now = manifest_files(P)
    bad = [f for f in set(man["files"]) | set(now) if man["files"].get(f) != now.get(f)]
    if bad: raise SystemExit(f"Files differ from frozen manifest: {bad}")
    seeds = list(range(*P["confirmation"]["plant_seeds_range"]))
    raw = run_block(seeds, P, models, prior, outdir / "confirmation_v3_raw_partial.json", budget)
    if raw is None: return
    res = {"warning": f"Plants {P['confirmation']['plant_seeds']} are SPENT.", "manifest": man,
           "protocol": P, "analysis": analyse(raw, P), "raw": raw}
    (outdir / "confirmation_v3_results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["analysis"]["decisions"], indent=1)); print("CONFIRMATION COMPLETE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(); g.add_argument("--audit-only", action="store_true")
    g.add_argument("--dev-rehearsal", action="store_true", help="run the full pipeline on development plants")
    ap.add_argument("--budget", type=float, default=float("inf"), help="seconds per invocation; progress saved per plant and resumed")
    a = ap.parse_args()
    main("audit" if a.audit_only else "rehearsal" if a.dev_rehearsal else "confirm", a.budget)
