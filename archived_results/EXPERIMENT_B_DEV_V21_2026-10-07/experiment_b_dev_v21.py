"""NeuroLight-AI B-Dev-v2.1: dynamic-observer adaptive PI.

LAST ADAPTIVE-CONTROLLER DEVELOPMENT STAGE.
Uses only already-spent tuning seeds 5000-5099 and validation seeds 5100-5149.
Loads the frozen B-Dev-v1.2 DR-ARX prior. No predictive action search.
No future confirmation seeds are declared or touched.
"""
from pathlib import Path
import argparse, json, hashlib
import numpy as np
from v02_robustness import ShiftedCircuit

SCALE=60.
SENS=[.60,.80,1.00,1.20]
TUNE=range(5000,5100)
VAL=range(5100,5150)
TARGETS=np.repeat([12.,30.,20.,40.,15.,35.],30)
KP0=.02
KI0=.15
LAM=.98
P_SCALE=1.0
# Small predeclared scheduling grid. alpha=0 is fixed PI and is included only
# as an internal sanity/reference point; the adaptive candidates are >0.
ALPHAS=[.25,.50,.75,1.00]
RATIO_MIN=.25
RATIO_MAX=4.0
EXPECTED_PRIOR_SHA="9da07cf88f96402d9faf2d1061f2d31acb0ae916fad42a41123785ca11a31797"

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def load_prior(path):
    got=sha256(path)
    if got != EXPECTED_PRIOR_SHA:
        raise RuntimeError(f"Frozen prior SHA mismatch: {got}")
    d=np.load(path)
    if set(d.files)!={"w0","covariance"}:
        raise RuntimeError(f"Unexpected prior keys: {d.files}")
    w=np.asarray(d["w0"],float)
    cov=np.asarray(d["covariance"],float)
    if w.shape!=(9,) or cov.shape!=(9,9):
        raise RuntimeError("Unexpected frozen prior dimensions")
    return w,cov,got

def phi(candidate,rh,ah):
    # EXACT frozen v1.2 information set:
    # candidate u_t + 4 rate histories + 3 most recent action histories + intercept.
    # It intentionally retains v1.2's omission of the oldest fourth action.
    return np.asarray([candidate,
        rh[-1]/SCALE,rh[-2]/SCALE,rh[-3]/SCALE,rh[-4]/SCALE,
        ah[-1],ah[-2],ah[-3],1.],float)

def metrics(rates,lights):
    r=np.asarray(rates); u=np.asarray(lights)
    return {"rmse_hz":float(np.sqrt(np.mean((r-TARGETS)**2))),
            "mean_intensity":float(u.mean()),
            "fraction_zero":float(np.mean(u<=1e-12)),
            "fraction_one":float(np.mean(u>=1-1e-12)),
            "mean_abs_action_change":float(np.mean(np.abs(np.diff(u))))}

def run_fixed(seed,sens):
    c=ShiftedCircuit(seed,noise=.5,sensitivity_scale=sens,bias_shift=0.)
    I=0.; rates=[]; lights=[]
    for target in TARGETS:
        e=target-c.rate
        I=float(np.clip(I+e*.05,-10,10))
        u=float(np.clip(KP0*e+KI0*I,0,1))
        rates.append(float(c.step(u))); lights.append(u)
    return metrics(rates,lights)

def run_dynamic_pi(seed,sens,alpha,w0,cov0):
    c=ShiftedCircuit(seed,noise=.5,sensitivity_scale=sens,bias_shift=0.)
    w=w0.copy()
    # Same v1.2 covariance-prior convention.
    P=P_SCALE*(cov0+1e-12*np.eye(9))
    b_nom=float(w0[0])
    rh=[0.]*4; ah=[0.]*4
    # Initialize observable rate history consistently from current plant rate.
    rh=[float(c.rate)]*4
    u_prev=0.
    e_prev=TARGETS[0]/SCALE-c.rate/SCALE

    rates=[]; lights=[]; btrace=[]; ratio_trace=[]; mult_trace=[]
    pred_err=[]; sat=0; aw=0; ratio_clamp=0

    for target_hz in TARGETS:
        y=c.rate/SCALE
        e=target_hz/SCALE-y

        # Observer-only scheduling statistic: current direct input coefficient.
        raw_ratio=float(w[0]/b_nom)
        ratio=float(np.clip(raw_ratio,RATIO_MIN,RATIO_MAX))
        ratio_clamp += int(ratio != raw_ratio)

        # alpha controls scheduling aggressiveness; alpha=1 is full reciprocal.
        multiplier=float(ratio**(-alpha))
        kp=KP0*multiplier
        ki=KI0*multiplier

        p_inc=kp*(e-e_prev)
        i_inc=ki*e
        v=u_prev+p_inc+i_inc
        u=float(np.clip(v,0.,1.))
        if u != v:
            sat += 1
            if (v>1. and i_inc>0.) or (v<0. and i_inc<0.):
                aw += 1
                u=float(np.clip(u_prev+p_inc,0.,1.))

        x=phi(u,rh,ah)
        yhat=float(x@w)
        yn=float(c.step(u))/SCALE
        err=yn-yhat

        # Standard RLS update with frozen lambda=.98.
        Px=P@x
        den=LAM+float(x@Px)
        K=Px/den
        w=w+K*err
        P=(P-np.outer(K,x)@P)/LAM
        P=.5*(P+P.T)

        rates.append(yn*SCALE); lights.append(u)
        btrace.append(float(w[0]))
        ratio_trace.append(float(w[0]/b_nom))
        mult_trace.append(multiplier); pred_err.append(err*SCALE)

        rh=(rh+[yn*SCALE])[-4:]
        ah=(ah+[u])[-4:]
        u_prev=u; e_prev=e

    out=metrics(rates,lights)
    bt=np.asarray(btrace); rt=np.asarray(ratio_trace); pe=np.asarray(pred_err)
    out.update({
        "observer_prediction_rmse_hz":float(np.sqrt(np.mean(pe**2))),
        "b0_mean":float(bt.mean()),"b0_final":float(bt[-1]),
        "gain_ratio_mean":float(rt.mean()),"gain_ratio_final":float(rt[-1]),
        "gain_ratio_rmse_vs_true_sensitivity":float(np.sqrt(np.mean((rt-sens)**2))),
        "gain_ratio_clamp_rate":float(ratio_clamp/len(TARGETS)),
        "control_saturation_attempt_rate":float(sat/len(TARGETS)),
        "anti_windup_activation_rate":float(aw/len(TARGETS)),
        "b0_trace":bt.tolist(),"gain_ratio_trace":rt.tolist(),
        "gain_multiplier_trace":mult_trace,
        "rate_trace_hz":rates,"light_trace":lights
    })
    return out

def evaluate(fn,seeds,sens):
    raw=[fn(seed,sens) for seed in seeds]
    scalar=[k for k,v in raw[0].items() if np.isscalar(v)]
    return {"mean":{k:float(np.mean([q[k] for q in raw])) for k in scalar},
            "std":{k:float(np.std([q[k] for q in raw])) for k in scalar},
            "raw":raw}

def main(out,prior):
    out.mkdir(parents=True,exist_ok=True)
    w0,cov0,prior_sha=load_prior(prior)

    payload={
      "warning":"B-DEV-v2.1 DEVELOPMENT ONLY; last adaptive-controller development stage; no confirmation seeds declared/touched.",
      "prior_path":str(prior),"prior_sha256":prior_sha,
      "prior_b0":float(w0[0]),"rls":{"lambda":LAM,"p_scale":P_SCALE},
      "tuning_seeds":"5000-5099","validation_seeds":"5100-5149",
      "sensitivities":SENS,
      "information_set":"frozen v1.2: candidate + 4 rates + 3 past actions + intercept; oldest fourth action omitted",
      "scheduling":{"alphas":ALPHAS,"ratio_bounds":[RATIO_MIN,RATIO_MAX],
                    "law":"multiplier=(clip(b0/b0_nom))**(-alpha)"},
      "selection_rule":"one global alpha minimizing equal-weight mean tuning RMSE across all four sensitivities"
    }

    # Observer validity diagnostic is evaluated for every candidate because
    # actions differ with alpha. True sensitivity is NEVER used by controller.
    search=[]
    for order,alpha in enumerate(ALPHAS):
        by=[]; rms=[]
        for sens in SENS:
            z=evaluate(lambda seed,s,a=alpha:run_dynamic_pi(seed,s,a,w0,cov0),TUNE,sens)
            m=z["mean"]; rms.append(m["rmse_hz"])
            by.append({"sensitivity":sens,"rmse_hz":m["rmse_hz"],
                       "observer_prediction_rmse_hz":m["observer_prediction_rmse_hz"],
                       "gain_ratio_mean":m["gain_ratio_mean"],
                       "gain_ratio_rmse":m["gain_ratio_rmse_vs_true_sensitivity"],
                       "ratio_clamp_rate":m["gain_ratio_clamp_rate"]})
        search.append({"order":order,"alpha":alpha,
                       "mean_tuning_rmse_hz":float(np.mean(rms)),
                       "by_sensitivity":by})

    best=min(search,key=lambda q:(q["mean_tuning_rmse_hz"],q["order"]))
    payload["search"]=sorted(search,key=lambda q:q["mean_tuning_rmse_hz"])
    payload["choice"]={"alpha":best["alpha"],
                       "mean_tuning_rmse_hz":best["mean_tuning_rmse_hz"]}

    payload["validation"]={}
    gain_means=[]
    for sens in SENS:
        fixed=evaluate(run_fixed,VAL,sens)
        adapt=evaluate(lambda seed,s:run_dynamic_pi(seed,s,best["alpha"],w0,cov0),VAL,sens)
        f=np.asarray([q["rmse_hz"] for q in fixed["raw"]])
        a=np.asarray([q["rmse_hz"] for q in adapt["raw"]])
        payload["validation"][f"{sens:.2f}"]={
          "fixed_pi":fixed,"dynamic_adaptive_pi":adapt,
          "adaptive_minus_fixed_mean_hz":float(np.mean(a-f)),
          "adaptive_better_fraction":float(np.mean(a<f))}
        gain_means.append(adapt["mean"]["gain_ratio_mean"])

    # Predeclared observer sanity diagnostic: does average b0/b0_nom order
    # monotonically with true sensitivity over the four validation conditions?
    payload["observer_sanity"]={
      "mean_gain_ratios_by_sensitivity":gain_means,
      "strictly_monotonic_increasing":bool(all(gain_means[i]<gain_means[i+1] for i in range(3))),
      "note":"diagnostic only; true sensitivity was not available to controller"
    }

    (out/"experiment_b_dev_v21.json").write_text(json.dumps(payload,indent=2)+"\n")
    print("B-DEV-v2.1 ONLY -- last adaptive-controller development stage")
    print("No future confirmation seeds declared or touched")
    print(f"frozen prior SHA: {prior_sha}")
    print(f"prior b0: {w0[0]:.9f}")
    print("choice:",payload["choice"])
    for sens in SENS:
        q=payload["validation"][f"{sens:.2f}"]
        a=q["dynamic_adaptive_pi"]["mean"]; f=q["fixed_pi"]["mean"]
        print(f"sensitivity {sens:.2f}: dynamic adaptive PI {a['rmse_hz']:.4f} | fixed {f['rmse_hz']:.4f} | "
              f"observer RMSE {a['observer_prediction_rmse_hz']:.4f} | "
              f"mean gain ratio {a['gain_ratio_mean']:.4f} | ratio RMSE {a['gain_ratio_rmse_vs_true_sensitivity']:.4f} | "
              f"ratio clamp {a['gain_ratio_clamp_rate']:.3f}")
    print("observer monotonic:",payload["observer_sanity"]["strictly_monotonic_increasing"])

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=Path("results_experiment_b_dev_v21"))
    p.add_argument("--prior",type=Path,default=Path(r"results_experiment_b_dev_v12\dr_arx_prior.npz"))
    a=p.parse_args(); main(a.out,a.prior)
