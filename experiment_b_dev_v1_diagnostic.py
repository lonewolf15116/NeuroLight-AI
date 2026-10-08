"""B-Dev-v1 diagnostic: cold-start RLS-ARX excitation. DEVELOPMENT ONLY."""
from pathlib import Path
import argparse, json
import numpy as np
from v02_robustness import ShiftedCircuit

H=4; SCALE=60.; SENS=[.60,.80,1.00,1.20]
SEEDS=range(5100,5150)
TARGETS=np.repeat([12.,30.,20.,40.,15.,35.],30)
CAND=np.linspace(0,1,101)
LAM=.98; P0=10.

class RLS:
    def __init__(self):
        self.w=np.zeros(9); self.P=np.eye(9)*P0
    def predict(self,z): return float(np.asarray(z)@self.w)
    def update(self,z,y):
        z=np.asarray(z,float); Pz=self.P@z
        k=Pz/(LAM+z@Pz)
        e=float(y-z@self.w)
        self.w += k*e
        self.P=(self.P-np.outer(k,z)@self.P)/LAM
        self.P=.5*(self.P+self.P.T)
        return e

def zvec(rh,ah,u):
    return np.array([u,rh[-1]/SCALE,rh[-2]/SCALE,rh[-3]/SCALE,rh[-4]/SCALE,
                     ah[-1],ah[-2],ah[-3],1.])

def diagnose(seed,sens,trace=False):
    c=ShiftedCircuit(seed,noise=.5,sensitivity_scale=sens,bias_shift=0.)
    q=RLS(); rh=[0.]*H; ah=[0.]*H; rates=[]; lights=[]; tr=[]; first=None
    for t,target in enumerate(TARGETS):
        preds=np.array([q.predict(zvec(rh,ah,u))*SCALE for u in CAND])
        costs=(preds-target)**2+2*CAND**2+2*(CAND-ah[-1])**2
        i=int(np.argmin(costs)); u=float(CAND[i])
        if first is None and u>1e-12: first=t
        z=zvec(rh,ah,u); pred=float(q.predict(z)*SCALE)
        b0=float(q.w[0]); pb0=float(q.P[0,0])
        r=float(c.step(u)); e=q.update(z,r/SCALE)
        if trace:
            tr.append(dict(t=t,target_hz=float(target),rate_before_hz=float(rh[-1]),
                action=u,predicted_next_hz=pred,observed_next_hz=r,
                prediction_error_normalized=e,b0_before=b0,b0_after=float(q.w[0]),
                P_b0_before=pb0,P_b0_after=float(q.P[0,0]),
                candidate_prediction_range_hz=float(preds.max()-preds.min())))
        rates.append(r); lights.append(u); rh=rh[1:]+[r]; ah=ah[1:]+[u]
    rates=np.asarray(rates); lights=np.asarray(lights); nz=lights>1e-12
    return dict(rmse_hz=float(np.sqrt(np.mean((rates-TARGETS)**2))),
        fraction_zero=float(np.mean(~nz)),nonzero_action_count=int(nz.sum()),
        distinct_action_count=int(len(np.unique(lights))),
        mean_intensity=float(lights.mean()),max_intensity=float(lights.max()),
        first_nonzero_timestep=first,final_b0=float(q.w[0]),
        final_P_b0=float(q.P[0,0]),final_parameters=[float(x) for x in q.w],
        trace=tr if trace else None)

def main(out):
    out.mkdir(parents=True,exist_ok=True)
    result={"status":"DEVELOPMENT DIAGNOSTIC ONLY",
      "hypothesis":"Cold-start zero coefficients induce zero action, preventing b0 identification.",
      "frozen_rls":{"forgetting":LAM,"P0":P0},
      "seeds":"5100-5149 (already spent in B-Dev-v1)","conditions":{}}
    for sens in SENS:
        raw=[diagnose(s,sens,trace=(s==5100)) for s in SEEDS]
        result["conditions"][str(sens)]={
          "mean_rmse_hz":float(np.mean([x["rmse_hz"] for x in raw])),
          "mean_fraction_zero":float(np.mean([x["fraction_zero"] for x in raw])),
          "plants_with_no_nonzero_action":int(sum(x["nonzero_action_count"]==0 for x in raw)),
          "mean_nonzero_action_count":float(np.mean([x["nonzero_action_count"] for x in raw])),
          "mean_distinct_action_count":float(np.mean([x["distinct_action_count"] for x in raw])),
          "mean_final_b0":float(np.mean([x["final_b0"] for x in raw])),
          "mean_final_P_b0":float(np.mean([x["final_P_b0"] for x in raw])),
          "representative_seed":5100,"representative":raw[0],
          "per_plant":[{k:v for k,v in x.items() if k!="trace"} for x in raw]}
    (out/"experiment_b_dev_v1_diagnostic.json").write_text(json.dumps(result,indent=2)+"\n")
    print("B-DEV-v1 DIAGNOSTIC ONLY -- no new seeds touched")
    for sens in SENS:
        x=result["conditions"][str(sens)]
        print(f"sensitivity {sens:.2f}: RMSE {x['mean_rmse_hz']:.4f} | zero fraction {x['mean_fraction_zero']:.4f} | no-nonzero {x['plants_with_no_nonzero_action']}/50 | b0 {x['mean_final_b0']:.6f} | P00 {x['mean_final_P_b0']:.3f}")

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--out",type=Path,default=Path("results_experiment_b_dev_v1_diagnostic"))
    main(p.parse_args().out)
