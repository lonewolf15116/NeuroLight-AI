"""NeuroLight-AI Experiment B FINAL confirmation.

Run --audit-only first. Audit mode does NOT instantiate confirmation plants.
After protocol/script/hash review, run once without --audit-only.

Primary endpoint: full 180-step tracking RMSE, matching development.
Secondary endpoint: steps 50:180 (zero-based; transient-excluded).
"""
from pathlib import Path
import argparse, json, hashlib
from statistics import NormalDist
import numpy as np
from v02_robustness import ShiftedCircuit

H=4; SCALE=60.
TRAIN=range(2000,2100)
CONF=range(6000,6050)
SENS=[.60,.80,1.00,1.20]
MODEL_SEEDS=[101,102,103,104,105]
TARGETS=np.repeat([12.,30.,20.,40.,15.,35.],30)
CAND=np.linspace(0.,1.,101)
LAM=.98; P_SCALE=1.0
KP=.02; KI=.15
NBOOT=10000
TRANSIENT_START=50

def sha256(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def arx10_data(seeds,steps=300):
    """Exact Experiment-A DR trajectories; affine linear regressor with exact H4 observable history."""
    Z=[]; Y=[]
    for seed in seeds:
        r=np.random.default_rng(seed+70000)
        c=ShiftedCircuit(seed,noise=float(r.uniform(.5,2)),
                         sensitivity_scale=float(r.uniform(.6,1.4)),
                         bias_shift=float(r.uniform(-2,2)))
        rh=[0.]*H; ah=[0.]*H
        for t in range(steps):
            if t%4==0: light=float(r.uniform())
            Z.append([light,
                      rh[-1]/SCALE,rh[-2]/SCALE,rh[-3]/SCALE,rh[-4]/SCALE,
                      ah[-1],ah[-2],ah[-3],ah[-4],1.])
            nxt=float(c.step(light)); Y.append(nxt/SCALE)
            rh=(rh+[nxt])[-H:]; ah=(ah+[light])[-H:]
    return np.asarray(Z),np.asarray(Y)

def fit_prior(Z,Y):
    w,_,rank,svals=np.linalg.lstsq(Z,Y,rcond=None)
    resid=Y-Z@w; dof=max(1,len(Y)-rank)
    sigma2=float(resid@resid/dof)
    cov=sigma2*np.linalg.pinv(Z.T@Z)
    return w,cov,{"rank":int(rank),"n_rows":int(len(Y)),
      "sigma2_normalized":sigma2,
      "train_rmse_hz":float(np.sqrt(np.mean(resid**2))*SCALE),
      "singular_values":[float(x) for x in svals],
      "w_prior":[float(x) for x in w]}

def zvec(rh,ah,u):
    return np.asarray([u,rh[-1]/SCALE,rh[-2]/SCALE,rh[-3]/SCALE,rh[-4]/SCALE,
                       ah[-1],ah[-2],ah[-3],ah[-4],1.],float)

class RLS:
    def __init__(self,w,cov):
        self.w=w.copy(); self.P=P_SCALE*(cov+1e-12*np.eye(10))
    def pred(self,z): return float(np.asarray(z)@self.w)
    def update(self,z,y):
        z=np.asarray(z,float); Pz=self.P@z
        k=Pz/(LAM+float(z@Pz)); e=float(y-z@self.w)
        self.w += k*e
        self.P=(self.P-np.outer(k,z)@self.P)/LAM
        self.P=.5*(self.P+self.P.T)
        return e

class FrozenH4:
    def __init__(self,p):
        z=np.load(p); self.w1=z["w1"]; self.b1=z["b1"]; self.w2=z["w2"]; self.b2=z["b2"]
        if self.w1.shape!=(9,64): raise RuntimeError(f"Unexpected H4 w1 shape {self.w1.shape}")
    def predict(self,x):
        return np.tanh(np.asarray(x)@self.w1+self.b1)@self.w2+self.b2

def metric_record(rates,lights):
    r=np.asarray(rates); u=np.asarray(lights)
    def rm(a,b): return float(np.sqrt(np.mean((r[a:b]-TARGETS[a:b])**2)))
    return {"rmse_hz_primary_full180":rm(0,len(TARGETS)),
            "rmse_hz_secondary_t50_179":rm(TRANSIENT_START,len(TARGETS)),
            "mean_intensity":float(u.mean()),
            "fraction_zero":float(np.mean(u<=1e-12)),
            "fraction_one":float(np.mean(u>=1-1e-12)),
            "mean_abs_action_change":float(np.mean(np.abs(np.diff(u))))}

def run_pi(seed,sens):
    c=ShiftedCircuit(seed,noise=.5,sensitivity_scale=sens,bias_shift=0.)
    integ=0.; rates=[]; lights=[]
    for target in TARGETS:
        e=target-c.rate
        integ=float(np.clip(integ+e*.05,-10,10))
        u=float(np.clip(KP*e+KI*integ,0,1))
        rates.append(float(c.step(u))); lights.append(u)
    return metric_record(rates,lights)

def run_arx(seed,sens,w0,cov0):
    c=ShiftedCircuit(seed,noise=.5,sensitivity_scale=sens,bias_shift=0.)
    q=RLS(w0,cov0); rh=[0.]*H; ah=[0.]*H; rates=[]; lights=[]; errs=[]
    for target in TARGETS:
        pred=np.asarray([q.pred(zvec(rh,ah,u))*SCALE for u in CAND])
        cost=(pred-target)**2+2*CAND**2+2*(CAND-ah[-1])**2
        u=float(CAND[np.argmin(cost)])
        z=zvec(rh,ah,u); y=float(c.step(u))
        errs.append(q.update(z,y/SCALE)); rates.append(y); lights.append(u)
        rh=(rh+[y])[-H:]; ah=(ah+[u])[-H:]
    o=metric_record(rates,lights)
    o.update(final_parameters=[float(x) for x in q.w],
             mean_abs_identification_error_normalized=float(np.mean(np.abs(errs))))
    return o

def run_h4(model,seed,sens):
    c=ShiftedCircuit(seed,noise=.5,sensitivity_scale=sens,bias_shift=0.)
    rh=[0.]*H; ah=[0.]*H; rates=[]; lights=[]
    for target in TARGETS:
        base=np.asarray(rh,float)/SCALE
        # Exact H4 order from Experiment A: 4 rates, 4 past actions, candidate.
        X=np.column_stack([np.tile(base,(len(CAND),1)),
                           np.tile(np.asarray(ah,float),(len(CAND),1)),CAND])
        pred=np.asarray(model.predict(X)).reshape(-1)*SCALE
        cost=(pred-target)**2+2*CAND**2+2*(CAND-ah[-1])**2
        u=float(CAND[np.argmin(cost)]); y=float(c.step(u))
        rates.append(y); lights.append(u)
        rh=(rh+[y])[-H:]; ah=(ah+[u])[-H:]
    return metric_record(rates,lights)

def bca_mean_ci(x,seed,nboot=NBOOT,alpha=.05):
    """BCa CI for the mean of paired plant-level differences."""
    x=np.asarray(x,float); n=len(x); theta=float(x.mean())
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,n,size=(nboot,n))
    boots=x[idx].mean(axis=1)
    nd=NormalDist()
    prop=float(np.mean(boots<theta))
    prop=min(max(prop,1/(2*nboot)),1-1/(2*nboot))
    z0=nd.inv_cdf(prop)
    jack=np.asarray([(np.delete(x,i)).mean() for i in range(n)])
    jm=jack.mean()
    num=np.sum((jm-jack)**3)
    den=6*(np.sum((jm-jack)**2)**1.5)
    acc=float(num/den) if den>0 else 0.
    qs=[]
    for p in (alpha/2,1-alpha/2):
        z=nd.inv_cdf(p)
        adj=nd.cdf(z0+(z0+z)/(1-acc*(z0+z)))
        qs.append(float(np.clip(adj,0,1)))
    lo,hi=np.quantile(boots,qs)
    return {"mean":theta,"bca95":[float(lo),float(hi)],
            "bootstrap_resamples":nboot,"bootstrap_seed":seed,
            "acceleration":acc,"bias_correction_z0":z0}

def audit(h4_dir,w0,cov0,prior):
    files={}
    for ms in MODEL_SEEDS:
        p=h4_dir/f"DR_H4_seed_{ms}.npz"
        if not p.exists(): raise FileNotFoundError(p)
        FrozenH4(p) # shape/key check
        files[str(ms)]={"path":str(p),"sha256":sha256(p)}
    return {"confirmation_seeds":"6000-6049 (NOT TOUCHED IN AUDIT MODE)",
            "h4_files":files,"arx10_prior_fit":prior,
            "arx10_prior_w":[float(x) for x in w0],
            "arx10_covariance_diag":[float(x) for x in np.diag(cov0)]}

def main(out,h4_dir,audit_only):
    out.mkdir(parents=True,exist_ok=True)
    Z,Y=arx10_data(TRAIN); w0,cov0,prior=fit_prior(Z,Y)
    aud=audit(h4_dir,w0,cov0,prior)
    (out/"experiment_b_final_audit.json").write_text(json.dumps(aud,indent=2)+"\n")
    print("EXPERIMENT-B FINAL AUDIT")
    print("ARX-10 prior rows",prior["n_rows"],"train RMSE",f'{prior["train_rmse_hz"]:.4f}')
    for k,v in aud["h4_files"].items(): print("H4 seed",k,v["sha256"])
    if audit_only:
        print("AUDIT ONLY -- confirmation seeds 6000-6049 were NOT instantiated.")
        return

    models=[(ms,FrozenH4(h4_dir/f"DR_H4_seed_{ms}.npz")) for ms in MODEL_SEEDS]
    result={"warning":"Seeds 6000-6049 are SPENT after this first completed confirmation run.",
            "primary_endpoint":"full 180-step tracking RMSE (Hz)",
            "secondary_endpoint":"tracking RMSE steps 50-179",
            "sensitivities":SENS,"plant_seeds":[int(x) for x in CONF],
            "model_seeds":MODEL_SEEDS,"audit":aud,"conditions":{}}

    for si,sens in enumerate(SENS):
        key=f"{sens:.2f}"
        pi=[run_pi(seed,sens) for seed in CONF]
        arx=[run_arx(seed,sens,w0,cov0) for seed in CONF]
        h4_by_model=[]
        for ms,m in models:
            h4_by_model.append([run_h4(m,seed,sens) for seed in CONF])

        p=np.asarray([q["rmse_hz_primary_full180"] for q in pi])
        a=np.asarray([q["rmse_hz_primary_full180"] for q in arx])
        hm=np.asarray([[q["rmse_hz_primary_full180"] for q in row] for row in h4_by_model])
        hp=hm.mean(axis=0)
        dpi=p-hp; darx=a-hp

        sec_p=np.asarray([q["rmse_hz_secondary_t50_179"] for q in pi])
        sec_a=np.asarray([q["rmse_hz_secondary_t50_179"] for q in arx])
        sec_hm=np.asarray([[q["rmse_hz_secondary_t50_179"] for q in row] for row in h4_by_model])
        sec_hp=sec_hm.mean(axis=0)

        result["conditions"][key]={
          "raw":{"fixed_pi":pi,"arx10_rls":arx,
                 "h4":[{"model_seed":ms,"plants":row} for (ms,_),row in zip(models,h4_by_model)]},
          "primary":{
            "fixed_pi_mean":float(p.mean()),"arx10_rls_mean":float(a.mean()),
            "h4_mean":float(hm.mean()),
            "h4_per_model_means":[float(x) for x in hm.mean(axis=1)],
            "h4_model_init_sd_of_means":float(np.std(hm.mean(axis=1),ddof=1)),
            "PI_minus_H4":bca_mean_ci(dpi,61000+si),
            "ARX_minus_H4":bca_mean_ci(darx,62000+si)},
          "secondary":{
            "fixed_pi_mean":float(sec_p.mean()),"arx10_rls_mean":float(sec_a.mean()),
            "h4_mean":float(sec_hm.mean()),
            "PI_minus_H4_mean":float(np.mean(sec_p-sec_hp)),
            "ARX_minus_H4_mean":float(np.mean(sec_a-sec_hp))}
        }

    d60=result["conditions"]["0.60"]["primary"]
    d12=result["conditions"]["1.20"]["primary"]
    pi60=d60["PI_minus_H4"]; ar60=d60["ARX_minus_H4"]; ar12=d12["ARX_minus_H4"]
    result["predeclared_hypotheses"]={
      "H1":{"rule":"PI-H4 mean > +1.0 Hz and BCa95 excludes 0 at s=0.60",
            "supported":bool(pi60["mean"]>1.0 and pi60["bca95"][0]>0)},
      "H2":{"rule":"ARX-H4 BCa95 entirely inside [-0.25,+0.25] Hz at s=0.60 (practical equivalence)",
            "supported":bool(ar60["bca95"][0]>=-.25 and ar60["bca95"][1]<=.25)},
      "H3":{"rule":"ARX-H4 mean > +0.50 Hz and BCa95 excludes 0 at s=1.20",
            "supported":bool(ar12["mean"]>.50 and ar12["bca95"][0]>0)}
    }
    (out/"experiment_b_final_results.json").write_text(json.dumps(result,indent=2)+"\n")
    print("CONFIRMATION COMPLETE -- seeds 6000-6049 are now SPENT")
    for s in SENS:
        q=result["conditions"][f"{s:.2f}"]["primary"]
        print(f"s={s:.2f}: PI {q['fixed_pi_mean']:.4f} | ARX10 {q['arx10_rls_mean']:.4f} | H4 {q['h4_mean']:.4f} | "
              f"ARX-H4 {q['ARX_minus_H4']['mean']:+.4f} {q['ARX_minus_H4']['bca95']}")
    print("hypotheses:",result["predeclared_hypotheses"])

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=Path("results_experiment_b_final"))
    p.add_argument("--h4-dir",type=Path,default=Path("archived_results/EXPERIMENT_A_FINAL_2026-10-07"))
    p.add_argument("--audit-only",action="store_true")
    a=p.parse_args(); main(a.out,a.h4_dir,a.audit_only)
