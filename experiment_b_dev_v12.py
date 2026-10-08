"""NeuroLight-AI B-Dev-v1.2: matched DR-ARX prior + online RLS.

DEVELOPMENT ONLY.
Offline prior data: exact Experiment-A datasets(TRAIN=2000-2099).
RLS tuning: 5000-5099.
Validation: 5100-5149.
No future confirmation block is declared or touched.

The DR-ARX prior uses the exact Experiment-A randomization and trajectory
construction, converted into a linear ARX regressor with the same H=4 history.
Frozen DR-H4 models are loaded from the Experiment-A archive.
"""
from pathlib import Path
import argparse, json
import numpy as np
from v02_robustness import ShiftedCircuit

H=4; SCALE=60.
TRAIN=range(2000,2100)
TUNE=range(5000,5100)
VAL=range(5100,5150)
SENS=[.60,.80,1.00,1.20]
MODEL_SEEDS=[101,102,103,104,105]
TARGETS=np.repeat([12.,30.,20.,40.,15.,35.],30)
CAND=np.linspace(0.,1.,101)

# Development-only grid. One global pair selected on tuning seeds.
LAM_GRID=[.98,.99,.995]
P_SCALE_GRID=[.1,.3,1.0]

def experiment_a_arx_data(seeds,steps=300):
    """Exact Experiment-A trajectory generator, represented as ARX rows.

    Experiment A constructs xh = [rh/60, ah, current_light], where rh/ah are
    oldest->newest H=4 histories before executing current_light, and target is
    next firing rate / 60.

    We reorder each row to:
    [u_t, y_t/60, y_t-1/60, y_t-2/60, y_t-3/60,
     u_t-1, u_t-2, u_t-3, 1].
    """
    Z=[]; Y=[]
    for seed in seeds:
        r=np.random.default_rng(seed+70000)
        c=ShiftedCircuit(seed,noise=float(r.uniform(.5,2)),
                         sensitivity_scale=float(r.uniform(.6,1.4)),
                         bias_shift=float(r.uniform(-2,2)))
        rh=[0.]*H; ah=[0.]*H
        for t in range(steps):
            if t%4==0:
                light=float(r.uniform())
            Z.append([light,
                      rh[-1]/SCALE,rh[-2]/SCALE,rh[-3]/SCALE,rh[-4]/SCALE,
                      ah[-1],ah[-2],ah[-3],1.])
            nxt=c.step(light)
            Y.append(nxt/SCALE)
            rh=(rh+[nxt])[-H:]
            ah=(ah+[light])[-H:]
    return np.asarray(Z),np.asarray(Y)

def fit_prior(Z,Y):
    # Stable least-squares solution; equivalent to OLS without explicitly
    # forming (Z'Z)^-1.
    w,_,rank,svals=np.linalg.lstsq(Z,Y,rcond=None)
    resid=Y-Z@w
    dof=max(1,len(Y)-rank)
    sigma2=float(resid@resid/dof)
    gram=Z.T@Z
    gram_inv=np.linalg.pinv(gram)
    covariance=sigma2*gram_inv
    return w,covariance,{
        "rank":int(rank),
        "sigma2_normalized":sigma2,
        "train_rmse_hz":float(np.sqrt(np.mean(resid**2))*SCALE),
        "singular_values":[float(x) for x in svals],
        "w_prior":[float(x) for x in w],
        "b0_prior":float(w[0])
    }

def metrics(rates,lights):
    r=np.asarray(rates); u=np.asarray(lights)
    return {"rmse_hz":float(np.sqrt(np.mean((r-TARGETS)**2))),
            "mean_intensity":float(u.mean()),
            "fraction_zero":float(np.mean(u<=1e-12)),
            "fraction_one":float(np.mean(u>=1-1e-12)),
            "mean_abs_action_change":float(np.mean(np.abs(np.diff(u))))}

class FrozenH4:
    def __init__(self,path):
        z=np.load(path); self.w1=z["w1"]; self.b1=z["b1"]; self.w2=z["w2"]; self.b2=z["b2"]
    def predict(self,x):
        return np.tanh(x@self.w1+self.b1)@self.w2+self.b2

def run_h4(model,seed,sens):
    c=ShiftedCircuit(seed,noise=.5,sensitivity_scale=sens,bias_shift=0.)
    rh=[0.]*H; ah=[0.]*H; rates=[]; lights=[]
    for target in TARGETS:
        base=np.r_[np.asarray(rh)/SCALE,np.asarray(ah)]
        x=np.column_stack([np.tile(base,(101,1)),CAND])
        pred=model.predict(x).ravel()*SCALE
        cost=(pred-target)**2+2*CAND**2+2*(CAND-ah[-1])**2
        u=float(CAND[np.argmin(cost)]); y=float(c.step(u))
        rates.append(y); lights.append(u)
        rh=(rh+[y])[-H:]; ah=(ah+[u])[-H:]
    return metrics(rates,lights)

def zvec(rh,ah,u):
    return np.array([u,rh[-1]/SCALE,rh[-2]/SCALE,rh[-3]/SCALE,rh[-4]/SCALE,
                     ah[-1],ah[-2],ah[-3],1.])

class RLS:
    def __init__(self,w0,P0,lam):
        self.w=w0.copy(); self.P=P0.copy(); self.lam=float(lam)
    def predict(self,z): return float(np.asarray(z)@self.w)
    def update(self,z,y):
        z=np.asarray(z,float); Pz=self.P@z
        k=Pz/(self.lam+z@Pz)
        e=float(y-z@self.w)
        self.w += k*e
        self.P=(self.P-np.outer(k,z)@self.P)/self.lam
        self.P=.5*(self.P+self.P.T)
        return e

def run_arx(seed,sens,w0,cov0,lam,p_scale):
    c=ShiftedCircuit(seed,noise=.5,sensitivity_scale=sens,bias_shift=0.)
    # Covariance-derived shape with a development-tuned scalar adaptation strength.
    # Add a tiny diagonal numerical floor only.
    P0=p_scale*(cov0+1e-12*np.eye(9))
    q=RLS(w0,P0,lam)
    rh=[0.]*H; ah=[0.]*H; rates=[]; lights=[]; errs=[]; b0=[float(q.w[0])]
    for target in TARGETS:
        pred=np.array([q.predict(zvec(rh,ah,u))*SCALE for u in CAND])
        cost=(pred-target)**2+2*CAND**2+2*(CAND-ah[-1])**2
        u=float(CAND[np.argmin(cost)])
        z=zvec(rh,ah,u); y=float(c.step(u))
        errs.append(q.update(z,y/SCALE)); b0.append(float(q.w[0]))
        rates.append(y); lights.append(u)
        rh=(rh+[y])[-H:]; ah=(ah+[u])[-H:]
    out=metrics(rates,lights)
    out.update({"final_b0":float(q.w[0]),
                "mean_b0":float(np.mean(b0)),
                "mean_abs_identification_error_normalized":float(np.mean(np.abs(errs))),
                "final_parameters":[float(x) for x in q.w]})
    return out

def evaluate(fn,seeds,sens):
    raw=[fn(s,sens) for s in seeds]
    scalar=[k for k,v in raw[0].items() if np.isscalar(v)]
    return {"mean":{k:float(np.mean([x[k] for x in raw])) for k in scalar},
            "std":{k:float(np.std([x[k] for x in raw])) for k in scalar},
            "raw":raw}

def main(out,h4_dir):
    out.mkdir(parents=True,exist_ok=True)

    Z,Y=experiment_a_arx_data(TRAIN)
    w0,cov0,prior=fit_prior(Z,Y)
    prior["n_rows"]=int(len(Y))
    prior["covariance_diag"]=[float(x) for x in np.diag(cov0)]

    models=[]
    for s in MODEL_SEEDS:
        p=h4_dir/f"DR_H4_seed_{s}.npz"
        if not p.exists(): raise FileNotFoundError(f"Missing frozen H4 model: {p}")
        models.append((s,FrozenH4(p)))

    result={"warning":"B-DEV-v1.2 DEVELOPMENT ONLY; no future confirmation seeds touched.",
            "offline_prior":{"source_seeds":"2000-2099","generator":"exact Experiment-A datasets logic","fit":prior},
            "tuning_seeds":"5000-5099","validation_seeds":"5100-5149",
            "sensitivities":SENS}

    # Select one global RLS adaptation configuration on tuning plants only.
    search=[]
    order=0
    for lam in LAM_GRID:
        for ps in P_SCALE_GRID:
            vals=[]
            for sens in SENS:
                ev=evaluate(lambda seed,s,w0=w0,cov0=cov0,lam=lam,ps=ps:
                            run_arx(seed,s,w0,cov0,lam,ps),TUNE,sens)
                vals.append(ev["mean"]["rmse_hz"])
            search.append({"order":order,"lambda":lam,"p_scale":ps,
                           "rmse_by_sensitivity":vals,
                           "mean_tuning_rmse_hz":float(np.mean(vals))})
            order+=1
    best=min(search,key=lambda x:(x["mean_tuning_rmse_hz"],x["order"]))
    result["rls_search"]=sorted(search,key=lambda x:x["mean_tuning_rmse_hz"])
    result["rls_choice"]={k:best[k] for k in ["lambda","p_scale","mean_tuning_rmse_hz"]}

    result["arx_validation"]={}
    result["h4_validation"]={}
    result["paired_development"]={}

    for sens in SENS:
        key=f"{sens:.2f}"
        arx=evaluate(lambda seed,s:run_arx(seed,s,w0,cov0,best["lambda"],best["p_scale"]),
                     VAL,sens)
        result["arx_validation"][key]=arx

        matrix=[]; records=[]
        for ms,m in models:
            ev=evaluate(lambda seed,s,m=m:run_h4(m,seed,s),VAL,sens)
            records.append({"model_seed":ms,"evaluation":ev})
            matrix.append([x["rmse_hz"] for x in ev["raw"]])
        a=np.asarray(matrix)
        hp=a.mean(0)
        result["h4_validation"][key]={
            "overall_mean_rmse_hz":float(a.mean()),
            "per_model_mean_rmse_hz":[float(x) for x in a.mean(1)],
            "plant_mean_rmse_hz":[float(x) for x in hp],
            "model_records":records}
        ar=np.asarray([x["rmse_hz"] for x in arx["raw"]])
        d=ar-hp
        result["paired_development"][key]={
            "ARX_minus_H4_mean_hz":float(d.mean()),
            "ARX_minus_H4_sd_hz":float(d.std()),
            "fraction_positive":float(np.mean(d>0))}

    (out/"experiment_b_dev_v12.json").write_text(json.dumps(result,indent=2)+"\n")
    np.savez(out/"dr_arx_prior.npz",w0=w0,covariance=cov0)

    print("B-DEV-v1.2 ONLY -- no future confirmation seeds touched")
    print("DR-ARX prior: rows",len(Y),"train RMSE",f"{prior['train_rmse_hz']:.4f}",
          "b0",f"{prior['b0_prior']:.6f}")
    print("RLS choice:",result["rls_choice"])
    for sens in SENS:
        k=f"{sens:.2f}"
        ar=result["arx_validation"][k]["mean"]["rmse_hz"]
        h=result["h4_validation"][k]["overall_mean_rmse_hz"]
        d=result["paired_development"][k]["ARX_minus_H4_mean_hz"]
        print(f"sensitivity {sens:.2f}: DR-ARX+RLS {ar:.4f} | H4 {h:.4f} | ARX-H4 {d:+.4f}")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=Path("results_experiment_b_dev_v12"))
    p.add_argument("--h4-dir",type=Path,default=Path("archived_results/EXPERIMENT_A_FINAL_2026-10-07"))
    a=p.parse_args(); main(a.out,a.h4_dir)
