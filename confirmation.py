"""NeuroLight-AI paper-grade confirmation.

IMPORTANT: protocol locked before confirmation data.
Never tune using confirmation plant seeds 3000-3049.
Synthetic study only.

Requires:
  experiment.py
  v02_robustness.py
"""
from pathlib import Path
import argparse, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment import Predictor
from v02_robustness import ShiftedCircuit, run_controller

H=4
SCALE=60.0
MODEL_SEEDS=[101,102,103,104,105]
TRAIN=range(2000,2100)
VAL=range(2100,2130)
CONF=list(range(3000,3050))
SENS=[.60,.80,1.00,1.20]
NOISE=[1.0,2.0,4.0]
KP=.02
KI=.15

class HistoryPredictor:
    def __init__(self,seed,hidden=64):
        rng=np.random.default_rng(seed)
        self.w1=rng.normal(0,.2,(2*H+1,hidden)); self.b1=np.zeros(hidden)
        self.w2=rng.normal(0,.15,(hidden,1)); self.b2=np.zeros(1)
    def predict(self,x):
        return np.tanh(x@self.w1+self.b1)@self.w2+self.b2
    def fit(self,x,y,xv,yv,epochs=900):
        ps=[self.w1,self.b1,self.w2,self.b2]
        m=[np.zeros_like(p) for p in ps]; v=[z.copy() for z in m]
        best=(float("inf"),None,0)
        for t in range(1,epochs+1):
            h=np.tanh(x@self.w1+self.b1); e=h@self.w2+self.b2-y
            d=2*e/len(x); dh=(d@self.w2.T)*(1-h*h)
            gs=[x.T@dh,dh.sum(0),h.T@d,d.sum(0)]
            for i,(p,g) in enumerate(zip(ps,gs)):
                m[i]=.9*m[i]+.1*g; v[i]=.999*v[i]+.001*g*g
                p-=.006*(m[i]/(1-.9**t))/(np.sqrt(v[i]/(1-.999**t))+1e-8)
            va=float(np.mean((self.predict(xv)-yv)**2))
            if va<best[0]:
                best=(va,[p.copy() for p in ps],t)
        for p,q in zip(ps,best[1]): p[...] = q
        return best[2]

def history_dataset(seeds,steps=300):
    xs,ys=[],[]
    for seed in seeds:
        rng=np.random.default_rng(seed+70000)
        c=ShiftedCircuit(seed,noise=float(rng.uniform(.5,2)),
            sensitivity_scale=float(rng.uniform(.6,1.4)),
            bias_shift=float(rng.uniform(-2,2)))
        rh=[0.]*H; ah=[0.]*H
        for t in range(steps):
            if t%4==0: light=float(rng.uniform())
            xs.append(np.r_[np.asarray(rh)/SCALE,np.asarray(ah),light])
            nxt=c.step(light); ys.append([nxt/SCALE])
            rh=(rh+[nxt])[-H:]; ah=(ah+[light])[-H:]
    return np.asarray(xs),np.asarray(ys)

def load_memoryless(path):
    z=np.load(path)
    m=Predictor()
    m.w1=z["w1"]; m.b1=z["b1"]; m.w2=z["w2"]; m.b2=z["b2"]
    return m

def learned_control(model,seed,sensitivity=1.,noise=.5):
    c=ShiftedCircuit(seed,noise=noise,sensitivity_scale=sensitivity,bias_shift=0)
    targets=np.repeat([12.,30.,20.,40.,15.,35.],30)
    previous=0.; rates=[]; lights=[]
    for target in targets:
        cand=np.linspace(0,1,101)
        x=np.column_stack([np.full(101,c.rate/60),np.full(101,previous),cand])
        pred=model.predict(x).ravel()*60
        cost=(pred-target)**2+2*cand**2+2*(cand-previous)**2
        light=float(cand[np.argmin(cost)])
        rates.append(c.step(light)); lights.append(light); previous=light
    return metrics(targets,rates,lights)

def history_control(model,seed,sensitivity=1.,noise=.5,shuffle=False):
    c=ShiftedCircuit(seed,noise=noise,sensitivity_scale=sensitivity,bias_shift=0)
    targets=np.repeat([12.,30.,20.,40.,15.,35.],30)
    rh=[0.]*H; ah=[0.]*H; rates=[]; lights=[]
    rng=np.random.default_rng(seed+91000)
    for target in targets:
        rr=np.asarray(rh,float); aa=np.asarray(ah,float)
        if shuffle:
            p=rng.permutation(H); rr=rr[p]; aa=aa[p]
        cand=np.linspace(0,1,101); base=np.r_[rr/60,aa]
        x=np.column_stack([np.tile(base,(101,1)),cand])
        pred=model.predict(x).ravel()*60
        cost=(pred-target)**2+2*cand**2+2*(cand-ah[-1])**2
        light=float(cand[np.argmin(cost)])
        actual=c.step(light); rates.append(actual); lights.append(light)
        rh=(rh+[actual])[-H:]; ah=(ah+[light])[-H:]
    return metrics(targets,rates,lights)

def pi_control(seed,sensitivity=1.,noise=.5):
    c=ShiftedCircuit(seed,noise=noise,sensitivity_scale=sensitivity,bias_shift=0)
    targets=np.repeat([12.,30.,20.,40.,15.,35.],30)
    integ=0.; rates=[]; lights=[]
    for target in targets:
        err=target-c.rate
        integ=float(np.clip(integ+err*.05,-10,10))
        light=float(np.clip(KP*err+KI*integ,0,1))
        rates.append(c.step(light)); lights.append(light)
    return metrics(targets,rates,lights)

def metrics(targets,rates,lights):
    r=np.asarray(rates); u=np.asarray(lights)
    return {
      "rmse_hz":float(np.sqrt(np.mean((r-targets)**2))),
      "mean_intensity":float(u.mean()),
      "fraction_zero":float(np.mean(u<=1e-12)),
      "fraction_one":float(np.mean(u>=1-1e-12)),
      "mean_abs_action_change":float(np.mean(np.abs(np.diff(u))))
    }

def boot(diff,seed):
    d=np.asarray(diff,float); rng=np.random.default_rng(seed)
    b=np.array([rng.choice(d,len(d),replace=True).mean() for _ in range(5000)])
    return {"mean":float(d.mean()),
      "ci95":[float(np.percentile(b,2.5)),float(np.percentile(b,97.5))],
      "positive_fraction":float(np.mean(d>0))}

def aggregate(records,key="rmse_hz"):
    a=np.asarray(records,float)
    return {"mean":float(a.mean()),"std":float(a.std())}

def main(out,memoryless_path):
    out.mkdir(parents=True,exist_ok=True)
    x,y=history_dataset(TRAIN); xv,yv=history_dataset(VAL)
    models=[]; train_info=[]
    for ms in MODEL_SEEDS:
        print("Training history model seed",ms,flush=True)
        m=HistoryPredictor(ms); ep=m.fit(x,y,xv,yv)
        info={"model_seed":ms,"best_epoch":ep,
          "train_rmse_hz":float(np.sqrt(np.mean((m.predict(x)-y)**2))*60),
          "val_rmse_hz":float(np.sqrt(np.mean((m.predict(xv)-yv)**2))*60)}
        train_info.append(info); models.append(m)
        np.savez(out/f"history_model_seed_{ms}.npz",w1=m.w1,b1=m.b1,w2=m.w2,b2=m.b2)

    mem=load_memoryless(memoryless_path)
    results={"sensitivity":{},"noise":{},"mechanism":{}}

    # Each model initialization is retained; no best-model selection.
    for si,sens in enumerate(SENS):
        key=f"{sens:.2f}"; print("CONFIRM sensitivity",key,flush=True)
        pi=np.array([pi_control(s,sensitivity=sens)["rmse_hz"] for s in CONF])
        mm=np.array([learned_control(mem,s,sensitivity=sens)["rmse_hz"] for s in CONF])
        hm=np.array([[history_control(m,s,sensitivity=sens)["rmse_hz"] for s in CONF] for m in models])
        # paired comparisons average H4 over predeclared model seeds per plant
        hplant=hm.mean(0)
        results["sensitivity"][key]={
          "pi":aggregate(pi),"memoryless":aggregate(mm),
          "history_all_model_plant":{"mean":float(hm.mean()),"std":float(hm.std())},
          "history_per_model_mean":[float(z.mean()) for z in hm],
          "pi_minus_history":boot(pi-hplant,26000+si),
          "memoryless_minus_history":boot(mm-hplant,27000+si)
        }

    for ni,n in enumerate(NOISE):
        key=f"{n:.1f}"; print("CONFIRM noise",key,flush=True)
        pi=np.array([pi_control(s,noise=n)["rmse_hz"] for s in CONF])
        mm=np.array([learned_control(mem,s,noise=n)["rmse_hz"] for s in CONF])
        hm=np.array([[history_control(m,s,noise=n)["rmse_hz"] for s in CONF] for m in models])
        hp=hm.mean(0)
        results["noise"][key]={
          "pi":aggregate(pi),"memoryless":aggregate(mm),
          "history_all_model_plant":{"mean":float(hm.mean()),"std":float(hm.std())},
          "history_per_model_mean":[float(z.mean()) for z in hm],
          "pi_minus_history":boot(pi-hp,28000+ni),
          "memoryless_minus_history":boot(mm-hp,29000+ni)
        }

    for mi,sens in enumerate([.60,.80]):
        key=f"{sens:.2f}"; print("CONFIRM mechanism",key,flush=True)
        ordered=np.array([[history_control(m,s,sensitivity=sens,shuffle=False)["rmse_hz"] for s in CONF] for m in models])
        shuffled=np.array([[history_control(m,s,sensitivity=sens,shuffle=True)["rmse_hz"] for s in CONF] for m in models])
        # paired by model seed and plant seed
        d=(shuffled-ordered).reshape(-1)
        results["mechanism"][key]={
          "ordered_mean":float(ordered.mean()),"joint_shuffle_mean":float(shuffled.mean()),
          "shuffle_minus_ordered":boot(d,30000+mi)
        }

    payload={
      "warning":"CONFIRMATION DATA. Seeds 3000-3049 are spent after this run.",
      "protocol":{"model_seeds":MODEL_SEEDS,"train":"2000-2099","validation":"2100-2129",
        "confirmation":"3000-3049","sensitivity":SENS,"noise":NOISE,
        "pi":{"kp":KP,"ki":KI},"delay_excluded":True},
      "training":train_info,"results":results}
    (out/"confirmation_results.json").write_text(json.dumps(payload,indent=2)+"\n")

    fig,ax=plt.subplots(figsize=(9,5))
    xs=np.array(SENS)
    ax.plot(xs,[results["sensitivity"][f"{s:.2f}"]["pi"]["mean"] for s in xs],marker="o",label="PI")
    ax.plot(xs,[results["sensitivity"][f"{s:.2f}"]["memoryless"]["mean"] for s in xs],marker="o",label="Memoryless learned")
    ax.plot(xs,[results["sensitivity"][f"{s:.2f}"]["history_all_model_plant"]["mean"] for s in xs],marker="o",label="H4 history")
    ax.set(xlabel="Sensitivity scale",ylabel="Tracking RMSE (Hz)")
    ax.legend(); fig.tight_layout(); fig.savefig(out/"confirmation_sensitivity.png",dpi=180); plt.close(fig)

    print(json.dumps(payload,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=Path("results_confirmation"))
    p.add_argument("--memoryless",type=Path,default=Path("results/predictor.npz"))
    a=p.parse_args()
    main(a.out,a.memoryless)
