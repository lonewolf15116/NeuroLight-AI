"""NeuroLight-AI v0.5: history-length + shuffled-history diagnostic ablation.

Development-only experiment. Does NOT use v0.4 final-test seeds 1000-1049.
Uses train 1100-1199, validation 1200-1229, diagnostic eval 1300-1349.
"""
from pathlib import Path
import argparse, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from v02_robustness import ShiftedCircuit, run_controller, tune_pi

SCALE=60.0

class MLP:
    def __init__(self,nin,hidden=64,seed=41):
        rng=np.random.default_rng(seed)
        self.w1=rng.normal(0,.2,(nin,hidden)); self.b1=np.zeros(hidden)
        self.w2=rng.normal(0,.15,(hidden,1)); self.b2=np.zeros(1)
    def predict(self,x): return np.tanh(x@self.w1+self.b1)@self.w2+self.b2
    def fit(self,x,y,xv,yv,epochs=900,patience=80):
        ps=[self.w1,self.b1,self.w2,self.b2]; m=[np.zeros_like(p) for p in ps]; v=[z.copy() for z in m]
        best=None; wait=0
        for t in range(1,epochs+1):
            h=np.tanh(x@self.w1+self.b1); e=h@self.w2+self.b2-y
            d=2*e/len(x); dh=(d@self.w2.T)*(1-h*h); gs=[x.T@dh,dh.sum(0),h.T@d,d.sum(0)]
            for i,(p,g) in enumerate(zip(ps,gs)):
                m[i]=.9*m[i]+.1*g; v[i]=.999*v[i]+.001*g*g
                p-=.006*(m[i]/(1-.9**t))/(np.sqrt(v[i]/(1-.999**t))+1e-8)
            va=float(np.mean((self.predict(xv)-yv)**2))
            if best is None or va < best[0]-1e-9:
                best=(va,[p.copy() for p in ps],t); wait=0
            else:
                wait+=1
                if wait>=patience: break
        for p,q in zip(ps,best[1]): p[...] = q
        return best[2]

def data(seeds,H,steps=300):
    xs,ys=[],[]
    for seed in seeds:
        rng=np.random.default_rng(seed+70000)
        c=ShiftedCircuit(seed,noise=float(rng.uniform(.5,2)),
            sensitivity_scale=float(rng.uniform(.6,1.4)),bias_shift=float(rng.uniform(-2,2)))
        rh=[0.]*H; ah=[0.]*H
        for t in range(steps):
            if t%4==0: light=float(rng.uniform())
            xs.append(np.r_[np.asarray(rh)/SCALE,np.asarray(ah),light])
            nxt=c.step(light); ys.append([nxt/SCALE])
            rh=(rh+[nxt])[-H:]; ah=(ah+[light])[-H:]
    return np.asarray(xs),np.asarray(ys)

def control(model,H,seed,sens,shuffle=False):
    c=ShiftedCircuit(seed,noise=.5,sensitivity_scale=sens,bias_shift=0)
    targets=np.repeat([12.,30.,20.,40.,15.,35.],30)
    rh=[0.]*H; ah=[0.]*H; rates=[]
    rng=np.random.default_rng(seed+88000)
    for target in targets:
        rr=np.asarray(rh); aa=np.asarray(ah)
        if shuffle and H>1:
            # Destroy temporal alignment while preserving the same values.
            rr=rr[rng.permutation(H)]; aa=aa[rng.permutation(H)]
        cand=np.linspace(0,1,101); base=np.r_[rr/SCALE,aa]
        x=np.column_stack([np.tile(base,(101,1)),cand])
        pred=model.predict(x).ravel()*SCALE
        light=float(cand[np.argmin((pred-target)**2+2*cand**2+2*(cand-ah[-1])**2)])
        actual=c.step(light); rates.append(actual)
        rh=(rh+[actual])[-H:]; ah=(ah+[light])[-H:]
    return float(np.sqrt(np.mean((np.asarray(rates)-targets)**2)))

def main(out):
    out.mkdir(parents=True,exist_ok=True)
    histories=[1,2,4,8]; models={}; meta={}
    for H in histories:
        print(f"Training H={H}",flush=True)
        x,y=data(range(1100,1200),H); xv,yv=data(range(1200,1230),H)
        model=MLP(2*H+1,seed=41+H); best=model.fit(x,y,xv,yv)
        models[H]=model
        meta[str(H)]={"best_epoch":best,
          "train_rmse_hz":float(np.sqrt(np.mean((model.predict(x)-y)**2))*60),
          "validation_rmse_hz":float(np.sqrt(np.mean((model.predict(xv)-yv)**2))*60)}
    _,kp,ki=tune_pi()
    sensitivities=np.round(np.arange(.5,1.5001,.05),2)
    seeds=range(1300,1350); results={}
    for sens in sensitivities:
        key=f"{sens:.2f}"; results[key]={}
        print("Diagnostic sensitivity",key,flush=True)
        pi=np.mean([run_controller(None,s,"pi",kp=kp,ki=ki,sensitivity_scale=float(sens))["rmse_hz"] for s in seeds])
        results[key]["pi_rmse_hz"]=float(pi)
        for H in histories:
            normal=np.array([control(models[H],H,s,float(sens),False) for s in seeds])
            shuffled=np.array([control(models[H],H,s,float(sens),True) for s in seeds])
            results[key][f"h{H}"]={"rmse_hz":float(normal.mean()),
                                   "shuffled_rmse_hz":float(shuffled.mean()),
                                   "shuffle_penalty_hz":float((shuffled-normal).mean())}
    payload={"protocol":{"purpose":"development-only diagnostic ablation",
              "train_seeds":"1100-1199","validation_seeds":"1200-1229","diagnostic_seeds":"1300-1349",
              "excluded_final_seeds":"1000-1049"},
             "training":meta,"tuned_pi":{"kp":kp,"ki":ki},"sensitivity":results}
    (out/"v05_ablation.json").write_text(json.dumps(payload,indent=2)+"\n")
    fig,ax=plt.subplots(figsize=(10,6))
    xs=sensitivities
    ax.plot(xs,[results[f"{s:.2f}"]["pi_rmse_hz"] for s in xs],label="PI",linewidth=2)
    for H in histories:
        ax.plot(xs,[results[f"{s:.2f}"][f"h{H}"]["rmse_hz"] for s in xs],label=f"H={H}")
    ax.set(xlabel="Sensitivity scale",ylabel="Tracking RMSE (Hz)")
    ax.legend(); fig.tight_layout(); fig.savefig(out/"v05_sensitivity_sweep.png",dpi=170); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,6))
    for H in histories[1:]:
        ax.plot(xs,[results[f"{s:.2f}"][f"h{H}"]["shuffle_penalty_hz"] for s in xs],label=f"H={H}")
    ax.axhline(0,linestyle="--"); ax.set(xlabel="Sensitivity scale",ylabel="Shuffled - ordered RMSE (Hz)")
    ax.legend(); fig.tight_layout(); fig.savefig(out/"v05_shuffle_ablation.png",dpi=170); plt.close(fig)
    print(json.dumps(payload,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--out",type=Path,default=Path("results_v05"))
    main(p.parse_args().out)
