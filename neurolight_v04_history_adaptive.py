"""NeuroLight-AI v0.4: history-conditioned adaptive control."""
from pathlib import Path
import argparse, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from v02_robustness import ShiftedCircuit, run_controller, tune_pi, load_model

H, SCALE = 4, 60.0

class HistoryPredictor:
    def __init__(self, history=H, hidden=64, seed=41):
        self.history=history
        rng=np.random.default_rng(seed); n=2*history+1
        self.w1=rng.normal(0,.2,(n,hidden)); self.b1=np.zeros(hidden)
        self.w2=rng.normal(0,.15,(hidden,1)); self.b2=np.zeros(1)
    def predict(self,x):
        return np.tanh(x@self.w1+self.b1)@self.w2+self.b2
    def fit(self,x,y,xv,yv,epochs=700,patience=60):
        ps=[self.w1,self.b1,self.w2,self.b2]
        m=[np.zeros_like(p) for p in ps]; v=[z.copy() for z in m]
        best=None; wait=0; curve=[]
        for t in range(1,epochs+1):
            h=np.tanh(x@self.w1+self.b1); e=h@self.w2+self.b2-y
            d=2*e/len(x); dh=(d@self.w2.T)*(1-h*h)
            gs=[x.T@dh,dh.sum(0),h.T@d,d.sum(0)]
            for i,(p,g) in enumerate(zip(ps,gs)):
                m[i]=.9*m[i]+.1*g; v[i]=.999*v[i]+.001*g*g
                p-=.006*(m[i]/(1-.9**t))/(np.sqrt(v[i]/(1-.999**t))+1e-8)
            tr=float(np.mean(e*e))*3600
            va=float(np.mean((self.predict(xv)-yv)**2))*3600
            curve.append((tr,va))
            if best is None or va < best[0]-1e-7:
                best=(va,[p.copy() for p in ps],t); wait=0
            else:
                wait+=1
                if wait>=patience: break
        for p,q in zip(ps,best[1]): p[...] = q
        return np.asarray(curve),best[2]

def make_dataset(seeds,steps=300):
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

def hcontrol(model,seed,noise=.5,sensitivity_scale=1.,bias_shift=0.,
             delay_steps=0,measurement_noise=0.):
    c=ShiftedCircuit(seed,noise=noise,sensitivity_scale=sensitivity_scale,
                     bias_shift=bias_shift)
    targets=np.repeat([12.,30.,20.,40.,15.,35.],30)
    rh=[0.]*H; ah=[0.]*H; queue=[0.]*(delay_steps+1)
    rng=np.random.default_rng(seed+90000); rates=[]; lights=[]
    for target in targets:
        obs=queue[0] if delay_steps else c.rate
        if measurement_noise: obs += rng.normal(0,measurement_noise)
        rh[-1]=obs
        cand=np.linspace(0,1,101)
        base=np.r_[np.asarray(rh)/SCALE,np.asarray(ah)]
        x=np.column_stack([np.tile(base,(101,1)),cand])
        pred=model.predict(x).ravel()*SCALE
        cost=(pred-target)**2+2*cand**2+2*(cand-ah[-1])**2
        light=float(cand[np.argmin(cost)])
        actual=c.step(light)
        queue.append(actual); queue.pop(0)
        rh=(rh+[obs])[-H:]; ah=(ah+[light])[-H:]
        rates.append(actual); lights.append(light)
    rates=np.asarray(rates); lights=np.asarray(lights)
    return {"rmse_hz":float(np.sqrt(np.mean((rates-targets)**2))),
            "mean_intensity":float(lights.mean())}

def summarize(hist,original,sc,seeds,kp,ki):
    rows=[]
    for s in seeds:
        kw=sc["kwargs"]
        rows.append((hcontrol(hist,s,**kw),
                     run_controller(original,s,"learned",**kw),
                     run_controller(None,s,"pi",kp=kp,ki=ki,**kw)))
    a=[np.array([r[i]["rmse_hz"] for r in rows]) for i in range(3)]
    rng=np.random.default_rng(20261008)
    def cmp(other):
        d=other-a[0]
        b=np.array([rng.choice(d,len(d),replace=True).mean() for _ in range(5000)])
        return float(d.mean()), [float(np.percentile(b,2.5)),float(np.percentile(b,97.5))], float(np.mean(d>0))
    ao,cio,wo=cmp(a[1]); ap,cip,wp=cmp(a[2])
    return {"history_rmse_hz":float(a[0].mean()),
            "original_rmse_hz":float(a[1].mean()),
            "pi_rmse_hz":float(a[2].mean()),
            "history_advantage_over_original_hz":ao,
            "vs_original_95ci_hz":cio,"win_fraction_vs_original":wo,
            "history_advantage_over_pi_hz":ap,
            "vs_pi_95ci_hz":cip,"win_fraction_vs_pi":wp}

def main(out,original_path):
    out.mkdir(parents=True,exist_ok=True)
    x,y=make_dataset(range(500,600))
    xv,yv=make_dataset(range(600,630))
    model=HistoryPredictor()
    curve,best=model.fit(x,y,xv,yv)
    tr=float(np.sqrt(np.mean((model.predict(x)-y)**2))*60)
    va=float(np.sqrt(np.mean((model.predict(xv)-yv)**2))*60)
    np.savez(out/"predictor_history.npz",history=H,w1=model.w1,b1=model.b1,w2=model.w2,b2=model.b2)

    original=load_model(original_path)
    _,kp,ki=tune_pi()
    pairs=[
      ("nominal",{}),
      ("sensitivity_0.6x",{"sensitivity_scale":.6}),
      ("sensitivity_0.8x",{"sensitivity_scale":.8}),
      ("sensitivity_1.2x",{"sensitivity_scale":1.2}),
      ("sensitivity_1.4x",{"sensitivity_scale":1.4}),
      ("noise_1.0",{"noise":1.}),
      ("noise_2.0",{"noise":2.}),
      ("noise_4.0",{"noise":4.}),
      ("delay_50ms",{"delay_steps":1}),
      ("delay_100ms",{"delay_steps":2}),
      ("delay_250ms",{"delay_steps":5}),
      ("sensor_noise_2hz",{"measurement_noise":2.}),
      ("sensor_noise_5hz",{"measurement_noise":5.}),
      ("bias_minus2",{"bias_shift":-2.}),
      ("bias_plus2",{"bias_shift":2.})]
    scenarios=[{"name":n,"kwargs":k} for n,k in pairs]

    seeds=range(1000,1050)
    results={}
    for sc in scenarios:
        print("FINAL TEST:",sc["name"],flush=True)
        results[sc["name"]]=summarize(model,original,sc,seeds,kp,ki)

    payload={
      "protocol":{"history_steps":H,"history_ms":200,
                  "train_seeds":"500-599","validation_seeds":"600-629",
                  "final_test_seeds":"1000-1049",
                  "warning":"Treat these final-test seeds as spent after this run."},
      "training":{"best_epoch":best,"train_rmse_hz":tr,"validation_rmse_hz":va},
      "tuned_pi":{"kp":kp,"ki":ki},
      "scenarios":results}
    (out/"v04_metrics.json").write_text(json.dumps(payload,indent=2)+"\n")

    fig,ax=plt.subplots(figsize=(8,4))
    ax.plot(curve[:,0],label="train"); ax.plot(curve[:,1],label="validation")
    ax.axvline(best-1,linestyle="--",label=f"best epoch {best}")
    ax.set(xlabel="Epoch",ylabel="MSE (Hz²)"); ax.legend()
    fig.tight_layout(); fig.savefig(out/"v04_training.png",dpi=170); plt.close(fig)

    names=[s["name"] for s in scenarios]; z=np.arange(len(names)); w=.26
    fig,ax=plt.subplots(figsize=(15,6))
    ax.bar(z-w,[results[n]["original_rmse_hz"] for n in names],w,label="Original learned")
    ax.bar(z,[results[n]["history_rmse_hz"] for n in names],w,label="History-conditioned")
    ax.bar(z+w,[results[n]["pi_rmse_hz"] for n in names],w,label="Tuned PI")
    ax.set_ylabel("Tracking RMSE (Hz)"); ax.set_xticks(z)
    ax.set_xticklabels(names,rotation=55,ha="right"); ax.legend()
    fig.tight_layout(); fig.savefig(out/"v04_final_test.png",dpi=170); plt.close(fig)
    print(json.dumps(payload,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=Path("results_v04"))
    p.add_argument("--original",type=Path,default=Path("results/predictor.npz"))
    a=p.parse_args()
    main(a.out,a.original)
