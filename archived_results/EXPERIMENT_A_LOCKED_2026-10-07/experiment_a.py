"""NeuroLight-AI Experiment A: matched history test.
FREEZE BEFORE evaluating seeds 4000-4049. Synthetic study only.
Requires v02_robustness.py.
"""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from v02_robustness import ShiftedCircuit

H=4; SCALE=60.; MODEL_SEEDS=[101,102,103,104,105]
TRAIN=range(2000,2100); VAL=range(2100,2130)
CONF=list(range(4000,4050)) # UNTOUCHED UNTIL LOCK
SENS=[.60,.80,1.00,1.20]; EPOCHS=900; LR=.006

def nparams(d,h): return d*h+h+h+1

class MLP:
 def __init__(self,d,h,seed):
  r=np.random.default_rng(seed)
  self.w1=r.normal(0,.2,(d,h)); self.b1=np.zeros(h)
  self.w2=r.normal(0,.15,(h,1)); self.b2=np.zeros(1)
 def predict(self,x): return np.tanh(x@self.w1+self.b1)@self.w2+self.b2
 def fit(self,x,y,xv,yv):
  ps=[self.w1,self.b1,self.w2,self.b2]; m=[np.zeros_like(p) for p in ps]; v=[z.copy() for z in m]
  best=(float("inf"),None,0)
  for t in range(1,EPOCHS+1):
   h=np.tanh(x@self.w1+self.b1); e=h@self.w2+self.b2-y; d=2*e/len(x); dh=(d@self.w2.T)*(1-h*h)
   gs=[x.T@dh,dh.sum(0),h.T@d,d.sum(0)]
   for i,(p,g) in enumerate(zip(ps,gs)):
    m[i]=.9*m[i]+.1*g; v[i]=.999*v[i]+.001*g*g
    p-=LR*(m[i]/(1-.9**t))/(np.sqrt(v[i]/(1-.999**t))+1e-8)
   va=float(np.mean((self.predict(xv)-yv)**2))
   if va<best[0]: best=(va,[p.copy() for p in ps],t)
  for p,q in zip(ps,best[1]): p[...] = q
  return best[2]

def datasets(seeds,steps=300):
 xh,xm,ys=[],[],[]
 for seed in seeds:
  r=np.random.default_rng(seed+70000)
  c=ShiftedCircuit(seed,noise=float(r.uniform(.5,2)),sensitivity_scale=float(r.uniform(.6,1.4)),bias_shift=float(r.uniform(-2,2)))
  rh=[0.]*H; ah=[0.]*H
  for t in range(steps):
   if t%4==0: light=float(r.uniform())
   xh.append(np.r_[np.asarray(rh)/SCALE,np.asarray(ah),light])
   xm.append([rh[-1]/SCALE,ah[-1],light])
   nxt=c.step(light); ys.append([nxt/SCALE])
   rh=(rh+[nxt])[-H:]; ah=(ah+[light])[-H:]
 return np.asarray(xh),np.asarray(xm),np.asarray(ys)

def metrics(targets,rates,lights):
 r=np.asarray(rates); u=np.asarray(lights)
 return {"rmse_hz":float(np.sqrt(np.mean((r-targets)**2))),"mean_intensity":float(u.mean()),
 "fraction_zero":float(np.mean(u<=1e-12)),"fraction_one":float(np.mean(u>=1-1e-12)),
 "mean_abs_action_change":float(np.mean(np.abs(np.diff(u))))}

def control(model,kind,seed,sensitivity):
 c=ShiftedCircuit(seed,noise=.5,sensitivity_scale=sensitivity,bias_shift=0)
 targets=np.repeat([12.,30.,20.,40.,15.,35.],30); rh=[0.]*H; ah=[0.]*H; rates=[]; lights=[]
 for target in targets:
  cand=np.linspace(0,1,101)
  if kind=="h4":
   base=np.r_[np.asarray(rh)/SCALE,np.asarray(ah)]
   x=np.column_stack([np.tile(base,(101,1)),cand])
  else:
   x=np.column_stack([np.full(101,rh[-1]/SCALE),np.full(101,ah[-1]),cand])
  pred=model.predict(x).ravel()*SCALE
  cost=(pred-target)**2+2*cand**2+2*(cand-ah[-1])**2
  light=float(cand[np.argmin(cost)]); actual=c.step(light)
  rates.append(actual); lights.append(light); rh=(rh+[actual])[-H:]; ah=(ah+[light])[-H:]
 return metrics(targets,rates,lights)

def boot(d,seed):
 d=np.asarray(d); r=np.random.default_rng(seed); idx=r.integers(0,len(d),(10000,len(d))); b=d[idx].mean(1)
 return {"mean":float(d.mean()),"ci95":[float(np.percentile(b,2.5)),float(np.percentile(b,97.5))],
 "positive_fraction":float(np.mean(d>0)),"n_plants":len(d)}

def train(name,d,h,x,y,xv,yv,out):
 ms=[]; info=[]
 for seed in MODEL_SEEDS:
  print("Training",name,"seed",seed,flush=True); m=MLP(d,h,seed); ep=m.fit(x,y,xv,yv)
  info.append({"seed":seed,"best_epoch":ep,"parameters":nparams(d,h),
   "train_rmse_hz":float(np.sqrt(np.mean((m.predict(x)-y)**2))*SCALE),
   "val_rmse_hz":float(np.sqrt(np.mean((m.predict(xv)-yv)**2))*SCALE)})
  np.savez(out/f"{name}_seed_{seed}.npz",w1=m.w1,b1=m.b1,w2=m.w2,b2=m.b2)
  ms.append(m)
 return ms,info

def summary(a):
 a=np.asarray(a); p=a.mean(0)
 return {"mean":float(a.mean()),"std":float(a.std()),"plant_sd_after_model_average":float(p.std()),
 "per_model_mean":[float(z.mean()) for z in a]}

def main(out):
 out.mkdir(parents=True,exist_ok=True)
 xh,xm,y=datasets(TRAIN); xhv,xmv,yv=datasets(VAL)
 h4,ih=train("DR_H4",9,64,xh,y,xhv,yv,out)
 m141,i141=train("DR_Memoryless_141",3,141,xm,y,xmv,yv,out)
 m64,i64=train("DR_Memoryless_64",3,64,xm,y,xmv,yv,out)
 results={}; raw={}
 for si,sens in enumerate(SENS):
  key=f"{sens:.2f}"; print("CONFIRM",key,flush=True)
  mats={}
  for name,models,kind in [("DR_H4",h4,"h4"),("DR_Memoryless_141",m141,"mem"),("DR_Memoryless_64",m64,"mem")]:
   rec=[[control(m,s,kind,sens) for s in CONF] for m in models]
   mats[name]=np.asarray([[q["rmse_hz"] for q in row] for row in rec])
   raw.setdefault(key,{})[name]=rec
  hp=mats["DR_H4"].mean(0); a=mats["DR_Memoryless_141"].mean(0); b=mats["DR_Memoryless_64"].mean(0)
  results[key]={"DR_H4":summary(mats["DR_H4"]),"DR_Memoryless_141":summary(mats["DR_Memoryless_141"]),
   "DR_Memoryless_64":summary(mats["DR_Memoryless_64"]),"M141_minus_H4":boot(a-hp,41000+si),"M64_minus_H4":boot(b-hp,42000+si)}
 payload={"warning":"Seeds 4000-4049 are SPENT after first completed run.","training":{"DR_H4":ih,"DR_Memoryless_141":i141,"DR_Memoryless_64":i64},
 "results":results,"raw_model_by_plant":raw}
 (out/"experiment_a_results.json").write_text(json.dumps(payload,indent=2)+"\n")
 fig,ax=plt.subplots(figsize=(9,5)); xs=np.asarray(SENS)
 for name,label in [("DR_H4","DR-H4 (705)"),("DR_Memoryless_141","DR-Memoryless-141 (706)"),("DR_Memoryless_64","DR-Memoryless-64 (321)")]:
  ax.plot(xs,[results[f"{s:.2f}"][name]["mean"] for s in xs],marker="o",label=label)
 ax.set(xlabel="Sensitivity scale",ylabel="Tracking RMSE (Hz)"); ax.legend(); fig.tight_layout()
 fig.savefig(out/"experiment_a_sensitivity.png",dpi=180); plt.close(fig)
 print(json.dumps(payload,indent=2))

if __name__=="__main__":
 p=argparse.ArgumentParser(); p.add_argument("--out",type=Path,default=Path("results_experiment_a")); a=p.parse_args(); main(a.out)
