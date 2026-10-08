"""B-Dev-v2: robust adaptive PI. DEVELOPMENT ONLY; no confirmation seeds."""
from pathlib import Path
import argparse,json,numpy as np
from v02_robustness import ShiftedCircuit
SCALE=60.; SENS=[.6,.8,1.,1.2]; TUNE=range(5000,5100); VAL=range(5100,5150)
TARGETS=np.repeat([12.,30.,20.,40.,15.,35.],30)
KP0=.02; KI0=.15; GAMMAS=[.01,.05,.10,.20]; DELTAS=[.001,.005,.010]
EPS=1e-4; GMIN=.20; GMAX=2.50; ETA=.25

def basic(r,u):
 r=np.asarray(r);u=np.asarray(u)
 return dict(rmse_hz=float(np.sqrt(np.mean((r-TARGETS)**2))),mean_intensity=float(u.mean()),
 fraction_zero=float(np.mean(u<=1e-12)),fraction_one=float(np.mean(u>=1-1e-12)),
 mean_abs_action_change=float(np.mean(np.abs(np.diff(u)))))

def nominal_gain():
 # Development-only dimensional reference: median positive normalized dy/du.
 a=[]
 for seed in TUNE:
  q=np.random.default_rng(seed+91000); c=ShiftedCircuit(seed,noise=.5,sensitivity_scale=1.,bias_shift=0)
  yp=c.rate/SCALE; up=0.
  for t in range(240):
   if t%4==0:u=float(q.uniform())
   y=c.step(u)/SCALE; du=u-up
   if abs(du)>.05 and (y-yp)/du>0:a.append((y-yp)/du)
   yp,up=y,u
 return float(np.median(a)),len(a)

def fixed(seed,s):
 c=ShiftedCircuit(seed,noise=.5,sensitivity_scale=s,bias_shift=0); I=0.;r=[];u=[]
 for tar in TARGETS:
  e=tar-c.rate; I=float(np.clip(I+e*.05,-10,10)); z=float(np.clip(KP0*e+KI0*I,0,1))
  r.append(c.step(z));u.append(z)
 return basic(r,u)

def adaptive(seed,s,gamma,delta,gn):
 c=ShiftedCircuit(seed,noise=.5,sensitivity_scale=s,bias_shift=0)
 g=float(np.clip(gn,GMIN,GMAX)); yp=c.rate/SCALE; up=0.; ep=TARGETS[0]/SCALE-yp
 r=[];us=[];st=[]; freeze=clamp=sat=0
 for tar in TARGETS:
  y=c.rate/SCALE;e=tar/SCALE-y; sh=max(g/gn,ETA); kp=KP0/sh;ki=KI0/sh
  pinc=kp*(e-ep);iinc=ki*e;v=up+pinc+iinc; z=float(np.clip(v,0,1))
  if z!=v:
   sat+=1
   # Actual anti-windup: suppress integral increment if it pushes farther into saturation.
   if (v>1 and iinc>0) or (v<0 and iinc<0): z=float(np.clip(up+pinc,0,1))
  yn=c.step(z)/SCALE;du=z-up;dy=yn-yp
  if abs(du)>delta:
   raw=g+gamma*du*(dy-g*du)/(EPS+du*du); g=float(np.clip(raw,GMIN,GMAX))
   clamp+=int(g<=GMIN+1e-12 or g>=GMAX-1e-12)
  else: freeze+=1
  r.append(yn*SCALE);us.append(z);st.append(g/gn);yp,up,ep=yn,z,e
 o=basic(r,us); st=np.asarray(st)
 o.update(gain_ratio_rmse_vs_true_sensitivity=float(np.sqrt(np.mean((st-s)**2))),
 dead_zone_freeze_rate=freeze/len(TARGETS),gain_boundary_clamp_rate=clamp/len(TARGETS),
 control_saturation_attempt_rate=sat/len(TARGETS),final_g=g,final_gain_ratio=g/gn,
 mean_gain_ratio=float(st.mean()),gain_ratio_trace=st.tolist(),light_trace=us,rate_trace_hz=r)
 return o

def ev(fn,seeds,s):
 raw=[fn(k,s) for k in seeds]; keys=[k for k,v in raw[0].items() if np.isscalar(v)]
 return {"mean":{k:float(np.mean([x[k] for x in raw])) for k in keys},
 "std":{k:float(np.std([x[k] for x in raw])) for k in keys},"raw":raw}

def main(out):
 out.mkdir(parents=True,exist_ok=True);gn,nn=nominal_gain()
 P={"warning":"B-DEV-v2 DEVELOPMENT ONLY; no confirmation seeds declared or touched.",
 "tuning_seeds":"5000-5099","validation_seeds":"5100-5149","sensitivities":SENS,
 "estimator":{"epsilon":EPS,"g_bounds":[GMIN,GMAX],"eta":ETA,"g_nom":gn,"g_nom_samples":nn}}
 search=[];order=0
 for ga in GAMMAS:
  for de in DELTAS:
   by=[];rr=[]
   for s in SENS:
    x=ev(lambda seed,ss,ga=ga,de=de:adaptive(seed,ss,ga,de,gn),TUNE,s)
    rr.append(x["mean"]["rmse_hz"]);by.append({"sensitivity":s,"rmse_hz":rr[-1],
    "gain_ratio_rmse":x["mean"]["gain_ratio_rmse_vs_true_sensitivity"],
    "freeze_rate":x["mean"]["dead_zone_freeze_rate"],"clamp_rate":x["mean"]["gain_boundary_clamp_rate"]})
   search.append({"order":order,"gamma":ga,"delta":de,"mean_tuning_rmse_hz":float(np.mean(rr)),"by_sensitivity":by});order+=1
 best=min(search,key=lambda x:(x["mean_tuning_rmse_hz"],x["order"]))
 P["search"]=sorted(search,key=lambda x:x["mean_tuning_rmse_hz"]);P["choice"]={k:best[k] for k in ["gamma","delta","mean_tuning_rmse_hz"]}
 P["validation"]={}
 for s in SENS:
  a=ev(lambda seed,ss:adaptive(seed,ss,best["gamma"],best["delta"],gn),VAL,s);f=ev(fixed,VAL,s)
  av=np.array([x["rmse_hz"] for x in a["raw"]]);fv=np.array([x["rmse_hz"] for x in f["raw"]])
  P["validation"][f"{s:.2f}"]={"adaptive_pi":a,"fixed_pi":f,
   "adaptive_minus_fixed_mean_hz":float(np.mean(av-fv)),"adaptive_better_fraction":float(np.mean(av<fv))}
 (out/"experiment_b_dev_v2.json").write_text(json.dumps(P,indent=2)+"\n")
 print("B-DEV-v2 ONLY -- no future confirmation seeds declared or touched")
 print(f"nominal normalized gain g_nom={gn:.6f} from {nn} positive development samples");print("choice:",P["choice"])
 for s in SENS:
  q=P["validation"][f"{s:.2f}"];a=q["adaptive_pi"]["mean"];f=q["fixed_pi"]["mean"]
  print(f"sensitivity {s:.2f}: adaptive {a['rmse_hz']:.4f} | fixed {f['rmse_hz']:.4f} | gain-ratio RMSE {a['gain_ratio_rmse_vs_true_sensitivity']:.4f} | freeze {a['dead_zone_freeze_rate']:.3f} | clamp {a['gain_boundary_clamp_rate']:.3f}")
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--out",type=Path,default=Path("results_experiment_b_dev_v2"));a=p.parse_args();main(a.out)
