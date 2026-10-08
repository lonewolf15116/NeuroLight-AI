"""R1c (development): unified PI family on ONE common tuning set, so the global plain PI, the global
anti-windup PI and the sensitivity-informed reference are all selected from the same data.
Family: positional PI, integrator clip +/-10, optional back-calculation anti-windup
    I <- clip(I + 0.05 e + kt (u_sat - u_raw)/Ki, -10, 10)      (kt = 0 -> plain PI, identical to PI_legacy form)
Grid: Kp {.005,.01,.015,.02,.03,.04,.06,.08} x Ki {.05,.1,.15,.2,.25,.3,.4,.5,.7,1.0} x kt {0,.1,.2,.5,1.0}.
Tuning plants 5000-5099 (even seeds, 50 plants), sigma 0.5, vectorised development simulator.
Usage: python t4_pi_unified.py --part P --nparts N --budget S   (resumable; one JSON per (s, kt) unit)"""
import os,sys,json,time,numpy as np; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from vsim import *
S9=[.5,.6,.7,.8,.9,1.,1.1,1.2,1.4]; TUNE=list(range(5000,5100,2))
KP=[.005,.01,.015,.02,.03,.04,.06,.08]; KI=[.05,.1,.15,.2,.25,.3,.4,.5,.7,1.]; KT=[0.,.1,.2,.5,1.]
GRID=[(a,b,c) for a in KP for b in KI for c in KT]
OUT=os.path.join(HERE,'t4'); os.makedirs(OUT,exist_ok=True)
def run_pi_family(seeds,s,kp,ki,kt,**kw):
    c=VCircuit(seeds,s,**kw); I=np.zeros(len(seeds)); R=[]
    for t in TARGETS:
        e=t-c.rate
        Ip=np.clip(I+.05*e,-10,10); up=np.clip(kp*e+ki*Ip,0,1)                     # plain (kt=0)
        Ib=I+.05*e; ur=kp*e+ki*Ib; ub=np.clip(ur,0,1); Ib=np.clip(Ib+kt*(ub-ur)/ki,-10,10)  # back-calculation
        plain=kt==0; u=np.where(plain,up,ub); I=np.where(plain,Ip,Ib); R.append(c.step(u))
    return rmse(np.array(R).T)
if __name__=='__main__':
    # Work unit = (sensitivity index j, kt value): 80 (Kp,Ki) configs x 50 plants. Cached per unit.
    import argparse; ap=argparse.ArgumentParser(); ap.add_argument('--part',type=int,default=0); ap.add_argument('--nparts',type=int,default=1)
    ap.add_argument('--budget',type=float,default=1e9); a=ap.parse_args(); T0=time.time(); S=len(TUNE)
    units=[(j,k) for j in range(len(S9)) for k in range(len(KT))][a.part::a.nparts]
    for j,k in units:
        f=os.path.join(OUT,f's{j}_kt{k}.json')
        if os.path.exists(f): continue
        if time.time()-T0>a.budget: print('budget reached',flush=True); break
        sub=[(p_,i_,KT[k]) for p_ in KP for i_ in KI]; G=len(sub); t0=time.time()
        kp=np.repeat([g[0] for g in sub],S); ki=np.repeat([g[1] for g in sub],S); kt=np.repeat([g[2] for g in sub],S)
        r=run_pi_family(TUNE*G,S9[j],kp,ki,kt,nseed=1000+10*j+k).reshape(G,S)
        json.dump({'s':S9[j],'kt':KT[k],'grid':sub,'mean_rmse':r.mean(1).tolist(),'seconds':round(time.time()-t0)},open(f,'w'))
        print('unit',j,k,'s',S9[j],'kt',KT[k],round(time.time()-t0),'s',flush=True)
    done=len([x for x in os.listdir(OUT) if x.endswith('.json')]); print(f'{done}/{len(S9)*len(KT)} units complete',flush=True)
