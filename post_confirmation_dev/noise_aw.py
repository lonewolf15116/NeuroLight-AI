import os
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys,numpy as np; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from vsim import *
VAL=list(range(5100,5150))
def run_pi_aw(seeds,s,kp,ki,**kw):
    c=VCircuit(seeds,s,**kw); I=np.zeros(len(seeds)); R=[]
    for t in TARGETS:
        e=t-c.rate; In=np.clip(I+.05*e,-10,10); u_raw=kp*e+ki*In
        # conditional integration: freeze integrator if it would push further into saturation
        sat=((u_raw>1)&(e>0))|((u_raw<0)&(e<0)); I=np.where(sat,I,In)
        R.append(c.step(np.clip(kp*e+ki*I,0,1)))
    return rmse(np.array(R).T), np.array(R).T
c=VCircuit(VAL[:10],1.,noise=2.,nseed=9); r=[c.step(np.zeros(10)) for _ in range(40)]; print('rate floor at u=0, noise 2: %.2f Hz'%np.mean(r[-20:]))
for g in [(.02,.15),(.02,.25),(.01,.25),(.02,.5)]:
    a,R=run_pi_aw(VAL,1.,*g,noise=2.,nseed=4); print('noise2 PI+antiwindup',g,round(a.mean(),3))
ms=[dict(np.load(ROOT+f'/archived_results/EXPERIMENT_A_FINAL_2026-10-07/DR_H4_seed_{m}.npz')) for m in range(101,106)]
