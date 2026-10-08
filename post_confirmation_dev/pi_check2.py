import os
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys,time,numpy as np; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from vsim import *
SENS=[.6,.8,1.,1.2]; TUNE=list(range(5000,5100)); VAL=list(range(5100,5150))
t0=time.time()
# sanity: vectorised PI(0.02,0.15) on VAL vs recorded dev numbers 2.300/1.586/1.245/1.162
print('PI orig VAL',[round(run_pi(VAL,s,.02,.15,nseed=1).mean(),3) for s in SENS],round(time.time()-t0))
grid=[(kp,ki) for kp in (.01,.02,.03,.04) for ki in (.15,.2,.25,.3,.4,.5,.7)]
G=len(grid); seeds=TUNE[::2]; S=len(seeds)
res={}
for s in SENS:
    allseeds=seeds*G; kp=np.repeat([g[0] for g in grid],S); ki=np.repeat([g[1] for g in grid],S)
    r=run_pi(allseeds,s,kp,ki,nseed=2).reshape(G,S).mean(1)
    for g,v in zip(grid,r): res.setdefault(g,[]).append(v)
rank=sorted(res,key=lambda g:np.mean(res[g]))
for g in rank[:6]: print('tune',g,[round(x,3) for x in res[g]],round(np.mean(res[g]),3))
print('tune orig',[round(x,3) for x in res[(.02,.15)]], round(time.time()-t0))
best=rank[0]
ms=[dict(np.load(ROOT+f'/archived_results/EXPERIMENT_A_FINAL_2026-10-07/DR_H4_seed_{m}.npz')) for m in range(101,106)]
for s in SENS:
    pb=run_pi(VAL,s,*best,nseed=3); po=run_pi(VAL,s,.02,.15,nseed=3)
    h=np.mean([run_h4(VAL,s,m,nseed=3) for m in ms],0); d=pb-h
    print(f's={s}: PIorig {po.mean():.3f} PIbest{best} {pb.mean():.3f} H4 {h.mean():.3f} PIbest-H4 {d.mean():+.3f} fracPIbetter {np.mean(d<0):.2f}',round(time.time()-t0))
