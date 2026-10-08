"""DEVELOPMENT: resumable long-budget retraining of DR-H4 / DR-Memoryless-141 (seeds 101-105).
Same data (train 2000-2099, val 2100-2129), initialisation, Adam settings and best-validation checkpointing as
experiment_a.MLP.fit; only the epoch budget differs. Checkpoints allow resuming across runs."""
import sys, os, json, time, argparse
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, experiment_a as A
ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=10000); ap.add_argument('--budget',type=float,default=1e9)
ap.add_argument('--only',default=None); a=ap.parse_args(); EPOCHS=a.epochs; T0=time.time()
OUT=os.path.join(ROOT,'prospective_dev',f'retrain_{EPOCHS}'); os.makedirs(OUT,exist_ok=True)
cache=os.path.join(ROOT,'prospective_dev','retrain_data_cache.npz')
if not os.path.exists(cache):
    xh,xm,y=A.datasets(A.TRAIN); xhv,xmv,yv=A.datasets(A.VAL); np.savez(cache,xh=xh,xm=xm,y=y,xhv=xhv,xmv=xmv,yv=yv)
z=np.load(cache); D={'DR_H4':(z['xh'],z['xhv'],9,64),'DR_Memoryless_141':(z['xm'],z['xmv'],3,141)}; y,yv=z['y'],z['yv']
names=[a.only] if a.only else list(D)
for name in names:
    X,XV,d,h=D[name]; logp=os.path.join(OUT,f'log_{name}.json'); log=json.load(open(logp)) if os.path.exists(logp) else {}
    for s in A.MODEL_SEEDS:
        key=f'{name}_{s}'
        if key in log: continue
        ck=os.path.join(OUT,f'ckpt_{key}.npz')
        if os.path.exists(ck):
            c=np.load(ck); ps=[c[f'p{i}'].copy() for i in range(4)]; mo=[c[f'm{i}'].copy() for i in range(4)]
            v=[c[f'v{i}'].copy() for i in range(4)]; bestp=[c[f'b{i}'].copy() for i in range(4)]
            t=int(c['t']); best_va=float(c['best_va']); best_t=int(c['best_t']); curve=c['curve'].tolist()
        else:
            m=A.MLP(d,h,s); ps=[m.w1,m.b1,m.w2,m.b2]; mo=[np.zeros_like(p) for p in ps]; v=[q.copy() for q in mo]
            bestp=[p.copy() for p in ps]; t=0; best_va=float('inf'); best_t=0; curve=[]
        w1,b1,w2,b2=ps
        while t<EPOCHS and time.time()-T0<a.budget:
            t+=1
            hh=np.tanh(X@w1+b1); e=hh@w2+b2-y; dd=2*e/len(X); dh=(dd@w2.T)*(1-hh*hh)
            gs=[X.T@dh,dh.sum(0),hh.T@dd,dd.sum(0)]
            for i,(p,g) in enumerate(zip(ps,gs)):
                mo[i]=.9*mo[i]+.1*g; v[i]=.999*v[i]+.001*g*g
                p-=A.LR*(mo[i]/(1-.9**t))/(np.sqrt(v[i]/(1-.999**t))+1e-8)
            va=float(np.mean((np.tanh(XV@w1+b1)@w2+b2-yv)**2))
            if t%50==0: curve.append([t,float(np.sqrt(va))*60])
            if va<best_va: best_va=va; best_t=t; bestp=[p.copy() for p in ps]
            if t%1000==0 or t==EPOCHS or time.time()-T0>=a.budget:
                np.savez(ck,**{f'p{i}':ps[i] for i in range(4)},**{f'm{i}':mo[i] for i in range(4)},**{f'v{i}':v[i] for i in range(4)},
                         **{f'b{i}':bestp[i] for i in range(4)},t=t,best_va=best_va,best_t=best_t,curve=np.array(curve))
        if t<EPOCHS: print('budget reached',key,t,flush=True); sys.exit(0)
        w1,b1,w2,b2=bestp; pr=lambda Q:np.tanh(Q@w1+b1)@w2+b2
        np.savez(os.path.join(OUT,f'{name}_seed_{s}.npz'),w1=w1,b1=b1,w2=w2,b2=b2)
        log[key]={'best_epoch':best_t,'val_rmse_hz':float(np.sqrt(np.mean((pr(XV)-yv)**2))*60),
                  'train_rmse_hz':float(np.sqrt(np.mean((pr(X)-y)**2))*60),'val_curve_every50':curve}
        json.dump(log,open(logp,'w'),indent=1); print('DONE',key,'best',best_t,'val',round(log[key]['val_rmse_hz'],4),round(time.time()-T0),'s',flush=True)
print('ALL DONE',name if a.only else '',flush=True)
