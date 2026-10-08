"""R1b (development): fixed PI with back-calculation anti-windup.
I <- clip(I + 0.05 e + kt (u_sat - u_raw)/Ki, -10, 10). Tuning plants 5000-5099 (every 4th, 25 plants);
selection = equal-weight mean over the 9-point sweep at sigma 0.5 (same rule as R1). Noise sigma=2 at s=1 reported only."""
import os,sys,json,time,numpy as np; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from vsim import *
S9=[.5,.6,.7,.8,.9,1.,1.1,1.2,1.4]; TUNE=list(range(5000,5100,4))
KP=[.01,.015,.02,.03]; KI=[.15,.2,.25,.3,.4]; KT=[.05,.2,.5,1.]
grid=[(a,b,c) for a in KP for b in KI for c in KT]
def run_pi_bc(seeds,s,kp,ki,kt,**kw):
    c=VCircuit(seeds,s,**kw); I=np.zeros(len(seeds)); R=[]
    for t in TARGETS:
        e=t-c.rate; I=I+.05*e; ur=kp*e+ki*I; u=np.clip(ur,0,1); I=np.clip(I+kt*(u-ur)/ki,-10,10)
        R.append(c.step(u))
    return rmse(np.array(R).T)
if __name__=='__main__':
    G=len(grid); S=len(TUNE); tab=np.zeros((G,len(S9))); t0=time.time()
    kp=np.repeat([g[0] for g in grid],S); ki=np.repeat([g[1] for g in grid],S); kt=np.repeat([g[2] for g in grid],S)
    for j,s in enumerate(S9):
        tab[:,j]=run_pi_bc(TUNE*G,s,kp,ki,kt,nseed=300+j).reshape(G,S).mean(1); print('s',s,round(time.time()-t0),flush=True)
    n2=run_pi_bc(TUNE*G,1.,kp,ki,kt,noise=2.,nseed=399).reshape(G,S).mean(1)
    m=tab.mean(1); order=np.argsort(m); b=int(order[0])
    out={'grid':grid,'sens':S9,'table':tab.tolist(),'noise2':n2.tolist(),'best':grid[b],'best_mean':float(m[b]),'best_by_s':tab[b].tolist(),
         'best_noise2':float(n2[b]),'top10':[(grid[i],float(m[i]),float(n2[i])) for i in order[:10]],
         'oracle_by_s':[{'s':s,'gain':grid[int(np.argmin(tab[:,j]))],'rmse':float(tab[:,j].min())} for j,s in enumerate(S9)]}
    json.dump(out,open(os.path.join(HERE,'t2b_pi_bc_results.json'),'w'),indent=1)
    for g,mm,nn in out['top10']: print(g,round(mm,4),'noise2',round(nn,3))
    print('best by s',[round(x,3) for x in out['best_by_s']]); print('DONE')
