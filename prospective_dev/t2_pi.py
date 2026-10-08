"""R1 (development): wide-grid fixed PI, with and without conditional-integration anti-windup.
Tuning plants 5000-5099 (even seeds, 50 plants), equal-weight mean over the 9-point sensitivity sweep, sigma 0.5.
Per-sensitivity oracle = best grid point at each s (upper bound only)."""
import os,sys,json,time,numpy as np; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from vsim import *
S9=[.5,.6,.7,.8,.9,1.,1.1,1.2,1.4]; TUNE=list(range(5000,5100,2))
KP=[.005,.01,.015,.02,.03,.04,.06,.08]; KI=[.05,.1,.15,.2,.25,.3,.4,.5,.7,1.]
grid=[(a,b) for a in KP for b in KI]
def run_pi_general(seeds,s,kp,ki,aw=False,**kw):
    c=VCircuit(seeds,s,**kw); I=np.zeros(len(seeds)); R=[]
    for t in TARGETS:
        e=t-c.rate; In=np.clip(I+.05*e,-10,10)
        if aw:
            ur=kp*e+ki*In; sat=((ur>1)&(e>0))|((ur<0)&(e<0)); I=np.where(sat,I,In)
        else: I=In
        R.append(c.step(np.clip(kp*e+ki*I,0,1)))
    return rmse(np.array(R).T)
if __name__=='__main__':
    out={'grid_kp':KP,'grid_ki':KI,'sens':S9,'tuning_plants':'5000-5099 even'}
    for aw in (False,True):
        G=len(grid); S=len(TUNE); tab=np.zeros((G,len(S9))); t0=time.time()
        for j,s in enumerate(S9):
            kp=np.repeat([g[0] for g in grid],S); ki=np.repeat([g[1] for g in grid],S)
            tab[:,j]=run_pi_general(TUNE*G,s,kp,ki,aw=aw,nseed=100+j).reshape(G,S).mean(1)
            print('aw',aw,'s',s,round(time.time()-t0),flush=True)
        m=tab.mean(1); b=int(np.argmin(m))
        out['aw' if aw else 'plain']={'table':tab.tolist(),'best':grid[b],'best_mean':float(m[b]),
           'best_by_s':tab[b].tolist(),'oracle_by_s':[{'s':s,'gain':grid[int(np.argmin(tab[:,j]))],'rmse':float(tab[:,j].min())} for j,s in enumerate(S9)]}
        json.dump(out,open(os.path.join(HERE,'t2_pi_results.json'),'w'),indent=1)
    print('DONE',flush=True)
