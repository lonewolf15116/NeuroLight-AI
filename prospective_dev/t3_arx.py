"""R2 (development): adaptive ARX-10/RLS. Grid over P0 scale (x sigma^2 (Z'Z)^+) and forgetting factor.
Tuning plants 5000-5099 (even, 50), equal-weight mean over 9-point sweep, sigma 0.5.
Records coefficient drift, prediction change, innovation MAE and numerical stability."""
import os,sys,json,time,numpy as np; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from vsim import *
S9=[.5,.6,.7,.8,.9,1.,1.1,1.2,1.4]; TUNE=list(range(5000,5100,2))
p=np.load(os.path.join(HERE,'arx10_prior.npz')); w0,cov0=p['w0'],p['cov0']
PS=[1,10,100,1e3,1e4,1e5]; LAM=[.9,.95,.98,.995,1.]
grid=[(a,b) for a in PS for b in LAM]
res={}; t0=time.time()
for ps,lam in grid:
    rows=[]
    for j,s in enumerate(S9):
        r,inf=run_arx(TUNE,s,w0,cov0,lam=lam,p_scale=ps,record=True,nseed=200+j)
        rows.append({'s':s,'rmse':float(np.nanmean(r)),'finite_frac':float(inf['finite'].mean()),
                     'max_coef_drift':float(np.nanmax(inf['max_coef_drift'])),'median_pred_change_hz':float(np.nanmedian(inf['pred_change_hz'])),
                     'innov_mae_hz':float(np.nanmean(inf['innov_mae_hz'])),'slew':float(np.nanmean(inf['slew']))})
    res[f'{ps:g}_{lam}']={'p_scale':ps,'lambda':lam,'mean_rmse':float(np.mean([q['rmse'] for q in rows])),'all_finite':all(q['finite_frac']==1 for q in rows),'by_s':rows}
    print(ps,lam,round(res[f'{ps:g}_{lam}']['mean_rmse'],4),res[f'{ps:g}_{lam}']['all_finite'],round(time.time()-t0),flush=True)
    json.dump(res,open(os.path.join(HERE,'t3_arx_results.json'),'w'),indent=1)
print('DONE',flush=True)
