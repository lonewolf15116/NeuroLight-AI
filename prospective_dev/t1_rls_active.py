"""T1 (development): was the frozen Experiment-B RLS (lambda .98, P scale 1) effectively active?
Evaluation plants 5100-5149; same noise stream for on/off so differences are paired."""
import os,sys,numpy as np; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from vsim import *
p=np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),'arx10_prior.npz')); w0,cov0=p['w0'],p['cov0']
VAL=list(range(5100,5150))
for s in [.6,.8,1.,1.2]:
    on,inf=run_arx(VAL,s,w0,cov0,adapt=True,record=True,nseed=11)
    off=run_arx(VAL,s,w0,cov0,adapt=False,nseed=11)
    print(f's={s}: RLS-on {on.mean():.4f}  fixed {off.mean():.4f}  on-off {np.mean(on-off):+.5f} (max |diff| {np.abs(on-off).max():.4f})  '
          f'max coef drift {inf["max_coef_drift"].max():.2e}  max pred change {inf["pred_change_hz"].max():.3f} Hz  innov MAE {inf["innov_mae_hz"].mean():.3f} Hz')
