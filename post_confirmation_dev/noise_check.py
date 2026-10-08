import os
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys,numpy as np; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from vsim import *
VAL=list(range(5100,5150))
ms=[dict(np.load(ROOT+f'/archived_results/EXPERIMENT_A_FINAL_2026-10-07/DR_H4_seed_{m}.npz')) for m in range(101,106)]
for nz in (1.,2.):
    po=run_pi(VAL,1.,.02,.15,noise=nz,nseed=4); pb=run_pi(VAL,1.,.02,.25,noise=nz,nseed=4)
    h=np.mean([run_h4(VAL,1.,m,noise=nz,nseed=4) for m in ms],0)
    print(f'noise={nz}: PI(.02,.15) {po.mean():.3f}  PI(.02,.25) {pb.mean():.3f}  H4 {h.mean():.3f}  PI.25-H4 {np.mean(pb-h):+.3f} frac PI better {np.mean(pb<h):.2f}')
