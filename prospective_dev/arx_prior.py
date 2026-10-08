"""Rebuild the exact Experiment-B ARX-10 prior (OLS + covariance) from training plants 2000-2099 and cache it."""
import os,sys,numpy as np
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,ROOT); os.chdir(ROOT)
import experiment_b_final as E
Z,Y=E.arx10_data(E.TRAIN); w0,cov0,prior=E.fit_prior(Z,Y)
np.savez(os.path.join(ROOT,'prospective_dev','arx10_prior.npz'),w0=w0,cov0=cov0)
print('train rmse',prior['train_rmse_hz'],'w0[0]',w0[0])
