"""Vectorised copy of experiment.Circuit (same per-plant parameters; noise drawn from one shared RNG).
DEVELOPMENT DIAGNOSTICS ONLY."""
import numpy as np
TARGETS=np.repeat([12.,30.,20.,40.,15.,35.],30); CAND=np.linspace(0,1,101)
class VCircuit:
    def __init__(self,seeds,sens,noise=.5,bias_shift=0.,nseed=0,n=64):
        B=len(seeds); self.v=np.empty((B,n)); self.bias=np.empty((B,n)); self.sens=np.empty((B,n))
        for i,s in enumerate(seeds):
            r=np.random.default_rng(s); self.v[i]=r.uniform(-65,-55,n); self.bias[i]=r.uniform(7,12,n); self.sens[i]=r.uniform(18,26,n)
        self.sens*=np.asarray(sens).reshape(-1,1) if np.ndim(sens) else sens; self.bias+=bias_shift
        self.gate=np.zeros((B,1)); self.rate=np.zeros(B); self.ref=np.zeros((B,n)); self.noise=noise; self.rng=np.random.default_rng(nseed); self.n=n
    def step(self,light):
        light=np.asarray(light,float).reshape(-1,1); spikes=np.zeros(len(self.rate))
        for _ in range(50):
            self.gate+=(light-self.gate)/10.; a=self.ref<=0
            dv=((-65-self.v)+self.bias+self.sens*self.gate)/20.+self.rng.normal(0,self.noise,self.v.shape)
            self.v=np.where(a,self.v+dv,self.v); f=a&(self.v>=-50); spikes+=f.sum(1)
            self.v[f]=-65; self.ref-=1; self.ref[f]=2
        self.rate=.5*self.rate+.5*spikes/self.n*20; return self.rate.copy()
def rmse(R): return np.sqrt(np.mean((R-TARGETS)**2,axis=1))
def run_pi(seeds,s,kp,ki,**kw):
    kp=np.broadcast_to(kp,(len(seeds),)); ki=np.broadcast_to(ki,(len(seeds),))
    c=VCircuit(seeds,s,**kw); I=np.zeros(len(seeds)); R=[]
    for t in TARGETS:
        e=t-c.rate; I=np.clip(I+.05*e,-10,10); R.append(c.step(np.clip(kp*e+ki*I,0,1)))
    return rmse(np.array(R).T)
def h4_predict(m,X): return (np.tanh(X@m['w1']+m['b1'])@m['w2']+m['b2']).reshape(X.shape[:-1])
def run_h4(seeds,s,m,shuffle=None,**kw):
    B=len(seeds); c=VCircuit(seeds,s,**kw); rh=np.zeros((B,4)); ah=np.zeros((B,4)); R=[]
    for t in TARGETS:
        base=np.concatenate([rh/60.,ah],1)
        X=np.concatenate([np.repeat(base[:,None,:],101,1),np.broadcast_to(CAND,(B,101))[...,None]],2)
        pred=h4_predict(m,X)*60; cost=(pred-t)**2+2*CAND**2+2*(CAND-ah[:,-1:])**2
        u=CAND[np.argmin(cost,1)]; y=c.step(u); R.append(y)
        rh=np.concatenate([rh[:,1:],y[:,None]],1); ah=np.concatenate([ah[:,1:],u[:,None]],1)
    return rmse(np.array(R).T)

# ---------------- vectorised ARX-10 (+ optional RLS), identical regressor/update to experiment_b_final.py
def arx_z(rh,ah,u):
    """rh, ah: (B,4) oldest->newest; u: (B,K) candidates -> (B,K,10) regressor
    [u, r_t, r_t-1, r_t-2, r_t-3 (/60), a_t-1..a_t-4, 1]"""
    B,K=u.shape
    hist=np.concatenate([rh[:,::-1]/60., ah[:,::-1], np.ones((B,1))],1)  # newest first
    return np.concatenate([u[...,None], np.broadcast_to(hist[:,None,:],(B,K,9))],2)
def run_arx(seeds,s,w0,cov0,lam=.98,p_scale=1.,adapt=True,p_identity=None,record=False,**kw):
    B=len(seeds); c=VCircuit(seeds,s,**kw); rh=np.zeros((B,4)); ah=np.zeros((B,4)); R=[];U=[]
    W=np.tile(w0,(B,1)).astype(float)
    P0=(p_identity*np.eye(10)) if p_identity is not None else p_scale*(cov0+1e-12*np.eye(10))
    P=np.tile(P0,(B,1,1)); innov=[]; drift=[]; ok=np.ones(B,bool)
    for t in TARGETS:
        Z=arx_z(rh,ah,np.broadcast_to(CAND,(B,101)))
        pred=np.einsum('bkj,bj->bk',Z,W)*60
        cost=(pred-t)**2+2*CAND**2+2*(CAND-ah[:,-1:])**2
        idx=np.argmin(cost,1); u=CAND[idx]; z=Z[np.arange(B),idx]
        y=c.step(u); e=y/60.-np.einsum('bj,bj->b',z,W); innov.append(e)
        if adapt:
            Pz=np.einsum('bij,bj->bi',P,z); k=Pz/(lam+np.einsum('bj,bj->b',z,Pz))[:,None]
            W=W+k*e[:,None]; P=(P-np.einsum('bi,bj,bjk->bik',k,z,P))/lam; P=.5*(P+P.transpose(0,2,1))
        drift.append(np.abs(W-w0).max(1)); R.append(y);U.append(u)
        rh=np.concatenate([rh[:,1:],y[:,None]],1); ah=np.concatenate([ah[:,1:],u[:,None]],1)
    R=np.array(R).T; U=np.array(U).T; out=rmse(R)
    finite=np.isfinite(W).all(1)&np.isfinite(R).all(1)
    if record:
        return out, dict(innov_mae_hz=60*np.mean(np.abs(np.array(innov)),0), max_coef_drift=np.array(drift).max(0),
                         final_W=W, finite=finite, slew=np.mean(np.abs(np.diff(U,axis=1)),1),
                         pred_change_hz=60*np.abs(W-w0).sum(1))
    return out
