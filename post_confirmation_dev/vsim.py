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
