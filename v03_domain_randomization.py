"""NeuroLight-AI v0.3: domain-randomized training + frozen evaluation."""
from pathlib import Path
import argparse, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiment import Predictor
from v02_robustness import ShiftedCircuit, run_controller, tune_pi, load_model

def randomized_dataset(seeds, steps=240):
    xs, ys = [], []
    for seed in seeds:
        rng=np.random.default_rng(seed+30000)
        sens=float(rng.uniform(.6,1.4)); bias=float(rng.uniform(-2,2)); noise=float(rng.uniform(.5,2))
        c=ShiftedCircuit(seed, noise=noise, sensitivity_scale=sens, bias_shift=bias)
        previous=0.
        for t in range(steps):
            if t%4==0: light=float(rng.uniform())
            xs.append([c.rate/60,previous,light]); ys.append([c.step(light)/60]); previous=light
    return np.asarray(xs),np.asarray(ys)

def summary(original, dr, scenario, seeds, kp, ki):
    rows=[]
    for s in seeds:
        kw=scenario["kwargs"]
        old=run_controller(original,s,"learned",**kw)
        new=run_controller(dr,s,"learned",**kw)
        pi=run_controller(None,s,"pi",kp=kp,ki=ki,**kw)
        rows.append((old,new,pi))
    old=np.array([r[0]["rmse_hz"] for r in rows])
    new=np.array([r[1]["rmse_hz"] for r in rows])
    pi=np.array([r[2]["rmse_hz"] for r in rows])
    rng=np.random.default_rng(20261007)
    def ci(d):
        b=np.array([rng.choice(d,len(d),replace=True).mean() for _ in range(5000)])
        return [float(np.percentile(b,2.5)),float(np.percentile(b,97.5))]
    a=old-new; b=pi-new
    return {"original_learned_rmse_hz":float(old.mean()),
            "domain_randomized_rmse_hz":float(new.mean()),"pi_rmse_hz":float(pi.mean()),
            "dr_advantage_over_original_hz":float(a.mean()),"dr_vs_original_95ci_hz":ci(a),
            "dr_advantage_over_pi_hz":float(b.mean()),"dr_vs_pi_95ci_hz":ci(b),
            "dr_win_fraction_vs_original":float(np.mean(a>0)),
            "dr_win_fraction_vs_pi":float(np.mean(b>0))}

def main(out, original_path):
    out.mkdir(parents=True,exist_ok=True)
    original=load_model(original_path)
    x,y=randomized_dataset(range(300,380)); xv,yv=randomized_dataset(range(380,400))
    dr=Predictor(seed=17); losses=dr.fit(x,y,epochs=700)
    tr=float(np.sqrt(np.mean((dr.predict(x)-y)**2))*60)
    va=float(np.sqrt(np.mean((dr.predict(xv)-yv)**2))*60)
    np.savez(out/"predictor_domain_randomized.npz",w1=dr.w1,b1=dr.b1,w2=dr.w2,b2=dr.b2)
    _,kp,ki=tune_pi()
    scenarios=[
      {"name":"nominal","kwargs":{}},
      {"name":"sensitivity_0.6x","kwargs":{"sensitivity_scale":.6}},
      {"name":"sensitivity_0.8x","kwargs":{"sensitivity_scale":.8}},
      {"name":"sensitivity_1.2x","kwargs":{"sensitivity_scale":1.2}},
      {"name":"sensitivity_1.4x","kwargs":{"sensitivity_scale":1.4}},
      {"name":"noise_1.0","kwargs":{"noise":1.0}},{"name":"noise_2.0","kwargs":{"noise":2.0}},
      {"name":"noise_4.0","kwargs":{"noise":4.0}},
      {"name":"delay_50ms","kwargs":{"delay_steps":1}},{"name":"delay_100ms","kwargs":{"delay_steps":2}},
      {"name":"delay_250ms","kwargs":{"delay_steps":5}},
      {"name":"sensor_noise_2hz","kwargs":{"measurement_noise":2.0}},
      {"name":"sensor_noise_5hz","kwargs":{"measurement_noise":5.0}},
      {"name":"bias_minus2","kwargs":{"bias_shift":-2.0}},{"name":"bias_plus2","kwargs":{"bias_shift":2.0}}]
    seeds=range(200,230); results={}
    for sc in scenarios:
        print("Running",sc["name"],flush=True); results[sc["name"]]=summary(original,dr,sc,seeds,kp,ki)
    payload={"training":{"train_rmse_hz":tr,"validation_rmse_hz":va,
             "randomization":{"sensitivity_scale":[.6,1.4],"bias_shift":[-2,2],"noise":[.5,2],
                              "train_trajectories":80,"validation_trajectories":20}},
             "evaluation_seeds":list(seeds),"tuned_pi":{"kp":kp,"ki":ki},"scenarios":results}
    (out/"v03_metrics.json").write_text(json.dumps(payload,indent=2)+"\n")
    names=[s["name"] for s in scenarios]; z=np.arange(len(names)); w=.26
    fig,ax=plt.subplots(figsize=(15,6))
    ax.bar(z-w,[results[n]["original_learned_rmse_hz"] for n in names],w,label="Original learned")
    ax.bar(z,[results[n]["domain_randomized_rmse_hz"] for n in names],w,label="Domain-randomized learned")
    ax.bar(z+w,[results[n]["pi_rmse_hz"] for n in names],w,label="Tuned PI")
    ax.set_ylabel("Tracking RMSE (Hz)"); ax.set_xticks(z); ax.set_xticklabels(names,rotation=55,ha="right"); ax.legend()
    fig.tight_layout(); fig.savefig(out/"v03_robustness.png",dpi=170); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4)); ax.plot(losses); ax.set(xlabel="Training epoch",ylabel="Training MSE (Hz²)")
    fig.tight_layout(); fig.savefig(out/"v03_training.png",dpi=170); plt.close(fig)
    print(json.dumps({"train_rmse_hz":tr,"validation_rmse_hz":va,"tuned_pi":{"kp":kp,"ki":ki},"scenarios":results},indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--out",type=Path,default=Path("results_v03"))
    p.add_argument("--original",type=Path,default=Path("results/predictor.npz")); a=p.parse_args()
    main(a.out,a.original)
