# Integrity check after an accidental overlapping invocation: deterministically recompute every plant in the
# conditions written while two processes overlapped (plant-conditions ~150-301 -> s0.80, s0.90, s1.00, s1.10)
# and compare with the recorded raw results. Recomputation only; nothing is tuned or changed.
import sys, json, numpy as np; sys.path.insert(0, '/home/claude/repo'); import os; os.chdir('/home/claude/repo')
import confirmation_v3 as C
P = json.load(open('PROTOCOL_V3.json')); models = C.load_models(P['models']); prior = dict(np.load(P['arx_prior']['path']))
d = json.load(open('results_v3/confirmation_v3_results.json'))['raw']
conds = {c['key']: c for c in C.conditions(P)}; seeds = list(range(7000, 7050)); maxdiff = 0.; n = 0
for key in sys.argv[1:]:
    for i, sd in enumerate(seeds):
        rec = C.run_plant(sd, conds[key], P, models, prior); old = d[key][i]
        for name in rec:
            a = rec[name] if isinstance(rec[name], list) else [rec[name]]; b = old[name] if isinstance(old[name], list) else [old[name]]
            for x, y in zip(a, b): maxdiff = max(maxdiff, abs(x['rmse'] - y['rmse']))
        n += 1
    print(key, 'checked', flush=True)
print('plants checked', n, 'max |rmse diff|', maxdiff)
