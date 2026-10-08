"""Development-only self-test: run every condition for THREE development plants (5100-5102) and execute the
complete pre-specified analysis, so every hypothesis family (P1, P2, B1, S1-S5) is exercised before freezing.
Results are not interpreted (n = 3)."""
import os, sys, json, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, confirmation_v3 as C
P = json.load(open('PROTOCOL_V3.json')); models = C.load_models(P['models']); prior = dict(np.load(P['arx_prior']['path']))
seeds = [5100, 5101, 5102]; t0 = time.time()
raw = {c['key']: [C.run_plant(s, c, P, models, prior) for s in seeds] for c in C.conditions(P)}
A = C.analyse(raw, P)
fam = {k: len(v) for k, v in A.items() if isinstance(v, list)}
print('families and interval counts:', fam)
print('levels:', {k: sorted({round(r['ci_level'], 5) for r in v}) for k, v in A.items() if isinstance(v, list)})
print('decisions:', A['decisions']); print('seconds:', round(time.time() - t0), '| plant-conditions:', 3 * len(raw))
json.dump({'note': 'self-test, n=3 development plants, not interpreted', 'interval_counts': fam}, open('results_v3/analysis_selftest.json', 'w'), indent=1)
