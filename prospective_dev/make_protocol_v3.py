"""Apply the pre-stated development selection rules mechanically and write PROTOCOL_V3.json (status DRAFT).
Rules
  PI_tuned / PI_tuned_AW : single (Kp, Ki) minimising equal-weight mean tuning RMSE over the 9-point sweep.
  PI_oracle              : per-sensitivity minimiser on the same grid (upper bound; reported only).
  ARX_RLS                : (P0 scale, lambda) minimising equal-weight mean tuning RMSE, among configs that stayed finite
                           on every tuning plant; pooled over t3 and t3b grids.
  Selected values must be interior to their grid; otherwise the script flags it and the grid must be extended.
  Models                 : 10000-epoch retrained DR-H4 / DR-Memoryless-141 (seeds 101-105, best validation checkpoint).
"""
import os, json, hashlib, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
rel = lambda p: os.path.relpath(p, ROOT).replace(os.sep, '/')
flags = []

pi = json.load(open(os.path.join(HERE, 't2_pi_results.json')))
S9 = pi['sens']
def interior(v, grid, name):
    if v in (grid[0], grid[-1]): flags.append(f'{name}={v} is on the grid edge {grid[0]}..{grid[-1]}')
sel = {}
kp, ki = pi['plain']['best']; interior(kp, pi['grid_kp'], 'PI_tuned.kp'); interior(ki, pi['grid_ki'], 'PI_tuned.ki')
sel['PI_tuned'] = {'kp': kp, 'ki': ki, 'kt': 0.0, '_dev_mean_tuning_rmse': pi['plain']['best_mean']}
bc = json.load(open(os.path.join(HERE, 't2b_pi_bc_results.json')))
kp, ki, kt = bc['best']
for v, g, n in ((kp, sorted({q[0] for q in bc['grid']}), 'kp'), (ki, sorted({q[1] for q in bc['grid']}), 'ki'), (kt, sorted({q[2] for q in bc['grid']}), 'kt')):
    interior(v, g, 'PI_tuned_AW.' + n)
sel['PI_tuned_AW'] = {'kp': kp, 'ki': ki, 'kt': kt, '_dev_mean_tuning_rmse': bc['best_mean'], '_form': 'back-calculation anti-windup'}
oracle = {f's{o["s"]:.2f}': {'kp': o['gain'][0], 'ki': o['gain'][1], 'kt': 0.0} for o in pi['plain']['oracle_by_s']}
for k, o in oracle.items():
    interior(o['kp'], pi['grid_kp'], f'PI_oracle[{k}].kp'); interior(o['ki'], pi['grid_ki'], f'PI_oracle[{k}].ki')

arx = {}
for f in ('t3_arx_results.json', 't3b_arx_results.json', 't3c_arx_results.json'):
    p = os.path.join(HERE, f)
    if os.path.exists(p): arx.update(json.load(open(p)))
ok = [q for q in arx.values() if q['all_finite']]
best = min(ok, key=lambda q: q['mean_rmse'])
allP = sorted({q['p_scale'] for q in arx.values()}); allL = sorted({q['lambda'] for q in arx.values()})
near = [q for q in ok if q['mean_rmse'] <= best['mean_rmse'] + 0.02]
plateau = {'configs_within_0.02Hz_of_best': sorted([(q['p_scale'], q['lambda'], round(q['mean_rmse'], 4)) for q in near])}
if best['p_scale'] == allP[-1]:
    flags.append(f"ARX_RLS.p_scale={best['p_scale']:g} is the largest tested; objective is a plateau (see arx_plateau), so not extended further")

def models(name, tag):
    out = []
    for s in range(101, 106):
        p = os.path.join(HERE, 'retrain_10000', f'{name}_seed_{s}.npz'); out.append({'path': rel(p), 'sha256': sha(p)})
    return out
log = {}
for n in ('DR_H4', 'DR_Memoryless_141'): log.update(json.load(open(os.path.join(HERE, 'retrain_10000', f'log_{n}.json'))))
def last_gain(v):
    c = dict((int(a), b) for a, b in v['val_curve_every50']); return c[9000] - min(b for e, b in c.items() if e >= 9000)
edge = {k: round(last_gain(v), 4) for k, v in log.items() if v['best_epoch'] >= 9950}
if edge: flags.append(f'best epoch >= 9950 for {list(edge)}; validation gain over last 1000 epochs (Hz): {edge} - treated as plateau')

prior = os.path.join(HERE, 'arx10_prior.npz')
P = {
 'study': 'NeuroLight-AI prospective confirmation v3 (study phase 2)',
 'status': 'DRAFT',
 'note': 'Set status to FROZEN only after review. Earlier confirmations (3000-3049, 4000-4049, 6000-6049) remain as historical results and are not re-run.',
 'confirmation': {'plant_seeds': '7000-7049', 'plant_seeds_range': [7000, 7050], 'n_plants': 50,
                  'rule': 'Untouched until FROZEN; spent after the first completed run regardless of outcome.'},
 'development_seeds': {'tuning': '5000-5099 (even seeds used for grids)', 'evaluation': '5100-5149'},
 'audit_smoke_plant': 5100,
 'sensitivities': S9, 'noise_nominal': 0.5, 'noise_conditions': [1.0, 2.0],
 'chronology_conditions': ['s0.60', 's0.80'],
 'endpoint': {'primary': 'tracking RMSE (Hz) over all 180 steps', 'secondary': 'RMSE steps 50-179; action metrics'},
 'controllers': {
   'PI_legacy': {'kp': 0.02, 'ki': 0.15, 'kt': 0.0},
   'PI_tuned': {k: v for k, v in sel['PI_tuned'].items() if not k.startswith('_')},
   'PI_tuned_AW': {k: v for k, v in sel['PI_tuned_AW'].items() if not k.startswith('_')},  # back-calculation anti-windup
   'PI_oracle': oracle,
   'ARX_fixed': {'lam': 1.0, 'p_scale': 1.0},
   'ARX_RLS': {'lam': best['lambda'], 'p_scale': best['p_scale']},
 },
 'arx_prior': {'path': rel(prior), 'sha256': sha(prior), 'source': 'OLS on training plants 2000-2099 (identical to Experiment B prior)'},
 'models': {'DR_H4': models('DR_H4', 'h4'), 'DR_Mem141': models('DR_Memoryless_141', 'm141')},
 'equivalence_margin_hz': 0.25,
 'statistics': {'unit': 'plant', 'learned_model_reduction': 'mean over five initialisations within plant before differencing',
                'interval': 'BCa bootstrap of mean paired plant-level difference', 'resamples': 10000},
 'analysis': {
   'P1': {'role': 'PRIMARY - history utility', 'contrast': 'DR_Mem141 - DR_H4', 'conditions': ['s0.60', 's0.80'],
          'alpha_each': 0.025, 'multiplicity': 'Bonferroni over 2 conditions',
          'decision': 'supported iff lower bound > 0 at both conditions'},
   'P2': {'role': 'PRIMARY - linear sufficiency', 'contrast': 'ARX_fixed - DR_H4', 'conditions': ['s0.60'], 'alpha': 0.05,
          'decision': 'supported iff whole interval within +/-0.25 Hz'},
   'B1': {'role': 'MANDATORY BENCHMARK - PI robustness', 'contrast': '{PI_tuned, PI_tuned_AW} - DR_H4',
          'conditions': 'all 9 sensitivities + 2 noise conditions', 'alpha_each': round(0.05 / 22, 6),
          'multiplicity': 'Bonferroni over 22 intervals (simultaneous 95%)', 'decision': 'none; no direction assumed'},
   'S1': {'role': 'SECONDARY - adaptation utility', 'contrast': 'ARX_RLS - ARX_fixed', 'conditions': 'all 11',
          'alpha_each': round(0.05 / 11, 6), 'decision': 'none; exploratory (no minimum effect justified)'},
   'S2': {'role': 'SECONDARY - regime dependence', 'output': 'mean RMSE and ranking per condition; crossovers reported descriptively'},
   'S3': {'role': 'SECONDARY - chronology', 'contrast': 'DR_H4_jointshuffle - DR_H4', 'conditions': ['s0.60', 's0.80'],
          'alpha_each': 0.025, 'note': 'plant-level, replaces the cell-level C0 interval'},
 },
 'development_summary': {'PI_tuned': sel['PI_tuned'], 'PI_tuned_AW': sel['PI_tuned_AW'],
                         'ARX_RLS': {k: best[k] for k in ('p_scale', 'lambda', 'mean_rmse')},
                         'retrained_best_epochs': {k: v['best_epoch'] for k, v in log.items()},
                         'retrained_val_rmse_hz': {k: round(v['val_rmse_hz'], 4) for k, v in log.items()}},
 'arx_plateau': plateau,
 'training_notes': ['DR_H4 seed 103 showed a transient validation spike (3.84 Hz) at epoch 10000; the best-validation checkpoint (epoch 9986, 1.393 Hz) is used.',
                    'Retraining was run on the laptop in resumable 165-s chunks (Adam state checkpointed); maths identical to experiment_a.MLP.fit apart from the 10000-epoch budget.'],
 'flags': flags,
 'prohibitions': ['No tuning or model selection on 7000-7049', 'No change of hypotheses, margins or alpha after FROZEN',
                  'No biological or clinical claim', 'Report all controllers and conditions regardless of outcome'],
}
json.dump(P, open(os.path.join(ROOT, 'PROTOCOL_V3.json'), 'w'), indent=2)
print('flags:', flags or 'none'); print(json.dumps(P['development_summary'], indent=1))
