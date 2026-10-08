"""Apply the pre-stated development selection rules mechanically and write PROTOCOL_V3.json (status DRAFT).

Selection rules (development plants only; tuning = 5000-5099 even seeds)
  PI family    : positional PI, integrator clip +/-10, back-calculation gain kt (kt = 0 -> plain PI).
                 One unified grid (t4): Kp x Ki x kt, evaluated on the same 50 tuning plants at all 9 sensitivities.
  PI_tuned     : kt = 0 member minimising the equal-weight mean RMSE over the 9-point sweep.
  PI_tuned_AW  : kt > 0 member minimising the same criterion.
  PI_sens_ref  : per-sensitivity minimiser over the whole family. Uses knowledge of s. Called the
                 "sensitivity-informed tuned PI reference": the best PI found by this search, not a bound on classical control.
  ARX_RLS      : (P0 scale, lambda) minimising equal-weight mean RMSE among configs finite on every tuning plant (t3, t3b, t3c).
  ARX_fixed    : same OLS prior, no online update.
  Models       : 10000-epoch retrained DR-H4 / DR-Memoryless-141, seeds 101-105, best-validation checkpoint.
Any selected value on the edge of its grid is listed under "flags".
"""
import os, json, glob, hashlib, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
rel = lambda p: os.path.relpath(p, ROOT).replace(os.sep, '/')
flags = []
S9 = [.5, .6, .7, .8, .9, 1., 1.1, 1.2, 1.4]

def edge(v, grid, name):
    g = sorted(set(grid))
    if v in (g[0], g[-1]): flags.append(f'{name}={v:g} is on the grid edge [{g[0]:g}, {g[-1]:g}]')

# ---------------- PI family (unified grid t4)
tab = {}
for f in glob.glob(os.path.join(HERE, 't4', '*.json')):
    d = json.load(open(f)); j = S9.index(d['s'])
    for g, m in zip(d['grid'], d['mean_rmse']): tab.setdefault(tuple(g), [None] * 9)[j] = m
G = sorted(tab); M = np.array([tab[g] for g in G], float)
assert not np.isnan(M).any() and len(G) == 400, 'unified PI grid incomplete'
KPg, KIg, KTg = [g[0] for g in G], [g[1] for g in G], [g[2] for g in G]
mean = M.mean(1)
plain = [i for i, g in enumerate(G) if g[2] == 0]; aw = [i for i, g in enumerate(G) if g[2] > 0]
bp = plain[int(np.argmin(mean[plain]))]; ba = aw[int(np.argmin(mean[aw]))]
PI_tuned = {'kp': G[bp][0], 'ki': G[bp][1], 'kt': 0.0}
PI_tuned_AW = {'kp': G[ba][0], 'ki': G[ba][1], 'kt': G[ba][2]}
for n, c in (('PI_tuned', PI_tuned), ('PI_tuned_AW', PI_tuned_AW)):
    edge(c['kp'], KPg, n + '.kp'); edge(c['ki'], KIg, n + '.ki')
    if c['kt'] > 0: edge(c['kt'], [k for k in KTg if k > 0], n + '.kt')
sens_ref = {}
for j, s in enumerate(S9):
    i = int(np.argmin(M[:, j])); g = G[i]
    sens_ref[f's{s:.2f}'] = {'kp': g[0], 'ki': g[1], 'kt': g[2]}
    edge(g[0], KPg, f'PI_sens_ref[s{s:.2f}].kp'); edge(g[1], KIg, f'PI_sens_ref[s{s:.2f}].ki')
    if g[2] > 0: edge(g[2], KTg, f'PI_sens_ref[s{s:.2f}].kt')
second_plain = sorted(plain, key=lambda i: mean[i])[1]

# ---------------- adaptive ARX
arx = {}
for f in ('t3_arx_results.json', 't3b_arx_results.json', 't3c_arx_results.json'):
    arx.update(json.load(open(os.path.join(HERE, f))))
ok = [q for q in arx.values() if q['all_finite']]
best = min(ok, key=lambda q: q['mean_rmse'])
near = sorted([(q['p_scale'], q['lambda'], round(q['mean_rmse'], 4)) for q in ok if q['mean_rmse'] <= best['mean_rmse'] + 0.02])
if best['p_scale'] == max(q['p_scale'] for q in arx.values()):
    flags.append(f"ARX_RLS.p_scale={best['p_scale']:g} is the largest tested; the objective is a plateau "
                 f"({len(near)} configs within 0.02 Hz of best, see arx_plateau), so the grid was not extended")

# ---------------- models
def models(name):
    out = []
    for s in range(101, 106):
        p = os.path.join(HERE, 'retrain_10000', f'{name}_seed_{s}.npz'); out.append({'path': rel(p), 'sha256': sha(p)})
    return out
def models900():
    out = []
    for s in range(101, 106):
        p = os.path.join(ROOT, 'archived_results', 'EXPERIMENT_A_FINAL_2026-10-07', f'DR_H4_seed_{s}.npz'); out.append({'path': rel(p), 'sha256': sha(p)})
    return out
expA = json.load(open(os.path.join(ROOT, 'archived_results', 'EXPERIMENT_A_FINAL_2026-10-07', 'experiment_a_results.json')))['training']['DR_H4']
log = {}
for n in ('DR_H4', 'DR_Memoryless_141'): log.update(json.load(open(os.path.join(HERE, 'retrain_10000', f'log_{n}.json'))))
def last_gain(v):
    c = dict((int(a), b) for a, b in v['val_curve_every50']); return c[9000] - min(b for e, b in c.items() if e >= 9000)
late = {k: round(last_gain(v), 4) for k, v in log.items() if v['best_epoch'] >= 9950}
if late: flags.append(f'best epoch >= 9950 for {sorted(late)}; validation gain over the last 1000 epochs {late} Hz - treated as converged in practice')

prior = os.path.join(HERE, 'arx10_prior.npz')
conds = [f's{s:.2f}' for s in S9] + ['n1.0', 'n2.0']
P = {
 'study': 'NeuroLight-AI prospective confirmation v3 (study phase 2)',
 'status': 'DRAFT',
 'historical_record': {
   'note': 'Phase-1 confirmations C0 (3000-3049), A (4000-4049) and B (6000-6049), their archives under archived_results/, '
           'and the manuscript drafts are historical and are neither re-run nor overwritten. V3 writes only to results_v3/.',
   'phase1_git_branch': 'phase1-historical (commit c348cb8)'},
 'confirmation': {'plant_seeds': '7000-7049', 'plant_seeds_range': [7000, 7050], 'n_plants': 50,
                  'rule': 'Untouched until FROZEN; spent after the first completed run regardless of outcome.'},
 'development_seeds': {'tuning': '5000-5099 (even seeds used in grids)', 'evaluation': '5100-5149',
                       'rehearsal': '5100-5149', 'rehearsal_range': [5100, 5150]},
 'audit_smoke_plant': 5100,
 'sensitivities': S9, 'noise_nominal': 0.5, 'noise_conditions': [1.0, 2.0],
 'conditions': {'count': len(conds), 'keys': conds,
                'definition': '9 sensitivities at sigma 0.5, plus sigma 1.0 and 2.0 at s 1.0. s 0.5 lies outside the 0.6-1.4 training range.'},
 'chronology_conditions': ['s0.60', 's0.80'],
 'endpoint': {'primary': 'tracking RMSE (Hz) over all 180 steps', 'secondary': 'RMSE steps 50-179; action metrics'},
 'controllers': {
   'PI_legacy': {'kp': 0.02, 'ki': 0.15, 'kt': 0.0},
   'PI_tuned': PI_tuned,
   'PI_tuned_AW': PI_tuned_AW,
   'PI_sens_ref': sens_ref,
   'ARX_fixed': {'lam': 1.0, 'p_scale': 1.0},
   'ARX_RLS': {'lam': best['lambda'], 'p_scale': best['p_scale']},
 },
 'controller_notes': {
   'DR_H4': 'PRIMARY learned model: 10000-epoch retrained, best-validation checkpoint. Selection rule fixed before the development rehearsal; stays primary regardless of S5.',
   'DR_H4_900': 'SECONDARY arm (S5 only): unmodified phase-1 Experiment-A weights (900-epoch budget). Not retrained or altered.',
   'PI_sens_ref': 'Sensitivity-informed tuned PI reference: per-sensitivity best member of the searched PI family. Uses knowledge of s, '
                  'so it is not deployable; it is the best PI this search found, not an upper bound on classical control. Not hypothesis-tested.',
   'ARX_fixed': 'OLS prior; online update disabled (lam and p_scale unused).'},
 'arx_prior': {'path': rel(prior), 'sha256': sha(prior), 'source': 'OLS on training plants 2000-2099 (identical to the Experiment B prior)'},
 'models': {'DR_H4': models('DR_H4'), 'DR_Mem141': models('DR_Memoryless_141'), 'DR_H4_900': models900()},
 'equivalence_margin_hz': 0.25,
 'equivalence_margin_justification': 'Carried over unchanged from the Experiment B protocol (locked 8 Oct 2026, before its confirmation). '
     'It is about 15% of DR-H4 RMSE at s 0.6 in every phase-1 confirmation and was not re-chosen after phase-2 development data.',
 'statistics': {'unit': 'plant', 'learned_model_reduction': 'mean over five initialisations within plant before differencing',
                'interval': 'BCa bootstrap of the mean paired plant-level difference', 'resamples': 10000},
 'analysis': {
   'P1': {'role': 'PRIMARY - history utility', 'contrast': 'DR_Mem141 - DR_H4', 'conditions': ['s0.60', 's0.80'], 'n_comparisons': 2,
          'alpha_each': 0.025, 'multiplicity': 'Bonferroni over 2', 'decision': 'supported iff the lower bound is > 0 at both conditions'},
   'P2': {'role': 'PRIMARY - linear sufficiency', 'contrast': 'ARX_fixed - DR_H4', 'comparator': 'fixed (non-adaptive) ARX',
          'conditions': ['s0.60'], 'n_comparisons': 1, 'alpha': 0.05,
          'decision': 'supported iff the whole interval lies within +/-0.25 Hz'},
   'B1': {'role': 'MANDATORY BENCHMARK - PI robustness', 'contrast': '{PI_tuned, PI_tuned_AW} - DR_H4', 'conditions': conds,
          'n_comparisons': 22, 'alpha_each': round(0.05 / 22, 6), 'multiplicity': 'Bonferroni over 22 (simultaneous 95%)',
          'decision': 'none; no direction assumed; all reported'},
   'S1': {'role': 'SECONDARY - adaptation utility', 'contrast': 'ARX_RLS - ARX_fixed', 'conditions': conds, 'n_comparisons': 11,
          'alpha_each': round(0.05 / 11, 6), 'decision': 'none; exploratory (no minimum effect justified)'},
   'S2': {'role': 'SECONDARY - regime dependence', 'n_comparisons': 0,
          'output': 'mean RMSE and ranking per condition (PI_sens_ref shown, excluded from ranking); crossovers described, none assumed'},
   'S3': {'role': 'SECONDARY - chronology', 'contrast': 'DR_H4_jointshuffle - DR_H4', 'conditions': ['s0.60', 's0.80'], 'n_comparisons': 2,
          'alpha_each': 0.025, 'note': 'plant-level; replaces the cell-level C0 interval'},
   'S4': {'role': 'SECONDARY - adaptive linear vs neural', 'contrast': 'ARX_RLS - DR_H4', 'conditions': conds, 'n_comparisons': 11,
          'alpha_each': round(0.05 / 11, 6), 'decision': 'none; exploratory'},
   'S5': {'role': 'SECONDARY - training length (prediction/control mismatch)', 'contrast': 'DR_H4_900 - DR_H4',
          'conditions': conds, 'n_comparisons': 11, 'alpha_each': round(0.05 / 11, 6), 'multiplicity': 'Bonferroni over 11 (simultaneous 95%)',
          'unit': 'plant; each arm averaged over all five initialisations (seeds 101-105) within plant; no seed selection',
          'decision': 'none; never promoted to primary whatever the outcome',
          'motivation': 'Added after the development rehearsal showed that the 10000-epoch models, despite lower one-step validation error, '
                        'tracked worse in closed loop at s >= 0.8 than the 900-epoch phase-1 models.'},
 },
 'comparison_count': {'conditions': len(conds), 'primary': 3, 'benchmark': 22, 'secondary_intervals': 35,
                      'note': 'Each family is corrected only within itself. Conditions (11) are not comparisons.'},
 'runtime_estimate': {'laptop_seconds_per_plant_condition': 2.1, 'plant_conditions': 50 * len(conds),
                      'estimate_minutes': 'about 20 on the laptop (resumable 160-s chunks); rehearsal without the 900-epoch arm took 15'},
 'development_summary': {
   'PI_tuned': {**PI_tuned, 'mean_tuning_rmse': round(float(mean[bp]), 4), 'by_s': [round(float(x), 3) for x in M[bp]],
                'runner_up': {'gain': G[second_plain], 'mean_tuning_rmse': round(float(mean[second_plain]), 4)}},
   'PI_tuned_AW': {**PI_tuned_AW, 'mean_tuning_rmse': round(float(mean[ba]), 4), 'by_s': [round(float(x), 3) for x in M[ba]]},
   'PI_sens_ref_by_s': {k: round(float(M[G.index((v['kp'], v['ki'], v['kt'])), j]), 3) for j, (k, v) in enumerate(sens_ref.items())},
   'ARX_RLS': {k: best[k] for k in ('p_scale', 'lambda', 'mean_rmse')},
   'retrained_best_epochs': {k: v['best_epoch'] for k, v in log.items()},
   'retrained_val_rmse_hz': {k: round(v['val_rmse_hz'], 4) for k, v in log.items()}},
 'arx_plateau': {'configs_within_0.02Hz_of_best': near},
 'prediction_metrics': {
   'note': 'One-step validation RMSE (plants 2100-2129) is a prediction metric, reported separately from closed-loop tracking RMSE; no causal link between them is assumed.',
   'DR_H4_10000_val_rmse_hz': {k: round(v['val_rmse_hz'], 4) for k, v in log.items() if k.startswith('DR_H4')},
   'DR_H4_900_val_rmse_hz': {f"DR_H4_{r['seed']}": round(r['val_rmse_hz'], 4) for r in expA}},
 'training_notes': ['DR_H4 seed 103 had a transient validation spike (3.84 Hz) at epoch 10000; its best-validation checkpoint (epoch 9986, 1.393 Hz) is used.',
                    'Retraining ran on the laptop in resumable 165-s chunks with Adam state checkpointed; maths identical to experiment_a.MLP.fit except the 10000-epoch budget.'],
 'flags': flags,
 'prohibitions': ['No tuning or model selection on 7000-7049', 'No change of hypotheses, margins or alpha after FROZEN',
                  'No biological or clinical claim', 'Report all controllers and conditions regardless of outcome'],
}
json.dump(P, open(os.path.join(ROOT, 'PROTOCOL_V3.json'), 'w'), indent=2)
print('flags:'); [print(' -', f) for f in flags]
print(json.dumps(P['development_summary'], indent=1)[:2000])
