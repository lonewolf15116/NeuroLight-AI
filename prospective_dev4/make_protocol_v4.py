"""Write PROTOCOL_V4.json (status DRAFT) from phase-4 development results. All selections were made on development plants:
I1 epochs on control-validation plants 5250-5299; everything else is fixed by construction (best-validation checkpoints,
fixed snapshot epochs, v3 controller settings)."""
import os, json, hashlib, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest(); rel = lambda p: os.path.relpath(p, ROOT).replace(os.sep, '/')
SEEDS = range(101, 106); i1 = json.load(open(os.path.join(HERE, 't43_i1.json')))['choice']
def files(fn): return [{'path': rel(p), 'sha256': sha(p)} for p in (fn(s) for s in SEEDS)]
P4 = lambda *a: os.path.join(HERE, *a)
models = {
 'H4_10k': files(lambda s: os.path.join(ROOT, 'prospective_dev', 'retrain_10000', f'DR_H4_seed_{s}.npz')),
 'H4_900': files(lambda s: os.path.join(ROOT, 'archived_results', 'EXPERIMENT_A_FINAL_2026-10-07', f'DR_H4_seed_{s}.npz')),
 'H4_I1': files(lambda s: P4('snapshots', f'DR_H4_{s}_e{i1[str(s)]}.npz')),
 'H4_I2': files(lambda s: P4('models_i2', f'i2_{s}_best.npz')),
 'H4_I3': files(lambda s: P4('models_i3', f'i3_{s}_best.npz')),
 'H4_snap_e900': files(lambda s: P4('snapshots', f'DR_H4_{s}_e900.npz')),
 'H4_snap_e9000': files(lambda s: P4('snapshots', f'DR_H4_{s}_e9000.npz')),
 'H4_I2_e900': files(lambda s: P4('models_i2', f'i2_{s}_e900.npz')),
 'H4_I2_e6000': files(lambda s: P4('models_i2', f'i2_{s}_e6000.npz')),
}
prior = os.path.join(ROOT, 'prospective_dev', 'arx10_prior.npz'); w = np.load(prior)['w0']
S9 = [.5, .6, .7, .8, .9, 1., 1.1, 1.2, 1.4]; conds = [f's{s:.2f}' for s in S9]; HI = ['s1.00', 's1.20', 's1.40']
P = {
 'study': 'NeuroLight-AI prospective confirmation v4 (phase 4: mechanisms)', 'status': 'DRAFT',
 'confirmation': {'plant_seeds': '8000-8049', 'plant_seeds_range': [8000, 8050], 'n_plants': 50},
 'development_seeds': {'mechanism_analysis': '5200-5249', 'I1_control_validation': '5250-5299'},
 'audit_smoke_plant': 5200, 'self_test_plants': [5200, 5201, 5202],
 'sensitivities': S9, 'noise': 0.5,
 'controllers': {'PI_tuned_AW': {'kp': 0.02, 'ki': 0.25, 'kt': 0.5}, 'ARX_RLS': {'lam': 0.98, 'p_scale': 1e9},
                 'ARX_gain': {'lam': 0.98, 'p0': 1000.0, 'act_idx': [0, 5, 6, 7, 8]}},
 'network_arms': list(models), 'models': models,
 'model_notes': {'H4_10k': 'v3 primary DR-H4 (10000-epoch, best validation)', 'H4_900': 'phase-1 900-epoch weights',
                 'H4_I1': f'I1 closed-loop early stopping: per-seed snapshot epoch chosen on plants 5250-5299: {i1}',
                 'H4_I2': 'I2: 10 inputs (DR-H4 inputs + one-parameter gain-RLS estimate g), best validation within 6000 epochs',
                 'H4_I3': 'I3: DR-H4 retrained on open-loop + own closed-loop data, best validation within 6000 epochs',
                 'H4_snap_*': 'raw DR-H4 weights at fixed epochs from the checkpointed retraining', 'H4_I2_e*': 'raw I2 weights at fixed epochs'},
 'arx_prior': {'path': rel(prior), 'sha256': sha(prior), 'w_u': float(w[0])},
 'statistics': {'unit': 'plant; network arms averaged over five seeds within plant', 'interval': 'BCa bootstrap', 'resamples': 10000},
 'analysis': {
  'Q1': {'role': 'PRIMARY', 'test': 'fraction of the full-RLS mean improvement over fixed ARX recovered by ARX_gain (ratio of plant-level means over the sweep)',
         'alpha': 0.05, 'threshold': 0.70, 'decision': 'lower CI bound >= 0.70', 'dev_value': 0.86},
  'Q2': {'role': 'PRIMARY', 'test': 'Spearman correlation of final g with s over 450 plant-conditions', 'threshold': 0.90, 'dev_value': 'about 1.0 on condition means'},
  'Q3': {'role': 'PRIMARY', 'test': 'H4_10k - H4_I2 RMSE', 'conditions': HI, 'alpha_each': round(0.05 / 3, 6), 'decision': 'all lower bounds > 0', 'dev_value': [0.87, 1.93, 3.20]},
  'Q4': {'role': 'PRIMARY', 'test': 'training length: (a) H4_snap_e9000 - H4_snap_e900 > 0 and (b) H4_I2_e6000 - H4_I2_e900 < 0', 'conditions': HI,
         'alpha_each': round(0.05 / 6, 6), 'decision': '(a) all lower bounds > 0 and (b) all upper bounds < 0'},
  'S1': {'role': 'secondary', 'test': 'H4_I2 - ARX_RLS', 'conditions': conds, 'alpha_each': round(0.05 / 9, 6)},
  'S2': {'role': 'secondary', 'test': 'H4_10k - H4_I1', 'conditions': conds, 'alpha_each': round(0.05 / 9, 6)},
  'S3': {'role': 'secondary', 'test': 'H4_10k - H4_I3', 'conditions': conds, 'alpha_each': round(0.05 / 9, 6)},
  'S4': {'role': 'secondary', 'test': 'slope of network dyhat/du vs s divided by slope of 60*g*w_u vs s, for H4_10k, H4_900, H4_I2', 'alpha_each': round(0.05 / 3, 6),
         'dev_value': {'H4_10k': 0.39, 'H4_900': 0.09, 'H4_I2': 0.84}},
  'S5': {'role': 'descriptive', 'test': 'mean RMSE per controller and condition'}},
 'prohibitions': ['No tuning or model selection on 8000-8049', 'No change of hypotheses, thresholds or alpha after FROZEN', 'Report all arms and conditions', 'No biological claim'],
}
json.dump(P, open(os.path.join(ROOT, 'PROTOCOL_V4.json'), 'w'), indent=2); print('arms', list(models), '| I1 epochs', i1)
