"""Figures for the v4 manuscript, generated only from the frozen results files."""
import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; OUT = Path(__file__).resolve().parent
v3 = json.load(open(ROOT / 'results_v3/confirmation_v3_results.json'))['analysis']['S2_mean_rmse_and_ranking']
R4 = json.load(open(ROOT / 'results_v4/confirmation_v4_results.json')); T4 = R4['analysis']['S5_mean_rmse']; raw = R4['raw']
S = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4]
COL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#4a3aa7']; MK = ['o', 's', '^', 'D', 'v', 'P']; LS = ['-', '--', '-', '--', '-', ':']
INK, MUTED = '#0b0b0b', '#898781'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e6e5e1', 'grid.linewidth': 0.6, 'lines.linewidth': 1.6, 'lines.markersize': 4.5, 'legend.frameon': False})
# colour, marker and line style follow the controller (entity), identically in every figure
STY = {'PI_tuned_AW': ('#2a78d6', 'o', '-'), 'ARX_RLS': ('#eb6834', 's', '--'), 'ARX_fixed': ('#1baf7a', '^', '-'),
       'H4_10k': ('#eda100', 'D', '--'), 'DR_H4': ('#eda100', 'D', '--'), 'H4_900': ('#e87ba4', 'v', '-'), 'DR_H4_900': ('#e87ba4', 'v', '-'),
       'H4_I2': ('#4a3aa7', 'P', ':'), 'ARX_gain': ('#1baf7a', 'X', '-.'), 'PI_tuned': ('#4a3aa7', 'h', ':')}
# Six validated categorical slots; ARX_fixed/ARX_gain and PI_tuned/H4_I2 share a slot but never appear in the same figure.
def series(ax, arms, table, keyf):
    for k, lab in arms:
        c, m, l = STY[k]; ax.plot(S, [table[keyf(s)][k] for s in S], color=c, marker=m, ls=l, label=lab)
    ax.set_xlabel('hidden sensitivity $s$'); ax.set_xticks(S); ax.set_xticklabels([f'{s:g}' for s in S])
# Fig 1: v3
f, ax = plt.subplots(figsize=(5.6, 3.1))
series(ax, [('PI_tuned', 'PI'), ('PI_tuned_AW', 'PI + anti-windup'), ('ARX_fixed', 'ARX fixed'), ('ARX_RLS', 'ARX + RLS'), ('DR_H4', 'DR-H4 (10k)'), ('DR_H4_900', 'DR-H4 (900)')], v3, lambda s: f's{s:.2f}')
ax.set_ylabel('tracking RMSE (Hz)'); ax.set_ylim(0.8, 6.3); ax.legend(ncol=2, fontsize=7.5, loc='upper left')
f.tight_layout(); f.savefig(OUT / 'fig1_v3_rmse.pdf'); plt.close(f)
# Fig 2: v4 main arms, two noise levels
arms = [('PI_tuned_AW', 'PI + anti-windup'), ('ARX_RLS', 'ARX + RLS (10 coef.)'), ('ARX_gain', 'ARX, gain only (1 param.)'), ('H4_10k', 'DR-H4 (10k)'), ('H4_900', 'DR-H4 (900)'), ('H4_I2', 'DR-H4 + gain input (I2)')]
f, axs = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=False)
series(axs[0], arms, T4, lambda s: f's{s:.2f}'); axs[0].set_title('σ = 0.5 (primary)', fontsize=9, color=INK); axs[0].set_ylabel('tracking RMSE (Hz)')
series(axs[1], arms, T4, lambda s: f'n2.0_s{s:.2f}'); axs[1].set_title('σ = 2.0 (robustness)', fontsize=9, color=INK)
axs[0].axvspan(0.95, 1.45, color='#f0efec', zorder=0); axs[0].text(1.2, 4.3, 'high-gain set', ha='center', fontsize=7.5, color=MUTED)
axs[0].legend(fontsize=7, loc='upper left')
f.tight_layout(); f.savefig(OUT / 'fig2_v4_rmse.pdf'); plt.close(f)
# Fig 3: matched-epoch training length
f, ax = plt.subplots(figsize=(5.6, 3.0))
for i, (k, lab) in enumerate([('H4_snap_e900', 'plain DR-H4, 900 epochs'), ('H4_snap_e6000', 'plain DR-H4, 6000 epochs'), ('H4_I2_e900', 'gain-input, 900 epochs'), ('H4_I2_e6000', 'gain-input, 6000 epochs')]):
    c = STY['H4_10k'][0] if k.startswith('H4_snap') else STY['H4_I2'][0]
    ax.plot(S, [T4[f's{s:.2f}'][k] for s in S], color=c, ls='--' if 'e900' in k else '-', marker='o' if 'e900' in k else 's', label=lab)
ax.axvspan(0.95, 1.45, color='#f0efec', zorder=0); ax.set_xlabel('hidden sensitivity $s$'); ax.set_xticks(S); ax.set_xticklabels([f'{s:g}' for s in S])
ax.set_ylabel('tracking RMSE (Hz)'); ax.legend(fontsize=7.5, loc='upper left'); f.tight_layout(); f.savefig(OUT / 'fig3_training_length.pdf'); plt.close(f)
# Fig 4: gain estimate g vs s, per plant
f, ax = plt.subplots(figsize=(3.4, 3.0))
G = np.array([[raw[f's{s:.2f}'][i]['ARX_gain']['g_final_mean60'] for s in S] for i in range(50)])
for i in range(50): ax.plot(S, G[i], color=COL[0], alpha=0.18, lw=0.8)
ax.plot(S, G.mean(0), color=INK, marker='o', lw=1.6, label='mean of 50 plants')
ax.set_xlabel('hidden sensitivity $s$'); ax.set_ylabel('gain estimate $g$ (last 60 steps)'); ax.set_xticks([0.5, 0.7, 0.9, 1.1, 1.4]); ax.legend(fontsize=7.5, loc='upper left')
f.tight_layout(); f.savefig(OUT / 'fig4_gain_estimate.pdf'); plt.close(f)
print('figures written')
