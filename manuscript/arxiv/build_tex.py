"""Builds main.tex for the v4 manuscript. Every result number is read from the frozen results files;
nothing in Section 4 is typed by hand. Placeholders <<name>> in TEMPLATE are filled from V."""
import json, re
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
A3 = json.load(open(ROOT / 'results_v3/confirmation_v3_results.json'))['analysis']
R4 = json.load(open(ROOT / 'results_v4/confirmation_v4_results.json')); A4 = R4['analysis']
T3 = A3['S2_mean_rmse_and_ranking']; T4 = A4['S5_mean_rmse']
S9 = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4]

def sg(x, d=2):  # signed number, LaTeX minus
    return ('+' if x >= 0 else '$-$') + f'{abs(x):.{d}f}'
def ci(lo, hi, d=2): return f'[{sg(lo, d)}, {sg(hi, d)}]'
def pci(est, lo, hi, d=2): return f'{sg(est, d)} {ci(lo, hi, d)}'
def get3(fam, cond, contrast=None): return next(r for r in A3[fam] if r['condition'] == cond and (contrast is None or r['contrast'] == contrast))
def get4(fam, cond): return next(r for r in A4[fam] if r['condition'] == cond)
def pct(x): return f'{100 * x:.0f}'
V = {}
# ---------------- v3
p1a, p1b, p2 = get3('P1_history', 's0.60'), get3('P1_history', 's0.80'), get3('P2_linear_sufficiency', 's0.60')
s3a, s3b = get3('S3_chronology', 's0.60'), get3('S3_chronology', 's0.80')
V.update(p1a=pci(p1a['mean'], *p1a['ci']), p1b=pci(p1b['mean'], *p1b['ci']), p1a_abs=f"{p1a['mean']:.2f}", p1a_lo=f"{p1a['ci'][0]:.2f}", p1a_hi=f"{p1a['ci'][1]:.2f}",
         p2=pci(p2['mean'], *p2['ci'], d=3), p2_abs=f"{p2['mean']:.3f}", p2_lo=f"{p2['ci'][0]:.3f}", p2_hi=f"{p2['ci'][1]:.3f}",
         s3a=pci(s3a['mean'], *s3a['ci']), s3b=pci(s3b['mean'], *s3b['ci']),
         arxf12=f"{T3['s1.20']['ARX_fixed']:.2f}", arxf14=f"{T3['s1.40']['ARX_fixed']:.2f}")
CONDS3 = ['s0.50', 's0.60', 's0.70', 's0.80', 's0.90', 's1.00', 's1.10', 's1.20', 's1.40', 'n1.0', 'n2.0']
lab3 = lambda c: f'$s={c[1:]}$' if c.startswith('s') else f'$\\sigma={c[1:]}$'
s4n = sum(get3('S4_adaptive_linear_vs_H4', c)['ci'][1] < 0 for c in CONDS3)
V.update(s4n=str(s4n), s1_lo=f"{abs(get3('S1_adaptation', 'n2.0')['mean']):.2f}", s1_hi=f"{abs(get3('S1_adaptation', 's1.40')['mean']):.2f}",
         aw06=sg(get3('B1_PI_benchmark', 's0.60', 'PI_tuned_AW - DR_H4')['mean']), aw07=f"{abs(get3('B1_PI_benchmark', 's0.70', 'PI_tuned_AW - DR_H4')['mean']):.2f}",
         aw14=f"{abs(get3('B1_PI_benchmark', 's1.40', 'PI_tuned_AW - DR_H4')['mean']):.2f}", awn2=sg(get3('B1_PI_benchmark', 'n2.0', 'PI_tuned_AW - DR_H4')['mean']),
         s5_07=f"{abs(get3('S5_training_length', 's0.70')['mean']):.2f}", s5_10=f"{abs(get3('S5_training_length', 's1.00')['mean']):.2f}", s5_14=f"{abs(get3('S5_training_length', 's1.40')['mean']):.2f}",
         rls14_vs_legacy=f"{T3['s1.40']['ARX_RLS'] - T3['s1.40']['PI_legacy']:.3f}", rls12=f"{T3['s1.20']['ARX_RLS']:.3f}", ref12=f"{T3['s1.20']['PI_sens_ref']:.3f}")
cols3 = [('PI_tuned', 'PI'), ('PI_tuned_AW', 'PI+AW'), ('ARX_fixed', 'ARX'), ('ARX_RLS', 'ARX+RLS'), ('DR_H4', 'H4-10k'), ('DR_H4_900', 'H4-900'), ('DR_Mem141', 'Mem-141'), ('PI_sens_ref', 'PI ref.$^\\dagger$')]
rows = []
for c in CONDS3:
    dep = [T3[c][k] for k, _ in cols3 if k != 'PI_sens_ref']; b = min(dep)
    cell = lambda k: '---' if k not in T3[c] else (f'\\textbf{{{T3[c][k]:.3f}}}' if (k != 'PI_sens_ref' and T3[c][k] == b) else f'{T3[c][k]:.3f}')
    rows.append(lab3(c) + ' & ' + ' & '.join(cell(k) for k, _ in cols3) + r' \\')
V['table_v3'] = '\n'.join(rows); V['table_v3_head'] = 'Condition & ' + ' & '.join(n for _, n in cols3) + r' \\'
rows = []
for c in CONDS3:
    a, b = get3('B1_PI_benchmark', c, 'PI_tuned - DR_H4'), get3('B1_PI_benchmark', c, 'PI_tuned_AW - DR_H4')
    rows.append(f"{lab3(c)} & {pci(a['mean'], *a['ci'], d=3)} & {pci(b['mean'], *b['ci'], d=3)} \\\\")
V['table_b1'] = '\n'.join(rows)
# ---------------- v4 primary
q1, q3, q4 = A4['Q1_gain_fraction'], A4['Q3_gain_input_vs_10k'], A4['Q4_differential_training_length']
V.update(q1=f"{q1['estimate']:.3f}", q1ci=f"[{q1['ci'][0]:.3f}, {q1['ci'][1]:.3f}]", q1pct=pct(q1['estimate']), q1pct_lo=pct(q1['ci'][0]), q1pct_hi=pct(q1['ci'][1]),
         q3=f"{q3['estimate']:.2f}", q3ci=f"[{q3['ci'][0]:.2f}, {q3['ci'][1]:.2f}]", q4=f"{q4['estimate']:.2f}", q4ci=f"[{q4['ci'][0]:.2f}, {q4['ci'][1]:.2f}]")
c = A4['S_Q4_components']; pl, gi = c['plain_e6000_minus_e900'], c['gaininput_e6000_minus_e900']
V.update(q4plain=pci(pl['estimate'], *pl['ci']), q4gain=pci(gi['estimate'], *gi['ci']), q4plain_abs=f"{pl['estimate']:.2f}", q4gain_abs=f"{abs(gi['estimate']):.2f}")
q2 = A4['S_Q2_gain_spearman']; cal = A4['S_calibration']
V.update(q2_min=f"{min(q2['per_plant']):.2f}", cal_slope=f"{cal['slope_dg_ds']['estimate']:.2f}", cal_slope_ci=f"[{cal['slope_dg_ds']['ci'][0]:.2f}, {cal['slope_dg_ds']['ci'][1]:.2f}]",
         cal_int=f"{cal['intercept']['estimate']:.3f}", cal_res=f"{cal['mean_abs_residual_from_linear']['estimate']:.3f}", cal_cv=f"{cal['cv_of_g_over_s']['estimate']:.3f}",
         cal_g1=f"{cal['g_at_nominal_s1_minus_1']['estimate']:.2f}")
sq3 = A4['S_Q3_per_setting']; V.update(sq3_lo=f"{min(r['mean'] for r in sq3):.2f}", sq3_hi=f"{max(r['mean'] for r in sq3):.2f}")
s4 = A4['S4_sensitivity_slope_ratio']
for k, n in [('H4_10k', 'sl10k'), ('H4_900', 'sl900'), ('H4_I2', 'slI2')]:
    V[n] = f"{s4[k]['estimate']:.2f}"; V[n + 'ci'] = f"[{s4[k]['ci'][0]:.2f}, {s4[k]['ci'][1]:.2f}]"
Rr = A4['R_sigma2.0']
V.update(rq1pct=pct(Rr['Q1_gain_fraction']['estimate']), rq1=f"{Rr['Q1_gain_fraction']['estimate']:.2f}", rq1ci=f"[{Rr['Q1_gain_fraction']['ci'][0]:.2f}, {Rr['Q1_gain_fraction']['ci'][1]:.2f}]",
         rq3=f"{Rr['Q3_gain_input_vs_10k']['estimate']:.3f}", rq3ci=f"[{Rr['Q3_gain_input_vs_10k']['ci'][0]:.3f}, {Rr['Q3_gain_input_vs_10k']['ci'][1]:.3f}]",
         rq4=f"{Rr['Q4_differential_training_length']['estimate']:.3f}", rq4ci=f"[{Rr['Q4_differential_training_length']['ci'][0]:.3f}, {Rr['Q4_differential_training_length']['ci'][1]:.3f}]")
s1 = A4['S1_I2_vs_ARX_RLS']; V.update(s1_min=sg(min(r['mean'] for r in s1), 3), s1_max=sg(max(r['mean'] for r in s1), 3),
    s1_better=', '.join(f"{float(r['condition'][1:]):.1f}" for r in s1 if r['ci'][1] < 0), s1_worse=', '.join(f"{float(r['condition'][1:]):.1f}" for r in s1 if r['ci'][0] > 0))
s6 = A4['S6_I2_vs_PI_AW']; V.update(s6_n=str(sum(r['ci'][1] < 0 for r in s6)), s6_lo=f"{min(abs(r['mean']) for r in s6):.3f}", s6_hi=f"{max(abs(r['mean']) for r in s6):.3f}")
s2, s3 = A4['S2_I1_vs_10k'], A4['S3_I3_vs_10k']
V.update(s2_05=pci(get4('S2_I1_vs_10k', 's0.50')['mean'], *get4('S2_I1_vs_10k', 's0.50')['ci'], d=3), s2_14=f"{get4('S2_I1_vs_10k', 's1.40')['mean']:.2f}", s3_14=f"{get4('S3_I3_vs_10k', 's1.40')['mean']:.2f}",
         s2_pos_from=f"{min(float(r['condition'][1:]) for r in s2 if r['ci'][0] > 0):.1f}", s3_pos_n=str(sum(r['ci'][0] > 0 for r in s3)))
mean9 = lambda k, pre='': sum(T4[f'{pre}s{s:.2f}'][k] for s in S9) / 9
for k, n in [('ARX_RLS', 'm_rls'), ('H4_I2', 'm_i2'), ('PI_tuned_AW', 'm_aw'), ('ARX_gain', 'm_gain'), ('H4_I1', 'm_i1'), ('H4_I3', 'm_i3'), ('H4_900', 'm_900'), ('H4_10k', 'm_10k'), ('ARX_fixed', 'm_fixed'), ('PI_tuned', 'm_pi')]:
    V[n] = f'{mean9(k):.3f}'; V[n + '2'] = f"{mean9(k, 'n2.0_'):.3f}"
# Table: v4 mean RMSE at sigma 0.5 and 2.0
cols4 = [('PI_tuned_AW', 'PI+AW'), ('ARX_fixed', 'ARX'), ('ARX_RLS', 'ARX+RLS'), ('ARX_gain', 'ARX-$g$'), ('H4_10k', 'H4-10k'), ('H4_900', 'H4-900'), ('H4_I1', 'I1'), ('H4_I3', 'I3'), ('H4_I2', 'I2')]
def t4rows(pre):
    out = []
    for s in S9 + ['mean']:
        if s == 'mean':
            vals = {k: mean9(k, pre) for k, _ in cols4}; lab = r'\midrule mean'
        else:
            vals = {k: T4[f'{pre}s{s:.2f}'][k] for k, _ in cols4}; lab = f'{s:.2f}'
        b = min(vals.values())
        out.append(lab + ' & ' + ' & '.join((f'\\textbf{{{vals[k]:.3f}}}' if vals[k] == b else f'{vals[k]:.3f}') for k, _ in cols4) + r' \\')
    return '\n'.join(out)
V['t4_head'] = '$s$ & ' + ' & '.join(n for _, n in cols4) + r' \\'
V['t4a'] = t4rows(''); V['t4b'] = t4rows('n2.0_')
# Table: v4 per-setting paired contrasts
rows = []
for s in S9:
    cnd = f's{s:.2f}'; cells = []
    for fam in ['S1_I2_vs_ARX_RLS', 'S6_I2_vs_PI_AW', 'S2_I1_vs_10k', 'S3_I3_vs_10k']:
        r = get4(fam, cnd); cells.append(pci(r['mean'], *r['ci'], d=3))
    rows.append(f'{s:.2f} & ' + ' & '.join(cells) + r' \\')
V['t_pairs'] = '\n'.join(rows)
rows = [f"{r['condition'][1:]} & {pci(r['mean'], *r['ci'], d=2)} & {100 * r['positive_fraction']:.0f}\\% \\\\" for r in sq3]
V['t_q3'] = '\n'.join(rows)
# integrity
V['sha4'] = '422b4078c07f8136bcb887469c1962ac24cf596a3d6dded5addb3a289e825b5c'
TEMPLATE = (HERE / 'main_template.tex').read_text()
out = re.sub(r'<<(\w+)>>', lambda m: V[m.group(1)], TEMPLATE)
missing = re.findall(r'<<\w+>>', out); assert not missing, missing
(HERE / 'main.tex').write_text(out); print('main.tex written,', len(V), 'values')
