// ---------------------------------------------------------------- data (read from the frozen results file)
const R = JSON.parse(fs.readFileSync('/home/claude/repo/results_v3/confirmation_v3_results.json'));
const A = R.analysis, T = A.S2_mean_rmse_and_ranking;
const f3 = x => (x >= 0 ? '+' : '−') + Math.abs(x).toFixed(3);
const g3 = x => x.toFixed(3);
const ci = r => `${f3(r.mean)} [${f3(r.ci[0])}, ${f3(r.ci[1])}]`;
const get = (fam, cond, contrast) => A[fam].find(r => r.condition === cond && (!contrast || r.contrast === contrast));
const CONDS = ['s0.50', 's0.60', 's0.70', 's0.80', 's0.90', 's1.00', 's1.10', 's1.20', 's1.40', 'n1.0', 'n2.0'];
const lab = c => c.startsWith('s') ? 's = ' + c.slice(1) : 'σ = ' + c.slice(1) + ' (s = 1.00)';
const p1a = get('P1_history', 's0.60'), p1b = get('P1_history', 's0.80'), p2 = get('P2_linear_sufficiency', 's0.60');
const s3a = get('S3_chronology', 's0.60'), s3b = get('S3_chronology', 's0.80');

const C = [];
C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [new TextRun({ text: 'Better Prediction, Worse Control: Temporal History, Linear Adaptation and Regime Dependence in Learned Control of a Synthetic Neural Population', bold: true, size: 32 })] }));
C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [new TextRun({ text: 'Mahesh Reddy Pagadala', size: 22 })] }));
C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 }, children: [new TextRun({ text: 'Manuscript draft v3 — 8 October 2026 — results from the frozen prospective confirmation (protocol v3)', size: 18, italics: true, color: '555555' })] }));
C.push(NOTE('**Drafting notes (remove before submission).** All numbers in Section 4 are generated directly from results_v3/confirmation_v3_results.json (sha256 7273…b58b). The provisional title is to be confirmed once the literature review in the roadmap is complete. References must be verified against publisher records.'));

// ---------------------------------------------------------------- Abstract
C.push(H1('Abstract'));
C.push(P(`Learned predictive controllers are often assumed to improve as their models become more accurate and more expressive. We test this in a synthetic population of 64 leaky integrate-and-fire neurons driven by a common phenomenological light input, under hidden multiplicative shifts *s* of light sensitivity. Five controller families — fixed PI with and without anti-windup, fixed and recursively adapted ARX predictors, and domain-randomised neural predictors with and without a 200-ms input–output history — were compared under a protocol frozen before 50 untouched confirmation plants were simulated, across nine sensitivities and two noise levels. Removing the history from a parameter-matched neural predictor increased tracking RMSE by ${p1a.mean.toFixed(2)} Hz at *s* = 0.60 (97.5% CI ${p1a.ci[0].toFixed(2)}–${p1a.ci[1].toFixed(2)}), and a fixed linear model with the same history was practically equivalent to the neural predictor (difference ${p2.mean.toFixed(3)} Hz, 95% CI ${p2.ci[0].toFixed(3)}–${p2.ci[1].toFixed(3)}; margin ±0.25 Hz). Online-adapted ARX had the lowest mean RMSE in ten of eleven conditions and beat the neural predictor in nine. Well-tuned fixed PI matched or beat the neural predictor at every sensitivity from 0.70 upwards. In a secondary comparison, neural predictors trained for 10 000 epochs had lower one-step prediction error than 900-epoch predictors but tracked worse in closed loop at every *s* ≥ 0.70, by up to ${Math.abs(get('S5_training_length','s1.40').mean).toFixed(2)} Hz. In this system, temporal information and online adaptation matter more than nonlinear capacity, and better one-step prediction does not imply better control.`));

// ---------------------------------------------------------------- 1
C.push(H1('1. Introduction'));
C.push(P('Closed-loop control of neural population firing rates is used to probe circuit dynamics and is a building block of proposed neural-interface systems. In optogenetic settings the effective plant gain can drift with opsin expression, optical attenuation and excitability, while the controller observes only the rate it produces. Learned dynamics models combined with sampling-based action selection are an attractive way to handle such systems, but whether their extra capacity helps — and whether improving their predictive accuracy improves control — is an empirical question. This work studies it in a synthetic plant only and makes no biological claim.'));
C.push(P('We ask four questions under hidden gain shifts: (Q1) does a learned one-step predictor need temporal input–output history, once training data and capacity are matched; (Q2) does a linear predictor with the same information suffice; (Q3) how do learned and linear predictive controllers compare with properly tuned classical feedback and with online linear adaptation; and (Q4) does a more accurate one-step predictor yield a better closed-loop controller?'));
C.push(P('Contributions. (i) A matched history ablation and an aligned-pair chronology perturbation. (ii) A pre-specified practical-equivalence test of linear sufficiency. (iii) A regime-stratified comparison against fixed PI tuned on development plants, with and without anti-windup, and against recursively adapted ARX. (iv) A prospective, pre-registered demonstration that longer training of the neural predictor lowers one-step error while worsening closed-loop tracking. (v) A documented methodological history: an earlier phase of this study used an under-tuned PI baseline and an RLS whose updates were negligible; both were identified, corrected in development, and replaced before a fresh confirmation.'));

// ---------------------------------------------------------------- 2
C.push(H1('2. Related Work'));
C.push(P('**Closed-loop optogenetic control.** PI-type optogenetic feedback control of population activity has been demonstrated experimentally [1], and model-based designs using identified input–output models, including robustness to gain mismatch, have been developed for in-vivo use [2, 3]. These works motivate the problem; they are not validated here.'));
C.push(P('**Identification and adaptive control.** ARX models with recursive least squares and certainty-equivalence adaptive control are classical tools [4, 5]; PI tuning and integrator anti-windup are covered in [6].'));
C.push(P('**Learned dynamics and objective mismatch.** Domain randomisation trains across simulated parameter variation [7]; neural dynamics models with sampling-based action selection are a common model-based control recipe [8]. That one-step model accuracy and control performance can diverge has been argued in model-based reinforcement learning [9].'));
C.push(P('**Novelty boundary.** We do not introduce any of these methods. The contribution is a controlled, pre-registered characterisation in one synthetic plant family of when history, nonlinearity and adaptation matter, together with a prospective test of prediction–control mismatch. A systematic literature search is still required before any “to our knowledge” claim.'));

// ---------------------------------------------------------------- 3
C.push(H1('3. Methods'));
C.push(H2('3.1 Plant'));
C.push(P('The plant is a population of *N* = 64 leaky integrate-and-fire-like neurons sharing a scalar light command *u* ∈ [0, 1]. Each neuron has initial potential *V*ᵢ(0) ~ U(−65, −55), bias *b*ᵢ ~ U(7, 12) and light sensitivity *κ*ᵢ ~ U(18, 26), drawn from a generator seeded by the plant seed (model voltage units). Each 50-ms control step is integrated with 50 Euler sub-steps of 1 ms. A phenomenological first-order gate (time constant 10 ms) filters the command, *g* ← *g* + (*u* − *g*)/10, and each non-refractory neuron updates as'));
C.push(EQ('Vᵢ ← Vᵢ + [ −(Vᵢ + 65) + bᵢ + s·κᵢ·g ] / 20 + σ·ξ,   ξ ~ N(0, 1)'));
C.push(P('A neuron reaching −50 spikes, resets to −65 and is refractory for 2 ms. The observed output is the exponentially smoothed population rate *y*ₜ = 0.5 *y*ₜ₋₁ + 0.5 *r*ₜ (Hz per neuron). The hidden sensitivity multiplier *s* scales every *κ*ᵢ; σ is per-sub-step process noise (nominal 0.5). No controller receives *s* or σ. Each episode lasts 180 steps (9 s) and tracks 12, 30, 20, 40, 15 and 35 Hz, each held for 30 steps. The gate is not an opsin kinetic model.'));
C.push(H2('3.2 Training data'));
C.push(P('All learned and linear predictors were fitted to plants 2000–2099 (training) and 2100–2129 (validation), 300 steps each, with per-plant σ ~ U(0.5, 2), *s* ~ U(0.6, 1.4) and bias shift ~ U(−2, 2), and the light command resampled from U(0, 1) every four steps. Rates are scaled by 1/60 for model input and output.'));
C.push(H2('3.3 Controllers'));
C.push(table([
  ['Controller', 'Definition', 'Selection'],
  ['DR-H4 (primary)', '9 → 64 tanh → 1; input: 4 past rates, 4 past actions, candidate u', '10 000-epoch Adam budget, best-validation checkpoint; 5 initialisations'],
  ['DR-H4-900 (secondary)', 'same architecture, unmodified phase-1 weights', '900-epoch budget; 5 initialisations'],
  ['DR-Mem-141', '3 → 141 → 1 tanh (706 parameters, matched to DR-H4); input: current rate, previous action, candidate', '10 000-epoch budget; 5 initialisations'],
  ['ARX fixed', 'linear in [u, 4 rates, 4 actions, 1]; OLS on training data', 'no online update'],
  ['ARX + RLS', 'same prior; recursive least squares', 'P₀ = 10⁹·σ̂²(ΦᵀΦ)⁺, λ = 0.98, from a development grid'],
  ['PI tuned', 'positional PI, integrator clip ±10', 'Kp 0.02, Ki 0.20 (development grid)'],
  ['PI tuned + AW', 'as above with back-calculation anti-windup', 'Kp 0.02, Ki 0.25, kₜ 0.5 (development grid)'],
  ['PI sens. ref.', 'per-sensitivity best member of the PI family', 'uses knowledge of s; reference only'],
  ['PI legacy', 'Kp 0.02, Ki 0.15 (phase-1 baseline)', 'continuity only'],
], [2000, 3900, 3125]));
C.push(P(''));
C.push(P('All predictive controllers apply the action minimising (ŷₜ₊₁(*u*) − *r*ₜ)² + 2*u*² + 2(*u* − *u*ₜ₋₁)² over 101 candidates in [0, 1]; this is one-step grid-search predictive control, not MPC or reinforcement learning. The back-calculation PI updates I ← clip(I + 0.05e + kₜ(u_sat − u_raw)/Kᵢ, −10, 10). The sensitivity-informed PI reference is the best PI found by our search when *s* is known; it is not deployable and not an upper bound on classical control.'));
C.push(H2('3.4 Study design and protocol'));
C.push(P('The study proceeded in two phases (Appendix A). In phase 1, three confirmations (plants 3000–3049, 4000–4049 and 6000–6049) used a fixed PI whose integral gain lay on the edge of its tuning grid and an ARX/RLS whose online updates proved negligible (switching them off changed RMSE by < 0.01 Hz). Phase 2 corrected both on development plants only (5000–5149): a unified PI search over 400 gain combinations, an adaptive-RLS search, retraining of the neural predictors to a 10 000-epoch budget, and a full rehearsal of the analysis. The protocol was then frozen, with SHA-256 hashes of the script, protocol, simulator modules, ARX prior and all fifteen weight files recorded, before plants 7000–7049 were simulated once with the exact scalar simulator.'));
C.push(P('The 900-epoch arm and its comparison were added after the development rehearsal showed that the 10 000-epoch models, despite lower validation error, tracked worse at higher sensitivity. Their primary status was fixed in advance and does not depend on that comparison.'));
C.push(H2('3.5 Hypotheses and statistics'));
C.push(P('Conditions: *s* ∈ {0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4} at σ = 0.5 (0.5 lies outside the training range), plus σ ∈ {1, 2} at *s* = 1; 50 plants each. The unit is the plant: learned-model RMSE (180 steps) is averaged over the five initialisations within each plant before forming paired differences. Intervals are BCa bootstrap intervals [10] of the mean paired difference (10 000 resamples), with Bonferroni correction within each family.'));
C.push(table([
  ['Family', 'Contrast', 'Conditions', 'Level', 'Decision'],
  ['P1 (primary)', 'DR-Mem-141 − DR-H4', 's 0.6, 0.8', '97.5% each', 'both lower bounds > 0'],
  ['P2 (primary)', 'ARX fixed − DR-H4', 's 0.6', '95%', 'interval within ±0.25 Hz'],
  ['B1 (benchmark)', 'PI tuned / PI tuned + AW − DR-H4', 'all 11', '99.77% each', 'reported, no direction'],
  ['S1', 'ARX + RLS − ARX fixed', 'all 11', '99.55% each', 'exploratory'],
  ['S3', 'DR-H4 joint shuffle − DR-H4', 's 0.6, 0.8', '97.5% each', 'exploratory'],
  ['S4', 'ARX + RLS − DR-H4', 'all 11', '99.55% each', 'exploratory'],
  ['S5', 'DR-H4-900 − DR-H4', 'all 11', '99.55% each', 'exploratory; never primary'],
], [1500, 2900, 1500, 1400, 1725]));
C.push(P(''));
C.push(P('The ±0.25-Hz margin was carried over unchanged from the phase-1 protocol. The joint shuffle applies one random permutation to the four historical (rate, action) pairs at every step, preserving pairing but destroying order.'));

// ---------------------------------------------------------------- 4 Results
C.push(H1('4. Results'));
C.push(P('Both primary hypotheses were supported. Table 1 and Figure 1 give mean tracking RMSE for every controller and condition.'));
C.push(CAP('**Table 1.** Mean tracking RMSE (Hz) over 50 confirmation plants (7000–7049). Learned models averaged over five initialisations. Lowest deployable controller in bold.'));
const cols = [['PI_tuned', 'PI'], ['PI_tuned_AW', 'PI+AW'], ['ARX_fixed', 'ARX'], ['ARX_RLS', 'ARX+RLS'], ['DR_H4', 'DR-H4'], ['DR_H4_900', 'DR-H4-900'], ['DR_Mem141', 'Mem-141'], ['PI_sens_ref', 'PI ref.†']];
const rows1 = [['Condition', ...cols.map(c => c[1])]];
for (const c of CONDS) {
  const dep = cols.filter(k => k[0] !== 'PI_sens_ref').map(k => T[c][k[0]]); const best = Math.min(...dep);
  rows1.push([lab(c), ...cols.map(([k]) => T[c][k] === undefined ? '—' : (k !== 'PI_sens_ref' && T[c][k] === best ? `**${g3(T[c][k])}**` : g3(T[c][k])))]);
}
C.push(table(rows1, [1520, 930, 930, 930, 1000, 930, 1050, 930, 1086]));
C.push(new Paragraph({ children: runs('† Uses knowledge of s; not deployable, excluded from bold. PI legacy (phase-1 gains) is reported in the results file.', { size: 17, color: '555555' }), spacing: { before: 60, after: 160 } }));
C.push(new Paragraph({ alignment: AlignmentType.CENTER, children: [new ImageRun({ type: 'png', data: fs.readFileSync('fig1.png'), transformation: { width: 600, height: 355 } })] }));
C.push(CAP('**Figure 1.** Mean tracking RMSE against hidden sensitivity (σ = 0.5). The memoryless network (2.4–5.6 Hz) is omitted for scale; values for every controller are in Table 1.'));

C.push(H2('4.1 Temporal history is necessary for the learned predictor (P1, S3)'));
C.push(P(`With training data and parameter count matched, removing the older history increased RMSE by ${ci(p1a)} Hz at *s* = 0.60 and ${ci(p1b)} Hz at *s* = 0.80 (97.5% intervals), on every one of the 50 plants. P1 was supported. Applying a joint permutation to the history raised RMSE by ${ci(s3a)} and ${ci(s3b)} Hz at the same sensitivities, again on every plant. Because permuted histories never occur in training, this shows that the trained controller depends on temporal order, not that it estimates the plant gain.`));
C.push(H2('4.2 A linear predictor with the same history suffices at low gain (P2)'));
C.push(P(`At *s* = 0.60 the fixed ARX model differed from DR-H4 by ${ci(p2)} Hz (95%), entirely within the ±0.25-Hz margin; P2 was supported. The interval excludes zero, so the linear model is very slightly but reliably worse. Away from low gain the fixed ARX degrades sharply (${g3(T['s1.20'].ARX_fixed)} Hz at *s* = 1.20, ${g3(T['s1.40'].ARX_fixed)} at 1.40).`));
C.push(H2('4.3 Tuned PI matches or beats the learned predictor across most of the range (B1)'));
C.push(CAP('**Table 2.** PI minus DR-H4 RMSE (Hz), simultaneous 95% intervals (99.77% each). Negative values favour PI.'));
const rows2 = [['Condition', 'PI tuned − DR-H4', 'PI + AW − DR-H4']];
for (const c of CONDS) rows2.push([lab(c), ci(get('B1_PI_benchmark', c, 'PI_tuned - DR_H4')), ci(get('B1_PI_benchmark', c, 'PI_tuned_AW - DR_H4'))]);
C.push(table(rows2, [2300, 3360, 3365]));
C.push(P(''));
const aw06 = get('B1_PI_benchmark', 's0.60', 'PI_tuned_AW - DR_H4'), aw07 = get('B1_PI_benchmark', 's0.70', 'PI_tuned_AW - DR_H4'), awn2 = get('B1_PI_benchmark', 'n2.0', 'PI_tuned_AW - DR_H4');
C.push(P(`DR-H4 retained an advantage only where the PI was saturated or noise-limited: at *s* ≤ 0.60 against both PIs and at 0.70 against plain PI (largest against plain PI, which winds up at *s* = 0.50) and at σ = 2, where the zero-light rate floor exceeds the 12-Hz target (anti-windup PI ${f3(awn2.mean)} Hz). Against anti-windup PI the low-gain advantage was small (${f3(aw06.mean)} Hz at *s* = 0.60). From *s* = 0.70 upwards anti-windup PI was better on every plant, by ${Math.abs(aw07.mean).toFixed(2)} Hz at 0.70 rising to ${Math.abs(get('B1_PI_benchmark', 's1.40', 'PI_tuned_AW - DR_H4').mean).toFixed(2)} Hz at 1.40.`));
C.push(H2('4.4 Online linear adaptation is the strongest controller (S1, S4)'));
const s4n = CONDS.filter(c => get('S4_adaptive_linear_vs_H4', c).ci[1] < 0).length;
C.push(P(`Enabling recursive least squares improved the fixed ARX in every condition, from ${Math.abs(get('S1_adaptation','n2.0').mean).toFixed(2)} Hz at σ = 2 to ${Math.abs(get('S1_adaptation','s1.40').mean).toFixed(2)} Hz at *s* = 1.40, with the largest gains where the fixed model was most mismatched. Adaptive ARX beat DR-H4 in ${s4n} of 11 conditions with the interval excluding zero and was statistically indistinguishable from it in the remaining two (*s* = 0.50 and σ = 2). It had the lowest mean RMSE among deployable controllers in ten of eleven conditions (at *s* = 1.40 legacy PI was lower by ${(T['s1.40'].ARX_RLS - T['s1.40'].PI_legacy).toFixed(3)} Hz), and came within 0.03 Hz of the sensitivity-informed PI reference at *s* = 1.20 (${g3(T['s1.20'].ARX_RLS)} vs ${g3(T['s1.20'].PI_sens_ref)} Hz).`));
C.push(H2('4.5 Better prediction, worse control (S5; secondary)'));
const s5 = c => get('S5_training_length', c);
C.push(P(`The 10 000-epoch DR-H4 models had lower one-step validation RMSE than the 900-epoch models (1.39–1.40 vs 1.61–1.85 Hz). In closed loop, the 900-epoch models were better at every *s* ≥ 0.70 and at σ = 1, by ${f3(s5('s0.70').mean).replace('−','')} Hz at *s* = 0.70, ${Math.abs(s5('s1.00').mean).toFixed(2)} Hz at 1.00 and ${Math.abs(s5('s1.40').mean).toFixed(2)} Hz at 1.40 (all intervals excluding zero; all 50 plants at *s* ≥ 0.80). The two were within 0.02 Hz at *s* ≤ 0.60 and at σ = 2. This comparison was added because of a development observation, it is secondary, and it does not alter the primary status of the 10 000-epoch models. Prediction and control metrics are reported separately; we do not infer a causal mechanism from their divergence.`));

// ---------------------------------------------------------------- 5
C.push(H1('5. Discussion'));
C.push(P('**What matters in this system is information and adaptation, not nonlinear capacity.** History is indispensable for a learned predictor facing a hidden gain (Q1), but once the history is available a linear model performs as well at low gain (Q2), and letting that linear model adapt online makes it the best controller tested across almost the entire range (Q3). A plausible explanation is structural: the hidden shift is a multiplicative input gain, which a linear model can re-estimate from a few recent steps, whereas a fixed network must have encoded the whole family of plants in its weights. We have not tested this explanation directly.'));
C.push(P('**Classical feedback is a hard baseline.** The apparent low-gain advantage of the learned controller in phase 1 largely disappears once PI gains are tuned on the same development data and anti-windup is added. The residual advantages occur where actuator limits or the rate floor bind. Any claim that a learned controller beats classical control should therefore state how the classical baseline was tuned.'));
C.push(P('**A more accurate model made a worse controller (Q4).** Longer training reduced one-step validation error yet worsened closed-loop tracking at most sensitivities, consistent with the objective-mismatch argument [9]. Possible contributors include over-fitting to the open-loop excitation used for training, which differs from closed-loop trajectories, and sharper prediction surfaces that the one-step action rule exploits. These are hypotheses for further work, not conclusions of this study.'));

// ---------------------------------------------------------------- 6
C.push(H1('6. Limitations'));
[
  '**Synthetic plant.** Uncoupled LIF-like neurons with a phenomenological light gate; no recurrence, inhibition, plasticity, spatial light distribution or opsin kinetics. One plant family supplies both training and test plants; the shift is a scalar gain plus process noise.',
  '**Controller scope.** One-step action selection only (no receding-horizon MPC); PI tuned by grid search within one family (no gain scheduling or robust synthesis); a single learned architecture.',
  '**Secondary results.** Adaptation (S1, S4) and training-length (S5) comparisons are secondary; S5 was motivated by development data.',
  '**Development choices.** The adaptive-RLS optimum lies on a broad plateau at the largest tested prior covariance; one sensitivity-informed PI gain lies on a grid edge. Neither is used for a primary claim.',
  '**Run integrity.** Two resumable chunks of the confirmation were launched concurrently by mistake; every affected plant was recomputed from the frozen code and matched exactly (Appendix B).',
].forEach(t => C.push(B(t)));

// ---------------------------------------------------------------- 7
C.push(H1('7. Conclusion'));
C.push(P('In a pre-registered comparison on untouched plants, short input–output history was necessary for a learned predictive controller to cope with hidden gain shifts; a linear model with the same history was practically equivalent at low gain; online linear adaptation outperformed the learned predictor across almost all conditions; well-tuned PI matched or beat it from moderate gain upwards; and training the predictor longer improved its predictions while worsening its control. For this class of problem, information, adaptation and a fairly tuned baseline matter more than model expressiveness.'));
C.push(H1('Code and data availability'));
C.push(P('Code, frozen protocols (PROTOCOL_V3.json, MANIFEST_V3.json), model weights, raw per-plant results and development records: github.com/lonewolf15116/NeuroLight-AI (currently private). The phase-1 state is preserved on branch phase1-historical.'));

C.push(H1('References'));
[
  'Newman JP, Fong M, Millard DC, Whitmire CJ, Stanley GB, Potter SM. Optogenetic feedback control of neural activity. *eLife* 4:e07192, 2015. doi:10.7554/eLife.07192',
  'Bolus MF, Willats AA, Whitmire CJ, Rozell CJ, Stanley GB. Design strategies for dynamic closed-loop optogenetic neurocontrol in vivo. *Journal of Neural Engineering* 15(2):026011, 2018. doi:10.1088/1741-2552/aaa506',
  'Grosenick L, Marshel JH, Deisseroth K. Closed-loop and activity-guided optogenetic control. *Neuron* 86(1):106–139, 2015. doi:10.1016/j.neuron.2015.03.034',
  'Åström KJ, Wittenmark B. *Adaptive Control*, 2nd ed. Addison-Wesley, 1995.',
  'Ljung L. *System Identification: Theory for the User*, 2nd ed. Prentice Hall, 1999.',
  'Åström KJ, Hägglund T. *PID Controllers: Theory, Design, and Tuning*, 2nd ed. ISA, 1995.',
  'Tobin J, Fong R, Ray A, Schneider J, Zaremba W, Abbeel P. Domain randomization for transferring deep neural networks from simulation to the real world. *IROS*, 2017. arXiv:1703.06907',
  'Nagabandi A, Kahn G, Fearing RS, Levine S. Neural network dynamics for model-based deep reinforcement learning with model-free fine-tuning. *ICRA*, 2018. arXiv:1708.02596',
  'Lambert N, Amos B, Yadan O, Calandra R. Objective mismatch in model-based reinforcement learning. *L4DC*, 2020. arXiv:2002.04523',
  'Efron B. Better bootstrap confidence intervals. *Journal of the American Statistical Association* 82(397):171–185, 1987.',
].forEach(t => C.push(N(t, 'refs')));

// ---------------------------------------------------------------- Appendices
C.push(H1('Appendix A. Study history'));
C.push(table([
  ['Stage', 'Plants', 'Outcome'],
  ['v0.1–v0.6 (exploratory)', '0–69, 1000–1049, 1400–1649', 'Learned controller brittle to gain shift; history and domain randomisation help'],
  ['C0 confirmation', '3000–3049', 'H4 vs legacy PI and nominal memoryless; chronology +2.1 / +5.4 Hz'],
  ['Experiment A', '4000–4049', 'Matched history ablation: +2.76 Hz at s = 0.6'],
  ['Experiment B', '6000–6049', 'Linear equivalence at s = 0.6 (+0.063 Hz); PI baseline later found under-tuned, RLS inactive'],
  ['Phase-2 development', '5000–5149', 'Unified PI search, adaptive RLS, 10 000-epoch retraining, rehearsal'],
  ['v3 confirmation (this paper)', '7000–7049', 'Frozen protocol; all results in Section 4'],
], [2600, 2600, 3825]));
C.push(P(''));
C.push(P('Phase-1 results are retained as historical records and are not combined with v3. Each confirmation plant block was used exactly once.'));
C.push(H1('Appendix B. Run integrity'));
C.push(P('The confirmation ran in nine resumable invocations on a laptop. Two invocations were accidentally launched concurrently and wrote the same progress file. The simulator is deterministic per plant and each write is a complete snapshot, so the only possible effect is duplicated work. All 200 plant-conditions in the four affected conditions (s = 0.80–1.10) were recomputed from the frozen code; the maximum absolute RMSE difference from the recorded values was 0.0.'));

const doc = new Document({
  creator: 'Mahesh Reddy Pagadala', title: 'Better Prediction, Worse Control — manuscript draft v3',
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { size: 28, bold: true, font: FONT, color: '1F3A5F' }, paragraph: { spacing: { before: 300, after: 120 }, outlineLevel: 0, keepNext: true } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { size: 23, bold: true, font: FONT, color: '1F3A5F' }, paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 1, keepNext: true } },
    ] },
  numbering: { config: [
    { reference: 'bul', levels: [{ level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
    { reference: 'refs', levels: [{ level: 0, format: LevelFormat.DECIMAL, text: '[%1]', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 600, hanging: 460 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1300, bottom: 1300, left: 1300, right: 1300 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, color: '777777' })] })] }) },
    children: C }]
});
Packer.toBuffer(doc).then(b => { fs.writeFileSync('NeuroLight_AI_Manuscript_v3.docx', b); console.log('written'); });
