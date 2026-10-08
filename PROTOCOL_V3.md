# NeuroLight-AI: Prospective Confirmation v3 (study phase 2)

**Status: DRAFT, not frozen.** Plants 7000–7049 have not been instantiated by any code. The machine-readable protocol is `PROTOCOL_V3.json`, generated mechanically from development results by `prospective_dev/make_protocol_v3.py`. Both files describe the same protocol; if they disagree, the JSON is what the script executes. **One decision is still open (§8): which DR-H4 / DR-Mem-141 models enter the confirmation.**

## 1. Historical record (preserved, not overwritten)

- The phase-1 confirmations remain as historical results and are not re-run: C0 (plants 3000–3049), A (4000–4049) and B (6000–6049).
- Their scripts, protocols, weights and results are under `archived_results/`. The repository state before phase 2 is git tag **`phase1-historical`**.
- The original manuscript drafts and the v2 manuscript are separate files and are not modified by v3.
- V3 writes only to `results_v3/`.
- Phase 1 used a fixed PI (Kp 0.02, Ki 0.15) whose integral gain sat on its tuning-grid edge. It also used an ARX/RLS whose online updates were negligible.
- **V3 is a different comparison, not a re-analysis of phase 1.**

## 2. Development evidence (plants 5000–5149 only)

| Step | Finding | Files |
|---|---|---|
| RLS activity check | Frozen phase-1 RLS on vs off changed mean RMSE by < 0.001 Hz (s ≤ 1.0) and 0.009 Hz (s 1.2). It was effectively fixed. | `t1_rls_active.py` |
| Unified PI search | One family: positional PI, integrator clip ±10, back-calculation gain k_t (k_t = 0 is plain PI). Kp {0.005–0.08} × Ki {0.05–1.0} × k_t {0, 0.1, 0.2, 0.5, 1.0} = 400 members. Same 50 tuning plants, all 9 sensitivities. | `t4_pi_unified.py`, `t4/` |
| → PI_tuned | k_t = 0 member with lowest equal-weight mean RMSE: **Kp 0.02, Ki 0.20** (1.645 Hz; runner-up Ki 0.25 at 1.648 Hz). | |
| → PI_tuned_AW | k_t > 0 member, same criterion: **Kp 0.02, Ki 0.25, k_t 0.5** (1.452 Hz). | |
| → PI_sens_ref | Per-sensitivity best member of the whole family (table in JSON). | |
| Adaptive ARX/RLS | P₀ scale 1–10⁹ × λ 0.8–1.0. The objective is flat (6 configs within 0.02 Hz) for P₀ ≥ 10⁷, λ 0.95–0.98; argmin **P₀ ×10⁹, λ 0.98** (1.345 Hz vs 2.46 fixed). | `t3*_arx.py` |
| Model retraining | DR-H4 and DR-Mem-141 retrained for 10 000 epochs (same data, seeds, optimiser). DR-H4 validation RMSE is 1.39–1.40 Hz (was 1.61–1.85 at 900 epochs). Gain over the last 1000 epochs is ≤ 0.007 Hz. | `retrain_chunked.py`, `retrain_10000/` |
| Dress rehearsal | Full v3 pipeline and analysis on evaluation plants 5100–5149 (§7). | `results_v3/dev_rehearsal_results.json` |

Edge flags recorded in the JSON:
- PI_sens_ref at s 0.70 chose k_t = 1.0, the top of the k_t grid.
- The ARX optimum sits at the largest P₀, inside a plateau.
- Three retrained models have their best epoch ≥ 9950, with gains ≤ 0.006 Hz over the last 1000 epochs.

## 3. Controllers

Every controller sees only the observed rate, its own past actions and the target.

| Name | Definition | Role |
|---|---|---|
| PI_legacy | Kp 0.02, Ki 0.15 (phase-1 baseline) | continuity only |
| PI_tuned | Kp 0.02, Ki 0.20, no anti-windup | benchmark B1 |
| PI_tuned_AW | Kp 0.02, Ki 0.25, back-calculation k_t 0.5 | benchmark B1 |
| PI_sens_ref | **Sensitivity-informed tuned PI reference:** per-sensitivity best member of the searched PI family. Uses knowledge of s, so it is not deployable. It is the best PI this search found, **not an upper bound on classical control**. | reported only, never tested |
| ARX_fixed | OLS prior fitted on training plants 2000–2099, no online update | P2 |
| ARX_RLS | same prior, RLS with P₀ ×10⁹ σ̂²(ΦᵀΦ)⁺, λ 0.98 | S1, S4 |
| DR_H4 | 9→64→1 tanh; 4 rates + 4 past actions + candidate; five initialisations | all |
| DR_Mem141 | 3→141→1 tanh, parameter-matched to DR_H4, same data; five initialisations | P1 |

Predictive controllers share the 101-candidate one-step rule (ŷ − r)² + 2u² + 2(u − u₋₁)².

## 4. Conditions and comparisons (these are different counts)

**Conditions: 11.**
- s ∈ {0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4} at σ = 0.5. s = 0.5 lies outside the 0.6–1.4 training range.
- σ ∈ {1.0, 2.0} at s = 1.0.
- 50 plants per condition, giving 550 plant-conditions, plus the chronology runs at s 0.6 and 0.8.

**Comparisons:** each family is corrected only within itself.

| Family | Contrast | Conditions | Comparisons | Interval level | Decision rule |
|---|---|---|---|---|---|
| **P1** (primary): history utility | DR_Mem141 − DR_H4 | s 0.6, 0.8 | 2 | 97.5% each (Bonferroni) | supported iff both lower bounds > 0 |
| **P2** (primary): linear sufficiency | **ARX_fixed** − DR_H4 | s 0.6 | 1 | 95% | supported iff the whole interval lies within ±0.25 Hz |
| **B1** (mandatory benchmark): PI robustness | PI_tuned − DR_H4 and PI_tuned_AW − DR_H4 | all 11 | 22 | 99.77% each (simultaneous 95%) | none; no direction assumed; all reported |
| S1: adaptation utility | ARX_RLS − ARX_fixed | all 11 | 11 | 99.55% each | none (exploratory) |
| S2: regime dependence | mean RMSE and ranking per condition | all 11 | 0 | — | descriptive; crossovers reported, none assumed |
| S3: chronology | DR_H4 joint-shuffle − DR_H4 | s 0.6, 0.8 | 2 | 97.5% each | none; plant-level (replaces the C0 cell-level interval) |
| S4: adaptive linear vs neural | ARX_RLS − DR_H4 | all 11 | 11 | 99.55% each | none (exploratory) |

Totals: 3 primary comparisons, 22 benchmark comparisons, 24 secondary intervals.

**Statistics.**
- Unit: plant. For learned models, RMSE is averaged over the five initialisations within each plant before differencing.
- Intervals are BCa bootstrap intervals of the mean paired difference, with 10 000 resamples.
- Primary endpoint: RMSE over all 180 steps. Secondary: RMSE over steps 50–179, plus action metrics.

**P2 comparator and margin.**
- P2 uses the **fixed** ARX, so it tests whether a linear representation with the same history suffices without online adaptation. The adaptive ARX is assessed separately in S1 and S4.
- The ±0.25 Hz margin is carried over unchanged from the Experiment B protocol, which was locked before its own confirmation. It is about 15% of DR-H4 RMSE at s 0.6 in every phase-1 confirmation, and it was not re-chosen after seeing phase-2 data.

## 5. Integrity: hashes and freezing

- `PROTOCOL_V3.json` records SHA-256 hashes for every model weight file and for the ARX prior. `confirmation_v3.py` aborts on any mismatch.
- **Freezing** (`python freeze_v3.py --i-confirm-freeze`, only after explicit approval):
  1. Sets status to FROZEN and timestamps it.
  2. Writes `MANIFEST_V3.json`. It holds SHA-256 hashes for the confirmation script, the protocol, the simulator modules (`experiment.py`, `v02_robustness.py`, `experiment_b_final.py`), the ARX prior and all ten weight files, plus the git commit.
- The confirmation mode refuses to start unless the status is FROZEN and every file still matches the manifest. The manifest and protocol are committed before the run.
- Current draft hashes are in `results_v3/audit_v3.json`.

## 6. Execution

| Step | Command | Touches |
|---|---|---|
| Audit | `python confirmation_v3.py --audit-only` | one development plant (5100) |
| Rehearsal | `python confirmation_v3.py --dev-rehearsal --budget 160` | development plants 5100–5149 |
| Freeze | `python freeze_v3.py --i-confirm-freeze`, then commit | — |
| Confirm | `python confirmation_v3.py --budget 160`, repeated until COMPLETE | plants 7000–7049 (spent after the first completed run) |

**Runtime:** the rehearsal took 15 minutes on the laptop (about 1.6 s per plant-condition) in six resumable 160-s chunks, and the confirmation is identical in size. Progress is saved after every plant-condition.

## 7. Development rehearsal (plants 5100–5149, current draft models): precision check

The rehearsal confirms that, with 50 plants, interval half-widths at the planned levels are 0.014–0.11 Hz for P1, P2, B1, S1 and S4, and 0.19–0.28 Hz for S3. That is narrow relative to every effect or margin in the protocol, so **50 plants supports the planned inference**.

Selected development values (not confirmatory):

| Condition | PI_tuned | PI_tuned_AW | PI_sens_ref | ARX_fixed | ARX_RLS | DR_H4 (10k) | DR_Mem141 (10k) |
|---|---|---|---|---|---|---|---|
| s 0.6 | 1.869 | 1.660 | 1.473 | 1.689 | 1.586 | 1.647 | 4.077 |
| s 0.8 | 1.257 | 1.142 | 1.128 | 1.287 | 1.078 | 1.398 | 2.410 |
| s 1.0 | 1.096 | 1.075 | 1.075 | 1.714 | 0.982 | 1.921 | 1.751 |
| s 1.2 | 1.163 | 1.245 | 1.058 | 3.337 | 1.090 | 3.009 | 1.928 |
| σ 2.0 | 3.217 | 2.259 | — | 2.198 | 2.128 | 2.155 | 2.980 |

P1 and P2 would be supported on development data:
- P1: +2.43 [2.38, 2.48] and +1.01 [0.97, 1.05] Hz.
- P2: +0.042 [0.027, 0.054] Hz.

ARX_RLS had the lowest mean RMSE in all 11 development conditions.

## 8. Open decision before freezing: which learned models

The rehearsal shows a systematic effect.
- The 10 000-epoch models improve one-step validation error over the phase-1 900-epoch models (DR-H4 1.39 vs 1.61–1.85 Hz).
- But they are **much worse in closed loop at s ≥ 0.8**. At s 1.2 they score 2.72–3.25 Hz across all five initialisations, against 1.82 Hz for the 900-epoch models on the same development plants (earlier vectorised development run, `post_confirmation_dev/`).
- At s 0.6 the two model sets are similar (1.65 vs 1.61 Hz); at s 0.8 the 10k models are already worse (1.40 vs 1.22 Hz).
- This is a development observation of prediction/control objective mismatch.

Options:

| Option | What enters v3 | Consequence |
|---|---|---|
| A (current draft) | 10k-epoch DR_H4 / DR_Mem141 | The selection rule (converged, best validation) was fixed before the rehearsal. DR-H4 will look weak at high gain. |
| B | phase-1 900-epoch models | Better closed loop at high gain. Under-trained by the protocol's own convergence criterion, and the choice would now be made after seeing closed-loop development data. |
| C | both: 10k pair primary (P1, P2, B1, S3, S4) plus 900-epoch pair as secondary S5 (DR_H4_900 − DR_H4_10k, all 11, Bonferroni 11) | Tests the objective-mismatch effect prospectively. Adds about 25% runtime. |

The choice belongs to the reviewer and must be made before freezing; the JSON and script will be regenerated to match.

## 9. Prohibitions

- No tuning or model selection on plants 7000–7049.
- No change of hypotheses, margins or interval levels after freezing.
- Every controller and condition is reported regardless of outcome.
- No biological or clinical claim.
