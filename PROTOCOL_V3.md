# NeuroLight-AI — Prospective Confirmation v3 (study phase 2)

**Status: DRAFT.** Nothing in this document has touched plants 7000–7049. The machine-readable protocol is `PROTOCOL_V3.json`, generated mechanically from development results by `prospective_dev/make_protocol_v3.py`. The confirmation script is `confirmation_v3.py`. To freeze, review both, set `"status": "FROZEN"` in the JSON, commit, and record the commit hash. The script refuses to run on confirmation plants unless the status is `FROZEN`.

## 1. Relationship to earlier work

The three earlier confirmations (C0: 3000–3049, A: 4000–4049, B: 6000–6049) stay as historical results and are not re-run. Their fixed-PI baseline (Kp 0.02, Ki 0.15) was under-tuned, and their ARX/RLS baseline was effectively non-adaptive. Phase 2 is a separate study that corrects both and adds a finer sensitivity sweep.

## 2. Development evidence used to fix the protocol (plants 5000–5149 only)

| Item | Finding | Script |
|---|---|---|
| Frozen RLS (λ 0.98, P₀ ×1) | Updates on vs off changed mean RMSE by < 0.001 Hz at s ≤ 1.0 and 0.009 Hz at s = 1.2: effectively fixed. | `t1_rls_active.py` |
| Fixed PI, wide grid | Kp 0.005–0.08 × Ki 0.05–1.0; equal-weight optimum (0.02, 0.25) is interior. | `t2_pi.py` |
| Anti-windup PI | Conditional integration was worse than plain PI, so it was rejected. Back-calculation optimum (0.02, 0.25, k_t 0.5) is interior. | `t2_pi.py`, `t2b_pi_bc.py` |
| Adaptive ARX/RLS | P₀ scale 1–10⁹ × λ 0.8–1.0. RMSE falls from 2.46 to ≈ 1.35 Hz. The objective is flat (±0.02 Hz) for P₀ ≥ 10⁷, λ 0.95–0.98; argmin taken. | `t3*_arx.py` |
| DR-H4 / DR-Mem-141 training | The 900-epoch models were not converged. Retrained with a 10 000-epoch budget, same data, seeds and optimiser. | `retrain_chunked.py` |

## 3. Controllers (all receive only observed rate, own past actions and target)

| Name | Definition | Role |
|---|---|---|
| PI_legacy | Kp 0.02, Ki 0.15 (earlier baseline) | continuity only |
| PI_tuned | single global gains from development (0.02, 0.25) | benchmark |
| PI_tuned_AW | same, with back-calculation anti-windup (k_t 0.5) | benchmark |
| PI_oracle | per-sensitivity best development gains; **uses knowledge of s** | upper bound; reported, never tested |
| ARX_fixed | OLS prior on training plants 2000–2099, no online update | P2 |
| ARX_RLS | same prior, online RLS with development-selected P₀ scale and λ | S1 |
| DR_H4 | 9→64→1 tanh, 4 rates + 4 actions + candidate; five retrained initialisations | all |
| DR_Mem141 | 3→141→1 tanh, parameter-matched, same data; five retrained initialisations | P1 |

Predictive controllers share the 101-point one-step action rule: (ŷ − r)² + 2u² + 2(u − u₋₁)².

## 4. Conditions

Sensitivity s ∈ {0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4} at σ = 0.5 (s = 0.5 is outside the 0.6–1.4 training range). Process noise σ ∈ {1, 2} at s = 1.0. The chronology test runs at s = 0.6 and 0.8. That makes 11 conditions × 50 plants, simulated with the exact scalar simulator used in all earlier confirmations.

## 5. Hypotheses and inference

Unit: plant. For learned models, RMSE is averaged over the five initialisations within each plant before differencing. Intervals are BCa bootstrap of the mean paired difference, 10 000 resamples, at the stated level.

| ID | Role | Contrast | Conditions | Level | Decision |
|---|---|---|---|---|---|
| **P1** | Primary: history utility | DR_Mem141 − DR_H4 | s = 0.6, 0.8 | 97.5% each (Bonferroni, 2) | Supported iff both lower bounds > 0 |
| **P2** | Primary: linear sufficiency | ARX_fixed − DR_H4 | s = 0.6 | 95% | Supported iff whole interval within ±0.25 Hz |
| **B1** | Mandatory benchmark: PI robustness | {PI_tuned, PI_tuned_AW} − DR_H4 | all 11 | simultaneous 95% (Bonferroni, 22) | None; no direction assumed, all reported |
| S1 | Secondary: adaptation utility | ARX_RLS − ARX_fixed | all 11 | simultaneous 95% (Bonferroni, 11) | Exploratory; no minimum effect justified |
| S2 | Secondary: regime dependence | mean RMSE and ranking per condition | all 11 | — | Descriptive; crossovers reported, none assumed |
| S3 | Secondary: chronology | DR_H4 joint-shuffle − DR_H4 | s = 0.6, 0.8 | 97.5% each | Plant-level; replaces the C0 cell-level interval |

P1 and P2 are distinct questions and each is controlled at its own level. The ±0.25 Hz margin is carried over unchanged from Experiment B and was not re-chosen after seeing development data. Primary endpoint: RMSE over all 180 steps. Secondary: RMSE over steps 50–179, plus action metrics.

## 6. Execution

1. `python confirmation_v3.py --audit-only` checks hashes and runs a smoke test on development plant 5100. It touches no confirmation plant.
2. Review, set the status to `FROZEN`, commit.
3. `python confirmation_v3.py` runs once (≈ 2–3 h). Plants 7000–7049 are spent after the first completed run, regardless of outcome.
4. Report every controller and condition. No retuning, no model selection, no change of hypotheses or levels.

## 7. Open choices for the reviewer before freezing

- Whether to keep σ ∈ {1, 2} in B1 (they add 4 of the 22 intervals).
- Whether 50 plants is enough. Plant-to-plant SD is about 0.05–0.3 Hz, so intervals will be narrow; more plants mainly guard against rare plant types.
- Whether PI_oracle should be tuned over the back-calculation grid as well as the plain grid.
