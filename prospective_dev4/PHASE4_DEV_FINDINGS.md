# Phase 4: development findings (8 October 2026)

**Development only.** Plants 5200–5249 (a new development block), vectorised simulator. Confirmation block 8000–8049 is untouched. None of this is confirmatory.

## 4.1 Why does adaptive ARX win? (`t41_arx_mechanism.py`, `t41b_scalar_gain.py`)

| s | fixed ARX | full RLS (10 coef.) | RLS on w_u only | RLS on w_u + intercept | **one scalar gain g on all 5 action terms** | g |
|---|---|---|---|---|---|---|
| 0.5 | 2.867 | 2.758 | 2.809 | 2.859 | 2.833 | 0.66 |
| 0.6 | 1.702 | 1.568 | 1.703 | 1.717 | 1.689 | 0.80 |
| 0.8 | 1.287 | 1.093 | 1.299 | 1.698 | 1.281 | 1.05 |
| 1.0 | 1.713 | 1.001 | 1.711 | 1.398 | 1.219 | 1.30 |
| 1.2 | 3.431 | 1.070 | 2.516 | 1.683 | 1.236 | 1.55 |
| 1.4 | 6.023 | 1.273 | 3.292 | 1.952 | 1.286 | 1.81 |
| **mean (9 s)** | 2.452 | 1.334 | — | — | 1.485 | |

(Mean tracking RMSE, Hz, 50 plants.)

- The full RLS's candidate-action coefficient tracks s almost exactly (correlation 1.00; 0.27 at s 0.5 to 0.72 at s 1.4; prior 0.41).
- Adapting that coefficient alone (or with the intercept) recovers little, because the hidden gain also enters through the four lagged-action terms (light gate plus rate smoothing).
- **One scalar g multiplying all five action terms** tracks s linearly (correlation 1.00, g ≈ 1.3 s) and recovers **86%** of the full-RLS improvement over fixed ARX. The remaining gap is mostly at s 0.9–1.2 (about 0.15–0.2 Hz).
- Interpretation: adaptive ARX wins mainly by re-estimating a single multiplicative input gain.

## 4.2 Why does longer training worsen control? (`train_snapshots.py`, `t42_mismatch.py`)

DR-H4 retrained (same data, seeds, optimiser) with raw weights saved at 12 epochs from 300 to 10 000 (60 checkpoints, trained on the laptop).

| epoch | val RMSE (open-loop) | track s 0.6 | 0.8 | 1.0 | 1.2 | 1.4 | dŷ/du s 0.6 | 1.0 | 1.4 |
|---|---|---|---|---|---|---|---|---|---|
| 600 | 1.80 | 1.70 | 1.24 | 1.26 | 1.62 | 2.18 | 23.4 | 24.8 | 25.2 |
| 900 | 1.68 | 1.63 | 1.22 | 1.32 | 1.82 | 2.64 | 23.2 | 24.7 | 25.1 |
| 3000 | 1.46 | 1.59 | 1.29 | 1.78 | 3.04 | 4.81 | 19.6 | 23.9 | 25.2 |
| 9000 | 1.40 | 1.64 | 1.37 | 1.89 | 2.97 | 4.42 | 17.1 | 23.3 | 25.5 |
| reference gain (60·g·w_u) | | | | | | | 19.4 | 31.7 | 44.2 |

(Means over 5 seeds. Epoch-10 000 raw weights include seed 103's transient spike and are excluded here.)

- **Validation error decouples from control as gain rises.** Across the 60 checkpoints, the correlation between validation RMSE and closed-loop RMSE is 0.81 at s 0.6, 0.73 at 0.8, 0.50 at 1.0, 0.18 at 1.2 and −0.08 at 1.4. Closed-loop tracking at s ≥ 1.0 is best around 600–900 epochs.
- **The network does not scale its action sensitivity with the hidden gain.**
  - dŷ/du at the chosen action stays at about 23–25 Hz per unit light across s, while the reference effective gain rises from about 19 to 44.
  - At high s the model underestimates the plant gain by roughly 40%, which would produce over-sized actions.
  - Longer training lowers the low-s sensitivity towards the reference but leaves the high-s underestimate unchanged.
  - The reference is the one-parameter gain-RLS estimate, a proxy rather than the plant's true local gain.
- **Discarded diagnostic.** On-policy one-step error correlated 1.00 with tracking error. That is close to circular: the controller chooses actions whose prediction equals the target, so on-policy prediction error is essentially tracking error. It is not evidence of a mechanism.

## What this suggests for a confirmation (draft protocol v4, `PROTOCOL_V4_DRAFT.md`)

- **4.1:** a one-parameter gain-adaptive ARX should recover most of the full-RLS benefit, and its g should be monotone in s.
- **4.2:** prospectively test (a) the decoupling of validation and closed-loop error with rising gain, and (b) the failure of the learned action sensitivity to scale with s.

Interventions still to develop before v4 is frozen: early stopping on a closed-loop validation set; training on closed-loop (on-policy) data; adding the scalar-gain estimate as an input feature.

## 4.2 interventions (`t43_interventions.py`; evaluation plants 5200–5249; I1 selection on 5250–5299)

| Controller | mean RMSE (9 s) | s 1.0 | s 1.4 | dŷ/du slope (reference ≈ 31) |
|---|---|---|---|---|
| H4 10 000 epochs | 2.289 | 1.913 | 4.464 | 12.2 |
| H4 900 epochs | 1.724 | 1.325 | 2.628 | 2.8 |
| I1: closed-loop early stopping (epochs 300–900) | 1.590 | 1.211 | 2.026 | — |
| I3: own closed-loop data (best-val) | 1.631 | 1.265 | 2.263 | 12.7 |
| **I2: gain estimate g as an extra input (best-val)** | **1.350** | **1.042** | **1.265** | **26.0** |
| ARX + RLS (full), for reference | 1.334 | 1.001 | 1.273 | — |

- **I2** learns to scale its action sensitivity with s and nearly matches the full adaptive ARX.
- Its validation RMSE is 1.04 Hz vs 1.39 Hz for H4 10k, so the gain input improves prediction too.
- Its control improves with training (mean 1.91 at 900 epochs → 1.86 → 1.54 → 1.35 at 6000), so the prediction–control mismatch disappears.
- **I3** shows a residual mismatch (900-epoch snapshot 1.58 vs best-val 1.63).
- **Interpretation (development):** the failure is that the network does not identify the plant gain from its history window; supplying the gain removes it.
