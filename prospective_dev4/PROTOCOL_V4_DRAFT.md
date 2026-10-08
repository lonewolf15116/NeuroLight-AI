# Protocol v4: DRAFT outline (not frozen, not executable yet)

**Status: OUTLINE.** The thresholds below are proposals, set conservatively relative to development values. The interventions for 4.2 are not yet developed. Plants **8000–8049** stay untouched until this protocol, its script and its manifest are frozen by the reviewer.

## Questions

1. **4.1:** Is the adaptive-ARX advantage explained by re-estimating one multiplicative input gain?
2. **4.2:** Why does longer training worsen closed-loop control?

## Proposed hypotheses

All use a 9-point sweep s ∈ {0.5, …, 1.4} at σ = 0.5 and 50 plants. The unit is the plant; intervals are BCa with 10 000 resamples.

| ID | Role | Test | Proposed criterion | Development value |
|---|---|---|---|---|
| Q1 | primary | Fraction of the full-RLS improvement over fixed ARX recovered by scalar-g ARX (mean over the sweep) | 95% CI lower bound ≥ 0.70 | 0.86 |
| Q2 | primary | Plant-level Spearman correlation between final g and s (450 plant-conditions) | ≥ 0.90 | ≈ 1.00 (condition means) |
| Q3 | primary | Closed-loop RMSE, DR-H4 at epoch 9000 minus epoch 900, at s ∈ {1.0, 1.2, 1.4} | lower bound > 0 at all three (Bonferroni over 3) | +0.57, +1.15, +1.78 Hz |
| Q4 | secondary | Slope of the learned dŷ/du against s, divided by the slope of the gain-RLS reference | upper CI bound < 0.5 | about 0.08 |
| Q5 | secondary | Correlation across checkpoints between validation RMSE and closed-loop RMSE, at s 0.6 vs s 1.4 | reported descriptively | 0.81 vs −0.08 |

## Still needed before freezing

- [ ] Develop and select 4.2 interventions on development plants: early stopping on closed-loop validation, on-policy data augmentation, and a g-estimate input feature.
- [ ] Decide whether the interventions enter v4 as hypotheses.
- [ ] Write `confirmation_v4.py` using the exact scalar simulator, with audit and rehearsal modes and a manifest of checkpoint hashes.
- [ ] Reviewer approval; then freeze.
