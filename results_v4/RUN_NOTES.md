# Protocol v4 confirmation: run notes (8 October 2026)

- **Protocol:** `PROTOCOL_V4.json`, frozen 2026-10-08T16:44:59Z at commit db2f4db.
- **Plants:** 8000–8049, now **SPENT**. Plant-conditions: 50 plants × 9 sensitivities × 2 noise levels = 900.
- **Where it ran:** the user's laptop, in resumable chunks: `python3 confirmation_v4.py --budget 155`, strictly one invocation at a time.
- **Results file:** `confirmation_v4_results.json`, sha256 `422b4078c07f8136bcb887469c1962ac24cf596a3d6dded5addb3a289e825b5c`.
- **Frozen files at completion:** `confirmation_v4.py`, `PROTOCOL_V4.json` and `MANIFEST_V4.json` matched their frozen md5s. The script also re-checks the manifest on every invocation.

## Interruptions

Three tool calls returned an infrastructure error ("Service Unavailable: server draining"), so their execution status was unknown. They occurred at about 465, 561 and 869 of 900.

Before each resume:
- no process could still be alive, because every call is hard-capped by `timeout 172`;
- the partial file's mtime was more than 3 minutes old and unchanged.

The runner skips completed plant-conditions and saves after each one, so a killed chunk can lose at most one plant-condition in progress, and that one is rerun from scratch.

## Integrity checks (`verify_v4.py`, run in a separate environment from the same frozen files)

- **Plant-conditions recomputed from scratch.** These cover both sides of every interruption point, plus the first and last primary plants:
  - n2.0_s0.50 plants 14, 15, 46 and 47;
  - n2.0_s0.60 plants 10 and 11;
  - n2.0_s1.20 plants 18 and 19;
  - s0.50 plant 0 and s1.40 plant 49.

  All 302 numeric fields matched exactly in every case (max |diff| = 0.0).
- **Analysis recomputed from the stored raw data:** max |diff| = 2.6e-16 across all reported statistics.

## Primary decisions (σ = 0.5)

| Hypothesis | Criterion | Estimate [95% BCa CI] | Decision |
|---|---|---|---|
| Q1: share of the full-RLS gain over fixed ARX recovered by one-parameter gain adaptation | lower bound ≥ 0.70 | 0.871 [0.863, 0.878] | Supported |
| Q3: H4_10k − H4_I2, mean over s ∈ {1.0, 1.1, 1.2, 1.4} | lower bound > 0 | 1.840 Hz [1.809, 1.879] | Supported |
| Q4: [plain e6000 − e900] − [gain-input e6000 − e900], high-gain set | lower bound > 0 | 2.160 Hz [2.125, 2.200] | Supported (differential effect) |

## Secondary and robustness results

These do not change the primary decisions.

- **Q4 components, reported separately:**
  - plain H4: +1.090 Hz [1.063, 1.119], so it gets worse with longer training;
  - gain-input H4: −1.069 Hz [−1.090, −1.052], so it gets better.

  The two directions are opposite, as pre-specified.
- **S_Q2:** the per-plant Spearman correlation between g and s is 1.0 for all 50 plants.
- **Calibration:**
  - slope dg/ds = 1.27 [1.256, 1.287] (99% CI);
  - coefficient of variation of g/s = 0.021.
- **S_Q3 per setting:** H4_10k − H4_I2 is positive at every high-gain setting, from 0.89 Hz at s = 1.0 to 3.21 Hz at s = 1.4. Every plant is positive.
- **S1: H4_I2 − ARX_RLS paired differences.** No equivalence claim was pre-specified.
  - I2 is lower (better) at s = 0.6: −0.037 Hz.
  - I2 is higher (worse) at s 0.9–1.2: +0.024 to +0.052 Hz.
  - Elsewhere the CI includes 0.
- **S6: H4_I2 − PI_tuned_AW.** I2 is better at all 9 sensitivities, by 0.025–0.273 Hz.
- **S2 and S3: I1 and I3 vs H4_10k.**
  - Both are better at s ≥ 0.7.
  - I1 is worse at s = 0.5.
  - Both remain far behind I2 at high s.
- **S4: learned action-sensitivity slope as a fraction of the reference.**
  - H4_10k: 0.39;
  - H4_900: 0.09;
  - H4_I2: 0.84.
- **Robustness at σ = 2.0** (98.3% CIs):
  - Q1 = 0.24 [0.17, 0.32]. This is **below the 0.70 criterion**: under high process noise, one-parameter gain adaptation recovers only a quarter of the full-RLS gain.
  - Q3 = 0.042 Hz [0.033, 0.054]. Positive but small.
  - Q4 = 0.018 Hz [0.008, 0.027]. Positive but small.
