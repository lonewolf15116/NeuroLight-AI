# NeuroLight-AI roadmap (from 8 October 2026)

## Where the project stands

| Evidence | Status |
|---|---|
| History is necessary for the learned predictor (P1) | **Confirmed**: v3, plants 7000–7049 |
| Fixed linear model with the same history is practically equivalent at s = 0.6 (P2) | **Confirmed**: v3 |
| Tuned PI matches or beats DR-H4 from s = 0.7 upwards (B1) | **Confirmed**: v3 benchmark |
| Online-adapted ARX is the strongest controller (S1, S4) | Secondary, pre-specified: v3 |
| Longer training gives better prediction but worse control (S5) | Secondary, motivated by development data: v3 |
| Mechanisms behind S4 and S5 | **Confirmed (v4, plants 8000–8049)**: one-parameter gain adaptation recovers 87% of the RLS benefit (Q1); a gain-estimate input removes the high-gain failure (Q3) and reverses the training-length effect (Q4). At σ = 2.0 Q1 drops to 0.24 |
| Biological / optogenetic relevance | **Not established**: synthetic plant |

Spent plant blocks (never reuse): 1000–1049, 3000–3049, 4000–4049, 6000–6049, 7000–7049, 8000–8049 (v4). Development plants: 0–69, 1400–1649, 5000–5149, 5200–5299. **Next fresh block: 9000–9049.**

## Phase 3: write up and submit (now → about 3 weeks)

| # | Task | Done when |
|---|---|---|
| 3.1 | ✅ (8 Oct, see LITERATURE_REVIEW.md) Literature search to test novelty. For every close competitor, record the DOI or arXiv link and quote the exact section or experiment that overlaps. Topics: adaptive optogenetic control, neural-population MPC, in-context / history-based system identification, adaptive MPC, objective mismatch, learned vs adaptive control benchmarks. | Table of the 5 closest papers with links and overlap quotes |
| 3.2 | ✅ mostly (Nagabandi ICRA DOI, book ISBNs, Bolus 2021 DOI still to confirm) Verify all references against publisher records | Every reference has a checked DOI or arXiv ID |
| 3.3 | Settle title and framing after 3.1 (draft: *Better Prediction, Worse Control*) | Title fixed; "to our knowledge" only if 3.1 supports it |
| 3.4 | Independent read of manuscript v3 by someone who hasn't seen the work (supervisor or peer) | Comments resolved |
| 3.5 | Pick a venue: an ML-for-control / ML-for-science workshop first, then a neural-engineering or control conference. Check the current calls and deadlines. | Venue and deadline chosen |
| 3.6 | Release: make the repo public, archive a tagged release (e.g. Zenodo DOI), update the README | Public DOI cited in the manuscript |

## Phase 4: explain the two secondary findings (new protocol, plants 8000–8049)

Each item needs its own development work on development plants, then a frozen protocol, before any fresh plant is touched.

| # | Question | Minimal experiment |
|---|---|---|
| 4.1 | Why does adaptive ARX win? Is it re-estimating the effective input gain? | Track the RLS input-gain coefficient against the true s; compare with an ARX that may only adapt that one coefficient |
| 4.2 | Why does longer training worsen control? | (a) Fit predictors on closed-loop rather than open-loop trajectories; (b) measure prediction error on the closed-loop states the controller actually visits; (c) check checkpoints between 900 and 10 000 epochs for a monotone trend |
| 4.3 | Is the result specific to one-step control? | Receding-horizon MPC (H = 3–5) with the same predictors |
| 4.4 | Stronger classical baselines | Gain-scheduled PI driven by the RLS gain estimate |

## Phase 5: external validity (optional; decides journal-level ambition)

| # | Extension |
|---|---|
| 5.1 | More realistic plant: Brian2 recurrent E/I network with a multi-state opsin model |
| 5.2 | Other shifts: time-varying (drifting) gain, bias drift, observation delay (the v0.4 delay bug must be fixed first) |
| 5.3 | Different learned architectures (GRU/sequence model) as a check that S5 isn't architecture-specific |

## Rules that carry forward

- Development on development plants only; freeze protocol, code and weight hashes before a fresh block; one run per block.
- Confirmation runs execute one resumable chunk at a time (no concurrent invocations).
- Secondary or exploratory results are never promoted to primary after the fact.
- No biological or clinical claims from synthetic results.
