# Literature review and novelty check (roadmap 3.1–3.2), 8 October 2026

**Method.** We searched arXiv, bioRxiv, PubMed/PMC, IEEE and publisher pages in two clusters:
- neural and optogenetic closed-loop control;
- learned vs classical/adaptive control, history-based identification, and objective mismatch.

Every paper below was opened at the link given. Quotes are from the fetched pages (abstract unless stated) and should be re-checked against the PDFs before quoting in the paper.

**Limitations of the search.** Google Scholar was not queried directly. Several PMC and eLife pages were bot-blocked. Classical "identification for control" texts were not opened. So this is a strong first pass, not an exhaustive review.

## The closest work

| # | Paper | What overlaps | What it does not do (our addition) |
|---|---|---|---|
| 1 | Fehrman C, Meliza CD. *Model Predictive Control on the Neural Manifold.* arXiv:2406.14801 (v2 2025). https://arxiv.org/abs/2406.14801 | Learned latent dynamics model plus MPC for simulated spiking networks, compared with PID. "MPC consistently produced more accurate control and required less hyperparameter tuning." | No hidden or varied plant gain, no adaptive linear baseline, no prediction-vs-control measure. Its conclusion (MPC beats PID) contrasts with ours (tuned PI matches or beats a learned one-step predictor above s = 0.7), so we must explain the differences: multi-step MPC vs one-step, their PID tuning vs our development-tuned PI with anti-windup, partial observability. |
| 2 | Fehrman C, Meliza CD. *Nonlinear Model Predictive Control of a Conductance-Based Neuron Model via Data-Driven Forecasting.* arXiv:2312.14274. https://arxiv.org/abs/2312.14274 | Data-driven forecaster with a time-delay (history) embedding, used for MPC of a single model neuron. A proportional controller beat MPC in one task; the authors stress the method "is not universally superior to simpler alternative methods". | Single neuron, no hidden gain shifts, no adaptive linear baseline, no matched history ablation. It supports our "simple feedback is competitive" finding. |
| 3 | Bolus MF, Willats AA, Rozell CJ, Stanley GB. *State-space optimal feedback control of optogenetically driven neural activity.* J Neural Eng 18(3), 2021; doi:10.1088/1741-2552/abb89c (bioRxiv 10.1101/2020.06.25.171785) | LQR with integral action and a parameter-adaptive Kalman filter for in vivo optogenetic rate control. Adaptation estimates a drifting disturbance. | Experimental (stronger than ours on realism). Gains and input matrix not adapted (the authors suggest re-estimating B or C as future work), no PI comparison, no learned nonlinear predictor. Our RLS-ARX re-estimates the input gain, which is the direction they suggest. |
| 4 | Lambert N, Amos B, Yadan O, Calandra R. *Objective Mismatch in Model-based Reinforcement Learning.* L4DC 2020, PMLR 120:761–770. arXiv:2002.04523 | Direct precedent for "better prediction, worse control". The abstract states one-step likelihood "is not always correlated with control performance"; §4.2 shows model loss and reward diverging during training (cartpole). | Not a hidden-gain plant, no classical or adaptive baselines, no pre-registered replication. **Our S5 is a controlled, prospectively replicated instance of a known phenomenon, not a discovery of it.** |
| 5 | Kunapuli P, Welde J, Jayaraman D, Kumar V. *Leveling the Playing Field: Carefully Comparing Classical and Learned Controllers for Quadrotor Trajectory Tracking.* arXiv:2506.17832 (2025) | Shows that correcting asymmetries (task access, tuning data, feedforward) shrinks learned-vs-classical gaps: "the gaps between the two controller classes are much smaller than previously published." | Quadrotor, RL policy vs geometric control. Supports our baseline-tuning argument; our phase-1 → phase-2 correction is a concrete case. |

**Counterexample to address.** Jian T, Dai T, Yu T. *Learning Nonlinear Systems In-Context: From Synthetic Data to Real-World Motor Control.* arXiv:2602.07173 (2026). An in-context transformer outperforms PI and physics-based feedforward on real motors; PI tuning is not described in the abstract. In the paper we should note that in-context learned models can beat PI on strongly nonlinear plants, and that our plant's shift is a multiplicative gain, which linear adaptation handles well.

## Other related work to cite

| Paper | Use |
|---|---|
| Yu W, Tan J, Liu CK, Turk G. *Preparing for the Unknown: Learning a Universal Policy with Online System Identification.* arXiv:1702.02453 | History-based online identification of hidden parameters under domain randomisation |
| Kumar A, Fu Z, Pathak D, Malik J. *RMA: Rapid Motor Adaptation for Legged Robots.* arXiv:2107.04034 | Canonical implicit adaptation from history |
| Nagabandi A et al. *Learning to Adapt in Dynamic, Real-World Environments Through Meta-Reinforcement Learning.* arXiv:1803.11347 | Dynamics models adapted from recent experience |
| Grimm C, Barreto A, Singh S, Silver D. *The Value Equivalence Principle for Model-Based Reinforcement Learning.* arXiv:2011.03506 | Theory: models should serve control, not prediction |
| Newman et al. 2015 (eLife) | PI optogenetic control; light requirement varied widely across preparations, which motivates hidden gain |

Lower-overlap items found (RL for deep brain stimulation, Koopman-MPC for seizure suppression, meta-learned neural MPC) do not compare against adaptive linear or tuned PI baselines. Two of them still need their author lists confirmed before citing.

## Novelty assessment

- **Not novel:** history-based identification and implicit adaptation; prediction–control mismatch; learned MPC for neural populations; classical baselines being competitive when tuned fairly.
- **Defensible ("to our knowledge", in this setting):** a pre-registered, prospectively confirmed comparison, under hidden gain shifts in neural-population rate control, of
  - a history-conditioned learned predictor,
  - a parameter-matched memoryless predictor,
  - fixed and RLS-adaptive linear predictors with the same information,
  - development-tuned PI with anti-windup,
  together with a prospective replication of prediction–control mismatch from training length.
- **Do not claim:** that we discovered objective mismatch; that adaptive linear control beating learned control is new in general; any first-ever statement.

**Title.** "Better Prediction, Worse Control" reads as a discovery claim given Lambert et al. 2020. A safer option: *Temporal History, Linear Adaptation and Prediction–Control Mismatch in Learned Control of a Synthetic Neural Population.*

## Reference verification (3.2)

| Ref | Status |
|---|---|
| Newman 2015 | Verified (Crossref) |
| Bolus 2018 | Verified (Crossref) |
| Grosenick 2015 | Verified; title in title case: "Closed-Loop and Activity-Guided Optogenetic Control" |
| Åström & Wittenmark 1995 | Authors, edition, publisher, year confirmed (library catalogue); ISBN and place unconfirmed |
| Ljung 1999 | Publisher is **Prentice Hall PTR**, Upper Saddle River NJ |
| Åström & Hägglund 1995 | Publisher **Instrument Society of America**; place unconfirmed |
| Tobin 2017 | Verified; add pp. 23–30, doi:10.1109/IROS.2017.8202133 |
| Nagabandi 2018 | arXiv verified; ICRA DOI and pages **unconfirmed**, so cite arXiv until checked |
| Lambert 2020 | Corrected: PMLR 120:761–770 (L4DC 2020) |
| Efron 1987 | Verified; add doi:10.1080/01621459.1987.10478410 |
