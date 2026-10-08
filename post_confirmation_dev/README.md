# Post-confirmation development diagnostics (8 Oct 2026)

**Exploratory. Development plants only** (tuning 5000–5099, evaluation 5100–5149). No spent confirmation seeds were used.

`vsim.py` is a vectorised copy of `experiment.Circuit`. It has identical per-plant parameters, but the noise comes from one shared generator, so results are not bit-identical to the scalar simulator. Its fixed-PI (0.02, 0.15) values reproduce the recorded development values to within 0.015 Hz.

| Script | Question |
|---|---|
| `pi_check2.py` | Does a single fixed PI with retuned gains (selected on tuning plants, equal weight across s) change the PI-vs-DR-H4 ranking? |
| `noise_check.py` | Same, under process noise σ = 1, 2 at s = 1. |
| `noise_aw.py` | Zero-light rate floor at σ = 2; PI with conditional-integration anti-windup. |

Recorded output (evaluation plants 5100–5149):

```
s=0.6: PI(.02,.15) 2.289  PI(.02,.25) 1.619  DR-H4 1.605  diff +0.013  PI better on 38% of plants
s=0.8: PI(.02,.15) 1.584  PI(.02,.25) 1.130  DR-H4 1.215  diff -0.086  98%
s=1.0: PI(.02,.15) 1.242  PI(.02,.25) 1.087  DR-H4 1.308  diff -0.221  100%
s=1.2: PI(.02,.15) 1.160  PI(.02,.25) 1.275  DR-H4 1.818  diff -0.543  100%
noise=1.0 (s=1): PI(.02,.25) 1.278  DR-H4 1.281
noise=2.0 (s=1): PI(.02,.25) 3.202  PI(.02,.25)+anti-windup 2.437  DR-H4 2.150   (rate floor at u=0: 14.4 Hz)
```
