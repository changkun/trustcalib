# Extension study: calibrating an opaque judge

Pre-registered in `experiment/judge_prereg.md`; no deviations. 10 seeds; the gateway sees only the judge's output, an 8-way command category and time.


## symmetric costs (band (0.35, 0.65))

| Policy | Judge input | Regret | Labels/action | False-allow | Blind-spot ALLOW (pooled) | Blind-spot false-allow (pooled) |
|---|---|---|---|---|---|---|
| judge-only | — | 0.296 ± 0.022 | 0.000 | 0.009 | 188/190 | 28/29 |
| judge-escalate | — | 0.088 ± 0.007 | 0.253 | 0.009 | 188/190 | 28/29 |
| always-ask | — | 0.720 ± 0.008 | 1.000 | nan | 0/190 | 0/29 |
| calibrated | verdict | 0.086 ± 0.016 | 0.118 | 0.022 | 179/190 | 23/29 |
| calibrated | score | 0.068 ± 0.010 | 0.119 | 0.019 | 182/190 | 25/29 |
| calibrated+audit | verdict | 0.075 ± 0.007 | 0.165 | 0.023 | 183/190 | 26/29 |
| calibrated+audit | score | 0.063 ± 0.015 | 0.164 | 0.020 | 188/190 | 28/29 |
| calibrated-product+audit | verdict | 0.146 ± 0.013 | 0.272 | 0.026 | 157/190 | 17/29 |
| calibrated-product+audit | score | 0.127 ± 0.013 | 0.256 | 0.020 | 154/190 | 21/29 |

## safety costs (band (0.25, 0.9))

| Policy | Judge input | Regret | Labels/action | False-allow | Blind-spot ALLOW (pooled) | Blind-spot false-allow (pooled) |
|---|---|---|---|---|---|---|
| judge-only | — | 0.565 ± 0.035 | 0.000 | 0.009 | 188/190 | 28/29 |
| judge-escalate | — | 0.173 ± 0.019 | 0.253 | 0.009 | 188/190 | 28/29 |
| always-ask | — | 0.516 ± 0.009 | 1.000 | nan | 0/190 | 0/29 |
| calibrated | verdict | 0.150 ± 0.017 | 0.467 | 0.001 | 64/190 | 3/29 |
| calibrated | score | 0.150 ± 0.026 | 0.473 | 0.003 | 63/190 | 8/29 |
| calibrated+audit | verdict | 0.168 ± 0.013 | 0.524 | 0.002 | 53/190 | 5/29 |
| calibrated+audit | score | 0.151 ± 0.015 | 0.507 | 0.002 | 61/190 | 7/29 |
| calibrated-product+audit | verdict | 0.462 ± 0.005 | 0.937 | 0.000 | 0/190 | 0/29 |
| calibrated-product+audit | score | 0.465 ± 0.009 | 0.942 | 0.000 | 0/190 | 0/29 |

## Judge reliability for this supervisor (calibrated+audit, verdict input, symmetric)

P(supervisor denies | judge ALLOW) and P(supervisor approves | judge BLOCK), truth vs Horvitz-Thompson estimate from the gateway's labels (known propensities), mean over seeds.

| Window | FA of judge ALLOW: truth | estimate (sd) | FB of judge BLOCK: truth | estimate (sd) |
|---|---|---|---|---|
| [560,750) | 0.029 | 0.015 (0.044) | 0.723 | 0.722 (0.325) |
| [750,1050) | 0.100 | 0.130 (0.121) | 0.486 | 0.425 (0.252) |
| [1050,1500) | 0.034 | 0.042 (0.038) | 0.706 | 0.691 (0.277) |

Without audits the estimate is undefined in every seed (10/10): auto-allowed actions have zero labelling propensity.


## How likely is a judge ALLOW to be right, by command category (calibrated+audit, verdict)

| Category | Gateway's mean p̂ on judge-ALLOWs | True mean approval | Actions (pooled) |
|---|---|---|---|
| db | 0.686 | 0.774 | 96 |
| deploy | 0.684 | 0.621 | 5 |
| exec | 0.801 | 0.869 | 900 |
| network | 0.758 | 0.918 | 146 |
| read | 0.877 | 0.985 | 2487 |
| search | 0.867 | 0.969 | 1882 |
| vcs | 0.796 | 0.911 | 518 |
| write | 0.834 | 0.917 | 987 |
