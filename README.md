# Progressive Autonomy as Preference Learning

This repository contains a short manuscript formalizing trust calibration for
agentic tool use as a preference-learning problem, and a self-contained
experiment that implements and stress-tests that formulation.

- `manuscript/` LaTeX source of the paper (`main.tex`, `references.bib`); the
  Simulation Study section and Conclusion report the results below.
- `experiment/` runnable implementation, tests, generated figures and report.

## Motivation

Today's coding agents gate actions with a binary switch: auto-run, or deny. A
hard deny does not pause the run for a human; the agent receives a refusal and
keeps going, often routing around the obstacle in ways that are worse than
asking would have been. The missing primitive is not *block* but *escalate*,
and the threshold for when to escalate should be learned from the supervisor's
own approve/deny history rather than hand-written. The manuscript formalizes
this as a preference-learning problem: structurally an instance of
Preferential Bayesian Optimization, specialized to unary approve/deny
feedback, with a three-tier allow/ask/block gateway.

## What the experiment is (and is not)

The manuscript models a policy gateway that maintains a Gaussian-process
posterior over a latent human risk-tolerance function, observed through a
probit approve/deny likelihood, and escalates to the human exactly where the
approval outcome is most uncertain.

No public dataset carries the signal this formulation needs: a single
supervisor's per-action approve/deny decisions tracked longitudinally as their
risk tolerance drifts. R-Judge has human safe/unsafe labels but is static and
aggregated; AgentSec (Zenodo 18369965) records agent provenance but has no
human-feedback, risk, or preference labels. The experiment is therefore a
**controlled simulation with a known ground-truth oracle that is the
manuscript's generative model** (Definition 1, Sections 4 to 6). This is the
standard evaluation protocol for Preferential Bayesian Optimization: "the
method recovers the oracle" means "the inference is correct", not "the real
world behaves like this". It is not a validation on real human data.

## Layout

```
experiment/
  data.py      synthetic action/context space + stream (manuscript Sec. 2)
  oracle.py    ground-truth latent f*, probit, Sec. 6 trust drift + changepoint
  kernel.py    product kernel k_tool * k_ctx * k_time (Sec. 4)
  gp.py        self-contained Laplace GP-probit (R&W Alg. 3.1/3.2)
  gp_botorch.py optional BoTorch PairwiseGP cross-check (Remark 1)
  gateway.py   three-tier allow/ask/block rule + prequential acquisition loop
  eval.py      metrics + baselines (random-query, no-correlation)
  run.py       orchestration; writes figures/ and report.md
  tests/       correctness tests (kernel PSD, Laplace mode, recovery, burden)
  figures/     generated PDFs (300 dpi, vector)
  report.md    generated results report
```

## Run

Dependencies are managed with `uv`.

```
uv sync                                # install
uv run python -m experiment.run        # regenerate report.md + figures/
uv run pytest                          # correctness tests
uv sync --extra botorch && uv run python -m experiment.run   # + cross-check
```

Each run is multi-seed and deterministic given the seeds. `report.md` is the
authoritative summary; the headline numbers are reproduced there.

## Key findings (see `experiment/report.md` for the full tables)

Supported by the simulation:

- The ASK band narrows as the posterior concentrates and the auto-approve
  rate rises substantially (toward, but not reaching, the manuscript's
  85-90% target band).
- A large reduction in human interruptions versus the always-escalate status
  quo, at high auto-decision accuracy and a bounded false-allow rate.
- Correlated generalization: the GP transfers evidence to an unqueried
  action-context combination far better than a per-tool baseline.
- The posterior tracks both gradual drift and the abrupt Section 6
  changepoint.

Reported honestly as a negative result:

- The Section 5 acquisition rule taken literally ("query inside the ASK
  band") is **not** more sample-efficient than random querying. An ablation
  shows the deficit is present even with a stationary target, so it is not
  caused by the Section 6 changepoint: it is the generic behaviour of
  uncertainty sampling under class imbalance (confident regions leave the ASK
  band and are never re-probed; `k_time` down-weights stale evidence but does
  not itself generate new probes). The three-tier escalate design is sound;
  only the rule for *what* to escalate is not, and a recency-aware or
  information-theoretic acquisition rule is the noted future-work remedy.

The posterior is also somewhat underconfident at the kernel-far tail
(calibration is partial, not perfect); this is shown rather than tuned away.
