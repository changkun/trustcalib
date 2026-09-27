# Progressive Autonomy as Preference Learning

This repository contains a manuscript that formalizes trust calibration for
agentic tool use, a simulation study that tests it, a Lean 4 formalization of
its theory, and a Go port of the gateway as an agent-harness plugin.

**Paper: [`manuscript/main.pdf`](manuscript/main.pdf)** (version 2; version 1
is [arXiv:2605.19151](https://arxiv.org/abs/2605.19151)).

- `manuscript/` LaTeX source (`main.tex`, `references.bib`); every number in
  the paper is a macro in `manuscript/generated/`, written by the experiment.
- `experiment/` runnable implementation, tests, figures and report.
- `lean/` machine-checked proofs of the paper's propositions.
- `go/` a Go port of the v1 gateway core for real agent harnesses.

## Motivation

Coding agents gate actions with a binary switch: auto-run, or deny. A hard
deny does not pause the run for a human; the agent receives a refusal and
keeps going, often routing around the obstacle in ways that are worse than
asking would have been. The missing primitive is not *block* but *escalate*,
and when to escalate should be learned from the supervisor's own approve/deny
history rather than hand-written.

## What the paper shows (version 2)

The gateway keeps a Gaussian-process posterior over a latent tolerance
`f(x, t) = tau(t) - r(x) + h(x, t)` (static action risk `r`, drifting
supervisor tolerance `tau`), observed through a probit likelihood on
approve/deny feedback, and decides ALLOW / ASK / BLOCK. Five propositions,
each machine-checked in `lean/`:

1. **Identifiability.** Unary approve/deny feedback identifies the allow/deny
   boundary; pairwise (preferential) feedback cannot, because it is invariant
   to shifting `f` by a constant. The gateway is GP classification with a
   reject option, a relative of preferential Bayesian optimization rather than
   an instance of it.
2. **Decision rule.** The three-tier rule is Chow's reject option: with costs
   for a false allow, a false block and an escalation, the thresholds are
   `tau_high = 1 - c_ask/c_FA` and `tau_low = c_ask/c_FB`, and the rule depends
   on the posterior only through the mean approval probability. Thresholds are
   specified, not tuned.
3. **Escalation floor.** Even a perfect posterior escalates every action whose
   true approval probability lies in the band, so the achievable automation
   rate is a property of the supervisor and the costs.
4. **Forgetting.** A separable kernel `k_x * k_time` decays every unlabelled
   action back to `p_hat = 1/2`, forcing re-escalation after at most
   `lambda * ln(m/z)` steps. An additive kernel forgets only what drifts.
5. **Audits.** Auto-decided actions are never labelled. Random audits give an
   unbiased false-allow estimate; certifying a false-allow rate below `alpha`
   at confidence `1 - delta` takes `ln(1/delta)/alpha` clean audits (300 for
   1% at 95%).

## What the simulation shows

No public dataset records a single supervisor's per-action decisions as their
tolerance drifts, so the experiment is a **controlled simulation with a known
oracle** (`experiment/oracle.py`). "The method recovers the oracle" means "the
inference is correct under the model", not "the real world behaves like this".
Headline numbers (10 seeds; `experiment/report.md` has all tables):

- The additive kernel cuts human labels per action from 28.4% to 13.5% and
  cost-weighted regret by 59% at a false-allow rate of about 1%, at the price
  of 1.4 points of auto-decision accuracy.
- Under safety-weighted costs (ALLOW only above 0.90) the v1 product kernel
  escalates 99% of actions; the additive kernel escalates 46% against a 31%
  floor.
- Version 1's negative result on uncertainty-targeted querying is explained:
  with the product kernel the ASK band is no better than random querying; with
  the additive kernel it beats random by about 3 points. The cause is
  forgetting, not class imbalance, and a BALD sampler does not help.
- 5% random audits halve ALLOWs inside the post-reset veto window and give an
  unbiased false-allow estimate.

Reported as limitations:

- Correlated generalization leaks: on a never-labelled dangerous combination
  (`git_force_push -> prod_infra`) every kernel model auto-allows about 11% of
  the oracle's denials.
- A Bayesian linear probit with the same time structure is competitive on
  average metrics, because the oracle's static term is linear in the features
  by construction; it is much worse on the veto interaction.
- Evidence (marginal-likelihood) selection prefers long memories that improve
  calibration but double veto-window allows.

Version 2 also corrects several claims of version 1 (listed in the paper's
appendix), including the burden ratio: version 1's "~1.8x" compared
full-stream queries against scored-phase actions.

## Version 3 draft: an opaque LLM judge instead of a tool taxonomy

Shell-centric harnesses have no fixed tool set; an LLM judge (such as Claude
Code's auto-mode classifier) decides allow/block from a restricted view. The
v3 draft (git tag `arxiv-v2` is the frozen v2) adds Section 12 and
`lean/TrustCalib/Judge.lean`:

- **Is the judge trustworthy?** A judge's false-allow rate for a supervisor
  is not identifiable from its verdicts, even together with the supervisor's
  approval rate (Prop. 6); it rises when the supervisor becomes stricter
  (Prop. 7); it is estimable without bias only from human labels with known
  propensities (Prop. 8), and certifying it below `alpha` takes
  `ln(1/delta)/alpha` clean audits while it drifts.
- **How likely is a verdict right?** The gateway's calibrated posterior over
  (judge output, command category, time) answers it per action.
- Pre-registered simulation (`experiment/judge_prereg.md`,
  `uv run python -m experiment.run_judge`): escalating the judge's blocks
  instead of enforcing them cuts regret 0.296 -> 0.088; the calibrated gateway
  matches that with half the labels (11.8% vs 25.3%). The same judge's
  false-allow rate for the supervisor goes 2.9% -> 10.0% -> 3.4% around the
  supervisor's trust reset. Calibration cannot catch a judge blind spot that
  the gateway's own features do not expose.

## Layout

```
experiment/
  data.py      synthetic action/context space and stream
  oracle.py    ground-truth latent f*, probit, trust drift and reset, veto
  kernel.py    product (v1), additive (v2) and linear kernels
  gp.py        Laplace GP-probit (R&W Alg. 3.1/3.2), evidence selection, BALD
  gateway.py   Chow thresholds, three-tier rule, audits, online loop
  eval.py      metrics and baselines (per-tool, per-cell, linear)
  run.py       orchestration; writes figures/, report.md, results.json and
               manuscript/generated/*.tex
  tests/       correctness tests (kernels, Laplace, Chow, forgetting, audits)
lean/          Lean 4 + Mathlib proofs (see lean/README.md)
manuscript/    paper source; `make` builds main.pdf, `make arxiv` the bundle
go/            Go port of the v1 gateway (library + CLI hook)
```

## Run

Dependencies are managed with `uv`; the paper builds with `tectonic`; the
proofs build with `lake` (elan).

```
uv sync                                  # install
uv run python -m experiment.run          # all tables, figures, paper macros (~1 min)
uv run python -m experiment.run_judge    # opaque-judge study (~1 min)
uv run pytest                            # correctness tests
(cd lean && lake exe cache get && lake build)   # check the proofs
(cd manuscript && make && make arxiv)    # paper and arXiv bundle
```

Each run is multi-seed and deterministic given the seeds.

The Go port in `go/` implements the version-1 gateway (product kernel, grid
threshold tuning). Its `Tune()` sees only escalated labels, whose predicted
probabilities all lie inside the band, so it returns the default band; the
version-2 cost-derived thresholds and additive kernel are not yet ported.
