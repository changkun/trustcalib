# Changelog

Release history of the paper and its artifact. Each release is a git tag.

## v3, 2026-09-28

The extension to opaque LLM judges, and a restructured paper under a new
title.

- **New title.** *Learning When to Ask: Trust Calibration for Agentic Tool
  Use, from Known Tools to Opaque LLM Judges*, previously *Progressive
  Autonomy as Preference Learning: A Formalization of Trust Calibration for
  Agentic Tool Use*.
- **Paper restructured.** The paper now follows a conventional layout:
  introduction with explicit research questions, related work, method (trust
  calibration as classification with a reject option, and calibration of an
  opaque LLM judge), experimental setup, three experiments, discussion and
  concluding remarks. It adds an overview figure of the escalation loop.
- **No version notes in the paper.** The paper no longer carries version
  history or repository paths; the corrections to v1 are recorded under v2
  below.
- **Figures redrawn.** The figures now share one style, and the drift-tracking
  and calibration plots are merged into the forgetting figure.
- **Artifact documentation.** The README is now an artifact guide: research
  questions, requirements, a claims-to-commands table, tests and citation.
  `CITATION.cff` is added.
- **Wording only.** Code comments and the Go and Lean READMEs describe the
  kernels by structure (product, additive) rather than by version. No numbers
  changed.
- **Opaque judge.** New section on an opaque LLM judge in front of the
  gateway, for shell-centric agent harnesses where no fixed tool taxonomy
  exists. It adds:
  - a definition of judge reliability for a supervisor;
  - Proposition 6: a judge's false-allow rate is not identifiable from its
    verdicts, even together with both marginals;
  - Proposition 7: reliability is relational, rising when the supervisor
    becomes stricter;
  - Proposition 8: Horvitz–Thompson estimation with known propensities.
- **Pre-registered simulation.** The judge simulation was pre-registered in
  `experiment/judge_prereg.md` (commit c742223) before any run, and was run
  with no deviations (`experiment/run_judge.py`).
- **Lean.** A formalization of Propositions 6–8 (`lean/TrustCalib/Judge.lean`),
  bringing the development to 31 theorems.
- **Codex comparison.** A comparison of Claude Code's auto-mode classifier
  with the open-source Codex auto-review reviewer: what each judge sees, what
  it returns, and when it hands control back to the human.
- **Go port.** The Go gateway gains:
  - the additive kernel;
  - cost-derived thresholds;
  - random audits with propensities logged at decision time, and a
    certification check;
  - a judge featurizer for shell-centric harnesses.

## v2 (e68182e), 2026-09-27

A revision of the theory and the simulation, with corrections to v1.

- **Relation to PBO.** v1 called the gateway an instance of preferential
  Bayesian optimization and the unary model a degenerate pairwise comparison.
  Pairwise feedback cannot identify the threshold (Proposition 1); the gateway
  is GP classification with a reject option.
- **Decision rule.** v1 gave no loss for the three-tier rule, and tuned its
  test-phase thresholds with oracle labels on every validation action. The
  rule is Chow's reject option (Proposition 2), and the thresholds now follow
  from stated costs.
- **Automation rate.** v1 asserted that an 85–90% auto-approval rate emerges
  without tuning. The achievable rate is bounded by the supervisor's ambiguity
  (Proposition 3).
- **Forgetting.** v1 stated that the time kernel does not generate new probes.
  It forces periodic re-escalation (Proposition 4), which is the main cause of
  v1's high escalation rate and underconfidence. v1 had attributed the
  underconfidence to the Laplace approximation.
- **Acquisition.** v1 attributed its acquisition result to class imbalance and
  proposed an information-theoretic rule as the remedy. The cause is
  forgetting: with an additive kernel the ASK band beats random querying, and
  a BALD sampler does not help.
- **Burden ratio.** v1 reported "508 vs 940 queries, a ~1.8x reduction", but
  its numerator counted the whole stream and its denominator only the scored
  phases. On matching phases the v1 protocol gives about 3.0x. The v2
  protocol gives 3.5x (product kernel) and 7.4x (additive kernel).
- **Transfer test and baselines.** v1's transfer test held out only a benign
  combination, and its per-tool baseline ignored target and task. v2 adds a
  dangerous hold-out, a per-cell baseline and a linear-probit baseline.
- **Smaller corrections.** v1 did not state its hyperparameters, described a
  sliding window as equivalent to the time kernel, and illustrated
  argument-level transfer (`DROP` to `TRUNCATE`) that its single
  destructive-argument feature cannot express. These are corrected.
- **Other additions.**
  - An additive kernel that separates static risk from drifting tolerance.
  - Random audits with a certification sample size (Proposition 5).
  - Evidence-selected and linear baselines.
  - A Lean 4 / Mathlib formalization of Propositions 1–5.
  - Every number in the paper generated as a LaTeX macro by the experiment.

## v1 (1706fc8), 2026-05-27, arXiv:2605.19151v1

The initial preprint: trust calibration for agentic tool use formulated as
preference learning with a Gaussian-process gateway. It came with a
simulation study and a Go port of the core algorithm as an agent-harness
plugin.
