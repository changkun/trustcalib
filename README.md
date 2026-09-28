# Learning When to Ask

**Trust Calibration for Agentic Tool Use, from Known Tools to Opaque LLM Judges**

Changkun Ou, Latere AI, Munich, Germany ·
[arXiv:2605.19151](https://arxiv.org/abs/2605.19151) ·
PDF: [`manuscript/main.pdf`](manuscript/main.pdf)

This repository is the research artifact for the paper. It contains:
- the LaTeX source;
- the simulation that produces every table, figure and number in the paper;
- a Lean 4 formalization of its propositions;
- a Go implementation of the gateway for real agent harnesses.

## Abstract

Coding agents gate tool calls with a binary switch: run automatically, or
refuse. The missing primitive is *escalation*, and the point at which to
escalate should be learned from the supervisor's own approve/deny decisions.

**Formulation.** We formalize this as Gaussian-process classification with a
reject option, learned online:
- The three-tier ALLOW / ASK / BLOCK gateway is Chow's Bayes-optimal
  reject-option rule, with thresholds `tau_high = 1 - c_ask/c_FA` and
  `tau_low = c_ask/c_FB` set by stated costs.

**Theory.** We prove four results about this gateway:
- Unary approve/deny feedback identifies the allow/deny boundary, while
  pairwise (preferential) feedback cannot.
- A separable time-decaying kernel forgets static action risk and forces every
  unlabelled action back into escalation.
- The irreducible escalation rate is a property of the supervisor.
- Random audits give unbiased false-allow estimates; certifying a rate below
  `alpha` takes `ln(1/delta)/alpha` clean audits.

**Simulation.** An additive kernel that separates static risk from drifting
tolerance halves human labels per action (28.4% to 13.5%) and cuts
cost-weighted regret by 59%, at an unchanged false-allow rate.

**Opaque LLM judges.** We then replace the tool taxonomy with an opaque LLM
judge, as in shell-centric agent harnesses:
- A judge's error rate for a supervisor is not identifiable from its verdicts.
- It changes when the supervisor's tolerance changes.
- It is estimable without bias only from labels with known propensities.

In a pre-registered simulation, the calibrated gateway matches the regret of
escalating the judge's blocks with half the human labels. It cannot, however,
repair a judge blind spot that its features do not expose.

All eight propositions are machine-checked in Lean 4 with Mathlib.

## Research questions

| | Question | Answered in |
|---|---|---|
| RQ1 | Which decision rule and thresholds should a three-tier gateway use? | Sec. 3.4, Sec. 6.1 |
| RQ2 | Which feedback identifies the allow/deny boundary? | Sec. 3.2, Related Work |
| RQ3 | What should a model of a drifting supervisor forget? | Sec. 3.5–3.6, Sec. 6 |
| RQ4 | How can the gateway know its own error rate when auto-decided actions are never labelled? | Sec. 3.7, Sec. 7 |
| RQ5 | When an opaque LLM judge replaces the tool taxonomy, is the judge trustworthy, and how likely is a verdict to be right? | Sec. 4, Sec. 8 |

Main findings (simulation with a known oracle, 10 seeds):

- **Additive kernel.** Separating static risk from drifting tolerance cuts
  human labels from 28.4% to 13.5% of actions and regret by 59%, at a
  false-allow rate of about 1%.
  - Under safety-weighted costs, the separable kernel escalates 99.1% of
    actions and the additive kernel 45.9%, against a floor of 31.3% that no
    learner can go below.
- **Calibrated judge.** A gateway calibrated over an LLM judge's verdicts
  matches the regret of escalating the judge's blocks (0.086 vs 0.088) with
  half the human labels (11.8% vs 25.3%).
- **Relational reliability.** The same judge's false-allow rate for the same
  supervisor moves from 2.9% to 10.0% and back to 3.4% around the
  supervisor's trust reset. Only audited labels with known propensities can
  track this.

## Repository structure

```
manuscript/   paper source (main.tex, references.bib); generated/ holds the
              tables and numbers written by the experiment; Makefile builds
              main.pdf and the arXiv bundle
experiment/   Python simulation: oracle, GP-probit gateway, baselines, the
              judge study and its pre-registration, tests, figures/, reports
lean/         Lean 4 + Mathlib proofs of Propositions 1–8 (see lean/README.md)
go/           Go library and CLI hook implementing the gateway for agent
              harnesses (see go/README.md)
```

Inside `experiment/`:

```
data.py        synthetic action/context space and decision stream
oracle.py      ground-truth supervisor: static risk, trust drift and reset, veto
kernel.py      product (separable), additive and linear kernels
gp.py          Laplace GP-probit classifier, evidence-based selection, BALD
gateway.py     Chow thresholds, three-tier rule, audits, online loop
eval.py        metrics and baselines (per-tool, per-cell)
judge.py       simulated opaque LLM judge and the calibrated-judge policies
run.py         main study: tables, figures, report.md, results.json, macros
run_judge.py   judge study: table, figure, report_judge.md, results_judge.json, macros
judge_prereg.md  pre-registration of the judge study
tests/         correctness tests
```

## Requirements

| Component | Needed for | Version |
|---|---|---|
| Python + [uv](https://docs.astral.sh/uv/) | simulation, tables, figures, tests | Python 3.12 (`.python-version`); numpy, scipy, matplotlib pinned in `uv.lock` |
| [elan](https://github.com/leanprover/elan) / `lake` | checking the proofs | Lean `v4.34.1` (`lean/lean-toolchain`), Mathlib `v4.34.1` |
| [tectonic](https://tectonic-typesetting.github.io/) | building the PDF | tested with 0.17 |
| Go | the gateway implementation | 1.26 or later (`go/go.mod`) |

No GPU is needed. Each simulation runner uses a process pool with all but two
CPU cores and takes about a minute on an 18-core machine; expect the time to
scale roughly inversely with the number of cores.

## Reproducing the paper

Run from the repository root.

```
uv sync                                         # install the Python environment
uv run python -m experiment.run                 # main study
uv run python -m experiment.run_judge           # judge study
uv run pytest                                   # Python tests
(cd lean && lake exe cache get && lake build)   # check the proofs
(cd manuscript && make)                         # build main.pdf
(cd manuscript && make arxiv)                   # build and test-compile arxiv.tar.gz
```

| Paper element | Command | Output | Time |
|---|---|---|---|
| Propositions 1–8 (formal proofs) | `cd lean && lake build` | build log (no `sorry`, standard axioms only) | ~1 min with the Mathlib cache |
| Propositions 2, 4, 5 (numerical checks on the implementation), judge-study sanity checks | `uv run pytest` | test report | seconds |
| Main comparison and safety-weighted costs (Sec. 6.1) | `uv run python -m experiment.run` | `manuscript/generated/table_main.tex`, `table_safety.tex` | ~1 min (one run makes every output of this row group) |
| Acquisition probe (Sec. 6.2) | same run | `manuscript/generated/table_acq.tex` | |
| Generalization to never-labelled actions (Sec. 6.3) | same run | `experiment/figures/transfer.pdf` | |
| Policy mix over time; forgetting, drift tracking and calibration | same run | `experiment/figures/policy_evolution.pdf`, `forgetting.pdf` | |
| Auditing auto-decisions (Sec. 7) | same run | `manuscript/generated/table_audit.tex` | |
| All numbers quoted in the text for Secs. 3 and 6–7 | same run | `manuscript/generated/results.tex`, `experiment/report.md`, `experiment/results.json` | |
| Calibrating a simulated LLM judge (Sec. 8): table, figure, numbers | `uv run python -m experiment.run_judge` | `manuscript/generated/judge.tex`, `experiment/figures/judge_reliability.pdf`, `experiment/report_judge.md`, `experiment/results_judge.json` | ~1 min |
| The PDF | `cd manuscript && make` | `manuscript/main.pdf` | under a minute (the first tectonic run also downloads TeX packages) |
| arXiv source bundle | `cd manuscript && make arxiv` | `manuscript/arxiv.tar.gz` | about a minute (compiles twice) |

Notes on reproducibility:

- **Every number is generated.** Every number in the paper is a LaTeX macro
  written by one of the two runners into `manuscript/generated/`; no result is
  typed by hand. The committed outputs are the ones the paper was built from.
- **Deterministic runs.** Each run is multi-seed (10 seeds; 20 paired seeds
  for the acquisition probe) and deterministic given the seeds, so a rerun
  reproduces the committed files.
- **The judge study was pre-registered.** Its judge model, policies, metrics
  and seeds were fixed in `experiment/judge_prereg.md` and committed (commit
  `c742223`) before the first run. `experiment/report_judge.md` states that
  there were no deviations.
- **What the simulation shows.** The simulation has a known oracle, because
  no public dataset records one supervisor's per-action decisions while their
  tolerance drifts. Recovering the oracle shows that the inference is correct
  under the model, not that real supervisors behave like it. The paper's
  Discussion lists the limitations.

## Tests

```
uv run pytest            # Python: kernels, Laplace inference, Chow rule, forgetting, audits, judge
(cd go && go test ./...) # Go: golden fixtures from the Python reference plus property tests
```

## Machine-checked proofs

The decision-theoretic and algebraic cores of all eight propositions are
formalized in Lean 4 with Mathlib: 31 theorems, no `sorry`, and only Lean's
standard axioms. The link function is kept abstract, so the results hold for
the probit and logistic links alike. The Laplace approximation itself is not
formalized. See [`lean/README.md`](lean/README.md) for the mapping from
propositions to theorems and the exact scope.

## Go implementation

[`go/`](go) implements the gateway as a library and a stateless CLI hook
(`trustcalib-hook decide | observe | tune | stats`) that an agent harness can
call before each tool call. It supports:
- the product and additive kernels;
- cost-derived thresholds;
- random audits with logged propensities, and a certification check;
- a featurizer over an LLM judge's verdict and a command category, for
  shell-centric harnesses.

Its numerics are tested against golden fixtures exported from the Python
reference. See [`go/README.md`](go/README.md).

## Citation

```bibtex
@misc{ou2026learning,
  title         = {Learning When to Ask: Trust Calibration for Agentic Tool
                   Use, from Known Tools to Opaque LLM Judges},
  author        = {Ou, Changkun},
  year          = {2026},
  eprint        = {2605.19151},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2605.19151}
}
```

`CITATION.cff` carries the same metadata.

## License

Apache License 2.0; see [`LICENSE`](LICENSE).

The release history (git tags) is in [`CHANGELOG.md`](CHANGELOG.md).
