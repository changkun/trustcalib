# Pre-registration: calibrating an opaque judge (extension study)

Committed before any run of `experiment/run_judge.py`. Any later change to
the settings below is reported as a deviation in the report.

## Question

Agent harnesses increasingly route actions through a general-purpose shell, so
no fixed tool taxonomy exists, and an LLM judge (for example an auto-mode
classifier) decides allow/block from a restricted view. We ask (1) whether the
judge is trustworthy for a given supervisor, and (2) how likely a given verdict
is to agree with that supervisor.

## Judge model (the oracle's view of the judge)

- The judge sees the action's static risk `r(x) = raw_risk(x)` (tool,
  target, destructive-argument flag) but not the supervisor's trust state,
  task offsets or veto.
- Score `s = sigmoid(8 * (r(x) - 0.5) + e)`, `e ~ N(0, 1)`, independent per
  action. Verdict: BLOCK if `s > 0.5`, else ALLOW.
- Blind spot: actions of category `exec` with the destructive-argument flag
  set are perceived as `r = 0.2` (for example obfuscated destructive shell
  commands).

## What the gateway sees

No tool taxonomy: only the judge's output, a coarse command category (8-way
one-hot, as a shell parser would give) and time. Two conditions:

- `verdict`: the binary verdict (0/1) plus category.
- `score`: the continuous score `s` plus category.

## Policies

1. `judge-only`: ALLOW if the judge allows, else BLOCK (no human).
2. `judge-escalate`: ALLOW if the judge allows, else ASK.
3. `always-ask`.
4. `calibrated`: Laplace GP-probit with the additive kernel
   `(s_static, s_global, s_inter) = (1.6, 1.0, 0.6)`, `lambda = 90`,
   `l_tool = 0.7` on the judge block, Chow thresholds, no audits.
5. `calibrated+audit`: as 4, with audits of `eps = 0.05` of auto-decisions.
6. `calibrated-product+audit`: product kernel `sigma2 = 1.6`, `lambda = 90`,
   `l_tool = 0.7`, audits `0.05`.

## Protocol and metrics

Same stream, oracle (reset at t=750), phases and seeds (10) as the main study;
symmetric and safety-weighted costs. Metrics: regret, labels per action,
false-allow rate, blind-spot false-allows (pooled counts), and the judge's
reliability for this supervisor, `P(y = 0 | judge ALLOW)` and
`P(y = 1 | judge BLOCK)`, estimated by Horvitz-Thompson with known
propensities (1 for ASK, `eps` for audited auto-decisions) in the windows
`[560, 750)`, `[750, 1050)` and `[1050, 1500)`, against the truth.
