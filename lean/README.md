# Machine-checked propositions

This directory is a Lean 4 formalization of the propositions in the paper
*Progressive Autonomy as Preference Learning* (the paper's appendix
"Machine-checked proofs" maps each proposition to its theorems). The build
contains no `sorry`, and every theorem depends only on Lean's standard axioms
(`propext`, `Classical.choice`, `Quot.sound`).

- Toolchain: Lean `v4.34.1` (`lean-toolchain`), Mathlib `v4.34.1`
  (`lakefile.toml`, pinned in `lake-manifest.json`).
- Build time: about a minute with the Mathlib cache; building Mathlib from
  source takes much longer.

## Build

Install [elan](https://github.com/leanprover/elan), then from this directory:

```
lake exe cache get   # download the prebuilt Mathlib cache (once)
lake build           # check every proof
```

A successful build ends with `Build completed successfully`. To list the
axioms a theorem depends on, add `#print axioms TrustCalib.<theorem>` to a
file that imports `TrustCalib` and build it.

## Propositions and theorems

| Paper | Module | Theorems |
|---|---|---|
| Prop. 1, identifiability of the allow/deny boundary | `TrustCalib/Identifiability.lean` | `unary_identifies`, `unary_identifies_boundary`, `pairwise_shift_invariant`, `pairwise_cannot_identify_threshold` |
| Prop. 2, Chow's reject-option rule | `TrustCalib/Chow.lean` | `expectedLoss_allow`, `expectedLoss_block`, `band_nonempty_iff`, `allow_strictly_optimal`, `block_strictly_optimal`, `ask_strictly_optimal`, `decide_optimal`, `ask_never_needed` |
| Prop. 3, escalation floor | `TrustCalib/Floor.lean` | `optimal_asks_on_band`, `escalation_floor` |
| Prop. 4, forgetting under the separable kernel | `TrustCalib/Forgetting.lean` | `kT_shift`, `meanProd_shift`, `quadForm_shift`, `exp_decay_tendsto`, `eventually_ask`, `reentry_bound`, `reentry_prob`, `meanAdd_shift`, `meanAdd_tendsto` |
| Prop. 5, audits | `TrustCalib/Audit.lean` | `audit_unbiased`, `one_sub_pow_le_exp`, `certify_sample_size` |
| Props. 6–8, opaque LLM judge | `TrustCalib/Judge.lean` | `judge_unidentified`, `judge_marginals_not_enough`, `judgeFA_mono`, `ht_unbiased` |
| Remark, epistemic margin | `TrustCalib/Identifiability.lean` | `lcb_iff` |

`TrustCalib.lean` imports every module.

## Scope

What is and is not formalized:

- The link function is abstract: strictly increasing (Prop. 1 and the margin
  remark), or continuous at 0 with value 1/2 (Prop. 4). This covers both the
  probit and the logistic link.
- The Laplace approximation itself is not formalized. Prop. 4 is stated for
  any predictor `μ(x*, t) = Σᵢ k((x*, t), (xᵢ, tᵢ)) αᵢ` with coefficients fixed
  by the training data. This is the form of the Laplace GP-probit mean
  (`α = ∇ log p(y | f̂)`) and of exact GP regression.
- Prop. 2(a) (the posterior mean is sufficient) is proved for an arbitrary
  probability measure over the latent value, using the Bochner integral.
- Prop. 3 is proved for finite action populations with nonnegative weights.
- Prop. 5(a) needs only `E[Aᵢ] = ε`, not independence. Prop. 5(b) is the
  deterministic inequality `(1 - α)^n ≤ exp(-α n) ≤ δ` behind the
  certification sample size.
- Prop. 6 is proved by explicit construction on the four cells of the
  (judge verdict, supervisor decision) table, and Prop. 7 for finite
  populations. Prop. 8 generalizes Prop. 5(a) to per-action propensities
  `E[Lᵢ] = πᵢ > 0`.

The Python tests in `../experiment/tests/` check the same statements
numerically on the implementation (`test_chow.py`, `test_forgetting.py`,
`test_audit.py`).
