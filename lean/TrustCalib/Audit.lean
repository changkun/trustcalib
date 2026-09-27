/-
Proposition 5 (manuscript §8, selective labels): random audits.

Only escalated actions receive a human label, so the false-allow rate of
auto-decided actions cannot be estimated from escalations. Auditing each
auto-allowed action with probability `ε > 0` gives (i) an unbiased
inverse-propensity estimate of the false-allow rate, using only
`E[Aᵢ] = ε` (no independence needed), and (ii) a certification sample size:
if the true false-allow rate is at least `α`, the probability that `n`
independent audits all come back clean is at most `(1 - α)^n ≤ exp(-α n)`,
which is `≤ δ` once `n ≥ log(1/δ)/α`.
-/
import Mathlib

namespace TrustCalib

open MeasureTheory

variable {Ω : Type*} [MeasurableSpace Ω] {N : ℕ}

/-- Unbiasedness of the inverse-propensity audit estimator. `A i` is the audit
indicator of action `i` (any integrable random variable with mean `ε`) and
`e i ∈ ℝ` its (fixed) false-allow indicator. -/
theorem audit_unbiased (P : Measure Ω) [IsProbabilityMeasure P] {ε : ℝ} (hε : 0 < ε)
    (hN : 0 < N) (A : Fin N → Ω → ℝ) (hA : ∀ i, Integrable (A i) P)
    (hmean : ∀ i, ∫ ω, A i ω ∂P = ε) (e : Fin N → ℝ) :
    ∫ ω, (∑ i, A i ω * e i) / (N * ε) ∂P = (∑ i, e i) / N := by
  have hN' : (N : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hN.ne'
  rw [integral_div, integral_finsetSum _ (fun i _ => (hA i).mul_const _)]
  simp_rw [integral_mul_const, hmean]
  rw [← Finset.mul_sum]
  field_simp

/-- `(1 - α)^n ≤ exp(-α n)` for `α ≤ 1`. -/
theorem one_sub_pow_le_exp {α : ℝ} (h1 : α ≤ 1) (n : ℕ) :
    (1 - α) ^ n ≤ Real.exp (-α * n) := by
  have h : 1 - α ≤ Real.exp (-α) := by
    have := Real.add_one_le_exp (-α)
    linarith
  calc (1 - α) ^ n ≤ Real.exp (-α) ^ n := by
        gcongr
    _ = Real.exp (-α * n) := by
        rw [← Real.exp_nat_mul]
        congr 1
        ring

/-- Certification sample size: `n ≥ log(1/δ)/α` clean audits make the
"all clean despite false-allow rate ≥ α" event have probability `≤ δ`. -/
theorem certify_sample_size {α δ : ℝ} (h0 : 0 < α) (h1 : α ≤ 1) (hδ0 : 0 < δ)
    (n : ℕ) (hn : Real.log (1 / δ) / α ≤ n) :
    (1 - α) ^ n ≤ δ := by
  refine (one_sub_pow_le_exp h1 n).trans ?_
  rw [div_le_iff₀ h0, one_div, Real.log_inv] at hn
  rw [← Real.le_log_iff_exp_le hδ0]
  linarith

end TrustCalib
