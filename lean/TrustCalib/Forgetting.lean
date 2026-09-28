/-
Proposition 4 (manuscript, drift and forgetting): forgetting under the
separable time kernel.

For the product kernel `k((x,t),(x',t')) = kX x x' * exp(-|t - t'|/λ)`, every
kernel predictor `μ(x*, t*) = Σᵢ kX x* xᵢ * exp(-|t* - tᵢ|/λ) * αᵢ` (this covers
the Laplace GP-probit posterior mean, with `α = ∇ log p(y | f̂)`, and exact GP
regression) decays geometrically once `t*` is past every training time:
`μ(x*, t* + Δ) = exp(-Δ/λ) μ(x*, t*)`. The variance reduction term
`k*ᵀ A k*` decays as `exp(-2Δ/λ)`. So the predictive approval probability of any
action that stops receiving labels returns to `Φ 0 = 1/2` and, for any band
containing `1/2`, the action is eventually escalated again.

Under the additive kernel `σs kX + σg kT + σd kX kT`, only the time-coupled parts
decay: the mean converges to the static component instead of to zero.
-/
import Mathlib

namespace TrustCalib

open Filter Topology

/-- The exponential (Ornstein-Uhlenbeck) time kernel. -/
noncomputable def kT (lam t s : ℝ) : ℝ := Real.exp (-|t - s| / lam)

/-- OU factorization: for `s ≤ t` and `Δ ≥ 0`,
`kT (t + Δ) s = exp(-Δ/λ) * kT t s`. -/
theorem kT_shift {lam t s Δ : ℝ} (hst : s ≤ t) (hΔ : 0 ≤ Δ) :
    kT lam (t + Δ) s = Real.exp (-Δ / lam) * kT lam t s := by
  unfold kT
  rw [← Real.exp_add]
  congr 1
  rw [abs_of_nonneg (by linarith), abs_of_nonneg (by linarith)]
  ring

variable {n : ℕ}

/-- A kernel predictor under the separable product kernel. -/
noncomputable def meanProd (lam : ℝ) (kx : Fin n → ℝ) (ts : Fin n → ℝ) (α : Fin n → ℝ)
    (t : ℝ) : ℝ :=
  ∑ i, kx i * kT lam t (ts i) * α i

/-- Geometric decay of the posterior mean past the last observation. -/
theorem meanProd_shift {lam t Δ : ℝ} (kx ts α : Fin n → ℝ) (hts : ∀ i, ts i ≤ t)
    (hΔ : 0 ≤ Δ) :
    meanProd lam kx ts α (t + Δ) = Real.exp (-Δ / lam) * meanProd lam kx ts α t := by
  unfold meanProd
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [kT_shift (hts i) hΔ]
  ring

/-- The variance-reduction quadratic form decays by `exp(-2Δ/λ)`. -/
theorem quadForm_shift {lam t Δ : ℝ} (kx ts : Fin n → ℝ) (A : Fin n → Fin n → ℝ)
    (hts : ∀ i, ts i ≤ t) (hΔ : 0 ≤ Δ) :
    (∑ i, ∑ j, (kx i * kT lam (t + Δ) (ts i)) * A i j * (kx j * kT lam (t + Δ) (ts j))) =
      Real.exp (-2 * Δ / lam) *
        ∑ i, ∑ j, (kx i * kT lam t (ts i)) * A i j * (kx j * kT lam t (ts j)) := by
  have he : Real.exp (-Δ / lam) * Real.exp (-Δ / lam) = Real.exp (-2 * Δ / lam) := by
    rw [← Real.exp_add]
    congr 1
    ring
  rw [← he, Finset.mul_sum]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [kT_shift (hts i) hΔ, kT_shift (hts j) hΔ]
  ring

/-- `exp(-Δ/λ) → 0` as `Δ → ∞` for `λ > 0`. -/
theorem exp_decay_tendsto {lam : ℝ} (hlam : 0 < lam) :
    Tendsto (fun Δ : ℝ => Real.exp (-Δ / lam)) atTop (𝓝 0) := by
  simp only [neg_div]
  exact Real.tendsto_exp_neg_atTop_nhds_zero.comp (tendsto_id.atTop_div_const hlam)

/-- Re-escalation: if the mean decays as `exp(-Δ/λ) m`, the predictive
probability is `Φ(μΔ / √(1 + vΔ))` with any nonnegative variances `vΔ`, `Φ` is
continuous at `0` with `Φ 0 = 1/2`, and the band contains `1/2`, then the action
is eventually strictly inside the ASK band. -/
theorem eventually_ask {lam m τlo τhi : ℝ} (hlam : 0 < lam) {Φ : ℝ → ℝ}
    (hΦc : ContinuousAt Φ 0) (hΦ0 : Φ 0 = 1 / 2) (hlo : τlo < 1 / 2) (hhi : 1 / 2 < τhi)
    (v : ℝ → ℝ) (hv : ∀ Δ, 0 ≤ v Δ) :
    ∀ᶠ Δ in atTop,
      τlo < Φ (Real.exp (-Δ / lam) * m / Real.sqrt (1 + v Δ)) ∧
        Φ (Real.exp (-Δ / lam) * m / Real.sqrt (1 + v Δ)) < τhi := by
  have hbound : ∀ Δ, ‖Real.exp (-Δ / lam) * m / Real.sqrt (1 + v Δ)‖ ≤
      Real.exp (-Δ / lam) * |m| := by
    intro Δ
    have hs : 1 ≤ Real.sqrt (1 + v Δ) := by
      rw [Real.one_le_sqrt]
      linarith [hv Δ]
    rw [Real.norm_eq_abs, abs_div, abs_mul, Real.abs_exp,
      abs_of_pos (lt_of_lt_of_le one_pos hs)]
    exact div_le_self (by positivity) hs
  have harg : Tendsto (fun Δ => Real.exp (-Δ / lam) * m / Real.sqrt (1 + v Δ)) atTop (𝓝 0) := by
    refine squeeze_zero_norm hbound ?_
    simpa using (exp_decay_tendsto hlam).mul_const |m|
  have hΦt := hΦc.tendsto.comp harg
  rw [hΦ0] at hΦt
  filter_upwards [hΦt.eventually (Ioo_mem_nhds hlo hhi)] with Δ h
  exact h

/-- Explicit re-escalation time. If the posterior mean past the last label is
`exp(-Δ/λ) m` with `m > 0`, then after `Δ ≥ λ log(m / z)` label-free steps the
mean is at most `z`. -/
theorem reentry_bound {lam m z Δ : ℝ} (hlam : 0 < lam) (hm : 0 < m) (hz : 0 < z)
    (hΔ : lam * Real.log (m / z) ≤ Δ) : Real.exp (-Δ / lam) * m ≤ z := by
  have h1 : Real.log (m / z) ≤ Δ / lam := by
    rw [le_div_iff₀ hlam]; linarith
  have h2 : Real.exp (-Δ / lam) ≤ z / m := by
    have : -Δ / lam ≤ Real.log (z / m) := by
      rw [Real.log_div hz.ne' hm.ne', neg_div]
      rw [Real.log_div hm.ne' hz.ne'] at h1
      linarith
    calc Real.exp (-Δ / lam) ≤ Real.exp (Real.log (z / m)) := Real.exp_le_exp.mpr this
      _ = z / m := Real.exp_log (div_pos hz hm)
  calc Real.exp (-Δ / lam) * m ≤ z / m * m := by gcongr
    _ = z := by field_simp

/-- Consequently, with a monotone link and any nonnegative predictive variance,
the predictive probability is at most `Φ z` after `λ log(m / z)` label-free
steps: taking `z = Φ⁻¹(τhi)`, the action is no longer auto-allowed. -/
theorem reentry_prob {lam m z Δ : ℝ} (hlam : 0 < lam) (hm : 0 < m) (hz : 0 < z)
    (hΔ : lam * Real.log (m / z) ≤ Δ) {Φ : ℝ → ℝ} (hΦ : Monotone Φ) {v : ℝ} (hv : 0 ≤ v) :
    Φ (Real.exp (-Δ / lam) * m / Real.sqrt (1 + v)) ≤ Φ z := by
  apply hΦ
  have hpos : 0 ≤ Real.exp (-Δ / lam) * m := by positivity
  have hs : 1 ≤ Real.sqrt (1 + v) := by
    have := Real.sqrt_le_sqrt (show (1 : ℝ) ≤ 1 + v by linarith)
    simpa using this
  calc Real.exp (-Δ / lam) * m / Real.sqrt (1 + v) ≤ Real.exp (-Δ / lam) * m :=
        div_le_self hpos hs
    _ ≤ z := reentry_bound hlam hm hz hΔ

/-- A kernel predictor under the additive kernel
`σs kX + σg kT + σd kX kT`. -/
noncomputable def meanAdd (lam σs σg σd : ℝ) (kx : Fin n → ℝ) (ts : Fin n → ℝ)
    (α : Fin n → ℝ) (t : ℝ) : ℝ :=
  ∑ i, (σs * kx i + σg * kT lam t (ts i) + σd * kx i * kT lam t (ts i)) * α i

/-- Under the additive kernel only the time-coupled part decays. -/
theorem meanAdd_shift {lam σs σg σd t Δ : ℝ} (kx ts α : Fin n → ℝ) (hts : ∀ i, ts i ≤ t)
    (hΔ : 0 ≤ Δ) :
    meanAdd lam σs σg σd kx ts α (t + Δ) =
      (∑ i, σs * kx i * α i) +
        Real.exp (-Δ / lam) * ∑ i, (σg * kT lam t (ts i) + σd * kx i * kT lam t (ts i)) * α i := by
  unfold meanAdd
  rw [Finset.mul_sum, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [kT_shift (hts i) hΔ]
  ring

/-- The additive-kernel mean converges to its static component (not to zero). -/
theorem meanAdd_tendsto {lam σs σg σd t : ℝ} (hlam : 0 < lam) (kx ts α : Fin n → ℝ)
    (hts : ∀ i, ts i ≤ t) :
    Tendsto (fun Δ : ℝ => meanAdd lam σs σg σd kx ts α (t + Δ)) atTop
      (𝓝 (∑ i, σs * kx i * α i)) := by
  have h : Tendsto (fun Δ : ℝ => (∑ i, σs * kx i * α i) +
      Real.exp (-Δ / lam) * ∑ i, (σg * kT lam t (ts i) + σd * kx i * kT lam t (ts i)) * α i)
      atTop (𝓝 (∑ i, σs * kx i * α i)) := by
    simpa using tendsto_const_nhds.add ((exp_decay_tendsto hlam).mul_const
      (∑ i, (σg * kT lam t (ts i) + σd * kx i * kT lam t (ts i)) * α i))
  refine h.congr' ?_
  filter_upwards [eventually_ge_atTop 0] with Δ hΔ
  exact (meanAdd_shift kx ts α hts hΔ).symm

end TrustCalib
