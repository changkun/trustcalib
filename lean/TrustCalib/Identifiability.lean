/-
Proposition 1 (manuscript §3; corrects Remark 1 of version 1): unary approve/deny feedback
identifies the latent tolerance, and hence the allow/deny boundary `{f > 0}`;
pairwise comparisons cannot, because they are invariant to adding a constant.

The link `Φ` is kept abstract; only strict monotonicity is used (the standard
normal CDF and the logistic function both qualify).
-/
import Mathlib

namespace TrustCalib

variable {X : Type*}

/-- Pairwise (preferential) likelihood `P(x ≻ x') = Φ(f x - f x')` is invariant
under shifting the latent function by any constant. -/
theorem pairwise_shift_invariant (Φ : ℝ → ℝ) (f : X → ℝ) (c : ℝ) (x x' : X) :
    Φ ((f x + c) - (f x' + c)) = Φ (f x - f x') := by
  congr 1
  ring

/-- Consequently pairwise data cannot locate the approval threshold: for any
latent `f` and any action `x` there is a `g` with *identical* pairwise
likelihoods whose sign at `x` (allow vs deny side) is flipped. -/
theorem pairwise_cannot_identify_threshold (Φ : ℝ → ℝ) (f : X → ℝ) (x : X) :
    ∃ g : X → ℝ, (∀ a b, Φ (g a - g b) = Φ (f a - f b)) ∧ (0 < f x ↔ ¬ 0 < g x) := by
  by_cases hx : 0 < f x
  · refine ⟨fun a => f a + (-f x - 1), fun a b => ?_, ?_⟩
    · exact pairwise_shift_invariant Φ f _ a b
    · simp only
      constructor
      · intro _ h
        linarith
      · intro _
        exact hx
  · refine ⟨fun a => f a + (-f x + 1), fun a b => ?_, ?_⟩
    · exact pairwise_shift_invariant Φ f _ a b
    · simp only
      constructor
      · intro h
        exact absurd h hx
      · intro h
        exact absurd (by linarith) h

/-- Unary likelihood `P(approve | x) = Φ(f x)` with a strictly monotone link
identifies `f` exactly. -/
theorem unary_identifies {Φ : ℝ → ℝ} (hΦ : StrictMono Φ) (f g : X → ℝ)
    (h : ∀ x, Φ (f x) = Φ (g x)) : f = g := by
  funext x
  exact hΦ.injective (h x)

/-- In particular the allow region `{x | Φ (f x) > Φ 0}` is identified. -/
theorem unary_identifies_boundary {Φ : ℝ → ℝ} (hΦ : StrictMono Φ) (f g : X → ℝ)
    (h : ∀ x, Φ (f x) = Φ (g x)) (x : X) : 0 < f x ↔ 0 < g x := by
  rw [unary_identifies hΦ f g h]

/-- Lower-confidence-bound rule (Remark on epistemic safety): with a strictly
monotone link and a threshold level `t₀` with `Φ t₀ = τ`, requiring
`Φ (μ - z σ) > τ` is equivalent to `μ - z σ > t₀`. -/
theorem lcb_iff {Φ : ℝ → ℝ} (hΦ : StrictMono Φ) {τ t₀ : ℝ} (ht : Φ t₀ = τ) (u : ℝ) :
    τ < Φ u ↔ t₀ < u := by
  rw [← ht]
  exact hΦ.lt_iff_lt

end TrustCalib
