/-
Proposition 3 (manuscript §5): the escalation floor is a property of the
supervisor, not of the learner.

Even with perfect knowledge of the true approval probability `q x = Φ(f*(x))`,
any policy that is optimal under the Chow loss must escalate every action with
`τlo < q x < τhi`. Hence, over any finite action population with nonnegative
weights, the escalated mass of such a policy is at least the mass of the band.
-/
import TrustCalib.Chow

namespace TrustCalib

open Finset

variable {X : Type*} [Fintype X]

/-- A policy is pointwise Chow-optimal for approval probabilities `q`. -/
def PointwiseOptimal (c : Costs) (q : X → ℝ) (π : X → Action) : Prop :=
  ∀ x a, c.loss (π x) (q x) ≤ c.loss a (q x)

omit [Fintype X] in
/-- Every pointwise-optimal policy escalates the whole open band. -/
theorem optimal_asks_on_band (c : Costs) (q : X → ℝ) (π : X → Action)
    (hπ : PointwiseOptimal c q π) (x : X) (hlo : c.tauLo < q x) (hhi : q x < c.tauHi) :
    π x = .ask := by
  obtain ⟨hA, hB⟩ := c.ask_strictly_optimal hlo hhi
  have h := hπ x .ask
  cases hx : π x with
  | allow => rw [hx] at h; exact absurd h (not_le.mpr hA)
  | ask => rfl
  | block => rw [hx] at h; exact absurd h (not_le.mpr hB)

/-- Escalation floor: the escalated mass of any pointwise-optimal policy is at
least the mass of actions whose true approval probability lies strictly inside
the band. -/
theorem escalation_floor (c : Costs) (q : X → ℝ) (w : X → ℝ) (hw : ∀ x, 0 ≤ w x)
    (π : X → Action) (hπ : PointwiseOptimal c q π) :
    (∑ x ∈ univ.filter (fun x => c.tauLo < q x ∧ q x < c.tauHi), w x) ≤
      ∑ x ∈ univ.filter (fun x => π x = .ask), w x := by
  refine Finset.sum_le_sum_of_subset_of_nonneg ?_ (fun x _ _ => hw x)
  intro x hx
  simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hx ⊢
  exact optimal_asks_on_band c q π hπ x hx.1 hx.2

end TrustCalib
