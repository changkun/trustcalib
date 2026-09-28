/-
Proposition 2 (manuscript, decision rule): the three-tier gateway rule is the
Bayes-optimal one-step decision under Chow's reject-option loss.

Costs: a false ALLOW costs `cFA`, a false BLOCK costs `cFB`, an escalation (ASK)
costs `cAsk`; all strictly positive. For a belief `p = P(approve)` the expected
losses are
  ALLOW : (1 - p) * cFA,   BLOCK : p * cFB,   ASK : cAsk.
The optimal action thresholds `p` at `τhi = 1 - cAsk/cFA` and `τlo = cAsk/cFB`,
and the ASK band is non-empty iff `cAsk * (cFA + cFB) < cFA * cFB`.

Because each expected loss is affine in the approval probability, the expected
loss under a posterior over the latent approval probability depends on that
posterior only through its mean `p̂ = E[Φ(f)]` (`expectedLoss_allow`,
`expectedLoss_block`). Hence thresholding `p̂` is Bayes-optimal.
-/
import Mathlib

namespace TrustCalib

open MeasureTheory

/-- The three gateway actions. -/
inductive Action
  | allow
  | ask
  | block
  deriving DecidableEq

/-- Chow cost structure with strictly positive costs. -/
structure Costs where
  cFA : ℝ
  cFB : ℝ
  cAsk : ℝ
  hFA : 0 < cFA
  hFB : 0 < cFB
  hAsk : 0 < cAsk

namespace Costs

variable (c : Costs)

/-- Expected loss of an action when the approval probability is `p`. -/
def loss : Action → ℝ → ℝ
  | .allow, p => (1 - p) * c.cFA
  | .block, p => p * c.cFB
  | .ask, _ => c.cAsk

/-- Upper threshold `τhi = 1 - cAsk / cFA`. -/
noncomputable def tauHi : ℝ := 1 - c.cAsk / c.cFA

/-- Lower threshold `τlo = cAsk / cFB`. -/
noncomputable def tauLo : ℝ := c.cAsk / c.cFB

/-- The three-tier rule of manuscript eq. (decision). -/
noncomputable def decide (p : ℝ) : Action :=
  if c.tauHi < p then .allow else if p < c.tauLo then .block else .ask

/-- The ASK band is non-empty iff `cAsk (cFA + cFB) < cFA cFB`. -/
theorem band_nonempty_iff : c.tauLo < c.tauHi ↔ c.cAsk * (c.cFA + c.cFB) < c.cFA * c.cFB := by
  have hFA := c.hFA
  have hFB := c.hFB
  unfold Costs.tauLo Costs.tauHi
  rw [one_sub_div hFA.ne', div_lt_div_iff₀ hFB hFA]
  constructor <;> intro h <;> nlinarith

/-- Above `τhi`, ALLOW is strictly better than both alternatives (band non-empty). -/
theorem allow_strictly_optimal (hband : c.tauLo < c.tauHi) {p : ℝ} (hp : c.tauHi < p) :
    c.loss .allow p < c.loss .ask p ∧ c.loss .allow p < c.loss .block p := by
  have hFA := c.hFA
  have hFB := c.hFB
  have hAsk := c.hAsk
  have hb := (c.band_nonempty_iff).1 hband
  simp only [Costs.loss]
  have hp' := hp
  rw [Costs.tauHi, sub_lt_comm, lt_div_iff₀ hFA] at hp'
  refine ⟨hp', ?_⟩
  -- `(1 - p) * cFA < cAsk`, and `cAsk * (cFA + cFB) < cFA * cFB`
  nlinarith [mul_lt_mul_of_pos_right hp' (add_pos hFA hFB)]

/-- Below `τlo`, BLOCK is strictly better than both alternatives (band non-empty). -/
theorem block_strictly_optimal (hband : c.tauLo < c.tauHi) {p : ℝ} (hp : p < c.tauLo) :
    c.loss .block p < c.loss .ask p ∧ c.loss .block p < c.loss .allow p := by
  have hFA := c.hFA
  have hFB := c.hFB
  have hAsk := c.hAsk
  have hb := (c.band_nonempty_iff).1 hband
  simp only [Costs.loss]
  have hp' := hp
  rw [Costs.tauLo, lt_div_iff₀ hFB] at hp'
  refine ⟨hp', ?_⟩
  nlinarith [mul_lt_mul_of_pos_right hp' (add_pos hFA hFB)]

/-- Strictly inside the band, ASK is strictly better than both auto-decisions. -/
theorem ask_strictly_optimal {p : ℝ} (hlo : c.tauLo < p) (hhi : p < c.tauHi) :
    c.loss .ask p < c.loss .allow p ∧ c.loss .ask p < c.loss .block p := by
  have hFA := c.hFA
  have hFB := c.hFB
  simp only [Costs.loss]
  rw [Costs.tauLo, div_lt_iff₀ hFB] at hlo
  rw [Costs.tauHi, lt_sub_comm, div_lt_iff₀ hFA] at hhi
  exact ⟨hhi, hlo⟩

/-- The three-tier rule attains the minimum expected loss among all actions. -/
theorem decide_optimal (hband : c.tauLo < c.tauHi) (p : ℝ) (a : Action) :
    c.loss (c.decide p) p ≤ c.loss a p := by
  have hFA := c.hFA
  have hFB := c.hFB
  unfold Costs.decide
  split_ifs with h1 h2
  · obtain ⟨ha, hb⟩ := c.allow_strictly_optimal hband h1
    cases a
    · exact le_rfl
    · exact ha.le
    · exact hb.le
  · obtain ⟨ha, hb⟩ := c.block_strictly_optimal hband h2
    cases a
    · exact hb.le
    · exact ha.le
    · exact le_rfl
  · rw [not_lt, Costs.tauHi, le_sub_comm, div_le_iff₀ hFA] at h1
    rw [not_lt, Costs.tauLo, div_le_iff₀ hFB] at h2
    cases a <;> simp only [Costs.loss, le_refl] <;> assumption

/-- If the band is empty, escalating is never strictly better than the better
auto-decision: some auto-decision is always at least as good as ASK. -/
theorem ask_never_needed (hempty : c.cFA * c.cFB ≤ c.cAsk * (c.cFA + c.cFB)) (p : ℝ) :
    min (c.loss .allow p) (c.loss .block p) ≤ c.loss .ask p := by
  have hFA := c.hFA
  have hFB := c.hFB
  simp only [Costs.loss]
  by_cases h : (1 - p) * c.cFA ≤ p * c.cFB
  · rw [min_eq_left h]
    nlinarith [mul_le_mul_of_nonneg_left h hFA.le]
  · have h := lt_of_not_ge h
    rw [min_eq_right h.le]
    nlinarith [mul_lt_mul_of_pos_left h hFB]

/-- Posterior-mean sufficiency for ALLOW: for any probability measure `μ` on the
latent value and any integrable link `Φ`, the posterior expected loss of ALLOW is
the loss evaluated at the posterior-mean approval probability `∫ Φ dμ`. -/
theorem expectedLoss_allow (μ : Measure ℝ) [IsProbabilityMeasure μ] (Φ : ℝ → ℝ)
    (hΦ : Integrable Φ μ) :
    ∫ f, c.loss .allow (Φ f) ∂μ = c.loss .allow (∫ f, Φ f ∂μ) := by
  simp only [Costs.loss]
  rw [integral_mul_const, integral_sub (integrable_const _) hΦ]
  simp

/-- Posterior-mean sufficiency for BLOCK (no integrability needed: the loss is
linear without an intercept, and a non-integrable `Φ` has integral `0` on both
sides by convention). -/
theorem expectedLoss_block (μ : Measure ℝ) [IsProbabilityMeasure μ] (Φ : ℝ → ℝ) :
    ∫ f, c.loss .block (Φ f) ∂μ = c.loss .block (∫ f, Φ f ∂μ) := by
  simp only [Costs.loss]
  rw [integral_mul_const]

end Costs

end TrustCalib
