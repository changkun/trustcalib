/-
Propositions 6-8 (manuscript, calibrating an opaque LLM judge): an opaque judge
in front of the gateway.

A judge emits ALLOW/BLOCK for each action; the supervisor approves with
probability `q x`. What matters to the supervisor is the judge's error rate
*for them*, e.g. `P(supervisor denies | judge ALLOW)`.

* Proposition 6 (non-identifiability): that rate is not a function of what can
  be observed without paired human labels. Even the two marginals together,
  how often the judge allows and how often the supervisor approves, leave it
  undetermined.
* Proposition 7 (relational reliability): for a fixed judge, the rate is
  monotone in the supervisor's tolerance, so the same judge becomes less
  reliable when the supervisor becomes stricter.
* Proposition 8 (audits with known propensities): labels collected with known,
  positive, possibly history-dependent propensities give an unbiased
  Horvitz-Thompson estimate.
-/
import Mathlib

namespace TrustCalib

open MeasureTheory Finset

/-- A joint distribution of (judge verdict, supervisor decision) on the four
cells: `aD` = judge ALLOW & supervisor denies, `aA` = ALLOW & approves,
`bD` = BLOCK & denies, `bA` = BLOCK & approves. -/
structure JointVD where
  aD : ℝ
  aA : ℝ
  bD : ℝ
  bA : ℝ
  nonneg : 0 ≤ aD ∧ 0 ≤ aA ∧ 0 ≤ bD ∧ 0 ≤ bA
  total : aD + aA + bD + bA = 1

namespace JointVD

/-- How often the judge allows. -/
def allowRate (p : JointVD) : ℝ := p.aD + p.aA

/-- How often the supervisor approves. -/
def approveRate (p : JointVD) : ℝ := p.aA + p.bA

/-- The judge's false-allow rate for this supervisor, `P(deny | judge ALLOW)`. -/
noncomputable def falseAllow (p : JointVD) : ℝ := p.aD / (p.aD + p.aA)

end JointVD

/-- Proposition 6(a): from the verdict marginal alone the false-allow rate is
completely unidentified: every allow rate `a ∈ (0,1)` is compatible with every
false-allow rate `φ ∈ [0,1]`. -/
theorem judge_unidentified {a φ : ℝ} (ha0 : 0 < a) (ha1 : a < 1) (hφ0 : 0 ≤ φ) (hφ1 : φ ≤ 1) :
    ∃ p : JointVD, p.allowRate = a ∧ p.falseAllow = φ := by
  refine ⟨⟨a * φ, a * (1 - φ), 1 - a, 0, ⟨by positivity, ?_, by linarith, le_refl _⟩,
    by ring⟩, ?_, ?_⟩
  · exact mul_nonneg ha0.le (by linarith)
  · simp only [JointVD.allowRate]; ring
  · simp only [JointVD.falseAllow]
    rw [show a * φ + a * (1 - φ) = a by ring]
    field_simp

/-- Proposition 6(b): even the two marginals together do not identify it. Two
joints with the same judge allow rate (0.8) and the same supervisor approval
rate (0.9) give false-allow rates 0 and 1/8. -/
theorem judge_marginals_not_enough :
    ∃ p q : JointVD, p.allowRate = q.allowRate ∧ p.approveRate = q.approveRate ∧
      p.falseAllow = 0 ∧ q.falseAllow = 1 / 8 := by
  refine ⟨⟨0, 8 / 10, 1 / 10, 1 / 10, ⟨le_refl _, by norm_num, by norm_num, by norm_num⟩,
    by norm_num⟩, ⟨1 / 10, 7 / 10, 0, 2 / 10, ⟨by norm_num, by norm_num, le_refl _, by norm_num⟩,
    by norm_num⟩, ?_, ?_, ?_, ?_⟩ <;>
    simp only [JointVD.allowRate, JointVD.approveRate, JointVD.falseAllow] <;> norm_num

variable {X : Type*}

/-- The judge's false-allow rate for a supervisor with approval probabilities
`q`, over a finite action population with weights `w`, where `A` is the set
of actions the judge allows. -/
noncomputable def judgeFA (w q : X → ℝ) (A : Finset X) : ℝ :=
  (∑ x ∈ A, w x * (1 - q x)) / ∑ x ∈ A, w x

/-- Proposition 7: reliability is relational. For a fixed judge (fixed `A`), if
supervisor `q'` is everywhere at most as permissive as `q` on the judge's
allowed actions, the judge's false-allow rate for `q'` is at least that for
`q`. -/
theorem judgeFA_mono (w q q' : X → ℝ) (A : Finset X) (hw : ∀ x ∈ A, 0 ≤ w x)
    (hpos : 0 < ∑ x ∈ A, w x) (hq : ∀ x ∈ A, q' x ≤ q x) :
    judgeFA w q A ≤ judgeFA w q' A := by
  unfold judgeFA
  apply (div_le_div_iff_of_pos_right hpos).mpr
  apply Finset.sum_le_sum
  intro x hx
  exact mul_le_mul_of_nonneg_left (by linarith [hq x hx]) (hw x hx)

variable {Ω : Type*} [MeasurableSpace Ω] {N : ℕ}

/-- Proposition 8: Horvitz-Thompson with per-action propensities. If label
indicator `L i` has mean `π i > 0`, then `Σ L i e i / π i` is unbiased for
`Σ e i`. (Escalations have propensity one, audited auto-decisions `ε`.) -/
theorem ht_unbiased (P : Measure Ω) [IsProbabilityMeasure P] (L : Fin N → Ω → ℝ)
    (hL : ∀ i, Integrable (L i) P) (π : Fin N → ℝ) (hπ : ∀ i, 0 < π i)
    (hmean : ∀ i, ∫ ω, L i ω ∂P = π i) (e : Fin N → ℝ) :
    ∫ ω, ∑ i, L i ω * e i / π i ∂P = ∑ i, e i := by
  rw [integral_finsetSum _ (fun i _ => ((hL i).mul_const _).div_const _)]
  refine Finset.sum_congr rfl (fun i _ => ?_)
  rw [integral_div, integral_mul_const, hmean i]
  field_simp [(hπ i).ne']

end TrustCalib
