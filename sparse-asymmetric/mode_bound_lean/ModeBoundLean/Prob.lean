import Mathlib

/-!
# Finite row distributions

Rows `ρ : Ω` carry weights `μ ρ ≥ 0` summing to `1` (in the experiments, the empirical distribution of the fitting
rows). An opener is an event `e : Ω → Prop`; `mass μ e` is its prevalence `p`, `condMean μ e f` the conditional mean
`E[f | e]` of a vector-valued `f`. Linear maps commute with conditional means (`map_condMean`), which is what makes
the opener's matrix `E[P | on] - E[P | off]` linear in the fingerprints.
-/

open Finset

noncomputable section

namespace ModeBound

variable {Ω : Type*} [Fintype Ω] (μ : Ω → ℝ)

/-- The prevalence `p = μ(e)` of an event. -/
def mass (e : Ω → Prop) [DecidablePred e] : ℝ := ∑ ρ ∈ univ.filter e, μ ρ

/-- The conditional mean `E[f | e]`. -/
def condMean {X : Type*} [AddCommGroup X] [Module ℝ X] (e : Ω → Prop) [DecidablePred e] (f : Ω → X) : X :=
  (mass μ e)⁻¹ • ∑ ρ ∈ univ.filter e, μ ρ • f ρ

/-- The mean `E[q]` of a real quantity. -/
def mean (q : Ω → ℝ) : ℝ := ∑ ρ, μ ρ * q ρ

theorem map_condMean {X Y : Type*} [AddCommGroup X] [Module ℝ X] [AddCommGroup Y] [Module ℝ Y] (g : X →ₗ[ℝ] Y)
    (e : Ω → Prop) [DecidablePred e] (f : Ω → X) : g (condMean μ e f) = condMean μ e (fun ρ => g (f ρ)) := by
  simp [condMean, map_sum, map_smul]

/-- The masses of an event and of its complement add up to the total mass. -/
theorem mass_add_mass_not (e : Ω → Prop) [DecidablePred e] :
    mass μ e + mass μ (fun ρ => ¬ e ρ) = ∑ ρ, μ ρ :=
  sum_filter_add_sum_filter_not univ e μ

theorem condMean_real (e : Ω → Prop) [DecidablePred e] (q : Ω → ℝ) :
    condMean μ e q = (∑ ρ ∈ univ.filter e, μ ρ * q ρ) / mass μ e := by
  simp [condMean, smul_eq_mul, div_eq_inv_mul]

end ModeBound

end
