import ModeBoundLean.Prob

/-!
# Variance budget (Theorem 5 (a), (c))

* **Theorem 5 (a)** (`contrast_sq_le_var`): a gate of prevalence `p` whose channel quantity `q` differs by `Δ` between
  on- and off-rows has `p(1-p)·Δ² ≤ Var q`: the contrast is paid for in variance (Cauchy–Schwarz against the indicator).
* **Theorem 5 (c)** (`sum_var_le`, `gate_count_var`): in the space of fingerprints with the Frobenius inner product, a
  channel quantity is `qₐ = ⟪Xₐ, P⟫` with `Xₐ = sₐsₐᵀ`. If the `Xₐ` have frame bound `B`, the variances share the total
  variance `κ = E‖P - P̄‖²`: `Σₐ Var qₐ ≤ B·κ`. So `m` gates with `p(1-p) ≥ c` and `|Δ| ≥ δ = γ/d` number at most
  `B·κ / (c·δ²)`, whatever the number of neurons. The fingerprint space is an arbitrary real inner product space `F`.
-/

open Finset
open scoped InnerProductSpace RealInnerProductSpace

noncomputable section

namespace ModeBound

variable {Ω : Type*} [Fintype Ω] (μ : Ω → ℝ)

/-- `Var q`. -/
def var (q : Ω → ℝ) : ℝ := ∑ ρ, μ ρ * (q ρ - mean μ q) ^ 2

/-- Weighted Cauchy–Schwarz. -/
theorem weighted_cauchy_schwarz (hμ : ∀ ρ, 0 ≤ μ ρ) (a b : Ω → ℝ) :
    (∑ ρ, μ ρ * a ρ * b ρ) ^ 2 ≤ (∑ ρ, μ ρ * a ρ ^ 2) * ∑ ρ, μ ρ * b ρ ^ 2 := by
  have h := sum_mul_sq_le_sq_mul_sq univ (fun ρ => √(μ ρ) * a ρ) (fun ρ => √(μ ρ) * b ρ)
  have e1 : ∀ ρ, √(μ ρ) * a ρ * (√(μ ρ) * b ρ) = μ ρ * a ρ * b ρ := fun ρ => by
    rw [show √(μ ρ) * a ρ * (√(μ ρ) * b ρ) = (√(μ ρ) * √(μ ρ)) * a ρ * b ρ by ring, Real.mul_self_sqrt (hμ ρ)]
  have e2 : ∀ (f : Ω → ℝ) ρ, (√(μ ρ) * f ρ) ^ 2 = μ ρ * f ρ ^ 2 := fun f ρ => by
    rw [mul_pow, Real.sq_sqrt (hμ ρ)]
  simpa only [e1, e2] using h

/-- **Theorem 5 (a)**: `p(1-p)·(E[q | on] - E[q | off])² ≤ Var q`. -/
theorem contrast_sq_le_var (hμ : ∀ ρ, 0 ≤ μ ρ) (hsum : ∑ ρ, μ ρ = 1) (e : Ω → Prop) [DecidablePred e]
    (q : Ω → ℝ) (hp : 0 < mass μ e) (hp' : mass μ e < 1) :
    mass μ e * (1 - mass μ e) * (condMean μ e q - condMean μ (fun ρ => ¬ e ρ) q) ^ 2 ≤ var μ q := by
  set p := mass μ e
  set m := mean μ q
  let ind : Ω → ℝ := fun ρ => if e ρ then 1 else 0
  set a := ∑ ρ ∈ univ.filter e, μ ρ * q ρ
  set b := ∑ ρ ∈ univ.filter (fun ρ => ¬ e ρ), μ ρ * q ρ
  have hoff : mass μ (fun ρ => ¬ e ρ) = 1 - p := by linarith [mass_add_mass_not μ e]
  have h1 : 0 < 1 - p := by linarith
  have hind : ∑ ρ, μ ρ * ind ρ = p := by simp [ind, p, mass, sum_filter]
  have hqind : ∑ ρ, μ ρ * q ρ * ind ρ = a := by simp [ind, a, sum_filter]
  have hm : m = a + b := (sum_filter_add_sum_filter_not univ e _).symm
  -- covariance of q with the indicator
  have hcov : ∑ ρ, μ ρ * (q ρ - m) * (ind ρ - p) = (1 - p) * a - p * b := by
    have : ∀ ρ, μ ρ * (q ρ - m) * (ind ρ - p) =
        μ ρ * q ρ * ind ρ - p * (μ ρ * q ρ) - m * (μ ρ * ind ρ) + m * p * μ ρ := fun ρ => by ring
    rw [sum_congr rfl fun ρ _ => this ρ]
    simp only [sum_add_distrib, sum_sub_distrib, ← mul_sum]
    rw [hqind, hind, hsum, show ∑ ρ, μ ρ * q ρ = m from rfl, hm]
    ring
  have hvar_ind : ∑ ρ, μ ρ * (ind ρ - p) ^ 2 = p * (1 - p) := by
    have : ∀ ρ, μ ρ * (ind ρ - p) ^ 2 = (1 - 2 * p) * (μ ρ * ind ρ) + p ^ 2 * μ ρ := fun ρ => by
      by_cases h : e ρ <;> simp [ind, h] <;> ring
    rw [sum_congr rfl fun ρ _ => this ρ, sum_add_distrib, ← mul_sum, ← mul_sum, hind, hsum]
    ring
  have hΔ : p * (1 - p) * (condMean μ e q - condMean μ (fun ρ => ¬ e ρ) q) = (1 - p) * a - p * b := by
    rw [condMean_real, condMean_real, hoff]
    change p * (1 - p) * (a / p - b / (1 - p)) = (1 - p) * a - p * b
    field_simp
  have hcs := weighted_cauchy_schwarz μ hμ (fun ρ => q ρ - m) (fun ρ => ind ρ - p)
  rw [hcov, hvar_ind, ← hΔ] at hcs
  have hc : 0 < p * (1 - p) := mul_pos hp h1
  refine le_of_mul_le_mul_left ?_ hc
  have hv : var μ q = ∑ ρ, μ ρ * (q ρ - m) ^ 2 := rfl
  rw [hv]
  nlinarith [hcs]

variable {F : Type*} [NormedAddCommGroup F] [InnerProductSpace ℝ F]

/-- **Theorem 5 (c)**, variance side: channel quantities `qₐ = ⟪Xₐ, Y⟫` of a random fingerprint `Y` with frame bound
`B` have `Σₐ Var qₐ ≤ B·E‖Y - Ȳ‖²`. -/
theorem sum_var_le (hμ : ∀ ρ, 0 ≤ μ ρ) (Y : Ω → F) {ι : Type*} [Fintype ι] (X : ι → F)
    {B : ℝ} (hB : ∀ y, ∑ a, ⟪X a, y⟫ ^ 2 ≤ B * ‖y‖ ^ 2) :
    ∑ a, var μ (fun ρ => ⟪X a, Y ρ⟫) ≤ B * ∑ ρ, μ ρ * ‖Y ρ - ∑ ρ', μ ρ' • Y ρ'‖ ^ 2 := by
  have hmean : ∀ a, mean μ (fun ρ => ⟪X a, Y ρ⟫) = ⟪X a, ∑ ρ', μ ρ' • Y ρ'⟫ := fun a => by
    simp [mean, inner_sum, inner_smul_right]
  simp only [var, hmean, ← inner_sub_right]
  rw [sum_comm, mul_sum]
  refine sum_le_sum fun ρ _ => ?_
  rw [← mul_sum, mul_left_comm]
  exact mul_le_mul_of_nonneg_left (hB _) (hμ ρ)

/-- **Theorem 5 (c)**, counted: gates `(e a, X a)` with `p(1-p) ≥ c` and `|E[qₐ | on] - E[qₐ | off]| ≥ δ` satisfy
`m · c · δ² ≤ B·κ`, `κ = E‖Y - Ȳ‖²`. With `δ = γ/d`: `m ≤ B·κ·d² / (c·γ²)`. -/
theorem gate_count_var (hμ : ∀ ρ, 0 ≤ μ ρ) (hsum : ∑ ρ, μ ρ = 1) (Y : Ω → F) {ι : Type*} [Fintype ι] (X : ι → F)
    {B : ℝ} (hB : ∀ y, ∑ a, ⟪X a, y⟫ ^ 2 ≤ B * ‖y‖ ^ 2) (e : ι → Ω → Prop) [∀ a, DecidablePred (e a)]
    (hp : ∀ a, 0 < mass μ (e a)) (hp' : ∀ a, mass μ (e a) < 1) {c δ : ℝ} (hc : ∀ a, c ≤ mass μ (e a) * (1 - mass μ (e a)))
    (hδ : 0 ≤ δ)
    (hδa : ∀ a, δ ≤ |condMean μ (e a) (fun ρ => ⟪X a, Y ρ⟫) - condMean μ (fun ρ => ¬ e a ρ) (fun ρ => ⟪X a, Y ρ⟫)|) :
    Fintype.card ι * (c * δ ^ 2) ≤ B * ∑ ρ, μ ρ * ‖Y ρ - ∑ ρ', μ ρ' • Y ρ'‖ ^ 2 := by
  calc (Fintype.card ι : ℝ) * (c * δ ^ 2) = ∑ _a : ι, c * δ ^ 2 := by simp
    _ ≤ ∑ a, var μ (fun ρ => ⟪X a, Y ρ⟫) := sum_le_sum fun a _ => by
        have h := contrast_sq_le_var μ hμ hsum (e a) (fun ρ => ⟪X a, Y ρ⟫) (hp a) (hp' a)
        have hpq : 0 ≤ mass μ (e a) * (1 - mass μ (e a)) := mul_nonneg (hp a).le (by linarith [hp' a])
        have hsq : δ ^ 2 ≤ (condMean μ (e a) (fun ρ => ⟪X a, Y ρ⟫) -
            condMean μ (fun ρ => ¬ e a ρ) (fun ρ => ⟪X a, Y ρ⟫)) ^ 2 := by
          rw [← sq_abs (condMean _ _ _ - _)]
          exact pow_le_pow_left₀ hδ (hδa a) 2
        calc c * δ ^ 2 ≤ mass μ (e a) * (1 - mass μ (e a)) * δ ^ 2 :=
              mul_le_mul_of_nonneg_right (hc a) (sq_nonneg δ)
          _ ≤ _ := mul_le_mul_of_nonneg_left hsq hpq
          _ ≤ _ := h
    _ ≤ _ := sum_var_le μ hμ Y X hB

end ModeBound

end
