import ModeBoundLean.GateSpace
import ModeBoundLean.Prob

/-!
# Trace budgets (Proposition 1 and Theorem 7)

A row's fingerprint is `P = GᵀG / ‖G‖²_F` for its sensitivity matrix `G` with rows `g k` (`fingerprint g`): positive,
with trace `1`. That conserved trace bounds gates in two ways.

* **Proposition 1** (`gate_budget`, `opener_budget`, `opener_budget_count`): an opener's matrix
  `M = E[P | on] - E[P | off]` has gate spaces of dimension at most `1/t = d/γ`, for any model and any opener.
* **Theorem 7** (`linear_budget_gate`, `linear_budget`, `gate_count_le`): each gate pays `p · (E[q | on] - E[q | off])`,
  `q = sᵀPs`, out of `E[q]`; channels with frame bound `B` share at most `B` in total. Hence `m` gates with prevalence
  `≥ p_min` and contrast `≥ γ` number at most `B·d / (p_min·γ)`; with orthonormal channels, `d / (p_min·γ)`.
  Nothing here depends on the number of neurons.
-/

open Module Submodule Finset
open scoped InnerProductSpace RealInnerProductSpace

set_option linter.unusedSectionVars false

noncomputable section

namespace ModeBound

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]

/-- For a positive operator and an orthonormal family, `Σᵢ ⟪P vᵢ, vᵢ⟫ ≤ tr P`. -/
theorem sum_inner_le_trace {P : E →ₗ[ℝ] E} (hP : P.IsPositive) {ι : Type*} [Fintype ι] {v : ι → E}
    (hv : Orthonormal ℝ v) : ∑ i, ⟪P (v i), v i⟫ ≤ LinearMap.trace ℝ E P := by
  classical
  have hinj : Function.Injective v := hv.linearIndependent.injective
  have hs : Orthonormal ℝ ((↑) : Set.range v → E) := (orthonormal_subtype_range hinj).mpr hv
  obtain ⟨u, b, hsu, hb⟩ := hs.exists_orthonormalBasis_extension
  rw [LinearMap.trace_eq_sum_inner P b, hb]
  have hsub : univ.image v ⊆ u := by
    intro x hx
    obtain ⟨i, -, rfl⟩ := mem_image.mp hx
    exact hsu ⟨i, rfl⟩
  calc ∑ i, ⟪P (v i), v i⟫ = ∑ x ∈ univ.image v, ⟪x, P x⟫ := by
        rw [sum_image fun i _ j _ h => hinj h]
        exact sum_congr rfl fun i _ => real_inner_comm _ _
    _ ≤ ∑ x ∈ u, ⟪x, P x⟫ := sum_le_sum_of_subset_of_nonneg hsub fun x _ _ => hP.inner_nonneg_right x
    _ = ∑ x : u, ⟪(x : E), P x⟫ := (sum_coe_sort u fun x => ⟪x, P x⟫).symm

/-- **Proposition 1**: if `P₁, P₀` are positive and `tr P₁ = 1`, every gate space `V` of `P₁ - P₀` at level `t` has
`t · dim V ≤ 1`. -/
theorem gate_budget {P₁ P₀ : E →ₗ[ℝ] E} (h₁ : P₁.IsPositive) (h₀ : P₀.IsPositive)
    (htr : LinearMap.trace ℝ E P₁ = 1) {t : ℝ} {V : Submodule ℝ E} (hV : IsGateSpace (P₁ - P₀) t V) :
    t * finrank ℝ V ≤ 1 := by
  set b := stdOrthonormalBasis ℝ V
  let v : Fin (finrank ℝ V) → E := fun i => b i
  have hv : Orthonormal ℝ v := b.orthonormal.comp_linearIsometry V.subtypeₗᵢ
  have key : ∀ i, t ≤ ⟪P₁ (v i), v i⟫ - ⟪P₀ (v i), v i⟫ := by
    intro i
    have := hV (v i) (b i).2
    rwa [hv.1 i, one_pow, mul_one, LinearMap.sub_apply, inner_sub_left] at this
  calc t * finrank ℝ V = ∑ _i : Fin (finrank ℝ V), t := by simp [mul_comm]
    _ ≤ ∑ i, (⟪P₁ (v i), v i⟫ - ⟪P₀ (v i), v i⟫) := sum_le_sum fun i _ => key i
    _ ≤ ∑ i, ⟪P₁ (v i), v i⟫ := sum_le_sum fun i _ => by linarith [h₀.inner_nonneg_left (v i)]
    _ ≤ 1 := htr ▸ sum_inner_le_trace h₁ hv

/-! ### Fingerprints -/

/-- The fingerprint `P = GᵀG / ‖G‖²_F` of a row whose sensitivity matrix `G` has rows `g k`. -/
def fingerprint {K : Type*} [Fintype K] (g : K → E) : E →ₗ[ℝ] E :=
  (∑ k, ‖g k‖ ^ 2)⁻¹ • ∑ k, (innerₗ E (g k)).smulRight (g k)

section fingerprint

variable {K : Type*} [Fintype K] (g : K → E)

theorem inner_fingerprint (x y : E) :
    ⟪fingerprint g x, y⟫ = (∑ k, ⟪g k, x⟫ * ⟪g k, y⟫) / ∑ k, ‖g k‖ ^ 2 := by
  simp [fingerprint, LinearMap.sum_apply, sum_inner, inner_smul_left, div_eq_inv_mul, mul_sum]

/-- `sᵀPs = ‖Gs‖² / ‖G‖²_F`: the row's gain along `s` relative to its average gain (times `1/d`). -/
theorem inner_fingerprint_self (s : E) :
    ⟪fingerprint g s, s⟫ = (∑ k, ⟪g k, s⟫ ^ 2) / ∑ k, ‖g k‖ ^ 2 := by
  simp [inner_fingerprint, sq]

theorem fingerprint_isPositive : (fingerprint g).IsPositive := by
  rw [LinearMap.isPositive_iff]
  refine ⟨fun x y => ?_, fun x => ?_⟩
  · rw [inner_fingerprint, real_inner_comm (fingerprint g y) x, inner_fingerprint]
    exact congrArg (· / _) (sum_congr rfl fun k _ => mul_comm _ _)
  · rw [inner_fingerprint_self]
    exact div_nonneg (sum_nonneg fun _ _ => sq_nonneg _) (sum_nonneg fun _ _ => sq_nonneg _)

theorem trace_fingerprint (hZ : ∑ k, ‖g k‖ ^ 2 ≠ 0) : LinearMap.trace ℝ E (fingerprint g) = 1 := by
  set b := stdOrthonormalBasis ℝ E
  have h : ∀ i, ⟪b i, fingerprint g (b i)⟫ = (∑ k, ⟪g k, b i⟫ * ⟪g k, b i⟫) / ∑ k, ‖g k‖ ^ 2 := fun i => by
    rw [real_inner_comm (fingerprint g (b i)) (b i), inner_fingerprint]
  rw [LinearMap.trace_eq_sum_inner _ b]
  simp_rw [h, ← sum_div]
  rw [sum_comm, div_eq_one_iff_eq hZ]
  refine sum_congr rfl fun k _ => ?_
  rw [← real_inner_self_eq_norm_sq, ← b.sum_inner_mul_inner (g k) (g k)]
  exact sum_congr rfl fun i _ => by rw [real_inner_comm (b i) (g k)]

end fingerprint

/-! ### Openers over a finite row distribution -/

variable {Ω : Type*} [Fintype Ω] (μ : Ω → ℝ)

theorem isPositive_condMean (hμ : ∀ ρ, 0 ≤ μ ρ) (e : Ω → Prop) [DecidablePred e] {P : Ω → E →ₗ[ℝ] E}
    (hP : ∀ ρ, (P ρ).IsPositive) : (condMean μ e P).IsPositive := by
  have hm : 0 ≤ mass μ e := sum_nonneg fun ρ _ => hμ ρ
  refine LinearMap.IsPositive.smul_of_nonneg ?_ (inv_nonneg.mpr hm)
  exact sum_induction _ LinearMap.IsPositive (fun _ _ ha hb => ha.add hb) LinearMap.isPositive_zero
    fun ρ _ => (hP ρ).smul_of_nonneg (hμ ρ)

theorem trace_condMean (e : Ω → Prop) [DecidablePred e] {P : Ω → E →ₗ[ℝ] E}
    (htr : ∀ ρ, LinearMap.trace ℝ E (P ρ) = 1) (hm : mass μ e ≠ 0) :
    LinearMap.trace ℝ E (condMean μ e P) = 1 := by
  rw [map_condMean, condMean_real]
  simp only [htr, mul_one]
  exact div_self hm

/-- **Proposition 1** for an opener: with positive, trace-one fingerprints, every gate space `V` of
`M = E[P | on] - E[P | off]` at level `t` has `t · dim V ≤ 1`. -/
theorem opener_budget (hμ : ∀ ρ, 0 ≤ μ ρ) (e : Ω → Prop) [DecidablePred e] {P : Ω → E →ₗ[ℝ] E}
    (hP : ∀ ρ, (P ρ).IsPositive) (htr : ∀ ρ, LinearMap.trace ℝ E (P ρ) = 1) (hp : 0 < mass μ e) {t : ℝ}
    {V : Submodule ℝ E} (hV : IsGateSpace (condMean μ e P - condMean μ (fun ρ => ¬ e ρ) P) t V) :
    t * finrank ℝ V ≤ 1 :=
  gate_budget (isPositive_condMean μ hμ e hP) (isPositive_condMean μ hμ _ hP) (trace_condMean μ e htr hp.ne') hV

/-- **Proposition 1**, counted: at level `t = γ/d`, every gate space has dimension at most `d/γ`, for any fingerprints
`P = GᵀG / ‖G‖²_F` and any opener, whatever the number of neurons. -/
theorem opener_budget_count (hμ : ∀ ρ, 0 ≤ μ ρ) (e : Ω → Prop) [DecidablePred e] {K : Type*} [Fintype K]
    (g : Ω → K → E) (hZ : ∀ ρ, ∑ k, ‖g ρ k‖ ^ 2 ≠ 0) (hp : 0 < mass μ e) {γ : ℝ} (hγ : 0 < γ)
    {V : Submodule ℝ E}
    (hV : IsGateSpace (condMean μ e (fun ρ => fingerprint (g ρ)) -
      condMean μ (fun ρ => ¬ e ρ) (fun ρ => fingerprint (g ρ))) (γ / finrank ℝ E) V) :
    (finrank ℝ V : ℝ) ≤ finrank ℝ E / γ := by
  have h := opener_budget μ hμ e (fun ρ => fingerprint_isPositive (g ρ)) (fun ρ => trace_fingerprint _ (hZ ρ)) hp hV
  have hVE : (finrank ℝ V : ℝ) ≤ finrank ℝ E := by exact_mod_cast Submodule.finrank_le V
  rw [le_div_iff₀ hγ]
  rcases (Nat.cast_nonneg (α := ℝ) (finrank ℝ E)).eq_or_lt with hd | hd
  · rw [← hd] at hVE ⊢
    nlinarith [Nat.cast_nonneg (α := ℝ) (finrank ℝ V)]
  · rw [div_mul_eq_mul_div, div_le_one hd] at h
    linarith

/-- **Theorem 7** (one gate): for a channel quantity `q ≥ 0` (`q = sᵀPs`) and an opener of prevalence `p`,
`p · (E[q | on] - E[q | off]) ≤ E[q]`. With contrast `A = d · (E[q | on] - E[q | off])`: `p·A ≤ d·sᵀP̄s`. -/
theorem linear_budget_gate (hμ : ∀ ρ, 0 ≤ μ ρ) (hsum : ∑ ρ, μ ρ = 1) (e : Ω → Prop) [DecidablePred e]
    {q : Ω → ℝ} (hq : ∀ ρ, 0 ≤ q ρ) (hp : 0 < mass μ e) (hp' : mass μ e < 1) :
    mass μ e * (condMean μ e q - condMean μ (fun ρ => ¬ e ρ) q) ≤ mean μ q := by
  have hoff : mass μ (fun ρ => ¬ e ρ) = 1 - mass μ e := by linarith [mass_add_mass_not μ e]
  rw [condMean_real, condMean_real, hoff]
  set a := ∑ ρ ∈ univ.filter e, μ ρ * q ρ
  set b := ∑ ρ ∈ univ.filter (fun ρ => ¬ e ρ), μ ρ * q ρ
  have hb : 0 ≤ b := sum_nonneg fun ρ _ => mul_nonneg (hμ ρ) (hq ρ)
  have hm : mean μ q = a + b := (sum_filter_add_sum_filter_not univ e _).symm
  have h1 : 0 < 1 - mass μ e := by linarith
  have ha : mass μ e * (a / mass μ e) = a := by field_simp
  have hc : 0 ≤ mass μ e * (b / (1 - mass μ e)) := mul_nonneg hp.le (div_nonneg hb h1.le)
  rw [hm, mul_sub, ha]
  linarith

/-- Channels `s a` with frame bound `B` (`Σₐ ⟪sₐ, x⟫² ≤ B‖x‖²`; `B = 1` for orthonormal channels) share one
fingerprint: `Σₐ sₐᵀ P sₐ ≤ B`. -/
theorem sum_inner_fingerprint_le {K : Type*} [Fintype K] (g : K → E) {ι : Type*} [Fintype ι] (s : ι → E) {B : ℝ}
    (hB : ∀ x, ∑ a, ⟪s a, x⟫ ^ 2 ≤ B * ‖x‖ ^ 2) (hB0 : 0 ≤ B) :
    ∑ a, ⟪fingerprint g (s a), s a⟫ ≤ B := by
  simp_rw [inner_fingerprint_self, ← sum_div]
  rcases (sum_nonneg fun k (_ : k ∈ univ) => sq_nonneg ‖g k‖).eq_or_lt with hZ | hZ
  · rw [← hZ, div_zero]
    exact hB0
  rw [div_le_iff₀ hZ, sum_comm, mul_sum]
  exact sum_le_sum fun k _ => by simpa [real_inner_comm] using hB (g k)

/-- **Theorem 7**: gates `(e a, s a)` on channels with frame bound `B` satisfy
`Σₐ pₐ · (E[qₐ | on] - E[qₐ | off]) ≤ B`, `qₐ = sₐᵀPsₐ`: in contrast units, `Σₐ pₐ Aₐ ≤ d·B`. -/
theorem linear_budget (hμ : ∀ ρ, 0 ≤ μ ρ) (hsum : ∑ ρ, μ ρ = 1) {K : Type*} [Fintype K] (g : Ω → K → E)
    {ι : Type*} [Fintype ι] (s : ι → E) {B : ℝ} (hB : ∀ x, ∑ a, ⟪s a, x⟫ ^ 2 ≤ B * ‖x‖ ^ 2) (hB0 : 0 ≤ B)
    (e : ι → Ω → Prop) [∀ a, DecidablePred (e a)] (hp : ∀ a, 0 < mass μ (e a)) (hp' : ∀ a, mass μ (e a) < 1) :
    ∑ a, mass μ (e a) * (condMean μ (e a) (fun ρ => ⟪fingerprint (g ρ) (s a), s a⟫) -
      condMean μ (fun ρ => ¬ e a ρ) (fun ρ => ⟪fingerprint (g ρ) (s a), s a⟫)) ≤ B := by
  calc _ ≤ ∑ a, mean μ (fun ρ => ⟪fingerprint (g ρ) (s a), s a⟫) :=
        sum_le_sum fun a _ => linear_budget_gate μ hμ hsum (e a)
          (fun ρ => (fingerprint_isPositive (g ρ)).inner_nonneg_left _) (hp a) (hp' a)
    _ = ∑ ρ, μ ρ * ∑ a, ⟪fingerprint (g ρ) (s a), s a⟫ := by
        simp only [mean, mul_sum]
        exact sum_comm
    _ ≤ ∑ ρ, μ ρ * B := sum_le_sum fun ρ _ =>
        mul_le_mul_of_nonneg_left (sum_inner_fingerprint_le (g ρ) s hB hB0) (hμ ρ)
    _ = B := by rw [← sum_mul, hsum, one_mul]

/-- **Theorem 7**, counted: if every gate has prevalence `≥ p_min` and `E[q | on] - E[q | off] ≥ δ` (contrast
`≥ γ = d·δ`), then `m · p_min · δ ≤ B`, i.e. `m ≤ B·d / (p_min·γ)`, whatever the number of neurons. -/
theorem gate_count_le (hμ : ∀ ρ, 0 ≤ μ ρ) (hsum : ∑ ρ, μ ρ = 1) {K : Type*} [Fintype K] (g : Ω → K → E)
    {ι : Type*} [Fintype ι] (s : ι → E) {B : ℝ} (hB : ∀ x, ∑ a, ⟪s a, x⟫ ^ 2 ≤ B * ‖x‖ ^ 2) (hB0 : 0 ≤ B)
    (e : ι → Ω → Prop) [∀ a, DecidablePred (e a)] (hp' : ∀ a, mass μ (e a) < 1) {pmin δ : ℝ} (hpmin : 0 < pmin)
    (hδ : 0 ≤ δ) (hpa : ∀ a, pmin ≤ mass μ (e a))
    (hδa : ∀ a, δ ≤ condMean μ (e a) (fun ρ => ⟪fingerprint (g ρ) (s a), s a⟫) -
      condMean μ (fun ρ => ¬ e a ρ) (fun ρ => ⟪fingerprint (g ρ) (s a), s a⟫)) :
    Fintype.card ι * (pmin * δ) ≤ B := by
  have hp : ∀ a, 0 < mass μ (e a) := fun a => hpmin.trans_le (hpa a)
  calc (Fintype.card ι : ℝ) * (pmin * δ) = ∑ _a : ι, pmin * δ := by simp
    _ ≤ _ := sum_le_sum fun a _ => mul_le_mul (hpa a) (hδa a) hδ (hp a).le
    _ ≤ B := linear_budget μ hμ hsum g s hB hB0 e hp hp'

/-- Orthonormal channels have frame bound `1` (Bessel). -/
theorem frame_of_orthonormal {ι : Type*} [Fintype ι] {s : ι → E} (hs : Orthonormal ℝ s) (x : E) :
    ∑ a, ⟪s a, x⟫ ^ 2 ≤ 1 * ‖x‖ ^ 2 := by
  simpa [sq_abs] using hs.sum_inner_products_le (x := x) (s := univ)

end ModeBound

end
