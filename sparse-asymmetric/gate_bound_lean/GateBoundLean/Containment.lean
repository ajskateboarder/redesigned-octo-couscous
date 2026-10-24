import GateBoundLean.Variance

/-!
# The gated subspace (Theorem 8): containment, counting, and the spiked–incoherent class

Rows `ρ` carry operators `P ρ` (fingerprints, or anything else), and an opener event `e` has matrix
`M = E[P | on] - E[P | off]`. A *gate direction* at level `t` is an eigenvector `M s = λ s` with `λ ≥ t`. Fix any subspace `V`
and write `w = s - Π_V s`.

* `containment`: if `⟪M w, w⟫ ≤ τ⊥‖w‖²` on `Vᗮ` (outside capacity) and `⟪M v, w⟫ ≤ τ×‖v‖‖w‖` for `v ∈ V`, `w ∈ Vᗮ`
  (cross capacity), then every eigenvector with `λ > τ⊥` has `((λ - τ⊥)² + τ×²)·‖s - Π_V s‖² ≤ τ×²‖s‖²`: strong gates are
  confined to `V`. No assumption on `M`, the rows, the events or how the gate is implemented.
* `card_le_of_contained`: unit channels with frame bound `B`, each with `‖s - Π_V s‖² ≤ ε`, number at most
  `B·dim V / (1 - ε)`.
* `abs_contrast_le_of_var`: a channel quantity with `Var q ≤ σ²` has `|E[q | on] - E[q | off]| ≤ σ / √(p(1-p))`, for any
  event (Theorem 5 (a)). This turns variance bounds into capacities.
* `spiked_containment`, `spiked_count`: **the spiked–incoherent class.** If every row splits as `P = S + N` with `S`
  living on `V` and `N`'s quadratic and bilinear forms off `V` of variance at most `σ⊥²‖w‖⁴` and `σ×²‖v‖²‖w‖²`, then every
  gate direction at level `t > τ⊥ := σ⊥/√(p(1-p))` has at most `ε = τ×² / (τ×² + (t - τ⊥)²)` of its mass off `V`
  (`τ× := σ×/√(p(1-p))`), and gate directions with frame bound `B` number at most `B·dim V / (1 - ε)`, whatever the events.
  For an isotropic remainder `N`, `σ⊥, σ×` are `O(1/d)` (gate_bound.md, Theorem 8 (c)), so `τ⊥, τ×` are `O(1)` in contrast
  units: only the `dim V` coherent directions can carry gates.
* `concentratedCount`, `card_le_concentratedCount` (**Corollary 8.1**): orthonormal gates number at most `N_c`, the largest
  number of orthonormal directions whose own channel quantity has variance `≥ c·δ²`. This sharpens the count of
  `spiked_count` from the dimension of the coherent subspace to its directions of concentrated variation.
-/

open Module Submodule Finset
open scoped InnerProductSpace RealInnerProductSpace

set_option linter.unusedSectionVars false

noncomputable section

namespace GateBound

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]

/-- **Containment.** With outside capacity `τ⊥` and cross capacity `τ×` of a subspace `V`, every eigenvector `M s = λ s`
with `λ > τ⊥` satisfies `((λ - τ⊥)² + τ×²)·‖s - Π_V s‖² ≤ τ×²·‖s‖²`. -/
theorem containment {M : E →ₗ[ℝ] E} (V : Submodule ℝ E) {τp τx lam : ℝ}
    (hperp : ∀ w ∈ Vᗮ, ⟪M w, w⟫ ≤ τp * ‖w‖ ^ 2)
    (hcross : ∀ v ∈ V, ∀ w ∈ Vᗮ, ⟪M v, w⟫ ≤ τx * (‖v‖ * ‖w‖)) (hτx : 0 ≤ τx) (hlam : τp < lam)
    {s : E} (hs : M s = lam • s) :
    ((lam - τp) ^ 2 + τx ^ 2) * ‖s - V.starProjection s‖ ^ 2 ≤ τx ^ 2 * ‖s‖ ^ 2 := by
  set v := V.starProjection s
  set w := s - v
  have hv : v ∈ V := V.starProjection_apply_mem s
  have hw : w ∈ Vᗮ := V.sub_starProjection_mem_orthogonal s
  have hvw : ⟪v, w⟫ = 0 := Submodule.inner_right_of_mem_orthogonal hv hw
  have hsvw : s = v + w := by simp [w]
  have hwv : ⟪w, v⟫ = 0 := by rw [real_inner_comm]; exact hvw
  have hpyth : ‖s‖ ^ 2 = ‖v‖ ^ 2 + ‖w‖ ^ 2 := by
    rw [hsvw, ← real_inner_self_eq_norm_sq, ← real_inner_self_eq_norm_sq, ← real_inner_self_eq_norm_sq]
    simp only [inner_add_left, inner_add_right, hvw, hwv]
    ring
  -- `λ‖w‖² = ⟪M s, w⟫ = ⟪M v, w⟫ + ⟪M w, w⟫ ≤ τ×‖v‖‖w‖ + τ⊥‖w‖²`
  have key : lam * ‖w‖ ^ 2 ≤ τx * (‖v‖ * ‖w‖) + τp * ‖w‖ ^ 2 := by
    have h1 : ⟪M s, w⟫ = lam * ‖w‖ ^ 2 := by
      rw [hs, real_inner_smul_left, hsvw, inner_add_left, hvw, zero_add, real_inner_self_eq_norm_sq]
    have h2 : ⟪M s, w⟫ = ⟪M v, w⟫ + ⟪M w, w⟫ := by rw [hsvw, map_add, inner_add_left]
    linarith [hcross v hv w hw, hperp w hw]
  set k := lam - τp
  have hk : 0 < k := by simp only [k]; linarith
  -- so `k‖w‖ ≤ τ×‖v‖`, hence `k²‖w‖² ≤ τ×²‖v‖²`
  have hsq : k ^ 2 * ‖w‖ ^ 2 ≤ τx ^ 2 * ‖v‖ ^ 2 := by
    rcases (norm_nonneg w).eq_or_lt with h0 | hpos
    · rw [← h0]
      simp only [ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow, mul_zero]
      positivity
    · have hlin : k * ‖w‖ ≤ τx * ‖v‖ := by
        have : k * ‖w‖ ^ 2 ≤ τx * ‖v‖ * ‖w‖ := by simp only [k]; nlinarith [key]
        have := (mul_le_mul_iff_of_pos_right hpos).mp (by nlinarith [this] : k * ‖w‖ * ‖w‖ ≤ τx * ‖v‖ * ‖w‖)
        exact this
      have h0 : 0 ≤ k * ‖w‖ := mul_nonneg hk.le (norm_nonneg w)
      nlinarith [mul_le_mul hlin hlin h0 (mul_nonneg hτx (norm_nonneg v))]
  rw [hpyth]
  nlinarith [hsq]

/-- **Counting contained channels.** Unit channels with frame bound `B` (`Σₐ ⟪sₐ, x⟫² ≤ B‖x‖²`), each with
`‖sₐ - Π_V sₐ‖² ≤ ε`, satisfy `m·(1 - ε) ≤ B·dim V`. -/
theorem card_le_of_contained (V : Submodule ℝ E) {ι : Type*} [Fintype ι] {s : ι → E} (hs : ∀ a, ‖s a‖ = 1)
    {B : ℝ} (hB : ∀ x, ∑ a, ⟪s a, x⟫ ^ 2 ≤ B * ‖x‖ ^ 2) {ε : ℝ}
    (hε : ∀ a, ‖s a - V.starProjection (s a)‖ ^ 2 ≤ ε) :
    Fintype.card ι * (1 - ε) ≤ B * finrank ℝ V := by
  set b := stdOrthonormalBasis ℝ V
  let e : Fin (finrank ℝ V) → E := fun i => b i
  have he : Orthonormal ℝ e := b.orthonormal.comp_linearIsometry V.subtypeₗᵢ
  have heV : ∀ i, e i ∈ V := fun i => (b i).2
  -- `‖Π_V x‖² = Σᵢ ⟪eᵢ, x⟫²`
  have hproj : ∀ x, ‖V.starProjection x‖ ^ 2 = ∑ i, ⟪e i, x⟫ ^ 2 := by
    intro x
    have hy := b.sum_sq_inner_right (⟨V.starProjection x, V.starProjection_apply_mem x⟩ : V)
    calc ‖V.starProjection x‖ ^ 2 = ‖(⟨V.starProjection x, V.starProjection_apply_mem x⟩ : V)‖ ^ 2 := rfl
      _ = _ := hy.symm
      _ = ∑ i, ⟪e i, x⟫ ^ 2 := sum_congr rfl fun i _ => by
          have h : ⟪e i, V.starProjection x⟫ = ⟪e i, x⟫ := by
            rw [← V.inner_starProjection_left_eq_right, (V.starProjection_eq_self_iff).mpr (heV i)]
          show ⟪e i, V.starProjection x⟫ ^ 2 = _
          rw [h]
  -- each channel keeps `≥ 1 - ε` of its mass in `V`
  have hmass : ∀ a, 1 - ε ≤ ‖V.starProjection (s a)‖ ^ 2 := by
    intro a
    have h := V.norm_sq_eq_add_norm_sq_starProjection (s a)
    have hmem : s a - V.starProjection (s a) ∈ Vᗮ := V.sub_starProjection_mem_orthogonal (s a)
    have hpyth : ‖s a‖ ^ 2 = ‖V.starProjection (s a)‖ ^ 2 + ‖s a - V.starProjection (s a)‖ ^ 2 := by
      have hvw : ⟪V.starProjection (s a), s a - V.starProjection (s a)⟫ = 0 :=
        Submodule.inner_right_of_mem_orthogonal (V.starProjection_apply_mem _) hmem
      have hwv : ⟪s a - V.starProjection (s a), V.starProjection (s a)⟫ = 0 := by rw [real_inner_comm]; exact hvw
      have hsum : s a = V.starProjection (s a) + (s a - V.starProjection (s a)) := by abel
      conv_lhs => rw [hsum]
      rw [← real_inner_self_eq_norm_sq, ← real_inner_self_eq_norm_sq, ← real_inner_self_eq_norm_sq]
      simp only [inner_add_left, inner_add_right, hvw, hwv]
      ring
    rw [hs a, one_pow] at hpyth
    linarith [hε a]
  calc (Fintype.card ι : ℝ) * (1 - ε) = ∑ _a : ι, (1 - ε) := by rw [sum_const, card_univ, nsmul_eq_mul]
    _ ≤ ∑ a, ‖V.starProjection (s a)‖ ^ 2 := sum_le_sum fun a _ => hmass a
    _ = ∑ i, ∑ a, ⟪s a, e i⟫ ^ 2 := by
        simp_rw [hproj, real_inner_comm (e _)]
        exact sum_comm
    _ ≤ ∑ _i : Fin (finrank ℝ V), B := sum_le_sum fun i _ => by
        simpa [he.1 i] using hB (e i)
    _ = B * finrank ℝ V := by simp [mul_comm]

variable {Ω : Type*} [Fintype Ω] (μ : Ω → ℝ)

/-- The contrast of a channel quantity with `Var q ≤ σ²` is at most `σ / √(p(1-p))`, for any event. -/
theorem abs_contrast_le_of_var (hμ : ∀ ρ, 0 ≤ μ ρ) (hsum : ∑ ρ, μ ρ = 1) (e : Ω → Prop) [DecidablePred e]
    (q : Ω → ℝ) (hp : 0 < mass μ e) (hp' : mass μ e < 1) {σ : ℝ} (hσ : 0 ≤ σ) (hvar : var μ q ≤ σ ^ 2) :
    |condMean μ e q - condMean μ (fun ρ => ¬ e ρ) q| ≤ σ / √(mass μ e * (1 - mass μ e)) := by
  have hc : 0 < mass μ e * (1 - mass μ e) := mul_pos hp (by linarith)
  have h := Real.sqrt_le_sqrt ((contrast_sq_le_var μ hμ hsum e q hp hp').trans hvar)
  rw [Real.sqrt_mul hc.le, Real.sqrt_sq_eq_abs, Real.sqrt_sq hσ] at h
  rw [le_div_iff₀ (Real.sqrt_pos.mpr hc), mul_comm]
  exact h

/-- Inner products commute with conditional means of operators. -/
theorem inner_condMean_apply (e : Ω → Prop) [DecidablePred e] (P : Ω → E →ₗ[ℝ] E) (x y : E) :
    ⟪condMean μ e P x, y⟫ = condMean μ e (fun ρ => ⟪P ρ x, y⟫) := by
  simp [condMean, LinearMap.sum_apply, sum_inner, real_inner_smul_left, smul_eq_mul, mul_sum]

/-- **The spiked–incoherent class, one gate.** Rows split as `P = S + N`, where `S` lives on `V`
(`⟪S x, y⟫ = 0` if `x` or `y` is in `Vᗮ`) and the remainder `N` has `Var ⟪N w, w⟫ ≤ σ⊥²‖w‖⁴` for `w ∈ Vᗮ` and
`Var ⟪N v, w⟫ ≤ σ×²‖v‖²‖w‖²` for `v ∈ V`, `w ∈ Vᗮ`. Then every gate direction `M s = λ s` of any event, with
`λ > τ⊥ = σ⊥/√(p(1-p))`, has `((λ - τ⊥)² + τ×²)·‖s - Π_V s‖² ≤ τ×²‖s‖²`, `τ× = σ×/√(p(1-p))`. -/
theorem spiked_containment (hμ : ∀ ρ, 0 ≤ μ ρ) (hsum : ∑ ρ, μ ρ = 1) (P S N : Ω → E →ₗ[ℝ] E)
    (V : Submodule ℝ E) (hPSN : ∀ ρ, P ρ = S ρ + N ρ) (hS : ∀ ρ x y, (x ∈ Vᗮ ∨ y ∈ Vᗮ) → ⟪S ρ x, y⟫ = 0)
    {σp σx : ℝ} (hσp : 0 ≤ σp) (hσx : 0 ≤ σx)
    (hNp : ∀ w ∈ Vᗮ, var μ (fun ρ => ⟪N ρ w, w⟫) ≤ σp ^ 2 * ‖w‖ ^ 4)
    (hNx : ∀ v ∈ V, ∀ w ∈ Vᗮ, var μ (fun ρ => ⟪N ρ v, w⟫) ≤ σx ^ 2 * (‖v‖ ^ 2 * ‖w‖ ^ 2))
    (e : Ω → Prop) [DecidablePred e] (hp : 0 < mass μ e) (hp' : mass μ e < 1) {lam : ℝ}
    (hlam : σp / √(mass μ e * (1 - mass μ e)) < lam) {s : E}
    (hs : (condMean μ e P - condMean μ (fun ρ => ¬ e ρ) P) s = lam • s) :
    ((lam - σp / √(mass μ e * (1 - mass μ e))) ^ 2 + (σx / √(mass μ e * (1 - mass μ e))) ^ 2) *
        ‖s - V.starProjection s‖ ^ 2 ≤ (σx / √(mass μ e * (1 - mass μ e))) ^ 2 * ‖s‖ ^ 2 := by
  set c := √(mass μ e * (1 - mass μ e))
  have hM : ∀ x y, ⟪(condMean μ e P - condMean μ (fun ρ => ¬ e ρ) P) x, y⟫ =
      condMean μ e (fun ρ => ⟪P ρ x, y⟫) - condMean μ (fun ρ => ¬ e ρ) (fun ρ => ⟪P ρ x, y⟫) := fun x y => by
    rw [LinearMap.sub_apply, inner_sub_left, inner_condMean_apply, inner_condMean_apply]
  -- off `V`, and across, only `N` is seen
  have hPN : ∀ ρ x y, (x ∈ Vᗮ ∨ y ∈ Vᗮ) → ⟪P ρ x, y⟫ = ⟪N ρ x, y⟫ := fun ρ x y h => by
    rw [hPSN, LinearMap.add_apply, inner_add_left, hS ρ x y h, zero_add]
  refine containment V (τp := σp / c) (τx := σx / c) (fun w hw => ?_) (fun v hv w hw => ?_)
    (div_nonneg hσx (Real.sqrt_nonneg _)) hlam hs
  · rw [hM]
    simp_rw [hPN _ w w (Or.inl hw)]
    have h := abs_contrast_le_of_var μ hμ hsum e (fun ρ => ⟪N ρ w, w⟫) hp hp' (σ := σp * ‖w‖ ^ 2)
      (by positivity) (by rw [mul_pow, ← pow_mul]; exact hNp w hw)
    calc _ ≤ |condMean μ e (fun ρ => ⟪N ρ w, w⟫) - condMean μ (fun ρ => ¬ e ρ) (fun ρ => ⟪N ρ w, w⟫)| := le_abs_self _
      _ ≤ σp * ‖w‖ ^ 2 / c := h
      _ = σp / c * ‖w‖ ^ 2 := by ring
  · rw [hM]
    simp_rw [hPN _ v w (Or.inr hw)]
    have h := abs_contrast_le_of_var μ hμ hsum e (fun ρ => ⟪N ρ v, w⟫) hp hp' (σ := σx * (‖v‖ * ‖w‖))
      (by positivity) (by rw [mul_pow, mul_pow]; exact hNx v hv w hw)
    calc _ ≤ |condMean μ e (fun ρ => ⟪N ρ v, w⟫) - condMean μ (fun ρ => ¬ e ρ) (fun ρ => ⟪N ρ v, w⟫)| := le_abs_self _
      _ ≤ σx * (‖v‖ * ‖w‖) / c := h
      _ = σx / c * (‖v‖ * ‖w‖) := by ring

/-- **The spiked–incoherent class, counted.** In the setting of `spiked_containment`, take unit channels `sₐ` with frame
bound `B`, each a gate direction `Mₐ sₐ = λₐ sₐ` of its own event `eₐ` (prevalence `p`) with `λₐ ≥ t > τ⊥`. Then
`m·(1 - ε) ≤ B·dim V` with `ε = τ×² / (τ×² + (t - τ⊥)²)`: however the gates are implemented and whatever their events,
only `V`'s directions can carry them. -/
theorem spiked_count (hμ : ∀ ρ, 0 ≤ μ ρ) (hsum : ∑ ρ, μ ρ = 1) (P S N : Ω → E →ₗ[ℝ] E)
    (V : Submodule ℝ E) (hPSN : ∀ ρ, P ρ = S ρ + N ρ) (hS : ∀ ρ x y, (x ∈ Vᗮ ∨ y ∈ Vᗮ) → ⟪S ρ x, y⟫ = 0)
    {σp σx : ℝ} (hσp : 0 ≤ σp) (hσx : 0 ≤ σx)
    (hNp : ∀ w ∈ Vᗮ, var μ (fun ρ => ⟪N ρ w, w⟫) ≤ σp ^ 2 * ‖w‖ ^ 4)
    (hNx : ∀ v ∈ V, ∀ w ∈ Vᗮ, var μ (fun ρ => ⟪N ρ v, w⟫) ≤ σx ^ 2 * (‖v‖ ^ 2 * ‖w‖ ^ 2))
    {ι : Type*} [Fintype ι] (e : ι → Ω → Prop) [∀ a, DecidablePred (e a)] {p : ℝ} (hpa : ∀ a, mass μ (e a) = p)
    (hp : 0 < p) (hp' : p < 1) {s : ι → E} (hs1 : ∀ a, ‖s a‖ = 1) {B : ℝ}
    (hB : ∀ x, ∑ a, ⟪s a, x⟫ ^ 2 ≤ B * ‖x‖ ^ 2) {t : ℝ} (ht : σp / √(p * (1 - p)) < t) (lam : ι → ℝ)
    (hlam : ∀ a, t ≤ lam a)
    (hs : ∀ a, (condMean μ (e a) P - condMean μ (fun ρ => ¬ e a ρ) P) (s a) = lam a • s a) :
    Fintype.card ι * (1 - (σx / √(p * (1 - p))) ^ 2 /
      ((σx / √(p * (1 - p))) ^ 2 + (t - σp / √(p * (1 - p))) ^ 2)) ≤ B * finrank ℝ V := by
  set τp := σp / √(p * (1 - p))
  set τx := σx / √(p * (1 - p))
  refine card_le_of_contained V hs1 hB fun a => ?_
  have h := spiked_containment μ hμ hsum P S N V hPSN hS hσp hσx hNp hNx (e a) ((hpa a).symm ▸ hp)
    ((hpa a).symm ▸ hp') (lam := lam a) (by rw [hpa a]; exact ht.trans_le (hlam a)) (hs a)
  rw [hpa a, hs1 a, one_pow, mul_one] at h
  have hd : 0 < τx ^ 2 + (t - τp) ^ 2 := by nlinarith [sq_nonneg τx, ht]
  have hmono : (t - τp) ^ 2 ≤ (lam a - τp) ^ 2 := by nlinarith [hlam a, ht]
  rw [le_div_iff₀ hd]
  nlinarith [h, hmono, sq_nonneg ‖s a - V.starProjection (s a)‖]

/-! ### Corollary 8.1: only directions with concentrated variation can gate -/

/-- `N_c`: the largest number of orthonormal directions `t` whose channel quantity `⟪P t, t⟫` has variance at least `v`.
At most `dim E`, since the directions are orthonormal. -/
def concentratedCount (P : Ω → E →ₗ[ℝ] E) (v : ℝ) : ℕ :=
  sSup {n | ∃ T : Finset E, T.card = n ∧ Orthonormal ℝ (fun t : T => (t : E)) ∧
    ∀ t ∈ T, v ≤ var μ (fun ρ => ⟪P ρ t, t⟫)}

/-- **Corollary 8.1.** Orthonormal gates `(eₐ, sₐ)` with `p(1-p) ≥ c` and `|E[qₐ | on] - E[qₐ | off]| ≥ δ`,
`qₐ = ⟪P sₐ, sₐ⟫`, opening or closing, with any events and any implementation, number at most
`N_{cδ²}`: each gate's own channel quantity must have variance `≥ c·δ²` (Theorem 5 (a)). -/
theorem card_le_concentratedCount (hμ : ∀ ρ, 0 ≤ μ ρ) (hsum : ∑ ρ, μ ρ = 1) (P : Ω → E →ₗ[ℝ] E)
    {ι : Type*} [Fintype ι] {s : ι → E} (hs : Orthonormal ℝ s) (e : ι → Ω → Prop) [∀ a, DecidablePred (e a)]
    (hp : ∀ a, 0 < mass μ (e a)) (hp' : ∀ a, mass μ (e a) < 1) {c δ : ℝ}
    (hc : ∀ a, c ≤ mass μ (e a) * (1 - mass μ (e a))) (hδ : 0 ≤ δ)
    (hδa : ∀ a, δ ≤ |condMean μ (e a) (fun ρ => ⟪P ρ (s a), s a⟫) -
      condMean μ (fun ρ => ¬ e a ρ) (fun ρ => ⟪P ρ (s a), s a⟫)|) :
    Fintype.card ι ≤ concentratedCount μ P (c * δ ^ 2) := by
  classical
  have hinj : Function.Injective s := hs.linearIndependent.injective
  set T := univ.image s
  have hcard : T.card = Fintype.card ι := by rw [card_image_of_injective _ hinj, card_univ]
  have hT : Orthonormal ℝ (fun t : T => (t : E)) := by
    rw [orthonormal_iff_ite]
    rintro ⟨x, hx⟩ ⟨y, hy⟩
    obtain ⟨a, -, rfl⟩ := mem_image.mp hx
    obtain ⟨b, -, rfl⟩ := mem_image.mp hy
    simp only [orthonormal_iff_ite.mp hs a b, Subtype.mk.injEq, hinj.eq_iff]
  have hvar : ∀ t ∈ T, c * δ ^ 2 ≤ var μ (fun ρ => ⟪P ρ t, t⟫) := by
    intro t ht
    obtain ⟨a, -, rfl⟩ := mem_image.mp ht
    have h := contrast_sq_le_var μ hμ hsum (e a) (fun ρ => ⟪P ρ (s a), s a⟫) (hp a) (hp' a)
    have hpq : 0 ≤ mass μ (e a) * (1 - mass μ (e a)) := mul_nonneg (hp a).le (by linarith [hp' a])
    have hsq : δ ^ 2 ≤ (condMean μ (e a) (fun ρ => ⟪P ρ (s a), s a⟫) -
        condMean μ (fun ρ => ¬ e a ρ) (fun ρ => ⟪P ρ (s a), s a⟫)) ^ 2 := by
      rw [← sq_abs (condMean _ _ _ - _)]
      exact pow_le_pow_left₀ hδ (hδa a) 2
    calc c * δ ^ 2 ≤ mass μ (e a) * (1 - mass μ (e a)) * δ ^ 2 := mul_le_mul_of_nonneg_right (hc a) (sq_nonneg δ)
      _ ≤ _ := mul_le_mul_of_nonneg_left hsq hpq
      _ ≤ _ := h
  have hbdd : BddAbove {n | ∃ T : Finset E, T.card = n ∧ Orthonormal ℝ (fun t : T => (t : E)) ∧
      ∀ t ∈ T, c * δ ^ 2 ≤ var μ (fun ρ => ⟪P ρ t, t⟫)} := by
    refine ⟨finrank ℝ E, fun n ⟨T', hn, hT', _⟩ => ?_⟩
    rw [← hn, ← Fintype.card_coe]
    exact hT'.linearIndependent.fintype_card_le_finrank
  rw [← hcard]
  exact le_csSup hbdd ⟨T, rfl, hT, hvar⟩

end GateBound

end
