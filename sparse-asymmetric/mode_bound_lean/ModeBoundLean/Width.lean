import ModeBoundLean.GateSpace

/-!
# Width (Proposition 6, Example 9 of the write-up)

A width-`H` MLP's curvature is a combination of the `H` rank-one terms `wᵢwᵢᵀ`. Counting signs gives a lower bound on
width:

* `inertia_le`: a quadratic form `Σᵢ cᵢ⟪wᵢ, x⟫²` is positive definite on at most `#{cᵢ > 0}` dimensions and negative
  definite on at most `#{cᵢ < 0}`; together at most `H`.
* `two_mul_le_width`: a width-`H` network with square activations that computes `m` bilinear gates
  `x ↦ Σₐ ⟪oₐ, x⟫⟪sₐ, x⟫` (opener `oₐ` scales the sensitivity along channel `sₐ`), with `o, s` orthonormal, has
  `H ≥ 2m`.
* `sqNet_bilinearGates`: `2m` neurons suffice, so the minimal width is exactly `2m`: two neurons per gate, the rate the
  planted-gate experiments reach for `d ≥ 128`.
-/

open Module Submodule Finset
open scoped InnerProductSpace RealInnerProductSpace

set_option linter.unusedSectionVars false

noncomputable section

namespace ModeBound

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]

/-- **Inertia count**: `Q(x) = Σᵢ cᵢ⟪wᵢ, x⟫²` from `H` neurons is positive definite on at most `#{cᵢ > 0}` dimensions
and negative definite on at most `#{cᵢ < 0}`, so the two dimensions add up to at most `H`. -/
theorem inertia_le {ι : Type*} [Fintype ι] (c : ι → ℝ) (w : ι → E) (Vpos Vneg : Submodule ℝ E)
    (hpos : ∀ x ∈ Vpos, x ≠ 0 → 0 < ∑ i, c i * ⟪w i, x⟫ ^ 2)
    (hneg : ∀ x ∈ Vneg, x ≠ 0 → ∑ i, c i * ⟪w i, x⟫ ^ 2 < 0) :
    finrank ℝ Vpos + finrank ℝ Vneg ≤ Fintype.card ι := by
  classical
  let Lp : E →ₗ[ℝ] ({i // 0 < c i} → ℝ) := LinearMap.pi fun i => innerₗ E (w i)
  let Ln : E →ₗ[ℝ] ({i // c i < 0} → ℝ) := LinearMap.pi fun i => innerₗ E (w i)
  have h1 : finrank ℝ Vpos ≤ Fintype.card {i // 0 < c i} := by
    refine (finrank_le_of_pos (fun x => ∑ i, c i * ⟪w i, x⟫ ^ 2) Lp (fun x hx => ?_) Vpos hpos).trans
      ((Submodule.finrank_le _).trans (Module.finrank_fintype_fun_eq_card ℝ).le)
    refine sum_nonpos fun i _ => ?_
    by_cases hi : 0 < c i
    · have h0 : ⟪w i, x⟫ = 0 := congrFun hx ⟨i, hi⟩
      simp [h0]
    · nlinarith [sq_nonneg ⟪w i, x⟫, not_lt.mp hi]
  have h2 : finrank ℝ Vneg ≤ Fintype.card {i // c i < 0} := by
    refine (finrank_le_of_pos (fun x => -∑ i, c i * ⟪w i, x⟫ ^ 2) Ln (fun x hx => ?_) Vneg
      (fun x hx h0 => neg_pos.mpr (hneg x hx h0))).trans
      ((Submodule.finrank_le _).trans (Module.finrank_fintype_fun_eq_card ℝ).le)
    rw [neg_nonpos]
    refine sum_nonneg fun i _ => ?_
    by_cases hi : c i < 0
    · have h0 : ⟪w i, x⟫ = 0 := congrFun hx ⟨i, hi⟩
      simp [h0]
    · nlinarith [sq_nonneg ⟪w i, x⟫, not_lt.mp hi]
  have h3 : Fintype.card {i // 0 < c i} + Fintype.card {i // c i < 0} ≤ Fintype.card ι := by
    rw [Fintype.card_subtype, Fintype.card_subtype, ← card_union_of_disjoint
      (disjoint_filter.mpr fun i _ h h' => lt_asymm h h')]
    exact card_le_univ _
  omega

/-- A width-`H` network with square activations: `f x = Σᵢ cᵢ(⟪wᵢ, x⟫ + bᵢ)² + ⟪β, x⟫ + β₀`. -/
def sqNet {ι : Type*} [Fintype ι] (c : ι → ℝ) (w : ι → E) (b : ι → ℝ) (β : E) (β₀ : ℝ) (x : E) : ℝ :=
  ∑ i, c i * (⟪w i, x⟫ + b i) ^ 2 + ⟪β, x⟫ + β₀

/-- `m` bilinear gates: opener `o a` scales the sensitivity along channel `s a`. -/
def bilinearGates {m : ℕ} (o s : Fin m → E) (x : E) : ℝ := ∑ a, ⟪o a, x⟫ * ⟪s a, x⟫

/-- The even part of a square-activation network is its curvature form `Σᵢ cᵢ⟪wᵢ, x⟫²`. -/
theorem sqNet_even {ι : Type*} [Fintype ι] (c : ι → ℝ) (w : ι → E) (b : ι → ℝ) (β : E) (β₀ : ℝ) (x : E) :
    (sqNet c w b β β₀ x + sqNet c w b β β₀ (-x) - 2 * sqNet c w b β β₀ 0) / 2 = ∑ i, c i * ⟪w i, x⟫ ^ 2 := by
  have h : ∑ i, c i * (⟪w i, x⟫ + b i) ^ 2 + ∑ i, c i * (-⟪w i, x⟫ + b i) ^ 2 - 2 * ∑ i, c i * b i ^ 2 =
      2 * ∑ i, c i * ⟪w i, x⟫ ^ 2 := by
    rw [← sum_add_distrib, mul_sum, mul_sum, ← sum_sub_distrib]
    exact sum_congr rfl fun i _ => by ring
  simp only [sqNet, inner_neg_right, inner_zero_right, zero_add]
  linarith

theorem bilinearGates_even {m : ℕ} (o s : Fin m → E) (x : E) :
    (bilinearGates o s x + bilinearGates o s (-x) - 2 * bilinearGates o s 0) / 2 = bilinearGates o s x := by
  simp [bilinearGates]

section orthonormal

variable {m : ℕ} {e : Fin m ⊕ Fin m → E} (he : Orthonormal ℝ e)
include he

/-- Read-outs of `x = Σₐ kₐ(oₐ + σ sₐ)` along an opener: `⟪o_b, x⟫ = k_b`. -/
theorem inner_opener (σ : ℝ) (k : Fin m → ℝ) (b : Fin m) :
    ⟪e (Sum.inl b), ∑ a, k a • (e (Sum.inl a) + σ • e (Sum.inr a))⟫ = k b := by
  simp [inner_sum, inner_smul_right, inner_add_right, orthonormal_iff_ite.mp he]

/-- ... and along a channel: `⟪s_b, x⟫ = σ k_b`. -/
theorem inner_channel (σ : ℝ) (k : Fin m → ℝ) (b : Fin m) :
    ⟪e (Sum.inr b), ∑ a, k a • (e (Sum.inl a) + σ • e (Sum.inr a))⟫ = σ * k b := by
  simp [inner_sum, inner_smul_right, inner_add_right, orthonormal_iff_ite.mp he, mul_comm]

/-- The directions `oₐ + σ sₐ` (`σ ≠ 0`) span an `m`-dimensional subspace on which the gates' form is `σ·Σₐ kₐ²`. -/
theorem gates_on_span {σ : ℝ} (hσ : σ ≠ 0) :
    finrank ℝ (span ℝ (Set.range fun a => e (Sum.inl a) + σ • e (Sum.inr a))) = m ∧
      ∀ x ∈ span ℝ (Set.range fun a => e (Sum.inl a) + σ • e (Sum.inr a)), x ≠ 0 →
        0 < σ * bilinearGates (e ∘ Sum.inl) (e ∘ Sum.inr) x := by
  have hli : LinearIndependent ℝ fun a => e (Sum.inl a) + σ • e (Sum.inr a) := by
    rw [Fintype.linearIndependent_iff]
    intro k hk b
    have h := inner_opener he σ k b
    rw [hk, inner_zero_right] at h
    exact h.symm
  refine ⟨by rw [finrank_span_eq_card hli, Fintype.card_fin], fun x hx hx0 => ?_⟩
  obtain ⟨k, rfl⟩ := (mem_span_range_iff_exists_fun ℝ).mp hx
  obtain ⟨b, hb⟩ : ∃ b, k b ≠ 0 := by
    by_contra h
    push Not at h
    exact hx0 (by simp [h])
  have hform : bilinearGates (e ∘ Sum.inl) (e ∘ Sum.inr) (∑ a, k a • (e (Sum.inl a) + σ • e (Sum.inr a))) =
      σ * ∑ a, k a ^ 2 := by
    simp only [bilinearGates, Function.comp_apply, inner_opener he, inner_channel he, mul_sum]
    exact sum_congr rfl fun a _ => by ring
  rw [hform, ← mul_assoc]
  exact mul_pos (mul_self_pos.mpr hσ)
    (sum_pos' (fun a _ => sq_nonneg _) ⟨b, mem_univ _, lt_of_le_of_ne (sq_nonneg _) (pow_ne_zero 2 hb).symm⟩)

/-- **Proposition 6** for bilinear gates: a width-`H` square-activation network that computes `m` bilinear gates with
orthonormal openers and channels has `H ≥ 2m`. -/
theorem two_mul_le_width {ι : Type*} [Fintype ι] (c : ι → ℝ) (w : ι → E) (b : ι → ℝ) (β : E) (β₀ : ℝ)
    (hf : ∀ x, sqNet c w b β β₀ x = bilinearGates (e ∘ Sum.inl) (e ∘ Sum.inr) x) : 2 * m ≤ Fintype.card ι := by
  have hQ : ∀ x, ∑ i, c i * ⟪w i, x⟫ ^ 2 = bilinearGates (e ∘ Sum.inl) (e ∘ Sum.inr) x := fun x => by
    rw [← sqNet_even, ← bilinearGates_even, hf, hf, hf]
  obtain ⟨d1, h1⟩ := gates_on_span he (σ := 1) one_ne_zero
  obtain ⟨d2, h2⟩ := gates_on_span he (σ := -1) (by norm_num)
  have h := inertia_le c w _ _ (fun x hx hx0 => by simpa [hQ] using h1 x hx hx0)
    (fun x hx hx0 => by have := h2 x hx hx0; rw [hQ]; linarith)
  rw [d1, d2] at h
  omega

end orthonormal

/-- `2m` square-activation neurons suffice for `m` bilinear gates: `⟪o, x⟫⟪s, x⟫ = ¼⟪o + s, x⟫² - ¼⟪o - s, x⟫²`.
With `two_mul_le_width`, the minimal width is exactly `2m`. -/
theorem sqNet_bilinearGates {m : ℕ} (o s : Fin m → E) :
    ∃ (c : Fin m ⊕ Fin m → ℝ) (w : Fin m ⊕ Fin m → E), ∀ x, sqNet c w 0 0 0 x = bilinearGates o s x := by
  refine ⟨Sum.elim (fun _ => 1 / 4) (fun _ => -(1 / 4)), Sum.elim (fun a => o a + s a) (fun a => o a - s a),
    fun x => ?_⟩
  simp only [sqNet, bilinearGates, Fintype.sum_sum_type, Sum.elim_inl, Sum.elim_inr, inner_add_left, inner_sub_left,
    Pi.zero_apply, add_zero, inner_zero_left]
  rw [← sum_add_distrib]
  exact sum_congr rfl fun a _ => by ring

end ModeBound

end
