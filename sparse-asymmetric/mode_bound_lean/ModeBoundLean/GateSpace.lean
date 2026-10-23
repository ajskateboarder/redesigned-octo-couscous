import Mathlib

/-!
# Gate spaces

`mode_bound.md` counts *gate modes*: eigenvalues `λ ≥ t` (with `t = γ/d`) of a symmetric operator `M`, the opener's
mean-difference matrix. Every bound in this formalization is proved for *gate spaces* instead: subspaces `V` on which
the quadratic form satisfies `⟪M x, x⟫ ≥ t‖x‖²`. The eigenvectors with eigenvalue `≥ t` span a gate space of exactly
that dimension (`exists_gateSpace_eigen`), so any bound on the dimension of gate spaces bounds the number of gate modes
(`card_eigen_le`). Modes that close (`λ ≤ -t`) are the gate modes of `-M`, i.e. `σ = -1` below.

`finrank_le_of_pos` is the dimension count behind the rank lemma, Theorems 3 and 3′, and the width bound: a subspace on
which a quadratic quantity is positive cannot meet the kernel of a map on whose kernel that quantity is `≤ 0`.
-/

open Module Submodule
open scoped InnerProductSpace RealInnerProductSpace

namespace ModeBound

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]

/-- `V` is a gate space of `T` at level `t`: the quadratic form of `T` is at least `t‖x‖²` on `V`. -/
def IsGateSpace (T : E →ₗ[ℝ] E) (t : ℝ) (V : Submodule ℝ E) : Prop :=
  ∀ x ∈ V, t * ‖x‖ ^ 2 ≤ ⟪T x, x⟫

/-- If `Q ≤ 0` wherever `L` vanishes, any subspace on which `Q` is positive (away from `0`) has dimension at most
`rank L`. -/
theorem finrank_le_of_pos {G : Type*} [AddCommGroup G] [Module ℝ G] (Q : E → ℝ) (L : E →ₗ[ℝ] G)
    (hL : ∀ x, L x = 0 → Q x ≤ 0) (V : Submodule ℝ E) (hV : ∀ x ∈ V, x ≠ 0 → 0 < Q x) :
    finrank ℝ V ≤ finrank ℝ (LinearMap.range L) := by
  have hinf : V ⊓ LinearMap.ker L = ⊥ := by
    rw [eq_bot_iff]
    intro x hx
    rw [mem_bot]
    by_contra h
    exact absurd (hV x (mem_inf.mp hx).1 h) (not_lt.mpr (hL x (LinearMap.mem_ker.mp (mem_inf.mp hx).2)))
  have h1 := finrank_sup_add_finrank_inf_eq V (LinearMap.ker L)
  have h2 := Submodule.finrank_le (V ⊔ LinearMap.ker L)
  have h3 := LinearMap.finrank_range_add_finrank_ker L
  rw [hinf, finrank_bot] at h1
  omega

/-- The eigenvectors of a symmetric `T` with `σ·λ ≥ t` span a gate space of `σ • T` at level `t` whose dimension is the
number of such eigenvalues (`σ = 1`: modes that open; `σ = -1`: modes that close). -/
theorem exists_gateSpace_eigen {T : E →ₗ[ℝ] E} (hT : T.IsSymmetric) {n : ℕ} (hn : finrank ℝ E = n) (σ t : ℝ) :
    ∃ V : Submodule ℝ E, IsGateSpace (σ • T) t V ∧
      finrank ℝ V = (Finset.univ.filter fun i => t ≤ σ * hT.eigenvalues hn i).card := by
  classical
  set b := hT.eigenvectorBasis hn
  set I := Finset.univ.filter fun i => t ≤ σ * hT.eigenvalues hn i
  let v : I → E := fun i => b i
  have hv : Orthonormal ℝ v := b.orthonormal.comp _ Subtype.val_injective
  refine ⟨span ℝ (Set.range v), ?_, ?_⟩
  · intro x hx
    obtain ⟨c, rfl⟩ := (mem_span_range_iff_exists_fun ℝ).mp hx
    have hTx : (σ • T) (∑ i, c i • v i) = ∑ i, (c i * (σ * hT.eigenvalues hn i)) • v i := by
      simp only [LinearMap.smul_apply, map_sum, map_smul, v, b, hT.apply_eigenvectorBasis, smul_smul,
        RCLike.ofReal_real_eq_id, id]
    rw [hTx, ← real_inner_self_eq_norm_sq, hv.inner_sum, hv.inner_sum, Finset.mul_sum]
    refine Finset.sum_le_sum fun i _ => ?_
    have hi : t ≤ σ * hT.eigenvalues hn i := (Finset.mem_filter.mp i.2).2
    simp only [conj_trivial]
    nlinarith [mul_self_nonneg (c i)]
  · rw [finrank_span_eq_card hv.linearIndependent, Fintype.card_coe]

/-- A bound on the dimension of every gate space of `σ • T` bounds the number of eigenvalues with `σ·λ ≥ t`. -/
theorem card_eigen_le {T : E →ₗ[ℝ] E} (hT : T.IsSymmetric) {n : ℕ} (hn : finrank ℝ E = n) (σ t : ℝ) {N : ℕ}
    (h : ∀ V, IsGateSpace (σ • T) t V → finrank ℝ V ≤ N) :
    (Finset.univ.filter fun i => t ≤ σ * hT.eigenvalues hn i).card ≤ N := by
  obtain ⟨V, hV, hd⟩ := exists_gateSpace_eigen hT hn σ t
  exact hd ▸ h V hV

/-- **Theorem 3′** (compression form). If the quadratic forms of `M` and `R` agree on `ker L`, every gate space `V` of
`M` has dimension at most `rank L + dim (V ⊓ ker L)`, and `V ⊓ ker L` is a gate space of `R`: every mode beyond
`rank L` is carried by `R` on `ker L`. -/
theorem gateSpace_compression {G : Type*} [AddCommGroup G] [Module ℝ G] (M R : E →ₗ[ℝ] E) (L : E →ₗ[ℝ] G)
    (hL : ∀ x, L x = 0 → ⟪M x, x⟫ = ⟪R x, x⟫) {t : ℝ} {V : Submodule ℝ E} (hV : IsGateSpace M t V) :
    IsGateSpace R t (V ⊓ LinearMap.ker L) ∧
      finrank ℝ V ≤ finrank ℝ (LinearMap.range L) + finrank ℝ ↥(V ⊓ LinearMap.ker L) := by
  refine ⟨fun x hx => ?_, ?_⟩
  · rw [← hL x (LinearMap.mem_ker.mp (mem_inf.mp hx).2)]
    exact hV x (mem_inf.mp hx).1
  · have h1 := finrank_sup_add_finrank_inf_eq V (LinearMap.ker L)
    have h2 := Submodule.finrank_le (V ⊔ LinearMap.ker L)
    have h3 := LinearMap.finrank_range_add_finrank_ker L
    omega

/-- **Theorem 3** (abstract). If moreover `⟪R x, x⟫ < t‖x‖²` for `x ≠ 0` (implied by `‖R‖ < t`), every gate space of `M`
at level `t` has dimension at most `rank L`. -/
theorem gateSpace_finrank_le {G : Type*} [AddCommGroup G] [Module ℝ G] (M R : E →ₗ[ℝ] E) (L : E →ₗ[ℝ] G)
    (hL : ∀ x, L x = 0 → ⟪M x, x⟫ = ⟪R x, x⟫) {t : ℝ} (hR : ∀ x, x ≠ 0 → ⟪R x, x⟫ < t * ‖x‖ ^ 2)
    {V : Submodule ℝ E} (hV : IsGateSpace M t V) : finrank ℝ V ≤ finrank ℝ (LinearMap.range L) := by
  obtain ⟨hR', hle⟩ := gateSpace_compression M R L hL hV
  have : V ⊓ LinearMap.ker L = ⊥ := by
    rw [eq_bot_iff]
    intro x hx
    rw [mem_bot]
    by_contra h
    exact absurd (hR' x hx) (not_le.mpr (hR x h))
  rw [this, finrank_bot] at hle
  omega

end ModeBound
