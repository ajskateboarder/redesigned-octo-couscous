import ModeBoundLean.GateSpace

/-!
# The rank lemma and the per-opener bound (Lemma 2, Theorems 3 and 3′)

Neurons are indexed by `S ⊕ R`: the opener's neuron set `S` and the rest `R`. Neuron `i` reads the whitened direction
`w i`. A row's fingerprint is `P_ρ = Σᵢⱼ (A_ρ)ᵢⱼ wᵢ wⱼᵀ` (`neuronOp w A_ρ`, with `A_ρ = F_ρᵀF_ρ / Z_ρ`), so an opener's
mean-difference matrix is `M = neuronOp w ΔA` with `ΔA = E[A | on] - E[A | off]`, by linearity (`neuronOp_sub`).

* `remainder w ΔA` is `E_S = W_R ΔA_RR W_Rᵀ`: `M` with every term that touches `S` deleted.
* `readS w` is `W_Sᵀ`, the read-outs of the neurons in `S`; `rank W_S ≤ |S|` (`finrank_range_readS_le`).
* **Lemma 2**: `M` and `E_S` have the same quadratic form on `ker W_Sᵀ` (`inner_neuronOp_eq_remainder`), so
  `K_S = M - E_S` has at most `rank W_S` positive and at most `rank W_S` negative directions (`rank_lemma`).
* **Theorem 3**: if `⟪E_S x, x⟫ < t‖x‖²` (resp. `-⟪E_S x, x⟫ < t‖x‖²`) for `x ≠ 0`, the opener has at most `|S|` gate modes
  that open (resp. close) at level `t` (`card_gateModes_le`).
* **Theorem 3′**: without that hypothesis, the modes beyond `rank W_S` form a gate space of `E_S` inside `ker W_Sᵀ`
  (`gateModes_compression`): they are carried by the neurons outside `S`.
-/

open Module Submodule
open scoped InnerProductSpace RealInnerProductSpace

set_option linter.unusedSectionVars false

noncomputable section

namespace ModeBound

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]

/-- `Σᵢⱼ Aᵢⱼ wᵢ wⱼᵀ`, as an operator on the whitened input space. -/
def neuronOp {ι : Type*} [Fintype ι] (w : ι → E) (A : Matrix ι ι ℝ) : E →ₗ[ℝ] E :=
  ∑ i, ∑ j, A i j • (innerₗ E (w j)).smulRight (w i)

section neuronOp

variable {ι : Type*} [Fintype ι] (w : ι → E)

theorem neuronOp_apply (A : Matrix ι ι ℝ) (x : E) :
    neuronOp w A x = ∑ i, ∑ j, (A i j * ⟪w j, x⟫) • w i := by
  simp [neuronOp, LinearMap.sum_apply, smul_smul]

theorem inner_neuronOp (A : Matrix ι ι ℝ) (x y : E) :
    ⟪neuronOp w A x, y⟫ = ∑ i, ∑ j, A i j * ⟪w j, x⟫ * ⟪w i, y⟫ := by
  simp [neuronOp_apply, sum_inner, inner_smul_left]

theorem neuronOp_sub (A B : Matrix ι ι ℝ) : neuronOp w (A - B) = neuronOp w A - neuronOp w B := by
  simp [neuronOp, sub_smul, Finset.sum_sub_distrib]

theorem neuronOp_neg (A : Matrix ι ι ℝ) : neuronOp w (-A) = -neuronOp w A := by
  simp [neuronOp, neg_smul, Finset.sum_neg_distrib]

/-- A symmetric neuron-space matrix gives a symmetric operator. -/
theorem neuronOp_isSymmetric {A : Matrix ι ι ℝ} (hA : A.IsSymm) : (neuronOp w A).IsSymmetric := by
  intro x y
  rw [real_inner_comm (neuronOp w A y) x, inner_neuronOp, inner_neuronOp]
  conv_rhs => rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
  rw [hA.apply i j]
  ring

end neuronOp

variable {S R : Type*} [Fintype S] [Fintype R] (w : S ⊕ R → E)

/-- `E_S = W_R ΔA_RR W_Rᵀ`: the opener's matrix recomputed from the neurons outside `S` alone. -/
def remainder (A : Matrix (S ⊕ R) (S ⊕ R) ℝ) : E →ₗ[ℝ] E :=
  neuronOp (w ∘ Sum.inr) (A.submatrix Sum.inr Sum.inr)

/-- `W_Sᵀ`: the read-outs `⟪w s, x⟫` of the neurons in `S`. -/
def readS : E →ₗ[ℝ] (S → ℝ) :=
  LinearMap.pi fun s => innerₗ E (w (Sum.inl s))

omit [Fintype R] in
/-- `rank W_S ≤ |S|`. -/
theorem finrank_range_readS_le : finrank ℝ (LinearMap.range (readS w)) ≤ Fintype.card S :=
  (Submodule.finrank_le _).trans (Module.finrank_fintype_fun_eq_card ℝ).le

omit [Fintype S] in
theorem remainder_neg (A : Matrix (S ⊕ R) (S ⊕ R) ℝ) : remainder w (-A) = -remainder w A := by
  simp [remainder, ← neuronOp_neg]
  rfl

/-- **Lemma 2 (i)**: on `ker W_Sᵀ`, the quadratic forms of `M` and `E_S` agree, i.e. `K_S = M - E_S` vanishes there. -/
theorem inner_neuronOp_eq_remainder (A : Matrix (S ⊕ R) (S ⊕ R) ℝ) (x : E) (hx : readS w x = 0) :
    ⟪neuronOp w A x, x⟫ = ⟪remainder w A x, x⟫ := by
  have h : ∀ s, ⟪w (Sum.inl s), x⟫ = 0 := fun s => congrFun hx s
  simp [inner_neuronOp, remainder, Fintype.sum_sum_type, h]

/-- **Lemma 2 (ii)**: `K_S = M - E_S` is positive on at most `rank W_S` dimensions and negative on at most `rank W_S`
dimensions: any subspace on which `±⟪K_S x, x⟫ > 0` (away from `0`) has dimension `≤ rank W_S ≤ |S|`. -/
theorem rank_lemma (A : Matrix (S ⊕ R) (S ⊕ R) ℝ) (σ : ℝ) (V : Submodule ℝ E)
    (hV : ∀ x ∈ V, x ≠ 0 → 0 < σ * ⟪(neuronOp w A - remainder w A) x, x⟫) :
    finrank ℝ V ≤ finrank ℝ (LinearMap.range (readS w)) := by
  refine finrank_le_of_pos _ (readS w) (fun x hx => ?_) V hV
  rw [LinearMap.sub_apply, inner_sub_left, inner_neuronOp_eq_remainder w A x hx, sub_self, mul_zero]

/-- **Theorem 3′**: every gate space of `M` at level `t` has dimension at most `rank W_S` plus the dimension of its
part inside `ker W_Sᵀ`, and that part is a gate space of `E_S`. -/
theorem gateModes_compression (A : Matrix (S ⊕ R) (S ⊕ R) ℝ) {t : ℝ} {V : Submodule ℝ E}
    (hV : IsGateSpace (neuronOp w A) t V) :
    IsGateSpace (remainder w A) t (V ⊓ LinearMap.ker (readS w)) ∧
      finrank ℝ V ≤ Fintype.card S + finrank ℝ ↥(V ⊓ LinearMap.ker (readS w)) := by
  obtain ⟨h1, h2⟩ := gateSpace_compression _ _ (readS w) (inner_neuronOp_eq_remainder w A) hV
  exact ⟨h1, h2.trans (Nat.add_le_add_right (finrank_range_readS_le w) _)⟩

/-- **Theorem 3** (gate spaces): if `⟪E_S x, x⟫ < t‖x‖²` for all `x ≠ 0`, every gate space of `M` at level `t` has
dimension at most `|S|`. -/
theorem gateModes_le (A : Matrix (S ⊕ R) (S ⊕ R) ℝ) {t : ℝ}
    (hE : ∀ x, x ≠ 0 → ⟪remainder w A x, x⟫ < t * ‖x‖ ^ 2) {V : Submodule ℝ E}
    (hV : IsGateSpace (neuronOp w A) t V) : finrank ℝ V ≤ Fintype.card S :=
  (gateSpace_finrank_le _ _ (readS w) (inner_neuronOp_eq_remainder w A) hE hV).trans (finrank_range_readS_le w)

/-- **Theorem 3** (eigenvalues): for a symmetric `ΔA`, if `±⟪E_S x, x⟫ < t‖x‖²` for all `x ≠ 0`, the opener's matrix
`M = neuronOp w ΔA` has at most `|S|` eigenvalues `≥ t` and at most `|S|` eigenvalues `≤ -t`, whatever the number of
neurons. -/
theorem card_gateModes_le (A : Matrix (S ⊕ R) (S ⊕ R) ℝ) (hA : A.IsSymm) {n : ℕ} (hn : finrank ℝ E = n) {t : ℝ}
    (hEpos : ∀ x, x ≠ 0 → ⟪remainder w A x, x⟫ < t * ‖x‖ ^ 2)
    (hEneg : ∀ x, x ≠ 0 → -⟪remainder w A x, x⟫ < t * ‖x‖ ^ 2) :
    (Finset.univ.filter fun i => t ≤ (neuronOp_isSymmetric w hA).eigenvalues hn i).card ≤ Fintype.card S ∧
      (Finset.univ.filter fun i => t ≤ -(neuronOp_isSymmetric w hA).eigenvalues hn i).card ≤ Fintype.card S := by
  constructor
  · have := card_eigen_le (neuronOp_isSymmetric w hA) hn 1 t (N := Fintype.card S)
      (fun V hV => gateModes_le w A hEpos (by simpa using hV))
    simpa using this
  · have := card_eigen_le (neuronOp_isSymmetric w hA) hn (-1) t (N := Fintype.card S) fun V hV =>
      gateModes_le w (-A) (fun x hx => by simpa [remainder_neg] using hEneg x hx)
        (by simpa [neuronOp_neg] using hV)
    simpa using this

end ModeBound

end
