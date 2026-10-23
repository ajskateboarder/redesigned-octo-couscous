import ModeBoundLean.Budget
import ModeBoundLean.Variance

/-!
# The gate bound for fingerprints (Theorem 5 (c) with `κ ≤ 1`)

`Variance.lean` proves Theorem 5 (c) in an abstract inner product space of fingerprints. This file states it for the
actual fingerprints `P_ρ = GᵀG / ‖G‖²_F` of `Budget.lean`, with the Frobenius (Hilbert–Schmidt) norm
`‖D‖²_F = tr(D²)` (`hsSq`), and closes it with `κ ≤ 1`:

* `hsSq_fingerprint_le_one`: every fingerprint has `‖P‖²_F ≤ 1` (Cauchy–Schwarz and Parseval; no spectral theorem).
* `kappa_le_one`: the total variation of fingerprints across rows, `κ = E‖P - P̄‖²_F`, is at most `1`.
* `sum_sq_inner_le_hsSq`: orthonormal channels `sₐ` have frame bound `1`: `Σₐ (sₐᵀ D sₐ)² ≤ ‖D‖²_F` for symmetric `D`.
* `gate_count_fingerprint`: gates `(eₐ, sₐ)` with `p(1-p) ≥ c` and `|E[qₐ | on] - E[qₐ | off]| ≥ δ`, `qₐ = sₐᵀPsₐ`,
  satisfy `m · c · δ² ≤ B·κ ≤ B`, where `B` is the channels' frame bound in the Frobenius sense.
* `gate_count_orthonormal`: with orthonormal channels, `m · c · δ² ≤ 1`. With contrast `γ = d·δ` this is
  `m ≤ d² / (p(1-p)·γ²)` for any model, any data and any number of neurons.
-/

open Module Finset
open scoped InnerProductSpace RealInnerProductSpace

set_option linter.unusedSectionVars false

noncomputable section

namespace ModeBound

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]

/-- The squared Frobenius (Hilbert–Schmidt) norm `‖D‖²_F = tr(D²)` of a symmetric operator. -/
def hsSq (D : E →ₗ[ℝ] E) : ℝ := LinearMap.trace ℝ E (D * D)

theorem hsSq_eq_sum {D : E →ₗ[ℝ] E} (hD : D.IsSymmetric) {ι : Type*} [Fintype ι]
    (b : OrthonormalBasis ι ℝ E) : hsSq D = ∑ i, ‖D (b i)‖ ^ 2 := by
  rw [hsSq, LinearMap.trace_eq_sum_inner _ b]
  refine sum_congr rfl fun i _ => ?_
  rw [Module.End.mul_apply, ← hD (b i) (D (b i)), real_inner_self_eq_norm_sq]

theorem hsSq_nonneg {D : E →ₗ[ℝ] E} (hD : D.IsSymmetric) : 0 ≤ hsSq D := by
  rw [hsSq_eq_sum hD (stdOrthonormalBasis ℝ E)]
  positivity

/-- Orthonormal channels have frame bound `1` for the Frobenius norm: `Σₐ (sₐᵀ D sₐ)² ≤ ‖D‖²_F`. -/
theorem sum_sq_inner_le_hsSq {D : E →ₗ[ℝ] E} (hD : D.IsSymmetric) {ι : Type*} [Fintype ι] {s : ι → E}
    (hs : Orthonormal ℝ s) : ∑ a, ⟪D (s a), s a⟫ ^ 2 ≤ hsSq D := by
  have hpos : (D * D).IsPositive := by
    rw [LinearMap.isPositive_iff]
    refine ⟨fun x y => ?_, fun x => ?_⟩
    · rw [Module.End.mul_apply, Module.End.mul_apply, hD, hD]
    · rw [Module.End.mul_apply, hD]
      exact real_inner_self_nonneg
  have hsq : ∀ a, ⟪(D * D) (s a), s a⟫ = ‖D (s a)‖ ^ 2 := fun a => by
    rw [Module.End.mul_apply, hD, real_inner_self_eq_norm_sq]
  calc ∑ a, ⟪D (s a), s a⟫ ^ 2 ≤ ∑ a, ⟪(D * D) (s a), s a⟫ := sum_le_sum fun a _ => by
        have h := abs_real_inner_le_norm (D (s a)) (s a)
        rw [hs.1 a, mul_one] at h
        rw [hsq, ← sq_abs]
        exact pow_le_pow_left₀ (abs_nonneg _) h 2
    _ ≤ hsSq D := sum_inner_le_trace hpos hs

/-- Every fingerprint has `‖P‖²_F ≤ 1`. -/
theorem hsSq_fingerprint_le_one {K : Type*} [Fintype K] (g : K → E) : hsSq (fingerprint g) ≤ 1 := by
  set b := stdOrthonormalBasis ℝ E
  rw [hsSq_eq_sum (fingerprint_isPositive g).isSymmetric b]
  set Z := ∑ k, ‖g k‖ ^ 2 with hZdef
  have hZ0 : 0 ≤ Z := sum_nonneg fun _ _ => sq_nonneg _
  have happ : ∀ x, fingerprint g x = Z⁻¹ • ∑ k, ⟪g k, x⟫ • g k := fun x => by
    simp [fingerprint, LinearMap.sum_apply, hZdef]
  -- `‖Px‖² ≤ Z⁻¹ Σₖ ⟪gₖ, x⟫²` by Cauchy–Schwarz on `Px = Z⁻¹ Σₖ ⟪gₖ, x⟫ gₖ`
  have hbound : ∀ x, ‖fingerprint g x‖ ^ 2 ≤ Z⁻¹ * ∑ k, ⟪g k, x⟫ ^ 2 := by
    intro x
    rw [happ, norm_smul, mul_pow, Real.norm_eq_abs, sq_abs]
    have h1 : ‖∑ k, ⟪g k, x⟫ • g k‖ ≤ ∑ k, |⟪g k, x⟫| * ‖g k‖ :=
      (norm_sum_le _ _).trans (le_of_eq (sum_congr rfl fun k _ => by rw [norm_smul, Real.norm_eq_abs]))
    have h2 := sum_mul_sq_le_sq_mul_sq univ (fun k => |⟪g k, x⟫|) (fun k => ‖g k‖)
    simp only [sq_abs] at h2
    have h3 : ‖∑ k, ⟪g k, x⟫ • g k‖ ^ 2 ≤ (∑ k, ⟪g k, x⟫ ^ 2) * Z :=
      (pow_le_pow_left₀ (norm_nonneg _) h1 2).trans h2
    rcases hZ0.eq_or_lt with h | h
    · rw [← h]
      simp
    · calc Z⁻¹ ^ 2 * ‖∑ k, ⟪g k, x⟫ • g k‖ ^ 2 ≤ Z⁻¹ ^ 2 * ((∑ k, ⟪g k, x⟫ ^ 2) * Z) :=
            mul_le_mul_of_nonneg_left h3 (by positivity)
        _ = Z⁻¹ * ∑ k, ⟪g k, x⟫ ^ 2 := by field_simp
  -- Parseval: `Σᵢ ⟪gₖ, bᵢ⟫² = ‖gₖ‖²`
  calc ∑ i, ‖fingerprint g (b i)‖ ^ 2 ≤ ∑ i, Z⁻¹ * ∑ k, ⟪g k, b i⟫ ^ 2 := sum_le_sum fun i _ => hbound (b i)
    _ = Z⁻¹ * Z := by
        rw [← mul_sum, sum_comm, hZdef]
        exact congrArg _ (sum_congr rfl fun k _ => b.sum_sq_inner_left (g k))
    _ ≤ 1 := by
        rcases hZ0.eq_or_lt with h | h
        · simp [← h]
        · rw [inv_mul_cancel₀ h.ne']

/-! ### The variation `κ` of fingerprints across rows -/

variable {Ω : Type*} [Fintype Ω] (μ : Ω → ℝ) {K : Type*} [Fintype K] (g : Ω → K → E)

/-- The mean fingerprint `P̄ = E[P]`. -/
def meanFP : E →ₗ[ℝ] E := ∑ ρ, μ ρ • fingerprint (g ρ)

/-- `κ = E‖P - P̄‖²_F`: how much rows differ in where their output is sensitive. -/
def kappa : ℝ := ∑ ρ, μ ρ * hsSq (fingerprint (g ρ) - meanFP μ g)

theorem meanFP_isSymmetric : (meanFP μ g).IsSymmetric := by
  intro x y
  simp only [meanFP, LinearMap.sum_apply, LinearMap.smul_apply, sum_inner, inner_sum, real_inner_smul_left,
    real_inner_smul_right, (fingerprint_isPositive _).isSymmetric _ _]

/-- **`κ ≤ 1`**: fingerprints vary across rows by at most `1` in total. -/
theorem kappa_le_one (hμ : ∀ ρ, 0 ≤ μ ρ) (hsum : ∑ ρ, μ ρ = 1) : kappa μ g ≤ 1 := by
  set P := fun ρ => fingerprint (g ρ)
  set Pbar := meanFP μ g
  have hexp : ∀ ρ, hsSq (P ρ - Pbar) =
      hsSq (P ρ) - LinearMap.trace ℝ E (P ρ * Pbar) - LinearMap.trace ℝ E (Pbar * P ρ) + hsSq Pbar := by
    intro ρ
    simp only [hsSq, mul_sub, sub_mul, map_sub]
    ring
  have hleft : ∑ ρ, μ ρ * LinearMap.trace ℝ E (P ρ * Pbar) = hsSq Pbar := by
    have h : ∑ ρ, μ ρ • (P ρ * Pbar) = Pbar * Pbar := by
      rw [show Pbar * Pbar = (∑ ρ, μ ρ • P ρ) * Pbar from rfl, Finset.sum_mul]
      simp_rw [smul_mul_assoc]
    rw [hsSq, ← h, map_sum]
    simp [map_smul, smul_eq_mul]
  have hright : ∑ ρ, μ ρ * LinearMap.trace ℝ E (Pbar * P ρ) = hsSq Pbar := by
    have h : ∑ ρ, μ ρ • (Pbar * P ρ) = Pbar * Pbar := by
      rw [show Pbar * Pbar = Pbar * ∑ ρ, μ ρ • P ρ from rfl, Finset.mul_sum]
      simp_rw [mul_smul_comm]
    rw [hsSq, ← h, map_sum]
    simp [map_smul, smul_eq_mul]
  have hκ : kappa μ g = ∑ ρ, μ ρ * hsSq (P ρ) - hsSq Pbar := by
    change ∑ ρ, μ ρ * hsSq (P ρ - Pbar) = _
    simp only [hexp, mul_add, mul_sub, sum_add_distrib, sum_sub_distrib, hleft, hright, ← sum_mul, hsum]
    ring
  rw [hκ]
  have h0 := hsSq_nonneg (meanFP_isSymmetric μ g)
  have h1 : ∑ ρ, μ ρ * hsSq (P ρ) ≤ ∑ ρ, μ ρ * 1 :=
    sum_le_sum fun ρ _ => mul_le_mul_of_nonneg_left (hsSq_fingerprint_le_one (g ρ)) (hμ ρ)
  rw [← sum_mul, hsum, one_mul] at h1
  linarith

/-! ### Gate bound -/

/-- **Theorem 5 (c)** for fingerprints: channel quantities `qₐ = sₐᵀPsₐ`, with channels of Frobenius frame bound `B`,
have `Σₐ Var qₐ ≤ B·κ`. -/
theorem sum_var_fingerprint_le (hμ : ∀ ρ, 0 ≤ μ ρ) {ι : Type*} [Fintype ι] (s : ι → E) {B : ℝ}
    (hB : ∀ D : E →ₗ[ℝ] E, D.IsSymmetric → ∑ a, ⟪D (s a), s a⟫ ^ 2 ≤ B * hsSq D) :
    ∑ a, var μ (fun ρ => ⟪fingerprint (g ρ) (s a), s a⟫) ≤ B * kappa μ g := by
  have hmean : ∀ a, mean μ (fun ρ => ⟪fingerprint (g ρ) (s a), s a⟫) = ⟪meanFP μ g (s a), s a⟫ := fun a => by
    simp [mean, meanFP, LinearMap.sum_apply, sum_inner, real_inner_smul_left]
  have hdiff : ∀ a ρ, ⟪fingerprint (g ρ) (s a), s a⟫ - ⟪meanFP μ g (s a), s a⟫ =
      ⟪(fingerprint (g ρ) - meanFP μ g) (s a), s a⟫ := fun a ρ => by
    rw [LinearMap.sub_apply, inner_sub_left]
  simp only [var, hmean, hdiff]
  rw [sum_comm, kappa, mul_sum]
  refine sum_le_sum fun ρ _ => ?_
  rw [← mul_sum, mul_left_comm]
  exact mul_le_mul_of_nonneg_left
    (hB _ ((fingerprint_isPositive (g ρ)).isSymmetric.sub (meanFP_isSymmetric μ g))) (hμ ρ)

/-- **The gate bound**: gates `(eₐ, sₐ)` on fingerprints, with `p(1-p) ≥ c` and `|E[qₐ | on] - E[qₐ | off]| ≥ δ`
(contrast `γ = d·δ`), on channels of Frobenius frame bound `B`, satisfy `m · c · δ² ≤ B·κ`, and `κ ≤ 1`. -/
theorem gate_count_fingerprint (hμ : ∀ ρ, 0 ≤ μ ρ) (hsum : ∑ ρ, μ ρ = 1) {ι : Type*} [Fintype ι] (s : ι → E)
    {B : ℝ} (hB : ∀ D : E →ₗ[ℝ] E, D.IsSymmetric → ∑ a, ⟪D (s a), s a⟫ ^ 2 ≤ B * hsSq D)
    (e : ι → Ω → Prop) [∀ a, DecidablePred (e a)] (hp : ∀ a, 0 < mass μ (e a)) (hp' : ∀ a, mass μ (e a) < 1)
    {c δ : ℝ} (hc : ∀ a, c ≤ mass μ (e a) * (1 - mass μ (e a))) (hδ : 0 ≤ δ)
    (hδa : ∀ a, δ ≤ |condMean μ (e a) (fun ρ => ⟪fingerprint (g ρ) (s a), s a⟫) -
      condMean μ (fun ρ => ¬ e a ρ) (fun ρ => ⟪fingerprint (g ρ) (s a), s a⟫)|) :
    Fintype.card ι * (c * δ ^ 2) ≤ B * kappa μ g ∧ kappa μ g ≤ 1 := by
  refine ⟨?_, kappa_le_one μ g hμ hsum⟩
  calc (Fintype.card ι : ℝ) * (c * δ ^ 2) = ∑ _a : ι, c * δ ^ 2 := by simp
    _ ≤ ∑ a, var μ (fun ρ => ⟪fingerprint (g ρ) (s a), s a⟫) := sum_le_sum fun a _ => by
        have h := contrast_sq_le_var μ hμ hsum (e a) (fun ρ => ⟪fingerprint (g ρ) (s a), s a⟫) (hp a) (hp' a)
        have hpq : 0 ≤ mass μ (e a) * (1 - mass μ (e a)) := mul_nonneg (hp a).le (by linarith [hp' a])
        have hsq : δ ^ 2 ≤ (condMean μ (e a) (fun ρ => ⟪fingerprint (g ρ) (s a), s a⟫) -
            condMean μ (fun ρ => ¬ e a ρ) (fun ρ => ⟪fingerprint (g ρ) (s a), s a⟫)) ^ 2 := by
          rw [← sq_abs (condMean _ _ _ - _)]
          exact pow_le_pow_left₀ hδ (hδa a) 2
        calc c * δ ^ 2 ≤ mass μ (e a) * (1 - mass μ (e a)) * δ ^ 2 :=
              mul_le_mul_of_nonneg_right (hc a) (sq_nonneg δ)
          _ ≤ _ := mul_le_mul_of_nonneg_left hsq hpq
          _ ≤ _ := h
    _ ≤ B * kappa μ g := sum_var_fingerprint_le μ g hμ s hB

/-- **The gate bound, orthonormal channels**: `m · c · δ² ≤ 1`, i.e. with contrast `γ = d·δ` and `p(1-p) ≥ c`, at most
`d² / (c·γ²)` gates, for any model, any data and any number of neurons. -/
theorem gate_count_orthonormal (hμ : ∀ ρ, 0 ≤ μ ρ) (hsum : ∑ ρ, μ ρ = 1) {ι : Type*} [Fintype ι] {s : ι → E}
    (hs : Orthonormal ℝ s) (e : ι → Ω → Prop) [∀ a, DecidablePred (e a)] (hp : ∀ a, 0 < mass μ (e a))
    (hp' : ∀ a, mass μ (e a) < 1) {c δ : ℝ} (hc : ∀ a, c ≤ mass μ (e a) * (1 - mass μ (e a))) (hδ : 0 ≤ δ)
    (hδa : ∀ a, δ ≤ |condMean μ (e a) (fun ρ => ⟪fingerprint (g ρ) (s a), s a⟫) -
      condMean μ (fun ρ => ¬ e a ρ) (fun ρ => ⟪fingerprint (g ρ) (s a), s a⟫)|) :
    Fintype.card ι * (c * δ ^ 2) ≤ 1 := by
  obtain ⟨h1, h2⟩ := gate_count_fingerprint μ g hμ hsum s (B := 1)
    (fun D hD => by simpa using sum_sq_inner_le_hsSq hD hs) e hp hp' hc hδ hδa
  linarith

end ModeBound

end
