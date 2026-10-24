import GateBoundLean.GateBound

/-!
# Sharper gate budgets (Theorems 7′ and 7*), and the sharpness of Theorem 7

Theorem 7 charges a gate `(E, s)` of prevalence `p` the amount `p · Δ`, `Δ = E[q | on] - E[q | off]`, `q = sᵀPs`, out of
the channel's mean `E[q]`. This file sharpens that accounting in three ways and shows that the count it gives cannot be
improved without information about the fingerprints.

* `mass_mul_contrast_eq`: `p · Δ = E[q] - E[q | off]` exactly, so Theorem 7 drops precisely the channel's off-mean.
* **Theorem 7′** (`linear_budget_on`, `gate_count_on`): `Σₐ pₐΔₐ ≤ E[Σ_{a open at ρ} qₐ(ρ)] ≤ B_on`, where `B_on` is a frame
  bound for the channels *open at the same row*, not for all channels. With orthonormal co-open channels, `m` gates of
  prevalence `≥ p_min` and contrast `≥ γ` number at most `d / (p_min γ)` however many channels the dictionary holds.
* **Theorem 7*** (`contrast_le_split`, `quantile_budget`, `gate_count_quantile`): for any symmetric `Y₀` and any split
  `P_ρ - Y₀ = Y⁺_ρ - Y⁻_ρ` into positive parts, `m · p(1-p) · δ ≤ B · E[(1-p) tr Y⁺ + p tr Y⁻]`. `Y₀ = 0`, `Y⁺ = P`,
  `Y⁻ = 0` is Theorem 7; a `Y₀` near the rows' typical fingerprint removes the part of each channel's mean that every row
  has, which Theorem 7 counts against the budget.
* **Two-sided budget** (`linear_budget_gate_abs`, `gate_count_abs`) and **Theorem 5 as a corollary**
  (`gate_count_orthonormal_linear`): closing gates pay `(1-p)|Δ|`, and orthonormal gates with `p(1-p) ≥ c` satisfy
  `m · c · δ ≤ 1`. Since `δ ≤ 1` (`contrast_abs_le_one`), this is stronger than Theorem 5's unconditional
  `m · c · δ² ≤ 1` (`gate_count_orthonormal`) by the factor `1/δ = d/γ`.
* **Sharpness** (`linear_budget_sharp`): `r` rows, each spreading its sensitivity evenly over `j` orthonormal channels
  of its own, give `r·j` gates with `p = 1/r` and `δ = 1/j`, so `m · p · δ = 1`: equality in `gate_count_le` with `B = 1`.
-/

open Module Finset
open scoped InnerProductSpace RealInnerProductSpace

set_option linter.unusedSectionVars false

noncomputable section

namespace GateBound

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]

/-- A frame bound `Σₐ ⟪sₐ, x⟫² ≤ B‖x‖²` gives `Σₐ ⟪Y sₐ, sₐ⟫ ≤ B · tr Y` for every positive `Y` (spectral theorem). -/
theorem sum_inner_le_frame_trace {Y : E →ₗ[ℝ] E} (hY : Y.IsPositive) {ι : Type*} [Fintype ι] (s : ι → E) {B : ℝ}
    (hB : ∀ x, ∑ a, ⟪s a, x⟫ ^ 2 ≤ B * ‖x‖ ^ 2) : ∑ a, ⟪Y (s a), s a⟫ ≤ B * LinearMap.trace ℝ E Y := by
  have hT := hY.isSymmetric
  set b := hT.eigenvectorBasis rfl
  set ev := hT.eigenvalues rfl
  have hYb : ∀ i, Y (b i) = ev i • b i := fun i => by simp [b, ev]
  have hx : ∀ x, ⟪Y x, x⟫ = ∑ i, ev i * ⟪b i, x⟫ ^ 2 := fun x => by
    rw [← b.sum_inner_mul_inner (Y x) x]
    refine sum_congr rfl fun i _ => ?_
    rw [hT x (b i), hYb, inner_smul_right, real_inner_comm x (b i)]
    ring
  have htr : LinearMap.trace ℝ E Y = ∑ i, ev i := by
    rw [LinearMap.trace_eq_sum_inner Y b]
    refine sum_congr rfl fun i _ => ?_
    rw [hYb, inner_smul_right, real_inner_self_eq_norm_sq, b.orthonormal.1 i]
    ring
  calc ∑ a, ⟪Y (s a), s a⟫ = ∑ i, ev i * ∑ a, ⟪b i, s a⟫ ^ 2 := by
        simp_rw [hx, mul_sum]
        exact sum_comm
    _ ≤ ∑ i, ev i * B := sum_le_sum fun i _ => by
        refine mul_le_mul_of_nonneg_left ?_ (hY.nonneg_eigenvalues rfl i)
        simpa [real_inner_comm, b.orthonormal.1 i] using hB (b i)
    _ = B * LinearMap.trace ℝ E Y := by rw [htr, ← sum_mul, mul_comm]

/-- Channel quantities of unit channels lie in `[0, 1]`. -/
theorem inner_fingerprint_le_one {K : Type*} [Fintype K] (g : K → E) {s : E} (hs : ‖s‖ = 1) :
    ⟪fingerprint g s, s⟫ ≤ 1 := by
  rw [inner_fingerprint_self]
  have hZ0 : 0 ≤ ∑ k, ‖g k‖ ^ 2 := sum_nonneg fun _ _ => sq_nonneg _
  rcases hZ0.eq_or_lt with h | h
  · rw [← h, div_zero]
    exact zero_le_one
  rw [div_le_one h]
  refine sum_le_sum fun k _ => ?_
  have h1 := abs_real_inner_le_norm (g k) s
  rw [hs, mul_one] at h1
  rw [← sq_abs]
  exact pow_le_pow_left₀ (abs_nonneg _) h1 2

variable {Ω : Type*} [Fintype Ω] (μ : Ω → ℝ)

/-- **Theorem 7 drops exactly the off-mean**: `p · (E[q | on] - E[q | off]) = E[q] - E[q | off]`. -/
theorem mass_mul_contrast_eq (hsum : ∑ ρ, μ ρ = 1) (e : Ω → Prop) [DecidablePred e] (q : Ω → ℝ)
    (hp : 0 < mass μ e) (hp' : mass μ e < 1) :
    mass μ e * (condMean μ e q - condMean μ (fun ρ => ¬ e ρ) q) = mean μ q - condMean μ (fun ρ => ¬ e ρ) q := by
  have hoff : mass μ (fun ρ => ¬ e ρ) = 1 - mass μ e := by linarith [mass_add_mass_not μ e]
  rw [condMean_real, condMean_real, hoff]
  set p := mass μ e
  set a := ∑ ρ ∈ univ.filter e, μ ρ * q ρ
  set b := ∑ ρ ∈ univ.filter (fun ρ => ¬ e ρ), μ ρ * q ρ
  have hm : mean μ q = a + b := (sum_filter_add_sum_filter_not univ e _).symm
  have h1 : 1 - p ≠ 0 := by linarith
  rw [hm]
  field_simp
  ring

/-- **Theorem 7*** (one gate). If `q = y + a - b` with `a, b ≥ 0` and `y` constant, then
`p(1-p) · (E[q | on] - E[q | off]) ≤ (1-p) · E[a] + p · E[b]`. With `y` the `p`-quantile of `q`, `a = (q - y)₊` and
`b = (q - y)₋`, this is an equality for the event `{q > y}`; `y = 0`, `a = q`, `b = 0` is Theorem 7. -/
theorem contrast_le_split (hμ : ∀ ρ, 0 ≤ μ ρ) (hsum : ∑ ρ, μ ρ = 1) (e : Ω → Prop) [DecidablePred e]
    {q a b : Ω → ℝ} (y : ℝ) (hq : ∀ ρ, q ρ = y + a ρ - b ρ) (ha : ∀ ρ, 0 ≤ a ρ) (hb : ∀ ρ, 0 ≤ b ρ)
    (hp : 0 < mass μ e) (hp' : mass μ e < 1) :
    mass μ e * (1 - mass μ e) * (condMean μ e q - condMean μ (fun ρ => ¬ e ρ) q) ≤
      (1 - mass μ e) * mean μ a + mass μ e * mean μ b := by
  have hoff : mass μ (fun ρ => ¬ e ρ) = 1 - mass μ e := by linarith [mass_add_mass_not μ e]
  rw [condMean_real, condMean_real, hoff]
  set p := mass μ e with hpdef
  set A := ∑ ρ ∈ univ.filter e, μ ρ * a ρ
  set A' := ∑ ρ ∈ univ.filter (fun ρ => ¬ e ρ), μ ρ * a ρ
  set Bn := ∑ ρ ∈ univ.filter e, μ ρ * b ρ
  set Bn' := ∑ ρ ∈ univ.filter (fun ρ => ¬ e ρ), μ ρ * b ρ
  have hsplit : ∀ (f : Ω → ℝ) (S : Finset Ω),
      ∑ ρ ∈ S, μ ρ * (y + a ρ - b ρ) = y * ∑ ρ ∈ S, μ ρ + ∑ ρ ∈ S, μ ρ * a ρ - ∑ ρ ∈ S, μ ρ * b ρ :=
    fun _ S => by rw [mul_sum, ← sum_add_distrib, ← sum_sub_distrib]; exact sum_congr rfl fun ρ _ => by ring
  have hon : ∑ ρ ∈ univ.filter e, μ ρ * q ρ = y * p + A - Bn := by
    simp_rw [hq]
    exact hsplit q _
  have hoffs : ∑ ρ ∈ univ.filter (fun ρ => ¬ e ρ), μ ρ * q ρ = y * (1 - p) + A' - Bn' := by
    simp_rw [hq]
    rw [hsplit q, ← hoff]
    rfl
  have hma : mean μ a = A + A' := (sum_filter_add_sum_filter_not univ e _).symm
  have hmb : mean μ b = Bn + Bn' := (sum_filter_add_sum_filter_not univ e _).symm
  have hA' : 0 ≤ A' := sum_nonneg fun ρ _ => mul_nonneg (hμ ρ) (ha ρ)
  have hBn : 0 ≤ Bn := sum_nonneg fun ρ _ => mul_nonneg (hμ ρ) (hb ρ)
  have h1 : 1 - p ≠ 0 := by linarith
  have hp0 : p ≠ 0 := hp.ne'
  have key : p * (1 - p) * ((y * p + A - Bn) / p - (y * (1 - p) + A' - Bn') / (1 - p)) =
      (1 - p) * (A - Bn) - p * (A' - Bn') := by
    field_simp
    ring
  rw [hon, hoffs, key, hma, hmb]
  have : (1 - p) * (A + A') + p * (Bn + Bn') - ((1 - p) * (A - Bn) - p * (A' - Bn')) = A' + Bn := by ring
  linarith

/-- **Theorem 7*** for fingerprints: for any `Y₀` and positive `Y⁺_ρ, Y⁻_ρ` with `P_ρ = Y₀ + Y⁺_ρ - Y⁻_ρ`, gates of
prevalence `p` on channels with frame bound `B` satisfy
`Σₐ p(1-p) · (E[qₐ | on] - E[qₐ | off]) ≤ B · E[(1-p) tr Y⁺ + p tr Y⁻]`. -/
theorem quantile_budget (hμ : ∀ ρ, 0 ≤ μ ρ) (hsum : ∑ ρ, μ ρ = 1) {K : Type*} [Fintype K] (g : Ω → K → E)
    {ι : Type*} [Fintype ι] (s : ι → E) {B : ℝ} (hB : ∀ x, ∑ a, ⟪s a, x⟫ ^ 2 ≤ B * ‖x‖ ^ 2)
    (Y₀ : E →ₗ[ℝ] E) (Yp Ym : Ω → E →ₗ[ℝ] E) (hYp : ∀ ρ, (Yp ρ).IsPositive) (hYm : ∀ ρ, (Ym ρ).IsPositive)
    (hY : ∀ ρ, fingerprint (g ρ) = Y₀ + Yp ρ - Ym ρ)
    (e : ι → Ω → Prop) [∀ a, DecidablePred (e a)] {p : ℝ} (hpa : ∀ a, mass μ (e a) = p) (hp : 0 < p) (hp' : p < 1) :
    ∑ a, p * (1 - p) * (condMean μ (e a) (fun ρ => ⟪fingerprint (g ρ) (s a), s a⟫) -
      condMean μ (fun ρ => ¬ e a ρ) (fun ρ => ⟪fingerprint (g ρ) (s a), s a⟫)) ≤
      B * ∑ ρ, μ ρ * ((1 - p) * LinearMap.trace ℝ E (Yp ρ) + p * LinearMap.trace ℝ E (Ym ρ)) := by
  calc _ ≤ ∑ a, ((1 - p) * mean μ (fun ρ => ⟪Yp ρ (s a), s a⟫) + p * mean μ (fun ρ => ⟪Ym ρ (s a), s a⟫)) :=
        sum_le_sum fun a _ => by
          have h := contrast_le_split μ hμ hsum (e a) (q := fun ρ => ⟪fingerprint (g ρ) (s a), s a⟫)
            (a := fun ρ => ⟪Yp ρ (s a), s a⟫) (b := fun ρ => ⟪Ym ρ (s a), s a⟫) ⟪Y₀ (s a), s a⟫
            (fun ρ => by simp only [hY ρ, LinearMap.sub_apply, LinearMap.add_apply, inner_sub_left, inner_add_left])
            (fun ρ => (hYp ρ).inner_nonneg_left _) (fun ρ => (hYm ρ).inner_nonneg_left _)
            ((hpa a).symm ▸ hp) ((hpa a).symm ▸ hp')
          rwa [hpa a] at h
    _ = ∑ ρ, μ ρ * ((1 - p) * ∑ a, ⟪Yp ρ (s a), s a⟫ + p * ∑ a, ⟪Ym ρ (s a), s a⟫) := by
        simp only [mean, mul_add, mul_sum, sum_add_distrib]
        rw [sum_comm (f := fun a ρ => (1 - p) * (μ ρ * ⟪Yp ρ (s a), s a⟫)),
          sum_comm (f := fun a ρ => p * (μ ρ * ⟪Ym ρ (s a), s a⟫))]
        congr 1 <;> exact sum_congr rfl fun ρ _ => sum_congr rfl fun a _ => by ring
    _ ≤ ∑ ρ, μ ρ * ((1 - p) * (B * LinearMap.trace ℝ E (Yp ρ)) + p * (B * LinearMap.trace ℝ E (Ym ρ))) :=
        sum_le_sum fun ρ _ => mul_le_mul_of_nonneg_left
          (add_le_add (mul_le_mul_of_nonneg_left (sum_inner_le_frame_trace (hYp ρ) s hB) (by linarith))
            (mul_le_mul_of_nonneg_left (sum_inner_le_frame_trace (hYm ρ) s hB) hp.le)) (hμ ρ)
    _ = _ := by
        rw [mul_sum]
        exact sum_congr rfl fun ρ _ => by ring

/-- **Theorem 7***, counted: `m` gates of prevalence `p` with `E[qₐ | on] - E[qₐ | off] ≥ δ` (contrast `γ = d·δ`)
satisfy `m · p(1-p) · δ ≤ B · E[(1-p) tr Y⁺ + p tr Y⁻]`. With `Y₀ = 0`, `Y⁺ = P`, `Y⁻ = 0` the right side is `B(1-p)`,
which is Theorem 7 (`gate_count_le`). -/
theorem gate_count_quantile (hμ : ∀ ρ, 0 ≤ μ ρ) (hsum : ∑ ρ, μ ρ = 1) {K : Type*} [Fintype K] (g : Ω → K → E)
    {ι : Type*} [Fintype ι] (s : ι → E) {B : ℝ} (hB : ∀ x, ∑ a, ⟪s a, x⟫ ^ 2 ≤ B * ‖x‖ ^ 2)
    (Y₀ : E →ₗ[ℝ] E) (Yp Ym : Ω → E →ₗ[ℝ] E) (hYp : ∀ ρ, (Yp ρ).IsPositive) (hYm : ∀ ρ, (Ym ρ).IsPositive)
    (hY : ∀ ρ, fingerprint (g ρ) = Y₀ + Yp ρ - Ym ρ)
    (e : ι → Ω → Prop) [∀ a, DecidablePred (e a)] {p δ : ℝ} (hpa : ∀ a, mass μ (e a) = p) (hp : 0 < p) (hp' : p < 1)
    (hδa : ∀ a, δ ≤ condMean μ (e a) (fun ρ => ⟪fingerprint (g ρ) (s a), s a⟫) -
      condMean μ (fun ρ => ¬ e a ρ) (fun ρ => ⟪fingerprint (g ρ) (s a), s a⟫)) :
    Fintype.card ι * (p * (1 - p) * δ) ≤
      B * ∑ ρ, μ ρ * ((1 - p) * LinearMap.trace ℝ E (Yp ρ) + p * LinearMap.trace ℝ E (Ym ρ)) := by
  have hpq : 0 ≤ p * (1 - p) := mul_nonneg hp.le (by linarith)
  calc (Fintype.card ι : ℝ) * (p * (1 - p) * δ) = ∑ _a : ι, p * (1 - p) * δ := by simp
    _ ≤ _ := sum_le_sum fun a _ => mul_le_mul_of_nonneg_left (hδa a) hpq
    _ ≤ _ := quantile_budget μ hμ hsum g s hB Y₀ Yp Ym hYp hYm hY e hpa hp hp'

/-- **Theorem 7′**: each gate pays at most its channel's mass on its own on-rows, so
`Σₐ pₐ · (E[qₐ | on] - E[qₐ | off]) ≤ E[Σ_{a open at ρ} qₐ(ρ)] ≤ B_on`, where `B_on` bounds the frame of the channels
open at any one row. -/
theorem linear_budget_on (hμ : ∀ ρ, 0 ≤ μ ρ) (hsum : ∑ ρ, μ ρ = 1) {K : Type*} [Fintype K] (g : Ω → K → E)
    {ι : Type*} [Fintype ι] (s : ι → E) (e : ι → Ω → Prop) [∀ a, DecidablePred (e a)] {B : ℝ} (hB0 : 0 ≤ B)
    (hB : ∀ ρ x, ∑ a ∈ univ.filter (fun a => e a ρ), ⟪s a, x⟫ ^ 2 ≤ B * ‖x‖ ^ 2)
    (hp : ∀ a, 0 < mass μ (e a)) :
    ∑ a, mass μ (e a) * (condMean μ (e a) (fun ρ => ⟪fingerprint (g ρ) (s a), s a⟫) -
      condMean μ (fun ρ => ¬ e a ρ) (fun ρ => ⟪fingerprint (g ρ) (s a), s a⟫)) ≤
      ∑ ρ, μ ρ * ∑ a ∈ univ.filter (fun a => e a ρ), ⟪fingerprint (g ρ) (s a), s a⟫ ∧
    ∑ ρ, μ ρ * ∑ a ∈ univ.filter (fun a => e a ρ), ⟪fingerprint (g ρ) (s a), s a⟫ ≤ B := by
  have hq : ∀ a ρ, 0 ≤ ⟪fingerprint (g ρ) (s a), s a⟫ := fun a ρ => (fingerprint_isPositive (g ρ)).inner_nonneg_left _
  refine ⟨?_, ?_⟩
  · calc _ ≤ ∑ a, ∑ ρ ∈ univ.filter (e a), μ ρ * ⟪fingerprint (g ρ) (s a), s a⟫ := sum_le_sum fun a _ => by
          rw [condMean_real, condMean_real, mul_sub, mul_div_cancel₀ _ (hp a).ne']
          have : 0 ≤ mass μ (e a) * ((∑ ρ ∈ univ.filter (fun ρ => ¬ e a ρ), μ ρ * ⟪fingerprint (g ρ) (s a), s a⟫) /
              mass μ (fun ρ => ¬ e a ρ)) :=
            mul_nonneg (hp a).le (div_nonneg (sum_nonneg fun ρ _ => mul_nonneg (hμ ρ) (hq a ρ))
              (sum_nonneg fun ρ _ => hμ ρ))
          linarith
      _ = _ := by
          simp_rw [sum_filter, mul_sum, mul_ite, mul_zero]
          exact sum_comm
  · calc _ ≤ ∑ ρ, μ ρ * B := sum_le_sum fun ρ _ => by
          refine mul_le_mul_of_nonneg_left ?_ (hμ ρ)
          have h := sum_inner_fingerprint_le (g ρ) (fun a : ↥(univ.filter fun a => e a ρ) => s a)
            (fun x => by rw [sum_coe_sort (univ.filter fun a => e a ρ) (fun a => ⟪s a, x⟫ ^ 2)]; exact hB ρ x) hB0
          rwa [sum_coe_sort (univ.filter fun a => e a ρ) (fun a => ⟪fingerprint (g ρ) (s a), s a⟫)] at h
      _ = B := by rw [← sum_mul, hsum, one_mul]

/-- **Theorem 7′**, counted: if the channels open at any one row have frame bound `B_on` (`1` if they are orthonormal),
gates with prevalence `≥ p_min` and `E[qₐ | on] - E[qₐ | off] ≥ δ` satisfy `m · p_min · δ ≤ B_on`, i.e.
`m ≤ B_on · d / (p_min γ)`, however many channels there are in all. -/
theorem gate_count_on (hμ : ∀ ρ, 0 ≤ μ ρ) (hsum : ∑ ρ, μ ρ = 1) {K : Type*} [Fintype K] (g : Ω → K → E)
    {ι : Type*} [Fintype ι] (s : ι → E) (e : ι → Ω → Prop) [∀ a, DecidablePred (e a)] {B : ℝ} (hB0 : 0 ≤ B)
    (hB : ∀ ρ x, ∑ a ∈ univ.filter (fun a => e a ρ), ⟪s a, x⟫ ^ 2 ≤ B * ‖x‖ ^ 2) {pmin δ : ℝ} (hpmin : 0 < pmin)
    (hδ : 0 ≤ δ) (hpa : ∀ a, pmin ≤ mass μ (e a))
    (hδa : ∀ a, δ ≤ condMean μ (e a) (fun ρ => ⟪fingerprint (g ρ) (s a), s a⟫) -
      condMean μ (fun ρ => ¬ e a ρ) (fun ρ => ⟪fingerprint (g ρ) (s a), s a⟫)) :
    Fintype.card ι * (pmin * δ) ≤ B := by
  have hp : ∀ a, 0 < mass μ (e a) := fun a => hpmin.trans_le (hpa a)
  obtain ⟨h1, h2⟩ := linear_budget_on μ hμ hsum g s e hB0 hB hp
  calc (Fintype.card ι : ℝ) * (pmin * δ) = ∑ _a : ι, pmin * δ := by simp
    _ ≤ _ := sum_le_sum fun a _ => mul_le_mul (hpa a) (hδa a) hδ (hp a).le
    _ ≤ B := h1.trans h2

/-! ### Two-sided budget, and Theorem 5 as a corollary -/

theorem condMean_mem_unit (hμ : ∀ ρ, 0 ≤ μ ρ) (e : Ω → Prop) [DecidablePred e] {q : Ω → ℝ} (h0 : ∀ ρ, 0 ≤ q ρ)
    (h1 : ∀ ρ, q ρ ≤ 1) (hp : 0 < mass μ e) : 0 ≤ condMean μ e q ∧ condMean μ e q ≤ 1 := by
  rw [condMean_real]
  refine ⟨div_nonneg (sum_nonneg fun ρ _ => mul_nonneg (hμ ρ) (h0 ρ)) hp.le, (div_le_one hp).mpr ?_⟩
  exact sum_le_sum fun ρ _ => by nlinarith [hμ ρ, h1 ρ]

/-- The contrast of a unit channel is at most `1` in size: `|E[q | on] - E[q | off]| ≤ 1`, i.e. `γ ≤ d`. -/
theorem contrast_abs_le_one (hμ : ∀ ρ, 0 ≤ μ ρ) {K : Type*} [Fintype K] (g : Ω → K → E) {s : E} (hs : ‖s‖ = 1)
    (e : Ω → Prop) [DecidablePred e] (hp : 0 < mass μ e) (hp' : 0 < mass μ (fun ρ => ¬ e ρ)) :
    |condMean μ e (fun ρ => ⟪fingerprint (g ρ) s, s⟫) -
      condMean μ (fun ρ => ¬ e ρ) (fun ρ => ⟪fingerprint (g ρ) s, s⟫)| ≤ 1 := by
  have h0 : ∀ ρ, 0 ≤ ⟪fingerprint (g ρ) s, s⟫ := fun ρ => (fingerprint_isPositive (g ρ)).inner_nonneg_left _
  have h1 : ∀ ρ, ⟪fingerprint (g ρ) s, s⟫ ≤ 1 := fun ρ => inner_fingerprint_le_one (g ρ) hs
  obtain ⟨a0, a1⟩ := condMean_mem_unit μ hμ e h0 h1 hp
  obtain ⟨b0, b1⟩ := condMean_mem_unit μ hμ _ h0 h1 hp'
  rw [abs_le]
  constructor <;> linarith

/-- **Two-sided budget** (one gate): opening and closing gates both pay `min(p, 1-p) · |Δ| ≤ E[q]`. A closing gate is an
opening gate of the complementary event, of prevalence `1 - p`. -/
theorem linear_budget_gate_abs (hμ : ∀ ρ, 0 ≤ μ ρ) (hsum : ∑ ρ, μ ρ = 1) (e : Ω → Prop) [DecidablePred e]
    {q : Ω → ℝ} (hq : ∀ ρ, 0 ≤ q ρ) (hp : 0 < mass μ e) (hp' : mass μ e < 1) :
    min (mass μ e) (1 - mass μ e) * |condMean μ e q - condMean μ (fun ρ => ¬ e ρ) q| ≤ mean μ q := by
  have hoff : mass μ (fun ρ => ¬ e ρ) = 1 - mass μ e := by linarith [mass_add_mass_not μ e]
  rcases le_or_gt 0 (condMean μ e q - condMean μ (fun ρ => ¬ e ρ) q) with h | h
  · rw [abs_of_nonneg h]
    exact (mul_le_mul_of_nonneg_right (min_le_left _ _) h).trans (linear_budget_gate μ hμ hsum e hq hp hp')
  · rw [abs_of_neg h]
    have hc := linear_budget_gate μ hμ hsum (fun ρ => ¬ e ρ) hq (by linarith) (by linarith)
    simp only [not_not, hoff] at hc
    calc min (mass μ e) (1 - mass μ e) * -(condMean μ e q - condMean μ (fun ρ => ¬ e ρ) q)
        ≤ (1 - mass μ e) * -(condMean μ e q - condMean μ (fun ρ => ¬ e ρ) q) :=
          mul_le_mul_of_nonneg_right (min_le_right _ _) (by linarith)
      _ = (1 - mass μ e) * (condMean μ (fun ρ => ¬ e ρ) q - condMean μ e q) := by ring
      _ ≤ mean μ q := hc

/-- **Two-sided budget**, counted: gates with `p_min ≤ pₐ ≤ 1 - p_min` and `|E[qₐ | on] - E[qₐ | off]| ≥ δ`, opening
or closing, on channels with frame bound `B`, satisfy `m · p_min · δ ≤ B`. -/
theorem gate_count_abs (hμ : ∀ ρ, 0 ≤ μ ρ) (hsum : ∑ ρ, μ ρ = 1) {K : Type*} [Fintype K] (g : Ω → K → E)
    {ι : Type*} [Fintype ι] (s : ι → E) {B : ℝ} (hB : ∀ x, ∑ a, ⟪s a, x⟫ ^ 2 ≤ B * ‖x‖ ^ 2) (hB0 : 0 ≤ B)
    (e : ι → Ω → Prop) [∀ a, DecidablePred (e a)] {pmin δ : ℝ} (hpmin : 0 < pmin) (hδ : 0 ≤ δ)
    (hpa : ∀ a, pmin ≤ mass μ (e a)) (hpa' : ∀ a, mass μ (e a) ≤ 1 - pmin)
    (hδa : ∀ a, δ ≤ |condMean μ (e a) (fun ρ => ⟪fingerprint (g ρ) (s a), s a⟫) -
      condMean μ (fun ρ => ¬ e a ρ) (fun ρ => ⟪fingerprint (g ρ) (s a), s a⟫)|) :
    Fintype.card ι * (pmin * δ) ≤ B := by
  calc (Fintype.card ι : ℝ) * (pmin * δ) = ∑ _a : ι, pmin * δ := by simp
    _ ≤ ∑ a, mean μ (fun ρ => ⟪fingerprint (g ρ) (s a), s a⟫) := sum_le_sum fun a _ => by
        have hp : 0 < mass μ (e a) := hpmin.trans_le (hpa a)
        have hp' : mass μ (e a) < 1 := by linarith [hpa' a]
        refine le_trans ?_ (linear_budget_gate_abs μ hμ hsum (e a)
          (fun ρ => (fingerprint_isPositive (g ρ)).inner_nonneg_left _) hp hp')
        exact mul_le_mul (le_min (hpa a) (by linarith [hpa' a])) (hδa a) hδ (le_min hp.le (by linarith))
    _ = ∑ ρ, μ ρ * ∑ a, ⟪fingerprint (g ρ) (s a), s a⟫ := by
        simp only [mean, mul_sum]
        exact sum_comm
    _ ≤ ∑ ρ, μ ρ * B := sum_le_sum fun ρ _ =>
        mul_le_mul_of_nonneg_left (sum_inner_fingerprint_le (g ρ) s hB hB0) (hμ ρ)
    _ = B := by rw [← sum_mul, hsum, one_mul]

/-- **Theorem 5's unconditional bound follows from the linear budget, `d/γ` times sharper**: orthonormal gates with
`p(1-p) ≥ c` and `|E[qₐ | on] - E[qₐ | off]| ≥ δ` satisfy `m · c · δ ≤ 1`. Since `δ ≤ 1` (`contrast_abs_le_one`), this
implies `m · c · δ² ≤ 1` (`gate_count_orthonormal`), and improves it by the factor `1/δ = d/γ`. -/
theorem gate_count_orthonormal_linear (hμ : ∀ ρ, 0 ≤ μ ρ) (hsum : ∑ ρ, μ ρ = 1) {K : Type*} [Fintype K]
    (g : Ω → K → E) {ι : Type*} [Fintype ι] {s : ι → E} (hs : Orthonormal ℝ s) (e : ι → Ω → Prop)
    [∀ a, DecidablePred (e a)] (hp : ∀ a, 0 < mass μ (e a)) (hp' : ∀ a, mass μ (e a) < 1) {c δ : ℝ}
    (hc : ∀ a, c ≤ mass μ (e a) * (1 - mass μ (e a))) (hδ : 0 ≤ δ)
    (hδa : ∀ a, δ ≤ |condMean μ (e a) (fun ρ => ⟪fingerprint (g ρ) (s a), s a⟫) -
      condMean μ (fun ρ => ¬ e a ρ) (fun ρ => ⟪fingerprint (g ρ) (s a), s a⟫)|) :
    Fintype.card ι * (c * δ) ≤ 1 := by
  -- `p(1-p) ≤ min(p, 1-p)`, since the other factor is at most `1`
  have hmin : ∀ a, c ≤ min (mass μ (e a)) (1 - mass μ (e a)) := fun a => by
    have h0 := hp a
    have h1 := hp' a
    refine (hc a).trans (le_min ?_ ?_) <;> nlinarith
  calc (Fintype.card ι : ℝ) * (c * δ) = ∑ _a : ι, c * δ := by simp
    _ ≤ ∑ a, mean μ (fun ρ => ⟪fingerprint (g ρ) (s a), s a⟫) := sum_le_sum fun a _ => by
        refine le_trans ?_ (linear_budget_gate_abs μ hμ hsum (e a)
          (fun ρ => (fingerprint_isPositive (g ρ)).inner_nonneg_left _) (hp a) (hp' a))
        rcases le_or_gt 0 c with hc0 | hc0
        · exact mul_le_mul (hmin a) (hδa a) hδ (le_min (hp a).le (by linarith [hp' a]))
        · exact (mul_nonpos_of_nonpos_of_nonneg hc0.le hδ).trans
            (mul_nonneg (le_min (hp a).le (by linarith [hp' a])) (abs_nonneg _))
    _ = ∑ ρ, μ ρ * ∑ a, ⟪fingerprint (g ρ) (s a), s a⟫ := by
        simp only [mean, mul_sum]
        exact sum_comm
    _ ≤ ∑ ρ, μ ρ * 1 := sum_le_sum fun ρ _ =>
        mul_le_mul_of_nonneg_left (sum_inner_fingerprint_le (g ρ) s (B := 1)
          (fun x => frame_of_orthonormal hs x) zero_le_one) (hμ ρ)
    _ = 1 := by rw [← sum_mul, hsum, one_mul]

/-! ### Theorem 7 is sharp -/

/-- **Theorem 7 is attained.** `r` equally weighted rows; row `ρ` has sensitivity vectors `b (ρ, 0), …, b (ρ, j-1)`
(orthonormal), so its fingerprint spreads its unit trace evenly over its own `j` channels. Gate `a = (ρ', k)` opens on
row `ρ'` along `b a`. Every gate has prevalence `1/r` and `E[q | on] - E[q | off] = 1/j`, so the `m = r·j` gates have
`m · p · δ = 1`, equality in `gate_count_le` with `B = 1`. In contrast units (`γ = d/j`, `p = 1/r`): `m = d/(pγ)`. -/
theorem linear_budget_sharp {r j : ℕ} (hr : 0 < r) (hj : 0 < j) (b : Fin r × Fin j → E) (hb : Orthonormal ℝ b)
    (a : Fin r × Fin j) :
    mass (fun _ : Fin r => (r : ℝ)⁻¹) (fun ρ => ρ = a.1) = (r : ℝ)⁻¹ ∧
    condMean (fun _ : Fin r => (r : ℝ)⁻¹) (fun ρ => ρ = a.1) (fun ρ => ⟪fingerprint (fun k => b (ρ, k)) (b a), b a⟫) -
      condMean (fun _ : Fin r => (r : ℝ)⁻¹) (fun ρ => ¬ ρ = a.1)
        (fun ρ => ⟪fingerprint (fun k => b (ρ, k)) (b a), b a⟫) = (j : ℝ)⁻¹ := by
  classical
  have hq : ∀ ρ, ⟪fingerprint (fun k => b (ρ, k)) (b a), b a⟫ = if ρ = a.1 then (j : ℝ)⁻¹ else 0 := by
    intro ρ
    rw [inner_fingerprint_self]
    have hden : ∑ k, ‖b (ρ, k)‖ ^ 2 = j := by simp [hb.1]
    have hnum : ∑ k, ⟪b (ρ, k), b a⟫ ^ 2 = if ρ = a.1 then 1 else 0 := by
      simp_rw [orthonormal_iff_ite.mp hb]
      obtain ⟨a1, a2⟩ := a
      by_cases h : ρ = a1 <;> simp [h, Prod.ext_iff]
    rw [hden, hnum]
    split_ifs <;> simp
  have hmass : mass (fun _ : Fin r => (r : ℝ)⁻¹) (fun ρ => ρ = a.1) = (r : ℝ)⁻¹ := by
    simp [mass, Finset.filter_eq']
  refine ⟨hmass, ?_⟩
  have hr0 : (r : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hr.ne'
  rw [condMean_real, condMean_real, hmass]
  simp_rw [hq]
  have hoff : ∑ ρ ∈ univ.filter (fun ρ => ¬ ρ = a.1), (r : ℝ)⁻¹ * (if ρ = a.1 then (j : ℝ)⁻¹ else 0) = 0 :=
    sum_eq_zero fun ρ hρ => by simp [(mem_filter.mp hρ).2]
  rw [hoff, zero_div, sub_zero]
  simp only [Finset.filter_eq', mem_univ, ite_true, sum_singleton]
  field_simp

end GateBound

end
