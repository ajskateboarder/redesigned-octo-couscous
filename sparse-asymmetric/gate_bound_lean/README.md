# gate_bound_lean

Lean 4 / Mathlib formalization of the bounds in `../gate_bound.md`. Build with `lake build` (Lean and Mathlib
v4.35.0; `lake exe cache get` fetches prebuilt Mathlib). Every theorem below depends only on Lean's standard axioms
(`propext`, `Classical.choice`, `Quot.sound`); there is no `sorry`.

The input space is any finite-dimensional real inner product space `E` (the whitened MLP input, `d = finrank E`).
Rows are a finite weighted set (`Prob.lean`). Gate directions are counted through *gate spaces*: subspaces on which
`⟪M x, x⟫ ≥ t‖x‖²` (`t = γ/d`). `card_eigen_le` turns any bound on gate spaces into a bound on the number of
eigenvalues `≥ t` (gate directions that open) or `≤ -t` (gate directions that close).

| gate_bound.md | Lean (namespace `GateBound`) | file |
|---|---|---|
| gate directions ↔ gate spaces | `exists_gateSpace_eigen`, `card_eigen_le` | GateSpace |
| Lemma 2 (i): `M = K_S + E_S`, `K_S` vanishes on `ker W_Sᵀ` | `neuronOp`, `remainder`, `readS`, `inner_neuronOp_eq_remainder` | RankLemma |
| Lemma 2 (ii): `K_S` has ≤ `rank W_S` positive and ≤ `rank W_S` negative directions | `rank_lemma`, `finrank_range_readS_le` | RankLemma |
| Theorem 3: `±⟪E_S x, x⟫ < t‖x‖²` ⇒ ≤ `\|S\|` gate directions that open and ≤ `\|S\|` that close | `gateDirections_le`, `card_gateDirections_le` | RankLemma |
| Theorem 3′: gate directions beyond `rank W_S` are a gate space of `E_S` inside `ker W_Sᵀ` | `gateDirections_compression`, `gateSpace_compression` | RankLemma, GateSpace |
| Proposition 1: `t · dim ≤ 1`, i.e. ≤ `d/γ` gate directions per opener | `gate_budget`, `opener_budget`, `opener_budget_count` | Budget |
| fingerprints `P = GᵀG/‖G‖²_F` are positive with trace 1 | `fingerprint`, `fingerprint_isPositive`, `trace_fingerprint` | Budget |
| Theorem 7 (one gate): `p·A ≤ d·sᵀP̄s` | `linear_budget_gate` | Budget |
| Theorem 7: `Σₐ pₐAₐ ≤ d·B`; `m ≤ B·d/(p_min γ)` | `linear_budget`, `gate_count_le`, `frame_of_orthonormal` | Budget |
| Theorem 5 (a): `p(1-p)Δ² ≤ Var q` | `contrast_sq_le_var` | Variance |
| Theorem 5 (c): `Σₐ Var qₐ ≤ B κ`; `m · c · δ² ≤ B κ` | `sum_var_le`, `gate_count_var` | Variance |
| `‖P‖²_F ≤ 1` for every fingerprint; `κ = E‖P - P̄‖²_F ≤ 1` | `hsSq_fingerprint_le_one`, `kappa_le_one` | GateBound |
| `κ ≤ 1 - 1/d` when every row has a nonzero gradient (`(tr D)² ≤ d·‖D‖²_F`, `tr P̄ = 1`) | `kappa_le_one_sub`, `trace_sq_le_hsSq`, `trace_meanFP` | GateBound |
| Theorem 5″(a): gate count with the frame bound on trace-zero matrices only | `gate_count_traceless`, `sum_var_fingerprint_le_traceless`, `trace_fingerprint_sub_meanFP` | GateBound |
| **Gate bound**: for fingerprints, `m · c · δ² ≤ B κ ≤ B`; orthonormal channels: `m ≤ d²/(p(1-p) γ²)` | `gate_count_fingerprint`, `gate_count_orthonormal` | GateBound |
| Thm 7 drops exactly the off-mean: `p·Δ = E[q] - E[q \| off]` | `mass_mul_contrast_eq` | Sharp |
| Theorem 7′: `Σₐ pₐΔₐ ≤ E[Σ_{a open} qₐ] ≤ B_on` (frame of co-open channels only); `m·p_min·δ ≤ B_on` | `linear_budget_on`, `gate_count_on` | Sharp |
| Theorem 7*: `p(1-p)Δ ≤ (1-p)E[a] + pE[b]` for `q = y + a - b`; `m·p(1-p)δ ≤ B·E[(1-p)tr Y⁺ + p tr Y⁻]` for `P - Y₀ = Y⁺ - Y⁻` | `contrast_le_split`, `quantile_budget`, `gate_count_quantile`, `sum_inner_le_frame_trace` | Sharp |
| two-sided budget `min(p,1-p)\|Δ\| ≤ E[q]`; `\|Δ\| ≤ 1`; Theorem 5's bound from it, `d/γ` times sharper: `m·c·δ ≤ 1` | `linear_budget_gate_abs`, `gate_count_abs`, `contrast_abs_le_one`, `gate_count_orthonormal_linear` | Sharp |
| Theorem 7 is attained: `r·j` gates with `p = 1/r`, `δ = 1/j` | `linear_budget_sharp` | Sharp |
| Theorem 8 (a): containment, `((λ - τ⊥)² + τ×²)·‖Π⊥s‖² ≤ τ×²‖s‖²` for gate directions, from outside / cross capacities of any subspace `V` | `containment` | Containment |
| Theorem 8 (b): channels with `‖Π⊥sₐ‖² ≤ ε` number `≤ B·dim V / (1 - ε)` | `card_le_of_contained` | Containment |
| Theorem 5 (a) as a capacity: `Var q ≤ σ²` ⇒ `\|Δ\| ≤ σ/√(p(1-p))` for any event | `abs_contrast_le_of_var`, `inner_condMean_apply` | Containment |
| Theorem 8 (c), (d): spiked–incoherent class `P = S + N`, `S` on `V`, variance bounds on `N` off `V` ⇒ containment and `m ≤ B·dim V/(1 - ε)` | `spiked_containment`, `spiked_count` | Containment |
| Corollary 8.1: orthonormal gates number at most `N_c`, the largest number of orthonormal directions whose share has variance `≥ c·δ²` | `concentratedCount`, `card_le_concentratedCount` | Containment |
| width: inertia of `Σᵢ cᵢ wᵢwᵢᵀ` ≤ `H` | `inertia_le` | Width |
| Prop. 6 / Example 9: `m` bilinear gates need exactly `2m` neurons | `two_mul_le_width`, `sqNet_bilinearGates` | Width |

## Hypotheses, as formalized

* Theorem 3 assumes `⟪E_S x, x⟫ < t‖x‖²` for `x ≠ 0` (largest eigenvalue of `E_S` below `t`), which is weaker than
  `‖E_S‖_op < t`. The closing half assumes the same of `-E_S`.
* Theorem 5 (c) is proved twice: in an arbitrary inner product space `F` of fingerprints (`Variance.lean`), and for
  the actual fingerprints with the Frobenius norm `‖D‖²_F = tr(D²)` and channels `qₐ = sₐᵀPsₐ` (`GateBound.lean`),
  where the frame bound is `Σₐ (sₐᵀDsₐ)² ≤ B‖D‖²_F` for symmetric `D` (`B = 1` for orthonormal channels).
* Frame bounds `B` are hypotheses of the form `Σₐ ⟪sₐ, x⟫² ≤ B‖x‖²`. For orthonormal channels `B = 1`
  (`frame_of_orthonormal`).
* The width results are for networks with square activations, whose curvature form can be read off exactly
  (`sqNet_even`).

## Not formalized

* Theorem 4 (matrix Bernstein; finite-sample statements) and Theorem 5″(b), (c) (matrix Chernoff, and the
  Kogan–Nandy–Huang edge theorem, for random channels): Mathlib has no matrix concentration or random-matrix results.
* The spectral forms that need von Neumann's trace inequality or the eigenvalues of `K`: Theorem 5 (b) and 5′, the
  spectral versions of Theorems 7 and 7*.
* The Haar fourth-moment computation of Theorem 8 (c), which turns isotropy of the remainder into the variance bounds
  that `spiked_containment` takes as hypotheses.
* The Theorem 5 half of the sharpness example (`Σₐ Var qₐ = κ` there; checked numerically), and the Marchenko–Pastur
  form of Theorem 7 for random dictionaries.
* Proposition 6 (a)–(c) for general activations: Hessians of an MLP via calculus, and the Hessian span of sigmoid gates.
* Everything empirical.
