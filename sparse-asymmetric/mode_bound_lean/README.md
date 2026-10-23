# mode_bound_lean

Lean 4 / Mathlib formalization of the bounds in `../mode_bound.md`. Build with `lake build` (Lean and Mathlib
v4.35.0; `lake exe cache get` fetches prebuilt Mathlib). Every theorem below depends only on Lean's standard axioms
(`propext`, `Classical.choice`, `Quot.sound`); there is no `sorry`.

The input space is any finite-dimensional real inner product space `E` (the whitened MLP input, `d = finrank E`).
Rows are a finite weighted set (`Prob.lean`). Gate modes are counted through *gate spaces*: subspaces on which
`⟪M x, x⟫ ≥ t‖x‖²` (`t = γ/d`). `card_eigen_le` turns any bound on gate spaces into a bound on the number of
eigenvalues `≥ t` (modes that open) or `≤ -t` (modes that close).

| mode_bound.md | Lean (namespace `ModeBound`) | file |
|---|---|---|
| gate modes ↔ gate spaces | `exists_gateSpace_eigen`, `card_eigen_le` | GateSpace |
| Lemma 2 (i): `M = K_S + E_S`, `K_S` vanishes on `ker W_Sᵀ` | `neuronOp`, `remainder`, `readS`, `inner_neuronOp_eq_remainder` | RankLemma |
| Lemma 2 (ii): `K_S` has ≤ `rank W_S` positive and ≤ `rank W_S` negative directions | `rank_lemma`, `finrank_range_readS_le` | RankLemma |
| Theorem 3: `±⟪E_S x, x⟫ < t‖x‖²` ⇒ ≤ `\|S\|` modes that open and ≤ `\|S\|` that close | `gateModes_le`, `card_gateModes_le` | RankLemma |
| Theorem 3′: modes beyond `rank W_S` are a gate space of `E_S` inside `ker W_Sᵀ` | `gateModes_compression`, `gateSpace_compression` | RankLemma, GateSpace |
| Proposition 1: `t · dim ≤ 1`, i.e. ≤ `d/γ` modes per opener | `gate_budget`, `opener_budget`, `opener_budget_count` | Budget |
| fingerprints `P = GᵀG/‖G‖²_F` are positive with trace 1 | `fingerprint`, `fingerprint_isPositive`, `trace_fingerprint` | Budget |
| Theorem 7 (one gate): `p·A ≤ d·sᵀP̄s` | `linear_budget_gate` | Budget |
| Theorem 7: `Σₐ pₐAₐ ≤ d·B`; `m ≤ B·d/(p_min γ)` | `linear_budget`, `gate_count_le`, `frame_of_orthonormal` | Budget |
| Theorem 5 (a): `p(1-p)Δ² ≤ Var q` | `contrast_sq_le_var` | Variance |
| Theorem 5 (c): `Σₐ Var qₐ ≤ B κ`; `m · c · δ² ≤ B κ` | `sum_var_le`, `gate_count_var` | Variance |
| `‖P‖²_F ≤ 1` for every fingerprint; `κ = E‖P - P̄‖²_F ≤ 1` | `hsSq_fingerprint_le_one`, `kappa_le_one` | GateBound |
| `κ ≤ 1 - 1/d` when every row has a nonzero gradient (`(tr D)² ≤ d·‖D‖²_F`, `tr P̄ = 1`) | `kappa_le_one_sub`, `trace_sq_le_hsSq`, `trace_meanFP` | GateBound |
| **Gate bound**: for fingerprints, `m · c · δ² ≤ B κ ≤ B`; orthonormal channels: `m ≤ d²/(p(1-p) γ²)` | `gate_count_fingerprint`, `gate_count_orthonormal` | GateBound |
| width: inertia of `Σᵢ cᵢ wᵢwᵢᵀ` ≤ `H` | `inertia_le` | Width |
| Prop. 6 / Example 9: `m` bilinear gates need exactly `2m` neurons | `two_mul_le_width`, `sqNet_bilinearGates` | Width |

## Hypotheses, as formalized

* Theorem 3 assumes `⟪E_S x, x⟫ < t‖x‖²` for `x ≠ 0` (largest eigenvalue of `E_S` below `t`), which is weaker than
  `‖E_S‖_op < t`. The closing-mode half assumes the same of `-E_S`.
* Theorem 5 (c) is proved twice: in an arbitrary inner product space `F` of fingerprints (`Variance.lean`), and for
  the actual fingerprints with the Frobenius norm `‖D‖²_F = tr(D²)` and channels `qₐ = sₐᵀPsₐ` (`GateBound.lean`),
  where the frame bound is `Σₐ (sₐᵀDsₐ)² ≤ B‖D‖²_F` for symmetric `D` (`B = 1` for orthonormal channels).
* Frame bounds `B` are hypotheses of the form `Σₐ ⟪sₐ, x⟫² ≤ B‖x‖²`. For orthonormal channels `B = 1`
  (`frame_of_orthonormal`).
* The width results are for networks with square activations, whose curvature form can be read off exactly
  (`sqNet_even`).

## Not formalized

* Theorem 4 (matrix Bernstein; finite-sample statements).
* The spectral forms that need von Neumann's trace inequality or the eigenvalues of `K`: Theorem 5 (b) and 5′, the
  spectral version of Theorem 7.
* Proposition 6 (a)–(c) for general activations: Hessians of an MLP via calculus, and the Hessian span of sigmoid gates.
* Everything empirical.
