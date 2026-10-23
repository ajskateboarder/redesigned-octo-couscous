# Dense-gate mode bound: proofs and empirical checks

This note proves the statements in *Dense-Gate Mode Bound: Theorem and Empirical Verification Plan*, sharpens two of them,
replaces the sketched budget step (ii) with a proof, and reports the checks of §3 of that plan. Code:
`amp_mode_bound.py` (checks on one run), `amp_mlp_width.py` (retrained MLPs for the d_mlp scaling test),
`amp_mode_bound.sh` (everything), `amp_mode_bound_table.py` (summary). Logs: `downstream_amp/checks/amp_mode_bound_*.log`.

## 0. Summary

* **Proved, with no assumptions on the model:**
  * The rank lemma, including the inertia claim (at most k eigenvalues of each sign).
  * The per-opener theorem, plus a strictly stronger variant (Theorem 3′) whose hypothesis can be checked from M_o
    alone. Its conclusion holds per sign: at most k positive and at most k negative gate modes.
  * The plan's "stated boundary": modes beyond r = rank w_S are modes of Q M_o Q, i.e. of the part carried outside S.
  * An unconditional per-opener budget, N_± ≤ d/γ (Proposition 1).
  * A total gate-count bound that replaces sketch (ii) (Theorem 5): m ≤ B·κ·d² / (p_min(1−p_min)γ²) ≤ B d²/(p_min(1−p_min)γ²).
    It has a sharper spectral form in terms of the eigenvalues of the fingerprint covariance K, and a further
    refinement (Theorem 5′).
  * A finite-sample statement (Theorem 4).
* **d_mlp-independence holds automatically.** P_ρ is a d×d matrix, so every count defined through it is at most d per
  opener and O(d²/γ²) in total, whatever d_mlp is. Saying the count is "bounded independently of d_mlp" therefore has no
  content by itself. The content is in the constants (κ, λ_i(K), B, ‖E_S‖). These are properties of the trained model,
  and §3 measures them.
* **Empirically.** Checked on 18 one-hidden-layer-into-readout MLPs at d = 512: the pretrained `gelu_1l` L0, the
  pretrained `gelu_2l` L1, and 16 retrained `gelu_1l` MLPs (§3). In all of them:
  * every qualifying opener has 1 positive gate mode at γ ≥ 50 and at most 2 at γ = 25. There are never any negative
    modes.
  * The plan's hypothesis ‖E_S‖ < γ/d fails for single-neuron openers. There ‖E_i‖ ≈ λ₁(M_i) (median ratio 0.97–1.00):
    the mode is carried by other, co-firing neurons.
  * For multi-neuron sets the hypothesis holds sometimes. It holds for the sets the earlier anatomy checks found, which
    contain the co-firing neurons. It holds less often for greedy sets.
  * Where it holds, the bound 2k (4–10) is true but loose: the observed count is 1.
* **The rank lemma does not explain the small count.** It is better described by the fingerprint covariance K. The
  channels have Var(sᵀPs) ≈ 0.3–0.95·λ₁(K), so they sit on the top few eigenvectors of K. In pretrained `gelu_1l`, K's
  top eigenvector is a rank-one matrix s*s*ᵀ (95%), and the one dominant channel *is* s* (|⟨ssᵀ, V₁⟩| = 0.97); 164
  openers open it.
* **d_mlp scaling at fixed d_model = 512.** The plan's prediction holds:
  * from d_mlp = 512 to 8192, the number of single-neuron openers that open a mode grows about 8× (71 → 557 at γ = 25);
  * the number of distinct channels falls (13.5 → 5 at γ = 25; 3 → 1 at γ = 50);
  * the pipeline's attractor count stays at 4.5–5.5;
  * the same holds for both seeds, with Muon in place of AdamW, and with ReLU or ReLU² in place of GELU.
  * The extra neurons become extra openers of the *same* channels.

* **Width (§4).** Planted-gate experiments separate two capacities.
  * *Storage* is width-limited: a width-H MLP stores about H/2 gates once d ≥ 128 (2 neurons per gate, the minimum
    Proposition 6 allows for exact storage), and H/3–H/5 at smaller d.
  * *Gates at a fixed contrast γ* are limited by a linear budget Σ_a p_a A_a ≤ d·B′ (Theorem 7), independent of width.
  * The trained LM is at its width limit only at d_mlp = 32 (about 13 channels from 32 neurons). From 256 up it is an
    order of magnitude below both limits, so for this model width is not what limits the number of gates.

## 1. Setup and notation

Rows ρ ~ μ. The MLP reads z_ρ ∈ ℝ^d through threshold neurons u = W_inᵀ z + b, h = φ(u), with H = d_mlp neurons.
Pushes s are unit vectors in whitened coordinates (push Σ^½ s). Define

* w_i = Σ^½ W_in[:, i] ∈ ℝ^d, the whitened read direction of neuron i, and W = [w_1 … w_H] ∈ ℝ^{d×H}.
* F_ρ = C_ρ^½ L_ρ D_ρ ∈ ℝ^{k×H}, where D_ρ = diag φ′(u_ρ), L_ρ = ∂(top-k logits)/∂h, C_ρ = ½(diag p − p pᵀ).
* G_ρ = C_ρ^½ J_ρ Σ^½ = F_ρ Wᵀ. Then g_ρ(s) = ‖G_ρ s‖² is exactly `downstream_amp.py`'s gain.
* P_ρ = G_ρᵀG_ρ / Z_ρ with Z_ρ = ‖G_ρ‖²_F. P_ρ is PSD with tr P_ρ = 1, and P_ρ = W A_ρ Wᵀ with A_ρ = F_ρᵀF_ρ / Z_ρ ⪰ 0.

The factorisation P_ρ = W A_ρ Wᵀ does not depend on the model being one layer deep. The push enters only through the
MLP's read directions, so everything downstream ends up in F_ρ, whether that is later layers, attention to other
positions or the readout. The rank lemma below therefore holds for any downstream network. Only the *computation* of
G_ρ is closed-form in the one-layer case.

An **opener** is an event E with p = μ(E) ∈ [p_min, 1 − p_min]. Its matrix is M = E[P | E] − E[P | Eᶜ].
A **gate mode at level γ** is an eigenpair (λ, s) of M with |λ| ≥ t := γ/d. Write N₊(M, t) = #{i : λ_i(M) ≥ t} and
N₋(M, t) = #{i : λ_i(M) ≤ −t}, counted with multiplicity.

The contrast A(o, s) = d·sᵀMs is `downstream_amp.py`'s objective, with its normaliser τ (the mean gain of 8 random
pushes) replaced by its expectation Z/d. Measured on `gelu_1l`, τ/(Z/d) has median 1.00 with a 10–90% range of
0.77–1.23. Because of that noise in τ, the pipeline's contrasts come out about 25–40% higher than the exact ones:
101 / 70 / 81 in the pipeline against 80 / 51 / 57 exact for the three attractors.

## 2. Results

### Proposition 1 (elementary facts; unconditional budget per opener)

(a) tr M = 0, and M = Cov(P, 1_E) / (p(1−p)), entrywise.
(b) −E[P | Eᶜ] ⪯ M ⪯ E[P | E], so ‖M‖_op ≤ 1.
(c) Σ_{λ_i>0} λ_i(M) = Σ_{λ_i<0} |λ_i(M)| ≤ 1 − tr(Π₊ E[P | Eᶜ]) ≤ 1, where Π₊ projects onto the positive eigenspace
of M.
(d) Hence N₊(M, γ/d) ≤ d/γ and N₋(M, γ/d) ≤ d/γ, for every opener and every model.

*Proof.* (a) Both conditional means have trace 1. Also E[P·1_E] − p E[P] = p(1−p)(E[P|E] − E[P|Eᶜ]).
(b) Both conditional means are PSD and have eigenvalues in [0, 1].
(c) Σ₊ = tr(Π₊M) = tr(Π₊E[P|E]) − tr(Π₊E[P|Eᶜ]) ≤ tr E[P|E] − tr(Π₊E[P|Eᶜ]), using 0 ⪯ Π₊ ⪯ I. Since tr M = 0,
Σ₊ = Σ₋.
(d) N₊·t ≤ Σ₊ ≤ 1. ∎

At d = 512 this gives N₊ ≤ 10 at γ = 50 and N₊ ≤ 5 at γ = 100. That is the size of the observed counts, and it holds
for every opener, neuron set or not, with no hypothesis about E_o.

### Lemma 2 (rank structure)

Fix any S ⊂ [H] with |S| = k, let R be its complement, let W_S = W[:, S] ∈ ℝ^{d×k} and r = rank W_S ≤ k. Let
ΔA = E[A | E] − E[A | Eᶜ]. Then

  M = K_S + E_S,  K_S = W_S Uᵀ + U W_Sᵀ,  U = ½ W_S ΔA_SS + W_R ΔA_RS ∈ ℝ^{d×k},  E_S = W_R ΔA_RR W_Rᵀ,

and

(i) xᵀK_S x = 0 for every x ⊥ range W_S;
(ii) K_S has at most r positive and at most r negative eigenvalues, so rank K_S ≤ 2r ≤ 2k.

Equivalently, E_S = Δ E[G_RᵀG_R / Z] with G_R = G − F_S W_Sᵀ. This is M recomputed with S's columns deleted from the
numerator, while Z, every neuron's slope, the readout Jacobian L and the distribution p are all left as they are.

*Proof.* Expand M = W ΔA Wᵀ in the blocks of W = [W_S W_R]. The three blocks that involve S are
W_S ΔA_SS W_Sᵀ + W_S ΔA_SR W_Rᵀ + W_R ΔA_RS W_Sᵀ, and these equal W_S Uᵀ + U W_Sᵀ (use ΔA_SRᵀ = ΔA_RS).
(i) If W_Sᵀx = 0 then xᵀK_S x = 2(W_Sᵀx)ᵀ(Uᵀx) = 0.
(ii) Let V₊ be the positive eigenspace of K_S. If dim V₊ > r, then V₊ meets (range W_S)^⊥, which has codimension r,
in some x ≠ 0. That x has xᵀK_S x > 0, contradicting (i). The negative eigenspace is handled the same way. ∎

The lemma is algebra and holds for every S and every event. Its content therefore lies entirely in E_S. ΔA_RR collects
everything that changes off S when E happens: the slopes D_R of the other neurons, the readout Jacobian and p (both
depend on h, and so on S's outputs), and the normaliser Z.

### Theorem 3 (per-opener bound, as in the plan)

If ‖E_S‖_op < t, then N₊(M, t) ≤ r and N₋(M, t) ≤ r. In particular the number of gate modes is at most 2r ≤ 2k.

*Proof.* By Weyl's inequality, λ_{r+1}(M) ≤ λ_{r+1}(K_S) + λ₁(E_S) ≤ 0 + ‖E_S‖ < t, because λ_{r+1}(K_S) ≤ 0 by
Lemma 2(ii). Symmetrically, λ_{d−r}(M) ≥ −‖E_S‖ > −t. ∎

### Theorem 3′ (compression form; stronger, and needs only M)

Let Q = I − Π_S, where Π_S projects onto range W_S. For every t > 0,

  N₊(M, t) ≤ r + N₊(QMQ|_{range Q}, t),  N₋(M, t) ≤ r + N₋(QMQ|_{range Q}, t).

Moreover QMQ = Q E_S Q, so ‖QMQ‖ ≤ ‖E_S‖, and Theorem 3′ implies Theorem 3. In particular, N_± ≤ r whenever
‖QMQ‖ < t.

*Proof.* Let V be the span of the eigenvectors of M with eigenvalue ≥ t, and N = dim V. Then
dim(V ∩ range Q) ≥ N − r. For x in that intersection, xᵀQMQx = xᵀMx ≥ t‖x‖². By Courant–Fischer, QMQ restricted to
range Q therefore has at least N − r eigenvalues ≥ t. For the identity, Q W_S = 0 gives Q K_S Q = 0. ∎

The plan's §3 calls ‖QMQ‖ "equivalent" to ‖E_S‖. It is not: ‖QMQ‖ ≤ ‖E_S‖, and the inequality can be strict
(measured below: 13.5 against 20.1, in units of 1/d). Theorem 3′ is what makes ‖QMQ‖ sufficient anyway.

Theorem 3′ is also the plan's "stated boundary", now a theorem: every gate mode beyond the first r is a mode of the
compression of E_S off span(w_S), i.e. it is carried by neurons outside S.

### Theorem 4 (finite samples)

Suppose the on-rows are i.i.d. from μ(·|E) (n₁ of them) and the off-rows i.i.d. from μ(·|Eᶜ) (n₀ of them). Let
v_j = ‖E[P | ·]‖_op and ε(n, v) = √(2v log(2d/δ)/n) + 2 log(2d/δ)/(3n). Then, with probability ≥ 1 − 2δ,
‖M̂ − M‖ ≤ ε := ε(n₁, v₁) + ε(n₀, v₀), and therefore

  N₊(M̂, t + ε) ≤ N₊(M, t) ≤ N₊(M̂, t − ε).

*Proof.* Apply matrix Bernstein to X_i = P_i − E P. Each ‖X_i‖ ≤ 1, because both terms have spectra in [0, 1]. Also
E X_i² ⪯ E P² ⪯ E P, because 0 ⪯ P ⪯ I gives P² ⪯ P. Finish with Weyl's inequality. ∎

Rows from the same window are not independent. The conservative valid version uses one row per window, and that is
vacuous at these sample sizes (see §3.9). Empirically, a window bootstrap and the held-out split stand in for it.

### Theorem 5 (total gate count; replaces sketch (ii))

Let K be the covariance operator of P on Sym(d) with the Frobenius inner product: K(X) = E[⟨P − P̄, X⟩(P − P̄)].
Write λ₁ ≥ λ₂ ≥ … for its eigenvalues, V_i for its eigenvectors, and κ = tr K = E‖P − P̄‖²_F ≤ 1 − 1/d.

Consider gates (E_a, s_a), a = 1…m, with unit s_a, prevalences p_a ∈ [p_min, 1 − p_min] and contrasts
d·s_aᵀM_a s_a ≥ γ. Let B = λ_max([(s_a·s_b)²]_{ab}). This is the upper frame bound of the projectors X_a = s_a s_aᵀ; it
equals 1 for orthogonal channels and is ≤ 1 + max_a Σ_{b≠a}(s_a·s_b)². Put c = p_min(1 − p_min)γ²/d². Then:

(a) every gate satisfies Var(s_aᵀPs_a) = ⟨X_a, K X_a⟩ ≥ p_a(1−p_a)γ²/d² ≥ c;
(b) m·c ≤ B Σ_{i≤m} λ_i, i.e. m ≤ max{m : (1/m)Σ_{i≤m} λ_i ≥ c/B};
(c) m ≤ B κ / c ≤ B d² / (p_min(1−p_min)γ²);
(d) per opener, any-sign modes: N·p(1−p)t² ≤ Σ_{i≤N} λ_i.

*Proof.* (a) sᵀMs = Cov(q, 1_E)/(p(1−p)) with q = sᵀPs. By Cauchy–Schwarz, |Cov(q, 1_E)| ≤ sd(q)·√(p(1−p)), so
γ/d ≤ sd(q)/√(p(1−p)); and p(1−p) ≥ p_min(1−p_min) on the allowed interval.
(b) Let T = Σ_a X_a⊗X_a, i.e. T(Y) = Σ_a⟨X_a, Y⟩X_a. T is PSD, has rank ≤ m, and ‖T‖ = λ_max(Gram) = B. By von Neumann's
trace inequality, Σ_a⟨X_a, KX_a⟩ = tr(KT) ≤ Σ_i λ_i(K)λ_i(T) ≤ B Σ_{i≤m} λ_i.
(c) Σ_{i≤m} λ_i ≤ κ, and κ ≤ E‖P‖²_F − ‖P̄‖²_F ≤ 1 − 1/d.
(d) The eigenprojectors of M are Frobenius-orthonormal. For each of them, ⟨X_i, M⟩² ≤ ⟨X_i, KX_i⟩/(p(1−p)) by the
same Cauchy–Schwarz step; then apply Ky Fan. ∎

(c) is the plan's (d_model/γ)² heuristic made exact, with the constant B/(p_min(1−p_min)). (b) is the data-dependent
form that plan step 1 (the kernel spectrum) calls for.

### Theorem 5′ (gates must sit on the top of K)

Under the hypotheses of Theorem 5, take any j with λ_{j+1} < c. Every gate then has
y_a := Σ_{i≤j}⟨X_a, V_i⟩² ≥ β_j := (c − λ_{j+1})/(λ₁ − λ_{j+1}), and m ≤ B j / β_j.
In particular, if γ > d·√(λ₂/(p(1−p))), every gate channel aligns with V₁ (⟨s sᵀ, V₁⟩² ≥ β₁) and m ≤ B/β₁.

*Proof.* c ≤ Var(s_aᵀPs_a) = Σ_i λ_i⟨X_a, V_i⟩² ≤ λ₁y_a + λ_{j+1}(1 − y_a), since Σ_i⟨X_a, V_i⟩² = ‖X_a‖² = 1. That
gives y_a ≥ β_j. Summing, mβ_j ≤ Σ_{i≤j}Σ_a⟨X_a, V_i⟩² ≤ Σ_{i≤j} B‖V_i‖² = Bj. ∎

### Theorem 5″ (overlapping channels: only the trace-zero part counts)

Theorem 5's B is the frame bound of the X_a on all of Sym(d). Every X_a = s_a s_aᵀ has the same component I/d along
the identity, so for m random channels B ≈ 1 + m/d, and (c) says nothing once m ≫ d. Perpendicular channels avoid
this, but there are at most d of them, so the orthonormal form of (c), m ≤ d²/(p(1−p)γ²), only beats the trivial m ≤ d
when p(1−p)γ² > d. For superposition the bound has to work with B on trace-zero matrices.

Let V = {D ∈ Sym(d) : tr D = 0}, of dimension N = d(d+1)/2 − 1, and let B₀ = λ_max([(s_a·s_b)² − 1/d]_{ab}). This is
the frame bound of the y_a = X_a − I/d, which are the X_a projected onto V, so Σ_a (s_aᵀDs_a)² ≤ B₀‖D‖²_F for D ∈ V.

**(a) Deterministic.** If every row has G ≠ 0, then m·p_min(1−p_min)γ²/d² ≤ B₀κ ≤ B₀(1 − 1/d).

*Proof.* tr P = 1 on every row, so tr(P − P̄) = 0, and the proof of Theorem 5(c) only applies the frame bound to
P − P̄ ∈ V. On V, ⟨X_a, D⟩ = ⟨y_a, D⟩. ∎

**(b) Random channels.** Let s_1…s_M be independent and uniform on the unit sphere. For every η ∈ (0, 1), with
probability at least 1 − η,

  B₀ < max{ 2e²M/(d(d+2)), log(d²/η) }.

A subset of a dictionary has a frame bound no larger than the dictionary's (T below is a sum of PSD terms). So with the
same probability, *every* set of gates on distinct channels from the dictionary, in any model trained on any data,
satisfies m ≤ B₀·d²/(p_min(1−p_min)γ²). For M ≤ d²·log(d²/η)/(2e²) this is m ≤ d²·log(d²/η)/(p_min(1−p_min)γ²).

*Proof.* Let T = Σ_a y_a y_aᵀ act on V. Then B₀ = λ_max(T), since T and the Gram matrix [⟨y_a, y_b⟩] = [(s_a·s_b)² − 1/d]
have the same nonzero eigenvalues.
1. ‖y_a‖² = 1 − 2/d + 1/d = 1 − 1/d =: R ≤ 1, so each y_a y_aᵀ is PSD with λ_max ≤ R.
2. E[y yᵀ] = 2/(d(d+2)) · Id_V. For D ∈ V, ⟨y, D⟩ = sᵀDs. Write a standard Gaussian vector as g = ‖g‖s, with ‖g‖
   independent of s. Then E(gᵀDg)² = 2‖D‖²_F + (tr D)² = 2‖D‖²_F and E‖g‖⁴ = d(d+2), so E(sᵀDs)² = 2‖D‖²_F/(d(d+2)).
   A symmetric operator whose quadratic form is a constant times ‖D‖² is that constant times the identity. So
   μ := λ_max(E T) = 2M/(d(d+2)).
3. Matrix Chernoff (Tropp 2012, *User-friendly tail bounds*, Thm 1.1): for independent PSD summands with λ_max ≤ R on an
   N-dimensional space, P{λ_max(T) ≥ (1+ε)μ} ≤ N·[e^ε/(1+ε)^{1+ε}]^{μ/R}. Put t = (1+ε)μ and use
   ε − (1+ε)log(1+ε) ≤ (1+ε)(1 − log(1+ε)). The bound is at most N·exp((t/R)(1 − log(t/μ))), which for t ≥ e²μ is at
   most N·e^{−t/R} ≤ N·e^{−t}.
4. Take t = max{e²μ, log(N/η)}. Since N ≤ d², P{B₀ ≥ t} ≤ η. ∎

Bound (b) carries a log d factor that the truth does not. In simulation, B₀ tracks (1 + √(2M)/d)², the
Marchenko–Pastur edge for M samples in N ≈ d²/2 dimensions. Over 5 draws for each setting:

| d | M | max B₀ | (1 + √(2M)/d)² | proved bound (η = 0.01) | full-space B ≈ 1 + M/d |
|---|---|---|---|---|---|
| 32 | 512 | 3.86 | 4.00 | 11.5 | 17 |
| 64 | 256 | 1.84 | 1.83 | 12.9 | 5 |
| 64 | 2048 | 3.95 | 4.00 | 12.9 | 33 |
| 128 | 2048 | 2.23 | 2.25 | 14.3 | 17 |

**(c) Random sign channels: the sharp edge.** Let s_a = x_a/√d with x_1…x_M independent and uniform on {±1}^d. As
d → ∞ with M/d² → α ∈ (0, ∞), almost surely

  B₀ → (1 + √(2α))².

So for large d, B₀ ≈ (1 + √(2M)/d)², and every set of gates on distinct channels from the dictionary satisfies
m ≤ (1 + o(1))·(1 + √(2M)/d)²·d²/(p_min(1−p_min)γ²). For M ≤ d²/2 the factor is at most 4.

*Proof.* Let X₂ be the M × C(d,2) matrix with entries [X₂]_{a,{k<l}} = x_{ak}x_{al}. Since x_{ak}² = 1,
(x_a·x_b)² = d + 2Σ_{k<l} x_{ak}x_{al}x_{bk}x_{bl}. So for a ≠ b, (s_a·s_b)² − 1/d = 2[X₂X₂ᵀ]_{ab}/d². On the
diagonal, 1 − 1/d = 2C(d,2)/d². Hence the Gram matrix of the y_a is exactly (2/d²)X₂X₂ᵀ.

Kogan, Nandy and Huang (arXiv:2410.17515, Theorem 1.2, case d = ℓ = 2, in their notation n = M, p = d) study
R₂ = X₂X₂ᵀ/√(M d²) for i.i.d. entries with mean 0, variance 1 and E|x|^k ≤ k^{βk}. ±1 signs satisfy this. Their result:
λ_max(R₂ − diag R₂) → 2/√2! + √α = √2 + √α almost surely, both for M ≤ C(d,2) and for M > C(d,2).

Since diag R₂ = (d−1)/(2√M)·I,

  (2/d²)X₂X₂ᵀ = (1 − 1/d)·I + (2√M/d)·(R₂ − diag R₂),

and 2√M/d → 2√α. So B₀ → 1 + 2√α(√2 + √α) = (1 + √(2α))². ∎

This is asymptotic: the theorem gives no rate in d. Bound (b) remains the finite-d guarantee, and its proof works the
same way for sign channels, with E[yyᵀ] ⪯ (2/d²)·Id_V, so 2e²M/d² replaces 2e²M/(d(d+2)). Uniform-sphere channels are not covered by (c), since their
coordinates are not independent. They give the same edge in simulation (largest of 3 draws per setting):

| d | M/d² | (1 + √(2M)/d)² | B₀, sign channels | B₀, sphere channels |
|---|---|---|---|---|
| 64 | 1/2 | 4.000 | 3.979 | 3.918 |
| 128 | 1/4 | 2.914 | 2.908 | 2.884 |
| 128 | 1/2 | 4.000 | 3.980 | 3.956 |
| 256 | 1/16 | 1.832 | 1.826 | 1.824 |
| 256 | 1/4 | 2.914 | 2.905 | 2.902 |

So up to M ≈ d²/2 random channels cost at most about 4× as much as perpendicular ones, instead of M/d times as much.

What (b) and (c) do not give: trained models do not draw their channels uniformly at random. For a given model, B₀ is
computable exactly from its channels, as λ_max([(s_a·s_b)² − 1/d]). As before, the count is of *distinct channels*:
many openers on one channel can share its variance.

### Theorem 7 (linear budget across gates)

Every gate (E, s) with prevalence p and contrast A satisfies

  p·A ≤ d·E[sᵀPs] = d·sᵀP̄s.

Hence, for gates (E_a, s_a), a = 1…m, with B′ = ‖Σ_a s_a s_aᵀ‖_op (the frame bound of the channel vectors; 1 for
orthonormal channels):

  Σ_a p_a A_a ≤ d·Σ_a s_aᵀP̄s_a ≤ d·B′·Σ_{i≤m} λ_i(P̄) ≤ d·B′.

So if every gate has p_a ≥ p_min and A_a ≥ γ, then m ≤ d·B′/(p_min γ). More sharply, m is at most the largest m with
(1/m)Σ_{i≤m} λ_i(P̄) ≥ p_min γ/(d B′).

*Proof.* Let q = sᵀPs ≥ 0. Since E[q] = pE[q|E] + (1−p)E[q|Eᶜ],

  p(E[q|E] − E[q|Eᶜ]) = (E[q·1_E] − pE[q])/(1−p) ≤ (E[q] − pE[q])/(1−p) = E[q].

For the sum, Σ_a s_aᵀP̄s_a = tr(P̄ S Sᵀ) with S = [s_1 … s_m]. By von Neumann's trace inequality this is
≤ Σ_i λ_i(P̄)λ_i(SSᵀ) ≤ B′ Σ_{i≤m} λ_i(P̄), because SSᵀ has rank ≤ m and norm B′. Finally tr P̄ = 1. ∎

Theorem 5 charges each gate p(1−p)γ² against a variance budget. Theorem 7 charges it pγ against the mean fingerprint,
so it is linear in γ. In the planted-gate experiments (§4) the teacher's contrasts follow exactly this law:
Σ_a p A_a ≈ const as m grows.

### Proposition 6 (width lower bound for exact gates)

Let f(x) = W₂φ(W₁x + b₁) + b₂ with width H and φ ∈ C². Every Hessian of every output lies in an H-dimensional space:

  ∇²f_k(x) = Σ_i (W₂)_{ki} φ″(w_i·x + b_i) w_i w_iᵀ ∈ V := span{w_i w_iᵀ},  dim V ≤ H.

(a) **Exact representation.** If f = f* on an open set, then H ≥ dim span{∇²f*_k(x) : k, x}.

(b) **Approximate representation.** Let C* be the second-moment operator of the target's Hessian field on Sym(d),
C* = E_x Σ_k ∇²f*_k ⊗ ∇²f*_k, with eigenvalues μ₁ ≥ μ₂ ≥ …. Then for any width-H MLP,
E_x Σ_k ‖∇²f_k − ∇²f*_k‖²_F ≥ Σ_{i>H} μ_i.

(c) **Planted gates.** Take the planted gate f*(x) = Σ_a σ((o_a·x − t)/T)(s_a·x)u_a, with gates in general position. Its
Hessians span {o_a s_aᵀ + s_a o_aᵀ, o_a o_aᵀ : a ≤ m}, which has dimension 2m when 2m ≤ d(d+1)/2. So storing m such gates
exactly takes H ≥ 2m: at least 2 neurons per gate.

*Proof.* (a) Differentiate f twice.
(b) The student's Hessian field takes values in V. The best approximation of the target field by any field with values
in a fixed H-dimensional subspace is the projection onto that subspace. Minimising over subspaces is Eckart–Young for
the operator C*.
(c) ∇²f*_k = Σ_a u_{ak}[σ′(z_a)/T·(o_a s_aᵀ + s_a o_aᵀ) + σ″(z_a)(s_a·x)/T²·o_a o_aᵀ]. Take a generic combination of
outputs, which makes every coefficient Σ_k v_k u_{ak} nonzero. The 2m scalar functions σ′(z_a) and σ″(z_a)(s_a·x) are
linearly independent, because they are ridge functions along distinct directions. So evaluating at 2m generic points
recovers each matrix separately. ∎

A gate as counted in §1 only needs the Jacobian to change between E and Eᶜ, and fitting the Jacobian does not control
the Hessian. So Proposition 6 bounds exact or Hessian-faithful storage, not the counted gates. It gives the benchmark of
2 neurons per gate that the experiments in §4 approach.

### What is proved, and what is not

**Proved, unconditionally:**
* d_mlp-free bounds on gate modes per opener (Proposition 1(d), Theorem 5(d)) and on distinct channels in total
  (Theorem 5).
* Under ‖QMQ‖ < γ/d (implied by ‖E_S‖ < γ/d): at most r ≤ k gate modes of each sign per k-neuron opener.

**Not proved:**
* That trained models satisfy ‖E_S‖ < γ/d. Section 3 shows that for single neurons they do not.
* That the observed small count has the rank-lemma cause. Section 3 shows that the bound 2k is not tight, and that the
  small count tracks the top of K's spectrum instead.
* Anything for multi-layer *computation* of G. The algebra carries over, but the checks here use closed-form G.
* A Lipschitz bound in θ that would turn the certified point into a certified neighbourhood (plan step 9, second
  item). Not attempted.

## 3. Empirical checks

### Setting

Models: MLPs that feed straight into the readout, so G_ρ = F_ρWᵀ is computed exactly from autodiff of the readout
alone. Checked:
* `gelu_1l` L0 and `gelu_2l` L1 (pretrained);
* 16 retrained `gelu_1l` MLPs (`amp_mlp_width.py`). Everything except the MLP is the pretrained model, frozen. Each MLP
  is trained from scratch on the same 59M OWT tokens. Variants:
  * d_mlp ∈ {512, 1024, 2048, 4096, 8192} × 2 seeds with GELU and AdamW;
  * ReLU and ReLU² at 2048;
  * Muon at 512, 2048 (two seeds) and 8192.

  All retrained MLPs recover 0.99–1.05 of the pretrained MLP's held-out loss reduction. Held-out loss falls
  monotonically with width (4.71 → 4.41), so the wider MLPs do use their extra neurons.

Rows: each run's own 480 fitting windows × 16 positions (7680 rows), plus 4096 held-out rows.
Sanity check: ‖G s‖² matches `downstream_amp.py`'s jvp gain to within 0.8% (TF32).

Openers:
* **k = 1:** every neuron, with on = the top 10% of its pre-activation. The threshold is set on the fitting rows.
* **k = 2–4:** greedy sets grown from the 8 best single neurons, maximising λ₁.
* **Given sets:** the sets the earlier `amp_neurons.py` / `amp_gate_anatomy.py` checks found.
* **The run's own openers o.**
* **Nulls:** 32 random neuron sets and 64 random row events.

The γ thresholds are in exact contrast units. The pipeline's attractors on `gelu_1l` have exact contrasts 51–83.

### 3.1 Kernel (plan step 1)

| model | κ | λ₁(K) | λ₁/κ | λ₂/κ | rank-one share of V₁ | ‖P_ρ‖²_F median |
|---|---|---|---|---|---|---|
| gelu_1l | 0.145 | 0.0100 | 0.069 | 0.029 | 0.95 | 0.124 |
| gelu_2l L1 | 0.165 | 0.0046 | 0.028 | 0.026 | 0.46 | 0.154 |
| retrained (16) | 0.138–0.165 | 0.0096–0.0126 | 0.07–0.08 | — | 0.94–0.97 | — |

κ is far from tiny, so there is plenty for openers to modulate. Each row's fingerprint is concentrated (effective rank
1/‖P‖²_F ≈ 7–8, λ₁(P_ρ) ≈ 0.3), while the average fingerprint is not (effective rank 35–46). In every `gelu_1l` variant,
K's top eigenvector is almost exactly a rank-one matrix ±vvᵀ. In `gelu_2l` L1 the top three eigenvalues of K are nearly
degenerate (0.028, 0.026, 0.024 of κ).

### 3.2–3.3 Residual test and mode counts (plan steps 2–3)

`gelu_1l`, γ = 25 / 50 / 100 (γ/d = 0.049 / 0.098 / 0.195):

| openers | with ≥ 1 mode | N₊ max | N₋ max | Thm 3 holds (‖E_S‖ < γ/d) | Thm 3′ holds (‖QMQ‖ < γ/d) |
|---|---|---|---|---|---|
| single neurons (2048) | 155 / 8 / 0 | 2 / 1 / – | 0 | 5/155, 2/8 | 18/155, 6/8 |
| greedy k = 2–4 (24) | 24 / 24 / 8 | 1 | 0 | 9/24, 20/24, 24/24 | 22/24, 22/24, 24/24 |
| anatomy sets {251,1608}, {251,1608,1942,690}, {2008,…} | 3 / 2 / 0 | 1 | 0 | 3/3 | 3/3 |
| random sets (32) | 2 / 0 / 0 | 1 | 0 | – | – |

**Single neurons.** For the top 50 single-neuron openers, ‖E_i‖/λ₁(M_i) has median 1.00 and ‖QMQ‖/λ₁ has median 0.98:
the mode is essentially orthogonal to the opener's own read direction. The same holds in every model (median
0.97–1.00), so for k = 1 the plan's hypothesis fails.

**Sets that contain the co-firing neurons.** Example: S = {251, 1608}, the pair the anatomy check found firing
together.
* d·‖E_S‖ = 17 and d·‖QMQ‖ = 10, against d·λ₁ = 88. Theorem 3's hypothesis holds at γ = 25 and 50.
* 94% of the gate mode lies in span(w_S): the channel is read by the opener neurons themselves.
* K_S has one large positive eigenvalue (86) and one negative partner (−17). The negative partner comes from the
  x yᵀ + y xᵀ structure: for k = 1, w aᵀ + a wᵀ has eigenvalues (w·a) ± ‖w‖‖a‖. E_S cancels it, which is why M itself
  has no negative mode.
* So the gate is a *pair-mediated* one, which is the plan's boundary case: firing neuron 1608 alone moves the channel
  only because 251 co-fires.

**The count is not tight against the bound.** Every opener has N₊ = 1 against 2k = 4–10. By the plan's own criterion,
something beyond the lemma limits the count: in the K_S eigenvalues above, the second positive eigenvalue is ≤ 5 in
every set.

`gelu_2l` L1 behaves the same way:
* single neurons: N₊ ≤ 2, N₋ = 0, ‖E_i‖/λ₁ median 0.97;
* run openers o with their top-5 neuron sets: Theorem 3 holds for 5/6 at γ = 25, and N₊ ∈ {1, 2} against 2k = 10.

Theorem 3′'s inequality N₊ ≤ r + N₊(QMQ) is never violated in any model. It is a theorem, so this is a check of the
code rather than of the model.

The run's attractors are M_o's top eigenvectors: on `gelu_1l` |cos(s, top mode)| is 0.98, 0.99 and 0.84. Their exact
contrasts are 80 / 51 / 57, against the pipeline's τ-normalised 101 / 70 / 81.

### 3.4 Overlap (plan step 4)

max_{j∉S}‖Π_S ŵ_j‖, the largest share of another neuron's read direction inside span(w_S), is 0.90–0.95 for every
gate set and 0.40–0.65 for random sets. It does not predict ‖E_S‖: the greedy 4-sets have overlap 0.92–0.93 in every
case, yet d·‖E_S‖ ranges from 20 to 51. What predicts ‖E_S‖ is whether S contains the co-firing partner.
Sets containing 251 have S-mass on the on-rows of 0.26–0.28 and d·‖E_S‖ ≈ 20. Sets without it have S-mass of 0.04–0.05
and d·‖E_S‖ = 45–51.

### 3.5 Budget quantities (plan step 5)

* **Fisher-mass shares are flat:** the top single neuron carries 1.5% of E[tr P], the top 10 carry 9%, and the top 100
  carry 27%. The diagonal shares sum to 1.20 (cross terms are −0.20).
* **The gates are not where the mass is:** a gate set holds only ~26% of the on-rows' Fisher mass.
* **Principal cosines:** between span(w_S) of disjoint openers of distinct channels, the median is 0.16–0.65, not near
  zero.
* **Frame bound:** B for the distinct channels is 1.0–3.1.
* **Theorem 5(a) is tight to within a small factor:** measured Var(sᵀPs) is 2.6–10.5× its lower bound p(1−p)(γ/d)².
* **The total bounds are loose:** Theorem 5(b) gives m ≤ 440 / 80 / 5 on `gelu_1l` at γ = 25 / 50 / 100, against 2 / 2 / 1
  observed. Theorem 5′ only helps at large γ (m ≤ 7.6 on `gelu_2l` at γ = 100), because λ₂(K) lies above c except there.
* **Proposition 1** (d·Σ₊λ/γ) is at most 7–8 at γ = 25 and 3–4 at γ = 50, against an observed max N₊ of 2 and 1. This is
  the tightest of the per-opener bounds, and it needs no hypothesis.

**What the channels are.** The pretrained `gelu_1l` has 2 distinct channels at γ = 25 and 50, and 1 at γ = 100.
Channel 0 is V₁(K): |⟨ssᵀ, V₁⟩| = 0.97 and Var(sᵀPs) = 0.95·λ₁(K). It is opened by 164 of the 189 (opener, mode) pairs
at γ = 25. Across the retrained models, channel 0 is V₁ in about half of them (|⟨ssᵀ, V₁⟩| 0.58–0.94) and lies elsewhere
in the top of K in the rest. In every retrained model channel 0 has Var(sᵀPs) = 0.61–0.90·λ₁(K). In `gelu_2l` the three γ = 50 channels split K's near-degenerate top
three (|⟨ssᵀ, V₁⟩| = 0.39, 0.57, 0.44).

### 3.6 Nulls (plan step 6)

* **Random row events of prevalence 0.1:** d·‖M‖ has median 3.0 and max 5.7–8.7. That is the sampling noise floor, well
  below γ = 25.
* **Random 2- and 4-neuron sets:** d·λ₁ is 6–11, and 2 of 32 cross γ = 25 (none cross 50). The anatomy and greedy sets
  reach 77–105.

### 3.7 Scaling in d_mlp at fixed d_model = 512 (plan step 7)

GELU / AdamW, mean of 2 seeds:

| d_mlp | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|
| held-out loss | 4.71 | 4.60 | 4.51 | 4.45 | 4.41 |
| neurons opening a mode, γ = 25 | 71 | 109 | 167 | 296 | 557 |
| distinct channels, γ = 25 | 13.5 | 9.0 | 6.0 | 5.5 | 5.0 |
| neurons opening a mode, γ = 50 | 20 | 22 | 34 | 62 | 114 |
| distinct channels, γ = 50 | 3.0 | 1.0 | 1.5 | 1.5 | 1.0 |
| pipeline attractors (`downstream_amp.py`) | 5.5 | 4.5 | 4.5 | 4.5 | 5.0 |
| openers of the dominant channel, γ = 25 | 58–103 | 123–134 | 184–195 | 325–331 | 616–620 |
| max N₊ per opener (γ = 25 / 50) | 2 / 1 | 2 / 1 | 2 / 1 | 2 / 1 | 2 / 1 |

**The prediction holds.** The gate count does not grow with d_mlp; at γ = 25 it falls. The number of neurons that act as
openers grows roughly in proportion to d_mlp, and they pile onto the same few channels. The strongest gate in each retrained model has contrast 70–98, with no trend in width. Only ReLU² reaches γ = 100
(105). The pretrained `gelu_1l` reaches 105.

### 3.8 Robustness (plan step 8)

| variant (d_mlp 2048 unless noted) | channels, γ = 25 / 50 | attractors | κ | λ₁(K) |
|---|---|---|---|---|
| GELU AdamW (2 seeds) | 7, 5 / 2, 1 | 4, 5 | 0.150 | 0.0116 |
| GELU Muon (2 seeds) | 6, 6 / 2, 1 | 3, 5 | 0.147 | 0.0109 |
| ReLU AdamW | 4 / 2 | 4 | 0.151 | 0.0112 |
| ReLU² AdamW | 6 / 2 (1 at γ = 100) | 5 | 0.165 | 0.0126 |
| Muon at 512 / 8192 | 7 / 5 (γ = 25) | 4 / 4 | 0.138 / 0.156 | — |
| pretrained gelu_1l | 2 / 2 | 3 | 0.145 | 0.0100 |
| pretrained gelu_2l L1 | 7 / 3 | 6 | 0.165 | 0.0046 |

κ and λ₁(K) stay in a narrow band (κ 0.14–0.17; λ₁ 0.010–0.013 except `gelu_2l`). So do the counts, and so does the
per-opener structure: one positive mode, no negative modes, and ‖E_i‖ ≈ λ₁ for single neurons. Theorem 3's hypothesis
for greedy sets is the variable part. It holds for 0–7 of 24 at γ = 25 in the retrained models, against 9/24 in the
pretrained one.

### 3.9 Statistics (plan step 9)

**Held-out replication.** Mode counts replicate: per-opener N₊ is identical on the fitting and held-out rows for all
anatomy, run-opener and greedy sets, except two greedy sets near threshold. The number of single neurons with a mode is 155 → 182 at γ = 25 and
8 → 13 at γ = 50; these are threshold effects near γ.

**Window bootstrap** (100 resamples). For the strongest openers, d·λ₁ is stable to ±3–4 and d·λ₂ to ±0.5, and the
5–95% interval of N₊ is 1–1 at γ = 25 and 50.

**Theorem 4.**
* With i.i.d. rows, the 95% matrix-Bernstein radius is d·ε = 47 (`gelu_1l`) to 50 (`gelu_2l`). The population-level
  count is therefore certified only for gaps larger than about 47 in contrast units, i.e. at γ ≈ 100 here, not at 25–50.
* Counting one row per window (the dependence-safe version), d·ε ≈ 250, which is vacuous. Rigorous population
  statements need about 4× (γ = 100), 15× (γ = 50) and 50× (γ = 25) more independent windows.

**Not done.** The Lipschitz-in-θ neighbourhood bound was not attempted.

### 3.9b Frame bounds of the trained channels (Theorem 5″)

All 26 runs were redone with γ = 10, 15, 25, 50, 100. The γ = 25 and 50 counts reproduce exactly. The noise floor
from random openers is d·‖M‖ ≈ 3–9, so γ = 10 sits just above it. Each row is the range over the 24 one-layer models
(widths 32–8192, AdamW and Muon, GELU/ReLU/ReLU²) and gelu_2l:

| γ | channels m | B | B₀ | B₀ of m random channels | k_eff | m/B₀ |
|---|---|---|---|---|---|---|
| 10 | 41–139 | 3.3–8.4 | 3.3–8.2 | 1.04–1.07 | 13–18 | 12–18 |
| 15 | 10–39 | 1.4–4.3 | 1.4–4.3 | 1.02–1.03 | 6–12 | 6–11 |

k_eff = (tr F)²/‖F‖²_F with F = Σ_a s_a s_aᵀ is the effective dimension of the span of the channels.

* **B₀ ≈ B.** With m ≪ d, the shared I/d part that B₀ removes is only about m/d.
* **The channels are confined, not duplicated.** They fill a subspace of dimension k_eff ≈ 15 at γ = 10, the same
  for every width from 32 to 8192, both optimizers, all three activations and the two-layer model. The top k_eff
  directions carry 82–94% of F. Inside that subspace they are spread like random channels: B₀ is at or a little below the
  value for m random unit vectors in the same subspace (≈ 1 + m/k_eff): within 10% at γ = 10, and up to 40% below at
  γ = 15, where m is small. Few pairs are near-duplicates.
* **What this does to the bound.** In a k-dimensional subspace, B₀ ≈ 1 + m/k grows with m. One change in how much of
  the row's sensitivity lies in the subspace registers in every channel inside it. So m·c ≤ B₀κ constrains m only
  when c > κ/k. Here c = p(1−p)γ²/d² ≈ 3·10⁻⁵ at γ = 10 and κ/k ≈ 0.01, so the theorem does not limit how many
  channels fit inside the subspace. That is correct, not slack: one opener that raises the subspace's total share
  opens all its directions at once.
* **m/B₀ ≈ k_eff.** For m ≫ k, m/B₀ ≈ m/(1 + m/k) ≈ k. The robust number in the trained models is therefore the
  dimension, about 15, of the gated subspace, not the channel count, which grows as γ falls.

### 3.10 Verdict

1. **The theorem is true.** Lemma 2, Theorems 3 and 3′, Proposition 1 and Theorem 5 are proved above, and Theorem 5
   replaces the sketched step (ii). The d_mlp-independence they assert is automatic for any count defined through P.
2. **Hypothesis (i) fails in the form the plan states.** For openers that are single neurons, ‖E_S‖ ≈ λ₁ in every model:
   gates are carried by co-firing neurons outside S. It holds for sets that include the co-firing partner. There, the
   compression form (Theorem 3′) certifies N₊ ≤ r.
3. **The certified bound is not the reason the count is small.** The bound is 2k = 4–10; the observed count is 1 per
   opener and 1–15 channels in total. The data point instead to K: per-row fingerprints are concentrated, and the
   variance of P across rows sits in a few rank-one-like directions at the top of K's spectrum. Every opener modulates
   one of those few directions.
4. **The falsifiable prediction survives.** At fixed d_model = 512, the gate count stays flat or falls from d_mlp 512 to
   8192, across 2 seeds, 2 optimizers and 3 activations, while the number of openers grows about 8×.

**Open.** The doc's mechanism is "bounded-bandwidth modulation plus a conserved sensitivity budget". The second half is
what the data support: Proposition 1 and Theorem 5 are the operative bounds. A proof that the count is O(1) would need a
statement about K, namely that training concentrates Var(P) in a few rank-one directions. That is a property of trained
models, not of the architecture, and nothing here proves it.

## 4. Width limits: how many gates can an MLP of width H store?

The bounds in §2 are all set by d (and γ, p); none depends on H. This section asks whether width limits gate storage at
all, from two sides. On the achievability side, planted gates (`amp_gate_capacity.py`) measure how many gates a width-H
MLP *can* store. On the observation side, the d_mlp sweep is extended down to 32 ≪ d = 512 (`amp_mlp_width.py`,
`amp_mode_bound.py`).

### 4.1 Planted gates

**Teacher** (ℝ^d → ℝ^d, x ~ N(0, I)): f*(x) = Qx + α√d·Σ_a σ((o_a·x − t)/0.25)(s_a·x)u_a.
* Q is a random orthogonal matrix, which gives an isotropic baseline.
* The o_a, s_a and u_a are random, with s_a ⟂ o_a.
* Each gate is open on a share p of inputs.

**Student:** a GELU MLP of width H with no skip connection, trained by MSE on fresh samples.

**Measurement:** both are measured exactly as the LM is (§1, with the identity as readout).

**Stored:** a planted gate counts as stored if both
* the student's contrast is at least half the teacher's, and
* the student's top mode of M_a has |cos| ≥ 0.8 with s_a.

m* is the largest m with ≥ 90% of gates stored. Noise floor: random events give |A|/d ≤ 0.001.

**Storage capacity is width-limited and linear in H** (p = 0.1, α = 1):

| | H = 32 | 64 | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|---|---|
| m*, d = 64 (seed 0 / 1) | 16 / 16 | 8 / 16 | 32 / 32 | 64 / 64 | 128 / 128 | 256 / 256 | ≥ 256 |
| m*, d = 128 | 16 | 32 | 64 | 128 | 256 | 512 | ≥ 512 |
| m*, d = 32 | 1 | 16 | 32 | 64 | 64 | 64 | – |

Finer m grid (11 values of m):

| d | 32 | 64 | 128 | 256 |
|---|---|---|---|---|
| m* at H = 256 / 512 / 1024 | 96 / 96 / – | 96 / 192 / – | 128 / 256 / – | 128 / 256 / 512 |
| neurons per gate at H = 512 | 5.3 | 2.7 | 2.0 | 2.0 |

* For d ≥ 128, storage costs exactly 2 neurons per gate. That is the minimum Proposition 6 allows for exact storage, so
  the bound is attained.
* At small d it costs more.
* At d = 32 storage stops at about 3d, whatever the width. There the *teacher's* own modes stop aligning with their
  channels (|cos| 0.88 at m = 128, 0.74 at m = 256): gates in excess of about 3d are not separately identifiable.

**Storage at a fixed contrast is not width-limited.** The teacher's per-gate contrast falls as gates are added, and for
large m it follows m·p·A/d ≈ const:

| | p = 0.02 | p = 0.1 | p = 0.1, α = 3 | p = 0.3 |
|---|---|---|---|---|
| m·p·A/d at large m | 0.20 | 0.31 | 0.31 | 0.45 |
| gates with A/d ≥ 0.1, any H ≥ 128 (d = 64) | 64 | 16 | 22 | 8 |
| same at d = 32 / d = 128 (p = 0.1) | – | 16 / 16 | – | – |

This is Theorem 7's linear budget, Σ_a p_a A_a ≤ d·B′. Gates that are open together share tr P = 1, so each additional
gate dilutes the rest. Stronger gates (α = 3) double a single gate's contrast but leave the many-gate limit unchanged.
The count at fixed γ/d is the same for d = 32, 64 and 128, and scales roughly as 1/p.

### 4.2 The LM at small widths

Retrained `gelu_1l` MLPs, GELU / AdamW, mean of 2 seeds:

| d_mlp | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|---|---|---|---|
| loss reduction recovered | 0.83 | 0.88 | 0.92 | 0.96 | 0.99 | 1.01 | 1.03 | 1.04 | 1.05 |
| neurons opening a mode (γ = 25) | 30 | 42 | 40 | 47 | 71 | 109 | 167 | 296 | 557 |
| distinct channels, γ = 25 | 13.5 | 12.0 | 12.5 | 13.5 | 13.5 | 9.0 | 6.0 | 5.5 | 5.0 |
| distinct channels, γ = 50 | 5.0 | 3.5 | 3.0 | 3.0 | 3.0 | 1.0 | 1.5 | 1.5 | 1.0 |
| distinct channels, γ = 100 | 3.0 | 2.0 | 1.5 | 0 | 0 | 0 | 0 | 0 | 0 |
| λ₁(K)/κ | 0.13 | 0.10 | 0.11 | 0.08 | 0.08 | 0.08 | 0.08 | 0.08 | 0.08 |

**Width saturates only at the narrow end.** At d_mlp = 32, 30 of the 32 neurons open a mode and there are about 13
distinct channels. That is the planted storage limit, H/c ≈ 32/2.5. So the narrowest MLP is at its width capacity, and
it spends it on gates: it has the *strongest* gates of any width, the only ones above γ = 100 besides ReLU².

**From d_mlp ≈ 256 up, the count is far below both limits.**
* Width would allow H/2–H/5 gates. The LM has 5–14.
* Theorem 7 allows m ≤ 31–220 at γ = 100–25 on `gelu_1l` (summed budget 42–880 across runs). Measured, Σ p·A uses
  10–45 of it, and each gate uses 0.15–0.33 of its own cap d·sᵀP̄s.
* The count then *falls* with width. Extra width goes into more openers of fewer, shared channels.

### 4.3 What this establishes about gate capacity

For a one-hidden-layer MLP (input dimension d, width H) with gates at prevalence p and contrast ≥ γ:

  N_gates ≲ min( H/c(d),  B′·d / (p·γ),  ~3d ),

where c(d) → 2 (Proposition 6) and the middle term is Theorem 7 (proved, with the sharper spectral form available).
* **The first term** is width: proved as a lower bound on neurons for exact storage (Proposition 6), measured as 2–5
  neurons per gate.
* **The second term** is the budget: proved, independent of width, and the term that sets how many gates of a given
  strength can coexist.
* **The third term** is identifiability. It is observed only at d = 32; it is not proved.

Trained LMs are width-limited only at d_mlp ≈ 32 (H ≲ 2.5·count). From d_mlp ≈ 256 up, they sit an order of magnitude
below every term. So for this model and data, width is not what limits the number of gates; something about what
training finds useful is. Neither the theory nor these experiments identify that something.

### 4.4 Still open

* **Proposition 6 bounds exact or Hessian-faithful storage, not the counted gates.** A matching lower bound for
  Jacobian-only gates, i.e. for neurons per counted gate, is not proved. c ≈ 2 is measured, not derived, for the counted
  notion.
* **The ~3d identifiability ceiling** is seen at d = 32 only and has no proof. Note that c(d) > 2 at small d may be the
  same effect.
* **The LM's count** (5–14 channels at d = 512) is below every bound. Explaining it needs a statement about training
  (what makes gates worth their budget), not about capacity.
* **Planted gates use one teacher family:** sigmoid × linear gates with random directions. Sparse superposed gates
  (p → 0, Theorem 7 allows m ∝ 1/p) were tested only down to p = 0.02.

## 5. Lean formalization

`mode_bound_lean/` checks the following in Lean 4 / Mathlib, with no `sorry` and only the standard axioms:
* the link between gate modes (eigenvalues) and gate spaces;
* Lemma 2, Theorem 3 (both signs) and Theorem 3′, for the neuron model `M = Σᵢⱼ ΔAᵢⱼ wᵢwⱼᵀ`;
* Proposition 1;
* the fingerprint `P = GᵀG/‖G‖²_F` being positive with trace 1;
* Theorem 7 with its count `m ≤ B·d/(p_min γ)`;
* Theorem 5 (a) and (c), `κ ≤ 1`, and `κ ≤ 1 − 1/d` when every row has a nonzero gradient. Together these give the
  gate bound for actual fingerprints, for any model, data and width: with orthonormal channels, at most
  `d² / (p(1-p)·γ²)` gates (`gate_count_orthonormal`; informative only when this is below `d`), and with overlapping
  channels `m·p(1-p)γ²/d² ≤ B₀·κ`, where `B₀` is the frame bound on trace-zero matrices (Theorem 5″(a),
  `gate_count_traceless`);
* the width results: the inertia count, and the result that `m` bilinear gates need exactly `2m` neurons.

`mode_bound_lean/README.md` maps each result to its Lean theorem and lists what is not formalized: Theorem 4, the
spectral forms (5 (b), 5′), the random-channel bounds 5″(b), (c) (like Theorem 4, they need random-matrix results that Mathlib lacks), and
Proposition 6 for general activations.
