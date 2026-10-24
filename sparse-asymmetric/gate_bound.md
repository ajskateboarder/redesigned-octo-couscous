# Dense-gate bound: proofs and empirical checks

This note proves the statements in *Dense-Gate Bound: Theorem and Empirical Verification Plan*, sharpens two of them,
replaces the sketched budget step (ii) with a proof, and reports the checks of §3 of that plan. Code:
`amp_gate_bound.py` (checks on one run), `amp_mlp_width.py` (retrained MLPs for the d_mlp scaling test),
`amp_gate_bound.sh` (everything), `amp_gate_bound_table.py` (summary), `amp_bound_tightness.py` (how tight Theorems 5 and 7 are, §3.9c), `amp_containment.py` (Theorem 8's class on trained models, §3.9d). Logs: `downstream_amp/checks/amp_gate_bound_*.log`.

## 0. Summary

* **Proved, with no assumptions on the model:**
  * The rank lemma, including the inertia claim (at most k eigenvalues of each sign).
  * The per-opener theorem, plus a strictly stronger variant (Theorem 3′) whose hypothesis can be checked from M_o
    alone. Its conclusion holds per sign: at most k positive and at most k negative gate directions.
  * The plan's "stated boundary": gate directions beyond r = rank w_S are gate directions of Q M_o Q, i.e. of the part carried outside S.
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
  * every qualifying opener has 1 positive gate direction at γ ≥ 50 and at most 2 at γ = 25. There are never any negative
    gate directions.
  * The plan's hypothesis ‖E_S‖ < γ/d fails for single-neuron openers. There ‖E_i‖ ≈ λ₁(M_i) (median ratio 0.97–1.00):
    the gate direction is carried by other, co-firing neurons.
  * For multi-neuron sets the hypothesis holds sometimes. It holds for the sets the earlier anatomy checks found, which
    contain the co-firing neurons. It holds less often for greedy sets.
  * Where it holds, the bound 2k (4–10) is true but loose: the observed count is 1.
* **The rank lemma does not explain the small count.** It is better described by the fingerprint covariance K. The
  channels have Var(sᵀPs) ≈ 0.3–0.95·λ₁(K), so they sit on the top few eigenvectors of K. In pretrained `gelu_1l`, K's
  top eigenvector is a rank-one matrix s*s*ᵀ (95%), and the one dominant channel *is* s* (|⟨ssᵀ, V₁⟩| = 0.97); 164
  openers open it.
* **d_mlp scaling at fixed d_model = 512.** The plan's prediction holds:
  * from d_mlp = 512 to 8192, the number of single-neuron openers that open a gate direction grows about 8× (71 → 557 at γ = 25);
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

* **Tightening Theorems 5 and 7 (§2, §3.9c).**
  * Theorem 7 is attained, and so is Theorem 5 given κ: neither can be improved from (d, p, γ) or (d, p, γ, κ).
  * Theorem 5's unconditional bound follows from the linear budget, and is d/γ times weaker than it.
  * Theorem 7′ needs a frame bound only for channels open at the same row, so superposed dictionaries gain nothing:
    m ≤ d/(pγ) if co-open channels are orthonormal.
  * Theorem 7* subtracts a matrix quantile Y₀ from every fingerprint. It is 1.5–2.5× tighter than Theorem 7 on the
    trained models.
  * Searched directly, with arbitrary events, the trained fingerprints admit only 9–13 orthonormal gates at γ = 25
    and 2–7 at γ = 50, all held-out replicated. Neuron and halfspace openers reach a third to all of them. The best
    bound is still 6–20× above that, because no sum of per-direction quantities sees whether the rows' top directions
    cluster.
* **The gated subspace (Theorem 8, §3.9d).**
  * A class-level theorem in the style of the computation-in-superposition literature. Suppose fingerprints are k
    coherent directions plus an incoherent remainder, for any metric, width, events or implementation. Then every gate
    above γ_N (about 2.5, independent of d) lies in the coherent subspace up to ε ≈ γ_×²/(γ − γ_N)², and there are at
    most about k/(1 − ε) gates.
  * Hypotheses about single rows cannot do this: rank-one rows attain d/(pγ).
  * Trained models belong to the tolerant version of the class with k ≈ 25 and a 5–9× more coherent remainder than
    isotropic. That V contains essentially all their gates, from d_mlp 32 to 8192.
  * The narrow MLP is in the class exactly, with N = 0 and m ≤ H.
  * Lean-checked core: `GateBoundLean/Containment.lean`.

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
A **gate direction at level γ** is an eigenpair (λ, s) of M with |λ| ≥ t := γ/d. Write N₊(M, t) = #{i : λ_i(M) ≥ t} and
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

If ‖E_S‖_op < t, then N₊(M, t) ≤ r and N₋(M, t) ≤ r. In particular the number of gate directions is at most 2r ≤ 2k.

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

Theorem 3′ is also the plan's "stated boundary", now a theorem: every gate direction beyond the first r is a gate direction of the
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
(d) per opener, gate directions of either sign: N·p(1−p)t² ≤ Σ_{i≤N} λ_i.

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

### Sharper budgets (Theorems 7′ and 7*), and how far Theorems 5 and 7 can be tightened

Write Δ = E[q | E] − E[q | Eᶜ] with q = sᵀPs, so a gate at level γ has Δ ≥ δ := γ/d. The results below sharpen
Theorem 7, show that Theorem 5 adds nothing to it unconditionally, and show that both are attained, so any further gain
has to use more of the fingerprint distribution than d, p, γ and κ.

**(a) What Theorem 7 drops.** Exactly, p·Δ = E[q] − E[q | Eᶜ]. Theorem 7's per-gate step p·Δ ≤ E[q] discards the
channel's off-mean, and nothing else. *Proof.* E[q] = pE[q|E] + (1−p)E[q|Eᶜ]. ∎

**(b) Theorem 7′ (co-open frame).** Let F_on(ρ) = Σ_{a : ρ ∈ E_a} s_a s_aᵀ, the channels open at row ρ. Then

  Σ_a p_a Δ_a ≤ E[⟨P_ρ, F_on(ρ)⟩] ≤ E[λ_max(F_on(ρ))] ≤ B_on := max_ρ λ_max(F_on(ρ)),

so m ≤ B_on·d/(p_min γ). Since F_on(ρ) ⪯ Σ_a s_a s_aᵀ, this is never weaker than Theorem 7, and it does not depend on
how many channels there are in all: if the channels open at any one row are orthonormal, m ≤ d/(p_min γ) for any
dictionary.

*Proof.* p·Δ ≤ p·E[q | E] = E[q 1_E], because E[q | Eᶜ] ≥ 0. Sum over a and exchange the sums:
Σ_a E[q_a 1_{E_a}] = E[Σ_{a open} s_aᵀP s_a] = E⟨P, F_on⟩. Finally ⟨P, F⟩ ≤ λ_max(F)·tr P for P ⪰ 0, and tr P = 1. ∎

This settles the superposition question of Theorem 5″. Parts (b) and (c) there allow up to about
4d²/(p(1−p)γ²) gates on a random dictionary. That is 4d/((1−p)γ) times more than d/(pγ), about 45× at d = 512, γ = 50.
Those gates cannot coexist unless many of them are open at the same rows along overlapping channels. Even plain
Theorem 7 excludes them. For m random unit channels, B′ → (1 + √(m/d))² (the Marchenko–Pastur edge), so
m·pγ/d ≤ (1 + √(m/d))², which for pγ > 1 forces m ≤ d/(√(pγ) − 1)² (asymptotically in d and m).

**(c) Theorem 7* (quantile shift).** For one gate, any constant y and any split q = y + a − b with a, b ≥ 0,

  p(1−p)·Δ ≤ (1−p)·E[a] + p·E[b],

with equality when y is the p-quantile of q, a = (q − y)₊, b = (q − y)₋ and E = {q > y}. (This is the CVaR formula
p(1−p)·max_{μ(E)=p} Δ = min_y [(1−p)E(q−y)₊ + pE(y−q)₊]. Taking y = 0, a = q is Theorem 7.) In matrix form, take any
symmetric Y₀ and positive Y⁺_ρ, Y⁻_ρ with P_ρ − Y₀ = Y⁺_ρ − Y⁻_ρ, and put y = sᵀY₀s, a = sᵀY⁺s, b = sᵀY⁻s. Then every
gate has p(1−p)Δ ≤ sᵀA s with A = E[(1−p)Y⁺ + pY⁻] ⪰ 0, and gates of prevalence p satisfy

  m·p(1−p)·δ ≤ B′·tr A,  and, as in Theorem 7,  m·p(1−p)·δ ≤ B′·Σ_{i≤m} λ_i(A).

The best split is Y^± = (P − Y₀)_±. Then tr A = β_p(Y₀) := E[(1−p) tr(P − Y₀)₊ + p tr(P − Y₀)₋]. Y₀ = 0 gives
A = (1−p)P̄, which is Theorem 7. A Y₀ near the rows' typical fingerprint (§3.9c uses Y₀ = θP̄) removes the part of each
channel's mean that all rows share. Theorem 7 counts that part as budget, but no event can spend it.

*Proof.* p(1−p)Δ = E[(q − y)(1_E − p)]. On E, (a − b)(1 − p) ≤ (1−p)a; off E, −p(a − b) ≤ pb. ∎

**(d) Theorem 5 adds nothing to the linear budget unconditionally.** Closing gates are opening gates of Eᶜ, so every
gate has min(p, 1−p)·|Δ| ≤ E[q]. Hence orthonormal gates with p(1−p) ≥ c satisfy m·c·δ ≤ 1, since
p(1−p) ≤ min(p, 1−p). Because 0 ≤ q ≤ 1 gives |Δ| ≤ 1, this implies Theorem 5's unconditional m·c·δ² ≤ 1 and improves
it by the factor 1/δ = d/γ (about 20 at γ = 25). Theorem 5 adds information only through a small κ (κ ≤ 1 − 1/d never
helps) or through the spectrum of K. In the trained models that happens only at γ = 100 (§3.9c).

**(e) Both theorems are attained.** Take r rows of weight 1/r and rj ≤ d orthonormal vectors b(ρ, k), k < j. Row ρ has
fingerprint P_ρ = (1/j)Σ_k b(ρ,k)b(ρ,k)ᵀ, and gate (ρ′, k) opens on row ρ′ along b(ρ′, k). Every gate has p = 1/r and
Δ = 1/j, so the m = rj gates have m·p·δ = 1: equality in Theorem 7 and in 7′ (B′ = B_on = 1). In contrast units this
is m = d/(pγ) with γ = d/j, for any p = 1/r and γ ≥ 1/p. The same construction makes Theorem 5 (a) and (c) equalities:
q_a is two-valued, so Var q_a = p(1−p)δ², and Σ_a Var q_a = κ = 1/j − 1/m. So Theorem 7 cannot be improved from
(d, p, γ), and Theorem 5 cannot be improved from (d, p, γ, κ). What is loose in Theorem 5 is only the relaxation κ ≤ 1.

**(f) Tried, small gain.** Var(sᵀPs) = ⟨ssᵀ, K ssᵀ⟩ only sees K on fully symmetric 4-tensors. So Theorem 5(b) holds
with K replaced by its full symmetrisation K̂(X) = ⅓E[⟨D, X⟩D + 2DXD], D = P − P̄ (tr K̂ = ⅔κ). On `gelu_1l` (one-off
800-step Lanczos, not in the scripts) this moves the spectral count from 437 to 357 at γ = 25 and from 58 to 47 at
γ = 50.

Lean: (a) `mass_mul_contrast_eq`; (b) `linear_budget_on`, `gate_count_on`; (c) `contrast_le_split`, `quantile_budget`,
`gate_count_quantile` (trace form; the spectral form needs von Neumann's inequality, as for Theorem 7); (d)
`linear_budget_gate_abs`, `gate_count_abs`, `contrast_abs_le_one`, `gate_count_orthonormal_linear`; (e)
`linear_budget_sharp` (the Theorem 7 half). All are in `GateBoundLean/Sharp.lean`.

### Theorem 8 (the gated subspace: containment, and the spiked–incoherent class)

The budgets above cannot be improved from (d, p, γ) alone, so a sharper bound needs a hypothesis. Hypotheses about
single rows do not help. A row with *rank-one* fingerprint vvᵀ, v = j^{-1/2}Σ_{a∈S_ρ} s_a, gives each of its j open
channels the share 1/j. Built from such rows, the construction of (e) attains d/(pγ). So no bound on the rank or
concentration of P_ρ reduces the count. What separates that construction from trained models is how rows *share*
directions. Theorem 8 makes this a class hypothesis, in the same spirit as the sparsity assumptions of work on
computation in superposition (Adler and Shavit, arXiv:2409.15318; Hänni et al., arXiv:2408.05451).

The setting is general: P_ρ is any PSD row operator. For a fingerprint, P_ρ = GᵀG/Z with any downstream network and
any PSD output metric. Events are arbitrary, and nothing is assumed about how a gate is implemented or how many neurons
carry it. Fix a subspace V of dimension k, with projectors Π, Π⊥. A *gate direction* at level t = γ/d is an eigenpair
M_E s = λs with λ ≥ t.

**(a) Containment.** Define the *outside capacity* τ⊥(V) = sup_E λ_max(Π⊥M_EΠ⊥) and the *cross capacity*
τ×(V) = sup_E ‖Π⊥M_EΠ‖. Then every gate direction with λ > τ⊥ has

  ((λ − τ⊥)² + τ×²)·‖Π⊥s‖² ≤ τ×²·‖s‖².

So strong gates are confined to V: the leaked share falls like τ×²/(λ − τ⊥)².

*Proof.* Write s = v + w with v ∈ V, w ⊥ V. Then λ‖w‖² = ⟨M s, w⟩ = ⟨M v, w⟩ + ⟨M w, w⟩ ≤ τ×‖v‖‖w‖ + τ⊥‖w‖², so
(λ − τ⊥)‖w‖ ≤ τ×‖v‖. Square this and use ‖s‖² = ‖v‖² + ‖w‖². ∎

**(b) Counting.** Unit channels with frame bound B, each with ‖Π⊥s_a‖² ≤ ε, satisfy m·(1 − ε) ≤ B·k.

*Proof.* With e_1…e_k an orthonormal basis of V, Σ_a‖Πs_a‖² = Σ_i Σ_a⟨e_i, s_a⟩² ≤ B·k, and each term on the left is
≥ 1 − ε. ∎

**(c) The spiked–incoherent class.** Suppose every row splits as P_ρ = S_ρ + N_ρ. The *coherent part* S_ρ ⪰ 0 is
supported in V. The *incoherent part* N_ρ = O_ρΛ_ρO_ρᵀ ⪰ 0 has an orientation O_ρ independent of (Λ_ρ, S_ρ) whose
fourth moments are those of Haar measure. Then for any event, any opener and any implementation:

  τ⊥ ≤ γ_N/d,  γ_N := √( (2E‖N‖²_F + Var tr N) / (p(1−p)) ),
  τ× ≤ γ_×/d,  γ_× := √( E‖N‖²_F / (p(1−p)) ) · d/√((d−1)(d+2)).

So for γ > γ_N every gate direction has ‖Π⊥s‖² ≤ ε = γ_×²/(γ_×² + (γ − γ_N)²), and m ≤ B·k/(1 − ε).

The constants do not depend on d, on the width, or on the orientation of anything. Because N ⪯ P,
E‖N‖²_F ≤ E‖P‖²_F ≤ 1 and Var tr N ≤ ¼, so γ_N ≤ √(2.25/(p(1−p))) (= 5 at p = 0.1) for *every* model in the class. With
the concentrated rows measured here (E‖P‖²_F ≈ 0.17), γ_N ≈ 2.5 and γ_× ≈ 1.4. At γ = 25 that gives ε ≈ 0.004.
**In this class, gates above γ_N lie in the coherent subspace, and there are at most about dim V of them.**

*Proof.* S lives in V, so Π⊥M_EΠ⊥ and Π⊥M_EΠ see only N. For a unit w ⊥ V, x = O_ρᵀw is uniform on the sphere. From
E x_i⁴ = 3/(d(d+2)) and E x_i²x_j² = 1/(d(d+2)):

  E[(wᵀNw)² | Λ] = ((tr Λ)² + 2‖Λ‖²_F)/(d(d+2)),  E[wᵀNw | Λ] = tr Λ/d,

hence Var(wᵀNw) ≤ (2E‖N‖²_F + Var tr N)/d². For orthonormal v ∈ V and w ⊥ V, the pair (Oᵀv, Oᵀw) has
E[x_i²y_i²] = 1/(d(d+2)) and E[x_iy_ix_jy_j] = −1/((d−1)d(d+2)) for i ≠ j. So vᵀNw has mean 0 and
E(vᵀNw)² = (d‖Λ‖²_F − (tr Λ)²)/((d−1)d(d+2)) ≤ ‖Λ‖²_F/((d−1)(d+2)). These bounds hold uniformly over directions, and
Theorem 5(a) (|Δ| ≤ sd/√(p(1−p)) for any event) turns them into the two capacities. Finish with (a) and (b). ∎

**(d) Tolerance.** Exact isotropy is not needed. The proof uses only the two variance bounds
Var(wᵀNw) ≤ σ⊥²‖w‖⁴ and Var(vᵀNw) ≤ σ×²‖v‖²‖w‖². If they hold with η⊥ and η× times the isotropic values (an
*η-incoherent* remainder), the constants become √η⊥·γ_N and √η×·γ_×. The class then has parameters (k, η⊥, η×, p) and
E‖N‖²_F, and nothing is fitted.

**Corollary 8.1 (only directions with concentrated variation gate).** Let N_c be the largest number of orthonormal
directions s whose own share has Var(sᵀPs) ≥ c, with c = p(1−p)(γ/d)². Every family of orthonormal gates at level γ
(prevalence p, any events, any implementation, opening or closing) satisfies m ≤ N_c. In the class of (c) and (d) this
combines with the count above to give m ≤ min(N_c, B·k/(1−ε)).

*Proof.* By Theorem 5(a) each gate's own channel quantity has variance ≥ p(1−p)Δ² ≥ c, and the gates are orthonormal. ∎

The corollary sharpens the count from the dimension of the coherent subspace to its directions of *concentrated*
variation. It excludes the coherent directions whose variation is spread thin (§3.9e). It adds rigor but little
explanation: that a gate needs variance concentrated in its own direction is close to the definition of a gate, so
"why about 10 gates" becomes "why about N_c concentrated directions". N_c is the maximum of a quartic form over
orthonormal families. The computable relaxations (Ky Fan on K or K̂) overcount, because they credit mixing and
off-diagonal variance. The values in §3.9d are greedy estimates, i.e. lower bounds on N_c, so they show what the bound
gives but do not certify it.

**What it explains, and what it does not.** It explains why the count does not grow with width, metric or
implementation: only coherent directions can carry gates, and sensitivity that is incoherent across rows cannot give
contrast beyond O(1). It does not explain how many of V's directions gate. Inside V, every statistic used here allows
all of them (§3.9d). Theorem 8 is a population statement. On a finite sample the variances carry sampling error,
which is why an isotropic remainder shows a cross capacity of 4–6 rather than 1.4 at N = 7680 (§3.9d).

Lean: (a) `containment`; (b) `card_le_of_contained`; Theorem 5(a) as a capacity bound, `abs_contrast_le_of_var`;
(c), (d) `spiked_containment` and `spiked_count`, with the two variance bounds as hypotheses (over any finite row
distribution); Corollary 8.1, `concentratedCount` and `card_le_concentratedCount`. All are in
`GateBoundLean/Containment.lean`. The Haar fourth-moment computation in (c) is not
formalized: Mathlib has no Haar moments.

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
* d_mlp-free bounds on gate directions per opener (Proposition 1(d), Theorem 5(d)) and on distinct channels in total
  (Theorem 5).
* Under ‖QMQ‖ < γ/d (implied by ‖E_S‖ < γ/d): at most r ≤ k gate directions of each sign per k-neuron opener.
* Sharper budgets (Theorems 7′, 7*), the domination of Theorem 5's unconditional bound by the linear budget, and a
  construction attaining Theorems 7 and 5 (given κ). So neither can be improved from (d, p, γ) or (d, p, γ, κ) alone.

**Not proved:**
* A computable bound close to the number of gates the trained fingerprints actually admit (§3.9c). Every bound here
  stays 6–20× above it at γ ≤ 50.
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

### 3.2–3.3 Residual test and gate-direction counts (plan steps 2–3)

`gelu_1l`, γ = 25 / 50 / 100 (γ/d = 0.049 / 0.098 / 0.195):

| openers | with ≥ 1 gate direction | N₊ max | N₋ max | Thm 3 holds (‖E_S‖ < γ/d) | Thm 3′ holds (‖QMQ‖ < γ/d) |
|---|---|---|---|---|---|
| single neurons (2048) | 155 / 8 / 0 | 2 / 1 / – | 0 | 5/155, 2/8 | 18/155, 6/8 |
| greedy k = 2–4 (24) | 24 / 24 / 8 | 1 | 0 | 9/24, 20/24, 24/24 | 22/24, 22/24, 24/24 |
| anatomy sets {251,1608}, {251,1608,1942,690}, {2008,…} | 3 / 2 / 0 | 1 | 0 | 3/3 | 3/3 |
| random sets (32) | 2 / 0 / 0 | 1 | 0 | – | – |

**Single neurons.** For the top 50 single-neuron openers, ‖E_i‖/λ₁(M_i) has median 1.00 and ‖QMQ‖/λ₁ has median 0.98:
the gate direction is essentially orthogonal to the opener's own read direction. The same holds in every model (median
0.97–1.00), so for k = 1 the plan's hypothesis fails.

**Sets that contain the co-firing neurons.** Example: S = {251, 1608}, the pair the anatomy check found firing
together.
* d·‖E_S‖ = 17 and d·‖QMQ‖ = 10, against d·λ₁ = 88. Theorem 3's hypothesis holds at γ = 25 and 50.
* 94% of the gate direction lies in span(w_S): the channel is read by the opener neurons themselves.
* K_S has one large positive eigenvalue (86) and one negative partner (−17). The negative partner comes from the
  x yᵀ + y xᵀ structure: for k = 1, w aᵀ + a wᵀ has eigenvalues (w·a) ± ‖w‖‖a‖. E_S cancels it, which is why M itself
  has no negative gate direction.
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

The run's attractors are M_o's top eigenvectors: on `gelu_1l` |cos(s, top gate direction)| is 0.98, 0.99 and 0.84. Their exact
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
Channel 0 is V₁(K): |⟨ssᵀ, V₁⟩| = 0.97 and Var(sᵀPs) = 0.95·λ₁(K). It is opened by 164 of the 189 (opener, direction) pairs
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
| neurons opening a gate direction, γ = 25 | 71 | 109 | 167 | 296 | 557 |
| distinct channels, γ = 25 | 13.5 | 9.0 | 6.0 | 5.5 | 5.0 |
| neurons opening a gate direction, γ = 50 | 20 | 22 | 34 | 62 | 114 |
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
per-opener structure: one positive gate direction, no negative gate directions, and ‖E_i‖ ≈ λ₁ for single neurons. Theorem 3's hypothesis
for greedy sets is the variable part. It holds for 0–7 of 24 at γ = 25 in the retrained models, against 9/24 in the
pretrained one.

### 3.9 Statistics (plan step 9)

**Held-out replication.** Gate-direction counts replicate: per-opener N₊ is identical on the fitting and held-out rows for all
anatomy, run-opener and greedy sets, except two greedy sets near threshold. The number of single neurons with a gate direction is 155 → 182 at γ = 25 and
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

### 3.9c How tight are Theorems 5 and 7? (`amp_bound_tightness.py`)

Theorems 5 and 7 bound gates over *arbitrary* events of prevalence p. For a fixed channel the best such event is the
top-p rows of q_s = sᵀPs, so the number of gates a model's fingerprints admit can be searched for directly. The search
is greedy over orthonormal channels: alternate E ← top-p rows of q_s and s ← top eigenvector of M_E compressed off the
channels already found, from 32 starts per channel. On held-out rows the channel's event is q_s above its
fitting-row threshold. The search gives a lower bound on the true maximum; the theorems give upper bounds.

p = 0.1, orthonormal channels (B = B′ = 1), counts at γ = 25 / 50 / 100. "Neuron" and "halfspace" count the same
channels when the opener is restricted: top 10% of one neuron's pre-activation, or of o·z with o the regression
direction of q_s on whitened z.

| model | admitted (held-out) | neuron | halfspace | Thm 7 spectral | **Thm 7*** spectral | Thm 5 spectral |
|---|---|---|---|---|---|---|
| gelu_1l | 9 / 2 / 1 (9 / 2 / 1) | 3 / 1 / 0 | 4 / 1 / 1 | 187 / 80 / 31 | 118 / 42 / 10 | 437 / 58 / 5 |
| gelu_2l L1 | 10 / 4 / 0 (9 / 4 / 0) | 5 / 1 / 0 | 7 / 3 / 0 | 191 / 84 / 34 | 121 / 45 / 13 | 543 / 74 / 5 |
| d_mlp 32 | 13 / 7 / 3 (13 / 7 / 3) | 7 / 3 / 0 | 12 / 7 / 3 | 204 / 102 / 51 | 82 / 41 / 17 | 600 / 145 / 30 |
| d_mlp 512 | 13 / 4 / 1 (13 / 4 / 1) | 5 / 0 / 0 | 9 / 3 / 0 | 197 / 88 / 36 | 117 / 46 / 14 | 531 / 92 / 10 |
| d_mlp 8192 | 12 / 4 / 1 (12 / 4 / 1) | 4 / 1 / 0 | 9 / 1 / 1 | 192 / 85 / 34 | 125 / 48 / 14 | 574 / 91 / 11 |
| d_mlp 2048, Muon | 11 / 4 / 1 (11 / 4 / 1) | 5 / 2 / 0 | 7 / 3 / 1 | 189 / 82 / 33 | 118 / 44 / 13 | 510 / 83 / 10 |
| d_mlp 2048, ReLU² | 11 / 4 / 1 (11 / 4 / 1) | 5 / 2 / 0 | 8 / 3 / 1 | 192 / 85 / 34 | 126 / 49 / 15 | 605 / 101 / 12 |

Theorem 7 without the spectrum is 205 / 102 / 51 in every row. Thm 7* uses Y₀ = θP̄ with the best of θ ∈ {0, 1, 2}.
Its trace β_p(θP̄) is 0.60–0.69, and 0.36 at d_mlp = 32, against Theorem 7's 0.9.

* **The opener class is not where the gap is.** With arbitrary events the fingerprints admit 9–13 orthonormal gates at
  γ = 25, 2–7 at γ = 50 and 0–3 at γ = 100. All of them replicate on held-out rows. Halfspace openers reach about
  half to all of them, and single neurons a third to a half. So the attractor counts of `downstream_amp.py` are within
  2–3× of everything the model's sensitivity structure allows at these contrasts, whatever the openers are.
* **Thm 7* is the tightest bound at γ ≤ 50**: 1.5–2.5× below Theorem 7 (the most at d_mlp = 32, whose rows share the
  most of P̄). At γ = 100 Theorem 5 spectral is the tightest (5–12), except at d_mlp = 32 (30 against 17). Theorem 5
  beats Thm 7* only there, as §2 (d) predicts.
* **What remains is 6–20× at γ ≤ 50.** Theorem 7's accounting on the admitted gates (γ = 25) shows where it goes:
  * Each gate spends only 24–31% of its channel's mean on its contrast. The rest is the off-mean (on/off ratio 4–6).
    This is the slack (a) identifies. Thm 7* recovers part of it.
  * The gates together use Σ p·Δ = 0.08–0.19 of the unit budget. Their on-rows give only 10–25% of their sensitivity
    to the channels open there (E⟨P, F_on⟩ = 0.10–0.25, about one open channel per row).
  * Rows' fingerprints are concentrated (effective rank 7–8, §3.1), but along directions that are not shared by 10%
    of rows. A gate needs one direction that carries ≥ γ/d of the mass on a tenth of the rows. All the bounds here are
    sums of per-direction or per-row quantities (means, quantile parts, variances, Ky Fan sums), and none of them sees
    whether the rows' top directions cluster. The construction in §2 (e) shows they cannot without more input: it
    meets them with equality.

### 3.9d Do trained fingerprints fit the spiked–incoherent class? (`amp_containment.py`)

Theorem 8 is a statement about a class, so measuring a model only says whether it belongs to the class and with which
parameters. For V = the top-k eigenvectors of R = E[(P − P̄)²], `amp_containment.py` measures:

* the outside and cross capacities, by search (alternating between the best event and the top eigen- or singular
  vector). Searches give lower bounds, so the counts derived from them are estimates, not certificates;
* the same capacities after Haar-rotating every row inside V⊥, which gives an isotropic remainder at this sample size,
  and the ratio η = (real / isotropic)²;
* the orthonormal gates a search finds inside V.

Columns for k = 25, in contrast units, p = 0.1:

| model | τ⊥: real / isotropic (η⊥) | τ×: real / isotropic (η×) | ε at γ = 25 / 50 / 100 | k/(1−ε), best k | gates inside V (k = 25) | gates admitted (§3.9c) |
|---|---|---|---|---|---|---|
| gelu_1l | 8.5 / 1.3 (45) | 14.8 / 4.6 (10) | 0.44 / 0.11 / 0.03 | 45 / 24 / 10 | 8 / 1 / 1 | 9 / 2 / 1 |
| gelu_2l L1 | 9.5 / 1.2 (61) | 15.7 / 5.7 (8) | 0.51 / 0.13 / 0.03 | 51 / 20 / 10 | 9 / 4 / 0 | 10 / 4 / 0 |
| d_mlp 32 | 8.1 / 0.2 (1185) | 11.4 / 1.4 (63) | 0.31 / 0.07 / 0.02 | 36 / 19 / 11 | 13 / 7 / 3 | 13 / 7 / 3 |
| d_mlp 512 | 8.9 / 1.0 (83) | 16.6 / 4.0 (17) | 0.52 / 0.14 / 0.03 | 52 / 26 / 10 | 12 / 4 / 0 | 13 / 4 / 1 |
| d_mlp 8192 | 9.3 / 1.2 (57) | 15.5 / 4.7 (11) | 0.49 / 0.13 / 0.03 | 49 / 22 / 10 | 10 / 3 / 1 | 12 / 4 / 1 |

"Best k" is the minimum over k ∈ {9, 15, 25, 40}. The best k is 25 at γ = 25, 9 or 15 at γ = 50, and 9 at γ = 100.

* **The gated subspace is real, and it is the same size in every model.** About 25 directions contain essentially
  every gate the fingerprints admit: the search inside V finds 8–13 of the 9–13 at γ = 25, and all of them at γ ≥ 50.
  Outside those directions the capacities (τ⊥ ≈ 8–9, τ× ≈ 11–17) sit well below γ = 25, so gates are confined
  (ε ≤ 0.5 at γ = 25, ≤ 0.14 at γ = 50, about 0.03 at γ = 100). The same holds from d_mlp 32 to 8192 and in the
  two-layer model.
* **The remainder is not isotropic, but it is η-incoherent with moderate η.** Its capacities are 5–9× (outside) and
  3–4× (cross) those of an isotropic remainder: η⊥ ≈ 45–85, η× ≈ 8–17. So trained models belong to the class of
  Theorem 8 (d) with k ≈ 25. They do not belong to the strict isotropic class (c).
* **The narrow MLP is in the class exactly.** With d_mlp = 32, every fingerprint W A Wᵀ lies in span(W), of dimension
  32. With V ⊇ span(W) the remainder is 0, so τ⊥ = τ× = 0, and Theorem 8 gives m ≤ H whatever the events. The width
  limit of §4 is the special case N = 0. (The isotropic-null η for d_mlp 32 is large because so little of the
  fingerprint lies outside the top 25 directions.)
* **The count k/(1−ε) is 3–5× above the admitted gates** (36–52 against 9–13 at γ = 25), because the theorem allows
  every direction of V to gate (§3.9c, "inside V"). Theorem 8 explains *where* gates live and why only coherent
  directions can carry them. It does not explain why about half of V's directions do not gate.
* **Corollary 8.1 closes most of the remaining gap.** Greedy estimates of N_c (`amp_containment.py --nc-only`), against
  the gates admitted and k/(1−ε), at γ = 25 / 50 / 100:

  | model | N_c (greedy, a lower bound) | gates admitted | k/(1−ε), best k |
  |---|---|---|---|
  | gelu_1l | 16 / 5 / 1 | 9 / 2 / 1 | 45 / 24 / 10 |
  | gelu_2l L1 | 19 / 7 / 1 | 10 / 4 / 0 | 51 / 20 / 10 |
  | d_mlp 32 | 15 / 9 / 5 | 13 / 7 / 3 | 36 / 19 / 11 |
  | d_mlp 512 | 16 / 8 / 1 | 13 / 4 / 1 | 52 / 26 / 10 |
  | d_mlp 8192 | 18 / 7 / 2 | 12 / 4 / 1 | 49 / 22 / 10 |

  At γ = 25 the estimate is 1.2–1.9× the admitted count, against 3–5× for the dimension bound. These are search
  estimates, not certificates.

### 3.9e Why only part of V gates (exploratory)

The directions of V that do not gate are not a separate kind of direction. One-off checks on gelu_1l and d_mlp 8192 (not
in the scripts) split V into the span G of the gates found inside it and its complement C, with dim C = 15–17:

* **Gates compete.** When a gate opens, 20–45% of the sensitivity it gains comes out of the other gates, and the gates'
  total share is anti-correlated with the rest (−0.88).
* **C's variation is spread thin.** Each direction of C has own-share amplitude of about 5 (gates: about 15), and its
  variance is below the cost c. The best *combination* of C's directions still reaches contrast 22–25, just under
  γ = 25.
* **Tilting of the gate channels is real but minor.** Directions that point mainly along a gate carry 24–39% of C's mass
  (2–9% for a random subspace of V of the same dimension). But they carry at most 17% of C's row-to-row variation.
  Each gate tilts into 10–11 of C's directions, not into private ones. 67–90% of C's variation comes from directions that
  point mainly into C itself.

So the non-gating part of V is mostly coherent structure below threshold, whose variation is spread over many
directions. Corollary 8.1 excludes it. Why training concentrates variation in about 10–15 directions and spreads the rest
is not explained.

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

5. **The theorems cannot explain the count either, and the opener class is not the reason.** With arbitrary events
   the fingerprints admit 9–13 gates at γ = 25 (§3.9c), against 82–205 from the sharpest bounds. Those bounds are
   attained by some distributions (§2 (e)), so closing the gap needs a statement about how the rows' sensitive
   directions cluster.

6. **Only coherent directions can carry gates (Theorem 8).** This is the explanatory statement the data support for
   a class of models, with nothing fitted. Trained models fit it with k ≈ 25 coherent directions. What it leaves open
   is why only about 10 of those 25 gate: that needs structure beyond means and variances, namely which directions
   share events.

**Open.** The doc's mechanism is "bounded-bandwidth modulation plus a conserved sensitivity budget". The second half is
what the data support: Proposition 1 and Theorem 5 are the operative bounds. A proof that the count is O(1) would need a
statement about K, namely that training concentrates Var(P) in a few rank-one directions. That is a property of trained
models, not of the architecture, and nothing here proves it.

## 4. Width limits: how many gates can an MLP of width H store?

The bounds in §2 are all set by d (and γ, p); none depends on H. This section asks whether width limits gate storage at
all, from two sides. On the achievability side, planted gates (`amp_gate_capacity.py`) measure how many gates a width-H
MLP *can* store. On the observation side, the d_mlp sweep is extended down to 32 ≪ d = 512 (`amp_mlp_width.py`,
`amp_gate_bound.py`).

### 4.1 Planted gates

**Teacher** (ℝ^d → ℝ^d, x ~ N(0, I)): f*(x) = Qx + α√d·Σ_a σ((o_a·x − t)/0.25)(s_a·x)u_a.
* Q is a random orthogonal matrix, which gives an isotropic baseline.
* The o_a, s_a and u_a are random, with s_a ⟂ o_a.
* Each gate is open on a share p of inputs.

**Student:** a GELU MLP of width H with no skip connection, trained by MSE on fresh samples.

**Measurement:** both are measured exactly as the LM is (§1, with the identity as readout).

**Stored:** a planted gate counts as stored if both
* the student's contrast is at least half the teacher's, and
* the student's top gate direction of M_a has |cos| ≥ 0.8 with s_a.

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
* At d = 32 storage stops at about 3d, whatever the width. There the *teacher's* own gate directions stop aligning with their
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
| neurons opening a gate direction (γ = 25) | 30 | 42 | 40 | 47 | 71 | 109 | 167 | 296 | 557 |
| distinct channels, γ = 25 | 13.5 | 12.0 | 12.5 | 13.5 | 13.5 | 9.0 | 6.0 | 5.5 | 5.0 |
| distinct channels, γ = 50 | 5.0 | 3.5 | 3.0 | 3.0 | 3.0 | 1.0 | 1.5 | 1.5 | 1.0 |
| distinct channels, γ = 100 | 3.0 | 2.0 | 1.5 | 0 | 0 | 0 | 0 | 0 | 0 |
| λ₁(K)/κ | 0.13 | 0.10 | 0.11 | 0.08 | 0.08 | 0.08 | 0.08 | 0.08 | 0.08 |

**Width saturates only at the narrow end.** At d_mlp = 32, 30 of the 32 neurons open a gate direction and there are about 13
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

`gate_bound_lean/` checks the following in Lean 4 / Mathlib, with no `sorry` and only the standard axioms:
* the link between gate directions (eigenvalues) and gate spaces;
* Lemma 2, Theorem 3 (both signs) and Theorem 3′, for the neuron model `M = Σᵢⱼ ΔAᵢⱼ wᵢwⱼᵀ`;
* Proposition 1;
* the fingerprint `P = GᵀG/‖G‖²_F` being positive with trace 1;
* Theorem 7 with its count `m ≤ B·d/(p_min γ)`;
* Theorem 5 (a) and (c), `κ ≤ 1`, and `κ ≤ 1 − 1/d` when every row has a nonzero gradient. Together these give the
  gate bound for actual fingerprints, for any model, data and width: with orthonormal channels, at most
  `d² / (p(1-p)·γ²)` gates (`gate_count_orthonormal`; informative only when this is below `d`), and with overlapping
  channels `m·p(1-p)γ²/d² ≤ B₀·κ`, where `B₀` is the frame bound on trace-zero matrices (Theorem 5″(a),
  `gate_count_traceless`);
* the width results: the inertia count, and the result that `m` bilinear gates need exactly `2m` neurons;
* the sharper budgets of §2 (`GateBoundLean/Sharp.lean`): Theorem 7′ (co-open frame), Theorem 7* in trace form,
  the two-sided budget with Theorem 5's unconditional bound as a `d/γ`-times-sharper corollary, and the construction
  attaining Theorem 7;
* Theorem 8 (`GateBoundLean/Containment.lean`): containment from outside and cross capacities, the count
  `m ≤ B·dim V/(1 - ε)`, and the spiked–incoherent class with its variance hypotheses (the Haar moments that supply them
  are proved on paper only).

`gate_bound_lean/README.md` maps each result to its Lean theorem and lists what is not formalized: Theorem 4, the
spectral forms (5 (b), 5′), the random-channel bounds 5″(b), (c) (like Theorem 4, they need random-matrix results that Mathlib lacks), and
Proposition 6 for general activations.
