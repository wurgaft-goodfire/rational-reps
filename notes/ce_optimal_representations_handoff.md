# Handoff: CE-optimal representations under a dimensionality budget

**Purpose.** This document specifies a replacement for the "optimal representation = SVD of the
log-conditional" claim in Section 2 of the *Resource Rational Analysis of Representations* draft,
extends it to the sparse/superposition setting of Section 5, and lays out an implementation plan.

**Status markers used throughout:**
`[DERIVED]` — follows from the math below, safe to build on.
`[HEURISTIC]` — plausible argument, needs numerical check before it goes in a paper.
`[VERIFY]` — citation or attribution stated from memory; confirm before citing.

---

## 0. Notation

| symbol | meaning | shape |
|---|---|---|
| $\theta$ | conditioning item (token, node, state) | index over $\mathcal{T}$ |
| $x$ | outcome | index over $\mathcal{X}$ |
| $L$ | log-conditional, $L[\theta,x] = \log p(x\mid\theta)$ | $\|\mathcal{T}\|\times\|\mathcal{X}\|$ |
| $\pi$ | marginal over items; $\Pi = \mathrm{diag}(\pi)$ | $\|\mathcal{T}\|$ |
| $q$ | marginal over outcomes, $q_x=\sum_\theta \pi_\theta p(x\mid\theta)$; $Q=\mathrm{diag}(q)$ | $\|\mathcal{X}\|$ |
| $F_\theta$ | Fisher / softmax Hessian at row $\theta$ | $\|\mathcal{X}\|\times\|\mathcal{X}\|$ |
| $E, D$ | embedding, decoder (dense case) | $\|\mathcal{T}\|\times d$, $d\times\|\mathcal{X}\|$ |
| $d$ | dimensionality budget | scalar |
| $m$ | number of features (sparse case) | scalar |
| $z\in\{0,1\}^m$ | $k$-sparse feature indicator | $m$ |
| $A$ | feature representation matrix, columns $a_i$ | $d\times m$ |
| $C$ | co-occurrence, $C=\mathbb{E}[zz^\top]$ | $m\times m$ |
| $\beta$ | inverse temperature on the target logits | scalar |

Sparse-case convention follows Garg–Kleinberg–Peng (GKP) so the two papers line up:
code for support $S$ is $e_S = Az$, logits are $e_S^\top D$.

---

## 1. The core correction: cross-entropy is not the Frobenius norm

The draft currently says: *the optimal linear representation is the best rank-$d$ factorization of
the log-conditional tensor*, justified by Eckart–Young. Eckart–Young is optimal for **unweighted
squared error on $L$**, which is not the loss anyone trains on.

Expand the actual loss. With model logits $\hat L = ED$ and residual
$\delta_\theta = \hat L[\theta,\cdot] - L[\theta,\cdot]$: `[DERIVED]`

$$
\mathrm{KL}\!\left(p_\theta \,\|\, \hat q_\theta\right)
= \tfrac{1}{2}\,\delta_\theta^\top F_\theta\, \delta_\theta + O(\delta^3),
\qquad
F_\theta = \mathrm{diag}(p_\theta) - p_\theta p_\theta^\top .
$$

So the true objective at budget $d$ is

$$
\boxed{\;\min_{\mathrm{rank}(\hat L)\le d}\ \sum_\theta \pi_\theta\,
\delta_\theta^\top F_\theta \delta_\theta \;}
$$

Three consequences, each of which changes the pipeline:

**(a) Gauge.** $F_\theta \mathbf{1} = 0$. Softmax only sees logit *differences*, so $L$ is fitted
modulo an arbitrary per-row constant. Equivalently $\delta^\top F_\theta \delta =
\mathrm{Var}_{x\sim p_\theta}\!\left(\delta_\theta(x)\right)$ — the variance form *is* the gauge fix.
Any pipeline that does not quotient this out spends a singular direction on nothing.

**(b) Weighting is the joint.** Dropping the rank-one term of $F_\theta$, the weight on entry
$(\theta,x)$ is $\pi_\theta\, p(x\mid\theta)$ — the **joint distribution**. CE weights each
log-probability error by how often that pair actually occurs. Unweighted SVD does the opposite: the
log stretches the tail, so Eckart–Young spends capacity fitting the log-probs of events that never
happen. On a chain under graph diffusion this over-serves far-apart node pairs, and should visibly
distort the horseshoe (Fig. 2).

**(c) Rank-one weights are the only closed-form case.** A weight matrix $W_{\theta x} = w_\theta v_x$
can be absorbed by rescaling, so SVD applies. The true Fisher weight $\pi_\theta p(x\mid\theta)$ is
full rank. General weighted low-rank approximation is NP-hard `[VERIFY: Gillis & Glineur 2011]`.
**The entire ladder below is a sequence of increasingly good rank-one approximations to the true
weight, then giving up on closed form.**

Note the discrepancy between the surrogate and the truth is exactly
$\pi_\theta p(x\mid\theta) = \pi_\theta q_x \, e^{\mathrm{PMI}(\theta,x)}$, so PMI controls *how
wrong* the closed-form rungs are.

---

## 2. PMI is the CE-optimal centering (not an analogy — an identity)

Fit the rank-0 model $\hat q(x\mid\theta) = \mathrm{softmax}(b)_x$, constant across $\theta$: `[DERIVED]`

$$
\mathbb{E}_\theta[\mathrm{CE}] = -\sum_x \Big(\sum_\theta \pi_\theta p(x\mid\theta)\Big)\log\hat q(x)
= -\sum_x q_x \log \hat q(x)
$$

minimized at $\hat q = q$, i.e. $b = \log q$. The residual left for the rank-$d$ part to explain is

$$
\log p(x\mid\theta) - \log q(x) = \mathrm{PMI}(\theta, x).
$$

**PMI is precisely what remains after the CE-optimal rank-0 fit.** This means Levy & Goldberg's
target is *derived from* the draft's own framework rather than borrowed by analogy — a framing
upgrade for Section 2.

Two riders:
- At readout you must add $\log q$ back. Factorizing PMI models the ratio $p(x\mid\theta)/q(x)$.
- $\log q$ is exactly optimal at $d=0$ and only near-optimal at $d>0$. A jointly fitted free bias
  beats it slightly; prefer fitting $L \approx \mathbf{1}c^\top + \mathbf{b}\mathbf{1}^\top + ED$
  with $c$ free.

PMI therefore appears twice: as the **target** and as the log of the **weight correction**. Worth
saying explicitly in the paper.

---

## 3. The ladder

| rung | target | metric | rows | solve | cost |
|---|---|---|---|---|---|
| 1 | raw $L$ | none | items | closed form | trivial |
| 2 | PMI | none | items | closed form | trivial |
| 3 | PMI | rank-1, linearized at independence | items | closed form | trivial |
| 3′ | PMI | rank-1, weight $C$ | supports | closed form | trivial |
| 4 | — | self-consistent Fisher | items | alternating convex | seconds |
| 4′ | — | self-consistent Fisher | supports | alternating convex | seconds–minutes |
| 5 | — | — | — | SAE, non-convex, $\ell_1$, overcomplete | **not needed** |

Rung 1 is what the draft currently does. **Rung 3 is the recommended default for all arity-1
figures. Rung 4 is the check. Rung 4′ is Section 5. Rung 5 is never required.**

Critically: **rung 4 is rung 3 iterated with self-consistent weights.** Each Newton step of a
multinomial logistic fit is a Fisher-reweighted least squares (IRLS). So this is one algorithm with
two stopping points, not two methods. Say that in the paper; it makes the ladder a single story.

---

## 4. Implementation, rung by rung

### 4.1 Rung 2 — PMI-SVD

```
PMI = log p(x|theta) - log q(x)
U, S, Vt = svd(PMI)
E = U[:, :d] @ diag(sqrt(S[:d]))
D = diag(sqrt(S[:d])) @ Vt[:d, :]
logits = E @ D + log q            # <-- do not forget the + log q
```

Strictly dominates rung 1 at equal $d$: $\log q$ carries a large singular value and no relational
content.

### 4.2 Rung 3 — marginal-weighted PMI-SVD (correspondence analysis in the log domain)

```
M      = Pi^(1/2) @ PMI @ Q^(1/2)          # whiten
U,S,Vt = svd(M); truncate to d
E      = Pi^(-1/2) @ U[:, :d] @ diag(sqrt(S[:d]))     # unwhiten  <-- REQUIRED
D      = diag(sqrt(S[:d])) @ Vt[:d, :] @ Q^(-1/2)     # unwhiten  <-- REQUIRED
logits = E @ D + log q
```

**Truncation happens in whitened space (correct); the geometry you plot lives in unwhitened space.**
On a chain these differ by $\sqrt{\text{degree}}$ and endpoints shrink. Plotting $\tilde U$ instead
of $\Pi^{-1/2}\tilde U$ is a silent bug.

Also: the balanced $\Sigma^{1/2}$ split (draft footnote 1) minimizes the L2 norm in the *unweighted*
metric. Re-derive it in the weighted metric or restate the footnote.

### 4.3 Rung 3′ — sparse supports, still closed form

Under log-linear composition — which is exactly Eq. (2) of the draft, the composition a linear code
natively implements — the target for support $S$ is $\sum_{i\in S}\ell_i$, so residuals add:
$\delta_S = \sum_{i\in S}\delta_i$ with $R = A^\top D - L$, $L\in\mathbb{R}^{m\times|\mathcal{X}|}$
the per-feature log-conditionals. Then `[DERIVED]`

$$
\sum_S \pi(S)\, \delta_S^\top \bar F \delta_S
= \operatorname{tr}\!\left(R^\top C R \bar F\right)
= \left\| C^{1/2}\left(A^\top D - L\right)\bar F^{1/2}\right\|_F^2,
\qquad C=\mathbb{E}[zz^\top].
$$

**$C$ is the co-occurrence matrix — already the central object of the paper — sitting in exactly the
slot $\Pi$ occupied in the dense case.** Two-sided fixed weights, so whiten / truncate / unwhiten
applies verbatim. No enumeration of $\binom{m}{k}$ supports; everything stays $m\times|\mathcal{X}|$.

For uniform random $k$-subsets, $C = \frac{k}{m}\left[I + \rho(\mathbf{1}\mathbf{1}^\top - I)\right]$
with $\rho = \frac{k-1}{m-1}$, giving the interpretable form

$$
\mathbb{E}[\mathrm{CE}] \;\propto\; \underbrace{\textstyle\sum_i \delta_i^\top \bar F \delta_i}_{\text{reconstruction}}
\;+\; \rho \underbrace{\textstyle\sum_{i\neq j} \delta_i^\top \bar F \delta_j}_{\text{cross-residual interference}}.
$$

Fisher-weighted reconstruction plus a **sparsity-scaled interference penalty**. The second term is
the $\langle b_i, a_j\rangle$ interference GKP control; its coefficient is the sparsity level.

$\bar F$ is a fixed surrogate Fisher — use $\mathrm{diag}(q) - qq^\top$. Using a *fixed* $\bar F$ is
the linearization; making it self-consistent is rung 4′.

**Fig. 11 rows 1–3 collapse to rank arithmetic on $C^{1/2}L\bar F^{1/2}$, no training:**

| Fig. 11 case | condition | outcome |
|---|---|---|
| disjoint supports | $C$ block-diagonal | row spaces orthogonal → ranks add |
| shared support, independent structures | orthogonal row spaces | ranks add (torus) |
| shared support, aligned relations | identical row spaces | rank saturates (one circle) |
| over capacity | see §5 | closed form invalid |

And capacity becomes a **number**: $d^*(\epsilon)$ = count of singular values of
$C^{1/2}L\bar F^{1/2}$ above the CE tolerance. The phase boundary is a spectrum crossing you can
plot rather than a qualitative claim.

### 4.4 Rung 4 — exponential-family PCA / Goodman RC

No closed form. The loss is convex in $\Theta = ED$ but $\mathrm{rank}(\Theta)\le d$ is not a convex
constraint. Gaussian is the sole exception (identity link ⇒ natural = mean parameters ⇒
Eckart–Young).

Alternating fits, both halves convex:

```
init E, D from rung 3
repeat:
    D, b  <- multinomial logistic regression, design E, targets p(x|theta), weights pi   # convex
    E     <- multinomial logistic regression, design D^T, weights pi                      # convex
until CE plateaus
```

Each half-step is convex ∘ affine, so Newton/IRLS converges fast and descent is monotone.

**Two structural facts that matter for the paper, not just the code:**

- **Solutions are not nested.** The optimal rank-$d$ subspace need not sit inside the optimal
  rank-$(d+1)$ one `[VERIFY: Srebro & Jaakkola 2003, weighted case]`. Under SVD, "budget $d$" means
  *truncate*; under CE it means *re-solve*, and the geometry can genuinely reorganize as $d$ changes.
  **For a paper about representations under a resource budget, this is a result, not a nuisance.**
  Sweep $d$ on the chain and on the hierarchy and show the restructuring.
- **No canonical basis.** $ED$ is defined only up to $GL(d)$; nothing orders the frame. After
  fitting, whiten and re-SVD the fitted logits in the Fisher metric to get a canonical ordering —
  otherwise cross-architecture comparisons (the convergence claim) compare gauge choices.

**Free certificate:** replace the rank constraint with a nuclear-norm penalty and the problem becomes
convex, hence globally solvable, with a penalty path that sweeps effective rank. Use it to confirm
ALS isn't stuck at a bad stationary point.

### 4.5 Rung 4′ — superposition, and it is *not* an SAE

Sample $N$ supports **once** into a fixed $Z\in\mathbb{R}^{N\times m}$. Logits are
$ZA^\top D + \mathbf{1}b^\top$.

- **$D$-step** ($A$ fixed): design $ZA^\top \in \mathbb{R}^{N\times d}$, logits affine in $D$.
  Multinomial logistic regression, $d|\mathcal{X}| + |\mathcal{X}|$ params. **Convex.**
- **$A$-step** ($D$ fixed): $\mathrm{vec}(\text{logits}) = (D^\top \otimes Z)\,\mathrm{vec}(A^\top)$,
  affine in $A$. **Convex**, $md$ params.

Same algorithm as rung 4, with $C$ in place of $\Pi$ and a sampled design in place of enumerated
rows. **The only thing added for Section 5 is sampling supports.**

Explicitly *not*: no overcomplete dictionary, no $\ell_1$ coefficient, no trained base model to
attach to, no SGD. For $m=100, d=10$: ~1000 variables, sub-second Newton solves.

**Sparse coding keeps its role as the *analysis* step** — recovering family blocks from a fitted $E$,
as in Fig. 9 — rather than as the objective. Worth stating that division explicitly in the text,
since it's easy to misread the current draft as proposing SAEs.

---

## 5. Superposition: why the closed-form rungs cannot show it, and what does

Under a quadratic loss with linear readout, the optimum at budget $d$ **is** the truncated weighted
SVD. Sparsity buys nothing beyond changing the metric. So superposition is definitionally what
happens when the quadratic approximation stops being valid. **The ladder's closed-form rungs end
exactly where Section 5 begins** — which is a feature: it tells you where the phase boundary is.

**Mechanism.** $F_\theta = \mathrm{diag}(p_\theta) - p_\theta p_\theta^\top \to 0$ as $p_\theta$
concentrates. Near-deterministic conditionals make logit error *free* until it flips the argmax —
the margin regime — and that is what permits non-orthogonal packing. Elhage et al. get the same
effect from ReLU clipping negative interference; strip the ReLU and you are back to PCA.
**The general requirement is a loss nonlinearity that makes interference on inactive features free.
Here it is softmax saturation, not the readout.** `[DERIVED]`

This sharpens the draft's uncertainty bullet: superposition does **not** require a nonlinear readout
in this setting. The softmax is the nonlinearity; you never need to recover $z$, only the logits, and
those are linear in the code by construction. GKP Corollary 13 (bias + monotone activation buys
nothing asymptotically) confirms nothing is left on the table.

**Falsifiable prediction the existing task set already spans:** `[HEURISTIC]`

> Superposition in **low-entropy** tasks (modular arithmetic, lattices, max, entailment);
> graceful mode-dropping in **high-entropy** tasks (graph diffusion, smooth families).

**The control variable is temperature.** Sweep $\beta$ on $\beta L$ for a *single* task:
$\beta\to 0$ gives pure truncation, $\beta\to\infty$ gives packing. One knob, one figure, the whole
phase boundary. (This is the same $\beta$ that is currently an implicit, unstated consequence of
whatever smoothing keeps $\log p$ finite — see §7.)

---

## 6. Diagnostics

**Capacity detector via warm start.** Initialize rung 4′ at the closed-form rung-3′ solution, run ALS.

- *Below capacity:* the fit refines but does not reorganize; the row space returns to where it started.
- *Above capacity:* the subspace genuinely moves, because packing is not a truncation.

Measure with **principal angles between initial and final row spaces**, plus the CE gap. That is a
scalar phase-boundary readout — *the deviation from closed form is the superposition signal.* Sweep
$\beta$ and you get a curve instead of a pentagon that has to be eyeballed.

**Reporting standard for every figure.** Report achieved CE (nats/token) for each rung at the same
$d$, alongside the geometry. If the geometry doesn't move between rungs, that is one honest sentence
in the paper. If it moves, that is a figure. Either way the claim is grounded.

**Suggested metric set:** CE at matched $d$; principal angles between subspaces; Procrustes distance
between embeddings; spectrum of $C^{1/2}L\bar F^{1/2}$; max $|\langle b_i, a_j\rangle|$ off-diagonal
coherence.

---

## 7. Pitfalls (order-of-operations bugs that silently change the geometry)

1. **Centering before scaling.** If you center rows with a flat mean and then scale columns by
   $\sqrt{q_x}$, the gauge direction becomes $\sqrt{q}$ in the new coordinates and rows are no longer
   orthogonal to it — SVD then spends a singular vector on a direction that is free under softmax.
   Fisher-correct row centering is by the row's own conditional mean:
   $\tilde L[\theta,x] = \log p(x\mid\theta) + H(p_\theta)$. **Simplest fix: fit a free bias and skip
   centering entirely.**
2. **Plotting the whitened embedding.** Plot $\Pi^{-1/2}\tilde E$, not $\tilde E$. See §4.2.
3. **Unstated smoothing = unstated temperature.** Whatever $\epsilon$-smoothing keeps $\log p$ finite
   on deterministic tasks is *setting the weight matrix and therefore the geometry.* State it as a
   modelling parameter, not an optimizer detail. Better: make it $\beta$ and sweep it (§5).
4. **Ridge in the margin regime.** In the near-deterministic regime the logistic subproblems become
   separable and weights diverge. A ridge on $A, D$ is required for finiteness. **This is the same
   weight decay that in Fig. 13 decides which frequency is learned** — it is doing modelling work, so
   report its scale as a parameter of the account. Same status as temperature.
5. **Balanced split.** Footnote 1's min-norm argument is stated in the unweighted metric; re-derive
   or re-scope it.

---

## 8. Experiments, mapped to draft figures

| # | Figure | Task | Action |
|---|---|---|---|
| E1 | Fig. 2 (chain) | rung 1 vs 2 vs 3 at matched $d$ | expected to move; degree-heterogeneous |
| E2 | Fig. 2 (ring) | same | expected **not** to move; vertex-transitive, uniform marginals |
| E3 | Fig. 3 (grid, SBM) | same | boundary/degree effects should appear |
| E4 | Fig. 6/7 (tree, hierarchy) | rung 3 vs 4 | strongly heterogeneous; check simplex-of-simplices survives |
| E5 | new | sweep $d$ on chain + hierarchy, rung 4 | **non-nestedness**: show reorganization, not truncation |
| E6 | Fig. 11 rows 1–3 | rung 3′ spectrum of $C^{1/2}L\bar F^{1/2}$ | replace qualitative claim with $d^*(\epsilon)$ |
| E7 | Fig. 11 row 4 | rung 3′ init → 4′ ALS, principal angles | capacity detector |
| E8 | new | $\beta$-sweep on one task | truncation → packing phase curve |
| E9 | Fig. 13 | modular addition, rung 4 with explicit ridge | frequency selection as a function of ridge scale |
| E10 | Fig. 9/10 | stack vs share under Fisher weighting | check saturation story is not an artifact of the unweighted metric |

**Where rungs 1–3 are safe vs. not:**

- **Safe:** vertex-transitive graphs — ring, torus, hypercube, mod-$n$. Uniform marginals mean the
  weighting is the identity, and symmetry pins the basis anyway.
- **Degraded:** degree heterogeneity — chain endpoints, grid boundary, SBM, tree/hierarchy.
  Marginals catch the leading correction; the residual $\theta$-dependence shapes the boundary,
  which is the interesting part.
- **Invalid:** near-deterministic conditionals — modular arithmetic, max, lattices. $F_\theta\to 0$,
  the quadratic expansion degenerates, CE becomes margin-like. **Do not patch the weights; go
  straight to rung 4.**

**Group-structured special case.** For ring / torus / hypercube / mod-$n$, $L$ and $W$ are both
$G$-equivariant, so Schur's lemma pins the basis to the irreps. What the budget decides is *which
modes survive at rank $d$*, and because the weight couples discarded modes this is a small explicit
subset-selection over a known quadratic cost rather than plain water-filling. **Semi-closed form —
and it gives a principled version of the Fig. 13 frequency-selection story that does not rest
entirely on margin-vs-weight-decay.** This is also the minimal-representation-theory route the draft
asks for in its uncertainties list. `[HEURISTIC — worth working out properly]`

---

## 9. Relation to Garg–Kleinberg–Peng (the uploaded published paper)

**Two capacities, differing by roughly a factor of $k$.**

GKP ask: how many neurons to make each feature *individually recoverable by a linear probe*? Answer
$d = \Omega_\epsilon\!\left(\frac{k^2}{\log k}\log\frac{m}{k}\right)$, upper bound
$O_\epsilon(k^2\log m)$.

Prediction never asks for that. You need the **logits** right, not $z$ back. So:

- read out for **prediction** → pay rank;
- read out for the **next layer** → pay $k^2$.

Their Proposition 9 is the same point from the other side: representation and probe vectors need not
align, and nothing in the factorization $E\times D$ forces them to either.

**A tension worth exploiting.** `[HEURISTIC — check numerically before claiming]`

GKP's bound is **worst-case over supports in $\ell_\infty$**. Their lower-bound proof explicitly
pigeonholes for *same-sign* interference so that $k$ terms of size $\epsilon/k$ compound linearly.
Under a data distribution with random supports and an *average CE* criterion, interference is a sum
of $\sim k$ random-sign terms and concentrates at $\sqrt{k}\mu$, not $k\mu$. Setting that to
$\epsilon$ gives $\mu\sim\epsilon/\sqrt{k}$ and

$$
d \sim \frac{k}{\epsilon^2}\log m \quad\text{— linear in } k,
$$

i.e. the quadratic penalty for linear accessibility largely evaporates in the average case. Note the
cross-residual terms in §4.3 can be **negative**: residuals can be arranged to cancel, a degree of
freedom that $\ell_\infty$ discards. If this holds numerically it is a genuine contribution and a
clean point of contact with a published result.

**Suggested check:** construct $A$ from a scaled Rademacher/Gaussian incoherent matrix at
$\mu=\epsilon/\sqrt{k}$, sample random $k$-supports, measure the empirical distribution of
$\max_i|\hat z_i - z_i|$ and of the CE. Compare $d$ required for average-CE tolerance vs. worst-case
$\ell_\infty$ tolerance as $k$ grows.

---

## 10. Naming and literature

**All of rungs 1–4 already exist in the contingency-table association-model literature, largely
statistics in the 1980s.** Rung 4 is Goodman's **RC(M) association model** `[VERIFY]`:

$$
\log p_{ij} = \mu + \alpha_i + \beta_j + \sum_{k=1}^{d}\sigma_k u_{ik}v_{jk}
$$

Free bias, log-multiplicative low-rank interaction, ML fit by alternating algorithms because there is
no closed form. That is rung 4 exactly, including the bias needed for the gauge. And the classical
result that **correspondence analysis approximates RC under near-independence** is precisely the
rung 3 → rung 4 step, already worked out.

| rung | names |
|---|---|
| 2 | PMI-SVD; LSA `[Levy & Goldberg 2014 — already cited]` |
| 3 | correspondence analysis; RC linearized at independence `[VERIFY]` |
| 4 | Goodman RC(M) `[VERIFY]`; exponential-family PCA `[VERIFY: Collins, Dasgupta & Schapire 2001]`; GLM-PCA; generalized low-rank models `[VERIFY: Udell et al. 2016]` |
| 4′ | **gap in the literature — the contribution** |

**Three payoffs beyond citation hygiene:**

1. **It names the moment-vs-natural-parameter uncertainty in the draft's open-questions list.**
   Low rank in the **mean** parameter is pLSA / NMF / multinomial PCA. Low rank in the **natural**
   parameter is RC / exponential-family PCA. Two literatures that largely do not cite each other.
   This framework is the latter — softmax makes it so — and that can now be said in one line with
   references rather than treated as a weeds-problem to avoid.

   Related: for fixed $D$ the model class is e-flat; stationarity is $(\hat P - P)D^\top = 0$ and
   $E^\top(\hat P - P) = 0$, i.e. residual orthogonality in **moment** space. The Fisher information
   is the Jacobian between the two coordinate systems. All the non-convexity lives in choosing the
   subspace.

2. **It explains Figures 2–3 mechanistically.** Marginal weighting on a symmetric diffusion kernel
   *is* the normalized graph Laplacian $D^{-1/2}WD^{-1/2}$ — so rung 3 on the graph tasks is
   Laplacian eigenmaps / diffusion maps `[VERIFY: Belkin & Niyogi; Coifman & Lafon]`. Ring → Fourier
   and $k$-NN sphere → degree-1 spherical harmonics **because those are the Laplacian eigenfunctions
   on those manifolds.** Currently presented as observations; this makes them derivations.

3. **It connects to GloVe.** GloVe fits $\log X_{ij}\approx w_i\cdot\tilde w_j + b_i + \tilde b_j$
   with a hand-tuned weighting $f(X_{ij})$. The Fisher derivation says that weighting function is a
   crude stand-in for $\pi_\theta p(x\mid\theta)$ — giving a principled account of two of the field's
   most-used embedding methods as adjacent rungs on one ladder.

**Also relevant:** the "softmax bottleneck" `[VERIFY: Yang et al. 2018]` is the LM-side name for
low-rank log-probability matrices — directly the object being factorized here.

**Differentiation** (addressing the draft's own worry about novelty vs. Wyart et al., semantic
cognition, and toy models of superposition): the contribution is not "we unified these." It is
(i) the CE-correct objective and the resulting ladder, (ii) rung 4′ — RC models were never posed over
sampled multi-hot supports, so nobody derived a superposition threshold from them, and (iii) the
entropy-based prediction of *when* superposition occurs. Elhage et al. **assume** sparsity and a
nonlinearity and characterize the resulting geometry; this predicts the regime from the generating
process. Sitting on a 40-year-old base makes the new part sharper, not smaller.

---

## 11. Open questions and what to verify

1. **Numerically test the $\sqrt{k}$ vs $k$ interference claim (§9).** Highest-value open item.
2. **Verify every citation in §10 by year and title.** Several are from memory.
3. **Work out the group-equivariant subset-selection properly (§8).** Could give a closed-form
   frequency-selection result for Fig. 13.
4. **How much does rung 3 → rung 4 actually move the geometry?** If the answer is "barely, except on
   trees and near-deterministic tasks," that is a clean, publishable, honest simplification.
5. **Arity-2.** The Fisher weighting argument carries over — the loss is still softmax CE, the weight
   is still the joint $\pi(a,b)p(c\mid a,b)$ — but CP decomposition has no Eckart–Young even
   unweighted, so rungs 1–3 have no closed-form analogue. Rung 4's alternating structure *does*
   generalize (alternate over the three factors; each step convex). **This is probably the cleanest
   path through the arity-2 section with minimal representation theory.**
6. **Non-nestedness (§4.4): confirm empirically**, then decide whether it becomes a headline result
   about resource-constrained representation rather than a technical footnote.
