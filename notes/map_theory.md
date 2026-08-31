# Superposition in a rank-limited softmax: the theory

For merging into `superposition.tex`. Setting: $m$ binary features
$z_i \sim \mathrm{Bernoulli}(p)$ iid, target $p_\star(x{=}i \mid z) =
\mathrm{softmax}_i(\alpha_i z_i)$, additive rank-$d$ model $\ell_k(z) = \sum_j z_j R_{jk}
+ b_k$. Standard instance: $m = 64$, $\alpha_i = 8(i{+}1)^{-1/2}$, $p = 0.3$,
$d \in \{3, 6, 12\}$. Every claim below is validated against exact oracles; accuracy
numbers quote $d = 3/6/12$.

## Two observables

Any solution supports two measurements that coincide only when reader and writer vectors
are tied per feature:

- **The map** $R$: regress row-centred logits on $z$ (`geometry.logit_map`).
  Gauge-invariant; what the model computes.
- **The representation**: balanced-SVD the logits, regress the code on $z$, read
  $\|E_i\|^2$ (`geometry.logit_features`). The weights gradient descent learns — the
  balance gauge is the implicit bias.

Optimal solutions tie directions but split norms (reader $= \kappa_i \times$ writer,
$\kappa$ rising from 1 at the head to $\approx 3$ at the threshold), so the two
observables diverge: the measured "amplification" ($\|E_1\|^2 \approx 11.6$ vs
$\alpha_1 = 8$) is a property of the balanced factorization of an asymmetric map, not of
the map. The tied-direction class is CE-equivalent to the free class (excess within
0.0003 at every $d$); forcing $\kappa \equiv 1$ changes the optimal map, not just the
gauge (+0.015/+0.003/+0.001 excess).

## The theory, in four claims

**1. The anchor law.** $\mathrm{diag}(R) = \alpha_i(1 - 1/m)$ exactly on the represented
set. Derivation: the bias and reader stationarity conditions match the on/off mean
probabilities of each output; inverting the logistic under noise pins the mean logit
displacement up to a variance-mismatch term, and the couplings make the model's noise
track the target's, so the correction is second order. Verified to $< 1\%$ away from the
representation threshold. The head's extra measured length is an anti-aligned
interference budget that cancels out of the response; the identity
$G_{ii} + p S_i = \alpha_i$ is this law's measurable shadow. Corollary: negative
couplings are rank-$d$ projection residue under the CE metric (the centred target map is
exactly diagonal); they concentrate on the threshold-adjacent features — interference is
priced onto the channel with the least Fisher weight to lose.

**2. The allocation formula.** Dimensional capacity $h_i$ (leverage: hat-matrix
diagonal of the feature vectors, invariant to any code reparametrization,
`geometry.leverage`) solves water-filling:

- calibrate biases: $\mathbb E[q_i] = \mathbb E[p_{\star i}]$;
- equalize pressure: $g_i(h_i) = \nu$ on the represented set, where
  $g_i = \alpha_i\, p\,(\mathbb E[p_{\star i}|z_i{=}1] - \mathbb E[q_i|z_i{=}1])$;
- spend the budget: $\sum_i h_i = d$.

The pressure is closed form: both conditional means are one-vs-rest logistics of "own
logit minus a random log-denominator"; the top-$K \approx 6$ competitors' on/off
patterns are enumerated exactly, the weak tail is lognormal moment-matched, and the 1-d
logistic integral is a fixed quadrature (`geometry.waterfill_h`). Closed-form threshold:
feature $i$ is represented iff $\alpha_i p(1-p)(\mathbb E[p_{\star i}|1] -
\mathbb E[p_{\star i}|0]) > \nu$.

Structure, verified at machine precision at the optimum of the certified evaluator: one
common pressure level (interior spread 0.0000), dropped directions below it, and **no
direction saturates at $h = 1$** — dedication is fractional up to the head. Accuracy:
corr 1.0000/0.9983/0.9691 against the Fantope, head exact at every budget (0.83/0.93/
0.98 vs trained 0.84/0.94/0.99), $\nu$ = 0.0448/0.0150/0.0047 vs measured
0.0442/0.0150/0.0052. First-order conditions suffice; the eigenbasis is feature-aligned
for iid features and a non-degenerate ladder (with correlated features, the same
construction applies in the target-covariance eigenbasis).

**3. The mechanical map.** Given the predicted map, everything measured — the
three-regime size curve including the amplified head and its S-shaped rise, spectra,
interference loads, Grams — follows deterministically from the balanced-SVD
factorization. No further physics.

**4. The boundary.** Trained networks live between two task-only class optima: the
additive optimum and the additive + strong-pairs optimum (pairwise interaction terms
$z_a z_b$ among strong features — per-context gain control that additivity cannot
express; they arise because rank-$d$ compression makes co-activation-specific errors and
the softmax prices exactly those contexts). The pairs class *beats* the trained networks
on CE at every budget, so training is dynamics-limited, not class-limited; amplification
and pair terms are substitutes (explicit pair vectors relax the head's anti-aligned
budget). Where a trained curve sits inside the bracket is a question about training
dynamics, out of scope by design.

**Degeneracy.** The optimum is a set, not a point: orientations are free, threshold
ties break arbitrarily (features swap $h \approx 0.5 \leftrightarrow 0.04$ across
runs), and the pair allocation is restart-degenerate. These directions are unpredictable
in principle; the theory predicts their statistics — the Fantope's fractional spectrum
is the tie-breaking occupancy distribution, matching the trained seed scatter.

## Certified instruments

- `fantope_leverage`: the allocation as a convex program — $\min_H \mathbb E\,
  \mathrm{KL}(p_\star \| \mathrm{softmax}(H J(\alpha \odot z)))$ over $\{0 \preceq H
  \preceq I, \mathrm{tr} H = d, H\mathbf 1 = 0\}$. Matches trained leverage at corr
  0.98--0.99 with counts 6/11/20 vs trained 7/11/$\sim$19; the exact evaluator of
  claim 2.
- `dc_map`: the map as a sequence of convex programs — truncated-nuclear
  difference-of-convex ($\mathrm{CE} + \lambda(\|R\|_* - \|R\|_{\mathrm{KyFan}\text-d})$,
  monotone; the penalty vanishes on rank $\le d$, so the kept spectrum is never shrunk)
  plus a convex coefficient polish. Excess CE 0.2031/0.1322/0.0780 vs the nonconvex
  oracle 0.1900/0.1264/0.0752 and trained 0.2101/0.1283/0.0758; finds the amplified
  basin unaided.
- `predicted_map` (nonconvex multi-start, benign in practice) and the pairs oracle:
  reference optima of the two classes in claim 4.

A single convex pass cannot produce the geometry: the last output directions are
saturation-shaped and no convex relaxation of the target proposes them (three chain
variants fail with final subspace cosines 0.56/0.43/0.28); extending the Fantope to the
pair dictionary is counterproductive (the relaxation gap grows with dictionary size).
Hard-rank benignity is not universal in this family — the dense-cell truncation basin is
a documented spurious minimum — so certification per step (DC) is the strongest
available guarantee.

## Method lesson: hybrid fields

Every smooth-field approximation in this problem failed the same way, and the same cure
fixed all three instances (cavity couplings, block objectives, allocation pressure):
the few strong competitors are spiky and discrete, and flattening them into a Gaussian
or lognormal is the error. Enumerate the top-$K$ strong features' $2^K$ patterns
exactly; smooth only the weak tail.

## Limits and open directions

- The one quantitative residual: the $d = 12$ representation threshold over-spreads
  (formula counts 36 vs 20). Localized and bounded: threshold features live at
  probabilities $\sim 10^{-3}$, where the moment-matched field has additive but not
  multiplicative accuracy in $m_1 - m_0$ at the $\nu = 0.005$ scale. Ruled out by direct
  test: enumeration width, centering variance, centering covariance (the exact pressure
  identity holds at the optimum with interior spread 0.0007).
- The pair sector's magnitude is computed (class optimum), not derived; its allocation
  is degenerate, so no deterministic second-order theory is warranted.
- Open: the training-dynamics selection principle (where inside the additive/pairs
  bracket finite training lands); correlated features via the target-covariance
  eigenbasis; a benign-landscape audit (Hessian spectra at the found optima).
