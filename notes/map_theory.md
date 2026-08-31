# The map, the gauge, and the cavity

Draft for merging into `superposition.tex`. Status: every claim below is validated
numerically against exact oracles (scripts referenced at the end); the analytic reduction
of the cavity integrals is the one remaining derivation.

## Setting

Binary features $z_i \sim \mathrm{Bernoulli}(p)$ iid, target
$p_\star(x{=}i \mid z) = \mathrm{softmax}_i(\alpha_i z_i)$, additive rank-$d$ model
$\ell_k(z) = \sum_j z_j R_{jk} + b_k$ with $\mathrm{rank}(R) \le d$.

## Two observables, one solution

Any solution supports two distinct measurements:

- **The map** $R$: regress row-centred logits on $z$. Gauge-invariant; describes what the
  model computes. (`geometry.logit_map`)
- **The representation**: balanced-SVD the logits, regress the code on $z$, read
  $\|E_i\|^2$. Describes the weights GD learns (the balance gauge is the implicit bias).
  (`balanced_code` + `feature_vectors`)

These coincide only when reader and writer vectors are tied per feature ($\kappa_i = 1$).
Trained solutions split norms ($D_i = \kappa_i E_i$ with tied directions, $\kappa$
heterogeneous), so the two observables diverge. The measured "amplification"
($\|E_1\|^2 \approx 11.6$ vs $\alpha_1 = 8$) is the balanced factorization of an
*asymmetric* map whose diagonal is not amplified at all. The empirical identity
$G_{ii} + p S_i \approx \alpha_i$ is the gauge shadow of $\mathrm{diag}(R) = \bar\alpha$:
the head vector's extra length is spent on anti-aligned components that cancel out of the
net response.

## Model classes

With writers $w_i = a_i u_i$ and readers $r_i = c_i u_i$ ($u_i$ unit):

| class | excess CE (d=3/6/12) | top measured $G_{ii}$ |
|---|---|---|
| free $AB^\top$ | 0.2206 / 0.1383 / 0.0795 | 11.63 / 11.58 / 7.12 |
| tied direction + free $\kappa$ | 0.2209 / 0.1386 / 0.0796 | 11.64 / 11.67 / 7.11 |
| symmetric ($\kappa \equiv 1$) | 0.2358 / 0.1410 / 0.0801 | 7.80 / 7.51 / 7.11 |

The tied-direction class is equivalent to the free class; forcing $\kappa = 1$ changes the
optimal map (weaker couplings), not just the gauge. Trained networks (free per-word codes)
beat all additive classes (0.2101 / 0.1283 / 0.0758) and additionally sharpen the actual
map diagonal (to $\approx 9\text{--}10$) through the non-additive part of their code
(additive $R^2 \approx 0.83\text{--}0.88$); that class effect is outside this theory.

## The theory, in four layers

Stationarity of exact CE per feature $i$ (with $q = \mathrm{softmax}(\ell)$):
bias $\mathbb E[q_i] = \mathbb E[p_{\star i}]$; reader
$\mathbb E[(q_i - p_{\star i})\,x(z)] = 0$ where $x = \sum_j z_j w_j$; writer
$\mathbb E[z_i \sum_k (q_k - p_{\star k})\, r_k] = 0$.

1. **Diagonal (exact anchor).** Matching on/off mean probabilities under noise pins the
   mean logit displacement: $R_{ii} - \bar\alpha_i \approx \tfrac12(\mathrm{Var}\,t_i -
   \mathrm{Var}\,u_i)\cdot\text{curvature}$, where $t_i, u_i$ are the target and model
   one-vs-rest logits. The couplings make the model's noise track the target's, so the
   correction is second order: oracles hit $\bar\alpha_i$ to $\lesssim 1\%$ away from the
   representation threshold.
2. **Support / allocation.** Dimensional capacity $h_i$ (leverage, `geometry.leverage`) is
   predicted by the convex Fantope program $\min_H \mathbb E\, \mathrm{KL}(p_\star \|
   \mathrm{softmax}(H J(\alpha \odot z)))$ over $\{0 \preceq H \preceq I,\ \mathrm{tr} H =
   d,\ H\mathbf 1 = 0\}$ (`geometry.fantope_leverage`): corr 0.98/0.995/0.994 with trained
   leverage, counts 5/9/19 vs trained 6/11/19 (old theory: 9/17/33). The relaxation is
   never tight ($\|H^2 - H\| / \|H\| \approx 0.22$--$0.25$) and its fractional spectrum
   matches the trained fractional dedication (top $h$ = 0.84/0.94/0.99 at $d$=3/6/12).
3. **Couplings (hybrid cavity, closed form).** The centred target map is exactly
   diagonal, so negative couplings are rank-$d$ projection residue under the CE metric:
   the reader condition is a logistic regression whose $\sigma'$ weights concentrate on
   uncertain states, where competitor presence anti-predicts $i$. The field
   $h = \sum_j z_j w_j$ must be treated as **exact enumeration over the top-$K \approx 8$
   strong features' $2^K$ patterns**; a pure Gaussian field fails (77% coupling error per
   block, compounding to $O(1)$ under self-consistency), while the hybrid field cuts the
   block error to ~1% and holds the exact oracle as a fixed point of the self-consistent
   sweep to +2% excess CE. The weak tail needs no integrals at all: its Gaussian only
   enters through the second-order Fisher correction, giving the fully analytic block
   objective
   $$\sum_{\sigma \in \{0,1\}^K} P(\sigma) \sum_{z_i} P(z_i)\Big[\mathrm{lse}(\bar\ell) +
   \tfrac12\big(q^\top \mathrm{diag}(C) - q^\top C q\big)\Big] - \text{(exact tilted
   linear term)},$$
   with $\bar\ell$ the mean logits at pattern $(\sigma, z_i)$, $q = \mathrm{softmax}
   (\bar\ell)$, and $C = R_{\rm read}\, \Sigma_{\rm tail}\, R_{\rm read}^\top$. Validated
   per block against the exact best response: sizes to $<1\%$, head coupling to $0.1\%$,
   direction cosines $\ge 0.96$; weak-feature couplings keep $O(0.1)$ absolute error and
   the dead-feature coupling sign is unreliable (threshold zone).
4. **Measured geometry (mechanical).** Given the predicted $R$, the balanced norms,
   Grams, spectra — including the "amplified" head and the S-shaped size curve — follow
   deterministically from the min-norm (balanced SVD) factorization of $R$. No further
   physics.

## The non-additive component: strong-pair interactions

The trained code's residual (after the additive fit; 7--16% of code variance) is **not**
the log-normalizer ($R^2 \le 0.006$ for $\log\sum_t e^{\alpha_t z_t}$, and adding it to
the oracle class changes nothing) and not the activity count. It is pairwise interaction
terms among strong features: top-6 pairs explain 60% / 40% / 3% of the residual at
$d = 3/6/12$. Adding those 15 pair directions to the additive class (code stays
$d$-dimensional) gives excess CE 0.1906 / 0.1302 / 0.0795 vs additive 0.2292 / 0.1383 /
0.0795 and trained 0.2101 / 0.1283 / 0.0758 — closing most of the gap at $d = 6$ and
reproducing the trained map signature (head diagonal sharpens to 8.4--8.9, additive $R^2$
drops to 0.85--0.88, represented count rises). Two corollaries: at $d = 3$ the pairs
class *beats* the trained network, so trained solutions are dynamics-limited rather than
class-limited; and with the pair set widened to the top 18 features the same holds at
$d = 12$ (residual $R^2$ 0.026 $\to$ 0.670; excess CE 0.0751 vs trained 0.0758). The
*allocation* of pair capacity is restart-degenerate (pair vectors replicate across
restarts at $|\cos| \approx 0.6$ only, one dominant sparse pair per budget, no stable
alignment with $-(R_a + R_b)$): the class-level statement is sharp, the specific pair
solution is not identifiable, so no deterministic second-order theory is warranted.

## Structure of the coupling block (measured on the oracles)

The represented off-diagonal block of the centred map is rank-one to $R^2$ = 1.000 /
0.954 at $d$ = 3/6 and drops to 0.49 at $d$ = 12; loadings concentrate on the head
(writer side) and the one-or-two *marginal* (threshold-adjacent) features (reader side),
with a weak uniform negative bulk underneath. Interference is priced onto the threshold
channel — the marginal feature has the least Fisher weight to lose. $\kappa_i$ rises
smoothly along the ladder (1 $\to$ 1.6/2/3 at $d$ = 3/6/12): weak features are
reader-heavy, matching the trained tie-check.

## The predictor in the notebook

`geometry.predicted_map(alpha, p, d)` — the tied-direction + $\kappa$ CE optimum on a
fixed-seed task sample — feeds every notebook figure through
`geometry.logit_features` (the identical balanced measurement applied to trained models
and predictions). Fully analytic joint objectives were tried and rejected: the
second-order Fisher tail is unbounded below once weak-feature logits grow (its
correction vanishes as softmax saturates), and any surrogate whose lse-term measure
differs from its linear-term measure has unbounded descent directions. A stratified
2^K-pattern sample reproduces the diagonal to corr 1.000 but overfits sparse tail draws.
The iid-sample optimum is the honest deterministic predictor; the closed form remains
open below.

## The certified solver: truncated-nuclear DC

Single-pass convex chains cannot find the geometry: three variants (output-Fantope
subspace + refit; extended-dictionary Fantope + refit; output-Fantope + extended-
dictionary refit) all fail on the same signature — the last one or two output directions
are saturation-shaped and no convex relaxation of the target proposes them (final
subspace cosines 0.56/0.43/0.28). The certified alternative that works: minimize CE +
lambda (||R||_* - ||R||_KyFan-d) by difference-of-convex iterations (each step a convex
program, monotone descent; the penalty vanishes on rank <= d, so lambda is free to be
large and the kept spectrum is never shrunk — repairing both defects of the plain
nuclear path). With a convex coefficient polish in the found subspace
(`geometry.dc_map`): excess CE 0.2031 / 0.1322 / 0.0780 vs the nonconvex pairs oracle
0.1900 / 0.1264 / 0.0752 and trained 0.2101 / 0.1283 / 0.0758 — it beats the trained
network at d = 3, finds the amplified basin unaided (measured head 11.5 at d = 3), and
the rank collapses to d exactly (tail = 0 before truncation).

## Allocation is water-filling: KKT of the Fantope, verified

At the (properly converged) Fantope optimum, in the eigenbasis of H*: every represented
direction's pressure — its eigenvalue of $-\nabla_H \mathbb E\,\mathrm{KL}$ — equals
one common level $\nu$ (spread 0.0000 at machine precision, all d), every dropped
direction sits below it, and **no direction saturates at h = 1** at any budget —
dedication is fractional all the way up, matching the trained top leverage 0.84/0.94/
0.99. The diagonal-restricted program reproduces the full optimum (corr 0.984--0.995),
so the allocation formula is: solve $g_i(h_i) = \nu$ per feature with $\sum_i h_i = d$,
where $g_i(h)$ is the marginal KL gain of feature i's gain under softmax saturation from
its co-active competitors. Remaining derivation: $g_i$ in analytic (or one-integral)
form — the corrected successor of the $\alpha_\star$ breakpoint scan, with the S-shaped
ramp generated by the saturation in $g_i$. First-order conditions suffice (no curvature
term). Basis caveat: feature-alignment of the eigenbasis rests on iid features and a
non-degenerate ladder; with correlated features the same construction applies in the
eigenbasis of the target logit covariance.

## The h(alpha) formula (`geometry.waterfill_h`)

The pressure has a closed form. In the diagonal program the logits are
$h_i \alpha_i z_i + b_i$, so $g_i = \alpha_i\, p\, (\mathbb E[p_{\star i}\mid z_i{=}1]
- \mathbb E[q_i \mid z_i{=}1])$, and both conditional means are one-vs-rest logistics of
"own logit minus a random log-denominator". The denominator is a sum of independent
terms, so its first two moments are elementary sums; a lognormal moment-match and a
fixed Gauss--Hermite quadrature of the logistic give $g_i$ in closed form (probit as the
coarse limit). The theory's formula is then the scalar system: calibrate $b$
($\mathbb E[q_i] = \mathbb E[p_{\star i}]$), equalize pressure ($g_i(h_i) = \nu$ on
the represented set), spend the budget ($\sum h_i = d$). Corollary in closed form:
feature $i$ is represented iff $\alpha_i\, p(1-p)\,(\mathbb E[p_{\star i}|1] -
\mathbb E[p_{\star i}|0]) > \nu$.

The pressure's field is hybrid, as everywhere in this theory: the top-$K \approx 6$
competitors' on/off patterns enumerated exactly, the weak tail lognormal moment-matched,
the 1-d logistic integral by fixed quadrature. Accuracy against the certified Fantope:
corr 1.0000 / 0.9983 / 0.9691 at $d$ = 3/6/12, max $|\Delta h|$ = 0.014 / 0.064 / 0.204,
and the fractional head exact at every budget (0.83/0.93/0.98 vs trained 0.84/0.94/0.99).
Counts 6/12/36 vs 6/11/20 (the old $\alpha_\star$ formula: 9/17/33). The one residual —
the $d = 12$ threshold over-spread — is localized and bounded: threshold features live at
probabilities $\sim 10^{-3}$ where the moment-matched field has additive but not
multiplicative accuracy in $m_1 - m_0$, at exactly the $\nu = 0.005$ scale. Ruled out by
direct test: enumeration width ($K = 10$ identical to $K = 6$), the centering variance,
and the centering covariance (the exact pressure identity was verified at the optimum:
the kept term $\alpha_i \mathbb E[(p_\star - q)_i z_i]$ carries the entire pressure,
interior spread 0.0007, and decays cleanly past the threshold). The Fantope remains the
exact evaluator.

## Open items
- Optional: benign-landscape audit (Hessian bottom eigenvalues at the found optima;
  classify the d=3 cavity stall as saddle vs local minimum). A universal benign-landscape
  theorem is false in this family (the dense-cell truncation basin is a documented
  spurious minimum); only a regime-restricted version could hold.

## Validation scripts (session 2026-08-29/30, /tmp)

`gauge_check.py` (two observables), `asym_cavity.py` (class equivalence),
`cavity_oracle.py` (symmetric cavity + statistics-suffice), `mf_single_feature.py` /
`hybrid_block_test.py` (Gaussian vs hybrid block), `mf_cavity_solver3.py`
(self-consistency), `fantope_test.py`, `leverage_test.py`, `finite_sample_check.py`
(no finite-sample effect).
