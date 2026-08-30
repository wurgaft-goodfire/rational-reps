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
class-limited; at $d = 12$ the top-6 pair set is too narrow (the residual lives in pairs
further down the ladder) — extend the pair set before quoting $d = 12$ numbers.

## Open items

- Assemble the scalar self-consistent equations from the analytic block objective
  (exchangeable-frame ansatz over the shared block) and compare the resulting closed-form
  size/coupling curves to the oracle across $d$.
- The cyclic solver stalls at $d = 3$ (+0.009 above the joint optimum); treat $d=3$ as
  the stress case for any fixed-point claim.
- Pairs extension at $d = 12$ with a wider pair set (top-18 features).

## Validation scripts (session 2026-08-29/30, /tmp)

`gauge_check.py` (two observables), `asym_cavity.py` (class equivalence),
`cavity_oracle.py` (symmetric cavity + statistics-suffice), `mf_single_feature.py` /
`hybrid_block_test.py` (Gaussian vs hybrid block), `mf_cavity_solver3.py`
(self-consistency), `fantope_test.py`, `leverage_test.py`, `finite_sample_check.py`
(no finite-sample effect).
