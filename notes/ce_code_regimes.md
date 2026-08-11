# When is each code prediction optimal?

The question throughout: a task is a conditional $p(x' \mid x)$; a model represents each item
$x$ as a $d$-dimensional code $e_x$ and reads out logits through a matrix,
$\hat{L}[x, x'] = e_x^\top d_{x'}$, followed by a softmax. Training minimizes the population
cross-entropy (CE). We want to *predict* the geometry the trained model ends up with, not just
bound its loss. Three regimes, three answers.

Throughout, write the logit table as a matrix product $\hat{L} = E D$ with $E$ ($N \times d$,
the codes) and $D$ ($d \times n_x$, the readout), and let $L^\*$ be the Bayes logits
$\log p(x' \mid x)$ (defined up to adding a constant per row, since the softmax ignores row
shifts).

## Regime 1 — full rank ($d \geq \operatorname{rank} L^\*$): exact fit + balanced split

When the budget doesn't bind, the model can hit Bayes exactly, and the CE optimum in
*function space* is unique: $ED = L^\*$ up to row shifts. Everything interesting is in which
*factorization* of that fixed table gradient descent picks, and the answer comes from a
conservation law rather than from the loss.

**Conservation.** For gradient flow on any loss $\mathcal{L}(ED)$,

$$\dot{E} = -\nabla_{\hat L}\mathcal{L}\, D^\top, \qquad \dot{D} = -E^\top \nabla_{\hat L}\mathcal{L},$$

so

$$\tfrac{d}{dt}\left(E^\top E - D D^\top\right)
  = \dot E^\top E + E^\top \dot E - \dot D D^\top - D \dot D^\top = 0.$$

The imbalance $E^\top E - DD^\top$ never moves. Small initialization means it starts near
zero, so the limit is a **balanced** factorization. Note the loss cancels out of this argument
entirely — sharpness, measure weighting, none of it can change the split.

**Balanced factorizations are the SVD split.** Write $L^\* = U S V^\top$ and a general
factorization $E = U\sqrt{S}\,G$, $D = G^{-1}\sqrt{S}\,V^\top$. Balance requires
$G^\top S G = G^{-1} S G^{-\top}$, i.e. $(GG^\top) S (GG^\top) = S$; for the positive-definite
matrix $H = GG^\top$ this forces $(S^{1/2} H S^{1/2})^2 = S^2$, hence $H = I$: $G$ is
orthogonal. So the prediction is $E = U\sqrt{S}$, $D = \sqrt{S}V^\top$ **up to rotation** —
the balanced $\sqrt{S}$ split of the *plain* SVD, in the parameter metric. This is
`factor_logcond`'s split; the measure weightings there matter for regime 2, not for this
endpoint.

**Caveats that re-open the budget.** The tail modes approach the fit at rate
$\ell(t) \approx \log t$ once the softmax saturates (the gradient toward a logit of size
$\ell$ decays like $e^{-\ell}$), so at fixed compute a sharp task shows an *effective* budget:
modes whose Bayes logits exceed $\sim \log T$ are missing. Weight decay $\lambda$ makes that
permanent: each mode stops where the marginal CE gain $\sim e^{-\ell}$ equals $\lambda \ell$.
Which modes make the finite-time cut is decided by the measure-weighted gradient — that's
where the Fisher weighting of regime 2 re-enters even at full rank.

## Regime 2 — budget binds, flat conditionals: whitened truncated SVD

With $d < \operatorname{rank} L^\*$ the model must approximate, and CE says which errors
matter. Let $\delta_x$ be row $x$'s logit error. Expanding row $x$'s KL to second order:

$$\mathrm{KL}\left(p_x \,\|\, q_x\right) \approx \tfrac12\, \delta_x^\top F_x\, \delta_x,
  \qquad F_x = \operatorname{diag}(p_x) - p_x p_x^\top,$$

the softmax Fisher at that row's own conditional. Note $\delta^\top F \delta =
\operatorname{Var}_p(\delta)$: a constant shift is free (softmax gauge), and errors on
outcomes $p_x$ ignores are cheap. Summing over items with weights $w_x$ and replacing every
$F_x$ by one shared $\bar F$ at the average conditional $\bar q = \sum_x w_x p_x$ gives

$$\text{excess CE} \approx \tfrac12 \left\| \sqrt{W}\,(L^\* - \hat L)\,\sqrt{\bar F} \right\|_F^2 ,$$

a *weighted* Frobenius norm. Eckart–Young then hands us the optimum in closed form: whiten
rows by $\sqrt{W}$ and columns by $\sqrt{\bar F}$, truncate the SVD at rank $d$, unwhiten.
That is `factor_features` (and `factor_logcond`'s $\sqrt{\nu}$ column scaling is the same
Fisher whitening with $\bar F$ approximated by its diagonal — identical once rows are
centered, since $\operatorname{Var}_q(\delta) = \mathbb{E}_q[\delta^2]$ for centered
$\delta$).

**Validity.** Two approximations were made, and both hold exactly when conditionals are flat:
(i) errors small enough for second order, (ii) every row's $F_x$ close to $\bar F$ (all rows
near the marginal). One rule from combining regimes 1 and 2: *the Fisher metric decides what
the code keeps; the parameter metric decides how it is split.* The truncation step uses the
whitened SVD; the factor split should still be the plain balanced one.

## Regime 3 — budget binds, sharp conditionals: the convex program

Sharp conditionals break both quadratic assumptions at once: the optimum sits at $O(1)$ error
where saturation matters (CE charges interference far less than quadratically once the right
outcome dominates — this is exactly what makes packing several features into shared
directions profitable), and the per-row Fishers become supported on nearly disjoint outcome
sets, so no shared metric is right. There is no closed form to find: the stationarity
condition couples the singular vectors of the solution to softmax residuals that are
transcendental in those same vectors — a self-consistent eigenproblem.

**But the problem is convex except for the rank.** CE is convex in the logit *map*
$R = A^\top D$ (logits are linear in $R$); only $\operatorname{rank} R \le d$ is nonconvex.
The convex surrogate for rank is the nuclear norm, and here it is not a heuristic:

$$\min\left\{ \tfrac12\left(\|A\|_F^2 + \|D\|_F^2\right) : A^\top D = R \right\} = \|R\|_*,$$

(per singular value, AM–GM: $\tfrac12(a^2 + d^2) \ge ad$, equality when balanced). So

$$\min_R\; \mathrm{CE}(R) + \lambda \|R\|_*$$

is exactly the global optimum of training the factorized model with weight decay $\lambda$ —
the implicit-bias prediction, made computable. `nuclear_features` solves it by proximal
gradient (gradient step on CE; singular-value soft-threshold as the prox), walks $\lambda$
down a regularization path, truncates each path point to rank $d$, and polishes with a
deterministic joint descent — the path proposes the basin, the polish converges within it.
No restarts, no init dependence. In the flat regime it reduces to regime 2's answer; at full
rank with $\lambda \to 0$ it reduces to regime 1's.

**What the sharp limit is *not*: max-margin packing.** One might guess that as conditionals
sharpen the problem tends to "maximize the worst classification margin," whose solutions are
the equally-spaced polygons familiar from the MSE superposition story. That fails here, for a
clean reason. The worst-case margin of feature $j$ against sign patterns is
$2\big(K_{jj} - \sum_{i \neq j} |K_{ij}|\big)$ with $K$ the feature Gram: signal minus total
interference. Six nonzero vectors in the plane can never make every row diagonally dominant,
so every full-support packing has a *negative* worst margin — while simply dropping a feature
makes its flip-pairs tie at margin exactly $0$. Max-min-margin therefore prefers truncation,
contradicting exact CE, where packing demonstrably wins. The resolution: in CE a tie costs a
bounded $\log(\text{degeneracy})$, while a negative margin at growing scale costs ever more.
The equally-spaced polygon of the MSE story does not appear here.

### The additive class has an exact closed form — and it is rigid

For codes additive on *both* sides ($G[x, x'] = s^\top K\, s'$ with $K = A^\top B$ the rank-$d$
feature matrix), the sharp regime turns out to need no approximation at all. The gap against
flipping a feature set $F$ is **additive over the flipped features**,

$$\Delta(F, s) = \sum_{j \in F} g_j(s), \qquad g_j(s) = 2\, s_j\, (s^\top K)_j ,$$

so the row partition function factorizes exactly,
$\sum_{x'} e^{-\Delta(x, x')} = \prod_j \big(1 + e^{-g_j(s)}\big)$, and the *entire* excess CE
collapses into per-feature potentials of the single-flip gaps:

$$\text{excess} \;=\; \mathbb{E}_s \sum_{j=1}^m \varphi_w\big(g_j(s)\big), \qquad
  \varphi_w(g) = \log\frac{1 + e^{-g}}{1 + e^{-2w}} + f\,(g - 2w),$$

with $f = 1/(1 + e^{2w})$ the per-feature flip probability. This is exact at every $\gamma$
(`geometry.additive_cube_excess`; verified against `expected_ce` to $10^{-9}$). Only the
*readout* needs to be feature-additive: the gap is linear in $s'$, so the identity holds with
an arbitrary code $e_x$ and $g_j(x) = 2 s'_j \langle e_x, d_j\rangle$, and it generalizes
beyond the cube to any product structure with a component-additive target (heat kernels on
product graphs), with the softplus becoming a per-component log-sum-exp. It does not survive
non-product outcome spaces or non-additive readouts — which is exactly the direction real
optima escape in. The potential
$\varphi_w$ is convex and asymmetric — softplus (exponential) cost for gap deficit,
linear-with-slope-$f$ cost for gap surplus, minimized exactly at the target gap $g = 2w$ — and
it prices ties correctly for free: $g = 0$ costs $\log 2$. Expanding
$\log(1 + e^{-g}) \approx e^{-g}$ for large gaps and averaging over the product measure gives
a factorized "log-cosh coherence" energy (the CE analogue of the MSE superposition energy,
with $\log\cosh$ interference in place of quadratic overlaps) — but that expansion is only the
non-degenerate asymptotics; the potential form above is the real object.

**And it explains why real optima leave the additive family.** Additivity forces every
multi-flip gap to be a sum of single-flip gaps: the $m$ scalar gap fields $g_j(s)$ determine
the entire interference structure, so nothing can be patched for specific outcomes. Measured
at the dense-sharp cell ($m=6$, $d=2$, $\gamma = 0.03$): truncation $1.58$ → best
fully-additive code $\approx 1.46$ (packing barely pays inside the class) → additive code
with a *free per-outcome readout* $1.07$ (`nuclear_features`) → free code and readout $0.92$
(trained k-linear). The sharp regime's packing gains come mostly from breaking additivity —
first in the readout, then in the code. So the honest sharp-end summary: the additive class
has an exact analytic theory (the potentials above), and the distance between its optimum and
the convex program's is a *measurement of how non-additive the true optimum is* — which is
itself the superposition phenomenon, seen from a new angle.

## Summary

| Regime | Condition | Optimal prediction | Status |
|---|---|---|---|
| Full rank | $d \geq \operatorname{rank} L^\*$ | exact fit; plain balanced $\sqrt{S}$ split (conservation law) | closed form |
| Budget + flat | errors small, rows near marginal | Fisher-whitened truncated SVD (`factor_features`) | closed form |
| Budget + sharp | saturation, heterogeneous Fishers | nuclear-penalized exact CE, path + polish (`nuclear_features`) | convex program |
| Budget + sharp, additive class | code and readout both additive in features | per-feature softplus potentials (`additive_cube_excess`) | closed form (exact), but the class is rigid — real optima leave it |

The closed forms stay in the toolkit for what they explain (spectra, harmonics, whitening —
*why* the geometry looks the way it does), and their deviation from the convex program is the
superposition signal itself: where and how much the true optimum departs from the
second-moment account.
