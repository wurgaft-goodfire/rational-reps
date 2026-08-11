# Which code does a trained softmax model learn?

Setup: a task is a conditional $p(x'|x)$ with Bayes logits $L^*[x,x'] = \log p(x'|x)$
(defined up to a constant per row — softmax ignores row shifts). The model stores a
$d$-dimensional code per item and reads out linearly: logits $\hat L = ED$ with $E$
($N \times d$) and $D$ ($d \times n_{x'}$), trained by gradient descent on population CE.
We want to predict the learned $(E, D)$, not just the loss. Three regimes.

## Regime 1 — full rank ($d \ge \operatorname{rank} L^*$)

Here $ED = L^*$ is achievable and the function-space optimum is unique, so the whole
question is *which factorization* of the fixed matrix $L^*$ training picks. Two short
derivations answer it.

**(1a) The imbalance is conserved.** For gradient flow on any loss $\mathcal L(ED)$, with
$M := \nabla_{\hat L}\mathcal L$:

$$\dot E = -M D^\top,\quad \dot D = -E^\top M
\;\Longrightarrow\;
\tfrac{d}{dt}(E^\top E - DD^\top)
= -\,D M^\top E - E^\top M D^\top + E^\top M D^\top + D M^\top E = 0 .$$

Small init $\Rightarrow$ $E^\top E \approx DD^\top$ forever. The loss cancelled out of the
derivation, so nothing about the task (sharpness, measure, Fisher) can move the split.

**(1b) Balanced factorizations are the plain-SVD split.** Write $L^* = USV^\top$ and any
factorization $E = U\sqrt S\,G$, $D = G^{-1}\sqrt S\,V^\top$. Balance requires
$G^\top S G = G^{-1} S G^{-\top}$, i.e. $HSH = S$ for $H = GG^\top \succ 0$, i.e.
$(S^{1/2}HS^{1/2})^2 = S^2$, and the PSD square root is unique, so $H = I$: $G$ is
orthogonal. **Endpoint prediction: $E = U\sqrt S$, $D = \sqrt S V^\top$ up to rotation —
the balanced split of the *plain, unweighted* SVD.**

### So why weight by $\sqrt\nu$ at full rank at all?

Not for the endpoint — (1a)+(1b) contain no measure. The weighting is needed because at
finite training time you observe the *path*, not the endpoint, and the path is
measure-ordered with a saturation bottleneck:

- **Which modes grow first.** The CE gradient in logit space is
  $\nabla_{\hat L}\text{CE} = \operatorname{diag}(w)(Q - P)$ — item-frequency times softmax
  residual. At small init this is the measure/Fisher-weighted target, and factorization
  dynamics grow modes at rates ordered by *its* singular values. So acquisition order = the
  **weighted** SVD's order, not the plain one.
- **How long the tail takes.** Once a row saturates, the gradient pushing a logit of size
  $\ell$ toward its Bayes value decays like $e^{-\ell}$, so $\dot\ell \propto e^{-\ell}
  \Rightarrow \ell(t) \approx \log t$: reaching Bayes logits of size $L$ costs $t \sim e^L$.
  Sharp tasks have large $L$ (on the cube, $2w$ per flipped feature, $w = \tfrac12\log\coth\gamma$).

Together: at fixed compute $T$, the modes present are the top ones **in the weighted
ordering** whose required logits are $\lesssim \log T$ — a finite-time *effective budget*
$d_{\text{eff}}(T)$ — and the modes that are present are split per (1a)+(1b). With weight
decay $\lambda$ the truncation becomes permanent (each mode stops where $e^{-\ell} = \lambda\ell$:
a soft-thresholded spectrum even at $t = \infty$). One sentence to remember:
**the Fisher/measure metric decides what the code contains; the parameter metric decides how
it is split.** At exact full rank and $t \to \infty$ with no decay, the first half is moot
and $\sqrt\nu$ drops out; in every observable situation it doesn't.

## Regime 2 — budget binds, flat conditionals

Now truncation is real and the only question is which rank-$d$ table to keep. Expand row
$x$'s KL to second order in its logit error $\delta_x = (L^* - \hat L)_x$:

$$\mathrm{KL}(p_x \| q_x) \approx \tfrac12 \delta_x^\top F_x \delta_x,
\qquad F_x = \operatorname{diag}(p_x) - p_x p_x^\top,$$

and note $\delta^\top F \delta = \operatorname{Var}_{p}(\delta)$: constant shifts are free
(the softmax gauge), errors on outcomes $p_x$ ignores are cheap. Replace each $F_x$ by one
shared $\bar F$ at the average conditional $\bar q = \sum_x w_x p_x$:

$$\text{excess CE} \approx \tfrac12\,\big\| \sqrt{W}\,(L^* - \hat L)\,\sqrt{\bar F} \big\|_F^2 .$$

This is a plain Frobenius norm in whitened coordinates, so Eckart–Young gives the optimum:
whiten by $\sqrt W$, $\sqrt{\bar F}$, truncate the SVD at rank $d$, unwhiten
(`factor_features`; `factor_logcond`'s row-centering + $\sqrt\nu$ is the same metric, since
$\operatorname{Var}_q(\delta) = \mathbb E_q[\delta^2]$ once $\delta$ is $q$-centered). Then
split the resulting table by regime 1's rule — plain balanced SVD of the truncation.

Two approximations were made, both exact in the flat limit and both broken by sharpness:
(i) second order in $\delta$ (misses saturation — CE charges interference *less* than
quadratically once the right answer dominates, which is what makes packing profitable);
(ii) one shared $\bar F$ (sharp rows have nearly disjoint supports, so no common metric is
right).

## Regime 3 — budget binds, sharp conditionals

No closed form exists: stationarity couples the singular vectors of the solution to softmax
residuals that are transcendental in those vectors (a self-consistent eigenproblem — the
proximal iteration below *is* its natural fixed-point solver).

**The convex program.** CE is convex in the logit map $R = A^\top D$ (logits are linear in
$R$); only $\operatorname{rank} R \le d$ is nonconvex. The nuclear norm is its convex
surrogate, and here it is exactly the implicit-bias object: per singular value,
$\tfrac12(a^2 + d^2) \ge ad$ with equality iff balanced, so

$$\min\{\tfrac12(\|A\|_F^2 + \|D\|_F^2) : A^\top D = R\} = \|R\|_* ,
\qquad\Rightarrow\qquad
\min_R\, \mathrm{CE}(R) + \lambda\|R\|_*$$

is the global optimum of training the factorized model with weight decay $\lambda$.
`nuclear_features` solves it by proximal gradient (CE gradient step; singular-value
soft-threshold prox), walks $\lambda$ down a path, truncates each path point to rank $d$,
polishes with joint L-BFGS. Deterministic; flat limit reduces to regime 2, full rank with
$\lambda \to 0$ to regime 1.

**Max-margin is not the sharp limit** (tempting, wrong). Worst-case margin of feature $j$
over sign patterns is $2(K_{jj} - \sum_{i\ne j}|K_{ij}|)$; six nonzero plane vectors can
never be diagonally dominant, so every full-support rank-2 packing has a negative worst
margin, while *dropping* a feature ties its flip-pairs at margin exactly 0. Max-min-margin
therefore prefers truncation — but CE charges a tie only $\log(1+\#\text{ties})$, which is
why packing wins under CE and no MSE-style polygon appears.

**The CE-optimal sharp code is not additive in the features — and the lift must match the
model class.** With an additive readout, the gap against flipping a feature set is a sum of
single-flip gaps (the gap is linear in $s'$), so additive models cannot manage interference
per outcome: the class is rigid. Measured at the dense-sharp cell ($m{=}6$, $d{=}2$,
$\gamma{=}0.03$): truncation $1.58$ → best fully-additive code $\approx 1.46$ → additive
code + free readout $1.07$ (`nuclear_features` on the feature lift) → trained k-linear
(free code and readout) $0.92$ → the same convex program in the **free-row lift** ($R$ =
the full table, `nuclear_features(logP, I, d)`) $0.90$. Packing gains come mostly from
leaving additivity, readout first; and once the lift matches the model class, the convex
program slightly beats the trained model — the variational account explains training in
full, and every "network beats the oracle" episode was class mismatch, not dynamics.

## Summary

| Regime | Optimal prediction | Status |
|---|---|---|
| Full rank | exact fit; plain balanced $\sqrt S$ split (conservation law); finite-$T$: weighted ordering sets $d_{\text{eff}}$ | closed form |
| Budget + flat | Fisher-whitened truncated SVD, plain-split (`factor_features`) | closed form |
| Budget + sharp | nuclear-penalized exact CE, path + polish (`nuclear_features`), lifted over the model's own class | convex program |

The closed forms explain (spectra, harmonics, whitening); the convex program predicts; the
gaps between them — quadratic-vs-exact, additive-vs-free — are the superposition signal.
