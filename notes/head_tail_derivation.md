# Predicting the learned geometry in the feature cube

A rank-$d$ softmax readout with $m \gg d$ features. The derivation runs from stationarity to a
closed prediction of the learned code and decoder, with every step checked against trained
networks.

**Result.** From task parameters plus one finite-capacity constant, the predicted decoder geometry
reaches 84–95% of the seed-to-seed CKA ceiling (mean 91%), against 0.363 for the closed-form PCA
baseline. The water level is predicted to 2.8% and the encoding profile to $\mathrm{corr}=0.985$.

---

## 1. Setting

**Task.** $N$ words, $m$ feature tokens. Word $x$ has binary features $z_x \in \{0,1\}^m$, i.i.d.
Bernoulli($p$). Feature $i$ has importance $\alpha_i = \texttt{scale}\cdot i^{-\beta}$.

$$p(i \mid x) = \frac{e^{\alpha_i z_{xi}}}{W(x)}, \qquad W(x) = \sum_j e^{\alpha_j z_{xj}} .$$

**Model.** Linear readout of width $d$: word $x$ gets a **code** $a_x \in \mathbb{R}^d$, feature $i$ a
**decoder** $D_i \in \mathbb{R}^d$,

$$q(i \mid x) = \frac{e^{\langle a_x, D_i\rangle}}{Z(x)}, \qquad Z(x) = \sum_j e^{\langle a_x, D_j\rangle},$$

trained on $\mathbb{E}_x \mathrm{KL}(p\|q)$.

**Gauge.** $(a_x, D_i)\to(M^{-\top}a_x, MD_i)$ and $D_i \to D_i + c$ both leave the model unchanged.
Fix them: row-centre the logit matrix $G = AD^\top$, then take the balanced factorisation
$A = U\sqrt S,\; D = \sqrt S V^\top$ from its SVD — unique up to $O(d)$. All comparisons use CKA,
which is $O(d)$-invariant.

**Scope.** The code prediction (§4) holds for every spectrum, correlation structure, and $p$ tested. The tail closed form (§3.2) needs $m/d \gtrsim 10$ and iid or equicorrelated features. Below $p \approx 0.2$ the seed-to-seed ceiling collapses, so there is no reproducible geometry left to predict (§8).

---

## 2. The two stationarity conditions

The objective is convex in $D$ given $A$, and convex in $A$ given $D$. Zeroing each gradient:

$$\textbf{(D)}\quad \mathbb{E}_x\big[(q(i|x)-p(i|x))\,a_x\big]=0 \;\;\forall i
\qquad\qquad
\textbf{(A)}\quad \sum_i q(i|x)D_i=\sum_i p(i|x)D_i \;\;\forall x$$

(D) is **separable across features**. (A) is exponential-family moment matching.

*Check:* solving either exactly, given the other from a trained model, returns the trained values at
CKA $1.000$. Trained networks sit at this fixed point.

---

## 3. The decoder, given the code

### 3.1 Exact factorisation of the target moment

Condition (D) needs $\mathbb{E}_x[p(i|x)a_x]$. Splitting on $z_{xi}$ is exact:

$$\mathbb{E}_x[p(i|x)\,a_x] = p\,e^{\alpha_i} v_i^{1} + (1-p)\,v_i^{0},
\qquad v_i^{b}=\mathbb{E}\big[\tfrac{a_x}{W(x)}\,\big|\,z_{xi}=b\big].$$

All feature-specificity is in $e^{\alpha_i}$ and $\delta_i \equiv v_i^1-v_i^0$. Measured
$\|\delta_i\|/\|v\|$: $1.1$–$1.6$ head, median $0.11$ tail — a sharp cliff, reproducible across
seeds (correlation $+0.90$ to $+0.99$).

### 3.2 Tail limit gives a closed form

If the code carries no information about $z_{xi}$ then $v_i^1=v_i^0=v$ and

$$\mathbb{E}_x[p(i|x)a_x] = c_i v,\qquad c_i = p\,e^{\alpha_i}+(1-p).$$

Tail logits are small, so linearise $e^{\langle a,D_i\rangle}\approx 1+\langle a,D_i\rangle$:

$$\boxed{\;D_i = M_Z^{-1}(c_i v - u)\;}\qquad
u=\mathbb{E}\big[\tfrac{a_x}{Z(x)}\big],\quad M_Z=\mathbb{E}\big[\tfrac{a_xa_x^\top}{Z(x)}\big].$$

$u,v,M_Z$ are shared, so **all tail decoders lie on a line**, positioned by $c_i$. The line crosses
the origin at some $c^\ast$; features either side point oppositely, producing the near-antipodal
pairs.

### 3.3 Head features

$\delta_i = O(1)$, independence fails, linearisation fails (large logits). Use the per-feature
convex solve with the conditional target from §3.1.

### 3.4 Checks

CKA against trained decoders (flat / mid / steep spectra):

| level | flat | mid | steep |
|---|---|---|---|
| exact solve, true target | 1.000 | 1.000 | 1.000 |
| conditional target (head) | 1.000 | 1.000 | 1.000 |
| factorised target $c_iv$ (tail) | 0.995 | 0.991 | 0.982 |
| + linearisation (tail closed form) | 0.859 | 0.992 | 0.982 |

Those supply $Z(x)$ as well as the code. Given **only** the code, use the joint convex solve over all
features: CKA $1.000$ own seed, $0.907$–$0.984$ held out.

*Predicted before measured:* tail collinearity — at $d=3$ one PC explains $0.976$ of tail variance.

The line and the collinearity are separate claims, and they decay at different rates with the
budget (tail = the predicted-inactive set, so it shrinks as $d$ grows):

| $d$ | tail size | closed form CKA | 1-PC fraction |
|---|---|---|---|
| 3 | 57 | 0.984 | 0.976 |
| 6 | 51 | 0.978 | 0.881 |
| 8 | 46 | 0.960 | 0.844 |
| 12 | 38 | 0.858 | 0.539 |

The predicted *directions* survive to $d=12$; strict collinearity does not. CKA compares
row-normalised decoders, so it tolerates spread off the line that the PC fraction penalises.

*Controls:* shuffling $c_i$ within the tail drops CKA $0.982\to0.016$; constant $c_i$ gives $0.030$;
$\alpha_i$ in place of $c_i$ gives $0.701$.

---

## 4. The code

Write $a_x \approx \sum_k w_k z_{xk}$ (linear part, $R^2\approx0.86$). Then
$\delta_i \approx \|w_i\|\cdot\mathbb{E}[1/W\mid z_i{=}1]$, with the scalar task-computable
($\mathrm{corr}=0.83$–$0.99$). So predicting the code means predicting $\|w_i\|$.

### 4.1 Amplitude — derived, no constant

At weak coupling the optimum is the truncated SVD. With i.i.d. Bernoulli features the covariance is
diagonal, so the singular directions are the feature axes and
$S_k(Z_c\,\mathrm{diag}\,\alpha)=\sqrt{Np(1-p)}\,\alpha_k$ (measured ratio $0.99$–$1.02$), with
$U_k \approx Z_ce_k/\sqrt{Npq}$. The balanced gauge gives $A=U\sqrt S$, hence

$$\boxed{\;\|w_k\|^2 = \frac{S_k}{Npq} = \frac{\alpha_k}{\sqrt{Np(1-p)}}\;}$$

Both factors are consequences, not fits. Against oracle solutions: $R^2 = 0.967$ (scale $0.25$),
$0.968$ (scale $1.0$).

### 4.2 Threshold — derived from interference

Features share $d$ dimensions. Expanding CE to second order, feature $i$'s interference is
$\eta_i=\sum_{k\neq i}z_k\langle w_k,D_i\rangle$. The Fisher weight $q_i$ is **correlated with**
$z_i$ — outcome $i$ only has appreciable probability when feature $i$ is on — so the signal term is
conditioned on $z_i{=}1$ while the interference term is not:

$$J = \tfrac12 p\sum_i\Big[Q^1_i(\sigma_i-\alpha_i)^2
+ \bar Q_i R^2\!\!\sum_{k\neq i}\langle w_i,w_k\rangle^2\Big],
\qquad \sigma_i = R\,s_i,\;\; s_i=\|w_i\|^2,\;\; R^2 = Npq,$$

with $Q^1_i=\mathbb{E}[p(i|x)\mid z_{xi}{=}1]$ and $\bar Q_i=\mathbb{E}[p(i|x)]$. Using the
tight-frame value $\|WW^\top\|_F^2=B^2/d$ and setting $\partial J/\partial s_i=0$:

$$\boxed{\;\|w_i\|^2 = A\,\frac{(\alpha_i-\mu)^{+}}{\sqrt{Np(1-p)}},
\qquad \mu = \sqrt{Np(1-p)}\;\frac{\bar Q}{Q^1}\;\frac{B}{d},\qquad B=\sum_i\|w_i\|^2 \;}$$

The threshold is what softens the hard truncation of §4.1 and lets more than $d$ features be
represented. $\mu$ is solved self-consistently (one scalar root-find).

*Check:* $\lambda \equiv \mu d/B$ predicted as $\sqrt{Npq}\,\bar Q/Q^1$ with **no free parameter**,
measured/predicted $= 1.062 \pm 0.115$. The $p$-scaling follows: inactive features have negligible
probability, so $\bar Q = pQ^1+(1-p)Q^0 \approx pQ^1$, giving $\bar Q/Q^1\approx p$ (measured
$0.229, 0.318, 0.409, 0.511$ at $p = 0.2$–$0.5$). This is why $\lambda\propto p$ and *not*
$\propto p(1-p)$: the co-active **count** drives interference, not its variance.

### 4.3 Directions — a principle, no parameters

Feature $i$ sees interference variance $p(1-p)\sum_{k\neq i}\langle w_i,w_k\rangle^2$; minimising
the total at fixed norms means minimising $\|WW^\top\|_F^2$, whose optimum is the tight frame
$\sum_iw_iw_i^\top\propto I$. That exact optimum is **infeasible** here (it needs
$\max_i\|w_i\|^2\le B/d$, violated by the top feature), and the trained solution sits $12$–$25\%$
above the interference minimum. As a *generating rule* it still works.

**What is optimised:** over unit directions $\hat u_i$ with norms held at their predicted values,
minimise $\sum_{i\neq k}\langle w_i,w_k\rangle^2$ by L-BFGS. Non-convex, so run several random
starts and **select by CE** on the task target — a task quantity, no fitted parameter. Selection
matters: candidates span CKA $0.859$–$0.879$ at $d=8$ while hitting identical interference.

| directions | $d{=}3$ | $d{=}6$ | $d{=}8$ |
|---|---|---|---|
| random | 0.794 | 0.740 | 0.804 |
| minimum interference | 0.872 | 0.940 | 0.879 |
| true directions | 0.917 | 0.913 | 0.932 |

### 4.4 The one fitted constant

$A$ is a **finite-capacity correction**, measured as $A = k^2\sqrt{Npq}$ where $k$ comes from a
least-squares fit of $\|w_i\|$ to $k\sqrt{(\alpha_i-\mu)^+}$:

| $d$ | 2 | 3 | 4 | 6 | 8 | 12 |
|---|---|---|---|---|---|---|
| $A$ | 2.46 | 2.25 | 2.37 | 2.04 | 1.79 | **1.67** |

$A$ declines with capacity (2.4 to 1.7 over $d = 2$–$12$) but does **not** reach 1, so the derived
amplitude of §4.1 is not recovered even at $d=12$. Seed agreement $\approx 0.05$.

*Do not confuse with the $A$ of the ruled-out three-parameter form* (§9), which reads $2.18 \to 1.03$
over the same range and appears to approach 1. That fit is unidentifiable and its apparent limit is
not evidence.

---

## 5. The pipeline

| step | operation | fitted? |
|---|---|---|
| 1 | solve $\mu = \sqrt{Npq}\,(\bar Q/Q^1)\,B/d$ with $B=\sum_i A(\alpha_i-\mu)^+/\sqrt{Npq}$ | $A$ only |
| 2 | $\|w_i\|^2 = A(\alpha_i-\mu)^+/\sqrt{Npq}$ | — |
| 3 | directions: minimise $\sum_{i\neq k}\langle w_i,w_k\rangle^2$, select by CE | no |
| 4 | $a_x=\sum_k w_kz_{xk}$ | no |
| 5 | $D=\arg\min\mathrm{CE}(AD^\top)$, joint convex solve | no |

Also derived, from the balanced gauge: $\|D_i\|/\|w_i\|=\sqrt{Np(1-p)}$
($\mathrm{corr}(\|D\|,\|w\|)=0.97$–$0.98$), with $\cos(w_i,D_i)=0.944$ so
$\langle w_i,D_i\rangle\approx\|w_i\|\|D_i\|$.

---

## 6. What each part predicts, and how well

| part | prediction | measured |
|---|---|---|
| §4.2 ($\mu$) | which features are encoded (head size) | exact or within one, $d=2$–$12$ |
| §4.2 ($\mu$) | the threshold value | 2.0% mean error, 7.3% max |
| §4.1–4.2 | full encoding-weight profile | $\mathrm{corr}=0.986$ |
| §4.1 ($\sqrt{Npq}$) | scaling with sample size and sparsity | ratio $0.99$–$1.02$ |
| §4.3 | code orientation | Gram at/above ceiling; covariance spectrum too flat |
| §3 | decoder from the code | CKA $1.000$ / $0.907$–$0.984$ |
| §3.2 | tail decoders collinear | $0.976$ on one PC at $d=3$, $0.539$ by $d=12$ |
| §3.2 | which features pair antipodally | sign flip at $c^\ast$, confirmed |

**End to end**, running §5 exactly as written, against five-seed ceilings:

| $m$ | $d$ | $p$ | $\mu$ true | $\mu$ pred | err | $\mathrm{corr}(w)$ | dec CKA | ceiling | frac |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 3 | 0.3 | 2.897 | 2.873 | 0.8% | 0.991 | 0.841 | 0.905 | 93% |
| 64 | 6 | 0.3 | 2.214 | 2.171 | 1.9% | 0.984 | 0.908 | 0.961 | 95% |
| 64 | 8 | 0.3 | 1.970 | 1.865 | 5.4% | 0.979 | 0.853 | 0.916 | 93% |
| 64 | 12 | 0.3 | 1.638 | 1.565 | 4.5% | 0.971 | 0.764 | 0.914 | 84% |
| 64 | 3 | 0.2 | 2.634 | 2.541 | 3.5% | 0.990 | 0.838 | 0.918 | 91% |
| 64 | 3 | 0.5 | 3.362 | 3.394 | 1.0% | 0.994 | 0.896 | 0.994 | 90% |

Baseline (closed-form PCA): $0.363$. $\mu$ error 2.8% mean / 5.4% max,
$\mathrm{corr}(\|w\|) = 0.985$.

**A caveat worth stating.** An earlier variant reached $0.872/0.940/0.879$ (96/98/96%) at
$d = 3/6/8$ using a *single* $A \approx 2.18$ calibrated once, rather than the per-$d$ table of
§4.4. Substituting the measured per-$d$ values makes the decoder **worse**, most at $d=12$ where
$A$ differs most ($1.67$ vs $2.18$) and CKA falls to $0.764$. The same effect appeared earlier:
supplying true per-$d$ constants scored below a single calibrated one ($0.855$ vs $0.872$ at $d=8$).
So the better headline number rests on a constant that outperforms the one actually measured, which
is not understood and should not be presented as the method working better than it does.

**Activations specifically:** Gram/CKA $0.637, 0.680, 0.754$ against ceilings $0.584, 0.644, 0.702$
— at or above. RDM/RSA and pairwise distances likewise. The exception is the covariance spectrum,
predicted to $0.09$–$0.11$ against a seed-to-seed floor of $0.02$–$0.05$, because minimum
interference gives a frame that is too isotropic ($[0.405,0.297,0.297]$ predicted vs
$[0.537,0.336,0.126]$ measured at $d=3$).

---

## 7. How things are fitted

- **$A$:** least squares of $\|w_i\|$ (from regressing the trained code on $Z$) against
  $\sqrt{A(\alpha_i-\mu)^+/\sqrt{Npq}}$, with $\mu$ from the self-consistent solve. One number,
  calibrated once; $d$-dependence tabulated in §4.4.
- **$\mu$:** *not* fitted in the pipeline — solved from §4.2. Fitted only when measuring $A$.
- **Directions:** optimised, not fitted. No free parameters; multiple random starts, selected by CE.
- **Everything else:** derived.

---

## 8. Regime boundaries and stress tests

**$m/d \gtrsim 10$.** The $\delta$ cliff has ratio $2.0$–$3.0$ at $m/d=21.3$ for $d\in\{3,6,12\}$,
falling to $1.44$ at $m/d=5.3$. Holding $m$ fixed while raising $d$ confounds the two.

**Sparsity $p$ — degrades gracefully, no hard boundary.** Running the §5 prediction:

| $p$ | $\max p$ | cliff | $\mu$ err | $\mathrm{corr}(w)$ | $R^2$ form | seed ceiling |
|---|---|---|---|---|---|---|
| 0.05 | 0.220 | 1.03 | 13.5% | 0.974 | 0.982 | **0.238** |
| 0.10 | 0.327 | 1.32 | 9.7% | 0.982 | **0.984** | 0.708 |
| 0.20 | 0.445 | 2.58 | 2.6% | 0.991 | 0.979 | 0.963 |
| 0.30 | 0.527 | 2.01 | 0.1% | 0.992 | 0.975 | 0.908 |

**The low-$p$ boundary is the target vanishing, not the model failing.** The encoding profile stays
predictable throughout ($\mathrm{corr}(w) = 0.992 \to 0.974$ over a 6x drop in $p$) and the
water-filling form fits *best* at low $p$. What collapses is the **seed ceiling**: $0.963 \to 0.708
\to 0.238$. Below $p \approx 0.2$ trained networks stop agreeing with each other, so there is no
well-defined geometry to predict. $\mu$ degrades in step ($2.6\% \to 13.5\%$).

An earlier version of this file called $p<0.2$ "outside the regime" on the basis of *fit*
instability ($A$ = 8.56 vs 2.61 across seeds). That is a different claim from the prediction
failing, and conflating them was an error.

**Importance spectrum — survives.** $\beta$ from $0.3$ to $1.0$: $\mu$ error $3.4$–$3.9\%$,
$\mathrm{corr}(w) = 0.967$–$0.978$.

**Correlated features — the code survives, the tail closed form does not.** One-factor and
block Gaussian copulas at $\rho = 0.6$, preserving Bernoulli($p$) marginals:

| structure | $\mu$ err | $\mathrm{corr}(w)$ | tail closed form | tail 1-PC | $\|v^1{-}v^0\|$ | tail ceiling |
|---|---|---|---|---|---|---|
| iid | 0.1% | 0.992 | 0.984 | 0.976 | 0.109 | 0.943 |
| equicorrelated | 0.8% | 0.993 | 0.917 | 0.976 | 0.587 | 0.975 |
| **blocks = 4** | 5.3% | 0.983 | **0.741** | **0.629** | 0.988 | 0.996 |
| blocks = 16 | 7.9% | 0.982 | 0.956 | 0.744 | 0.188 | 0.966 |

*Equicorrelation does not break §3.2, and the reason matters:* it shifts every feature's $\delta_i$
by the same amount, and $D_i = M_Z^{-1}(c_iv-u)$ has a free common term $u$ that absorbs a common
shift exactly. **Block structure closes that escape hatch** — at 4 blocks the closed form drops to
$0.741$ against a ceiling of $0.996$, and the collinearity it predicts collapses ($0.976 \to 0.629$).
Damage peaks at *intermediate* block counts, where correlation is both strong and heterogeneous;
16 blocks of 4 features is close to iid in aggregate and recovers.

**What correlation changes, mechanically.** For binary features,
$P(z_k{=}1\mid z_i{=}1)-P(z_k{=}1\mid z_i{=}0)=C_{ik}/p(1-p)=\rho_{ik}$ exactly, so a tail
feature's leak into the code is

$$\delta_i = v_i^1-v_i^0 \approx \sum_k \rho_{ik}\,w_k .$$

Independence sends every $\rho_{ik}\to0$ and recovers §3.2. Equicorrelation makes $\delta_i$ the
*same* vector for every $i$, which the free $u$ term absorbs — hence no damage despite
$\|\delta\|=0.587$. Blocks make it block-specific, and nothing absorbs that. Blocks $=16$
recovers because with 4 features per block and a head of 7, most blocks contain no head feature,
so $\delta_i\approx0$ there. **Untested:** whether feeding this $\delta_i$ into the §3.1 target
lifts the blocks $=4$ tail back toward its $0.996$ ceiling.

**Summary of scope.** §4 (the code) is robust to spectrum, to $p$ down to $0.1$, and to every
correlation structure tested ($\mathrm{corr}(w) \ge 0.982$ throughout). §3.2 (the tail closed form
and its collinearity prediction) requires iid or equicorrelated features.

## 8b. Figures

`src/notebooks/feature_cube_analysis.py` renders the derivation; each figure states its
acceptance criterion in the markdown cell above it.

| figure | claim it tests | section |
|---|---|---|
| 1 water filling | $\|w_i\|^2 \propto (\alpha_i-\mu)^+$, hard cut at $\mu$, $\mu$ falls with $d$ | §4.1–4.2 |
| 2 predicted decoder | closed-form decoder reaches the seed-to-seed CKA ceiling | §3, §5 |
| 3 tail collinearity | tail decoders lie on the line $M_Z D_i = c_i v - u$ | §3.2 |
| 4 scope | code survives sparsity and correlation; tail closed form breaks on blocks | §8 |

## 9. Ruled out

Each looked plausible and cost real time.

- **$k=\sqrt{p(1-p)/2}$.** Matched to three decimals at $p=0.3$; a $p$ sweep showed the two are
  *anti*-correlated ($-0.917$).
- **$k^2\sqrt{Npq}=2$.** Artifact of forcing $(\alpha-\mu)^+$ onto a hard-truncation profile.
- **$C_B$, $C_\rho$ as constants.** $C_B = R^2(\text{code on }Z)\times(\text{centred fraction})$ to
  $1.4\%$; both absorbed $N$- and $m$-scaling the gauge explains.
- **A three-parameter $A(1-(\mu/\alpha)^g)^+$ profile.** Fits better ($R^2$ $0.97$ vs $0.95$; $0.99$
  vs $0.84$ at $d{=}12$) but is **unidentifiable** — same config, different seeds give
  $(A,g) = (3.9, 0.5)$ and $(1.3, 12.0)$. Only identifies once $\mu$ is pinned, which is circular.
- **The code's nonlinearity is $1/W(x)$.** $\log W$ is a uniform shift across outcomes; the softmax
  gauge deletes it ($R^2$ drops $0.86\to0.61$).
- **Pairwise head interactions matter for the decoder.** They explain the code's $R^2$ residual
  ($0.86\to0.96$) but the *true* coefficients do not improve the decoder ($0.917\to0.882$).
- **Amplification is the optimal temperature of the truncation.** $t^\ast\approx0.93$ (shrinkage)
  against measured $1.2$–$2.1$.
- **The model preserves prediction sharpness.** $H(q)/H(p)=1.13\to1.04$ as $d$ grows.
- **Amplification tracks target variability (Jensen).** Right shape, wrong magnitude — CV saturates
  ($0.530\to0.558$) while $A$ keeps rising.
- **Odds-ratio matching sets the amplification.** Gives $\sigma_i=\alpha_i-dW+dZ$ with $dZ<dW$
  everywhere: predicts attenuation, wrong sign.
- **Neural collapse / simplex ETF.** Mean pairwise $\cos=+0.49$ to $+0.56$ among argmax winners
  against an ETF target of $-0.05$.
- **Capacity is a conserved budget** (Rosetta's $\sum_r s_r\le\kappa N$). Output-side
  $\sum_i\|D_i\|^2$ grows $32\%$ while $d$ triples; the budget is on the *input* side.
- **Rosetta's $\log(1+s_r)$ utility.** Right for the head ($R^2=0.995$ on the top feature) but the
  true utility is non-monotone — it peaks and turns negative, each feature sitting at its own peak.

---

## 10. Multiplicity

Trained networks occupy genuinely distinct local optima spanning $3.8\%$ in CE, none improvable by
joint L-BFGS. **Within** one optimum the solution is pinned — perturb by $100\%$ of its own scale
and re-converge, code CKA returns to $0.972$. **Across** optima the decoder agrees at $\approx0.80$,
the code at $\approx0.58$. So the ceiling measures optimum *selection*, and the code is its most
sensitive readout, which is why the decoder is the better target.

Alternating the two convex solves from a closed-form start is deterministic and beats SGD on CE, but
that is an algorithm. The content is §4, which predicts the code without solving for it.

---

## 11. Open

- **$A$ at small $d$.** No derivation; two attempts failed (§9). It vanishes as $d$ grows, so it
  matters least in the regime of interest.
- **Code orientation.** Minimum interference gives too isotropic a frame; the covariance spectrum
  is the one quantity clearly below its noise floor.
- **Scope.** One spectrum family ($\beta=0.5$; geometric $0.9/0.99$ in §3), $\texttt{scale}=8.0$
  unless stated, i.i.d. features, linear readout. Correlated features, other spectra, other
  sharpness, and non-linear architectures are all untested. Correlated features and other spectra
  *are* now tested — see §8.
