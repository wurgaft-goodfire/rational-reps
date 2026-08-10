# Two vectors per concept: embedding-side vs decoder-side

Every concept in a linear-softmax pipeline has **two** natural vectors, and most interpretability
practice quietly assumes they are the same one. This note states the two objects, the theorem
that makes them coincide in symmetric worlds, the minimal example where they split, and what the
split does to probes and steering.

## 1. Setup: three vector families

Factor the environment's centered log-conditional: `log p(x | i) = ⟨e_i, d_x⟩` (+ per-row
constants). Inference over sources given iid observations `x_1..x_t` is then linear in one state:

```
log w(j)  =  ⟨ e_j , S ⟩ + const,        S = Σ_t d_{x_t}
```

Three families of vectors, one per concept/outcome:

| symbol | definition | role |
|---|---|---|
| `d_x` | column of the factorization | **write**: what one observation of x adds to the state |
| `e_j` | row of the factorization | **read**: the linear functional that scores hypothesis j; also the output code when resolved |
| `m_j = Σ_x p(x|j) d_x` | expected write under source j | **rest**: where the (normalized) state converges when the source is j |

One identity that holds in *any* geometry (why Bayes never breaks):

```
⟨ e_j , m_i ⟩ = Σ_x p(x|i) log p(x|j) = −H(p_i) − KL(p_i ‖ p_j)   →  argmax_j = i.
```

## 2. The symmetric case: when the space is self-legible

On a homogeneous symmetric world (the ring), two theorems align all three families:

1. **e = d** (matrix symmetry): a node's concept point equals its prototype.
2. **m_j = ρ·e_j** (shift theorem): the resting point lies on the concept's own ray, shrunk by
   the kernel's Fourier coefficient ρ < 1.

Corollaries — the folk-geometric readings, as theorems:

- **Radius = confidence.** With equal norms `‖e_j‖ = a`, the posterior is
  `w(j) ∝ exp(a‖S‖·cos(θ_j − φ_S))` — a von Mises over concepts with concentration `κ = a‖S‖`.
  The state's angle picks the winner; its radius *is* the posterior concentration.
- **Nearest-neighbor validity.** `‖S − e_j‖² = ‖S‖² − 2⟨e_j, S⟩ + ‖e_j‖²`: minimizing distance
  equals maximizing the correct score **minus a penalty `‖e_j‖²/2`**. Flat norms ⇒ proximity is
  the Bayes classifier.

Both readings are licensed by the symmetry, not by the linear algebra.

## 3. The minimal split: two correlated binary features

Entities carry features `x = (x₁, x₂) ∈ {−1,+1}²`; a query about feature f must be answered
with log-odds `c·x_f`. Centered conditional: `Lc[entity, (f, y/n)] = ±(c/2)·x_f`.

Two candidate "concept vectors" for feature f:

- **decoder / read**: `r_f = d(yes_f) − d(no_f)` — the direction the answer consults;
- **diff-in-means / write-side**: `Δe_f = E[e | x_f=+1] − E[e | x_f=−1]` — what every
  mean-based probe recipe estimates.

**Independent, balanced features (ρ = 0): `Δe_f ∥ r_f` exactly.** One orthogonal axis per
feature; embeddings at square corners; the textbook linear-representation world.

**Correlated features, `P(x₁=x₂) = (1+ρ)/2`: they split.** The SVD axes become the principal
axes of the feature covariance (shared axis weighted 1+ρ, contrast axis 1−ρ), and the
diff-in-means vector absorbs the covariance once more than the decoder direction does.
Measured angle between `Δe₁` and `r₁` (n = 20k entities, balanced factorization):

| ρ | 0.0 | 0.3 | 0.6 | 0.9 |
|---|---|---|---|---|
| angle | 0° | 8° | 16° | 19° |

(Exact closed form: TODO — requires careful √σ bookkeeping through the balanced split and the
row/column measures; first attempt was wrong, numbers above are the ground truth.)

**Mechanism.** Diff-in-means reports the feature *as it co-occurs*: carriers of x₁ are
typically also carriers of x₂, so the mean difference tilts toward the shared axis. The decoder
reports the feature *as it must be answered*: a query about x₁ must come out right even on
atypical entities, so it stays disentangled.

**Where it bites: the atypical entity** `(x₁=+1, x₂=−1)` (rare when ρ is large). The
diff-in-means probe under-scores feature 1 on it — its projection is dragged down by the
negative partner-feature component. The decoder answers it correctly. Probe generalization
failure on atypical feature combinations *is* the split.

## 3b. Rows, columns, and types: why a property is a column difference

Write the factorization elementwise: `log p(x|i) = Σ_j E[i,j]·D[j,x]`.

- **Column of D** (`d_x ∈ R^k`): the k coordinates of one outcome — a point in the space.
  This is the **decoder vector** for x: the unembedding direction the readout dots the code
  against to produce x's logit.
- **Row of D** (`φ_j : X → R`): the j-th coordinate of *every* outcome — a function over
  outcomes, not a vector in the concept space; its entries are single coordinates of points.
  A row is not a decoder vector for anything.
- **Property direction**: for a binary assertion with outcomes yes/no, the operational vector is
  the **difference of two decoder columns** `u = d_yes − d_no ∈ R^k`, since `⟨e_i, u⟩` is the
  log-odds of the answer. Differences are forced: under a softmax the columns are defined only
  up to per-concept shifts — outcomes live in an affine space, and only differences are honest
  vectors. No single decoder vector is the property; the contrast between two answers is.

In a disentangled world (independent factors, one assertion per factor), the row for a factor is
supported exactly on its own assertion columns, so `u` points along a single axis and the row
serves to *name* that axis. With correlation or shared outcomes, `u` remains well-defined (and
Bayes-correct) but aligns with no axis and no row — "female" is then a generic direction of the
space, which is the honest general case.

## 4. Cousins of the same split

- **Base-rate asymmetry (one rare feature)**: norms split rather than angles — the rare
  feature's vectors lengthen (diagnosticity > typicality) but stay collinear (one feature = one
  dimension). Chain-of-graph boundary: both at once, with a spatial ordering
  (cos(e_i, d_i): 1.00 in the bulk → 0.89 at the ends; ‖d‖/‖e‖ → 1.3).
- **Continuous families (Gaussian)**: the extreme case — evidence direction `(x, x²)` vs
  certainty ray `(2x, −1)`, nearly orthogonal.

## 5. Consequences for probing and steering

Both standard probe recipes are linear classifiers `argmax_j (w_jᵀ S + b_j)`; they differ in
which family the weights estimate:

| recipe | weights learned | valid when |
|---|---|---|
| diff-in-means / class-centroid / LDA on activations | `m_j` (resting points) | `m ∝ e`: symmetric, decorrelated, flat-norm worlds |
| decoder rows / unembedding directions | `e_j` (read directions) | always (the Bayes readout) |

Nearest-prototype classification carries the `−‖e_j‖²/2` bias: it votes against long-norm
(diagnostic/boundary/rare) concepts even when the inner product favors them.

Steering: three different interventions that coincide only under symmetry —

- add `d_x`: simulate *one observation* of x;
- add `m_j`: simulate *typical accumulated evidence* for j;
- add `e_j` (unembedding direction): manipulate *the score/belief* of j directly.

**Weight tying** (input embedding = unembedding) is the architectural assertion `e = d`:
optimal in homogeneous statistics, and predictably wrong — write-vectors longer than and
rotated from read-vectors for rare/boundary/correlated concepts — the moment the statistics
have structure.

## 6. Open items

- Exact angle formula for the correlated-feature split (and its maximum over ρ).
- Empirical LLM test: compare diff-in-means vs unembedding directions for the same concept as a
  function of concept frequency/correlation; prediction: angle grows with correlational
  structure, norm ratio grows with rarity.
- Notebook figure for claim 3 built around *one broken reading* (nearest-neighbor decision
  regions on the chain diverging from Bayes near the boundary) rather than the raw split.
