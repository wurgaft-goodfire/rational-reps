# Relational settings: candidates and what the oracle predicts

Two inputs, one conditional `T[a, b, c]`. The oracle factors the row-centred log-conditional; the
predicted code is whatever the log-conditional is linear in. This note records every candidate we
considered, what it predicts, and why each was kept or dropped.

## The two criteria that did the work

**Operands, not samples.** A *relation* combines two operands by an operation (`a ∧ b`, `LCA(a,b)`).
If the two inputs are instead exchangeable *samples from a common source*, the task is evidence
accumulation — inference with `n = 2` — and belongs in the inference section. Commutativity does not
distinguish them: `∧`, `max`, `LCA`, `+` are all symmetric.

**Derived grading, not a chosen kernel.** A delta target says only "which class", and one-hot
outcomes carry no metric, so the pair geometry is a size-weighted simplex with no metric content
(verified: every entry of the pair Gram is a function of the two classes' sizes alone, residual
1e-15). Grading fixes that — but only if the spread follows from what the outcome *is*. Placing a
Gaussian at chosen outcome positions reads back the assumption: choose helical positions for sums
and you predict a helix, choose a line and you predict a line (both R² = 1.00 against their own
assumption). Set-valued answers grade for free, because "give me an element of the result" has
uniform as its answer when the set is all you know.

## Kept

Every scored relation is graded: a delta answer makes all outcomes equidistant, so its pair code
is a simplex with no geometry to predict. Modular addition is that case, and survives only as the
factorization figure, where the DFT gauge is the point.

| relation | operands | answer | predicted geometry |
|---|---|---|---|
| `approx_add` | two magnitudes | `a + b` under Weber noise | a line: rank 1, code = `log(a+b)` |
| `pitch_transpose` | note, interval | the transposed note, confusable in height and in chroma | a helix: rank 3, one linear axis plus a degenerate cos/sin pair |
| `between` | two positions | the interval `[min, max]` | intervals by midpoint; least compressible of the six (eff-rank 6.7, null 0.11) |
| `feature_taxonomy` | two consistent feature sets | is a drawn feature shared? | a grid, one axis per chain — the nesting is what stops it being a cube |
| `set_difference` | two subsets | is a drawn feature in `a` but not `b`? | a box: rank nbit, axes scaled by salience, and asymmetric |
| `entailment` | two tree nodes | a common ancestor of `a, b` | the tree, read as an ultrametric over the leaves |

Two design rules earned by measurement:

* **Ask about every feature, not the answer's members.** Sampling from `a ∧ b` makes a one-member
  answer a sharper row than a three-member one, which stretches the cube and needs a null outcome
  to reach rank nbit at all. Asking "draw a feature — does the predicate hold?" weights every
  coordinate equally and the cube comes out exact (edges 1.000 / 1.412 / 1.728).
* **The predicate cannot change the geometry.** In that form `∩`, `∪` and `⊕` all measure
  identically (spectrum 1, 1, .84, 0), because each feature contributes an independent binary
  coordinate. Only *dependencies between features* break the cube — which is what
  `feature_taxonomy` adds, and why it is numerically identical to gcd on the divisors of 36.

`entailment` uses the uniform-over-common-ancestors form, not the Bayesian one, to keep it a relation.

`∧` and `△` are the two halves of **Tversky's contrast model** —
`sim(a,b) = f(A∩B) − α f(A−B) − β f(B−A)` — common features and distinctive features in one feature
space, so a model that represents one and not the other is a finding. `△` also does double duty: it
is addition in ℤ₂ⁿ, so as a *class label* it would be flat-spectrum and gauge-only exactly like
modular addition, while as a *set* it is rank nbit with the code pinned to the bits. That is the
appendix's claim shown on a single operation instead of across different ones.

All four set operations give the *same* prediction (rank 3 at nbit=4, eff-rank 3.0, 15 distinct
targets, code = the result's bits), and two are outright redundant: union is intersection under bit
complement, and `a ∧ ¬b` is intersection with `b` complemented. So the gallery takes two of the four.
`a ∧ ¬b` was the only non-commutative candidate, so nothing in the final set tests whether the code
encodes operand order — a deliberate omission for tightness.

A retraction worth recording: there is no computational-difficulty gradient across the four in this
model set. Per bit, `AND = a·b`, `OR = a+b−ab`, `XOR = a+b−2ab`, `DIFF = a(1−b)` — all bilinear, none
linear — so all four are equally hard for a linear-concat readout and equally easy for a bilinear
one. With `pair_inputs` the linear model has a free row per pair, so it is a lookup table and learns
any of them trivially.

Approximate addition earns its place because the log geometry is *not* smuggled in: outcome
positions are the plain integers, and swapping Weber noise (proportional to magnitude) for
constant-width noise flips the prediction to a linear line (R² = 1.0000 either way). The
psychophysical law does the work, and it is measured rather than chosen.

## Moved to the inference section

All of these are exchangeable samples, not operands.

| task | reference | note |
|---|---|---|
| Bayesian generalization, interval hypotheses | Shepard 1987; Tenenbaum & Griffiths 2001 | rank 9, eff-rank 3.6, effectively 2-D: location + breadth. Size principle `1/|h|²` |
| Bayesian generalization, subtree hypotheses | Xu & Tenenbaum 2007 | rank 7, eff-rank 3.2, spectrum paired by tree level |
| the number game | Tenenbaum 1999 | mixed hypothesis kinds; iid accumulation |
| key finding from two pitches | Krumhansl & Schmuckler | rank 11, paired cyclic spectrum (the key torus), 78/144 distinct |
| blicket detector | Gopnik & Sobel 2000 | single trial is degenerate: target is just the indicator of `{a,b}`, spectrum flat. Backward blocking needs two trials |

The belief itself is a poor target: outputting the posterior over hypotheses is essentially full rank
(14 of 15, 54 of 55), because "which hypotheses contain both" is a conjunction whose patterns span
the space. The posterior *predictive* is what creates low-rank structure, since it projects a
55-dimensional belief down onto the items. The softmax then factors `log(A·belief)` rather than
`A·belief` — one log in the way — which sharpens the spectrum rather than blurring it (2nd singular
value 0.79 vs 0.90) but costs the closed form.

## Dropped, with the reason

| candidate | why |
|---|---|
| `set_intersection` | in the flag form every bitwise predicate gives the identical cube, so it duplicated `set_difference` — which keeps the slot for being asymmetric |
| `common_divisors` | eff-rank 7.9 at n=12 and *worse* as n grows (21.1 at n=30): more primes mean more near-orthogonal directions. Its grid content survives as `feature_taxonomy` |
| `maximum` | throws away one operand: 64 pairs collapse onto 8 targets, and holding `a` fixed every smaller `b` gives the same answer. Replaced by `between` |
| `standard_mult` | 42 nearly-unique products, so the spectrum is flat (eff-rank 40). Multiplication is addition in log space only on the *exponent lattice*, and the prime exponents explain just 0.05 of the code. Replaced by `common_divisors` |
| `standard_add` | sums have no canonical associated set, so no derived grading |
| `averaging` | the Gaussian at the midpoint is a chosen kernel. Was the only graded relation; replaced |
| product of experts / convolution / mixture | vector addition in disguise; requires choosing the item distributions. PoE is exactly rank 2 with the code = resultant vector, but the geometry comes from the choice |
| number line with digit-wise outcome positions | predicts a helix because helical positions were assumed. Note the *finding* is still real: addition is helical iff the outcome is tokenized digit-wise (rank 3, R² = 1.0000), linear if tokenized as magnitude (rank 2) |
| scalar MSE (predict a number, squared error) | graded by definition, but the decoder is one row so the predicted code is rank 1 — no geometry left. Needs a vector-valued target |
| mental rotation | output is binary (same/different) or a group element (degenerate like modular addition); its content is the RT slope, about process not representation |
| Hinton family trees | delta and mostly empty: with gendered relations only 4 of 144 `(person, relation)` cases are multi-answer, and `(person, person) → relation` has **zero**. Still valuable as a benchmark against a known result — see below |
| genealogy DAG common ancestors | 12 distinct targets of 121 pairs; needs a much larger pedigree |
| analogy `a:b::c:?` as a target | eff-rank 13.1, 22 distinct targets of 490: `a:b` only selects which of 3 relations, so most of the input is redundant. Analogy belongs as a *probe* on item geometry (the parallelogram), not as a training target |
| chord completion | genuinely relational and always 2 answers (72 answerable pairs, all multi-answer), cyclic paired spectrum — but eff-rank 10.4 and 24 distinct targets. Charming, uncompressed |
| category-based induction | mathematically identical to Bayesian LCA; the diversity effect *is* the size principle. A framing plus human data, not a new task |
| Rogers & McClelland semantic network | already covered by the static diffusion-tree setting |

## Hinton family trees as a benchmark, not a relation

Hinton 1986: two isomorphic families (English, Italian), 12 people each, 12 gendered relations,
**104 triples** — reproduced exactly. Inputs are heterogeneous (person × relation), unlike every
other relation here. Factoring the `person × relation × person` tensor gives two codes with
different predicted content, both checkable against a published result:

* **person code** — generation first (R² 0.91 by 4 dims), then nationality and branch (0.83 by 12),
  and **sex nearly absent (0.19)**. Hinton reported nationality, generation, branch — and not sex.
* **relation code** — blood vs marriage dominates (0.94 by 6 dims), ahead of generation shift (0.59)
  and the gender of the answer (0.50).

Both are high eff-rank (18.9, 10.6), so the claim is an *ordering* over which features load first,
not a geometric shape. Measure decodability (feature from code), not the reverse: asking "is the
code a function of nationality" regresses 1 predictor on 6 targets and understates it badly.

## The lapse rate

A target with zero-probability outcomes is not identified by the loss: `KL(p || q)` weights an
outcome by its true probability, so an impossible one is unscored and its logit free. These
relations answer with a set, so 55-92% of the entries are zeros, and `E*` — built from the whole
matrix — was mostly encoding structure nothing constrained. Measured on the six: logit R² averaged
0.475 (as low as -0.98) at KL ~1e-6.

Mixing the conditional with the uniform (`lapse: 0.05`, "with 5% probability the response is
random") makes every outcome possible without moving the prediction: `CKA(E*(0.05), E*(0)) > 0.99`.
After it, logit R² is 0.997 and CKA against `E*` rises 0.813 -> 0.956, with the identifiability
term of the decomposition at 0.998 — so the residual gap is the balanced-sqrt(S) split alone.

It matters most for modular addition. With a delta target one frequency sufficed (a 2-D code
determines `s = a+b`, and scaling the readout drives KL to 0), so architectures truncated to an
arbitrary irrep and `mlp_paired` sat at circulance 0.26-0.45. The lapse closes that route — the
best rank-2 solution bottoms out at KL 0.56 — and every architecture now carries every irrep at
equal energy (`mlp_paired` circulance 0.95-0.99).

## Still open

* A sixth relation. The gap is a cognitively real relation that is operand-based, derives its
  grading, and compresses (low eff-rank). "Shared features in a real feature space" is the shape
  that keeps surviving — phonological distinctive features, colour terms, semantic feature norms —
  each the same operation as `set_intersection` in a different space, and each needing published
  feature data so the geometry is an empirical claim rather than a structural one.
* Whether `entailment` should use the uniform or the Bayesian (size-principle) spread; the Bayesian
  version is a stronger prediction (eff-rank 3.2) but its generative story is sampling.
