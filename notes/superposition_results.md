# Superposition: where the results stand, and what's next

Companion to `superposition.tex` (the theory). Setting: m = 64 Bernoulli(0.3) features,
α_i = 8 (i+1)^{-1/2}, N = 2048 states, d ∈ {3, 6, 12}, five runs per d with task and training
seeds coupled. Numbers below are 5-seed means at d = 3 / 6 / 12 unless stated.

## The pipeline as it stands

1. **Allocation** — water filling (`allocate_d`): d_i vs trained leverage 0.98 / 0.99 / 0.99.
2. **Orientation** — the linear Fantope program on the compression pressure a_ij, diagonal
   free, then a fixed-diagonal realization (`fantope_geometry` + `tight_frame(init=P)`):
   placement (off-diagonal correlation, top-16 whitened Gram, seed means) 0.61 / 0.72 / 0.33
   against a pairs-class ceiling of 0.85 / 0.64 / 0.43. Produces the soft tail.
3. **Completion** — decoder and pair block solved exactly at the fixed frame by convex block
   alternation (`repair`): CE 0.316 → 0.228 / 0.205 → 0.143 / 0.145 → 0.092 (trained
   0.207 / 0.130 / 0.077); head sizes 11.5 / 10.2 / 9.0 vs trained 11.2 / 11.0 / 9.7.
4. **Benchmark** — the additive-plus-pairs class optimum, which the SGD pairs runner
   (`runner=paired_features`) reaches (biconvex refinement moved its CE in the fourth decimal):
   reproduces the trained network's CE within ~3% and its placement at 0.85 / 0.64 / 0.43.

## What is established

- **Allocation is water filling.** Pricing the interference (a certified toll) is consistent
  but does not improve it (0.971 / 0.970 / 0.972 vs 0.977 / 0.993 / 0.989). The toll is small
  (≈ 0.02–0.04) because CE charges interference at the victim outcome's softmax variance ∼ 1/m.
- **Signs and placement come from the compression residual.** a_ij (exact first derivative of
  the tied-map CE at the fractional predictor) predicts trained overlap signs at 90 / 82 / 64%;
  trained frames sit at percentile ≈ 0 of the pressure score over random compatible frames.
  Coarse sign is generic to compression (the quadratic-loss analogue agrees); graded
  placement is softmax-specific.
- **Placement is selected by the joint optimum.** With the frame held fixed, the class's loss
  is flat across placements (trained / Fantope / random frame: 0.243 / 0.232 / 0.235 at d=3)
  because decoder and pairs compensate. With frame, decoder and pairs optimized together, the
  pressure placement wins: gradient descent (the runner's recipe) started from the
  *anti-pressure* frame abandons it within ~1000 steps and ends at the pressure pattern
  (placement vs trained −0.39 → +0.53 at d=3; corr with its own init ≈ 0.1) at the common loss
  0.212, below every fixed-frame value; exact block descent from the same start stays in the
  anti basin at 0.247. The analytic pressure is the handle on this optimum because the
  additive class's own CE points the other way (anti-aligned at −0.25): it cannot repair what
  pressure-following frames create. Loose end: a start *at* the pressure frame with the tied
  decoder drifts to only +0.08 vs trained at d=3 while staying near its init (0.73) — the tied
  start over-shoots the head bias and GD rebalances it.
- **Soft tail.** Features water filling drops carry small anti-aligned vectors with leverage
  d_j = ‖a·_j‖² / (4δ²) (slope 2 in log–log, corr 0.92 / 0.84, no fitted constant); 45% / 34% of
  trained off-diagonal energy at d = 3 / 6 lives there. The Fantope program produces a tail
  but a smaller one than trained at d = 3, 6 (0.05 / 0.17 vs 0.31 / 0.39).
- **The completion must be solved, not linearized.** A one-step quadratic in decoder
  resizing and pair vectors at the tied baseline recovers only 0.323 / 0.211 / 0.172 (exact
  conditional 0.232 / 0.152 / 0.106) and mispredicts resizing (corr −0.64 at d=3). The pair
  sector is conditional on the decoder: the same Fisher/Newton step reproduces trained pair
  maps at 0.82 / 0.80 / 0.81 given the network's own decoder, 0.38 / 0.31 / 0.56 given the
  additive-optimal one. The trained decoder resizes the fractional mid features (s ≈ 2–3.6 at
  d=3), not the head (s ≈ 1.0).
- **Second-order corrections to the frame do not help.** The additive curvature H_gg blurs the
  Fantope solution into an unrealizable mixture (η = 0.9–3.7); the pair-integrated a_eff, H_eff
  destroys the orientation at the compressed background and is neutral at the realized one;
  integrating out resizing + pairs (Schur) reverses the pressure — expansion-point artifacts.

## Figures (notebook `superposition_analysis.py`)

Every figure has its own p / β / ρ selectors and reads the sweep grid (540 cells × linear, pairs,
additive) lazily from each run's saved evaluation (`eval.npz`: logits, representation, readout);
no model is constructed in the notebook.

Every model is put in one gauge: its logit table over states Λ = E Dᵀ (theory: [Z, φ(Z), 1]·map_rep),
row-centred, balanced SVD at rank d (`factor_logcond(Λ, rank=d)`, the repo-wide `predicted_geometry`),
then the first-order vectors E_i of the least-squares fit h(z) ≈ h̄ + Σ E_i z_i to the balanced code.
Why a common gauge: the loss is flat along E → EA, D → DA⁻ᵀ; GD from small init would balance E and D
but the runs use AdamW, which leaves the state table 4.6–7.8× larger than the readout (SGD retrain:
balanced to 0.4%). Raw scales are an optimizer artefact.

1. Allocation: Fantope diagonal (water filling + soft tail) vs trained leverage.
2. Feature sizes ‖E_i‖² in state-table units (α_i/√(Np(1−p)) as reference); one theory series.
3. Feature Grams E Eᵀ (standard display, strong diagonal): trained / pairs / additive / theory.
4. Spectrum of the first-order map E Wᵀ (gauge-free).
5. Readout Grams Dᵀ D of the balanced readout.
6. Soft tail: trained tail leverage vs ‖a·_j‖²/4δ² and the Fantope tail (log–log).
7. Gram of the regression vectors of the balanced code on [1, z − z̄, φ(z)] over the top 16 features
   and the 15 pairs among the top 6, colour scale set by the pair blocks. Trained feature–pair blocks carry definite signs at d = 3, 6, which the
   theory reproduces in broad strokes at d = 3; pair–pair blocks are near-diagonal everywhere.
   At d = 12 the theory's pair vectors anti-align systematically with specific features (a
   stripe) while trained pairs are nearly orthogonal to the features — the completion
   over-repairs there, the same regime as the tail overshoot.

## Sweep over sparsity, ladder slope and rank (2026-09-04)

`conf/setting/superposition.yaml` (a Hydra multirun per model class): p ∈ {0.05, 0.15, 0.3, 0.5} ×
β ∈ {0.25, 0.5, 1.0} × ρ ∈ {0, 0.3, 0.6} × d ∈ {3, 6, 12}, 5 seeds per cell. The theory is a model
class of its own (`runner=theory`): `geometry.rank_optimal_logits` solved per cell and stored as a
rank-d linear run beside the trained classes, read like them.
Notebook Figs 8a/8b (allocation small multiples) and 9 (score heatmaps; placement on the leverage
projectors).

- **Allocation transfers everywhere**: leverage corr 0.83–1.00 over all 36 cells; weakest at
  β = 0.25 (flat ladder) and at d = 12.
- **Soft vs hard regime.** The water filling is a fractional relaxation. Where features are dense
  (p = 0.5, d ≥ 6) or the ladder steep (β = 1), trained networks allocate hard — about d features at
  leverage ≈ 1, the rest dropped — while the theory spreads fractional leverage over ~2d features.
  Where features are sparse and the ladder flat (p = 0.05, β = 0.25) the trained tail is broader and
  noisier than predicted. The reference cell (p = 0.3, β = 0.5) sits in the soft regime the theory
  describes best.
- **Orientation is only defined in the soft regime.** Hard allocations have ~0 off-diagonals, so
  placement correlations at β = 1 are noise (−0.28 to +0.81). In the soft cells placement is +0.2 to
  +0.7 (reference +0.60 / +0.68 / +0.25 at d = 3 / 6 / 12, β = 0.5); it is ~0 for p = 0.05 at d = 6 and
  is weak throughout β = 0.25.
- **Soft-tail slope** is ≈ 2–3 only for sparse features at d ≤ 6 (p ≤ 0.15, β = 0.5: 2.3–3.3); it is
  0.5–1.8 at the reference cell and below 1 at d = 12 everywhere. The slope-2 law is a sparse,
  low-rank statement, not general.

## Correlated features (2026-09-04, in progress)

Pipeline generalised to a logit map: `alpha` may be an (m, m) matrix (`_logit_map`); all stages
take (W, X) and `whiten(Z)` gives the predictive modes u = z C^{-1/2} with W = diag(α) C^{1/2}
(`geometry.modes`). Pair features are residualised on [1, z]. Tests: whitening
identities, vector/matrix equivalence, exact no-op on the full cube, derivative identities for
a full map, pair-block orthogonality, oracle test at ρ = 0.6 (m = 16, d = 3), antipodal sign test.

- **Oracle checks (exact additive optimum by biconvex, m = 16, d = 3, n = 4096).** ρ = 0.3:
  theory CE raw 0.371 vs modes 0.317 (oracle 0.269), placement −0.03 vs +0.38. ρ = 0.6: 0.350 vs
  0.254 (oracle 0.216), placement −0.21 vs +0.42. Modes win clearly on shared-factor correlation.
- **Caveat.** A mutually exclusive pair inside iid features (m = 16, d = 3): both coordinates
  predict the antipodal sign, but raw is closer to the oracle (CE 0.269 vs 0.297; oracle 0.237).
  The relaxation is basis-dependent; modes are the right basis for shared-factor correlation,
  which the ρ sweep uses. Open: a basis choice that covers both (e.g. pick the lower completed CE).
- **Runs (Fig 10a/b): ρ ∈ {0.3, 0.6} at the reference cell, d ∈ {3, 6, 12}, 5 seeds, theory on the
  modes.** Mode-space allocation transfers (leverage corr 0.96–0.99 at every ρ). Placement
  (leverage projectors, top 16) falls with correlation: ρ = 0: +0.60 / +0.69 / +0.23 (d = 3 / 6 / 12);
  ρ = 0.3: +0.06 / +0.19 / +0.16; ρ = 0.6: +0.14 / +0.35 / +0.35. In feature space the trained Grams
  at ρ = 0.3, 0.6 stay near-diagonal with weak negative overlaps, much like ρ = 0; the theory at
  ρ = 0.6 adds a positive block among features 5–8 the networks do not have. One-factor
  equicorrelation is a weak probe of "correlated features align": every feature shares the same
  factor, so there is no specific pair to align. Block correlation is the sharper test.
- The ρ = 0 reference numbers are unchanged by the residualised pair block (placement +0.60 / +0.69 /
  +0.23 vs +0.60 / +0.68 / +0.25 before), so the earlier grid results stand.

## Exact tied-map Fantope program (2026-09-04, tested and shelved)

Replacing the linearised objective (exact CE of the diagonal + aᵀg) by the exact tied-map CE
E[CE(softmax(A P z))] over the whole Fantope is still one convex program (`tied_map_ce`,
`fantope_geometry(a=None)`), and its KKT conditions are water filling on the eigenvalues of P.
But its optimum exploits the relaxation: at the reference cell, d = 3, η jumps from 0.24 to
1.11, the realised frame's CE is worse (0.333 vs 0.323) and the placement against trained
collapses (+0.03 vs +0.57). At ρ = 0.6 it lowers the tied CE substantially (0.254 vs 0.313 raw)
but predicts no orientation either. The linear off-diagonal term is what keeps the linearised
optimum near the projectors; the pipeline keeps it. Kept as an option for the record.

Also measured: at ρ = 0.6 the trained networks have no consistent orientation across seeds
(placement self-agreement +0.23 at d = 3, +0.02 at d = 6), so orientation is not a target there;
allocation is (leverage corr 0.98–0.996 for every variant). The whitened linearised pipeline
gives the lowest realised CE at ρ = 0.6 (0.196 / 0.123 at d = 3 / 6 vs raw 0.313 / 0.211).

## Correlated features are captured, and how (2026-09-06)

The pipeline never assumed independence: every stage evaluates its objective on the data. What
correlation changes is the coordinates. The inputs are written in their predictive modes
u = z C^{-1/2} (C the feature covariance, symmetric root so mode k stays closest to feature k), and the
Bayes logits become u Wᵀ with the logit map W = diag(α) C^{1/2}: outcome i reads a mixture of modes,
which is the only structural change. Water filling, pressure, Fantope program, frame and completion run
unchanged on (W, u); trained models are measured in the same coordinates (`geometry.modes`,
`geometry.whiten`). On raw features the relaxation charges twice for a shared component (a duplicated
feature is the extreme case); on the modes it does not.

Across the full grid, theory against the additive class's own optimum (mean over p, 5 seeds each):

| ρ | β | leverage corr, d = 3 / 6 / 12 | theory CE ÷ additive KL, d = 3 / 6 / 12 | placement corr, d = 3 / 6 / 12 |
|---|---|---|---|---|
| 0.3 | 0.25 | 0.93 / 0.96 / 0.95 | 1.0 / 1.1 / 1.2 | −0.26 / 0.00 / +0.32 |
| 0.3 | 0.5 | 0.96 / 0.97 / 0.97 | 1.1 / 1.1 / 1.2 | +0.12 / +0.25 / +0.26 |
| 0.3 | 1.0 | 0.98 / 0.97 / 0.81 | 1.2 / 1.5 / 2.6 | +0.45 / +0.50 / +0.39 |
| 0.6 | 0.25 | 0.92 / 0.98 / 0.98 | 1.0 / 1.0 / 1.1 | −0.26 / +0.11 / +0.28 |
| 0.6 | 0.5 | 0.96 / 0.98 / 0.97 | 1.0 / 1.1 / 1.2 | −0.02 / +0.24 / +0.31 |
| 0.6 | 1.0 | 0.98 / 0.92 / 0.70 | 1.2 / 1.8 / 3.9 | +0.81 / +0.55 / +0.32 |

So for β ≤ 0.5 the theory reaches its class optimum to within 0–20% in CE at every ρ, p and d, with
leverage correlations 0.92–0.98: shared-factor correlation is handled by the change of coordinates and
nothing else. Orientation is weak where the ladder is flat (β = 0.25: no consistent orientation to
predict) and modest otherwise. The failure is confined to the steep ladder β = 1 at d ≥ 6, growing
with ρ and d (1.5× → 3.9×), and that is the regime analysed next.

## Where the relaxation misses its own class (2026-09-06)

Cell p = 0.3, β = 1, ρ = 0.6, d = 12, five seeds. The theory's allocation there is noisy across
seeds while every trained model is not; the analysis below is what that noise means.

| | excess CE / KL | feature-space leverage, features 0–15 | seed SD (features 4–15) |
|---|---|---|---|
| network (free state table) | 0.0022 | 1.0 × 12, then 0.87, 0.05, 0.07, 0.01 | ≤ 0.05 |
| additive model (the theory's class) | 0.0022 | 1.0 × 12, then 0.79, 0.19, 0.05, 0.01 | ≤ 0.05 |
| pairs model | 0.0023 | same hard profile | ≤ 0.07 |
| theory, predictive modes | 0.0098 ± 0.0005 | 0.97 0.96 0.95 0.9 0.49 0.12 0.04 0.2 0.25 0.01 … 0.55 | 0.12 – 0.26 |
| theory, raw features | 0.0220 | smooth decline 0.96 → 0.19 | |

- **Not the solver.** Running the Fantope program 5000 instead of 1000 iterations moves the objective
  by 1e-5 and the diagonal by 0.002; it does move the off-diagonals by 0.09 and η from 0.91 to 0.32, so
  the orientation is a flat valley there and the reported η depends on the stopping point.
- **Not the network's extra capacity.** The additive model is the theory's own class and reaches the
  network's KL with the network's hard profile. The relaxation misses its class optimum by 4.5×.
- **Not the basis alone.** Raw features order the features right but soften the cliff into a slope
  (leverage corr 0.94 with the additive model, CE 0.022); the predictive modes give the lower CE but a
  scrambled feature-space allocation (corr 0.50). A hard top-12 frame in *mode* space completes to
  0.038, so the optimum is 12 features, not 12 modes.
- **The noise itself.** Mode importances 6–20 are nearly tied (0.34–0.42), so the water filling splits a
  fixed budget among ties and each seed's sample covariance decides which mode gets 0.3 and which 0.
  The hard optimum has no such tie and is seed-stable.
- **Where the theory stands.** Stage one minimises C(d), the tied predictor's CE, which counts the
  benefit of leverage and none of its cost. Realising a fractional diagonal forces off-diagonal energy
  Σ_j P_ij² = d_i(1 − d_i), whose interference cost is the B(h) = Σ h(1 − h) term of the two-term
  theory. Dropping the projector constraint deletes B; that is right exactly when the completion refunds
  it. At the reference cell free pairs refunded 78–84% of B, which is why C alone predicted the pairs
  model and the network and why a priced allocation was worse there. Here the pairs model is as hard as
  the additive one: the refund has collapsed. The standing assumption of the frame stage is therefore
  "pairs cancel the forced interference", and this is the regime where it fails.
- **Why the refund fails, two computable mechanisms.** (i) The CE cost of an interference logit
  δℓ_i = α_i Σ_j P_ij (z_j − p̄_j) is, to second order, Σ_i q_i(1 − q_i) δℓ_i²; δℓ² is a pair term, so it
  is refundable only while the Fisher weight q_i(1 − q_i) is nearly constant across states. A steep ladder
  makes the softmax peaked and the weight swing with which head feature is on: the cost becomes third
  order and higher in z, outside span{1, z, φ}. (ii) Correlation makes co-activation group-wise, so
  interference events are no longer pairwise-sparse and more of the cost sits in the higher Walsh
  levels.
- **Proposed alteration.** Carry the *residual* price: β_eff(d) = the B-cost at the fractional predictor
  minus its projection onto span{1, z, φ(z)}, computable from the Fisher curvature we already have, no
  fitted constant. It reduces to plain water filling where pairs refund (the reference cell) and to a hard
  allocation where they do not; the 540-cell sweep tests the predicted soft/hard boundary. The decisive
  measurement first: repeat the reference-cell refund experiment (additive vs pairs biconvex oracles,
  B-slope and refund fraction) at this cell; if the refund has collapsed to near zero, the residual price
  is the theory of the boundary.

## What is open

- **Soft/hard transition**: predict where the fractional allocation stops being the optimum
  (dense p, steep β). Diagnosis and proposed residual-price alteration in the section above; the sweep
  gives the boundary empirically.
- **Optimizer check (SGD lr 2, 40k steps vs AdamW 8k, 5 seeds, d = 3 and 6; /tmp/sgd_seeds.py).** Losses
  match (SGD within ±0.005). Placement agreement between the two optimizers' 5-seed mean Grams
  (+0.46 / +0.49) equals their seed-to-seed self-agreement (Adam +0.46 / +0.38, SGD +0.53 / +0.53), and
  the theory matches both alike (theory–Adam +0.51 / +0.58, theory–SGD +0.38 / +0.48). Orientation is
  not an optimizer artefact; the scale split is (Adam leaves E 4.6–7.8× larger than D, SGD balances
  to 0.4%), which the common gauge removes. No reason to retrain with SGD.
- **Transfer (p ∈ {0.15, 0.3, 0.5}, β ∈ {0.25, 0.5, 1.0}, d ∈ {3, 6}, 3 seeds each; /tmp/transfer.py).**
  Allocation transfers everywhere (leverage corr 0.94–0.99). Orientation transfers at the
  reference cell (placement +0.42 / +0.66) and at d = 6 for p ≠ 0.3 (+0.30 / +0.32), but the
  head-pair sign fails at d = 3 away from p = 0.3 and at β = 0.25 for both d (trained G₀₁ turns
  positive, +0.05 to +0.13, where the pressure says negative); at β = 1.0 the pressure is ~0 and
  placement uninformative. The soft-tail log–log slope is ≈ 2 only at p = 0.15 (1.7 / 2.0); 1.1 at
  the reference cell, 0.4–1.0 elsewhere, so the slope-2 law is not general.
- **Common gauge, d = 12**: in the state-table gauge the theory's sizes fall below every trained
  class for features 1–8 (0.17 vs 0.30 at feature 2) while matching at d = 3, 6; the Fig 7
  cross block shows the completion's pair vectors anti-aligned with specific features (a blue
  stripe) that no trained class has.
- **d = 12**: (a) placement — a structured, mutually negative packing of the fractional tail
  (features 4–15, α ≈ 2–3.6) that both reference models reproduce (0.48) and no frame score
  predicts (~0); not pairs of dropped features, not head-vs-tail, not undertraining (4× steps
  change nothing), not the second-order term. (b) The Fantope diagonal moves too much mass
  into the tail, and (c) the completion's pair vectors anti-align with features where trained
  pairs are orthogonal to them. Common thread: the water level δ ≈ 0.005 is far below the
  second-order interference scale, so the near-degenerate tail (nearly uniform pressures) is
  resolved by costs the linear program does not carry. Candidate test: perturb the ladder
  (equal tail α → a clean simplex ring should appear; distinct α → it should vanish).
- **Soft-tail magnitude**: with the diagonal free, the Fantope program's tail matches trained
  at d = 3, 6 in the notebook (Fig 1) but at d = 12 it moves too much mass from the head to
  the tail (head 0.87 vs trained 0.93; tail above trained) — the linear pressure over-rewards
  tail overlaps where the water level is tiny. Likely fix: the effective price δ′ (interference
  floor) near threshold.
- **Completion residual**: third feature at d=3 (5.4 vs 6.8) and the second spectral mode
  (9.3 vs 12.1). Not pairs beyond the top 18 (the sweep rules that out): higher Walsh levels,
  or memorization by the network's free per-state table.
- **Rate–distortion framing**: the frontier Δ(d) vs d with slope δ has not been drawn.

## Next steps

1. (done) `rank_optimal_logits` = water filling → linear Fantope orientation → fixed-diagonal
   frame E★ → exact completion at E★ (the random-frame control is a test on the pressure score); retired analytic-completion
   functions deleted, `interference_curvature` kept; notebook rebuilt on the new outputs.
2. (done) Pair-set sweep of the biconvex pairs class, n_top ∈ {0, 6, 12, 18, 24, 32, 48, 64}.
   The class saturates at n_top ≈ 12–18: fresh-draw CE is lowest there (0.223 / 0.139 / 0.084)
   and rises beyond 32 (interpolation; at n_top = 64, 2081 features > 2048 states, fresh-draw
   CE 3–5). Placement peaks at 6–12 (0.74 / 0.65) and head sizes are flat past 6
   ([11.5, 6.4, 5.4] at d=3 vs trained [11.2, 8.1, 6.8]). Widening pairs does not approach the
   network: the remaining ~6% CE gap (0.2195 vs 0.2073 on the training states) and the
   second/third-feature sizes are beyond pairs — higher Walsh levels, or the network's free
   per-state table fitting its own draw (it has no fresh-draw evaluation; the states are its
   vocabulary). n_top = 18 stays.
3. Notebook: a soft-tail panel (trained tail leverage vs ‖a·_j‖²/4δ² and vs the Fantope tail).
4. d = 12: test whether the tail packing is set by the near-degenerate α ladder (perturb the
   ladder slope; a simplex-like arrangement should follow the equal-α block) before looking
   for a new mechanism.
5. (done) Hysteresis test: GD from the anti-pressure frame returns to the pressure pattern,
   so orientation is an optimality statement of the joint class (tex updated accordingly).
6. Commit.
