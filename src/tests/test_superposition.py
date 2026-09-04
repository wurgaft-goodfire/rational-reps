"""The first-order geometry pipeline (geometry.py), from notes/superposition.tex:
convex allocation -> tight-frame realization -> convex decoder -> balanced SVD.
Every test pins an analytic answer or an independent audit."""
import numpy as np
import pytest

from src.code import geometry
from src.code.geometry import allocate_d, decoder_map, feature_vectors, tight_frame

LADDER = 8.0 * (1.0 + np.arange(64)) ** -0.5          # the standard power-law importance ladder

# trained 5-seed mean leverage at d = 3 (features 0..7), the reference the allocation
# must reproduce (measured 2026-09-01; the deep-tail cutoff residual lives at d = 12)
TRAINED_H3 = np.array([0.839, 0.710, 0.686, 0.293, 0.234, 0.137, 0.041, 0.014])


def _bernoulli(m, n=16384, p=0.3, seed=0):
    return (np.random.default_rng(seed).random((n, m)) < p).astype(float)


def test_allocate_d_splits_evenly_under_equal_importance():
    # exchangeable features: the convex optimum is symmetric, a_i = d/m
    a, delta = allocate_d(np.full(16, 3.0), _bernoulli(16), 4)
    assert abs(a.sum() - 4) < 1e-9
    assert np.abs(a - 0.25).max() < 0.03
    assert delta > 0


def test_allocate_d_orders_allocation_by_importance():
    a, _ = allocate_d(LADDER[:8], _bernoulli(8), 3)
    assert np.all(np.diff(a) < 0.02)                  # decreasing along the ladder
    assert a[0] > a[-1] + 0.2


@pytest.mark.slow
def test_allocate_d_passes_an_independent_kkt_audit():
    # audit the returned allocation with an evaluator independent of the solver:
    # utilities on a large fresh dataset must equalize at delta on the interior and
    # stay below it on the dropped set (guards the water-level dilution bug class).
    from scipy.special import logsumexp
    a, delta = allocate_d(LADDER, _bernoulli(64), 6)
    Z = _bernoulli(64, n=262144, seed=99)
    G = Z * LADDER
    P = np.exp(G - logsumexp(G, axis=1, keepdims=True))
    L = Z * (LADDER * a)
    q = np.exp(L - logsumexp(L, axis=1, keepdims=True))
    u = -LADDER * (Z * (q - P)).mean(0)
    interior = (a > 1e-3) & (a < 1 - 1e-3)
    dropped = a <= 1e-3
    assert interior.any() and dropped.any()
    assert np.abs(u[interior] - delta).max() < 0.06 * delta
    assert u[dropped].max() < 1.02 * delta


@pytest.mark.slow
def test_allocate_d_reproduces_trained_leverage():
    a, _ = allocate_d(LADDER, _bernoulli(64), 3)
    ref = np.zeros(64)
    ref[:len(TRAINED_H3)] = TRAINED_H3
    assert np.corrcoef(a, ref)[0, 1] > 0.95


def test_tight_frame_satisfies_the_frame_identities():
    # Parseval: E1^T E1 = I; prescribed row norms; forced-overlap identity
    rng = np.random.default_rng(0)
    raw = rng.uniform(0, 1, 12)
    a = raw * 4.0 / raw.sum()
    E1 = tight_frame(a, seed=1)
    assert np.allclose(E1.T @ E1, np.eye(4), atol=1e-7)
    assert np.allclose((E1 ** 2).sum(1), a, atol=1e-6)
    G = E1 @ E1.T
    off = (G ** 2).sum(1) - np.diag(G) ** 2
    assert np.allclose(off, a * (1 - a), atol=1e-6)


def test_decoder_map_recovers_the_target_when_the_frame_contains_it():
    # analytic oracle: give the frame the target map's own column space; the convex
    # decoder solve must return L1 = diag(alpha) J exactly (row-centred gauge).
    m = 8
    alpha = 3.0 * (1.0 + np.arange(m)) ** -0.5
    T = np.diag(alpha) @ (np.eye(m) - 1.0 / m)
    E1 = np.linalg.qr(T)[0][:, :m - 1]
    L1, o = decoder_map(alpha, _bernoulli(m, n=8192), E1)
    assert np.abs(L1 - T).max() < 0.05
    assert np.abs(o - o.mean()).max() < 0.05          # the offset is pure softmax gauge


def test_decoder_map_satisfies_the_joint_stationarity_conditions():
    # KKT of the joint (D, e0) solve, in returned observables (see the derivation in
    # the tex): frame moments are rank one along the mean residual, and L1 dbar = 0.
    from scipy.special import logsumexp
    m, d = 12, 3
    alpha = 6.0 * (1.0 + np.arange(m)) ** -0.5
    Z = _bernoulli(m, n=8192)
    a, _ = allocate_d(alpha, Z, d)
    E1 = tight_frame(a, seed=2)
    L1, o = decoder_map(alpha, Z, E1)
    G = Z * alpha
    P = np.exp(G - logsumexp(G, axis=1, keepdims=True))
    L = Z @ L1 + o
    q = np.exp(L - logsumexp(L, axis=1, keepdims=True))
    dbar = (q - P).mean(0)
    M = E1.T @ (Z.T @ (q - P)) / len(Z)
    M_perp = M - np.outer(M @ dbar, dbar) / (dbar @ dbar)
    assert np.abs(M_perp).max() < 2e-4
    assert np.abs(L1 @ dbar).max() < 2e-4


@pytest.mark.slow
def test_predicted_map_uses_exactly_the_allocated_subspace():
    # leverage of the predicted first-order map equals the allocation
    Z = _bernoulli(64)
    a, _ = allocate_d(LADDER, Z, 6)
    E1 = tight_frame(a, seed=3)
    L1, _ = decoder_map(LADDER, Z, E1)
    h = geometry.leverage(L1)[0]
    assert np.corrcoef(h, a)[0, 1] > 0.999


def test_signed_pressure_is_the_covariance_form_and_points_head_against_tail():
    # a_ij = alpha_j Cov(z_i, q_j - p*_j) + alpha_i Cov(z_j, q_i - p*_i) at the fractional
    # predictor (independent covariance evaluation); the head presses every weaker
    # represented feature toward negative overlap (a_0j > 0), while the two head
    # features press each other the other way (a_01 < 0).
    from scipy.special import logsumexp
    alpha = LADDER[:16]
    Z = _bernoulli(16, n=8192)
    di, _ = allocate_d(alpha, Z, 3)
    a = geometry.signed_pressure(alpha, Z, di)
    G = Z * alpha
    P = np.exp(G - logsumexp(G, axis=1, keepdims=True))
    L = Z * (alpha * di)
    q = np.exp(L - logsumexp(L, axis=1, keepdims=True))
    r = q - P
    ref = np.array([[alpha[j] * np.cov(Z[:, i], r[:, j], bias=True)[0, 1]
                     + alpha[i] * np.cov(Z[:, j], r[:, i], bias=True)[0, 1]
                     for j in range(16)] for i in range(16)])
    np.fill_diagonal(ref, 0.0)
    assert np.allclose(a, ref, atol=1e-10)
    rep = np.flatnonzero(di > 0.05)
    assert (a[0, rep[3:]] > 0).all() and (a[1, rep[3:]] > 0).all()
    assert a[0, 1] < 0


@pytest.mark.slow
def test_predict_geometry_reports_frame_ensemble_statistics():
    # one realized CE per frame, the mean Gram's diagonal is the mean sizes, and
    # the spectrum has exactly d modes (the rank cutoff is not a mode)
    out = geometry.predict_geometry(LADDER[:16], _bernoulli(16, n=4096), 3, n_frames=6, n_top=8)
    ce = out["frame_ce"]
    assert ce.shape == (6,) and ce.min() > 0
    assert np.allclose(np.diag(out["gram_mean"]), out["sizes_mean"])
    assert out["spectra_mean"].shape == (3,) and out["spectra_sd"].shape == (3,)
    assert np.allclose((out["E_star"] ** 2).sum(1), out["d_star"], atol=1e-5)
    assert abs(out["d_star"].sum() - 3) < 1e-6 and 0 <= out["eta"] < 3
    assert out["ce_rep"] < out["ce_add"] <= out["ce_star"] + 1e-9   # completion lowers the CE at E*
    # the completion's logit map over its dictionary [z; phi(z); 1], and the dictionary in
    # balanced coordinates: the rank-d factorization is exact with equal scale in both factors
    M = np.vstack([out["E_star"], out["E2_rep"], out["e0_rep"]]) @ out["C_rep"]
    assert out["map_rep"].shape == (16 + 28 + 1, 16) and np.allclose(out["map_rep"], M)
    Mc, B, D, S = geometry.factor_logcond(M, rank=3)
    assert B.shape == (45, 3) and D.shape == (3, 16) and S.shape == (3,)
    assert np.allclose(B @ D, Mc, atol=1e-8)
    assert np.allclose(np.linalg.norm(B, axis=0), np.linalg.norm(D, axis=1), atol=1e-6)
    assert np.allclose(geometry.predicted_geometry(M, rank=3), B)


@pytest.mark.slow
def test_repair_lowers_the_ce_monotonically_and_keeps_the_frame():
    # stage two is block-coordinate descent on the additive-plus-pairs class: every
    # round lowers the excess CE, and the first-order map keeps the frame's leverage
    # (pairs are population-orthogonal to features, so E1 is untouched)
    alpha = LADDER[:16]
    Z = _bernoulli(16, n=4096)
    di, _ = allocate_d(alpha, Z, 3)
    P, _ = geometry.fantope_geometry(alpha, Z, 3, geometry.signed_pressure(alpha, Z, di))
    E1 = tight_frame(np.diag(P), init=P)
    out = geometry.repair(alpha, Z, E1, n_top=10, rounds=4)
    assert (np.diff(out["ce"]) < 1e-9).all()
    assert out["ce"][-1] < 0.8 * out["ce_additive"]
    assert np.corrcoef(geometry.leverage(out["L1"])[0], np.diag(P))[0, 1] > 0.999
    assert out["E2"].shape == (45, 3)


def test_feature_model_builds_the_centered_dictionary_and_is_bias_free():
    # the reference-class runners: phi's first block is the centered features, the pair
    # block is standardized and population-orthogonal to it, the code is e0 + phi E,
    # and no per-output bias exists anywhere.
    import torch
    from src.code.models import FeatureModel, ModelConfig
    rng = np.random.default_rng(0)
    Z = (rng.random((512, 6)) < 0.3).astype(float)
    cfg = ModelConfig(vocab_size=520, num_classes=6, embed_dim=3,
                      n_features=6, feature_order=2, n_top=6)
    model = FeatureModel(cfg)
    model.load_features(Z)
    phi = model.phi.numpy()
    Zc = Z - Z.mean(0)
    assert np.allclose(phi[:512, :6], Zc, atol=1e-6)
    pairs = phi[:512, 6:]
    assert np.allclose(pairs.std(0, ddof=1), 1.0, atol=1e-3)   # torch std is Bessel-corrected
    assert np.abs(Zc.T @ pairs / 512).max() < 0.1     # centered blocks nearly orthogonal
    ids = torch.arange(8).reshape(8, 1)
    L = model(ids)["logits"].detach().numpy()
    h = model.input_reps(ids).detach().numpy()
    W = model.readout.detach().numpy()
    assert np.allclose(L, h @ W.T, atol=1e-6)
    assert not any("bias" in n for n, _ in model.named_parameters())


def test_feature_vectors_recover_an_additive_code_exactly():
    # H = Z V + v0 with no noise: least squares must return V and v0 exactly
    rng = np.random.default_rng(0)
    Z = (rng.random((40, 5)) < 0.4).astype(float)
    V_true, v0_true = rng.normal(size=(5, 3)), rng.normal(size=3)
    V, v0 = feature_vectors(Z @ V_true + v0_true, Z)
    assert np.allclose(V, V_true, atol=1e-9) and np.allclose(v0, v0_true, atol=1e-9)


def test_leverage_gives_full_scores_to_an_orthogonal_frame_and_satisfies_the_packing_identity():
    W = np.zeros((5, 8))
    W[np.arange(5), np.arange(5)] = [3.0, 2.0, 1.5, 1.0, 0.5]
    h, P = geometry.leverage(W)
    assert np.allclose(h, 1.0)
    assert np.allclose(P - np.diag(np.diag(P)), 0.0)

    rng = np.random.default_rng(0)
    W = rng.normal(size=(12, 3))
    h, P = geometry.leverage(W)
    assert np.all(h > -1e-12) and np.all(h < 1 + 1e-12)
    assert abs(h.sum() - 3) < 1e-9
    off = (P ** 2).sum(1) - np.diag(P) ** 2
    assert np.allclose(off, h * (1 - h), atol=1e-9)


def test_leverage_is_invariant_to_any_invertible_transform_of_the_code():
    rng = np.random.default_rng(1)
    W = rng.normal(size=(10, 4))
    A = rng.normal(size=(4, 4)) + 4 * np.eye(4)
    assert np.allclose(geometry.leverage(W)[1], geometry.leverage(W @ A)[1], atol=1e-8)


def test_logit_map_recovers_a_planted_additive_map_exactly():
    rng = np.random.default_rng(2)
    Z = (rng.random((256, 6)) < 0.4).astype(float)
    R0 = rng.normal(size=(6, 6))
    b0 = rng.normal(size=6)
    shift = rng.normal(size=(256, 1))                 # softmax gauge: per-row constant
    L = Z @ R0 + b0 + shift
    R, b, r2 = geometry.logit_map(L, Z)
    J = np.eye(6) - 1.0 / 6
    assert np.allclose(R, R0 @ J, atol=1e-9)
    assert abs(r2 - 1.0) < 1e-12


def test_logit_features_recovers_planted_balanced_sizes():
    rng = np.random.default_rng(4)
    m, d, p, n = 32, 3, 0.3, 8192
    Z = (rng.random((n, m)) < p).astype(float)
    E = np.zeros((m, d))
    E[0, 0], E[1, 1], E[2, 2] = 2.0, 1.5, 1.0        # orthogonal writers, sizes 4/2.25/1
    L = (Z - Z.mean(0)) @ (E @ E.T)                  # mean-free code: sizes are exact
    sizes = (geometry.logit_features(L, Z, d) ** 2).sum(1)
    assert np.allclose(sizes[:3], [4.0, 2.25, 1.0], rtol=0.05)
    assert sizes[3:].max() < 0.05


def _tied_ce(alpha, Z, di, G_off):
    # exact CE of the tied map: own logits alpha_k d_k z_k plus interference
    # alpha_k sum_{i != k} (z_i - zbar_i) G_ik; the object the local expansion approximates
    from scipy.special import logsumexp
    G, P, _ = geometry._softmax_stats(alpha, Z)
    X = Z - Z.mean(0)
    L = Z * (alpha * di) + (X @ G_off) * alpha
    return float((logsumexp(L, axis=1) - (P * L).sum(1)).mean())


def test_pressure_and_curvature_are_the_derivatives_of_the_tied_map_ce():
    # a_g = first derivative, H_gg = second derivative of the tied-map CE in the
    # off-diagonal coordinates at the diagonal background, by central differences
    alpha = LADDER[:10]
    Z = _bernoulli(10, n=4096)
    di, _ = allocate_d(alpha, Z, 3)
    a = geometry.signed_pressure(alpha, Z, di)
    pairs = np.triu_indices(10, 1)
    H = geometry.interference_curvature(alpha, Z, di)
    h = 1e-3
    def ce(deltas):
        G = np.zeros((10, 10))
        for (i, j), v in deltas:
            G[i, j] = G[j, i] = v
        return _tied_ce(alpha, Z, di, G)
    for p in (0, 5, 20):
        i, j = pairs[0][p], pairs[1][p]
        fd = (ce([((i, j), h)]) - ce([((i, j), -h)])) / (2 * h)
        assert abs(fd - a[i, j]) < 1e-6 * max(1.0, abs(a[i, j])) + 1e-7
        fdd = (ce([((i, j), h)]) - 2 * ce([]) + ce([((i, j), -h)])) / h ** 2
        assert abs(fdd - H[p, p]) < 2e-3 * abs(H[p, p])
    p, q = 0, 5                                        # pairs (0,1) and (0,6) share feature 0
    i, j = pairs[0][p], pairs[1][p]; k, l = pairs[0][q], pairs[1][q]
    fdd = (ce([((i, j), h), ((k, l), h)]) - ce([((i, j), h), ((k, l), -h)])
           - ce([((i, j), -h), ((k, l), h)]) + ce([((i, j), -h), ((k, l), -h)])) / (4 * h ** 2)
    assert abs(fdd - H[p, q]) < 2e-3 * abs(H[p, q]) + 1e-9
    assert np.allclose(H, H.T) and np.all(np.linalg.eigvalsh(H) > -1e-9)


def test_fantope_geometry_reduces_to_water_filling_without_pressure():
    # a = 0, H = 0: the objective is C(diag P) alone, so the Fantope optimum's diagonal
    # is the water-filling allocation (its off-diagonal is undetermined)
    alpha = LADDER[:12]
    Z = _bernoulli(12, n=4096)
    di, _ = allocate_d(alpha, Z, 3)
    P, eta = geometry.fantope_geometry(alpha, Z, 3, np.zeros((12, 12)))
    assert np.abs(np.diag(P) - di).max() < 0.02
    assert abs(np.trace(P) - 3) < 1e-8
    lam = np.linalg.eigvalsh(P)
    assert lam.min() > -1e-8 and lam.max() < 1 + 1e-8 and 0 <= eta <= 3


def test_fantope_geometry_with_pressure_grows_a_soft_tail_with_the_pressure_signs():
    # the linear program: every feature water filling drops gets a positive share
    # (the soft tail), the off-diagonals follow the pressure on net and in most large
    # entries (PSD coupling forces a few the other way), and the objective sits below
    # the water-filling point's
    alpha = LADDER[:12]
    Z = _bernoulli(12, n=4096)
    di, _ = allocate_d(alpha, Z, 3)
    a = geometry.signed_pressure(alpha, Z, di)
    P, eta = geometry.fantope_geometry(alpha, Z, 3, a)
    dropped = di < 1e-3
    assert dropped.any() and (np.diag(P)[dropped] > 1e-4).all()
    iu = np.triu_indices(12, 1)
    assert float(a[iu] @ P[iu]) < 0                          # net: follows the pressure
    big = np.abs(P[iu]) > 0.1                                 # the large overlaps mostly do
    assert big.sum() >= 5 and np.mean(np.sign(P[iu][big]) == -np.sign(a[iu][big])) >= 0.75
    assert 0 <= eta < 0.1 * 3                                 # close to a rank-d projector
    obj = lambda Q: geometry._fantope_objective(alpha, Z, Q, a, None)
    assert obj(P) < obj(np.diag(di)) - 1e-6


def test_fixed_diagonal_realization_keeps_the_diagonal_and_is_parseval():
    # realization B: alternating projections from a Fantope point onto rank-d
    # projectors with that point's diagonal
    rng = np.random.default_rng(0)
    V = np.linalg.qr(rng.normal(size=(12, 3)))[0]
    P = 0.7 * V @ V.T + 0.3 * np.diag(np.full(12, 0.25))      # inside the Fantope, not a projector
    E = tight_frame(np.diag(P), init=P)
    assert np.allclose(E.T @ E, np.eye(3), atol=1e-7)
    assert np.allclose((E ** 2).sum(1), np.diag(P), atol=1e-6)
    Ud = geometry.top_eigenspace(P, 3)
    assert np.allclose(Ud.T @ Ud, np.eye(3), atol=1e-10)


@pytest.mark.slow
def test_biconvex_pairs_never_increases_the_objective_and_beats_the_additive_class():
    # Part II benchmark: exact CE of the additive-plus-pairs class by alternating convex
    # block solves; the objective trace is nonincreasing and ends below the additive
    # class realized at a compatible frame
    alpha = LADDER[:16]
    Z = _bernoulli(16, n=4096)
    out = geometry.biconvex_pairs(alpha, Z, 3, n_top=10, rounds=4, seed=0)
    assert (np.diff(out["ce_trace"]) < 1e-9).all()
    di, _ = allocate_d(alpha, Z, 3)
    G, P, _ = geometry._softmax_stats(alpha, Z)
    L1, o = decoder_map(alpha, Z, tight_frame(di, seed=1))
    assert out["ce"] < geometry._excess_ce(Z @ L1 + o, G, P)
    assert out["B"].shape == (1 + 16 + 45, 3) and out["D"].shape == (3, 16)




def test_sweep_grid_enumerates_unique_cells_with_unique_run_names_and_paths():
    from src.code import sweep_superposition as sw
    cs = sw.cells()
    assert len(cs) == 4 * 3 * 3 * 5 and len(set(cs)) == len(cs)
    assert len({sw.run_name(*c) for c in cs}) == len(cs)
    assert len({sw.cell_path(*c) for c in cs}) == len(cs)
    assert (0.3, 0.5, 6, 0) in cs and sw.run_name(0.3, 0.5, 6, 0) == "sp_p0.3_b0.5_d6_s0"
