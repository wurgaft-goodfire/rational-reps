"""The first-order geometry pipeline (geometry.py), from notes/superposition.tex:
convex allocation -> tight-frame realization -> convex decoder -> balanced SVD.
Every test pins an analytic answer or an independent audit."""
import os
import pathlib

import numpy as np
import pytest

from src.code import geometry
from src.code.analysis_utils import feature_vectors, leverage, logit_map
from src.code.geometry import allocate_d, decoder_map, tight_frame

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
    h = leverage(L1)[0]
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
def test_predicted_rank_constrained_geometry_is_the_completed_model_at_the_pressure_frame():
    # H = z E* + phi(z) E2 + e0 with E* the fixed-diagonal realization of the Fantope solution
    # and (E2, e0, D) the completion at E*; the logits are H D^T
    alpha, Z, d = LADDER[:16], _bernoulli(16, n=4096), 3
    H, D = geometry.predicted_rank_constrained_geometry(alpha, Z, d, n_top=8)
    d_wf, _ = allocate_d(alpha, Z, d)
    P, _ = geometry.fantope_geometry(alpha, Z, d, geometry.signed_pressure(alpha, Z, d_wf))
    E_star = tight_frame(np.diag(P), init=P)
    rep = geometry.repair(alpha, Z, E_star, n_top=8)
    assert H.shape == (4096, 3) and D.shape == (16, 3)
    assert np.allclose(H, Z @ E_star + geometry.pair_features(Z, 8) @ rep["E2"] + rep["e0"])
    assert np.allclose(D, rep["D"].T)
    assert np.allclose(feature_vectors(H, Z)[0], E_star, atol=1e-8)   # the frame is the first-order code


def test_pressure_frame_scores_below_every_random_frame_with_the_same_allocation():
    # the control behind the orientation step: among tight frames at the water-filling
    # diagonal, random ones score ~0 on the signed pressure sum_{i != j} a_ij G_ij while the
    # realized Fantope frame sits below all of them
    alpha, Z, d = LADDER[:16], _bernoulli(16, n=4096), 3
    d_wf, _ = allocate_d(alpha, Z, d)
    a = geometry.signed_pressure(alpha, Z, d_wf)
    P, _ = geometry.fantope_geometry(alpha, Z, d, a)
    score = lambda E: float((a * (E @ E.T)).sum())
    star = score(tight_frame(np.diag(P), init=P))
    random = np.array([score(tight_frame(d_wf, seed=k)) for k in range(1, 9)])
    assert star < random.min() and star < 0
    assert abs(random.mean()) < abs(star) / 3


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
    assert np.corrcoef(leverage(out["L1"])[0], np.diag(P))[0, 1] > 0.999
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
    h = model(ids, output_hidden_states=True)["hidden_states"][-1][:, -1].detach().numpy()
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
    h, P = leverage(W)
    assert np.allclose(h, 1.0)
    assert np.allclose(P - np.diag(np.diag(P)), 0.0)

    rng = np.random.default_rng(0)
    W = rng.normal(size=(12, 3))
    h, P = leverage(W)
    assert np.all(h > -1e-12) and np.all(h < 1 + 1e-12)
    assert abs(h.sum() - 3) < 1e-9
    off = (P ** 2).sum(1) - np.diag(P) ** 2
    assert np.allclose(off, h * (1 - h), atol=1e-9)


def test_leverage_is_invariant_to_any_invertible_transform_of_the_code():
    rng = np.random.default_rng(1)
    W = rng.normal(size=(10, 4))
    A = rng.normal(size=(4, 4)) + 4 * np.eye(4)
    assert np.allclose(leverage(W)[1], leverage(W @ A)[1], atol=1e-8)


def test_logit_map_recovers_a_planted_additive_map_exactly():
    rng = np.random.default_rng(2)
    Z = (rng.random((256, 6)) < 0.4).astype(float)
    R0 = rng.normal(size=(6, 6))
    b0 = rng.normal(size=6)
    shift = rng.normal(size=(256, 1))                 # softmax gauge: per-row constant
    L = Z @ R0 + b0 + shift
    R, b, r2 = logit_map(L, Z)
    J = np.eye(6) - 1.0 / 6
    assert np.allclose(R, R0 @ J, atol=1e-9)
    assert abs(r2 - 1.0) < 1e-12


def test_leverage_of_the_first_order_vectors_is_gauge_invariant_and_their_gram_is_not():
    # a representation additive in the features with orthogonal writers of sizes 4 / 2.25 / 1:
    # the first-order vectors recover the writers, their leverages are (1, 1, 1, then 0) with a
    # diagonal projector, and a change of gauge (H A, D A^-T) leaves leverage alone while it
    # changes the Gram
    rng = np.random.default_rng(4)
    m, d, p, n = 32, 3, 0.3, 8192
    Z = (rng.random((n, m)) < p).astype(float)
    E = np.zeros((m, d))
    E[0, 0], E[1, 1], E[2, 2] = 2.0, 1.5, 1.0
    H = (Z - Z.mean(0)) @ E
    E1 = feature_vectors(H, Z)[0]
    h, P = leverage(E1)
    assert np.allclose(E1, E, atol=1e-8)
    assert np.allclose(h[:3], 1.0, atol=1e-6) and h[3:].max() < 1e-6
    assert np.allclose(P[:3, :3], np.eye(3), atol=1e-6)
    A = np.diag([3.0, 0.5, 2.0])
    E2 = feature_vectors(H @ A, Z)[0]
    assert np.allclose(leverage(E2)[0], h) and np.allclose(leverage(E2)[1], P)
    assert not np.allclose(E2 @ E2.T, E1 @ E1.T)


def _tied_ce(alpha, Z, di, G_off):
    # exact CE of the tied map: own logits alpha_k d_k z_k plus interference
    # alpha_k sum_{i != k} (z_i - zbar_i) G_ik; the object the local expansion approximates
    from scipy.special import logsumexp
    G, P, _ = geometry._softmax_stats(alpha, Z)
    X = Z - Z.mean(0)
    L = Z * (alpha * di) + (X @ G_off) * alpha
    return float((logsumexp(L, axis=1) - (P * L).sum(1)).mean())


def test_pressure_is_the_derivative_of_the_tied_map_ce():
    # a_g = first derivative of the tied-map CE in the off-diagonal coordinates at the
    # diagonal background, by central differences
    alpha = LADDER[:10]
    Z = _bernoulli(10, n=4096)
    di, _ = allocate_d(alpha, Z, 3)
    a = geometry.signed_pressure(alpha, Z, di)
    pairs = np.triu_indices(10, 1)
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
    obj = lambda Q: geometry._fantope_objective(alpha, Z, Q, a)
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



def test_modes_are_the_features_when_independent_and_the_predictive_modes_when_not():
    Z = _bernoulli(4, n=64)
    W, X = geometry.modes(np.ones(4), Z, 0.0)
    assert W.shape == (4,) and np.array_equal(X, Z)
    W, X = geometry.modes(np.ones(4), Z, 0.4)
    assert W.shape == (4, 4) and np.allclose(X @ geometry.whiten(Z)[1], Z)


# ── the sweep as a setting preset ──────────────────────────────────────────────────────────

def test_superposition_setting_declares_the_grid_and_couples_the_seeds():
    # the sweep is Hydra's: every sweeper param is a valid override of the composed config, and
    # the last value of each composes to the cell it names, with the task seed following the
    # training seed
    from hydra.core.override_parser.overrides_parser import OverridesParser
    from omegaconf import OmegaConf
    from pyprojroot import here
    from src.code.train import resolve_config
    params = OmegaConf.load(here() / "conf" / "setting" / "superposition.yaml").hydra.sweeper.params
    grid = {o.key_or_group: list(o.sweep_string_iterator()) if o.is_sweep_override() else [str(o.value())]
            for o in OverridesParser.create().parse_overrides([f"{k}={v}" for k, v in params.items()])}
    assert [len(v) for v in grid.values()] == [4, 3, 3, 3, 5, 1]           # p, beta, rho, d, seed, steps
    for runner in ("linear", "additive_features", "paired_features"):
        config = resolve_config(["setting=superposition", f"runner={runner}"] + [f"{k}={v[-1]}" for k, v in grid.items()])
        sk = config.cfg.structure_kwargs
        assert (sk.N, sk.m, sk.scale, sk.spectrum) == (2048, 64, 8.0, "power")
        assert sk.seed == config.cfg.random_seed == 4                      # task seed follows the training seed
        assert (sk.p, sk.beta, sk.rho) == (0.5, 1.0, 0.6) and config.cfg.training_kwargs.train_steps == 8000
        assert config.cfg.model_kwargs.model_config.embed_dim == 12
        assert os.path.basename(config.run_dir) == runner                  # the model class is the leaf


def test_has_run_is_false_before_training(tmp_path):
    from src.code.analysis_utils import has_run
    assert not has_run(["setting=superposition", "runner=linear"], cache_dir=str(tmp_path))
    assert not (tmp_path / "graph-cooccurrence").exists()                  # and looking created nothing


# ── correlated features: predictive modes ──────────────────────────────────────────────────

def _correlated(m, n=8192, p=0.3, rho=0.5, seed=0):
    from src.code.data import feature_cube
    return feature_cube(n, m, p, scale=8.0, seed=seed, spectrum="power", beta=0.5, rho=rho)[1]["Z"]


def _tied_ce_map(W, Z, di, G_off):
    # exact CE of the tied map for a general logit map W: logits (z d + X G_off) W^T
    from scipy.special import logsumexp
    G, P, _ = geometry._softmax_stats(W, Z)
    X = Z - Z.mean(0)
    L = (Z * di + X @ G_off) @ W.T
    return float((logsumexp(L, axis=1) - (P * L).sum(1)).mean())


def test_whiten_decorrelates_the_features_and_inverts():
    # u = C^{-1/2} z has identity covariance and Z = U C^{1/2}
    Z = _correlated(8)
    U, C_half = geometry.whiten(Z)
    assert np.allclose(np.cov(U.T, bias=True), np.eye(8), atol=1e-8)
    assert np.allclose(U @ C_half, Z, atol=1e-8)
    assert np.allclose(C_half @ C_half, np.cov(Z.T, bias=True), atol=1e-8)


def test_a_diagonal_logit_map_reproduces_the_vector_form():
    # every stage accepts the (m, m) logit map; diag(alpha) is the vector path exactly
    alpha, Z = LADDER[:10], _bernoulli(10, n=4096)
    W = np.diag(alpha)
    di_v, dl_v = allocate_d(alpha, Z, 3); di_m, dl_m = allocate_d(W, Z, 3)
    assert np.allclose(di_v, di_m, atol=1e-9) and abs(dl_v - dl_m) < 1e-9
    assert np.allclose(geometry.signed_pressure(alpha, Z, di_v), geometry.signed_pressure(W, Z, di_v))
    a = geometry.signed_pressure(alpha, Z, di_v)
    P_v, _ = geometry.fantope_geometry(alpha, Z, 3, a); P_m, _ = geometry.fantope_geometry(W, Z, 3, a)
    assert np.allclose(P_v, P_m, atol=1e-8)
    E1 = tight_frame(di_v)
    assert np.allclose(decoder_map(alpha, Z, E1)[0], decoder_map(W, Z, E1)[0], atol=1e-6)


def test_whitening_is_a_no_op_on_exactly_independent_features():
    # on the full cube {0,1}^k the sample covariance is exactly diagonal, so the whitened
    # problem (alpha C^{1/2}, C^{-1/2} Z) is the raw one up to a uniform rescaling
    k = 8
    Z = np.array(np.meshgrid(*[[0, 1]] * k, indexing="ij")).reshape(k, -1).T.astype(float)
    alpha = LADDER[:k]
    U, C_half = geometry.whiten(Z)
    W = np.diag(alpha) @ C_half.T
    di, _ = allocate_d(alpha, Z, 3); dw, _ = allocate_d(W, U, 3)
    assert np.allclose(di, dw, atol=1e-6)
    a, aw = geometry.signed_pressure(alpha, Z, di), geometry.signed_pressure(W, U, dw)
    assert np.allclose(a, aw, atol=1e-8)
    P, _ = geometry.fantope_geometry(alpha, Z, 3, a); Pw, _ = geometry.fantope_geometry(W, U, 3, aw)
    assert np.allclose(P, Pw, atol=1e-6)


def test_pressure_is_the_derivative_for_a_full_logit_map():
    # the derivative identity holds on whitened correlated features, where the logit map is
    # a full matrix
    Z = _correlated(8, n=4096)
    U, C_half = geometry.whiten(Z)
    W = np.diag(LADDER[:8]) @ C_half.T
    di, _ = allocate_d(W, U, 3)
    a = geometry.signed_pressure(W, U, di)
    pairs = np.triu_indices(8, 1)
    h = 1e-3
    def ce(deltas):
        G = np.zeros((8, 8))
        for (i, j), v in deltas:
            G[i, j] = G[j, i] = v
        return _tied_ce_map(W, U, di, G)
    for p in (0, 4, 15):
        i, j = pairs[0][p], pairs[1][p]
        fd = (ce([((i, j), h)]) - ce([((i, j), -h)])) / (2 * h)
        assert abs(fd - a[i, j]) < 1e-6 * max(1.0, abs(a[i, j])) + 1e-7


def test_pair_features_are_orthogonal_to_the_first_order_dictionary():
    # residualised on [1, z], the pair block cannot leak into the first-order vectors
    # even when the features are correlated
    Z = _correlated(8, n=4096)
    U = geometry.whiten(Z)[0]
    phi = geometry.pair_features(U, 6)
    X = np.column_stack([np.ones(len(U)), U])
    assert phi.shape == (4096, 15)
    assert np.abs(X.T @ phi).max() < 1e-8
    assert np.allclose(phi.std(0), 1.0)


@pytest.mark.slow
def test_whitening_brings_the_theory_to_the_class_optimum_on_correlated_features(tmp_path):
    # equicorrelated features (rho = 0.6, m = 16, d = 3): the pairs class trained to its
    # optimum (the paired_features runner, full batch) is the oracle, compared through the
    # gauge-invariant measurements only (excess CE, leverage, leverage projector). On the
    # predictive modes the theory closes most of the CE gap to the oracle that raw features
    # leave open. Leverage and placement are reported, not asserted: measured invariantly they
    # do not favor the modes (leverage a tie, placement 0.05 vs 0.28 raw, 2026-09-08); open
    # item in notes/superposition_results.md.
    from src.code.train import train_inline
    m, d, k = 16, 3, 8
    model, dataset, evaluator, *_ = train_inline(
        ["setting=superposition", "runner=paired_features", "structure_kwargs.N=4096",
         f"structure_kwargs.m={m}", "structure_kwargs.rho=0.6", f"model_kwargs.model_config.embed_dim={d}",
         "random_seed=0"], cache_dir=str(tmp_path))
    alpha, Z = dataset.info["alpha"], np.asarray(dataset.features, float)
    iu = np.triu_indices(k, 1)
    h_opt, P_opt = leverage(feature_vectors(evaluator.evaluate(model)["hidden_states"][-1][:, 0], Z)[0])
    G, P, _ = geometry._softmax_stats(alpha, Z)
    scores = {}
    for name, correlated in [("raw", 0.0), ("white", 0.6)]:
        H, D = geometry.predicted_rank_constrained_geometry(*geometry.modes(alpha, Z, correlated), d)
        h_th, P_th = leverage(feature_vectors(H, Z)[0])
        scores[name] = (geometry._excess_ce(H @ D.T, G, P), np.corrcoef(h_th, h_opt)[0, 1],
                        np.corrcoef(P_th[:k, :k][iu], P_opt[:k, :k][iu])[0, 1])
    print("(excess CE, leverage corr, placement corr) vs the pairs optimum:", scores)
    assert scores["white"][0] < scores["raw"][0] - 0.05, scores          # CE gap closed by > 0.05


def test_mutually_exclusive_features_are_predicted_antipodal_in_both_bases():
    # two features that never co-occur inside an iid background (m = 16, d = 3): the
    # predicted overlap of the pair's writers is negative (gauge-invariantly, in the leverage
    # projector) on raw features and on the predictive modes alike (the TMS antipodal
    # arrangement)
    rng = np.random.default_rng(0)
    n, m, d = 4096, 16, 3
    c = rng.choice(3, size=n, p=[0.3, 0.3, 0.4])
    Z = np.column_stack([c == 0, c == 1, rng.random((n, m - 2)) < 0.3]).astype(float)
    alpha = LADDER[:m]
    for correlated in (0.0, 1.0):
        H, _ = geometry.predicted_rank_constrained_geometry(*geometry.modes(alpha, Z, correlated), d)
        P = leverage(feature_vectors(H, Z)[0])[1]
        assert P[0, 1] < -0.02, P[0, 1]


# ── the exact tied-map Fantope program ─────────────────────────────────────────────────────

def test_tied_map_ce_gradient_matches_finite_differences_on_the_fantope():
    # F(P) = E CE(softmax(A P z)); its gradient is Z^T (q - p*) W / n, symmetrised
    alpha, Z = LADDER[:8], _correlated(8, n=2048, rho=0.4)
    rng = np.random.default_rng(0)
    P = geometry._fantope_project(rng.normal(size=(8, 8)), 3)
    f, G = geometry.tied_map_ce(alpha, Z, P)
    h = 1e-5
    for i, j in [(0, 0), (0, 1), (2, 5), (7, 7)]:
        Dm = np.zeros((8, 8)); Dm[i, j] = Dm[j, i] = h
        fd = (geometry.tied_map_ce(alpha, Z, P + Dm)[0] - geometry.tied_map_ce(alpha, Z, P - Dm)[0]) / (2 * h)
        assert abs(fd - (G[i, j] + G[j, i]) * (1.0 if i != j else 0.5)) < 1e-6


def test_exact_fantope_program_lowers_the_tied_ce_by_leaving_the_projectors():
    # the exact tied-map program minimises the objective the linearised one expands, so
    # its tied CE is no worse at its own optimum; it gets there by moving far from the
    # projectors (a larger relaxation gap eta), which is why the pipeline keeps the
    # linearised program for the frame
    alpha, Z = LADDER[:16], _bernoulli(16, n=4096)
    di, _ = allocate_d(alpha, Z, 3)
    a = geometry.signed_pressure(alpha, Z, di)
    P_lin, eta_lin = geometry.fantope_geometry(alpha, Z, 3, a)
    P_ex, eta_ex = geometry.fantope_geometry(alpha, Z, 3)
    assert geometry.tied_map_ce(alpha, Z, P_ex)[0] <= geometry.tied_map_ce(alpha, Z, P_lin)[0] + 1e-9
    assert eta_ex > 2 * eta_lin
    assert abs(np.trace(P_ex) - 3) < 1e-6 and np.abs(np.diag(P_ex) - np.diag(P_lin)).max() < 0.1


