"""The superposition theory (geometry.py, notes/superposition_new.tex): the rank-d
cross-entropy optimum by descent of the profiled CE on the decoder subspace, the representation
profiled out by exponential-family moment matching; and the measurements every model class is
read with. Every test pins an analytic answer, an oracle, or a null control."""
import os

import numpy as np
import pytest

from src.code import geometry
from src.code.analysis_utils import feature_vectors, leverage, logit_map, walsh_design, walsh_vectors

LADDER = 8.0 * (1.0 + np.arange(64)) ** -0.5          # the standard power-law importance ladder


def _bernoulli(m, n=16384, p=0.3, seed=0):
    return (np.random.default_rng(seed).random((n, m)) < p).astype(float)


# ── the theory ─────────────────────────────────────────────────────────────────────────────

def _task(m, n, p=0.3, seed=0):
    """A feature-cube cell: ladder alpha, features Z and the target (G, P, Gc) they define."""
    Z = _bernoulli(m, n=n, p=p, seed=seed)
    alpha = LADDER[:m]
    return (alpha, Z) + geometry.softmax_stats(alpha, Z)


def _centred_frame(rng, d, m):
    """d orthonormal rows orthogonal to the ones vector: a random decoder subspace."""
    D = rng.normal(size=(d, m))
    D -= D.mean(1, keepdims=True)
    U, _, Vt = np.linalg.svd(D, full_matrices=False)
    return U @ Vt


def _ce(L, G, P):
    return geometry.excess_ce(L, G, P)


def test_moment_match_is_the_i_projection_and_beats_least_squares_in_the_same_subspace():
    # for a fixed decoder subspace the CE-optimal representation matches the retained moments
    # D q = D p exactly, and its CE is below that of the least-squares logits in the subspace
    alpha, Z, G, P, Gc = _task(8, 512)
    D = _centred_frame(np.random.default_rng(0), 3, 8)
    E, Q = geometry.moment_match(D, P)
    assert np.abs((Q - P) @ D.T).max() < 1e-8
    assert _ce(E @ D, G, P) < _ce(Gc @ D.T @ D, G, P) - 1e-3


def test_rank_constrained_geometry_recovers_the_target_at_full_rank():
    # d = m - 1 spans every centred logit direction: the optimum is the Bayes table itself
    alpha, Z, G, P, Gc = _task(6, 256)
    H, D = geometry.rank_constrained_geometry(alpha, Z, 5)
    L = H @ D.T
    assert _ce(L, G, P) < 1e-6
    assert np.abs(L - L.mean(1, keepdims=True) - Gc).max() < 1e-3


def test_rank_constrained_geometry_is_a_stationary_point_of_the_joint_problem_and_not_a_truncation():
    # the returned (H, D) is a stationary point of the joint rank-d problem: moment matching zeroes
    # the E-gradient, the projected gradient zero makes the D-gradient follow, and joint L-BFGS
    # over (E, D) started there does not move. The problem is multimodal even at m = 6, d = 2
    # (random starts reach 0.160-0.217 nats), so no start-independent oracle exists. The fit sits
    # below the moment frame it starts from and below the SVD truncation of the target logits.
    from scipy.optimize import minimize
    from scipy.special import logsumexp
    alpha, Z, G, P, Gc = _task(6, 256, seed=1)
    d, n, m = 2, len(Z), 6
    H, D = geometry.rank_constrained_geometry(alpha, Z, d)
    L = H @ D.T
    q = np.exp(L - logsumexp(L, axis=1, keepdims=True))
    assert np.abs((q - P) @ D).max() < 1e-8 and np.abs(H.T @ (q - P) / n).max() < 1e-6
    ce_fit = _ce(L, G, P)

    def fg(x):
        E_, D_ = x[:n * d].reshape(n, d), x[n * d:].reshape(d, m)
        L_ = E_ @ D_
        lse = logsumexp(L_, axis=1)
        q_ = np.exp(L_ - lse[:, None])
        return (float((lse - (P * L_).sum(1)).mean()),
                np.concatenate([((q_ - P) @ D_.T).ravel(), (E_.T @ (q_ - P)).ravel()]) / n)
    x = minimize(fg, np.concatenate([H.ravel(), D.T.ravel()]), jac=True, method="L-BFGS-B",
                 options=dict(maxiter=50000, ftol=1e-15, gtol=1e-12)).x
    ce_joint = _ce(x[:n * d].reshape(n, d) @ x[n * d:].reshape(d, m), G, P)
    assert abs(ce_fit - ce_joint) < 1e-7, (ce_fit, ce_joint)

    D0 = geometry._moment_frame(Z, P, d)
    E0, _ = geometry.moment_match(D0, P, iters=100, tol=1e-10)
    V = np.linalg.svd(Gc, full_matrices=False)[2][:d]
    assert ce_fit < _ce(E0 @ D0, G, P) - 1e-4 and ce_fit < _ce(Gc @ V.T @ V, G, P) - 1e-3


def test_decoder_frames_are_centred_orthonormal_and_the_fit_is_deterministic():
    alpha, Z, G, P, Gc = _task(8, 256)
    D0 = geometry._moment_frame(Z, P, 3)
    H, D = geometry.rank_constrained_geometry(alpha, Z, 3)
    for F in (D0, D.T):
        assert np.allclose(F @ F.T, np.eye(3), atol=1e-10) and np.abs(F.sum(1)).max() < 1e-10
    H2, D2 = geometry.rank_constrained_geometry(alpha, Z, 3)
    assert np.array_equal(H, H2) and np.array_equal(D, D2)


def test_the_fit_grows_a_soft_decoder_tail_beyond_the_moment_frame():
    # the moment frame is a hard cut on the head outputs; the CE optimum shares its dimensions
    # with the tail outputs, whose total share grows past the frame's
    alpha, Z, G, P, Gc = _task(12, 2048)
    d = 3
    D0 = geometry._moment_frame(Z, P, d)
    H, D = geometry.rank_constrained_geometry(alpha, Z, d)
    tail0, tail = np.diag(D0.T @ D0)[d:].sum(), np.diag(D @ D.T)[d:].sum()
    print("tail share: moment frame", tail0, "fit", tail)
    assert tail > 2 * tail0 and tail > 0.2


def test_shuffled_targets_destroy_the_first_order_structure():
    # the measurement (first-order vectors -> leverage) sees the ladder when the target follows
    # the features, and nothing when the target rows are shuffled against them
    alpha, Z, G, P, Gc = _task(12, 2048)
    d = 3

    def r2(H):
        X = np.column_stack([Z, np.ones(len(Z))])
        R = H - X @ np.linalg.lstsq(X, H, rcond=None)[0]
        return 1.0 - (R ** 2).sum() / ((H - H.mean(0)) ** 2).sum()
    H, D = geometry.profiled_fit(Z, P, d)
    lev = leverage(feature_vectors(H, Z)[0])[0]
    assert r2(H) > 0.5 and lev[:d].sum() > 0.6 * d
    perm = np.random.default_rng(0).permutation(len(Z))
    H_s, D_s = geometry.profiled_fit(Z, P[perm], d)
    lev_s = leverage(feature_vectors(H_s, Z)[0])[0]
    print("first-order R^2: paired", r2(H), "shuffled", r2(H_s), "| top-3 leverage: paired", lev[:d].sum(), "shuffled", lev_s[:d].sum())
    assert r2(H_s) < 0.05 and lev_s[:d].sum() < 0.6 * d


@pytest.mark.slow
def test_rank_constrained_geometry_reproduces_a_converged_trained_cell():
    # the reference cell (p 0.3, beta 0.5, rho 0, d 3, seed 4): the trained network sits at the
    # optimum (excess CE equal to four digits) and so does its geometry: the decoder subspaces
    # coincide and the decoder shares agree output by output (pilot of 2026-09-15)
    from src.code.analysis_utils import has_run, load_run
    cell = ["setting=superposition", "runner=linear", "structure_kwargs.p=0.3", "structure_kwargs.beta=0.5",
            "structure_kwargs.rho=0.0", "model_kwargs.model_config.embed_dim=3", "random_seed=4",
            "training_kwargs.train_steps=8000"]
    if not has_run(cell):
        pytest.skip("the reference cell is not trained in this cache")
    run = load_run(cell)
    alpha, Z = run["dataset"].info["alpha"], np.asarray(run["dataset"].features, float)
    G, P, _ = geometry.softmax_stats(alpha, Z)
    L_tr = run["representation"] @ run["readout"].T
    H, D = geometry.rank_constrained_geometry(alpha, Z, 3)
    assert abs(_ce(H @ D.T, G, P) - _ce(L_tr, G, P)) < 1e-3
    V_tr = np.linalg.svd(L_tr - L_tr.mean(1, keepdims=True), full_matrices=False)[2][:3]
    assert (np.linalg.svd(D.T @ V_tr.T, compute_uv=False) ** 2).sum() / 3 > 0.95
    assert np.abs(np.diag(D @ D.T) - np.diag(V_tr.T @ V_tr))[:8].max() < 0.1


# ── the measurements every class is read with ─────────────────────────────────────────────

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


def test_feature_vectors_are_the_first_walsh_level_of_an_additive_code():
    # H = Z V + v0 with no noise: on the standardized features the coefficients are std_i V_i
    # and the constant absorbs the means, both exactly
    rng = np.random.default_rng(0)
    Z = (rng.random((40, 5)) < 0.4).astype(float)
    V_true, v0_true = rng.normal(size=(5, 3)), rng.normal(size=3)
    V, v0 = feature_vectors(Z @ V_true + v0_true, Z)
    assert np.allclose(V, Z.std(0)[:, None] * V_true, atol=1e-9)
    assert np.allclose(v0, v0_true + Z.mean(0) @ V_true, atol=1e-9)


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
    assert np.allclose(R, Z.std(0)[:, None] * (R0 @ J), atol=1e-9)
    assert np.allclose(b, (b0 + Z.mean(0) @ R0) @ J, atol=1e-9)
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
    assert np.allclose(E1, Z.std(0)[:, None] * E, atol=1e-8)
    assert np.allclose(h[:3], 1.0, atol=1e-6) and h[3:].max() < 1e-6
    assert np.allclose(P[:3, :3], np.eye(3), atol=1e-6)
    A = np.diag([3.0, 0.5, 2.0])
    E2 = feature_vectors(H @ A, Z)[0]
    assert np.allclose(leverage(E2)[0], h) and np.allclose(leverage(E2)[1], P)
    assert not np.allclose(E2 @ E2.T, E1 @ E1.T)


def test_modes_are_the_features_when_independent_and_the_predictive_modes_when_not():
    Z = _bernoulli(4, n=64)
    W, X = geometry.modes(np.ones(4), Z, 0.0)
    assert W.shape == (4,) and np.array_equal(X, Z)
    W, X = geometry.modes(np.ones(4), Z, 0.4)
    assert W.shape == (4, 4) and np.allclose(X @ geometry.whiten(Z)[1], Z)


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


def _correlated(m, n=8192, p=0.3, rho=0.5, seed=0):
    from src.code.data import feature_cube
    return feature_cube(n, m, p, scale=8.0, seed=seed, spectrum="power", beta=0.5, rho=rho)[1]["Z"]


def test_whiten_decorrelates_the_features_and_inverts():
    # u = C^{-1/2} z has identity covariance and Z = U C^{1/2}
    Z = _correlated(8)
    U, C_half = geometry.whiten(Z)
    assert np.allclose(np.cov(U.T, bias=True), np.eye(8), atol=1e-8)
    assert np.allclose(U @ C_half, Z, atol=1e-8)
    assert np.allclose(C_half @ C_half, np.cov(Z.T, bias=True), atol=1e-8)


def test_walsh_design_is_orthogonal_across_levels_and_standardized():
    # the constant, the standardized features, and the pair products residualized on both, so no
    # pair term can leak into the first-order vectors, even when the features are correlated
    Z = _correlated(8, n=4096)
    U = geometry.whiten(Z)[0]
    A, blocks = walsh_design(U, order=2, n_top=6)
    assert A.shape == (4096, 1 + 8 + 15)
    assert np.allclose(A[:, blocks["const"]], 1.0)
    first, second = A[:, blocks["first"]], A[:, blocks["second"]]
    assert np.abs(first.mean(0)).max() < 1e-10 and np.allclose(first.std(0), 1.0)
    assert np.abs(np.column_stack([np.ones(len(U)), first]).T @ second).max() < 1e-8
    assert np.allclose(second.std(0), 1.0)


def test_walsh_vectors_recover_a_planted_expansion_exactly():
    rng = np.random.default_rng(3)
    Z = (rng.random((512, 6)) < 0.3).astype(float)
    A, blocks = walsh_design(Z, order=2, n_top=6)
    C = rng.normal(size=(A.shape[1], 3))
    V, b = walsh_vectors(A @ C, Z, order=2, n_top=6)
    assert np.allclose(V, C, atol=1e-9) and b == blocks


# ── the note's closed forms on the signed feature cube ─────────────────────────────────────

def test_weak_signed_features_are_truncated_to_the_strongest_by_svd():
    # weak logits (notes/superposition_new.tex): the rank-d optimum keeps the d strongest features
    # and drops the rest, and its CE matches the SVD truncation of the target logits to O(eps^4)
    from src.code.data import feature_cube
    L, info = feature_cube(m=6, signed=True, all_bundles=True, scale=0.1, spectrum="power", beta=0.5)
    Z, P = info["Z"], np.exp(L)
    H, D = geometry.profiled_fit(Z, P, 3)
    lev = leverage(feature_vectors(H, Z)[0])[0]
    assert np.allclose(lev[:3], 1.0, atol=0.05) and lev[3:].max() < 0.05, lev
    Lc = L - L.mean(1, keepdims=True)
    U, S, Vt = np.linalg.svd(Lc, full_matrices=False)
    ce_fit, ce_svd = geometry.excess_ce(H @ D.T, L, P), geometry.excess_ce((U[:, :3] * S[:3]) @ Vt[:3], L, P)
    assert ce_fit <= ce_svd + 1e-9 and ce_svd - ce_fit < 1e-4, (ce_fit, ce_svd)


def test_sharp_signed_singletons_pack_feature_lines_evenly_in_the_plane():
    # sharp singletons of equal strength at d = 2: the optimum's feature lines D_i+ - D_i- are
    # evenly spaced, the largest |cos| between distinct features at cos(pi/m): the regular 2m-gon
    from src.code.data import feature_cube
    m = 4
    L, info = feature_cube(m=m, k=1, signed=True, all_bundles=True, scale=10.0, decay=1.0)
    Z, P = info["Z"], np.exp(L)
    H, D = geometry.profiled_fit(Z, P, 2)
    U = D[:m] - D[m:]
    C = np.abs(U @ U.T) / np.outer(np.linalg.norm(U, axis=1), np.linalg.norm(U, axis=1))
    np.fill_diagonal(C, 0.0)
    print("largest |cos| between feature lines", C.max(), "vs cos(pi/m)", np.cos(np.pi / m), "| excess CE", geometry.excess_ce(H @ D.T, L, P))
    assert abs(C.max() - np.cos(np.pi / m)) < 0.05, C.max()
