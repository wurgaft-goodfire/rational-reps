"""The superposition oracle (geometry.py): closed-form geometry of a rank-limited softmax
on Bernoulli features, from notes/superposition.tex. Every test pins an analytic answer."""
import numpy as np

from src.code import geometry
from src.code.geometry import (balanced_code, feature_directions, feature_objective,
                               feature_sizes, feature_vectors, fitted_features,
                               predicted_features)

LADDER = 8.0 * (1.0 + np.arange(64)) ** -0.5          # the standard power-law importance ladder


def test_objective_scores_strength_error_and_overlap_by_hand():
    # orthogonal at target strength: J = 0. Two identical unit vectors at target strength:
    # only the overlap term survives, <w1,w2>^2 = 1 counted for both orderings -> J = p.
    W = np.array([[2.0, 0.0], [0.0, 3.0]])            # exact in binary: J = 0 exactly
    assert feature_objective(W, np.array([4.0, 9.0]), 0.3) == 0.0
    W = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert np.isclose(feature_objective(W, np.array([1.0, 1.0]), 0.3), 0.3)


def test_sizes_equal_importance_closed_form_and_exact_at_full_budget():
    # all alpha equal: everything is shared, s = alpha / (1 - p + p m/d); at d = m
    # orthogonality is free and s = alpha exactly.
    alpha = np.full(8, 2.0)
    s, _ = feature_sizes(alpha, 0.3, d=4)
    assert np.allclose(s, 2.0 / (1 - 0.3 + 0.3 * 8 / 4))
    s, _ = feature_sizes(alpha, 0.3, d=8)
    assert np.allclose(s, 2.0)


def test_a_dominant_feature_takes_its_own_dimension_at_exact_strength():
    # alpha = [10, 1 x7], p = 0.3, d = 2. Budget with the top feature dedicated:
    # alpha_star = 7 / (0.7 + 0.3 * 7) = 2.5, so 10 > alpha_star is dedicated at s = 10
    # and the seven others share one dimension at s = (1 - 0.75)/0.7 = 5/14.
    alpha = np.array([10.0] + [1.0] * 7)
    s, astar = feature_sizes(alpha, 0.3, d=2)
    assert np.isclose(astar, 2.5)
    assert s[0] == 10.0
    assert np.allclose(s[1:], 5.0 / 14.0)


def test_capacity_level_satisfies_the_dimension_budget_exactly():
    # the one scalar alpha_star must spend exactly d dimensions: dedicated features count 1,
    # shared features their fraction, dropped features 0.
    for d in (3, 6, 12):
        _, astar = feature_sizes(LADDER, 0.3, d)
        frac = np.clip((LADDER / astar - 0.3) / 0.7, 0.0, 1.0)
        assert np.isclose(frac.sum(), d, atol=1e-9)


def test_shared_features_sit_at_a_common_threshold():
    # in the shared regime alpha_i - (1-p) s_i = p alpha_star, the same for every feature
    sizes, astar = feature_sizes(LADDER, 0.3, d=6)
    own = LADDER > astar
    shared = ~own & (sizes > 0)
    assert np.allclose(LADDER[shared] - 0.7 * sizes[shared], 0.3 * astar, atol=1e-9)
    assert own.sum() >= 1                             # the ladder head forces a dedicated axis
    assert shared.sum() > 6                           # superposition: more features than dims


def test_directions_hit_the_analytic_minimum_overlap():
    # row norms exact, dedicated rows exactly orthogonal to everything, and the shared
    # block's total overlap equals (sum s)^2/k - sum s^2 -- the tight-frame minimum.
    rng = np.random.default_rng(0)
    for _ in range(5):
        k = 5
        s_shared = rng.uniform(0.5, 1.5, 12)          # max < sum/k: an even spread exists
        sizes = np.concatenate([[4.0, 3.0], s_shared])
        own = np.array([True, True] + [False] * 12)
        W = feature_directions(sizes, own, d=k + 2)
        assert np.allclose((W ** 2).sum(1), sizes, atol=1e-9)
        G = W @ W.T
        assert np.allclose(G[0, 1:], 0.0, atol=1e-12) and np.allclose(G[1, 2:], 0.0, atol=1e-12)
        off = G.copy()
        np.fill_diagonal(off, 0.0)
        target = s_shared.sum() ** 2 / k - (s_shared ** 2).sum()
        assert np.isclose((off ** 2).sum(), target, atol=1e-8)


def test_spectrum_is_dedicated_spikes_over_a_flat_frame():
    # spec+(W^T W) = {alpha_i : dedicated} plus alpha_star repeated on the shared block
    W, sizes, own, astar = predicted_features(LADDER, 0.3, d=6)
    ev = np.sort(np.linalg.eigvalsh(W.T @ W))[::-1]
    expected = np.sort(np.concatenate([LADDER[own], np.full(6 - own.sum(), astar)]))[::-1]
    assert np.allclose(ev, expected, atol=1e-9)


def test_closed_form_matches_direct_minimization_of_the_objective():
    # the oracle: L-BFGS on the objective itself, from random starts, must land on the
    # closed form's sizes and value.
    alpha = LADDER[:16]
    p, d = 0.3, 3
    W_pred, s_pred, own, _ = predicted_features(alpha, p, d)
    W_fit = fitted_features(alpha, p, d, n_starts=4)
    assert np.allclose((W_fit ** 2).sum(1), s_pred, atol=2e-3)
    J_pred = feature_objective(W_pred, alpha, p)
    J_fit = feature_objective(W_fit, alpha, p)
    assert J_pred <= J_fit * (1 + 1e-6)               # the formula is at least as good


def test_full_budget_represents_everything_orthogonally():
    # m <= d: capacity does not bind, every feature gets its own axis at s = alpha
    alpha = np.array([3.0, 2.0, 1.0])
    W, sizes, own, astar = predicted_features(alpha, 0.3, d=5)
    assert np.allclose(sizes, alpha) and own.all()
    assert np.allclose(W @ W.T, np.diag(alpha), atol=1e-12)


def test_feature_vectors_recover_an_additive_code_exactly():
    # H = Z V + v0 with no noise: least squares must return V and v0 exactly
    rng = np.random.default_rng(0)
    Z = (rng.random((40, 5)) < 0.4).astype(float)
    V_true, v0_true = rng.normal(size=(5, 3)), rng.normal(size=3)
    V, v0 = feature_vectors(Z @ V_true + v0_true, Z)
    assert np.allclose(V, V_true, atol=1e-9) and np.allclose(v0, v0_true, atol=1e-9)


def test_balanced_code_factors_the_row_centred_logits_evenly():
    # the gauge every comparison uses: H @ D^T reproduces the centred logits and the two
    # factors carry equal scale in every direction
    rng = np.random.default_rng(1)
    E, readout = rng.normal(size=(30, 3)), rng.normal(size=(7, 3))
    H, D = balanced_code(E, readout)
    G = E @ readout.T
    assert np.allclose(H @ D.T, G - G.mean(1, keepdims=True), atol=1e-9)
    assert np.allclose(np.linalg.norm(H, axis=0), np.linalg.norm(D, axis=0), atol=1e-9)
    assert H.shape == (30, 3) and D.shape == (7, 3)


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


def test_fantope_projection_returns_a_feasible_point_and_fixes_feasible_inputs():
    rng = np.random.default_rng(3)
    A = rng.normal(size=(7, 7)); A = 0.5 * (A + A.T)
    H = geometry._fantope_project(A, 3)
    lam = np.linalg.eigvalsh(H)
    assert abs(np.trace(H) - 3) < 1e-8
    assert lam.min() > -1e-10 and lam.max() < 1 + 1e-10
    P = np.zeros((7, 7)); P[:3, :3] = np.eye(3)       # a rank-3 projector is a fixed point
    assert np.allclose(geometry._fantope_project(P, 3), P, atol=1e-8)


def test_fantope_leverage_is_symmetric_under_equal_importance():
    h, H = geometry.fantope_leverage(np.full(8, 3.0), 0.3, 3,
                                     n_samples=16384, iters=400)
    assert abs(h.sum() - 3) < 1e-6
    assert h.max() - h.min() < 0.03                   # exchangeable features, equal shares
    lam = np.linalg.eigvalsh(H)
    assert lam.min() > -1e-8 and lam.max() < 1 + 1e-8


def test_fantope_leverage_orders_allocation_by_importance():
    alpha = 8.0 * (1.0 + np.arange(8)) ** -0.5
    h, _ = geometry.fantope_leverage(alpha, 0.3, 3, n_samples=16384, iters=400)
    assert np.all(np.diff(h) < 0.02)                  # decreasing along the ladder
    assert h[0] > h[-1] + 0.2
