"""Regression tests for the RSA metric, locking the tree edge case that a tie-breaking bug broke.

Tree geometries are *ultrametric*: their RDM has only a few distinct distances, heavily tied (e.g.
the depth-3 binary tree below has just 3 distinct pairwise distances across 28 pairs). An ordinal
tie-break (ranks by array position) scrambles those ties under float noise and deflates RSA even
between identical geometries — a rotation of a tree once scored 0.81 instead of 1.0. `rsa` must
round to a tolerance and average tied ranks. We pin: the distinct-distance structure, rotation- and
tie-noise-invariance, symmetry/self-identity, and agreement with scipy's Spearman (an independent
oracle) once distances are tolerance-rounded."""
import numpy as np

from src.code.data import graph_conditional
from src.code.geometry import predicted_geometry
from src.code.eval import rsa

TREE = {"structure": "tree", "n": 3, "k": 2}          # ultrametric: few distinct, heavily-tied distances
RING = {"structure": "ring", "n": 12}
GRID = {"structure": "grid", "n": 5, "m": 5}


def _rotate(E, seed=0):
    Q, _ = np.linalg.qr(np.random.RandomState(seed).randn(E.shape[1], E.shape[1]))
    return E @ Q


def test_tree_rdm_is_heavily_tied():
    # documents the edge case: the tree's RDM collapses to a handful of distinct distances.
    E = predicted_geometry(graph_conditional(TREE))
    D = np.sqrt(((E[:, None] - E[None]) ** 2).sum(-1))[np.triu_indices(len(E), 1)]
    assert len(np.unique(np.round(D, 6))) <= 3 < len(D)        # 3 distinct distances / 28 pairs


def test_rsa_is_rotation_invariant_even_on_ultrametric_trees():
    # the exact failure: an orthogonal rotation preserves every distance, so RSA must be 1.0 —
    # including for trees, whose tied distances previously broke ordinal tie-breaking.
    for spec in (TREE, RING, GRID):
        E = predicted_geometry(graph_conditional(spec))
        assert rsa(E, _rotate(E)) > 0.9999, spec


def test_rsa_ignores_tie_breaking_float_noise_on_a_tree():
    # perturbing a tree's code far below its distance gaps must not change the ranking -> RSA ~ 1.0.
    E = predicted_geometry(graph_conditional(TREE))
    E_noisy = E + 1e-9 * np.random.RandomState(0).randn(*E.shape)
    assert rsa(E, E_noisy) > 0.9999


def test_rsa_self_identity_and_symmetry():
    E = predicted_geometry(graph_conditional(TREE))
    F = _rotate(E, seed=1) + 0.1 * np.random.RandomState(2).randn(*E.shape)
    assert rsa(E, E) > 0.9999
    assert abs(rsa(E, F) - rsa(F, E)) < 1e-12


def test_rsa_matches_scipy_spearman_with_tie_rounding():
    # independent oracle: our RSA must equal scipy's Spearman once distances are tolerance-rounded
    # the same way (the rounding — not the ranking library — is what fixes the tie bug).
    from scipy.stats import spearmanr

    def scipy_rounded(X, Y):
        d = lambda M: ((M[:, None] - M[None]) ** 2).sum(-1)[np.triu_indices(len(M), 1)]
        rnd = lambda v: np.round(v / (v.max() + 1e-12), 6)
        return spearmanr(rnd(d(X)), rnd(d(Y))).statistic

    E = predicted_geometry(graph_conditional(TREE))
    for seed in range(4):
        F = E + 0.15 * np.random.RandomState(seed).randn(*E.shape)   # a genuinely different geometry
        assert abs(rsa(E, F) - scipy_rounded(E, F)) < 1e-6, seed
