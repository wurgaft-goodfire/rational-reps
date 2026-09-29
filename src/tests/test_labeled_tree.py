"""A tree whose edges carry labels (relation types): log p(y|x) = -gamma/2 ||p(x) - p(y)||^2 with p(x) the sum,
over the edges on x's root path, of a private unit step plus sqrt(alpha) times the edge label's shared step.
The positions are polar: every edge vector has the same length (distance marks an edge), same-label edges
share a direction with cosine alpha / (1 + alpha), different labels are orthogonal."""
import numpy as np
import pytest

from src.code.data import (adjacency_conditional, graph_conditional, labeled_tree, labeled_tree_positions,
                           tree_adjacency, tree_parents)

PARENTS = tree_parents(n=2, k=3)                                               # 13 nodes, 12 edges
LABELS = [0] + [PARENTS[1:c].count(PARENTS[c]) for c in range(1, 13)]         # birth order among siblings


def _edge_vectors(E, parents):
    return np.array([E[c] - E[p] for c, p in enumerate(parents) if p >= 0])


def _cosines(V):
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    return V @ V.T


@pytest.mark.parametrize("alpha", [0.25, 1.0, 4.0])
def test_positions_are_polar_with_cosine_alpha_over_one_plus_alpha(alpha):
    V = _edge_vectors(labeled_tree_positions(PARENTS, LABELS, alpha), PARENTS)
    assert np.allclose(np.linalg.norm(V, axis=1), np.sqrt(1 + alpha))         # distance marks an edge
    lab = np.array(LABELS[1:]); same, off = lab[:, None] == lab[None], ~np.eye(12, dtype=bool)
    C = _cosines(V)
    assert np.allclose(C[same & off], alpha / (1 + alpha)) and np.allclose(C[~same], 0.0)


def test_doubly_centered_table_is_the_gram_of_the_positions():
    # centering over outcomes and items leaves gamma Pc Pc^T exactly: the positions are the code of a
    # readout with a bias. Bias-free, the depth term -gamma/2 ||p_y||^2 stays as a column effect.
    L = labeled_tree(n=2, k=3, alpha=1.0, gamma=0.7)
    Pc = labeled_tree_positions(PARENTS, LABELS, 1.0); Pc = Pc - Pc.mean(0)
    Lcc = L - L.mean(1, keepdims=True); Lcc = Lcc - Lcc.mean(0)
    assert np.allclose(Lcc, 0.7 * Pc @ Pc.T, atol=1e-9)


def test_alpha_zero_is_the_geodesic_tree_kernel():
    # without labels the kernel is -gamma/2 d_tree: the geodesic kernel with decay exp(-gamma/2)
    L = labeled_tree(n=2, k=3, alpha=0.0, gamma=1.5)
    G = adjacency_conditional(tree_adjacency(PARENTS), np.exp(-0.75), "geodesic")
    assert np.allclose(L, G, atol=1e-9)


def test_explicit_labels_set_the_shared_directions():
    # a path root -> 1 -> 2 -> 3 with labels a, a, b: edges 1 and 2 share a direction, 3 is orthogonal
    parents, labels = [-1, 0, 1, 2], [-1, 0, 0, 1]
    C = _cosines(_edge_vectors(labeled_tree_positions(parents, labels, 1.0), parents))
    assert np.allclose(C, [[1, 0.5, 0], [0.5, 1, 0], [0, 0, 1]])


def test_labeled_tree_is_a_registered_structure():
    L = graph_conditional({"structure": "labeled_tree", "n": 2, "k": 3, "alpha": 1.0, "gamma": 1.0})
    assert L.shape == (13, 13) and np.allclose(np.exp(L).sum(1), 1.0)
