"""The predicted geometry of each latent structure carries its expected signature:
the symmetry/topology of the structure sets the shape (demo_graphs §1)."""
import numpy as np

from src.code.data import graph_conditional as build_conditional
from src.code.geometry import factor_logcond, predicted_geometry


def _ES(spec, gamma=0.8):
    _, E, _, S, _ = factor_logcond(build_conditional(spec, gamma))
    return E, S


def test_geodesic_cooccurrence_is_self_similar_and_decays_with_graph_distance():
    spec = {"structure": "chain", "n": 6}
    P = np.exp(build_conditional(spec, cooccurrence="geodesic"))   # p(x | i) from the geodesic kernel
    for i in range(6):
        assert P[i].argmax() == i                                  # a node predicts itself most (dist 0)
    assert P[0, 0] > P[0, 1] > P[0, 2] > P[0, 3]                   # decays monotonically with distance


def test_adjacency_cooccurrence_is_the_raw_one_step_transition():
    from src.code.data import grid
    spec = {"structure": "grid", "n": 5, "m": 5}
    A = grid(5, 5)
    L_adj = build_conditional(spec, cooccurrence="adjacency")
    L_diff = build_conditional(spec, cooccurrence="diffusion")
    for i in range(len(A)):                                   # raw: top outcomes are exactly neighbours
        nbrs = set(np.where(A[i] > 0)[0])
        assert set(np.argsort(L_adj[i])[-len(nbrs):]) == nbrs
    assert not np.allclose(L_adj, L_diff)                     # diffusion spreads beyond neighbours


def test_chain_is_an_ordered_horseshoe():
    E, _ = _ES({"structure": "chain", "n": 40})
    assert abs(np.corrcoef(E[:, 0], np.arange(40))[0, 1]) > 0.95   # PC1 monotonic in position


def test_grid_is_rank_two():
    _, S = _ES({"structure": "grid", "n": 6, "m": 6})
    assert S[1] / S[0] > 0.8 and S[2] / S[1] < 0.6                 # a 2-D sheet


def test_sphere_is_rank_three_harmonics():
    _, S = _ES({"structure": "sphere", "n": 150, "k": 8})
    assert S[2] / S[0] > 0.5 and S[3] / S[0] < 0.3                 # degree-1 harmonics x,y,z


def test_complete_graph_is_a_simplex():
    E, _ = _ES({"structure": "complete", "n": 6})
    D = np.sqrt(((E[:, None] - E[None]) ** 2).sum(-1))
    iu = np.triu_indices(6, 1)
    assert D[iu].std() / D[iu].mean() < 0.02                       # all items equidistant


def test_hypercube_supports_analogies():
    E = predicted_geometry(build_conditional({"structure": "hypercube", "n": 3}))
    idx = lambda bits: sum(b << k for k, b in enumerate(bits))
    man, woman, king, queen = idx((0, 0, 0)), idx((0, 1, 0)), idx((1, 0, 0)), idx((1, 1, 0))
    pred = E[king] - E[man] + E[woman]                             # king - man + woman
    assert int(np.argmin(((E - pred) ** 2).sum(1))) == queen       # == queen


def test_ring_product_ring_is_rank_four_torus():
    _, S = _ES({"structure": "product",
                "factors": [{"structure": "ring", "n": 12}, {"structure": "ring", "n": 8}]})
    assert S[3] > 2 * S[4]                                          # two Fourier pairs (a torus)


def test_stack_of_two_rings_is_two_circles():
    E, _ = _ES({"structure": "stack",
                "factors": [{"structure": "ring", "n": 12}, {"structure": "ring", "n": 12}]})
    # each disjoint block embeds as its own circle in its own subspace (union geometry);
    # recover each block's local top-2 by centering + SVD on that block's rows.
    for block in (slice(0, 12), slice(12, 24)):
        Xb = E[block] - E[block].mean(0)
        U, S, _ = np.linalg.svd(Xb, full_matrices=False)
        r = np.linalg.norm((U[:, :2] * S[:2]), axis=1)
        assert r.std() / r.mean() < 0.1


def test_tree_includes_inner_nodes_and_supports_unbalanced():
    # leaves-only by default: balanced depth-2 branch-3 -> 9 leaves
    assert build_conditional({"structure": "tree", "n": 2, "k": 3}).shape == (9, 9)
    # opt into inner-node items (Park et al.): 1 + 3 + 9 = 13 nodes
    assert build_conditional({"structure": "tree", "n": 2, "k": 3, "leaves_only": False}).shape == (13, 13)
    # unbalanced via an explicit parent structure (the Park Organism->...->Dog example)
    L = build_conditional({"structure": "tree", "parents": [-1, 0, 0, 1, 1, 1, 1, 5, 5, 5], "leaves_only": False})
    assert L.shape == (10, 10)


def test_tree_is_orthogonal_direct_sum_of_simplices():
    # Park et al.: hierarchically related concepts occupy orthogonal subspaces, each a simplex.
    # nodes (DFS pre-order): 0=root; each level-1 node is followed by its 3 leaves,
    # so the level-1 inner nodes are [1, 5, 9] and their leaves [2,3,4]/[6,7,8]/[10,11,12].
    E = predicted_geometry(build_conditional({"structure": "tree", "n": 2, "k": 3, "leaves_only": False}))
    E = E - E.mean(0)

    def edge_cv(P):
        D = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1)); iu = np.triu_indices(len(P), 1)
        return D[iu].std() / D[iu].mean()

    def dirs(M):
        Mc = M - M.mean(0); U, S, Vt = np.linalg.svd(Mc, full_matrices=False)
        return Vt[S > 1e-6 * S[0]]

    top = E[[1, 5, 9]]                                      # top split over the inner-node reps
    within = np.vstack([E[g] - E[g].mean(0) for g in ([2, 3, 4], [6, 7, 8], [10, 11, 12])])
    assert edge_cv(top) < 0.05                              # regular simplex at the top split
    assert np.abs(dirs(top) @ dirs(within).T).max() < 0.05  # orthogonal direct sum


def test_log_line_compresses_large_magnitudes():
    E = predicted_geometry(build_conditional({"structure": "log_line", "n": 30}))
    gaps = np.abs(np.diff(E[:, 0]))
    assert gaps[0] > 5 * gaps[-1]                                   # Weber: small numbers spread, large bunch


def test_view_defaults_per_structure_and_override():
    from src.code.plot_utils import _view
    assert _view({"structure": "complete", "n": 6}) == "3d"          # a simplex -> 3-D
    assert _view({"structure": "hypercube", "n": 4}) == "3d"
    assert _view({"structure": "sphere", "n": 50}) == "3d"
    assert _view({"structure": "tree", "n": 2, "k": 3}) == "ultrametric"
    assert _view({"structure": "ring", "n": 12}) == "2d"
    assert _view({"structure": "ring", "n": 12, "view": "3d"}) == "3d"   # yaml structure_kwarg override


def test_view_kwarg_is_presentation_only_and_never_reaches_builders():
    from src.code.data import graph_conditional, structure_adjacency
    assert graph_conditional({"structure": "ring", "n": 8, "view": "3d"}).shape == (8, 8)
    assert structure_adjacency({"structure": "ring", "n": 8, "view": "3d"}).shape == (8, 8)


def test_structure_adjacency_supports_tree_and_log_line():
    # Unified on adjacency: every structure yields an adjacency, so the HMM latent walk and the
    # structure-drawing path accept tree/log_line too — not just the graph-conditional path.
    from src.code.data import structure_adjacency
    A_tree = structure_adjacency({"structure": "tree", "n": 2, "k": 3})   # all nodes: 1 + 3 + 9 = 13
    assert A_tree.shape == (13, 13)
    assert np.allclose(A_tree, A_tree.T) and A_tree.sum() > 0             # symmetric tree edges
    A_line = structure_adjacency({"structure": "log_line", "n": 8})       # log_line's graph is a line
    assert A_line.shape == (8, 8)
    assert A_line[np.arange(7), np.arange(1, 8)].sum() == 7               # consecutive-node edges
