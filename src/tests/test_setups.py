"""The stacking (demo_graphs §2) setup, the uncertainty (§3) plot, and per-structure draw layouts."""
import matplotlib
matplotlib.use("Agg")
import numpy as np

from src.code.data import graph_conditional, stacked_conditional
from src.code.geometry import predicted_geometry


def test_hypercube_and_complete_embeddings_draw_graph_edges():
    # cube / simplex embeddings draw the graph edges (legible polytope), a plain scatter does not
    import matplotlib.pyplot as plt
    from src.code.plot_utils import draw_embedding
    for spec, edges in [({"structure": "hypercube", "n": 3}, True),
                        ({"structure": "complete", "n": 6}, True),
                        ({"structure": "ring", "n": 10}, False)]:
        E = predicted_geometry(graph_conditional(spec))
        fig = plt.figure(); draw_embedding(fig, 111, E, spec, "t")
        assert (len(fig.axes[0].lines) > 0) == edges
        plt.close(fig)


def test_two_ring_product_defaults_to_torus_view():
    from src.code.plot_utils import _view
    ring, chain = {"structure": "ring", "n": 10}, {"structure": "chain", "n": 8}
    assert _view({"structure": "product", "factors": [ring, ring]}) == "torus"
    assert _view({"structure": "product", "factors": [ring, chain]}) == "3d"   # cylinder, not a torus
    assert _view({"structure": "product", "factors": [ring, ring], "view": "3d"}) == "3d"   # override


def test_ring_ring_structure_draws_a_torus_not_a_cylinder():
    # a torus is 4-D; drawn in 3-D it must be a donut (inner+outer radius), not a fixed-radius cylinder
    from src.code.plot_utils import _structure_graph
    _, pos = _structure_graph({"structure": "product",
                               "factors": [{"structure": "ring", "n": 10}, {"structure": "ring", "n": 8}]})
    assert pos.shape[1] == 3
    radial = np.hypot(pos[:, 0], pos[:, 1])
    assert radial.std() / radial.mean() > 0.1        # donut has a range of radii (a cylinder ~ 0)
    assert np.ptp(pos[:, 2]) > 0                      # and height from the minor circle


def test_product_structure_layout_is_order_independent():
    # chain (1-D) x ring (2-D) draws as a 3-D cylinder regardless of factor order — the earlier
    # code padded each factor to 2-D and truncated to 3 columns, dropping the ring's second axis
    # when the chain came first (it looked like a flat grid).
    from src.code.plot_utils import _structure_graph
    for factors in ([{"structure": "chain", "n": 8}, {"structure": "ring", "n": 10}],
                    [{"structure": "ring", "n": 10}, {"structure": "chain", "n": 8}]):
        _, pos = _structure_graph({"structure": "product", "factors": factors})
        assert pos.shape[1] == 3
        assert int((pos.std(0) > 1e-6).sum()) == 3       # all three axes informative (a full cylinder)


# ── stacking ──────────────────────────────────────────────────────────────────

def test_stacking_modes_have_expected_node_counts():
    a, b = {"structure": "ring", "n": 10}, {"structure": "ring", "n": 8}
    assert stacked_conditional([a, b], "union").shape[0] == 18       # disjoint -> sum
    assert stacked_conditional([a, b], "product").shape[0] == 80     # Cartesian -> product
    two = [{"structure": "ring", "n": 10}, {"structure": "ring", "n": 10}]
    assert stacked_conditional(two, "share").shape[0] == 10          # aligned -> same nodes


def test_share_of_aligned_rings_is_a_single_circle():
    L = stacked_conditional([{"structure": "ring", "n": 16}, {"structure": "ring", "n": 16}], "share")
    E = predicted_geometry(L)
    r = np.linalg.norm(E[:, :2], axis=1)
    assert r.std() / r.mean() < 0.05


def test_product_stack_is_rank_four_torus():
    from src.code.geometry import factor_logcond
    L = stacked_conditional([{"structure": "ring", "n": 12}, {"structure": "ring", "n": 8}], "product")
    _, _, _, S, _ = factor_logcond(L)
    assert S[3] > 2 * S[4]                                           # two Fourier pairs


# ── uncertainty plot ──────────────────────────────────────────────────────────

def test_uncertainty_plot_renders():
    from src.code import plot_utils
    E = predicted_geometry(graph_conditional({"structure": "ring", "n": 40}))
    fig = plot_utils.plot_uncertainty(E)
    assert len(fig.axes) == 3                                        # circle + two belief panels


def test_stacking_plot_renders():
    from src.code import plot_utils
    fig = plot_utils.plot_stacking(n=10, q=8)
    assert len(fig.axes) == 3                                        # union / product / share


# ── per-structure draw layouts ────────────────────────────────────────────────

def test_tree_triptych_default_is_ultrametric_dendrogram():
    from src.code import plot_utils
    spec = {"structure": "tree", "n": 2, "k": 2}
    L = graph_conditional(spec)
    fig = plot_utils.plot_predicted_geometry(spec, L, predicted_geometry(L))
    assert len(fig.axes) == 3                                   # structure + conditional + dendrogram


def test_tree_triptych_simplex_view_draws_one_panel_per_split():
    from src.code import plot_utils
    # unbalanced Park example: splits at node 0 (segment), node 1 (tetrahedron/3-D), node 5 (triangle)
    spec = {"structure": "tree", "parents": [-1, 0, 0, 1, 1, 1, 1, 5, 5, 5], "view": "simplex"}
    L = graph_conditional(spec)
    fig = plot_utils.plot_predicted_geometry(spec, L, predicted_geometry(L))
    assert len(fig.axes) == 2 + 3                               # structure + conditional + one per split


def test_tree_leaves_only_drops_inner_nodes_from_conditional_and_code():
    from src.code import plot_utils
    full = graph_conditional({"structure": "tree", "n": 2, "k": 2, "leaves_only": False})
    leaves = graph_conditional({"structure": "tree", "n": 2, "k": 2})   # leaves-only by default
    assert full.shape[0] == 7 and leaves.shape[0] == 4         # 7 nodes vs 4 leaves
    fig = plot_utils.plot_predicted_geometry({"structure": "tree", "n": 2, "k": 2},
                                             leaves, predicted_geometry(leaves))
    assert len(fig.axes) == 3                                   # dendrogram over the 4 leaves


def test_nontree_triptych_keeps_the_embedding_panel():
    from src.code import plot_utils
    spec = {"structure": "ring", "n": 12}
    L = graph_conditional(spec)
    fig = plot_utils.plot_predicted_geometry(spec, L, predicted_geometry(L))
    assert len(fig.axes) == 3                                   # structure + conditional + embedding


def test_structure_graph_layouts_for_all_structures():
    from src.code import plot_utils
    specs = [
        {"structure": "ring", "n": 12}, {"structure": "chain", "n": 10},
        {"structure": "grid", "n": 4, "m": 4}, {"structure": "complete", "n": 5},
        {"structure": "sphere", "n": 40, "k": 6}, {"structure": "hypercube", "n": 3},
        {"structure": "communities", "k": 3, "n": 8},
        {"structure": "tree", "n": 3, "k": 2},
        {"structure": "product", "factors": [{"structure": "ring", "n": 8}, {"structure": "chain", "n": 5}]},
        {"structure": "stack", "factors": [{"structure": "ring", "n": 8}, {"structure": "ring", "n": 8}]},
    ]
    for s in specs:
        A, pos = plot_utils._structure_graph(s)
        assert A.shape[0] == A.shape[1] == len(pos), s
        assert pos.shape[1] in (2, 3), s
