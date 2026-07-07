"""The stacking (demo_graphs §2) setup, the uncertainty (§3) plot, and per-structure draw layouts."""
import matplotlib
matplotlib.use("Agg")
import numpy as np

from src.code.data import graph_conditional, stacked_conditional
from src.code.geometry import predicted_geometry


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
    fig = plot_utils.plot_stacking(n=10, q=8, n_features=5, sae_steps=200)
    assert len(fig.axes) == 4                                        # union / product / share / superpose


# ── per-structure draw layouts ────────────────────────────────────────────────

def test_structure_graph_layouts_for_all_structures():
    from src.code import plot_utils
    specs = [
        {"structure": "ring", "n": 12}, {"structure": "chain", "n": 10},
        {"structure": "grid", "n": 4, "m": 4}, {"structure": "complete", "n": 5},
        {"structure": "sphere", "n": 40, "k": 6}, {"structure": "hypercube", "n": 3},
        {"structure": "communities", "k": 3, "n": 8},
        {"structure": "tree", "n": 3, "k": 2}, {"structure": "log_line", "n": 15},
        {"structure": "product", "factors": [{"structure": "ring", "n": 8}, {"structure": "chain", "n": 5}]},
        {"structure": "stack", "factors": [{"structure": "ring", "n": 8}, {"structure": "ring", "n": 8}]},
    ]
    for s in specs:
        A, pos = plot_utils._structure_graph(s)
        assert A.shape[0] == A.shape[1] == len(pos), s
        assert pos.shape[1] in (2, 3), s
