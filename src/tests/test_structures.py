"""The predicted geometry of each latent structure carries its expected signature:
the symmetry/topology of the structure sets the shape (demo_graphs §1)."""
import numpy as np

from src.code.data import graph_conditional as build_conditional
from src.code.geometry import factor_logcond, predicted_geometry


def _ES(spec, gamma=0.8):
    _, E, _, S, _ = factor_logcond(build_conditional(spec, gamma))
    return E, S


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


def test_log_line_compresses_large_magnitudes():
    E = predicted_geometry(build_conditional({"structure": "log_line", "n": 30}))
    gaps = np.abs(np.diff(E[:, 0]))
    assert gaps[0] > 5 * gaps[-1]                                   # Weber: small numbers spread, large bunch
