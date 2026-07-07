"""Order-2 gallery: graded/semilattice operations factor cleanly into their geometry
(demo_graphs §4). Group operations are degenerate under factor_tensor -> their circle is
validated via a trained model (see test_convergence)."""
import numpy as np

from src.code.data import maximum, averaging, set_intersection, entailment, relation_tensor as build_tensor
from src.code.geometry import factor_tensor
from src.code.eval import rsa


def test_maximum_is_an_ordered_line():
    E, _ = factor_tensor(maximum(10))
    assert abs(np.corrcoef(E[:, 0], np.arange(10))[0, 1]) > 0.95


def test_averaging_is_a_flat_plane():
    E, _ = factor_tensor(averaging(4))
    grid = np.array([(i, j) for i in range(4) for j in range(4)], float)
    assert rsa(E[:, :2], grid) > 0.9                               # recovers the 2-D grid


def test_set_intersection_is_a_hypercube():
    _, S = factor_tensor(set_intersection(3))
    assert S[2] > 0.5 * S[0] and S[3] < 0.7 * S[0]                 # rank = nbit = 3 binary coords


def test_entailment_is_a_nested_hierarchy():
    # nodes: root=0, children=1..3, grandchildren of child1={4,5,6}, of child2={7,8,9}
    E, _ = factor_tensor(entailment(depth=2, branch=3))
    def mean_dist(idx_a, idx_b):
        return np.mean([np.linalg.norm(E[a] - E[b]) for a in idx_a for b in idx_b if a != b])
    within = mean_dist([4, 5, 6], [4, 5, 6])
    across = mean_dist([4, 5, 6], [7, 8, 9])
    assert within < across                                         # same-parent leaves are closer


def test_build_tensor_dispatch_and_arity():
    assert build_tensor({"operation": "modular_add", "p": 5}).shape == (5, 5, 5)      # arity 2
    assert build_tensor({"operation": "modular_add", "p": 5, "arity": 3}).ndim == 4   # arity 3
