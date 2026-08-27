"""Order-2 gallery: graded/semilattice operations factor cleanly into their geometry
(demo_graphs §4). Group operations are degenerate -> their circle is validated via a trained
model (see test_convergence)."""
import numpy as np

from src.code.data import (maximum, averaging, set_intersection, entailment, apply_lapse,
                          relation_tensor as build_tensor)
from src.code.geometry import factor_logcond, log_conditional


def _item_geometry(T):
    """Per-item code and spectrum of a relation: its log-conditional with items on the first axis
    and every other axis flattened into columns."""
    L = log_conditional(T)
    _, E, _, S = factor_logcond(L.reshape(len(L), -1))
    return E, S


def test_maximum_is_an_ordered_line():
    E, _ = _item_geometry(maximum(10))
    assert abs(np.corrcoef(E[:, 0], np.arange(10))[0, 1]) > 0.95


def test_averaging_is_an_ordered_line_in_the_item_value():
    """In log space the average's conditional is a function of a + b, so the item code is an
    ordered line. Its top pair is degenerate, so the order lives in the plane they span, not in
    either axis alone."""
    E, _ = _item_geometry(averaging(4))
    Ec = E - E.mean(0)
    U = np.linalg.svd(Ec, full_matrices=False)[0][:, :2]
    v = np.arange(len(Ec), dtype=float) - (len(Ec) - 1) / 2
    fit = U @ np.linalg.lstsq(U, v, rcond=None)[0]
    assert 1 - ((v - fit) ** 2).sum() / (v ** 2).sum() > 0.99


def test_set_intersection_is_a_hypercube():
    _, S = _item_geometry(set_intersection(3))
    assert S[2] > 0.5 * S[0] and S[3] < 0.7 * S[0]                 # rank = nbit = 3 binary coords


def test_entailment_is_a_nested_hierarchy():
    # nodes: root=0, children=1..3, grandchildren of child1={4,5,6}, of child2={7,8,9}
    E, _ = _item_geometry(entailment(depth=2, branch=3))
    def mean_dist(idx_a, idx_b):
        return np.mean([np.linalg.norm(E[a] - E[b]) for a in idx_a for b in idx_b if a != b])
    within = mean_dist([4, 5, 6], [4, 5, 6])
    across = mean_dist([4, 5, 6], [7, 8, 9])
    assert within < across                                         # same-parent leaves are closer


def test_build_tensor_dispatch_and_arity():
    assert build_tensor({"relation": "modular_add", "p": 5}).shape == (5, 5, 5)      # arity 2
    assert build_tensor({"relation": "modular_add", "p": 5, "arity": 3}).ndim == 4   # arity 3


def test_every_relation_builds_from_the_setting_defaults_alone():
    """``relation=X`` must be enough: the setting cannot carry one relation's kwargs (a ``p`` for
    modular arithmetic) into builders that take ``n`` or ``nbit``, or selecting a relation fails
    on a TypeError nobody asked for."""
    from src.code.data import _RELATION_BUILDERS, init_dataset
    from src.code.train import resolve_config

    for name in _RELATION_BUILDERS:
        cfg = resolve_config(["setting=relational", f"relation={name}", "runner=linear"])
        ds = init_dataset(cfg)
        assert ds.n_items > 1 and ds.n_outcomes > 1, name


def _rows(T):
    return T.reshape(-1, T.shape[-1])


def test_graded_set_relations_spread_over_the_answer_and_name_the_empty_set():
    """A graded set relation answers "give me an element of the result", so the row is uniform over
    the result's members. The empty set is a real element of the lattice, so it gets its own outcome
    rather than being faked as a uniform row over everything."""
    from src.code.data import set_intersection, symmetric_difference
    nbit = 4
    for build, op in [(set_intersection, lambda a, b: a & b),
                      (symmetric_difference, lambda a, b: a ^ b)]:
        T = build(nbit=nbit, graded=True)
        assert T.shape == (2**nbit, 2**nbit, 2**nbit + 1)      # + the empty-set outcome
        R = _rows(T)
        assert np.allclose(R.sum(1), 1.0)
        for a in (0, 5, 15):
            for b in (0, 3, 15):
                s, row = op(a, b), T[a, b]
                members = [1 << i for i in range(nbit) if s >> i & 1]
                if not members:
                    assert row[-1] == 1.0, f"{a},{b}: empty answer must be the null outcome"
                else:
                    assert row[-1] == 0.0
                    assert np.allclose(row[members], 1 / len(members))
                    assert np.isclose(row.sum() - row[members].sum(), 0.0)
        # delta form is unchanged: one-hot on the result, no null outcome
        D = build(nbit=nbit, graded=False)
        assert D.shape == (2**nbit,) * 3
        assert D[5, 3, op(5, 3)] == 1.0 and D[5, 3].sum() == 1.0


def test_graded_entailment_spreads_over_the_common_ancestors():
    """Common ancestors of a and b are exactly the ancestors of their LCA, so the graded row covers
    that path and never needs a null outcome (the root is always common)."""
    from src.code.data import entailment
    T = entailment(depth=2, branch=3, graded=True)
    R = _rows(T)
    assert np.allclose(R.sum(1), 1.0)
    assert (R > 0).sum(1).min() >= 1                      # never empty
    # a node paired with itself: its own ancestor path, root included
    assert np.isclose(T[0, 0, 0], 1.0)                    # the root's only ancestor is itself
    deep = T.shape[0] - 1
    assert T[deep, deep].argmax() in range(T.shape[0])
    assert (T[deep, deep] > 0).sum() == 3                 # leaf, parent, root


def test_between_covers_the_interval_and_needs_both_operands():
    """`between` spreads over [min, max], so unlike `maximum` it cannot ignore the smaller input."""
    from src.code.data import between
    T = between(n=8)
    assert np.allclose(_rows(T).sum(1), 1.0)
    assert np.allclose(T[3, 5], [0, 0, 0, 1/3, 1/3, 1/3, 0, 0])
    assert np.allclose(T[5, 3], T[3, 5])                  # symmetric
    assert len(np.unique(_rows(T), axis=0)) == 36         # every unordered pair, vs max's 8


def test_common_divisors_is_the_divisors_of_the_gcd():
    from math import gcd
    from src.code.data import common_divisors
    n = 12
    T = common_divisors(n=n)
    assert np.allclose(_rows(T).sum(1), 1.0)
    for a, b in [(1, 1), (6, 9), (8, 12), (7, 11)]:
        g = gcd(a, b)
        want = [d for d in range(1, n + 1) if g % d == 0]
        assert np.allclose(np.flatnonzero(T[a - 1, b - 1]) + 1, want), (a, b)


def test_approximate_addition_is_graded_by_weber_noise():
    """Scalar variability: noise proportional to magnitude, i.e. Gaussian in log space. The outcome
    positions are the plain integers, so the log geometry comes from the noise law, not the layout."""
    from src.code.geometry import factor_logcond
    from src.code.data import approx_add
    m = 12
    T = approx_add(m=m)
    R = _rows(T)
    assert np.allclose(R.sum(1), 1.0)
    # the mode sits on the true sum
    for a, b in [(2, 3), (7, 7), (12, 12)]:
        assert T[a - 1, b - 1].argmax() + 2 == a + b        # outcomes are the sums 2..2m
    S = factor_logcond(np.log(np.maximum(R, 1e-12)))[3]
    assert int((S > 1e-9 * S[0]).sum()) == 2              # rank 2: linear in log(a+b)
    E = factor_logcond(np.log(np.maximum(R, 1e-12)))[1][:, :2]
    s = np.array([a + b for a in range(1, m + 1) for b in range(1, m + 1)], float)
    X = np.c_[np.log(s), np.ones(len(s))]
    r2 = 1 - ((E - X @ np.linalg.lstsq(X, E, rcond=None)[0])**2).sum() / ((E - E.mean(0))**2).sum()
    assert r2 > 0.999, f"code should be log(a+b), got R2={r2:.4f}"


def test_a_lapse_rate_removes_impossible_outcomes_without_moving_the_geometry():
    """A target with zeros is not identified by the loss: KL puts no weight on outcomes the task
    calls impossible, so their logits are free and E* asserts structure nothing constrains. Mixing
    in a lapse rate fixes that, and it must do so without changing the prediction being tested."""
    from src.code.geometry import predicted_geometry, log_conditional
    from src.code.eval import cka

    T = set_intersection(4)
    M = np.asarray(T, float).reshape(-1, T.shape[-1])
    E0 = predicted_geometry(log_conditional(M))

    lapsed = apply_lapse(M, 0.05)
    assert (lapsed > 0).all()                                  # no outcome is impossible any more
    assert np.allclose(lapsed.sum(1), 1.0)
    assert np.allclose(lapsed, 0.95 * M + 0.05 / M.shape[1])   # a uniform mixture, nothing else
    assert cka(predicted_geometry(np.log(lapsed)), E0) > 0.99  # same predicted geometry
