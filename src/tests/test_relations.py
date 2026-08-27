"""Order-2 gallery: graded/semilattice operations factor cleanly into their geometry
(demo_graphs §4). Group operations are degenerate -> their circle is validated via a trained
model (see test_convergence)."""
import numpy as np

from src.code.data import (maximum, averaging, set_intersection, set_difference, entailment, apply_lapse,
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

def test_shared_feature_relations_are_an_exact_cube():
    """Sampling a feature from the *intersection* makes a 1-member answer a sharper distribution
    than a 3-member one, which stretches the cube and needs a null outcome to reach rank nbit.
    Sampling from all nbit features and reporting a yes/no flag gives nbit equal binary
    coordinates: rank nbit exactly, and edges in the cube's 1 : sqrt(2) : sqrt(3) ratios."""
    from src.code.geometry import predicted_geometry
    from src.code.data import apply_lapse

    for build, answer in [(set_intersection, lambda a, b: a & b),
                          (lambda nbit: set_difference(nbit, salience=None),
                           lambda a, b: a & ~b & 7)]:
        T = build(nbit=3)
        n = 8
        assert T.shape == (n, n, 6)                                  # 3 features x {yes, no}
        pts_all = predicted_geometry(np.log(apply_lapse(np.asarray(T, float).reshape(-1, 6), 0.05)))
        ans = np.array([answer(a, b) for a in range(n) for b in range(n)])
        pts = np.stack([pts_all[ans == c].mean(0) for c in np.unique(ans)])
        pts = pts - pts.mean(0)

        S = np.linalg.svd(pts, compute_uv=False)
        assert np.sum(S > 1e-6 * S[0]) == 3, f"rank {S}"
        D = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1))
        hd = np.array([[bin(x ^ y).count("1") for y in range(n)] for x in range(n)])
        edge = D[hd == 1].mean()
        assert D[hd == 1].std() < 1e-6 * edge                        # every edge the same length
        for h in (2, 3):
            assert abs(D[hd == h].mean() / edge - np.sqrt(h)) < 0.01, f"h={h}"


def test_set_difference_is_asymmetric_and_intersection_is_not():
    """Tversky's two terms: common features are symmetric, distinctive features of a are not."""
    I, D = set_intersection(nbit=3), set_difference(nbit=3)
    assert np.allclose(I, I.transpose(1, 0, 2))
    assert not np.allclose(D, D.transpose(1, 0, 2))


def test_entailment_defaults_to_a_binary_tree_of_depth_three():
    """Branching by 3 makes E*'s top two singular values exactly degenerate, so that plane is only
    pinned up to a rotation; branching by 2 removes the symmetry. Depth 3 also cuts the share of
    pairs whose only common ancestor is the root."""
    from src.code.geometry import predicted_geometry
    from src.code.data import apply_lapse

    T = entailment()
    n = T.shape[0]
    assert n == 15                                                   # 1 + 2 + 4 + 8
    M = apply_lapse(np.asarray(T, float).reshape(-1, n), 0.05)
    root_only = np.mean([np.count_nonzero(r > r.min()) == 1 for r in M])
    assert root_only < 0.6, f"{root_only:.2f} of pairs share only the root"

    E = np.asarray(predicted_geometry(np.log(M)), float)
    S = np.linalg.svd(E - E.mean(0), compute_uv=False)
    assert S[1] < 0.95 * S[0], f"top pair still degenerate: {S[:3] / S[0]}"


def test_lapse_leaves_a_target_that_already_has_full_support_alone():
    """The lapse exists to identify outcomes the loss cannot see -- ones with zero probability.
    A target that is already dense needs nothing, and mixing in uniform would clip its tails,
    which is exactly what raises the rank spuriously (approx_add: rank 1 -> 21)."""
    from src.code.data import approx_add, apply_lapse
    from src.code.geometry import predicted_geometry

    dense = np.asarray(approx_add(), float).reshape(-1, 23)
    assert (dense > 0).all()
    assert apply_lapse(dense, 0.05) is dense                     # untouched, not merely close

    E = predicted_geometry(np.log(apply_lapse(dense, 0.05)))
    S = np.linalg.svd(E - E.mean(0), compute_uv=False)
    assert np.sum(S > 1e-8 * S[0]) <= 4, f"dense target should stay low rank, got {S[:6]/S[0]}"

    sparse = np.asarray(set_difference(3), float).reshape(-1, 6)
    assert (sparse == 0).any() and (apply_lapse(sparse, 0.05) > 0).all()   # still applied here


def test_salience_weighted_difference_pins_every_axis():
    """Equal salience makes two cube axes exactly the same length, so that plane is only fixed up
    to a rotation. Unequal salience -- Tversky's weights -- gives distinct axes, so the prediction
    is a box with stated edge ratios rather than a cube up to gauge."""
    from src.code.geometry import predicted_geometry
    from src.code.data import apply_lapse

    T = set_difference(nbit=3)
    M = apply_lapse(np.asarray(T, float).reshape(-1, np.shape(T)[-1]), 0.05)
    E = predicted_geometry(np.log(M))
    S = np.linalg.svd(E - E.mean(0), compute_uv=False)[:3] / np.linalg.svd(E - E.mean(0), compute_uv=False)[0]
    assert S[1] < 0.98 and S[2] < 0.95 * S[1], f"axes still degenerate: {S}"
    assert not np.allclose(T, T.transpose(1, 0, 2))               # and still asymmetric


def test_feature_taxonomy_is_a_grid_not_a_cube():
    """Features in chains: the specific one entails the general, so items are the consistent sets
    and the shared-feature answer is a pair of chain levels -- a grid. Independent features would
    give a cube, whose axes come in a degenerate pair."""
    from src.code.geometry import predicted_geometry
    from src.code.data import apply_lapse, feature_taxonomy

    T = feature_taxonomy(chains=2, depth=2)
    assert T.shape == (9, 9, 8)                                   # 9 consistent items, 4 features x2
    E = predicted_geometry(np.log(apply_lapse(np.asarray(T, float).reshape(-1, 8), 0.05)))
    assert np.sum(np.linalg.svd(E - E.mean(0), compute_uv=False)
                  > 1e-6 * np.linalg.svd(E - E.mean(0), compute_uv=False)[0]) == 4   # one per feature

    # the grid is a claim about the answers, so read it off the answer classes: shared level per
    # chain. Over raw pairs the spectrum is reweighted by how many pairs share an answer.
    items = [(i, j) for i in range(3) for j in range(3)]
    cls = [tuple(min(x, y) for x, y in zip(A, B)) for A in items for B in items]
    keys = sorted(set(cls))
    pts = np.stack([E[[i for i, c in enumerate(cls) if c == k]].mean(0) for k in keys])
    pts = pts - pts.mean(0)
    U = np.linalg.svd(pts, full_matrices=False)[0][:, :2]
    S = np.linalg.svd(pts, compute_uv=False)
    assert S[1] < 0.95 * S[0], f"top pair degenerate like a cube: {S[:3] / S[0]}"
    for axis, v in enumerate(zip(*keys)):                         # top-2 plane is the two chains
        v = np.asarray(v, float) - np.mean(v)
        fit = U @ np.linalg.lstsq(U, v, rcond=None)[0]
        assert 1 - ((v - fit) ** 2).sum() / (v ** 2).sum() > 0.9, f"chain {axis} not in the plane"


def test_pitch_transpose_is_a_helix():
    """Pitch is height plus chroma: responses confuse neighbours (Gaussian in height) and notes an
    octave apart (von Mises in chroma). log p is then quadratic plus cosine in (c - s), so the code
    is one linear dimension plus a degenerate cos/sin pair -- a helix."""
    from src.code.geometry import predicted_geometry
    from src.code.data import pitch_transpose

    T = pitch_transpose()
    M = np.asarray(T, float).reshape(-1, np.shape(T)[-1])
    assert (M > 0).all()                                          # dense: the lapse must not touch it
    E = predicted_geometry(np.log(M))
    S = np.linalg.svd(E - E.mean(0), compute_uv=False)
    assert np.sum(S > 1e-6 * S[0]) == 3, f"not a helix: {S[:6] / S[0]}"
    assert abs(S[1] - S[2]) < 0.1 * S[1]                          # the circular pair is degenerate
    assert S[1] < 0.6 * S[0]                                      # height dominates chroma


def test_answer_classes_group_pairs_that_share_a_target():
    """Panels show one point per distinct answer, so each relation says which class a pair falls
    in. Indices follow the relation's own structure -- cube vertex, tree node -- so a figure can
    draw that structure over the code."""
    from src.code.data import answer_classes, set_difference, entailment

    cls, spec = answer_classes("set_difference")
    assert spec == {"structure": "hypercube", "n": 3}
    T = np.asarray(set_difference(3), float).reshape(-1, 6)
    for c in np.unique(cls):                                     # a class is one target row
        assert np.allclose(T[cls == c], T[cls == c][0])
    assert set(cls) == set(range(8))                             # indexed as cube vertices

    cls, spec = answer_classes("entailment")
    assert spec["structure"] == "tree" and len(spec["parents"]) == 15
    T = np.asarray(entailment(), float).reshape(-1, 15)
    for c in np.unique(cls):
        assert np.allclose(T[cls == c], T[cls == c][0])

    # relations with no structure to draw fall back to grouping identical target rows
    cls, spec = answer_classes("approx_add")
    assert spec is None and len(np.unique(cls)) == 23
