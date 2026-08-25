"""Unit tests for the predicted-geometry oracle (geometry.py).

The closed-form prediction must recover the structure's harmonics: a ring's co-occurrence
must factor into a clean circle, and a modular-addition tensor likewise.
"""
import numpy as np

from src.code.geometry import factor_logcond, factor_tensor, predicted_geometry
from src.code.data import ring, cooccur_logcond


def _radius_cv(emb2):
    r = np.linalg.norm(emb2, axis=1)
    return r.std() / r.mean()


def test_ring_factorizes_into_a_circle():
    L = cooccur_logcond(ring(24), gamma=0.8)
    Lc, E, D, S = factor_logcond(L)
    # translation symmetry -> degenerate top Fourier pair, then a spectral gap
    assert np.isclose(S[0], S[1], rtol=0.02), f"top pair not degenerate: {S[:4]}"
    assert S[2] < 0.9 * S[1], f"no gap after the top pair: {S[:4]}"
    # the top-2 code lies on a circle: near-constant radius
    assert _radius_cv(E[:, :2]) < 0.05
    # walking the ring index sweeps the angle monotonically (a clean loop)
    ang = np.unwrap(np.sort(np.arctan2(E[:, 1], E[:, 0])))
    assert np.all(np.diff(ang) > 0)


def test_max_tensor_factorizes_into_an_ordered_line():
    # max is a non-group (semilattice) operation -> a clean, non-degenerate 1-D order.
    # (Abelian group indicators like modular addition are degenerate under factor_tensor;
    #  their circle is the analytic single-frequency member, handled in the relational slice.)
    n = 10
    T = np.zeros((n, n, n))                        # T[a, b, c] = 1[max(a, b) == c]
    for a in range(n):
        for b in range(n):
            T[a, b, max(a, b)] = 1.0
    E, S = factor_tensor(T)
    order = np.abs(np.corrcoef(E[:, 0], np.arange(n))[0, 1])
    assert order > 0.95, f"max should embed on an ordered line, got |corr|={order:.3f}"


def test_predicted_geometry_dispatches_on_rank():
    L = cooccur_logcond(ring(8))
    assert predicted_geometry(L).shape[0] == 8            # 2-D conditional -> factor_logcond
    T = np.zeros((5, 5, 5))
    for a in range(5):
        for b in range(5):
            T[a, b, (a + b) % 5] = 1.0
    assert predicted_geometry(T).shape[0] == 5            # 3-D conditional -> factor_tensor


def test_factor_logcond_is_the_plain_svd_of_the_row_centered_log_conditional():
    # no measure weighting: Lc is L minus its plain row mean, and (E, D) is its SVD with the
    # scale split balanced, so each factor carries sqrt(S).
    L = cooccur_logcond(ring(12), gamma=0.8)
    Lc, E, D, S = factor_logcond(L)
    assert np.allclose(Lc, L - L.mean(1, keepdims=True))
    assert np.allclose(S, np.linalg.svd(Lc, compute_uv=False))
    assert np.allclose(E @ D, Lc)
    # balanced split: the two factors carry equal scale in every direction
    assert np.allclose(np.linalg.norm(E, axis=0), np.linalg.norm(D, axis=1))


def test_capacity_allocation_splits_the_budget_by_squared_decoder_norm():
    # analytic case: balanced decoder rows [[1,-1,0,0],[0,0,2,-2]] (already row-centred, so the
    # softmax gauge is a no-op). Column norms 1,1,2,2 -> squares 1,1,4,4 -> s = 2*[1,1,4,4]/10.
    from src.code.geometry import capacity_allocation
    Db = np.array([[1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 2.0, -2.0]])
    A = np.diag(np.sqrt(np.diag(Db @ Db.T)))                    # balanced: A A^T = D D^T
    s = capacity_allocation(A, Db, 2)
    assert np.isclose(s.sum(), 2.0)                             # the budget is the code width
    assert np.allclose(s, [0.2, 0.2, 0.8, 0.8])


def test_predicted_allocation_is_symmetric_for_two_features():
    # m=2: the centring projector H = v v^T with v = [1,-1]/sqrt(2) makes the second-moment
    # matrix exactly rank one along v, for any alpha. So the single dimension is shared evenly.
    from src.code.geometry import predicted_allocation
    for alpha in ([3.0, 1.0], [1.0, 1.0], [8.0, 0.5]):
        s = predicted_allocation(np.array(alpha), p=0.3, d=1)
        assert np.isclose(s.sum(), 1.0)
        assert np.allclose(s, [0.5, 0.5])
