"""Unit tests for the predicted-geometry oracle (geometry.py).

The closed-form prediction must recover the structure's harmonics: a ring's co-occurrence
must factor into a clean circle, and a modular-addition tensor likewise.
"""
import numpy as np
import pytest

from src.code.geometry import factor_logcond, log_conditional, predicted_geometry
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
    # (Abelian group indicators like modular addition are degenerate; their circle is the
    #  analytic single-frequency member, handled in the relational slice.)
    n = 10
    T = np.zeros((n, n, n))                        # T[a, b, c] = 1[max(a, b) == c]
    for a in range(n):
        for b in range(n):
            T[a, b, max(a, b)] = 1.0
    E = predicted_geometry(log_conditional(T).reshape(n, -1))
    order = np.abs(np.corrcoef(E[:, 0], np.arange(n))[0, 1])
    assert order > 0.95, f"max should embed on an ordered line, got |corr|={order:.3f}"


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


def test_canonical_gauge_recovers_the_dft_on_modular_addition():
    """Within a degenerate singular block the SVD basis is an arbitrary rotation. Rotating onto a
    target basis (here the DFT modes of Z_p) recovers structured factors without changing E @ D."""
    from src.code.data import relation_tensor
    from src.code.geometry import canonical_gauge, factor_logcond, fourier_basis

    p = 12
    M = relation_tensor({"relation": "modular_add", "p": p}).reshape(p * p, p)
    Lc, E, D, S = factor_logcond(M)
    Eg, Dg = canonical_gauge(E, D, fourier_basis(p))

    assert np.allclose(Eg @ Dg, E @ D, atol=1e-9)              # same factorization
    F = fourier_basis(p)
    for row in Dg[:p - 1]:                                     # every gauged row is one DFT mode
        corr = np.abs(F @ row) / (np.linalg.norm(F, axis=1) * np.linalg.norm(row) + 1e-12)
        assert corr.max() > 0.99
    # the item code inherits the same modes: the frequency-1 pair is cos/sin of s = (a + b) mod p
    s = (np.arange(p * p) // p + np.arange(p * p) % p) % p
    target = np.c_[np.cos(2 * np.pi * s / p), np.sin(2 * np.pi * s / p)]
    fit = Eg[:, :2] @ np.linalg.lstsq(Eg[:, :2], target, rcond=None)[0]
    assert np.corrcoef(fit.ravel(), target.ravel())[0, 1] > 0.99


def test_canonical_gauge_leaves_a_nondegenerate_factorization_alone():
    """Distinct singular values already pin the basis: the gauge must be (nearly) the identity."""
    from src.code.geometry import canonical_gauge, factor_logcond

    rng = np.random.default_rng(0)
    L = rng.normal(size=(20, 6)) @ np.diag([5.0, 3.0, 2.0, 1.0, 0.5, 0.2])
    _, E, D, _ = factor_logcond(L)
    Eg, Dg = canonical_gauge(E, D, rng.normal(size=(6, 6)))
    assert np.allclose(np.abs(Dg), np.abs(D), atol=1e-8)       # up to sign, unchanged


def _circle_code(p, k, d=32, seed=0, squash=1.0):
    """Codes for all (a, b) pairs carrying exactly frequency k of s = (a+b) mod p, lifted into d
    dimensions by a random rotation. ``squash`` shrinks the sine axis to a known circularity."""
    s = (np.arange(p * p) // p + np.arange(p * p) % p) % p
    ang = 2 * np.pi * k * s / p
    xy = np.c_[np.cos(ang), squash * np.sin(ang)]
    Q = np.linalg.qr(np.random.default_rng(seed).normal(size=(d, 2)))[0]
    return xy @ Q.T, s


def test_group_fraction_is_one_on_a_pure_group_code_and_chance_when_labels_are_shuffled():
    """Exact recovery plus the shuffle control: a code that is a function of s alone puts all of
    its variance on s; permuting the labels destroys the signal and leaves about 1/p."""
    from src.code.geometry import group_fraction

    p = 11
    X, s = _circle_code(p, k=1)
    assert group_fraction(X, s) == pytest.approx(1.0, abs=1e-9)
    shuffled = np.random.default_rng(0).permutation(s)
    assert group_fraction(X, shuffled) == pytest.approx(1 / p, abs=0.03)


def test_frequency_plane_puts_all_energy_on_the_frequency_that_is_present():
    """A code built from frequency 3 alone reports energy 1 at k=3 and 0 elsewhere."""
    from src.code.geometry import frequency_plane

    p = 11
    X, s = _circle_code(p, k=3)
    energy = {k: frequency_plane(X, s, k)[1] for k in range(1, p // 2 + 1)}
    assert energy[3] == pytest.approx(1.0, abs=1e-9)
    assert max(v for k, v in energy.items() if k != 3) < 1e-9


def test_circularity_reports_a_known_axis_squash():
    """Circularity is sigma_2 / sigma_1 of the fitted plane, so squashing the sine axis by a
    known factor must read back that factor."""
    from src.code.geometry import frequency_plane

    X, s = _circle_code(11, k=1, squash=0.4)
    assert frequency_plane(X, s, 1)[2] == pytest.approx(0.4, abs=1e-6)


def test_circulant_fraction_ignores_radii_but_catches_a_broken_symmetry():
    """A ring is a Gram that depends only on s - s'. Unequal radii across frequencies keep that
    property (the rnn's case), so they must not be penalised; displacing one class must be."""
    from src.code.geometry import circulant_fraction

    p, s = 11, (np.arange(11 * 11) // 11 + np.arange(11 * 11) % 11) % 11
    ang = 2 * np.pi * np.arange(p) / p
    radii = [1.0, 0.3, 0.9, 0.1, 0.5]                          # deliberately uneven per frequency
    means = np.concatenate([r * np.c_[np.cos(k * ang), np.sin(k * ang)]
                            for k, r in zip(range(1, p // 2 + 1), radii)], axis=1)
    assert circulant_fraction(means[s], s) == pytest.approx(1.0, abs=1e-9)
    moved = means.copy()
    moved[3] *= 2.5                                            # one class pushed off the ring
    assert circulant_fraction(moved[s], s) < 0.8


def test_log_conditional_clamps_zeros_and_leaves_probabilities_alone():
    """Softmax reads logits, so every prediction factors log p, never p. Zero-probability outcomes
    would be -inf, so the log is clamped at a single floor shared by the whole codebase."""
    from src.code.geometry import LOG_FLOOR, log_conditional

    P = np.array([[0.5, 0.5, 0.0], [1.0, 0.0, 0.0]])
    L = log_conditional(P)
    assert np.allclose(L[0, :2], np.log(0.5))
    assert (L[[0, 1], [2, 1]] == LOG_FLOOR).all()          # exact zeros hit the floor
    assert L.min() >= LOG_FLOOR                            # and nothing sits below it


def test_log_conditional_floors_impossible_outcomes_not_merely_unlikely_ones():
    """LOG_FLOOR stands in for log 0, which has no value. Clipping *small but real* probabilities
    to it flattens a distribution's tails onto a constant and invents structure: pitch_transpose's
    Gaussian reaches exp(-32) at the far end, and clipping it takes the code from rank 3 to 19."""
    from src.code.data import pitch_transpose
    from src.code.geometry import LOG_FLOOR, log_conditional, predicted_geometry

    P = np.asarray(pitch_transpose(), float).reshape(-1, 25)
    assert (P > 0).all() and P.min() < np.exp(LOG_FLOOR)      # dense, but with a very thin tail

    L = log_conditional(P)
    assert np.allclose(L, np.log(P))                          # nothing is impossible, nothing floored

    E = predicted_geometry(L)
    S = np.linalg.svd(E - E.mean(0), compute_uv=False)
    assert np.sum(S > 1e-6 * S[0]) == 3, f"tail clipping inflated the rank: {S[:6] / S[0]}"

    # a genuine zero still gets the floor
    Z = np.array([[0.5, 0.5, 0.0]])
    assert log_conditional(Z)[0, 2] == LOG_FLOOR
