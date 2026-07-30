"""Spectral emergence of a latent noun class from noun-context cooccurrence.

The gender/noun-class DGP: nouns carry a hidden binary class that is invisible in the token and
revealed only through a pair of *marked* (agreement) contexts, seen with probability ``p`` per
token and biased by ``delta`` toward the class's preferred form. The class exists for a learner
exactly when the class direction of the correspondence-analysis residuals crosses the sampling
noise floor — a BBP-type transition at ``N* = sqrt(V*C) / (p * delta^2)`` — and crossing it is
what buys one-shot generalization: classifying a held-out noun from a single marked token.
"""
import numpy as np
import pytest

from src.code.emergence import (acquisition_N, bbp_threshold, ca_residuals, class_context_joint,
                                identification, identification_curve, mp_edge, one_shot_accuracy,
                                payoff_curve, per_noun_correct, permutation_null, sample_counts,
                                threshold_N)


def test_class_direction_and_one_shot_turn_on_with_data():
    """Integration: the same DGP, scarce vs abundant data. Below N* the top singular value sits in
    the permutation null band and the class direction is unidentified; well above N* the direction
    is found and supports one-shot classification of nouns from a single marked context."""
    Q, cls, _ = class_context_joint(400, 10, p=0.1, delta=0.8, zipf_a=0.0, seed=0)
    rng = np.random.default_rng(0)
    train, test = np.arange(200), np.arange(200, 400)

    M_small = sample_counts(Q, 200, rng)                       # N* ~ 1e3 here
    null = permutation_null(M_small, n_perm=20, rng=rng)
    top = np.linalg.svd(ca_residuals(M_small), compute_uv=False)[0]
    assert top < 1.1 * null.max()                              # undetectable: spectrum is null
    assert identification(M_small, cls) < 0.3                  # unidentified: direction is noise

    M_big = sample_counts(Q, 50_000, rng)
    assert identification(M_big, cls) > 0.9
    assert one_shot_accuracy(M_big[train], cls[train], cls[test], delta=0.8, n_marked=4, rng=rng) > 0.8
    # below threshold the paradigms' sign pattern is noise: one-shot is chance *on average* (any
    # single small corpus can land lucky on some pairs, so the claim is over fresh corpora)
    small_accs = [one_shot_accuracy(sample_counts(Q, 200, rng)[train], cls[train], cls[test],
                                    delta=0.8, n_marked=4, rng=rng) for _ in range(10)]
    assert 0.35 < np.mean(small_accs) < 0.65


def test_joint_margins_and_class_bias():
    Q, cls, f = class_context_joint(50, 6, p=0.2, delta=0.6, n_marked=3, zipf_a=1.2, seed=1)
    assert Q.shape == (50, 12) and cls.shape == (50,) and f.shape == (50,)
    assert Q.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(Q.sum(1), f)                    # noun marginal is the frequency
    np.testing.assert_allclose(Q[:, :6].sum(1), 0.2 * f)       # marked mass = p per token
    for k in range(3):                                         # class-0 prefers each pair's even form
        share = Q[:, 2 * k] / Q[:, 2 * k:2 * k + 2].sum(1)
        np.testing.assert_allclose(share, np.where(cls == 0, 0.8, 0.2))


def test_population_residuals_are_rank_one_with_the_bbp_signal_value():
    """On expected counts N*Q (uniform frequencies, balanced classes) the residual matrix is
    exactly the planted spike: one singular value sqrt(N) * delta * sqrt(p), rest zero."""
    N, p, delta = 10_000, 0.3, 0.5
    Q, _, _ = class_context_joint(64, 8, p=p, delta=delta, zipf_a=0.0, seed=0)
    S = np.linalg.svd(ca_residuals(N * Q), compute_uv=False)
    assert S[0] == pytest.approx(np.sqrt(N * p) * delta, rel=1e-6)
    assert S[1] < 1e-6 * S[0]


def test_zero_count_nouns_contribute_zero_residuals():
    M = np.array([[4, 0, 2], [0, 0, 0], [1, 3, 0]], dtype=float)
    R = ca_residuals(M)
    np.testing.assert_array_equal(R[1], 0.0)
    assert np.isfinite(R).all()


def test_permutation_null_matches_the_marchenko_pastur_edge():
    Q, _, _ = class_context_joint(200, 10, p=0.1, delta=0.0, seed=2)   # no signal at all
    rng = np.random.default_rng(2)
    M = sample_counts(Q, 100_000, rng)
    null = permutation_null(M, n_perm=10, rng=rng)
    edge = mp_edge(M.shape)
    assert 0.8 * edge < np.median(null) < 1.1 * edge
    top = np.linalg.svd(ca_residuals(M), compute_uv=False)[0]  # the delta=0 corpus is itself null
    assert top < 1.1 * edge


def test_threshold_scales_inversely_with_p_and_delta_squared():
    """The scaling law N* ~ 1/(p * delta^2): quadrupling p (or doubling delta) cuts the measured
    identification threshold by ~4x."""
    Ns = np.unique(np.geomspace(50, 40_000, 12).astype(int))

    def nstar(p, delta, seed):
        Q, cls, _ = class_context_joint(400, 10, p=p, delta=delta, zipf_a=0.0, seed=seed)
        curves = identification_curve(Q, cls, Ns, n_seeds=4, seed=seed)
        return threshold_N(Ns, curves, level=0.5)

    assert 2.0 < nstar(0.1, 0.8, 0) / nstar(0.4, 0.8, 0) < 8.0
    assert 2.0 < nstar(0.1, 0.4, 1) / nstar(0.1, 0.8, 1) < 8.0


def test_bbp_threshold_formula():
    assert bbp_threshold(400, 12, p=0.1, delta=0.8) == pytest.approx(np.sqrt(400 * 12) / (0.1 * 0.64))


def test_payoff_curve_runs_chance_to_ceiling():
    Q, cls, _ = class_context_joint(400, 10, p=0.1, delta=0.8, zipf_a=0.0, seed=0)
    Ns = np.array([100, 200_000])
    acc = payoff_curve(Q, cls, Ns, delta=0.8, n_marked=4, n_seeds=8, seed=0)
    assert acc.shape == (8, 2)
    assert 0.35 < acc[:, 0].mean() < 0.65                      # below N*: chance on average
    assert acc[:, 1].mean() > 0.85                             # far above N*: near the 0.9 ceiling


def test_acquisition_age_tracks_frequency():
    """Age of acquisition (first corpus size from which a noun stays correctly classified) is
    driven by the noun's own attestation before the abstraction emerges, so frequent nouns are
    acquired earlier than rare ones."""
    Q, cls, f = class_context_joint(400, 10, p=0.1, delta=0.8, zipf_a=1.2, seed=5)
    Ns = np.unique(np.geomspace(100, 1_000_000, 20).astype(int))
    aoa = acquisition_N(Q, cls, Ns, seed=5)
    assert np.isfinite(aoa).mean() > 0.95                      # nearly all acquired by 1M tokens
    order = np.argsort(f)
    rare, freq = order[:100], order[-100:]
    assert np.nanmedian(aoa[freq]) < np.nanmedian(aoa[rare])


def test_frequent_nouns_are_classified_before_rare_ones():
    """Mid-transition, per-noun correctness tracks noun frequency (pre-abstraction regime: each
    noun learned by brute attestation); with abundant data rare nouns come along for free."""
    Q, cls, f = class_context_joint(400, 10, p=0.1, delta=0.8, seed=3)
    rng = np.random.default_rng(3)
    order = np.argsort(f)
    rare, freq = order[:100], order[-100:]
    M_mid = sample_counts(Q, 2_000, rng)
    correct_mid = per_noun_correct(M_mid, cls)
    assert correct_mid[freq].mean() > correct_mid[rare].mean() + 0.1
    M_big = sample_counts(Q, 400_000, rng)
    assert per_noun_correct(M_big, cls).mean() > 0.95
