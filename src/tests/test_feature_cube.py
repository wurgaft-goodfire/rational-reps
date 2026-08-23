"""Budgeted CE-optimal codes (geometry.factor_logcond truncation / nuclear_features) and the
feature-cube setting: hypercube heat kernel + sparse item measure, log-conditional linear in
the code."""
import itertools

import numpy as np
from omegaconf import OmegaConf

from src.code.config import Config
from src.code.data import FeatureCubeDataset, feature_cube
from src.code.geometry import (factor_logcond, expected_ce, nuclear_features,
                               principal_angles, superposition_measure)


def truncation(L, d, w=None):
    """Rank-d prefix of the correspondence-analysis SVD: the free-row truncation oracle.
    Returns (A (d, N), D (d, n_x)) for the identity lift."""
    _, E, D, _, _ = factor_logcond(L, w=w)
    return E[:, :d].T, D[:d]


def all_supports(m, k):
    """All k-of-m supports as a binary (N, m) matrix — the fixed-k oracle test fixture."""
    combos = list(itertools.combinations(range(m), k))
    Z = np.zeros((len(combos), m))
    for r, c in enumerate(combos):
        Z[r, list(c)] = 1.0
    return Z


def _setup(m, k, beta, decay=1.0):
    Lf = beta * np.diag(decay ** np.arange(m))
    Z = all_supports(m, k)
    return Lf, Z


##### The budgeted-code oracles (generic, checked on fixed-k supports) #####

def test_surrogate_is_the_support_averaged_fisher_quadratic():
    # tr(R^T C R Fbar) == E_S delta_S^T Fbar delta_S, enumerated directly
    rng = np.random.default_rng(0)
    m, k, nx = 5, 2, 6
    Z = all_supports(m, k)
    R = rng.normal(size=(m, nx))
    q = rng.dirichlet(np.ones(nx))
    Fbar = np.diag(q) - np.outer(q, q)
    w = np.full(len(Z), 1.0 / len(Z))
    lhs = sum(wi * (z @ R) @ Fbar @ (z @ R) for wi, z in zip(w, Z))
    C = (Z * w[:, None]).T @ Z
    rhs = np.trace(R.T @ C @ R @ Fbar)
    assert np.isclose(lhs, rhs)


def test_full_rank_truncation_recovers_bayes():
    Lf, Z = _setup(m=6, k=2, beta=2.0, decay=0.8)
    L = Z @ Lf
    I = np.eye(len(Z))
    A, D = truncation(L, d=len(Z))
    ce, bayes = expected_ce(I, L, A, D, np.zeros(L.shape[1]))
    assert np.isclose(ce, bayes, atol=1e-8)


def test_truncation_ce_is_monotone_in_budget():
    Lf, Z = _setup(m=6, k=2, beta=2.0, decay=0.8)
    L = Z @ Lf
    I = np.eye(len(Z))
    ces = []
    for d in range(1, 7):
        A, D = truncation(L, d)
        ces.append(expected_ce(I, L, A, D, np.zeros(L.shape[1]))[0])
    assert all(a >= b - 1e-9 for a, b in zip(ces, ces[1:]))


def test_superposition_at_high_beta_truncation_at_low_beta():
    # equal-importance one-hot features, m=6 into d=2: the packing-vs-truncation phase
    for beta, packed in [(0.3, False), (8.0, True)]:
        Lf, Z = _setup(m=6, k=1, beta=beta)
        L = Z @ Lf
        I = np.eye(len(Z))
        A0, D0 = truncation(L, d=2)
        A, D, b, _ = nuclear_features(L, I, d=2, fit_bias=False)
        _, bayes = expected_ce(I, L, A0, D0, np.zeros(L.shape[1]))
        ex_cf = expected_ce(I, L, A0, D0, np.zeros(L.shape[1]))[0] - bayes
        ex_nuc = expected_ce(I, L, A, D, b)[0] - bayes
        if packed:
            assert ex_nuc < 0.6 * ex_cf          # packing beats truncation
        else:
            assert ex_nuc > 0.8 * ex_cf          # quadratic regime: closed form ~ optimal


def test_prox_nuclear_soft_thresholds_the_spectrum():
    from src.code.geometry import _prox_nuclear
    rng = np.random.default_rng(0)
    R = rng.normal(size=(4, 7))
    U, S, Vt = np.linalg.svd(R, full_matrices=False)
    t = S[2]                                                  # kills modes 3-4, shrinks 1-2
    P = _prox_nuclear(R, t)
    assert np.allclose(np.linalg.svd(P, compute_uv=False), np.maximum(S - t, 0.0))
    assert np.allclose(P, (U * np.maximum(S - t, 0.0)) @ Vt)
    assert np.allclose(_prox_nuclear(R, 0.0), R)


def test_nuclear_features_full_rank_recovers_bayes_with_balanced_factors():
    logP, pi, Z, Lf = feature_cube(np.array([0.4, 0.2, 0.1]), gamma=0.3)
    A, D, b, S = nuclear_features(Lf, Z, d=3, w=pi, fit_bias=False)
    ce, bayes = expected_ce(Z, Lf, A, D, b, w=pi)
    assert ce - bayes < 1e-5
    assert np.allclose(A @ A.T, D @ D.T, atol=1e-8)           # plain balanced split


def test_nuclear_features_flat_matches_closed_form_and_sharp_packs():
    # the convex program reduces to the whitened SVD in the flat regime and packs at sharp
    for gamma, packed in [(0.6, False), (0.03, True)]:
        logP, pi, Z, Lf = feature_cube(0.5 * 0.7 ** np.arange(6), gamma)
        I = np.eye(len(Z))
        A0, D0 = truncation(logP, d=2, w=pi)
        ce_cf, bayes = expected_ce(I, logP, A0, D0, np.zeros(logP.shape[1]), w=pi)
        A, D, b, S = nuclear_features(logP, I, d=2, w=pi, fit_bias=False)
        ce, _ = expected_ce(I, logP, A, D, b, w=pi)
        assert np.allclose(b, 0.0)
        assert ce <= ce_cf + 1e-4                          # never worse than truncation
        if packed:
            assert (ce - bayes) < 0.8 * (ce_cf - bayes)    # sharp: packing beats truncation
        else:
            assert (ce - bayes) > 0.8 * (ce_cf - bayes)    # flat: closed form ~ optimal


def test_nuclear_features_escapes_the_truncation_basin():
    # dense cell of the importance-ladder cube: every lambda-path truncation proposes the
    # truncation basin (excess 1.64); the small-init candidate must find the packed one
    logP, pi, Z, Lf = feature_cube(np.full(6, 0.5), 0.03, w_decay=0.9)
    I = np.eye(len(Z))
    A, D, b, S = nuclear_features(logP, I, 2, w=pi, fit_bias=False)
    bayes = -(pi[:, None] * np.exp(logP) * logP).sum()
    ce, _ = expected_ce(I, logP, A, D, b, w=pi)
    assert ce - bayes < 1.45


def test_superposition_measure():
    A = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.0]])              # feature 3 overlaps feature 1
    s = superposition_measure(A)
    assert np.allclose(s[[0, 2]], [0.25, 1.0]) and s[1] == 0.0    # (a_hat_i . a_j)^2 summed


def test_principal_angles():
    rng = np.random.default_rng(1)
    M = rng.normal(size=(3, 8))
    Q = rng.normal(size=(3, 3))
    assert np.allclose(principal_angles(M, Q @ M), 0.0, atol=1e-6)
    a = np.zeros((1, 4)); a[0, 0] = 1.0
    c = np.zeros((1, 4)); c[0, 2] = 1.0
    assert np.isclose(principal_angles(a, c)[0], np.pi / 2)
    # rank-deficient rows must not inflate the subspace: zero-padding changes nothing
    a_pad = np.vstack([a, np.zeros((3, 4))])
    c_pad = np.vstack([c, np.zeros((3, 4))])
    assert np.isclose(principal_angles(a_pad, c_pad)[0], np.pi / 2)


##### Feature cube (hypercube heat kernel + sparse item measure) #####

def test_feature_cube_conditional_structure():
    p, gamma = np.array([0.35, 0.245, 0.17]), 0.25
    logP, pi, Z, Lf = feature_cube(p, gamma)
    P = np.exp(logP)
    assert P.shape == (8, 8) and Z.shape == (8, 3) and Lf.shape == (3, 8)
    assert np.allclose(P.sum(1), 1.0)
    assert np.allclose(P, P.T)                                    # symmetric heat kernel
    assert np.isclose(pi.sum(), 1.0)
    on = Z > 0
    assert np.allclose(pi, (p ** on * (1 - p) ** ~on).prod(1))    # product-Bernoulli measure
    R = logP - Z @ Lf                                             # purely bilinear + one constant
    assert np.allclose(R, R[0, 0])
    assert np.allclose(Z * Z, 1.0)                                # +-1 centered codes
    f = (1 - np.exp(-2 * gamma)) / 2                              # per-feature flip probability
    w = 0.5 * np.log((1 - f) / f)
    assert np.allclose(Lf, w * Z.T)                               # coupling set by the heat time


def test_feature_cube_importance_ladder():
    # per-feature couplings w_i = w0 * w_decay^i decouple importance from frequency:
    # still a symmetric, purely bilinear kernel; the measure only sees p
    p, gamma, wd = np.full(3, 0.3), 0.1, 0.8
    logP, pi, Z, Lf = feature_cube(p, gamma, w_decay=wd)
    P = np.exp(logP)
    assert np.allclose(P.sum(1), 1.0)
    assert np.allclose(P, P.T)
    w = 0.5 * np.log(1 / np.tanh(gamma)) * wd ** np.arange(3)
    assert np.allclose(Lf, w[:, None] * Z.T)
    R = logP - Z @ Lf
    assert np.allclose(R, R[0, 0])                                # bilinear + one constant
    assert np.allclose(pi, feature_cube(p, gamma)[1])             # measure independent of w


def test_cube_full_rank_recovers_bayes():
    logP, pi, Z, Lf = feature_cube(np.array([0.4, 0.2, 0.1, 0.05]), gamma=0.3)
    I = np.eye(len(Z))
    A, D = truncation(logP, d=len(Z), w=pi)
    ce, bayes = expected_ce(I, logP, A, D, np.zeros(logP.shape[1]), w=pi)
    assert ce - bayes < 1e-8


##### Trainable dataset #####

def _cube_config(sample_mode):
    cfg = OmegaConf.create({
        "setting": "feature_cube", "n_features": 4, "gamma": 0.25,
        "sparsity": 0.4, "p_decay": 0.7, "w_decay": 0.8, "sample_mode": sample_mode,
        "random_seed": 0,
        "model_kwargs": {"model_type": "klinear"},
        "training_kwargs": {"batch_size": 64, "context_length": 3},
        "eval_kwargs": {"num_eval_batches": 2, "output_hidden_states": False},
    })
    return Config(cfg)


def test_cube_dataset_pointwise_samples_the_sparse_marginal():
    ds = FeatureCubeDataset(_cube_config("pointwise"))
    logP, pi, Z, Lf = feature_cube(0.4 * 0.7 ** np.arange(4), 0.25, w_decay=0.8)
    assert np.allclose(ds.pi, pi) and np.allclose(ds.Lf, Lf) and np.allclose(ds.Z, Z)
    assert ds.num_dims == 16 and ds.vocab_size == 16 and ds.arity == 1
    b = ds[0]
    assert b["input_ids"].shape == (64, 1)
    items = b["input_ids"][:, 0].cpu().numpy()
    assert np.allclose(b["labels"].cpu().numpy(), np.exp(logP)[items], atol=1e-6)
    counts = np.zeros(16)
    for _ in range(50):
        idx = ds[0]["input_ids"][:, 0].cpu().numpy()
        np.add.at(counts, idx, 1)
    assert abs(counts[0] / counts.sum() - pi[0]) < 0.05             # all-off item ~ pi[0]


def test_cube_dataset_episodes_are_scored_exactly():
    # short episodes from the sparse marginal (the symmetric kernel's walk drifts dense,
    # so episodes are kept short and resampled rather than run as long stationary chains)
    ds = FeatureCubeDataset(_cube_config("sequence"))
    b = ds[0]
    ids = b["input_ids"]
    el = ds.optimal_predictions(ids)
    assert ids.shape == (64, 4) and el.shape == (64, 3, 16)
    assert (ids[:, 0] == ds.start_token).all() and (ids[:, 1:] < 16).all()
    P = np.exp(ds.conditional)
    assert np.allclose(el[:, 0].cpu().numpy(), ds.pi, atol=1e-6)    # after start: the marginal
    for t in range(1, 3):                                           # after item x: P[x]
        assert np.allclose(el[:, t].cpu().numpy(), P[ids[:, t].cpu().numpy()], atol=1e-6)
