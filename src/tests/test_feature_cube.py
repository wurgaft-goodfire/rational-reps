"""Budgeted CE-optimal codes (geometry.factor_features / nuclear_features) and the feature-cube
setting: hypercube heat kernel + sparse item measure, log-conditional linear in the code."""
import itertools

import numpy as np
from omegaconf import OmegaConf

from src.code.config import Config
from src.code.data import FeatureCubeDataset, feature_cube
from src.code.geometry import (factor_features, expected_ce, nuclear_features,
                               principal_angles, superposition_measure)


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


def test_full_rank_closed_form_recovers_bayes():
    Lf, Z = _setup(m=6, k=2, beta=2.0, decay=0.8)
    A, D, b, S = factor_features(Lf, Z, d=6)
    ce, bayes = expected_ce(Z, Lf, A, D, b)
    assert np.isclose(ce, bayes, atol=1e-8)


def test_closed_form_ce_is_monotone_in_budget():
    Lf, Z = _setup(m=6, k=2, beta=2.0, decay=0.8)
    ces = []
    for d in range(1, 7):
        A, D, b, _ = factor_features(Lf, Z, d)
        ces.append(expected_ce(Z, Lf, A, D, b)[0])
    assert all(a >= b - 1e-9 for a, b in zip(ces, ces[1:]))


def test_superposition_at_high_beta_truncation_at_low_beta():
    # equal-importance one-hot features, m=6 into d=2: the packing-vs-truncation phase
    for beta, packed in [(0.3, False), (8.0, True)]:
        Lf, Z = _setup(m=6, k=1, beta=beta)
        A0, D0, b0, _ = factor_features(Lf, Z, d=2)
        A, D, b, _ = nuclear_features(Lf, Z, d=2)
        _, bayes = expected_ce(Z, Lf, A0, D0, b0)
        ex_cf = expected_ce(Z, Lf, A0, D0, b0)[0] - bayes
        ex_nuc = expected_ce(Z, Lf, A, D, b)[0] - bayes
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
        cf = factor_features(Lf, Z, d=2, w=pi, fit_bias=False)
        ce_cf, bayes = expected_ce(Z, Lf, *cf[:3], w=pi)
        A, D, b, S = nuclear_features(Lf, Z, d=2, w=pi, fit_bias=False)
        ce, _ = expected_ce(Z, Lf, A, D, b, w=pi)
        assert np.allclose(b, 0.0)
        assert ce <= ce_cf + 1e-4                          # never worse than truncation
        if packed:
            assert (ce - bayes) < 0.8 * (ce_cf - bayes)    # sharp: packing beats truncation
        else:
            assert (ce - bayes) > 0.8 * (ce_cf - bayes)    # flat: closed form ~ optimal


def test_additive_cube_excess_is_exact():
    # the per-feature softplus decomposition equals the exact expected CE of the same code
    from src.code.geometry import additive_cube_excess
    rng = np.random.default_rng(0)
    p = np.array([0.5, 0.35, 0.2, 0.14, 0.1])
    for gamma in (0.6, 0.03):
        logP, pi, Z, Lf = feature_cube(p, gamma)
        f = (1 - np.exp(-2 * gamma)) / 2
        w = 0.5 * np.log((1 - f) / f)
        A, B = rng.normal(size=(2, 5)), rng.normal(size=(2, 5))
        ce, bayes = expected_ce(Z, Lf, A, B @ Z.T, np.zeros(len(Z)), w=pi)
        assert np.isclose(additive_cube_excess(A.T @ B, w, p), ce - bayes, atol=1e-9)


def test_additive_class_is_rigid_at_sharp_gamma():
    # gaps are additive over flipped features, so the fully-additive class cannot manage
    # interference per outcome: its optimum barely improves on truncation, while the free
    # readout of nuclear_features buys most of the packing
    from src.code.geometry import additive_cube_excess, minimize_additive_excess
    p = 0.5 * 0.7 ** np.arange(6)
    logP, pi, Z, Lf = feature_cube(p, 0.03)
    f = (1 - np.exp(-2 * 0.03)) / 2
    w = 0.5 * np.log((1 - f) / f)
    cf = factor_features(Lf, Z, 2, w=pi, fit_bias=False)
    coef = np.linalg.lstsq(Z, cf[1].T, rcond=None)[0]             # cf readout, additive part
    K0 = cf[0].T @ coef.T
    ce_cf, bayes = expected_ce(Z, Lf, *cf[:3], w=pi)
    assert np.isclose(additive_cube_excess(K0, w, p), ce_cf - bayes, atol=1e-8)
    K, ex = minimize_additive_excess(K0, 2, w, p)
    assert np.isclose(additive_cube_excess(K, w, p), ex)
    assert ex < ce_cf - bayes - 0.05                              # some packing in the class
    nuc = nuclear_features(Lf, Z, 2, w=pi, fit_bias=False)
    ex_nuc = expected_ce(Z, Lf, *nuc[:3], w=pi)[0] - bayes
    assert ex_nuc < ex - 0.2                                      # free readout beats it all


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


def test_cube_full_rank_recovers_bayes():
    logP, pi, Z, Lf = feature_cube(np.array([0.4, 0.2, 0.1, 0.05]), gamma=0.3)
    A, D, b, _ = factor_features(Lf, Z, d=4, w=pi, fit_bias=False)
    assert np.allclose(b, 0.0)
    ce, bayes = expected_ce(Z, Lf, A, D, b, w=pi)
    assert ce - bayes < 1e-8


##### Trainable dataset #####

def _cube_config(sample_mode):
    cfg = OmegaConf.create({
        "setting": "feature_cube", "n_features": 4, "gamma": 0.25,
        "sparsity": 0.4, "p_decay": 0.7, "sample_mode": sample_mode, "random_seed": 0,
        "model_kwargs": {"model_type": "klinear"},
        "training_kwargs": {"batch_size": 64, "context_length": 3},
        "eval_kwargs": {"num_eval_batches": 2, "output_hidden_states": False},
    })
    return Config(cfg)


def test_cube_dataset_pointwise_samples_the_sparse_marginal():
    ds = FeatureCubeDataset(_cube_config("pointwise"))
    logP, pi, Z, Lf = feature_cube(0.4 * 0.7 ** np.arange(4), 0.25)
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
