"""Budgeted CE-optimal codes (geometry.factor_features / als_features) and the feature-cube
setting: hypercube heat kernel + sparse item measure, log-conditional linear in the code."""
import itertools

import numpy as np
from omegaconf import OmegaConf

from src.code.config import Config
from src.code.data import FeatureCubeDataset, feature_cube
from src.code.geometry import (factor_features, als_features, expected_ce, principal_angles,
                               superposition_measure)


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


def test_als_descends_from_the_closed_form_init():
    Lf, Z = _setup(m=6, k=2, beta=4.0, decay=0.8)
    A0, D0, b0, _ = factor_features(Lf, Z, d=3)
    A, D, b, ce_path = als_features(Lf, Z, d=3, init=(A0, D0, b0))
    assert all(a >= b - 1e-7 for a, b in zip(ce_path, ce_path[1:]))
    ce0 = expected_ce(Z, Lf, A0, D0, b0)[0]
    assert expected_ce(Z, Lf, A, D, b)[0] <= ce0 + 1e-7


def test_superposition_at_high_beta_truncation_at_low_beta():
    # equal-importance one-hot features, m=6 into d=2: the packing-vs-truncation phase
    for beta, packed in [(0.3, False), (8.0, True)]:
        Lf, Z = _setup(m=6, k=1, beta=beta)
        A0, D0, b0, _ = factor_features(Lf, Z, d=2)
        A, D, b, _ = als_features(Lf, Z, d=2, init=(A0, D0, b0))
        _, bayes = expected_ce(Z, Lf, A0, D0, b0)
        ex_cf = expected_ce(Z, Lf, A0, D0, b0)[0] - bayes
        ex_als = expected_ce(Z, Lf, A, D, b)[0] - bayes
        if packed:
            assert ex_als < 0.6 * ex_cf          # ALS packs; truncation is far from optimal
        else:
            assert ex_als > 0.8 * ex_cf          # quadratic regime: closed form ~ optimal


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


def test_bias_free_closed_form_matches_the_class():
    # flat kernel: truncation is optimal within the bias-free class, so bias-free ALS
    # warm-started from the bias-free closed form should not beat it by much
    logP, pi, Z, Lf = feature_cube(np.array([0.4, 0.2, 0.1]), gamma=1.0)
    cf = factor_features(Lf, Z, d=2, w=pi, fit_bias=False)
    ce_cf, bayes = expected_ce(Z, Lf, *cf[:3], w=pi)
    A, D, b, _ = als_features(Lf, Z, 2, init=cf[:3], w=pi, fit_bias=False)
    assert np.allclose(b, cf[2])                                  # bias frozen at its init
    ce_als, _ = expected_ce(Z, Lf, A, D, b, w=pi)
    assert ce_als <= ce_cf + 1e-7                                 # ALS refines, never worse
    assert (ce_als - bayes) > 0.8 * (ce_cf - bayes)               # flat: closed form ~ optimal


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
