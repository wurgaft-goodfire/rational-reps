"""Word -> attribute family: words are hypercube feature bundles, outcomes are the m
attribute tokens plus a null ("nothing said") reference pinning the softmax gauge."""
import numpy as np
from omegaconf import OmegaConf

from src.code.config import Config
from src.code.data import DistributionFamilyDataset, word_attribute
from src.code.geometry import expected_ce, nuclear_features
from src.tests.test_feature_cube import truncation


def test_word_attribute_kernel_structure():
    m, gamma, r_a, db = 4, 0.1, 0.8, 0.7
    logP, info = word_attribute(m, gamma, alpha_decay=r_a, beta_step=db)
    Z, alpha, beta = info["Z"], info["alpha"], info["beta"]
    N = 2 ** m
    assert logP.shape == (N, m + 1) and Z.shape == (N, m)
    assert np.allclose(np.exp(logP).sum(1), 1.0)
    assert np.allclose(Z * Z, 1.0)                                # +-1 feature codes
    assert np.allclose(alpha, 0.5 * np.log(1 / np.tanh(gamma)) * r_a ** np.arange(m))
    assert np.allclose(beta, -db * np.arange(m))
    # gauge-free identity: log p(i|x)/p(null|x) = alpha_i x_i + beta_i, exactly
    G = logP[:, :m] - logP[:, m:]
    assert np.allclose(G, Z * alpha + beta)
    # the pinned-gauge logit matrix is rank m: feature directions with beta welded in
    assert np.linalg.matrix_rank(G) == m


def test_word_attribute_odds_swings():
    m = 3
    logP, info = word_attribute(m, gamma=0.3, alpha_decay=0.9, beta_step=0.5)
    Z, alpha = info["Z"], info["alpha"]
    G = logP[:, :m] - logP[:, m:]
    for i in range(m):
        x_has = np.flatnonzero(Z[:, i] > 0)[0]
        x_lacks = np.flatnonzero([(Z[j, i] < 0) and (np.delete(Z[j], i) == np.delete(Z[x_has], i)).all()
                                  for j in range(len(Z))])[0]
        assert np.isclose(G[x_has, i] - G[x_lacks, i], 2 * alpha[i])   # importance = swing
        j = (i + 1) % m
        assert np.isclose(G[x_has, j], G[x_lacks, j])                  # other attributes untouched


def test_word_attribute_full_rank_is_m_and_budget_binds_below():
    m = 4
    logP, _ = word_attribute(m, gamma=0.25, alpha_decay=0.9, beta_step=0.5)
    I = np.eye(len(logP))
    A, D = truncation(logP, d=m)
    ce, bayes = expected_ce(I, logP, A, D, np.zeros(logP.shape[1]))
    assert ce - bayes < 1e-8                                       # d = m recovers Bayes
    A, D = truncation(logP, d=m - 1)
    ce, _ = expected_ce(I, logP, A, D, np.zeros(logP.shape[1]))
    assert ce - bayes > 1e-3                                       # d = m - 1 does not


def test_word_attribute_nuclear_beats_truncation_under_budget():
    logP, _ = word_attribute(4, gamma=0.03, alpha_decay=0.9, beta_step=0.5)
    I = np.eye(len(logP))
    A0, D0 = truncation(logP, d=2)
    ce_cf, bayes = expected_ce(I, logP, A0, D0, np.zeros(logP.shape[1]))
    A, D, b, _ = nuclear_features(logP, I, d=2, fit_bias=False)
    ce, _ = expected_ce(I, logP, A, D, b)
    assert np.allclose(b, 0.0)
    assert ce <= ce_cf + 1e-4


def _wa_config():
    cfg = OmegaConf.create({
        "setting": "distribution-family", "family": "word_attribute",
        "family_kwargs": {"m": 4, "gamma": 0.25, "alpha_decay": 0.8, "beta_step": 0.5},
        "random_seed": 0,
        "model_kwargs": {"model_type": "klinear"},
        "training_kwargs": {"batch_size": 16},
        "eval_kwargs": {"num_eval_batches": 1, "output_hidden_states": False},
    })
    return Config(cfg)


def test_word_attribute_dataset_exposes_exact_targets():
    ds = DistributionFamilyDataset(_wa_config())
    logP, _ = word_attribute(4, 0.25, 0.8, 0.5)
    assert ds.vocab_size == 16 and ds.num_dims == 5 and ds.arity == 1
    assert np.allclose(ds.conditional, logP)
    b = ds[0]
    assert b["input_ids"].shape == (16, 1)
    assert (b["input_ids"][:, 0].cpu().numpy() == np.arange(16)).all()
    assert np.allclose(b["labels"].cpu().numpy(), np.exp(logP), atol=1e-6)


def test_attribute_stats_recover_the_ladders():
    from src.code.geometry import attribute_stats
    logP, info = word_attribute(5, gamma=0.1, alpha_decay=0.85, beta_step=0.6)
    a_hat, b_hat = attribute_stats(logP, info["Z"])
    assert np.allclose(a_hat, info["alpha"])
    assert np.allclose(b_hat, info["beta"])


def test_feature_vectors_recover_additive_fit():
    from src.code.geometry import feature_vectors
    rng = np.random.default_rng(0)
    Z = 2.0 * (rng.random((32, 5)) > 0.5) - 1.0
    W, c = rng.normal(size=(5, 3)), rng.normal(size=3)
    V, v0 = feature_vectors(Z @ W + c, Z)
    assert np.allclose(V, W) and np.allclose(v0, c)


def test_sparse_words_lexicon_and_kernel():
    from src.code.data import sparse_words
    logP, info = sparse_words(n_words=32, m=6, p_active=0.25, gamma=0.1,
                              alpha_decay=0.8, seed=3)
    X, alpha = info["Z"], info["alpha"]
    assert X.shape == (32, 6) and logP.shape == (32, 7)
    assert set(np.unique(X)) <= {0.0, 1.0}                       # sparse 0/1 bundles
    assert 0.05 < X.mean() < 0.5                                 # Bernoulli(p) activity
    assert np.allclose(np.exp(logP).sum(1), 1.0)
    G = logP[:, :-1] - logP[:, -1:]                              # pin gauge at null
    assert np.allclose(G, X * alpha)                             # kernel = columns scaled by alpha
    L2, info2 = sparse_words(n_words=32, m=6, p_active=0.25, gamma=0.1,
                             alpha_decay=0.8, seed=3)
    assert np.allclose(L2, logP)                                 # lexicon fixed by seed
