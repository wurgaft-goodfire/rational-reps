"""Word -> attribute family: N words with i.i.d. Bernoulli(p) features in {0, 1},
outcomes the m attribute tokens, softmax logits alpha_i Z_xi with alpha_i = scale * decay^i."""
import numpy as np
from omegaconf import OmegaConf

from src.code.config import Config
from src.code.data import CooccurrenceDataset, word_attribute
from src.code.geometry import (attribute_stats, expected_ce, feature_vectors,
                               nuclear_features, truncation_features as truncation)


def test_word_attribute_kernel_structure():
    N, m, p, decay, scale = 64, 5, 0.25, 0.8, 1.5
    logP, info = word_attribute(N, m, p, decay, scale, seed=3)
    Z, alpha = info["Z"], info["alpha"]
    assert logP.shape == (N, m) and Z.shape == (N, m)
    assert np.allclose(np.exp(logP).sum(1), 1.0)
    assert set(np.unique(Z)) <= {0.0, 1.0}                        # sparse 0/1 bundles
    assert 0.05 < Z.mean() < 0.5                                  # Bernoulli(p) activity
    assert np.allclose(alpha, scale * decay ** np.arange(m))
    # gauge-free identity: within-row logit contrasts equal alpha_i z_i - alpha_j z_j
    G = logP - logP[:, :1]
    assert np.allclose(G, Z * alpha - (Z * alpha)[:, :1])
    L2, _ = word_attribute(N, m, p, decay, scale, seed=3)
    assert np.allclose(L2, logP)                                  # lexicon fixed by seed


def test_word_attribute_effective_rank_is_m_minus_1():
    # the softmax gauge absorbs one dimension: d = m - 1 recovers Bayes, d = m - 2 does not
    logP, info = word_attribute(N=128, m=5, p=0.4, decay=0.9, scale=1.75, seed=0)
    I = np.eye(len(logP))
    m = logP.shape[1]
    A, D = truncation(logP, d=m - 1)
    ce, bayes = expected_ce(I, logP, A, D, np.zeros(m))
    assert ce - bayes < 1e-8
    A, D = truncation(logP, d=m - 2)
    ce, _ = expected_ce(I, logP, A, D, np.zeros(m))
    assert ce - bayes > 1e-3


def test_word_attribute_nuclear_beats_truncation_under_budget():
    logP, _ = word_attribute(N=128, m=5, p=0.4, decay=0.9, scale=1.75, seed=0)
    I = np.eye(len(logP))
    A0, D0 = truncation(logP, d=2)
    ce_cf, bayes = expected_ce(I, logP, A0, D0, np.zeros(logP.shape[1]))
    A, D, b, _ = nuclear_features(logP, I, d=2, fit_bias=False)
    ce, _ = expected_ce(I, logP, A, D, b)
    assert np.allclose(b, 0.0)
    assert ce <= ce_cf + 1e-4


def test_attribute_stats_recover_the_ladder():
    logP, info = word_attribute(N=128, m=6, p=0.3, decay=0.85, scale=1.6, seed=1)
    a_hat, b_hat = attribute_stats(logP, info["Z"])
    assert np.allclose(a_hat, info["alpha"])                      # swing = alpha, exactly
    assert np.allclose(b_hat, b_hat[0])                           # no attribute intercepts


def test_feature_vectors_recover_additive_fit():
    rng = np.random.default_rng(0)
    Z = (rng.random((32, 5)) > 0.5).astype(float)
    W, c = rng.normal(size=(5, 3)), rng.normal(size=3)
    V, v0 = feature_vectors(Z @ W + c, Z)
    assert np.allclose(V, W) and np.allclose(v0, c)


def _wa_config():
    cfg = OmegaConf.create({
        "setting": "graph-cooccurrence", "structure": "word_attribute",
        "structure_kwargs": {"N": 32, "m": 4, "p": 0.3, "decay": 0.8, "scale": 1.5, "seed": 2},
        "random_seed": 0,
        "model_kwargs": {"model_type": "linear"},
        "training_kwargs": {"batch_size": 16},
        "eval_kwargs": {"num_eval_batches": 1, "output_hidden_states": False},
    })
    return Config(cfg)


def test_word_attribute_dataset_exposes_exact_targets():
    ds = CooccurrenceDataset(_wa_config())
    logP, _ = word_attribute(32, 4, 0.3, 0.8, 1.5, 2)
    assert ds.vocab_size == ds.start_token + 1 and ds.num_dims == 4 and ds.arity == 1
    assert np.allclose(ds.conditional, logP)
    b = ds[0]
    assert b["input_ids"].shape == (32, 1)
    assert (b["input_ids"][:, 0].cpu().numpy() == np.arange(32)).all()
    assert np.allclose(b["labels"].cpu().numpy(), np.exp(logP), atol=1e-6)
