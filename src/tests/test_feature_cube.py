"""Word -> feature family: N words with i.i.d. Bernoulli(p) features in {0, 1},
outcomes the m feature tokens, softmax logits alpha_i Z_xi with alpha_i = scale * decay^i."""
import numpy as np
from omegaconf import OmegaConf

from src.code.config import Config
from src.code.data import KernelDataset, feature_cube
from src.code.geometry import (feature_stats, expected_ce, feature_vectors,
                               nuclear_features, truncation_features as truncation)


def test_features_kernel_structure():
    N, m, p, decay, scale = 64, 5, 0.25, 0.8, 1.5
    logP, info = feature_cube(N, m, p, decay, scale, seed=3)
    Z, alpha = info["Z"], info["alpha"]
    assert logP.shape == (N, m) and Z.shape == (N, m)
    assert np.allclose(np.exp(logP).sum(1), 1.0)
    assert set(np.unique(Z)) <= {0.0, 1.0}                        # sparse 0/1 bundles
    assert 0.05 < Z.mean() < 0.5                                  # Bernoulli(p) activity
    assert np.allclose(alpha, scale * decay ** np.arange(m))
    # gauge-free identity: within-row logit contrasts equal alpha_i z_i - alpha_j z_j
    G = logP - logP[:, :1]
    assert np.allclose(G, Z * alpha - (Z * alpha)[:, :1])
    L2, _ = feature_cube(N, m, p, decay, scale, seed=3)
    assert np.allclose(L2, logP)                                  # lexicon fixed by seed


def test_features_effective_rank_is_m_minus_1():
    # the softmax gauge absorbs one dimension: d = m - 1 recovers Bayes, d = m - 2 does not
    logP, info = feature_cube(N=128, m=5, p=0.4, decay=0.9, scale=1.75, seed=0)
    I = np.eye(len(logP))
    m = logP.shape[1]
    A, D = truncation(logP, d=m - 1)
    ce, bayes = expected_ce(I, logP, A, D, np.zeros(m))
    assert ce - bayes < 1e-8
    A, D = truncation(logP, d=m - 2)
    ce, _ = expected_ce(I, logP, A, D, np.zeros(m))
    assert ce - bayes > 1e-3


def test_features_nuclear_beats_truncation_under_budget():
    logP, _ = feature_cube(N=128, m=5, p=0.4, decay=0.9, scale=1.75, seed=0)
    I = np.eye(len(logP))
    A0, D0 = truncation(logP, d=2)
    ce_cf, bayes = expected_ce(I, logP, A0, D0, np.zeros(logP.shape[1]))
    A, D, b, _ = nuclear_features(logP, I, d=2, fit_bias=False)
    ce, _ = expected_ce(I, logP, A, D, b)
    assert np.allclose(b, 0.0)
    assert ce <= ce_cf + 1e-4


def test_feature_stats_recover_the_ladder():
    logP, info = feature_cube(N=128, m=6, p=0.3, decay=0.85, scale=1.6, seed=1)
    a_hat, b_hat = feature_stats(logP, info["Z"])
    assert np.allclose(a_hat, info["alpha"])                      # swing = alpha, exactly
    assert np.allclose(b_hat, b_hat[0])                           # no feature intercepts


def test_feature_vectors_recover_additive_fit():
    rng = np.random.default_rng(0)
    Z = (rng.random((32, 5)) > 0.5).astype(float)
    W, c = rng.normal(size=(5, 3)), rng.normal(size=3)
    V, v0 = feature_vectors(Z @ W + c, Z)
    assert np.allclose(V, W) and np.allclose(v0, c)


def _wa_config():
    cfg = OmegaConf.create({
        "setting": "graph-cooccurrence", "structure": "feature_cube",
        "structure_kwargs": {"N": 32, "m": 4, "p": 0.3, "decay": 0.8, "scale": 1.5, "seed": 2},
        "random_seed": 0,
        "model_kwargs": {"model_type": "linear"},
        "training_kwargs": {"batch_size": 16},
        "eval_kwargs": {"num_eval_batches": 1, "output_hidden_states": False},
    })
    return Config(cfg)


def test_features_dataset_exposes_exact_targets():
    ds = KernelDataset(_wa_config())
    logP, _ = feature_cube(32, 4, 0.3, 0.8, 1.5, 2)
    assert ds.vocab_size == 32 + 4 and ds.start_token is None   # words + feature block
    assert ds.num_dims == 4 and ds.arity == 1
    assert np.allclose(ds.conditional, logP)
    b = ds[0]
    assert b["input_ids"].shape == (32, 2)                       # every word, then a draw
    assert (b["input_ids"][:, 0].cpu().numpy() == np.arange(32)).all()
    assert np.allclose(b["distribution_labels"].cpu().numpy(), np.exp(logP), atol=1e-6)


def test_sae_trains_from_the_shared_pipeline_on_feature_codes(tmp_path):
    # the SAE reads the item feature codes, so its frozen embedding is exactly Z
    from src.code.train import train_inline
    model, ds, evaluator, *_ = train_inline(
        ["setting=feature_cube", "runner=sae", "structure_kwargs.N=64", "structure_kwargs.m=5",
         "model_kwargs.model_config.hidden_size=2", "training_kwargs.train_steps=50"],
        name="sae_pipeline_test", cache_dir=str(tmp_path))
    codes = model.embedding.weight.detach().cpu().numpy()[:ds.n_items]
    assert np.allclose(codes, ds.features)
    assert not model.embedding.weight.requires_grad
    assert evaluator.item_representations(model).shape == (64, 2)   # the bottleneck code


def test_power_spectrum_gives_a_power_law_importance_ladder():
    # alpha_i = scale * (i+1)^-beta, versus the default geometric scale * decay^i
    from src.code.data import feature_cube
    _, info = feature_cube(N=64, m=6, p=0.3, scale=2.0, seed=0, spectrum="power", beta=1.5)
    expected = 2.0 * (1.0 + np.arange(6)) ** -1.5
    assert np.allclose(info["alpha"], expected)
    _, geo = feature_cube(N=64, m=6, p=0.3, decay=0.9, scale=2.0, seed=0)
    assert np.allclose(geo["alpha"], 2.0 * 0.9 ** np.arange(6))          # default unchanged


def test_rho_correlates_features_and_preserves_marginals():
    # one-factor Gaussian copula: rho=0 must reproduce the independent path exactly,
    # rho>0 raises pairwise correlation while leaving P(z_i=1) = p.
    from src.code.data import feature_cube
    _, base = feature_cube(N=4000, m=16, p=0.3, seed=0)
    _, same = feature_cube(N=4000, m=16, p=0.3, seed=0, rho=0.0)
    assert np.array_equal(base["Z"], same["Z"])

    _, corr = feature_cube(N=4000, m=16, p=0.3, seed=0, rho=0.6)
    assert abs(corr["Z"].mean() - 0.3) < 0.02                    # marginals preserved
    off = lambda Z: np.corrcoef(Z.T)[np.triu_indices(16, 1)].mean()
    assert off(corr["Z"]) > 0.3                                  # clearly correlated
    assert abs(off(base["Z"])) < 0.05                            # baseline is not


def test_blocks_give_within_block_correlation_only():
    # with blocks>1 each block shares its own factor, so within-block pairs correlate
    # and cross-block pairs do not.
    from src.code.data import feature_cube
    _, info = feature_cube(N=6000, m=16, p=0.3, seed=0, rho=0.6, blocks=4)
    C = np.corrcoef(info["Z"].T)
    blk = np.arange(16) // 4
    same = np.array([[blk[i] == blk[j] for j in range(16)] for i in range(16)])
    iu = np.triu_indices(16, 1)
    assert C[iu][same[iu]].mean() > 0.3
    assert abs(C[iu][~same[iu]].mean()) < 0.06
