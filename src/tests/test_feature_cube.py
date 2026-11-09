"""Word -> feature family: N words with i.i.d. Bernoulli(p) features in {0, 1},
outcomes the m feature tokens, softmax logits alpha_i Z_xi with alpha_i = scale * decay^i."""
import numpy as np
import pytest
from omegaconf import OmegaConf

from src.code.config import Config
from src.code.data import KernelDataset, feature_cube
from src.code.geometry import factor_logcond


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
    # the softmax gauge absorbs one dimension: row-centring the log-conditional leaves
    # exactly m - 1 nonzero singular values, so a rank-(m-1) code is already exact.
    logP, _ = feature_cube(N=128, m=5, p=0.4, decay=0.9, scale=1.75, seed=0)
    _, _, _, S = factor_logcond(logP)
    assert (S[:4] > 1e-8).all() and S[4] < 1e-8


def _wa_config():
    cfg = OmegaConf.create({
        "setting": "graph-cooccurrence", "structure": "feature_cube",
        "structure_kwargs": {"N": 32, "m": 4, "p": 0.3, "decay": 0.8, "scale": 1.5, "seed": 2},
        "random_seed": 0,
        "model_kwargs": {"model_type": "linear"},
        "training_kwargs": {"batch_size": 16},
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
    assert evaluator.evaluate(model)["hidden_states"][-1][:, 0].shape == (64, 2)   # the bottleneck code


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


def _paired(m, decay=1.0, scale=2.0):
    """The yes/no world: every bundle once, a (k+, k-) token pair per feature, columns
    [1+, ..., m+, 1-, ..., m-]."""
    return feature_cube(m=m, decay=decay, scale=scale, paired=True, all_bundles=True)


def test_paired_kernel_is_alpha_z_and_alpha_one_minus_z():
    # Lc(z, k+) = alpha_k z_k, Lc(z, k-) = alpha_k (1 - z_k); L = log softmax(Lc). The
    # normalizer is the same for every word, so L differs from Lc by one global constant.
    from src.code.data import cube_codes
    m = 2
    L, info = _paired(m)
    Z, alpha = info["Z"], info["alpha"]
    assert L.shape == (2 ** m, 2 * m) and Z.shape == (2 ** m, m)
    assert np.array_equal(Z, cube_codes(m))
    Lc = np.c_[Z * alpha, (1 - Z) * alpha]
    assert np.allclose(L - Lc, (L - Lc)[0, 0])
    # within-pair log-odds is that feature alone
    assert np.allclose(L[:, :m] - L[:, m:], alpha * (2 * Z - 1))


def test_paired_rank_is_one_dimension_per_feature():
    # equal alpha: rank m; unequal alpha adds the pair-budget constant (rank m + 1);
    # the single-token cube loses one dimension to the softmax gauge (rank m - 1)
    rank = lambda L: int((factor_logcond(L)[3] > 1e-8).sum())
    assert rank(_paired(2)[0]) == 2
    assert rank(_paired(2, decay=0.9)[0]) == 3
    assert rank(feature_cube(m=3, decay=1.0, all_bundles=True)[0]) == 2


def test_paired_geometry_is_axes_and_corners():
    # balanced factorization: the two decoders of a pair are antipodal, pairs are orthogonal,
    # and a word is the sum of its chosen prototype per feature, scaled by 1/sqrt(2)
    L, info = _paired(2)
    _, E, D, S = factor_logcond(L)
    E, D = E[:, :2], D[:2]                                        # the exact rank-2 code
    d = {("+", k): D[:, k] for k in range(2)} | {("-", k): D[:, 2 + k] for k in range(2)}
    for k in range(2):
        assert np.allclose(d["+", k], -d["-", k])
    assert abs(d["+", 0] @ d["+", 1]) < 1e-8
    for z, e in zip(info["Z"], E):
        proto = sum(d["+" if z[k] else "-", k] for k in range(2))
        assert np.allclose(e, proto / np.sqrt(2))


def test_single_token_cube_collapses_all_on_and_all_off():
    # control: with one token per feature and equal alpha, the words 00 and 11 emit the
    # same distribution, so the square is projected onto a line
    L, info = feature_cube(m=2, decay=1.0, all_bundles=True)
    Z = info["Z"]
    i00 = np.flatnonzero((Z == 0).all(1))[0]; i11 = np.flatnonzero((Z == 1).all(1))[0]
    assert np.allclose(L[i00], L[i11])
    assert int((factor_logcond(L)[3] > 1e-8).sum()) == 1


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


# ── signed features: {-1, 0, 1} with one yes/no token pair per feature (notes/superposition_new.tex) ──

def test_signed_kernel_is_plus_minus_alpha_z_and_ties_absent_pairs():
    # Lc(z, i+) = alpha_i z_i, Lc(z, i-) = -alpha_i z_i, one softmax over the 2m outputs; a feature
    # at 0 leaves its pair tied; all_bundles enumerates {-1, 0, 1}^m once
    m = 3
    L, info = feature_cube(m=m, decay=0.8, scale=1.5, signed=True, all_bundles=True)
    Z, alpha = info["Z"], info["alpha"]
    assert L.shape == (3 ** m, 2 * m) and Z.shape == (3 ** m, m)
    assert set(np.unique(Z)) == {-1.0, 0.0, 1.0} and len(np.unique(Z, axis=0)) == 3 ** m
    Lc = np.c_[Z * alpha, -Z * alpha]
    assert np.allclose(L - Lc, (L - Lc)[:, :1])                     # equal up to each row's normalizer
    assert np.allclose(L[:, :m] - L[:, m:], 2 * alpha * Z)          # within-pair log-odds
    absent = Z == 0
    assert np.allclose(L[:, :m][absent], L[:, m:][absent])


def test_signed_k_supports_have_exactly_k_active_features_with_fair_signs():
    N, m, k = 6000, 8, 2
    _, info = feature_cube(N=N, m=m, k=k, signed=True, seed=1)
    Z = info["Z"]
    assert Z.shape == (N, m) and np.all((Z != 0).sum(1) == k)
    assert np.abs((Z != 0).mean(0) - k / m).max() < 0.03           # uniform supports
    assert np.abs(Z.mean(0)).max() < 0.03                           # fair, independent signs
    assert np.abs(np.corrcoef(Z.T)[np.triu_indices(m, 1)]).max() < 0.05


def test_signed_all_bundles_with_k_enumerates_every_signed_support():
    # k = 1 gives the 2m signed singletons; k = 2 the C(m, 2) 2^2 signed pairs
    m = 4
    for k, n in ((1, 2 * m), (2, 6 * 4)):
        _, info = feature_cube(m=m, k=k, signed=True, all_bundles=True)
        Z = info["Z"]
        assert Z.shape == (n, m) and len(np.unique(Z, axis=0)) == n and np.all((Z != 0).sum(1) == k)


def test_signed_mixing_hands_the_model_mixed_coordinates_of_singleton_patterns():
    # z = A z' with z' the signed singletons and A the symmetric root of the AR(1) correlation
    # matrix mix^|i-j|; the logits are defined on the patterns z', the model sees z
    m, rho = 4, 0.5
    L, info = feature_cube(m=m, k=1, signed=True, all_bundles=True, mix=rho, scale=2.0, decay=1.0)
    Z, Zp, A, alpha = info["Z"], info["patterns"], info["A"], info["alpha"]
    Sigma = rho ** np.abs(np.arange(m)[:, None] - np.arange(m)[None])
    assert np.allclose(A @ A, Sigma) and np.allclose(Z, Zp @ A.T)
    assert np.allclose(L[:, :m] - L[:, m:], 2 * alpha * Zp)
    assert np.allclose(np.corrcoef(Z.T), Sigma)                      # the features the model sees are correlated


def test_frequency_ladder_weights_each_signed_singleton_by_half_its_feature_frequency():
    # p(z = +-e_i) = pi_i / 2 with pi_i proportional to (i + 1)^-freq_beta
    m, beta = 5, 1.0
    _, info = feature_cube(m=m, k=1, signed=True, all_bundles=True, freq_beta=beta)
    Z, w = info["Z"], info["weights"]
    pi = (1.0 + np.arange(m)) ** -beta
    pi /= pi.sum()
    feature = np.abs(Z).argmax(1)
    assert np.isclose(w.sum(), 1.0) and np.allclose(w, pi[feature] / 2)


def test_weighted_batch_cross_entropy_is_the_frequency_weighted_mean():
    # the batch scales each target row by n w_z, so cross-entropy (linear in the target) over the
    # tiled batch equals sum_z w_z CE_z exactly
    import torch
    import torch.nn.functional as F
    from src.code.train import graph_overrides, resolve_config
    m = 4
    config = resolve_config(graph_overrides({"structure": "feature_cube", "m": m, "k": 1, "signed": True,
                                             "all_bundles": True, "scale": 1.0, "decay": 1.0, "freq_beta": 1.0}))
    ds = KernelDataset(config)
    b = ds[0]
    logits = torch.randn(len(b["input_ids"]), ds.n_outcomes, generator=torch.Generator().manual_seed(0)).to(b["distribution_labels"])
    rows = b["input_ids"][:, 0].cpu().numpy()
    per_row = -(ds.P[rows] * torch.log_softmax(logits, -1)).sum(-1).cpu().numpy()
    w = ds.info["weights"][rows]
    expected = (w * per_row).sum() / w.sum()
    assert np.isclose(F.cross_entropy(logits, b["distribution_labels"]).item(), expected, rtol=1e-5)


@pytest.mark.slow
def test_weak_logits_keep_the_most_frequent_features_and_follow_reversed_frequencies(tmp_path):
    # equal weak strengths, rank 3 of 6: the weighted SVD keeps the modes with the largest pi_i, so a
    # trained linear network dedicates its dimensions to features 1-3; reversing the ladder
    # (freq_beta = -1) moves them to features 4-6, which rules out an effect of feature order
    from src.code.analysis_utils import leverage, representation
    from src.code.train import graph_overrides, train_inline
    for beta, kept in ((1.0, [0, 1, 2]), (-1.0, [3, 4, 5])):
        spec = {"structure": "feature_cube", "m": 6, "k": 1, "signed": True, "all_bundles": True,
                "scale": 0.5, "decay": 1.0, "freq_beta": beta}
        model, ds, evaluator, *_ = train_inline(
            graph_overrides(spec) + ["runner=linear", "model_kwargs.model_config.embed_dim=3", "random_seed=0",
                                     "training_kwargs.train_steps=8000", "training_kwargs.optimizer=sgd",
                                     "training_kwargs.learning_rate=0.5"],
            name=f"freq_{beta}", cache_dir=str(tmp_path))
        E = np.asarray(representation(evaluator.evaluate(model)["hidden_states"], static=True))
        plus = [int(np.flatnonzero(ds.features[:, i] == 1)[0]) for i in range(6)]   # the +e_i states
        lev = leverage(E[plus])[0]
        assert np.allclose(np.sort(np.argsort(lev)[-3:]), kept), lev
        assert lev[kept].min() > 0.9 and np.delete(lev, kept).max() < 0.1, lev
