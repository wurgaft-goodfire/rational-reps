"""Exchangeable-family dataset: each sequence draws theta from a family's atom pool and emits
iid tokens over the binned support. The Bayes-optimal output is the posterior predictive under
the uniform prior over atoms (exact filtering); the predicted in-context representation is its
centered log — the logit code. Generalizes the Dirichlet-categorical task to any family."""
import numpy as np
import torch
from omegaconf import OmegaConf

from src.code.config import Config
from src.code.data import ExchangeableFamilyDataset


def _cfg(family="von_mises", family_kwargs=None, seed=0, batch_size=8, context_length=12,
         labels="tokens"):
    return Config(OmegaConf.create({
        "setting": "exchangeable-family", "family": family,
        "family_kwargs": family_kwargs or {}, "sample_mode": "sequence", "labels": labels,
        "random_seed": seed,
        "model_kwargs": {"model_type": "transformer"},
        "training_kwargs": {"batch_size": batch_size, "context_length": context_length},
        "eval_kwargs": {"num_eval_batches": 1, "output_hidden_states": False},
    }))


def test_batch_shapes_and_start_token():
    ds = ExchangeableFamilyDataset(_cfg("von_mises", {"n_mu": 8, "kappas": (2.0,), "x_n": 10}))
    b = ds[0]
    assert ds.num_dims == 10 and ds.vocab_size == 11            # support bins + start token
    assert b["input_ids"].shape == (8, 13)                      # start + L emitted tokens
    assert (b["input_ids"][:, 0] == 10).all()                   # start token first
    el = ds.optimal_predictions(b["input_ids"])
    assert el.shape == (8, 12, 10)                              # Bayes-optimal next-token dists
    assert torch.allclose(el.sum(-1), torch.ones(8, 12, device=el.device), atol=1e-5)


def test_latent_labels_use_never_input_atom_tokens_and_exact_posterior():
    ds = ExchangeableFamilyDataset(_cfg(
        "gaussian1d", {"n_mu": 5, "n_sig": 2, "x_n": 12}, labels="latent"))
    b = ds[0]

    assert ds.num_dims == ds.M
    assert ds.item_tokens.tolist() == list(range(ds.K + 1, ds.K + 1 + ds.M))
    assert ds.vocab_size == ds.K + 1 + ds.M
    assert not torch.isin(b["input_ids"], ds.item_tokens.to(ds.device)).any()
    assert torch.isin(b["labels"][:, 1:], ds.item_tokens.to(ds.device)).all()
    assert (b["labels"][:, 1:] == b["labels"][:, 1:2]).all()  # one atom per sequence

    x = b["input_ids"][:, 1:]
    expected = torch.cat([
        torch.full((len(x), 1, ds.M), 1.0 / ds.M, dtype=torch.float64, device=ds.device),
        ds.posterior_weights(x)[:, :-1],
    ], dim=1).float()
    assert torch.allclose(ds.optimal_predictions(b["input_ids"]), expected, atol=1e-6)


def test_eval_labels_are_the_exact_posterior_predictive_over_atoms():
    ds = ExchangeableFamilyDataset(_cfg("gaussian1d", {"n_mu": 5, "n_sig": 2, "x_n": 12}))
    b = ds[0]
    el = ds.optimal_predictions(b["input_ids"])
    x = b["input_ids"][:, 1:].cpu().numpy()
    P = ds.atoms.cpu().numpy()                                  # (M, K) emission probs per atom
    # position 0 predicts x_1 from the prior predictive (uniform mixture of atoms)
    prior = torch.tensor(P.mean(0), dtype=el.dtype, device=el.device)
    assert torch.allclose(el[:, 0], prior.expand_as(el[:, 0]), atol=1e-5)
    # position t: brute-force Bayes over the atom pool given x_{1:t}
    for t in (1, 5, 11):
        with np.errstate(divide="ignore"):                      # zero-prob atoms drop out exactly
            logw = np.log(P)[:, x[:, :t]].sum(-1)               # (M, B) log-likelihood per atom
        w = np.exp(logw - logw.max(0)); w /= w.sum(0)
        expected = torch.tensor((w.T @ P), dtype=el.dtype, device=el.device)
        assert torch.allclose(el[:, t], expected, atol=1e-5)


def test_logit_code_is_the_centered_log_of_the_bayes_outputs():
    ds = ExchangeableFamilyDataset(_cfg("von_mises", {"n_mu": 8, "kappas": (2.0,), "x_n": 10}))
    b = ds[0]
    x = b["input_ids"][:, 1:]
    K = ds.num_dims
    code = ds.logit_code(x)[:, :-1].reshape(-1, K).cpu().numpy()          # code after x_{1:t}
    logp = np.log(ds.optimal_predictions(b["input_ids"])[:, 1:]
                  .reshape(-1, K).cpu().numpy())                          # Bayes output after x_{1:t}
    np.testing.assert_allclose(code, logp - logp.mean(1, keepdims=True), atol=1e-5)
    np.testing.assert_allclose(code.sum(1), 0.0, atol=1e-6)


def test_categorical_family_matches_the_dirichlet_conjugate_in_the_large_pool_limit():
    # family=categorical with many atoms ~ Dir(alpha) -> filtering over the pool approaches the
    # conjugate closed form (alpha + counts)/(alpha_0 + t): the Dirichlet task is the special case
    ds = ExchangeableFamilyDataset(_cfg("categorical", {"K": 3, "n_items": 60000, "alpha": 1.0},
                                        batch_size=16, context_length=8))
    b = ds[0]
    x = b["input_ids"][:, 1:]
    counts = torch.cumsum(torch.nn.functional.one_hot(x, 3).float(), dim=1)
    t = torch.arange(1, 9, device=x.device, dtype=torch.float32).view(1, -1, 1)
    conjugate = (1.0 + counts) / (3.0 + t)
    assert (ds.posterior_predictive(x) - conjugate).abs().max() < 0.02


import pytest


@pytest.mark.parametrize("runner", ["exchangeable_transformer", "gaussian_transformer",
                                    "categorical_transformer"])
def test_shipped_exchangeable_config_has_learnable_signal(runner):
    # shipped-config guardrail: Bayes must beat the prior predictive, else training converges
    # to a constant
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from pyprojroot import here
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(here() / "conf")):
        cfg = compose(config_name="config", overrides=[f"runner={runner}"])
    ds = ExchangeableFamilyDataset(Config(cfg))
    bayes = ds.optimal_predictions(ds[0]["input_ids"])         # (B, L, K) posterior predictives
    prior = ds.atoms.mean(0).expand_as(bayes)
    kl = (bayes * ((bayes + 1e-12) / (prior + 1e-12)).log()).sum(-1)
    assert kl[:, 1:].mean() > 0.05


def test_logit_code_stays_finite_beyond_float32_exp_range():
    # narrow atoms on a wide support: tail bins have log-probs far below the float32 exp range,
    # so the mixed predictive underflows; the logit code must come from logsumexp in log space
    ds = ExchangeableFamilyDataset(_cfg("gaussian1d", {"n_mu": 9, "n_sig": 2, "x_n": 36,
                                                       "x_lim": 7.0, "mu_lim": 4.0,
                                                       "sig_lo": 0.3, "sig_hi": 0.5},
                                        batch_size=16, context_length=64))
    b = ds[0]
    x = b["input_ids"][:, 1:]
    code = ds.logit_code(x)
    assert torch.isfinite(code).all()
    # consistent with the direct route where the predictive is representable
    p = ds.posterior_predictive(x).double()
    direct = p.log() - p.log().mean(-1, keepdim=True)
    mask = (p > 1e-30).all(-1)
    assert (code[mask] - direct[mask]).abs().max() < 1e-3


def test_belief_references_are_float64_and_batches_float32():
    # references are the instrument (float64: no underflow, junk at ~1e-14); batch tensors handed
    # to the trainer stay float32 (the network side neither needs nor benefits from more)
    ds = ExchangeableFamilyDataset(_cfg("gaussian1d", {"n_mu": 4, "n_sig": 2, "x_n": 12}))
    b = ds[0]
    x = b["input_ids"][:, 1:]
    assert ds.atoms.dtype == torch.float64 and ds.log_atoms.dtype == torch.float64
    assert ds.posterior_weights(x).dtype == torch.float64
    assert ds.posterior_predictive(x).dtype == torch.float64
    assert ds.logit_code(x).dtype == torch.float64
    assert ds.optimal_predictions(b["input_ids"]).dtype == torch.float32
