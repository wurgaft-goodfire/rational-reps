"""Dirichlet-categorical dataset: each sequence draws theta ~ Dir(alpha) and emits iid tokens.
The Bayes-optimal output is the posterior predictive; the predicted in-context representation is
its centered log — the logit code (`DirichletCategoricalDataset.logit_code`)."""
import numpy as np
import torch
from omegaconf import OmegaConf

from src.code.config import Config
from src.code.data import DirichletCategoricalDataset


def _centered_log(alpha, counts):
    g = np.log(alpha + np.asarray(counts, float))
    return g - g.mean(1, keepdims=True)


def _cfg(K=3, alpha=1.0, seed=0, batch_size=16, context_length=12):
    return Config(OmegaConf.create({
        "setting": "dirichlet-categorical", "n_categories": K, "alpha": alpha,
        "sample_mode": "sequence", "random_seed": seed,
        "model_kwargs": {"model_type": "transformer"},
        "training_kwargs": {"batch_size": batch_size, "context_length": context_length},
        "eval_kwargs": {"num_eval_batches": 1, "output_hidden_states": False},
    }))


def test_batch_shapes_and_start_token():
    ds = DirichletCategoricalDataset(_cfg(K=3, batch_size=16, context_length=12))
    b = ds[0]
    assert ds.vocab_size == 4 and ds.num_dims == 3
    assert b["input_ids"].shape == (16, 13)                    # start + L emitted tokens
    assert (b["input_ids"][:, 0] == 3).all()                   # start token first
    el = ds.optimal_predictions(b["input_ids"])
    assert el.shape == (16, 12, 3)                             # Bayes-optimal next-token dists
    assert torch.allclose(el.sum(-1), torch.ones(16, 12, device=el.device), atol=1e-5)


def test_eval_labels_are_the_posterior_predictive():
    ds = DirichletCategoricalDataset(_cfg(K=3, alpha=1.0))
    b = ds[0]
    el = ds.optimal_predictions(b["input_ids"])
    x = b["input_ids"][:, 1:]
    # position 0 predicts x_1 from the flat prior; position t from counts of x_{1:t}
    prior = torch.full((3,), 1.0 / 3, device=x.device)
    assert torch.allclose(el[:, 0], prior.expand_as(el[:, 0]))
    for t in (1, 5, 11):
        c = torch.stack([(x[:, :t] == k).sum(1) for k in range(3)], 1).float()
        expected = (1.0 + c) / (3.0 + t)
        assert torch.allclose(el[:, t], expected, atol=1e-5)


def test_logit_code_is_the_centered_log_of_the_bayes_outputs():
    # centered log(alpha + counts): centering removes the shared -log(alpha_0 + t) normalizer
    ds = DirichletCategoricalDataset(_cfg(K=3, alpha=1.0))
    b = ds[0]
    x = b["input_ids"][:, 1:]
    code = ds.logit_code(x)[:, :-1].reshape(-1, 3).cpu().numpy()          # code after x_{1:t}
    logp = np.log(ds.optimal_predictions(b["input_ids"])[:, 1:]
                  .reshape(-1, 3).cpu().numpy())                          # Bayes output after x_{1:t}
    np.testing.assert_allclose(code, logp - logp.mean(1, keepdims=True), atol=1e-5)
    np.testing.assert_allclose(code.sum(1), 0.0, atol=1e-6)


def test_logit_code_is_the_belief_weighted_prototype_combination():
    # posterior-averaging prototype codes (SVD of log p(x|theta), theta ~ prior) = the logit code
    # up to the Jensen (log-of-mean vs mean-of-log) correction -> equal up to a linear map
    from src.code.data import categorical
    from src.code.geometry import factor_logcond
    from src.code.eval import cca
    rng = np.random.default_rng(0)
    L, info = categorical(K=3, n_items=4000, alpha=1.0, seed=1)   # prototype theta ~ Dir(1)
    _, E_proto, _, S, _ = factor_logcond(L)
    E_proto = E_proto[:, S > 1e-9 * S[0]]
    counts = rng.multinomial(1, np.full(3, 1 / 3), size=(50, 8)).sum(1)   # states with t = 8
    logw = counts @ np.log(info["P"].T)                            # log-likelihood of each prototype
    w = np.exp(logw - logw.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)   # posterior weights
    mc_code = w @ E_proto                                          # belief-weighted prototypes
    assert cca(mc_code, _centered_log(1.0, counts)) > 0.99


def test_factoring_the_log_outputs_recovers_the_logit_code():
    # observables-only route: factor_logcond of log p(next|context) = the logit code, rank K-1
    from src.code.geometry import factor_logcond
    from src.code.eval import cca
    rng = np.random.default_rng(0)
    counts = rng.multinomial(1, np.full(3, 1 / 3), size=(60, 10)).sum(1)
    outputs = (1.0 + counts) / (3.0 + counts.sum(1, keepdims=True))   # Bayes outputs per context
    _, E, _, S, _ = factor_logcond(np.log(outputs))
    E = E[:, S > 1e-9 * S[0]]
    assert E.shape[1] == 2                                     # exactly rank K-1
    assert cca(E, _centered_log(1.0, counts)) > 0.999


def test_shipped_dirichlet_config_has_learnable_signal():
    # shipped-config guardrail: Bayes must beat the flat prior, else training converges to a constant
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from pyprojroot import here
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(here() / "conf")):
        cfg = compose(config_name="config", overrides=["runner=dirichlet_transformer"])
    ds = DirichletCategoricalDataset(Config(cfg))
    bayes = ds.optimal_predictions(ds[0]["input_ids"])         # (B, L, K) posterior predictives
    prior = torch.full_like(bayes, 1.0 / ds.num_dims)
    kl = (bayes * (bayes / prior).log()).sum(-1)
    assert kl[:, 1:].mean() > 0.05


def test_belief_references_are_float64_and_batches_float32():
    ds = DirichletCategoricalDataset(_cfg())
    b = ds[0]
    x = b["input_ids"][:, 1:]
    assert ds.posterior_predictive(x).dtype == torch.float64
    assert ds.logit_code(x).dtype == torch.float64
    assert ds.optimal_predictions(b["input_ids"]).dtype == torch.float32
