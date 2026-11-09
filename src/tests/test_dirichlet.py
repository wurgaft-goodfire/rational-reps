"""Dirichlet-categorical streams: z ~ Dir(1, ..., 1) once per stream, x_1:t i.i.d. Cat(z). The
Bayes next-token predictive is (1 + n_x) / (K + t); the posterior over z, read on a lattice of the
simplex, is the flat prior times prod_x z_x^{n_x}."""
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.code.analysis_utils import heldout_decode_r2
from src.code.config import Config
from src.code.data import DirichletDataset
from src.code.eval import Evaluator


def _cfg(K=3, T=12, B=64, seed=0, **extra):
    return Config(OmegaConf.create({
        "setting": "dirichlet", "n_outcomes": K, "lattice": 12, "sample_mode": "sequence",
        "random_seed": seed, "model_kwargs": {"model_type": "transformer"},
        "training_kwargs": {"batch_size": B, "context_length": T}, **extra,
    }))


def _counts(obs, K):
    """Counts before each position: (B, L) -> (B, L, K), row t = counts of obs[:, :t]."""
    onehot = torch.nn.functional.one_hot(obs, K).double()
    return torch.cat([torch.zeros_like(onehot[:, :1]), onehot.cumsum(1)[:, :-1]], 1)


def test_bayes_predictive_is_add_one_smoothing():
    ds = DirichletDataset(_cfg())
    seq = ds.batch()["input_ids"]
    n = _counts(seq[:, 1:], 3)
    t = n.sum(-1, keepdim=True)
    assert torch.allclose(ds.optimal_predictions(seq).double(), (1 + n) / (3 + t), atol=1e-6)


def test_streams_draw_one_flat_dirichlet_z_each():
    ds = DirichletDataset(_cfg(B=20000, T=4))
    b = ds.batch()
    z = b["latents"]
    assert z.shape == (20000, 3) and torch.allclose(z.sum(-1), torch.ones(20000, device=z.device, dtype=z.dtype))
    assert torch.allclose(z.mean(0), torch.full((3,), 1 / 3, device=z.device, dtype=z.dtype), atol=0.01)
    assert abs(z[:, 0].var().item() - 1 / 18) < 0.003                  # Dir(1,1,1) marginal: Beta(1, 2)
    assert b["input_ids"][:, 0].eq(ds.start_token).all() and b["input_ids"][:, 1:].lt(3).all()


def test_a_uniform_draw_of_exactly_zero_still_gives_a_valid_z(monkeypatch):
    # torch.rand can return 0.0 for one symbol; the exponential draw must stay finite there
    ds = DirichletDataset(_cfg(B=8))
    def one_zero(*shape, generator=None, device=None):
        u = torch.full(shape, 0.5, device=device); u[:, 0] = 0.0
        return u
    monkeypatch.setattr(torch, "rand", one_zero)
    z = ds.batch()["latents"]
    assert torch.isfinite(z).all()


def test_lattice_is_the_interior_of_the_simplex_with_a_flat_prior():
    ds = DirichletDataset(_cfg())
    G = ds.P.double()
    assert (G > 0).all() and torch.allclose(G.sum(-1), torch.ones(len(G), device=G.device, dtype=G.dtype))
    assert torch.allclose(torch.as_tensor(ds.conditional, device=G.device), G.log())
    assert torch.allclose(ds.prior, torch.full_like(ds.prior, 1 / len(G)))


def test_latent_target_is_rejected():
    with pytest.raises(ValueError):
        DirichletDataset(_cfg(infer_latent=True))


@pytest.mark.slow
def test_attention_writes_the_predictive_and_the_posterior_over_z_up_to_its_length():
    # a 1-layer transformer trained to near-Bayes on next-token prediction. Attention averages, so
    # the residual right after it (before the MLP) holds the frequencies (1 + n) / (K + t): one
    # ridge map reads the predictive, while the posterior over z (linear in the counts n, i.e.
    # t times a frequency) is read exactly by one ridge map per position t
    from src.code.models import init_model
    config = _cfg(B=256)
    ds = DirichletDataset(config)
    config.cfg.model_kwargs.model_config = {
        "hidden_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2, "intermediate_size": 64,
        "use_parallel_residual": False, "vocab_size": ds.vocab_size, "max_position_embeddings": ds.seq_len}
    model = init_model(config)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(1500):
        out = model(**ds.batch())
        opt.zero_grad(); out["loss"].backward(); opt.step()
    ev = Evaluator(config, ds).evaluate(model)
    assert ev["loss"] < 0.02
    H = ev["attention_states"][0][:, :-1]                                 # (B, T, width): states after x_1:t, t = 0..T-1
    n = _counts(torch.as_tensor(ev["inputs"][:, 1:]), 3).numpy()         # (B, T, K)
    logG = np.log(ds.P.double().cpu().numpy())
    LP = n @ (logG - logG.mean(0)).T                                     # z-centred log-posterior on the lattice
    predictive = (1 + n) / (3 + n.sum(-1, keepdims=True))
    streams = np.repeat(np.arange(len(H)), H.shape[1])                   # held-out streams are wholly unseen
    assert heldout_decode_r2(H.reshape(-1, H.shape[-1]), predictive.reshape(-1, 3), streams) > 0.95
    per_t = [heldout_decode_r2(H[:, t], LP[:, t], np.arange(len(H))) for t in range(1, H.shape[1])]
    assert min(per_t) > 0.95, per_t
