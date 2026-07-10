"""HMM dataset: latent states walk a graph + emit tokens. The forward algorithm's belief
p(s_t | x_{1:t}) recovers the latent geometry when emissions are informative; a token alone does
not (the geometry lives in belief space, not the emission alphabet)."""
import numpy as np
import torch
from omegaconf import OmegaConf

from src.code.config import Config
from src.code.data import HMMDataset
from src.code.geometry import predicted_geometry
from src.code.eval import cka


def _cfg(emission="bump", sigma=1.0, n=16, M=16, seed=0):
    return Config(OmegaConf.create({
        "setting": "hmm", "structure": "ring", "structure_kwargs": {"n": n},
        "gamma": 0.8, "cooccurrence": "diffusion",
        "emission": emission, "emission_sigma": sigma, "n_symbols": M,
        "sample_mode": "sequence", "random_seed": seed,
        "model_kwargs": {"model_type": "transformer"},
        "training_kwargs": {"batch_size": 64, "context_length": 40},
        "eval_kwargs": {"num_eval_batches": 1, "output_hidden_states": False},
    }))


def test_forward_algorithm_returns_valid_beliefs():
    ds = HMMDataset(_cfg())
    x = ds[0]["input_ids"][:, 1:]                              # drop the start token
    b = ds.forward_algorithm(x)
    assert b.shape == (x.shape[0], x.shape[1], ds.n_states)
    assert torch.allclose(b.sum(-1), torch.ones_like(b.sum(-1)), atol=1e-4)


def test_batch_shapes_and_normalised_targets():
    ds = HMMDataset(_cfg()); b = ds[0]
    B, L = ds._batch_size(), ds.config.training_kwargs["context_length"]
    assert b["input_ids"].shape == (B, L + 1)                  # start + L emitted tokens
    assert b["eval_labels"].shape == (B, L, ds.num_dims)       # Bayes-optimal next-token dists
    ones = torch.ones(B, L, device=b["eval_labels"].device)
    assert torch.allclose(b["eval_labels"].sum(-1), ones, atol=1e-4)


def _belief_geometry_cka(sigma):
    """CKA between the per-latent-state mean belief-weighted code and the predicted latent geometry."""
    ds = HMMDataset(_cfg(sigma=sigma))
    E = predicted_geometry(ds.latent_conditional)             # predicted latent ring
    L = 600; s = torch.empty(1, L, dtype=torch.long, device=ds.device)
    s[:, 0] = torch.multinomial(ds.pi, 1, generator=ds.rng)
    for t in range(1, L):
        s[:, t] = torch.multinomial(ds.T[s[:, t - 1]], 1, generator=ds.rng).squeeze(1)
    x = torch.multinomial(ds.Em[s.reshape(-1)], 1, generator=ds.rng).reshape(1, L)
    b = ds.forward_algorithm(x)[0].cpu().numpy(); st = s[0].cpu().numpy()
    codes = np.vstack([b[st == k].mean(0) for k in range(ds.n_states)]) @ E    # (n_states, d)
    return cka(codes, E)


def test_informative_emissions_recover_latent_geometry_in_belief_space():
    assert _belief_geometry_cka(sigma=0.8) > 0.9             # sharp emissions -> belief is the ring
    assert _belief_geometry_cka(sigma=0.8) > _belief_geometry_cka(sigma=8.0) + 0.2   # >> aliased


def test_bump_and_permuted_emissions_have_uniform_per_state_entropy():
    """Both emission modes give every state the same peakiness -> even belief coverage."""
    for emission in ("bump", "permuted"):
        Em = HMMDataset(_cfg(emission=emission, sigma=1.0, n=16, M=16)).Em.cpu().numpy()
        ent = -(Em * np.log(Em + 1e-12)).sum(1)
        assert ent.std() < 0.02 * ent.mean()                # near-identical entropy across states


def test_spectral_hmm_matches_forward_algorithm_up_to_a_linear_map():
    from src.code.geometry import spectral_hmm, spectral_beliefs
    from src.code.eval import cca
    ds = HMMDataset(_cfg(emission="permuted", sigma=1.0, n=12, M=24))   # M > n so spectral rank is meaningful
    pi, T, Em = ds.pi.cpu().numpy(), ds.T.cpu().numpy(), ds.Em.cpu().numpy()
    U, B, b1, b_inf = spectral_hmm(pi, T, Em)
    assert U.shape == (24, 12)                                # observable subspace has rank n_states
    # sample observation sequences and compare spectral vs forward beliefs
    Bsz, L = 32, 30
    s = torch.empty(Bsz, L, dtype=torch.long, device=ds.device)
    s[:, 0] = torch.multinomial(ds.pi, Bsz, replacement=True, generator=ds.rng)
    for t in range(1, L):
        s[:, t] = torch.multinomial(ds.T[s[:, t-1]], 1, generator=ds.rng).squeeze(1)
    x = torch.multinomial(ds.Em[s.reshape(-1)], 1, generator=ds.rng).reshape(Bsz, L)
    fwd = ds.forward_algorithm(x).cpu().numpy().reshape(-1, ds.n_states)
    spec = spectral_beliefs(B, b1, b_inf, x.cpu().numpy()).reshape(-1, ds.n_states)
    assert cca(spec, fwd) > 0.999                             # identical up to an invertible linear map
    assert cka(spec, fwd) < cca(spec, fwd)                    # ... but not up to rotation (CKA under-reports)
