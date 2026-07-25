"""HMM dataset: latent states walk a graph + emit tokens. The forward algorithm's belief
p(s_t | x_{1:t}) recovers the latent geometry when emissions are informative; a token alone does
not (the geometry lives in belief space, not the emission alphabet)."""
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.code.config import Config
from src.code.data import HMMDataset
from src.code.geometry import predicted_geometry
from src.code.eval import cka


def _cfg(emission="bump", sigma=1.0, n=16, M=16, seed=0, structure="ring", structure_kwargs=None):
    kwargs = {"n": n} if structure_kwargs is None else structure_kwargs
    return Config(OmegaConf.create({
        "setting": "hmm", "structure": structure,
        "structure_kwargs": {"gamma": 0.8, "cooccurrence": "diffusion", **kwargs},
        "emission": emission, "emission_sigma": sigma, "n_symbols": M,
        "sample_mode": "sequence", "random_seed": seed,
        "model_kwargs": {"model_type": "transformer"},
        "training_kwargs": {"batch_size": 64, "context_length": 40},
        "eval_kwargs": {"num_eval_batches": 1, "output_hidden_states": False},
    }))


@pytest.mark.parametrize("structure,kwargs", [
    ("ring", {"n": 12}), ("chain", {"n": 12}), ("grid", {"n": 4, "m": 4}),
    ("sphere", {"n": 30}), ("communities", {"n": 6, "k": 3}), ("hypercube", {"n": 4}),
    ("complete", {"n": 10}), ("tree", {"n": 3, "k": 2}),
    ("product", {"factors": [{"structure": "ring", "n": 6}, {"structure": "ring", "n": 4}]}),
])
def test_every_structure_works_as_hmm_latent(structure, kwargs):
    # Every structure yields an adjacency + conditional, so any of them can be the HMM latent graph
    # (tree/product included, not only the plain graphs).
    ds = HMMDataset(_cfg(structure=structure, structure_kwargs=kwargs))
    x = ds[0]["input_ids"][:, 1:]                              # drop the start token
    b = ds.forward_algorithm(x)
    assert b.shape == (x.shape[0], x.shape[1], ds.n_states)    # a belief per position over the states
    assert torch.allclose(b.sum(-1), torch.ones_like(b.sum(-1)), atol=1e-4)   # valid distributions


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


def test_shipped_hmm_config_has_learnable_signal():
    """Guardrail on the *shipped* config, not a hand-rolled one: the Bayes-optimal next-token
    predictor must beat the stationary baseline by a clear margin. This is the invariant an
    over-wide emission (or any silent data-pipeline change) breaks first — beliefs never leave
    the prior, KL(bayes || stationary) -> 0, and training 'converges' to a constant."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from pyprojroot import here
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(here() / "conf")):
        cfg = compose(config_name="config", overrides=["runner=hmm_transformer"])
    ds = HMMDataset(Config(cfg))
    bayes = ds[0]["eval_labels"]                               # (B, L, M) forward-algorithm targets
    stationary = ds.pi @ ds.Em
    kl = (bayes * (bayes / stationary).log()).sum(-1)
    assert kl[:, 1:].mean() > 0.05                             # past the first token, context matters


def test_bump_emissions_are_gaussian_in_hop_distance():
    """The HMM treats every structure as its adjacency graph: bump emissions are a Gaussian in
    hop (geodesic) distance, so sigma means 'hops aliased' — not an embedding artifact."""
    ds = HMMDataset(_cfg(sigma=1.0, n=12))
    hops = np.minimum(np.arange(12), 12 - np.arange(12)).astype(float)
    expected = np.exp(-hops ** 2 / 2.0); expected /= expected.sum()
    np.testing.assert_allclose(ds.Em[0].cpu().numpy(), expected, atol=1e-6)

    # tree: states are all nodes of the tree graph; hops through the tree order the aliasing.
    # DFS pre-order for n=2,k=2: 0 root, 1 (2 3) subtree, 4 (5 6) subtree.
    ds = HMMDataset(_cfg(structure="tree", structure_kwargs={"n": 2, "k": 2}))
    assert ds.n_states == 7
    Em = ds.Em.cpu().numpy()
    assert Em[2, 1] > Em[2, 3] > Em[2, 5]                      # leaf 2: parent (1 hop) > sibling (2) > cousin (4)


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
