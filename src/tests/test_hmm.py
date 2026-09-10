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
from src.code.analysis_utils import cka


def _cfg(emission="bump", sigma=1.0, n=16, M=16, seed=0, structure="ring",
         structure_kwargs=None, latent=False, T=40, **extra):
    kwargs = {"n": n} if structure_kwargs is None else structure_kwargs
    return Config(OmegaConf.create({
        "setting": "hmm", "structure": structure,
        "structure_kwargs": {"gamma": 0.8, "cooccurrence": "diffusion", **kwargs},
        "emission": emission, "emission_sigma": sigma, "n_symbols": M,
        "sample_mode": "sequence", "infer_latent": latent, "random_seed": seed,
        "model_kwargs": {"model_type": "transformer"},
        "training_kwargs": {"batch_size": 64, "context_length": T}, **extra,
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
    el = ds.optimal_predictions(b["input_ids"])
    assert el.shape == (B, L, ds.num_dims)                     # Bayes-optimal next-token dists
    assert torch.allclose(el.sum(-1), torch.ones(B, L, device=el.device), atol=1e-4)


def test_latent_labels_use_never_input_state_tokens_and_predictive_beliefs():
    ds = HMMDataset(_cfg(emission="permuted", n=7, M=11, latent=True)); b = ds[0]

    assert ds.num_dims == ds.n_states
    assert ds.item_tokens.tolist() == list(range(ds.n_symbols + 1,
                                                 ds.n_symbols + 1 + ds.n_states))
    assert ds.vocab_size == ds.n_symbols + 1 + ds.n_states
    assert not torch.isin(b["input_ids"], ds.item_tokens.to(ds.device)).any()
    assert torch.isin(b["labels"][:, 1:], ds.item_tokens.to(ds.device)).all()

    x = b["input_ids"][:, 1:]
    expected = torch.cat([
        ds.prior.expand(len(x), 1, -1),
        ds.forward_algorithm(x)[:, :-1] @ ds.transition,
    ], dim=1).float()
    assert torch.allclose(ds.optimal_predictions(b["input_ids"]), expected, atol=1e-6)


def _belief_geometry_cka(sigma):
    """CKA between the per-latent-state mean belief-weighted code and the predicted latent geometry."""
    ds = HMMDataset(_cfg(sigma=sigma))
    E = predicted_geometry(ds.latent_conditional)             # predicted latent ring
    L = 600; s = torch.empty(1, L, dtype=torch.long, device=ds.device)
    s[:, 0] = torch.multinomial(ds.prior, 1, generator=ds.rng)
    for t in range(1, L):
        s[:, t] = torch.multinomial(ds.transition[s[:, t - 1]], 1, generator=ds.rng).squeeze(1)
    x = torch.multinomial(ds.P[s.reshape(-1)], 1, generator=ds.rng).reshape(1, L)
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
    bayes = ds.optimal_predictions(ds[0]["input_ids"])         # (B, L, M) forward-algorithm targets
    stationary = ds.prior @ ds.P
    kl = (bayes * (bayes / stationary).log()).sum(-1)
    assert kl[:, 1:].mean() > 0.05                             # past the first token, context matters


def test_bump_emissions_are_gaussian_in_hop_distance():
    """The HMM treats every structure as its adjacency graph: bump emissions are a Gaussian in
    hop (geodesic) distance, so sigma means 'hops aliased' — not an embedding artifact."""
    ds = HMMDataset(_cfg(sigma=1.0, n=12))
    hops = np.minimum(np.arange(12), 12 - np.arange(12)).astype(float)
    expected = np.exp(-hops ** 2 / 2.0); expected /= expected.sum()
    np.testing.assert_allclose(ds.P[0].cpu().numpy(), expected, atol=1e-6)

    # tree: states are all nodes of the tree graph; hops through the tree order the aliasing.
    # DFS pre-order for n=2,k=2: 0 root, 1 (2 3) subtree, 4 (5 6) subtree.
    ds = HMMDataset(_cfg(structure="tree", structure_kwargs={"n": 2, "k": 2}))
    assert ds.n_states == 7
    Em = ds.P.cpu().numpy()
    assert Em[2, 1] > Em[2, 3] > Em[2, 5]                      # leaf 2: parent (1 hop) > sibling (2) > cousin (4)


def test_bump_and_permuted_emissions_have_uniform_per_state_entropy():
    """Both emission modes give every state the same peakiness -> even belief coverage."""
    for emission in ("bump", "permuted"):
        Em = HMMDataset(_cfg(emission=emission, sigma=1.0, n=16, M=16)).P.cpu().numpy()
        ent = -(Em * np.log(Em + 1e-12)).sum(1)
        assert ent.std() < 0.02 * ent.mean()                # near-identical entropy across states


def test_belief_references_are_float64_and_batches_float32():
    ds = HMMDataset(_cfg())
    b = ds[0]
    x = b["input_ids"][:, 1:]
    assert ds.transition.dtype == torch.float64 and ds.P.dtype == torch.float64
    assert ds.forward_algorithm(x).dtype == torch.float64
    assert ds.optimal_predictions(b["input_ids"]).dtype == torch.float32


def test_bayes_infers_the_state_instead_of_reading_it_off_the_batch():
    # the run never sees a state token, so Bayes starts at the prior predictive and sharpens
    ds = HMMDataset(_cfg(n=12, M=12, sigma=0.5))
    b = ds[0]
    assert (b["input_ids"][:, 1:] < ds.n_outcomes).all()          # symbols only
    pred = ds.optimal_predictions(b["input_ids"])
    prior_predictive = (ds.prior @ ds.P).float()
    assert torch.allclose(pred[:, 0], prior_predictive.expand(len(pred), -1), atol=1e-5)
    entropy = -(pred * (pred + 1e-12).log()).sum(-1).mean(0)
    assert entropy[-1] < entropy[0]                               # the prefix is informative


def test_directed_ring_is_the_cyclic_shift():
    """``directed=True`` turns the ring into the cycle i -> i+1: the adjacency is the shift
    matrix, and hop distance (used by the bump emission) is still the undirected ring's."""
    from src.code.data import ring, structure_distance
    A = ring(6, directed=True)
    np.testing.assert_array_equal(A, np.roll(np.eye(6), 1, axis=1))
    hops = structure_distance({"structure": "ring", "n": 6, "directed": True})
    np.testing.assert_array_equal(hops, structure_distance({"structure": "ring", "n": 6}))


def _clock_cfg(n=24, transition_floor=0.05, sigma=3.0, emission_floor=0.6, T=24, seed=0):
    """A clock: the latent advances one state per step (up to a uniform leak), weak bump emissions."""
    return _cfg(sigma=sigma, n=n, M=n, seed=seed, T=T, transition_floor=transition_floor, emission_floor=emission_floor,
                structure_kwargs={"n": n, "directed": True, "gamma": 0.8, "cooccurrence": "adjacency"})


def test_floors_mix_transition_and_emission_with_uniform():
    """``transition_floor`` / ``emission_floor`` mix each row with the uniform distribution, so no
    latent or symbol is ever ruled out: on the clock the transition is exactly (1 - e) shift + e / n."""
    n, eT, eE = 12, 0.05, 0.6
    ds = HMMDataset(_clock_cfg(n=n, transition_floor=eT, emission_floor=eE))
    shift = np.roll(np.eye(n), 1, axis=1)
    np.testing.assert_allclose(ds.transition.cpu().numpy(), (1 - eT) * shift + eT / n, atol=1e-9)
    Em = ds.P.cpu().numpy()
    np.testing.assert_allclose(Em.sum(1), np.ones(n), atol=1e-9)
    assert Em.min() >= eE / n - 1e-12
    np.testing.assert_allclose(ds.prior.cpu().numpy(), np.full(n, 1 / n), atol=1e-6)    # stationary: uniform


def test_clock_belief_spirals_out():
    """On the clock the filtered belief sharpens through the whole context while its phase advances
    one state per step: mean entropy falls at every step, and the circular mean of the belief turns
    2 pi / n per step (the spiral the moving-latent figure shows)."""
    n, T = 24, 24
    ds = HMMDataset(_clock_cfg(n=n, T=T))
    x = ds.batch(4000)["input_ids"][:, 1:]
    b = ds.forward_algorithm(x).cpu().numpy()                       # (B, T, n)
    ent = -(b * np.log(b)).sum(-1).mean(0)
    assert np.all(np.diff(ent) < 0) and ent[0] - ent[-1] > 0.4     # sharpening never stalls, ~3.1 -> ~2.6 nats
    phase = np.angle(b @ np.exp(2j * np.pi * np.arange(n) / n))     # circular mean of the belief
    turn = np.angle(np.exp(1j * np.diff(phase, axis=1))).mean()
    assert abs(turn - 2 * np.pi / n) < 0.1 * 2 * np.pi / n


def test_uniform_emissions_leave_the_belief_at_the_prior():
    """Null control: with ``emission_floor=1`` the symbols carry no information, so the filter's
    belief is the stationary prior at every position."""
    ds = HMMDataset(_clock_cfg(n=12, emission_floor=1.0))
    x = ds.batch(64)["input_ids"][:, 1:]
    b = ds.forward_algorithm(x)
    torch.testing.assert_close(b, ds.prior.expand_as(b), atol=1e-9, rtol=0)


def test_sensors_are_bumps_around_chosen_states():
    """``sensors`` lists the states the symbols are centred on: symbol k is a bump in hop distance to
    centre k, so ``n_symbols = len(sensors)``; ``sensors`` unset keeps one symbol per state."""
    n = 12
    Em_all = HMMDataset(_cfg(sigma=1.0, n=n)).P.cpu().numpy()
    Em_listed = HMMDataset(_cfg(sigma=1.0, n=n, sensors=list(range(n)))).P.cpu().numpy()
    np.testing.assert_allclose(Em_listed, Em_all, atol=1e-12)                        # all states: today's bump
    ds = HMMDataset(_cfg(sigma=0.8, n=n, sensors=[0, 4, 8]))
    Em = ds.P.cpu().numpy()
    assert Em.shape == (n, 3) and ds.n_symbols == 3
    np.testing.assert_allclose(Em.sum(1), np.ones(n), atol=1e-12)
    hops = np.minimum(np.arange(n), n - np.arange(n)).astype(float)                     # from state 0
    expected = np.exp(-hops[[0, 4, 8]] ** 2 / (2 * 0.8 ** 2)); expected /= expected.sum()
    np.testing.assert_allclose(Em[0], expected, atol=1e-9)
    np.testing.assert_allclose(Em[2, 0], Em[2, 1], atol=1e-12)                        # state 2 sits midway between centres 0 and 4
