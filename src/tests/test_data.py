"""Unit tests for the graph co-occurrence dataset: conditional shape, sampler modes."""
import numpy as np
import torch
from omegaconf import OmegaConf

from src.code.config import Config
from src.code.data import KernelDataset, RelationalDataset


def _config(sample_mode, n=24, batch_size=8, context_length=6):
    cfg = OmegaConf.create({
        "setting": "graph-cooccurrence",
        "structure": "ring",
        "structure_kwargs": {"n": n},
        "gamma": 0.8,
        "sample_mode": sample_mode,
        "random_seed": 0,
        "training_kwargs": {"batch_size": batch_size, "context_length": context_length},
    })
    return Config(cfg)


def test_conditional_is_square_and_arity_one():
    ds = KernelDataset(_config("pointwise", n=24))
    assert ds.conditional.shape == (24, 24)
    assert ds.arity == 1
    assert ds.num_dims == 24
    assert ds.vocab_size == 24  # pointwise: no start token


def test_pointwise_batch_is_full_batch_soft_targets():
    ds = KernelDataset(_config("pointwise", n=24, batch_size=8))
    batch = ds.__getitem__()
    assert batch["input_ids"].shape == (24, 1)             # full batch: every item
    assert batch["labels"].shape == (24, 24)               # soft targets = true conditional
    assert ds.optimal_predictions(batch["input_ids"]).shape == (24, 24)
    assert torch.allclose(batch["labels"].sum(1), torch.ones(24, device=batch["labels"].device))


def test_sequence_batch_shapes_and_start_token():
    ds = KernelDataset(_config("tokens", n=24, batch_size=8, context_length=6))
    batch = ds.__getitem__()
    assert batch["input_ids"].shape == (8, 7)              # start + context_length
    assert (batch["input_ids"][:, 0] == 24).all()          # start token first
    assert batch["labels"].shape == (8, 7)
    assert ds.optimal_predictions(batch["input_ids"]).shape == (8, 6, 24)   # aligns with logits[:, :-1]


def test_ring_conditional_is_circulant():
    ds = KernelDataset(_config("pointwise", n=12))
    L = ds.conditional
    # circulant: every row is a shift of the first (translation symmetry)
    for i in range(1, 12):
        assert np.allclose(L[i], np.roll(L[0], i), atol=1e-9)


def test_relational_dataset_infers_arity_and_shapes():
    for arity, op_kwargs in [(2, {"p": 7}), (3, {"p": 5, "arity": 3})]:
        cfg = OmegaConf.create({
            "setting": "relational", "operation": "modular_add", "operation_kwargs": op_kwargs,
            "random_seed": 0, "training_kwargs": {"batch_size": 8, "context_length": 6},
        })
        ds = RelationalDataset(Config(cfg))
        assert ds.arity == arity
        batch = ds.__getitem__()
        assert batch["input_ids"].shape == (ds.n_items ** arity, arity)   # full batch: all tuples
        assert batch["labels"].shape == (ds.n_items ** arity, ds.n_outcomes)  # soft targets
        # modular addition is deterministic: argmax(c | a,b,...) == (sum of inputs) mod p
        s = batch["input_ids"].sum(1) % ds.n_items
        assert (batch["labels"].argmax(1) == s).all()


def _wa_dataset(mode, n=16, m=4, context_length=4, **extra):
    from omegaconf import OmegaConf
    return KernelDataset(Config(OmegaConf.create({
        "setting": "graph-cooccurrence", "structure": "word_attribute",
        "structure_kwargs": {"N": n, "m": m}, "random_seed": 0, "sample_mode": mode,
        "training_kwargs": {"batch_size": 8, "context_length": context_length}, **extra})))


def test_tokens_exist_only_when_the_sampler_emits_them():
    # an unchainable kernel emits outcome tokens only in sequence mode, and never a start token
    point, tokens = _wa_dataset("pointwise"), _wa_dataset("tokens")
    assert not point.chained and not tokens.chained
    assert point.start_token is None and point.vocab_size == 16   # words only
    assert tokens.start_token is None and tokens.vocab_size == 20  # words + attributes
    assert tokens.label_tokens.tolist() == list(range(16, 20))


def test_chainable_kernel_walks_from_a_start_token():
    ds = KernelDataset(_config("tokens", n=12, batch_size=8, context_length=5))
    assert ds.chained and ds.start_token == 12 and ds.vocab_size == 13
    b = ds.__getitem__()
    assert (b["input_ids"][:, 0] == ds.start_token).all()
    assert b["input_ids"].shape == (8, 6)


def test_latent_input_is_inferred_from_the_prior_not_read_off_the_batch():
    # z is hidden: Bayes starts at the prior predictive and converges on the true row
    ds = _wa_dataset("tokens", n=8, m=5, latent=True)
    assert ds.latent and ds.start_token == ds.n_outcomes          # z is never a token
    z = 3
    obs = torch.multinomial(ds.P[z], 60, replacement=True, generator=ds.rng).unsqueeze(0)
    pred = ds._posterior_predictive(obs)[0]
    prior_predictive = (ds.prior @ ds.P.double()).float()
    assert torch.allclose(pred[0], prior_predictive, atol=1e-5)   # nothing seen yet
    assert torch.allclose(pred[-1], ds.P[z], atol=2e-2)           # evidence pins the row
    entropy = -(pred * (pred + 1e-12).log()).sum(-1)
    assert entropy[-1] < entropy[0]                               # inference sharpens
