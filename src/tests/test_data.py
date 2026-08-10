"""Unit tests for the graph co-occurrence dataset: conditional shape, sampler modes."""
import numpy as np
import torch
from omegaconf import OmegaConf

from src.code.config import Config
from src.code.data import GraphCooccurrenceDataset, DistributionFamilyDataset, RelationalDataset


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
    ds = GraphCooccurrenceDataset(_config("pointwise", n=24))
    assert ds.conditional.shape == (24, 24)
    assert ds.arity == 1
    assert ds.num_dims == 24
    assert ds.vocab_size == 25  # + start token


def test_pointwise_batch_is_full_batch_soft_targets():
    ds = GraphCooccurrenceDataset(_config("pointwise", n=24, batch_size=8))
    batch = ds.__getitem__()
    assert batch["input_ids"].shape == (24, 1)             # full batch: every item
    assert batch["labels"].shape == (24, 24)               # soft targets = true conditional
    assert ds.optimal_predictions(batch["input_ids"]).shape == (24, 24)
    assert torch.allclose(batch["labels"].sum(1), torch.ones(24, device=batch["labels"].device))


def test_sequence_batch_shapes_and_start_token():
    ds = GraphCooccurrenceDataset(_config("sequence", n=24, batch_size=8, context_length=6))
    batch = ds.__getitem__()
    assert batch["input_ids"].shape == (8, 7)              # start + context_length
    assert (batch["input_ids"][:, 0] == 24).all()          # start token first
    assert batch["labels"].shape == (8, 7)
    assert ds.optimal_predictions(batch["input_ids"]).shape == (8, 6, 24)   # aligns with logits[:, :-1]


def test_ring_conditional_is_circulant():
    ds = GraphCooccurrenceDataset(_config("pointwise", n=12))
    L = ds.conditional
    # circulant: every row is a shift of the first (translation symmetry)
    for i in range(1, 12):
        assert np.allclose(L[i], np.roll(L[0], i), atol=1e-9)


def test_distribution_family_dataset_shapes():
    cfg = OmegaConf.create({
        "setting": "distribution-family", "family": "gaussian1d", "random_seed": 0,
        "training_kwargs": {"batch_size": 8, "context_length": 6},
    })
    ds = DistributionFamilyDataset(Config(cfg))
    assert ds.conditional.shape == (ds.n_items, ds.n_outcomes)
    assert ds.arity == 1 and ds.num_dims == ds.n_outcomes and ds.vocab_size == ds.n_items
    batch = ds.__getitem__()
    assert batch["input_ids"].shape == (ds.n_items, 1)          # full batch: every distribution
    assert batch["labels"].shape == (ds.n_items, ds.n_outcomes)  # soft targets


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
