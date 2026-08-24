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


def test_pointwise_batch_pairs_every_item_with_a_draw_and_its_exact_row():
    ds = KernelDataset(_config("pointwise", n=24, batch_size=8))
    batch = ds.__getitem__()
    assert batch["input_ids"].shape == (24, 2)             # full batch: every item, then a draw
    assert batch["labels"].shape == (24, 2)                # the LM predicts the outcome token
    assert batch["distribution_labels"].shape == (24, 24)  # readouts get the exact conditional
    assert torch.allclose(batch["distribution_labels"], ds.P)
    assert ds.optimal_predictions(batch["input_ids"]).shape == (24, 24)
    assert (batch["labels"] == batch["input_ids"]).all()   # the LM target is the token stream


def test_sequence_batch_shapes_and_start_token():
    ds = KernelDataset(_config("sequence", n=24, batch_size=8, context_length=6))
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
        # full batch: every tuple, then its sampled outcome
        assert batch["input_ids"].shape == (ds.n_items ** arity, arity + 1)
        assert batch["distribution_labels"].shape == (ds.n_items ** arity, ds.n_outcomes)
        # modular addition is deterministic: argmax(c | a,b,...) == (sum of inputs) mod p
        inputs = batch["input_ids"][:, :arity]
        assert (batch["distribution_labels"].argmax(1) == inputs.sum(1) % ds.n_items).all()


def _cube_dataset(mode, n=16, m=4, context_length=4, cls=KernelDataset, **extra):
    from omegaconf import OmegaConf
    return cls(Config(OmegaConf.create({
        "setting": "graph-cooccurrence", "structure": "feature_cube",
        "structure_kwargs": {"N": n, "m": m}, "random_seed": 0, "sample_mode": mode,
        "training_kwargs": {"batch_size": 8, "context_length": context_length}, **extra})))


def test_tokens_exist_only_when_the_sampler_emits_them():
    # a bipartite kernel names its outcomes; only a hidden input drops out of the vocabulary
    point, seq = _cube_dataset("pointwise"), _cube_dataset("sequence")
    assert point.bipartite and seq.bipartite
    assert point.start_token is None and point.vocab_size == 20   # inputs + outcome block
    assert point.label_tokens.tolist() == list(range(16, 20))
    assert seq.start_token == 4 and seq.vocab_size == 5           # a hidden input is not a token


def test_chainable_kernel_walks_from_a_start_token():
    ds = KernelDataset(_config("sequence", n=12, batch_size=8, context_length=5))
    assert ds.connected and ds.start_token == 12 and ds.vocab_size == 13
    b = ds.__getitem__()
    assert (b["input_ids"][:, 0] == ds.start_token).all()
    assert b["input_ids"].shape == (8, 6)


def test_hidden_input_is_inferred_from_the_prior_not_read_off_the_batch():
    # z -> x_{1:t} with z latent: it leaves the stream and Bayes infers it from the prefix
    ds = _cube_dataset("sequence", n=8, m=5, infer_latent=True)
    assert ds.bipartite and ds.start_token == ds.n_outcomes and ds.num_dims == ds.n_items
    b = ds.__getitem__()
    assert (b["input_ids"][:, 1:] < ds.n_outcomes).all()          # outcomes only, never z
    assert torch.isin(b["labels"][:, 1:], ds.item_tokens.to(ds.device)).all()

    obs = torch.multinomial(ds.P[3], 60, replacement=True, generator=ds.rng).unsqueeze(0)
    ds.infer_latent = False                                       # score the next outcome
    pred = ds._posterior_predictive(obs)[0]
    assert torch.allclose(pred[0], (ds.prior @ ds.P.double()).float(), atol=1e-5)
    assert torch.allclose(pred[-1], ds.P[3], atol=2e-2)           # evidence pins the row


