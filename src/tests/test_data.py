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
    assert (batch["labels"][:, -1] == batch["input_ids"][:, -1]).all()   # the outcome is the target
    assert (batch["labels"][:, :-1] == -100).all()                       # the inputs are not scored


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
            "setting": "relational", "relation": "modular_add", "relation_kwargs": op_kwargs,
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
    # the scored head slice is the latent block: eval compares z-token logits to the z posterior
    assert torch.equal(ds.label_tokens, ds.item_tokens)
    b = ds.__getitem__()
    assert (b["input_ids"][:, 1:] < ds.n_outcomes).all()          # outcomes only, never z
    assert torch.isin(b["labels"][:, 1:], ds.item_tokens.to(ds.device)).all()

    obs = torch.multinomial(ds.P[3], 60, replacement=True, generator=ds.rng).unsqueeze(0)
    ds.infer_latent = False                                       # score the next outcome
    pred = ds._posterior_predictive(obs)[0]
    assert torch.allclose(pred[0], (ds.prior @ ds.P.double()).float(), atol=1e-5)
    assert torch.allclose(pred[-1], ds.P[3], atol=2e-2)           # evidence pins the row


def test_pointwise_tiles_to_batch_size():
    """Pointwise tiles the full enumeration to ``batch_size``, each copy with freshly sampled
    outcomes: an LM averages outcome noise, a readout sees duplicated exact rows."""
    ds = KernelDataset(_config("pointwise", n=6, batch_size=20))
    batch = ds[0]
    assert batch["input_ids"].shape == (24, 2)                   # ceil(20/6) = 4 copies of 6 items
    z = batch["input_ids"][:, 0]
    assert torch.equal(z.cpu(), torch.arange(6).repeat(4))       # every copy enumerates all items
    assert torch.allclose(batch["distribution_labels"], ds.P[z])  # exact rows follow the tiling
    x = batch["input_ids"][:, 1]
    assert not torch.equal(x[:6], x[6:12]) or not torch.equal(x[:6], x[12:18])  # fresh samples
    assert ds.start_token is None and ds.vocab_size == 6         # no start token outside sequence mode



def test_gaussian_cooccurrence_is_exactly_rank_two():
    """Gaussian tuning in canonical coordinates: quadratic log-kernel on the chain, exact von
    Mises (kappa = gamma) on the ring — rank 2 after row-centering, so a 2-D code is exact.
    ``m`` puts a coarse outcome grid on the same manifold (continuous latent, few outcomes)."""
    from src.code.data import graph_conditional
    for spec in [{"structure": "chain", "n": 9}, {"structure": "ring", "n": 7},
                 {"structure": "chain", "n": 21, "m": 5}, {"structure": "ring", "n": 28, "m": 7}]:
        L = graph_conditional({**spec, "cooccurrence": "gaussian", "gamma": 1.5})
        assert L.shape == (spec["n"], spec.get("m", spec["n"]))
        Lc = L - L.mean(1, keepdims=True)
        s = np.linalg.svd(Lc, compute_uv=False)
        assert s[2] < 1e-9 * s[0]
    L = graph_conditional({"structure": "ring", "cooccurrence": "gaussian", "n": 7, "gamma": 1.5})
    expected = 1.5 * np.cos(2 * np.pi * np.arange(7) / 7)
    assert np.allclose(L[0] - L[0].mean(), expected - expected.mean(), atol=1e-9)


def test_relational_lm_is_scored_only_on_the_outcome():
    """With two input slots the stream ``[a, b, c]`` contains one unlearnable transition (b is
    independent of a). Scoring it would cost log p nats of irreducible loss and spend half the
    gradient on noise, so every input position is masked out."""
    cfg = OmegaConf.create({
        "setting": "relational", "relation": "modular_add", "relation_kwargs": {"p": 7},
        "random_seed": 0, "training_kwargs": {"batch_size": 8, "context_length": 6},
    })
    ds = RelationalDataset(Config(cfg))
    batch = ds[0]                                                # one draw: outcomes are sampled
    labels = batch["labels"]
    assert (labels[:, :2] == -100).all()                         # a and b are inputs, not targets
    assert (labels[:, 2] == batch["input_ids"][:, 2]).all()
    # exactly one scored transition per row, so a perfect model reaches zero loss
    assert int((labels[:, 1:] != -100).sum()) == len(labels)


def test_relational_pair_inputs_flatten_to_a_matrix_kernel():
    """``pair_inputs`` makes each (a, b) tuple one token: arity 1, a (p^2, p) matrix conditional
    whose plain SVD (no tensor factorization) carries the degenerate Fourier circles."""
    cfg = OmegaConf.create({
        "setting": "relational", "relation": "modular_add", "relation_kwargs": {"p": 7},
        "pair_inputs": True, "random_seed": 0, "lapse": 0.0,
        "training_kwargs": {"batch_size": 8, "context_length": 6},
    })
    ds = RelationalDataset(Config(cfg))
    assert ds.arity == 1 and ds.n_items == 49 and ds.n_outcomes == 7 and ds.bipartite
    assert np.asarray(ds.conditional).shape == (49, 7)
    b = ds[0]
    assert b["input_ids"].shape == (49, 2)                       # one pair token, then the outcome
    t = b["input_ids"][:, 0].cpu()
    assert (b["input_ids"][:, 1].cpu() == (t // 7 + t % 7) % 7 + 49).all()   # c = (a+b) mod p


def test_every_relation_builder_returns_its_tensor():
    """Each builder must hand back a (n,)*arity + (n_outcomes,) tensor whose rows are
    distributions — a builder that falls off the end returns None and fails silently."""
    from src.code.data import _RELATION_BUILDERS, relation_tensor
    small = {"modular_add": {"p": 5}, "modular_mult": {"p": 5}, "standard_add": {"m": 4},
             "standard_mult": {"m": 4}, "product": {"p": 3, "q": 2}, "maximum": {"n": 4},
             "averaging": {"grid": 2}, "set_intersection": {"nbit": 2},
             "entailment": {"depth": 1, "branch": 2},
             "symmetric_difference": {"nbit": 2}, "set_difference": {"nbit": 2},
             "between": {"n": 4},
             "common_divisors": {"n": 6}, "approx_add": {"m": 5},
             "feature_taxonomy": {"chains": 2, "depth": 1}, "pitch_transpose": {"n": 4}}
    for name in _RELATION_BUILDERS:
        T = relation_tensor({"relation": name, **small[name]})
        assert T is not None, f"{name} returned None"
        assert T.ndim >= 3 and T.shape[0] == T.shape[1]
        assert np.allclose(T.reshape(-1, T.shape[-1]).sum(1), 1.0), f"{name} rows are not normalised"


def test_input_tuples_enumerates_once_and_the_batch_tiles_it():
    """``input_tuples`` is the enumeration itself — every input the task defines, each once — and
    the pointwise batch is that enumeration tiled. Analysis wants the former, training the latter."""
    cfg = OmegaConf.create({
        "setting": "relational", "relation": "modular_add", "relation_kwargs": {"p": 6},
        "random_seed": 0, "training_kwargs": {"batch_size": 100, "context_length": 6},
    })
    ds = RelationalDataset(Config(cfg))
    tuples = ds.input_tuples()
    assert tuples.shape == (36, 2)                          # every (a, b) once, untiled
    assert torch.equal(tuples, torch.cartesian_prod(*([ds.item_tokens.to(ds.device)] * 2)))
    assert ds[0]["input_ids"][:, :2].shape == (108, 2)      # ceil(100/36) = 3 copies

    ring = KernelDataset(_config("pointwise", n=8, batch_size=8))
    assert ring.input_tuples().shape == (8, 1)              # arity 1 keeps the column


def test_relational_conditional_is_the_flat_log_conditional():
    """Every dataset hands the oracle the same object: log p(outcome | input tuple), one row per
    tuple. The relation's tensor is flattened and logged here, not at each call site."""
    from src.code.geometry import LOG_FLOOR

    cfg = OmegaConf.create({
        "setting": "relational", "relation": "modular_add", "relation_kwargs": {"p": 7},
        "random_seed": 0, "lapse": 0.0, "training_kwargs": {"batch_size": 8},
    })
    ds = RelationalDataset(Config(cfg))

    L = np.asarray(ds.conditional)
    assert ds.arity == 2 and L.shape == (49, 7)                  # rows = (a, b) pairs, not a tensor
    s = (np.arange(49) // 7 + np.arange(49) % 7) % 7
    assert np.allclose(L[np.arange(49), s], 0.0)                 # log 1 on the true outcome
    assert L.min() == LOG_FLOOR                                  # every other outcome at the floor


def test_model_inputs_prepend_the_start_token_when_the_layout_has_one():
    """One accessor feeds the model, whatever the sampler is: the input tuple with the start token
    in front when the layout has one, so code extraction needs no per-architecture fixup."""
    pointwise = KernelDataset(_config("pointwise", n=24))
    assert torch.equal(pointwise.model_inputs(), pointwise.input_tuples())   # no start token

    sequential = KernelDataset(_config("sequence", n=24))
    ids = sequential.model_inputs()
    assert ids.shape == (24, 2) and (ids[:, 0] == sequential.start_token).all()
    assert torch.equal(ids[:, 1:], sequential.input_tuples())


def test_bipartite_is_declared_by_the_structure_not_read_off_the_shapes():
    # a 4-word cube with 2 paired yes/no tokens has as many outcomes as items, yet the words are
    # hidden inputs: the structure says so, and the sequence sampler must treat them as such
    ds = _cube_dataset("sequence", m=2, infer_latent=True, structure_kwargs={"m": 2, "paired": True, "all_bundles": True})
    assert ds.n_items == ds.n_outcomes == 4
    assert ds.bipartite and not ds.connected
    b = ds.__getitem__()
    assert (b["input_ids"][:, 1:] < ds.n_outcomes).all()          # outcomes only, never z
    # a plain ring still chains: its outcomes are its nodes
    ring = KernelDataset(_config("sequence"))
    assert ring.connected and not ring.bipartite
