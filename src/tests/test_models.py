"""Unit tests for the model ladder: shapes, arity, embedding table, loss."""
import numpy as np
import torch

from src.code.models import (
    ModelConfig, LinearModel, MLPModel, RNNModel, MODEL_CLASSES,
)


def _cfg(**kw):
    base = dict(vocab_size=13, num_classes=12, embed_dim=8, arity=1, hidden_size=16, num_layers=1)
    base.update(kw)
    return ModelConfig(**base)


def test_linear_arity1_is_linear_readout():
    model = LinearModel(_cfg(arity=1))
    assert model.readout.shape == (12, 8)                 # (num_classes, embed_dim)
    out = model(torch.randint(0, 12, (5, 1)), distribution_labels=torch.rand(5, 12).softmax(-1))
    assert out["logits"].shape == (5, 12)
    assert out["loss"].ndim == 0


def test_linear_arity2_is_bilinear_readout():
    model = LinearModel(_cfg(arity=2))
    assert model.readout.shape == (12, 8, 8)              # (num_classes, d, d)
    out = model(torch.randint(0, 12, (5, 2)))
    assert out["logits"].shape == (5, 12)
    assert "loss" not in out                               # no labels -> no loss (eval path)


def test_mlp_is_a_pointwise_readout():
    model = MLPModel(_cfg(arity=1))
    out = model(torch.randint(0, 12, (5, 1)), distribution_labels=torch.rand(5, 12).softmax(-1))
    assert out["logits"].shape == (5, 12)                  # (B, num_classes)
    assert model.get_input_embeddings().weight.shape == (13, 8)
    assert model.input_reps(torch.arange(12).unsqueeze(1)).shape == (12, 16)   # penultimate


def test_rnn_is_a_causal_lm():
    model = RNNModel(_cfg(arity=1))
    seq = torch.randint(0, 12, (5, 7))                     # (B, T) walk
    out = model(seq, labels=seq)
    assert out["logits"].shape == (5, 7, 13)              # (B, T, vocab_size) next-token
    assert out["loss"].ndim == 0
    ids = torch.stack([torch.full((12,), 12), torch.arange(12)], dim=1)   # [start, item]
    assert model.input_reps(ids).shape == (12, 16)                        # GRU hidden


def test_rnn_exposes_per_position_hidden_states():
    # hidden_states = (recurrent state,) — read at every position exactly like a transformer's
    # residual stream. Embeddings are excluded: they are input, not state, and their width differs
    # from the hidden (the eval path stacks layers into one equal-width tensor).
    model = RNNModel(_cfg(arity=1))
    out = model(torch.randint(0, 12, (5, 7)), output_hidden_states=True)
    assert [tuple(h.shape) for h in out["hidden_states"]] == [(5, 7, 16)]
    assert "hidden_states" not in model(torch.randint(0, 12, (5, 7)))


def test_registry_covers_the_ladder():
    assert set(MODEL_CLASSES) == {"linear", "mlp", "rnn", "sae", "transformer"}


def test_readout_heads_are_bias_free():
    # the readout (decoder) is a pure matrix everywhere; hidden-layer biases are code-side
    from src.code.models import SAEModel
    assert MLPModel(_cfg()).mlp[-1].bias is None
    assert RNNModel(_cfg()).head.bias is None
    assert SAEModel(_cfg(hidden_size=3, init_scale=0.1)).decoder.bias is None


def test_a_multi_input_code_is_per_tuple_not_per_item():
    """A concat-MLP reads all input slots at once, so its pre-decode state belongs to the tuple,
    not to either item: one row per tuple, at the penultimate width."""
    model = MLPModel(_cfg(arity=2))
    assert model.mlp[0].in_features == 16                                    # slots concatenated
    assert model.input_reps(torch.randint(0, 12, (7, 2))).shape == (7, 16)


def test_a_bilinear_readout_reports_no_vector_code():
    """The linear model at arity > 1 contracts an outer product, so no vector the decoder reads
    exists. It says so with ``None`` rather than offering a stand-in."""
    assert LinearModel(_cfg(arity=2)).input_reps(torch.randint(0, 12, (7, 2))) is None
    assert LinearModel(_cfg(arity=1)).input_reps(torch.arange(12).unsqueeze(1)).shape == (12, 8)


def test_input_reps_times_the_decoder_reproduces_the_logits():
    """``input_reps`` is the pre-decode code for a batch of input tuples, so composing it with the
    decoder rows must give back the model's own logits. That pins the extraction to the thing the
    readout actually reads, rather than to whatever activation looks plausible."""
    from src.code.analysis_utils import decoder_weights

    class _DS:                                                 # the outcome block is the vocabulary
        label_tokens = torch.arange(12)

    ids = torch.randint(0, 12, (7, 2))
    for model in (MLPModel(_cfg(arity=2)), RNNModel(_cfg(arity=2, vocab_size=12))):
        code = model.input_reps(ids)
        W = torch.as_tensor(decoder_weights(model, _DS()), dtype=code.dtype)
        expected = model(ids)["logits"]
        expected = expected[:, -1] if expected.ndim == 3 else expected
        assert torch.allclose(code @ W.T, expected, atol=1e-4), type(model).__name__


def test_input_code_is_one_row_per_input_and_keeps_what_the_decoder_reads():
    """One accessor for every architecture: the pre-decode code of each model input. ``project``
    maps it onto the readout's row space, which by construction cannot change the logits — only
    the per-row constant a softmax ignores."""
    from src.code.analysis_utils import decoder_weights, input_code, readout_basis

    class _DS:
        label_tokens = torch.arange(12)
        def model_inputs(self):
            return torch.arange(12).unsqueeze(1)

    model, ds = MLPModel(_cfg(arity=1)), _DS()
    raw, proj = input_code(model, ds, project=False), input_code(model, ds)
    B = readout_basis(model, ds)
    assert raw.shape == (12, 16) and proj.shape == (12, len(B))

    W = decoder_weights(model, ds)
    lo, lp = raw @ W.T, proj @ (B @ W.T)
    assert np.allclose(lo - lo.mean(1, keepdims=True), lp - lp.mean(1, keepdims=True), atol=1e-4)


def test_reverse_kl_loss_is_zero_at_the_target_with_a_hand_checkable_value():
    # loss="reverse_kl" trains on KL(q || p*): zero exactly at logits = log p* (CE would be
    # the entropy), and at uniform q the value is (1/3) sum log((1/3)/p_x) = 0.32429.
    import torch.nn.functional as F
    from src.code.models import ModelConfig, LinearModel

    model = LinearModel(ModelConfig(vocab_size=4, num_classes=3, embed_dim=2, arity=1,
                                    loss="reverse_kl"))
    p = torch.tensor([[0.7, 0.2, 0.1]])
    assert torch.isclose(model._finish(p.log(), p)["loss"], torch.tensor(0.0), atol=1e-7)
    assert torch.isclose(model._finish(torch.zeros(1, 3), p)["loss"],
                         torch.tensor(0.32429), atol=1e-4)
    ce_model = LinearModel(ModelConfig(vocab_size=4, num_classes=3, embed_dim=2, arity=1))
    assert torch.isclose(ce_model._finish(p.log(), p)["loss"], F.cross_entropy(p.log(), p))
