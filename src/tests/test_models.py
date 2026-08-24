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
    out = model(torch.randint(0, 12, (5, 1)), labels=torch.randint(0, 12, (5,)))
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
    out = model(torch.randint(0, 12, (5, 1)), labels=torch.randint(0, 12, (5,)))
    assert out["logits"].shape == (5, 12)                  # (B, num_classes)
    assert model.get_input_embeddings().weight.shape == (13, 8)
    assert model.item_reps(torch.arange(12)).shape == (12, 16)   # penultimate activation


def test_rnn_is_a_causal_lm():
    model = RNNModel(_cfg(arity=1))
    seq = torch.randint(0, 12, (5, 7))                     # (B, T) walk
    out = model(seq, labels=seq)
    assert out["logits"].shape == (5, 7, 13)              # (B, T, vocab_size) next-token
    assert out["loss"].ndim == 0
    assert model.item_reps(torch.arange(12), start_token=12).shape == (12, 16)   # GRU hidden


def test_rnn_exposes_per_position_hidden_states():
    # hidden_states = (recurrent state,) — read at every position exactly like a transformer's
    # residual stream. Embeddings are excluded: they are input, not state, and their width differs
    # from the hidden (the eval path stacks layers into one equal-width tensor).
    model = RNNModel(_cfg(arity=1))
    out = model(torch.randint(0, 12, (5, 7)), output_hidden_states=True)
    assert [tuple(h.shape) for h in out["hidden_states"]] == [(5, 7, 16)]
    assert "hidden_states" not in model(torch.randint(0, 12, (5, 7)))


def test_registry_covers_the_ladder():
    assert set(MODEL_CLASSES) == {"linear", "mlp", "rnn", "sae"}


def test_readout_heads_are_bias_free():
    # the readout (decoder) is a pure matrix everywhere; hidden-layer biases are code-side
    from src.code.models import SAEModel
    assert MLPModel(_cfg()).mlp[-1].bias is None
    assert RNNModel(_cfg()).head.bias is None
    assert SAEModel(_cfg(hidden_size=3, init_scale=0.1)).decoder.bias is None
