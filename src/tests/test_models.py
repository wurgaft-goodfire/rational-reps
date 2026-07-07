"""Unit tests for the readout-head ladder: shapes, arity, embedding table, loss."""
import numpy as np
import torch

from src.code.models import (
    ReadoutConfig, KLinearReadout, MLPReadout, RNNReadout, READOUT_CLASSES,
    train_sparse_autoencoder,
)


def _cfg(**kw):
    base = dict(vocab_size=13, num_classes=12, embed_dim=8, arity=1, hidden_size=16, num_layers=1)
    base.update(kw)
    return ReadoutConfig(**base)


def test_klinear_arity1_is_linear_readout():
    model = KLinearReadout(_cfg(arity=1))
    assert model.readout.shape == (12, 8)                 # (num_classes, embed_dim)
    out = model(torch.randint(0, 12, (5, 1)), labels=torch.randint(0, 12, (5,)))
    assert out["logits"].shape == (5, 12)
    assert out["loss"].ndim == 0


def test_klinear_arity2_is_bilinear_readout():
    model = KLinearReadout(_cfg(arity=2))
    assert model.readout.shape == (12, 8, 8)              # (num_classes, d, d)
    out = model(torch.randint(0, 12, (5, 2)))
    assert out["logits"].shape == (5, 12)
    assert "loss" not in out                               # no labels -> no loss (eval path)


def test_mlp_is_a_pointwise_readout():
    model = MLPReadout(_cfg(arity=1))
    out = model(torch.randint(0, 12, (5, 1)), labels=torch.randint(0, 12, (5,)))
    assert out["logits"].shape == (5, 12)                  # (B, num_classes)
    assert model.get_input_embeddings().weight.shape == (13, 8)
    assert model.item_reps(torch.arange(12)).shape == (12, 16)   # penultimate activation


def test_rnn_is_a_causal_lm():
    model = RNNReadout(_cfg(arity=1))
    seq = torch.randint(0, 12, (5, 7))                     # (B, T) walk
    out = model(seq, labels=seq)
    assert out["logits"].shape == (5, 7, 13)              # (B, T, vocab_size) next-token
    assert out["loss"].ndim == 0
    assert model.item_reps(torch.arange(12), start_token=12).shape == (12, 16)   # GRU hidden


def test_registry_covers_the_ladder():
    assert set(READOUT_CLASSES) == {"klinear", "mlp", "rnn"}


def test_sparse_autoencoder_superposes_five_features_as_a_pentagon():
    # over capacity (5 features, 2 dims) -> the code superposes them into a regular polygon
    W = train_sparse_autoencoder(5, d=2, sparsity=0.05, steps=2500, seed=0)
    ang = np.sort(np.arctan2(W[1], W[0]))
    gaps = np.degrees(np.diff(np.r_[ang, ang[0] + 2 * np.pi]))
    assert gaps.min() > 50.0                               # evenly spread (ideal pentagon = 72 deg)
