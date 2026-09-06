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


def test_rnn_is_a_causal_lm():
    model = RNNModel(_cfg(arity=1))
    seq = torch.randint(0, 12, (5, 7))                     # (B, T) walk
    out = model(seq, labels=seq)
    assert out["logits"].shape == (5, 7, 13)              # (B, T, vocab_size) next-token
    assert out["loss"].ndim == 0


def test_rnn_exposes_per_position_hidden_states():
    # hidden_states = (recurrent state,) — read at every position exactly like a transformer's
    # residual stream. Embeddings are excluded: they are input, not state, and their width differs
    # from the hidden (the eval path stacks layers into one equal-width tensor).
    model = RNNModel(_cfg(arity=1))
    out = model(torch.randint(0, 12, (5, 7)), output_hidden_states=True)
    assert [tuple(h.shape) for h in out["hidden_states"]] == [(5, 7, 16)]
    assert "hidden_states" not in model(torch.randint(0, 12, (5, 7)))


def test_registry_covers_the_ladder():
    assert set(MODEL_CLASSES) == {"linear", "mlp", "rnn", "sae", "transformer",
                                  "features"}


def test_readout_heads_are_bias_free():
    # the readout (decoder) is a pure matrix everywhere; hidden-layer biases are code-side
    from src.code.models import SAEModel
    assert MLPModel(_cfg()).mlp[-1].bias is None
    assert RNNModel(_cfg()).head.bias is None
    assert SAEModel(_cfg(hidden_size=3, init_scale=0.1)).decoder.bias is None


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


def test_load_model_rebuilds_a_saved_small_model_exactly(tmp_path):
    # the in-house classes reload from config + state dict without the HF loader: same
    # parameters, same logits
    from src.code.models import MODEL_CLASSES, load_model
    from omegaconf import OmegaConf
    cls = MODEL_CLASSES["linear"]
    model = cls(cls.config_class(vocab_size=20, num_classes=5, embed_dim=3, arity=1))
    model.save_pretrained(tmp_path)
    config = OmegaConf.create({"model_kwargs": {"model_type": "linear"}})
    loaded, tok = load_model(config, str(tmp_path), device="cpu")
    assert tok is None
    x = torch.randint(0, 20, (4, 2))
    with torch.no_grad():
        assert torch.allclose(loaded(x)["logits"], model.cpu()(x)["logits"])
    assert loaded.output_key == "logits"


def _ladder():
    # one small instance of every class, with the input batch each reads
    from src.code.models import SAEModel, FeatureModel, TransformerModel
    from transformers import GPTNeoXConfig
    ids1, ids2 = torch.randint(0, 12, (7, 2)), torch.randint(0, 12, (7, 3))
    seq = torch.randint(0, 12, (7, 5))
    feat = FeatureModel(_cfg(n_features=4, feature_order=2, n_top=4))
    feat.load_features((torch.rand(12, 4) < 0.5).float())
    tr = TransformerModel(GPTNeoXConfig(vocab_size=13, hidden_size=16, num_hidden_layers=2,
                                        num_attention_heads=2, intermediate_size=32,
                                        max_position_embeddings=8))
    return [(LinearModel(_cfg(arity=1)), ids1), (MLPModel(_cfg(arity=1)), ids1),
            (MLPModel(_cfg(arity=2)), ids2), (SAEModel(_cfg(hidden_size=3, init_scale=0.1)), ids1),
            (feat, ids1), (RNNModel(_cfg(arity=1)), seq), (tr, seq)]


def test_every_model_exposes_the_code_its_readout_reads():
    """One contract for the ladder: ``output_hidden_states=True`` returns a tuple of (B, T, width)
    tensors whose last entry is what the readout consumes, and ``readout_matrix`` (outcomes,
    width) applied to it at the last position reproduces the logits there. For the transformer
    that pins the last hidden state to the post-final-norm residual stream."""
    for model, ids in _ladder():
        model.eval()
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        hs = out["hidden_states"]
        assert isinstance(hs, tuple) and all(h.ndim == 3 for h in hs), type(model).__name__
        logits = out["logits"]
        last = logits[:, -1] if logits.ndim == 3 else logits
        W = model.readout_matrix
        assert W.shape == (last.shape[-1], hs[-1].shape[-1]), type(model).__name__
        assert torch.allclose(hs[-1][:, -1] @ W.T, last, atol=1e-4), type(model).__name__
        assert "hidden_states" not in model(ids)


def test_a_bilinear_readout_has_no_vector_representation():
    """The linear model at arity > 1 contracts an outer product: its hidden states are the slot
    embeddings and there is no readout matrix."""
    model = LinearModel(_cfg(arity=2))
    out = model(torch.randint(0, 12, (7, 2)), output_hidden_states=True)
    assert out["hidden_states"][0].shape == (7, 2, 8) and model.readout_matrix is None


def test_input_reps_is_gone():
    for cls in MODEL_CLASSES.values():
        assert not hasattr(cls, "input_reps")
