"""Slice-2 acceptance test: representational convergence. Different architectures (k-linear,
MLP, RNN) trained on the same ring co-occurrence process all learn the predicted circle.
In-process training (tiny models, pointwise mode); marked slow."""
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.code.config import Config
from src.code.data import CooccurrenceDataset
from src.code.models import init_model
from src.code.eval import Evaluator

pytestmark = pytest.mark.slow


def _train_and_score(model_type, steps=500, lr=0.01, n=12):
    mc = {"vocab_size": 4, "num_classes": 4, "arity": 1, "embed_dim": 16,
          "hidden_size": 64, "num_layers": 1, "init_scale": 0.3}
    sample_mode = "sequence" if model_type == "rnn" else "pointwise"   # rnn is a causal LM
    cfg = OmegaConf.create({
        "setting": "graph-cooccurrence", "structure": "ring", "structure_kwargs": {"n": n},
        "gamma": 0.8, "sample_mode": sample_mode, "random_seed": 0,
        "model_kwargs": {"model_type": model_type, "model_config": mc},
        "training_kwargs": {"batch_size": 256, "context_length": 6},
        "eval_kwargs": {"num_eval_batches": 2, "output_hidden_states": False},
    })
    config = Config(cfg)
    dataset = CooccurrenceDataset(config)
    config.cfg.model_kwargs.model_config.vocab_size = dataset.vocab_size
    config.cfg.model_kwargs.model_config.num_classes = dataset.num_dims
    config.cfg.model_kwargs.model_config.arity = dataset.arity
    model = init_model(config)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        batch = dataset.__getitem__()
        out = model(batch["input_ids"], labels=batch["labels"])
        opt.zero_grad(); out["loss"].backward(); opt.step()
    metrics, _ = Evaluator(config, dataset).evaluate(model)
    return metrics


@pytest.mark.parametrize("model_type", ["linear", "mlp", "rnn"])
def test_architecture_converges_to_predicted_circle(model_type):
    metrics = _train_and_score(model_type)
    # every architecture recovers the circle geometry (random init ~ 0)
    assert metrics["rsa_vs_predicted"] > 0.6, f"{model_type}: {metrics}"


def _train_word_attribute(steps=600, lr=0.01):
    mc = {"vocab_size": 4, "num_classes": 4, "arity": 1, "embed_dim": 16,
          "hidden_size": 64, "num_layers": 1, "init_scale": 0.3}
    cfg = OmegaConf.create({
        "setting": "graph-cooccurrence", "structure": "word_attribute", "random_seed": 0,
        "structure_kwargs": {"N": 32, "m": 4, "p": 0.3},
        "model_kwargs": {"model_type": "linear", "model_config": mc},
        "training_kwargs": {"batch_size": 256, "context_length": 6},
        "eval_kwargs": {"num_eval_batches": 2, "output_hidden_states": False},
    })
    config = Config(cfg)
    dataset = CooccurrenceDataset(config)
    config.cfg.model_kwargs.model_config.vocab_size = dataset.vocab_size
    config.cfg.model_kwargs.model_config.num_classes = dataset.num_dims
    config.cfg.model_kwargs.model_config.arity = dataset.arity
    model = init_model(config)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        batch = dataset.__getitem__()
        out = model(batch["input_ids"], labels=batch["labels"])
        opt.zero_grad(); out["loss"].backward(); opt.step()
    return Evaluator(config, dataset).evaluate(model)[0]


def test_linear_converges_to_word_attribute_geometry():
    # the learned per-item embedding recovers the predicted natural-parameter manifold
    assert _train_word_attribute()["rsa_vs_predicted"] > 0.6


def _cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0); Kx, Ky = X @ X.T, Y @ Y.T
    return (Kx * Ky).sum() / (np.linalg.norm(Kx) * np.linalg.norm(Ky))


def test_bilinear_modular_addition_learns_a_circle():
    # order-2 group task: weight decay selects a single Fourier frequency -> a circle
    # (the tensor is degenerate, so we check the learned embedding is a circle directly).
    from src.code.data import RelationalDataset
    p = 7
    mc = {"vocab_size": 4, "num_classes": 4, "arity": 2, "embed_dim": 10, "init_scale": 0.3}
    cfg = OmegaConf.create({
        "setting": "relational", "operation": "modular_add", "operation_kwargs": {"p": p},
        "random_seed": 1, "model_kwargs": {"model_type": "linear", "model_config": mc},
        "training_kwargs": {"batch_size": 256, "context_length": 6},
        "eval_kwargs": {"num_eval_batches": 2, "output_hidden_states": False},
    })
    config = Config(cfg)
    dataset = RelationalDataset(config)
    config.cfg.model_kwargs.model_config.vocab_size = dataset.vocab_size
    config.cfg.model_kwargs.model_config.num_classes = dataset.num_dims
    config.cfg.model_kwargs.model_config.arity = dataset.arity
    model = init_model(config)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=3.0)
    model.train()
    for _ in range(6000):
        batch = dataset.__getitem__()
        out = model(batch["input_ids"], labels=batch["labels"])
        opt.zero_grad(); out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # stabilize the high-wd regime
        opt.step()
    emb = model.get_input_embeddings().weight.detach().cpu().numpy()[:p]
    embc = emb - emb.mean(0)
    nn = np.arange(p)
    best = max(_cka(embc, np.stack([np.cos(2 * np.pi * k * nn / p), np.sin(2 * np.pi * k * nn / p)], 1))
               for k in range(1, p // 2 + 1))
    assert best > 0.8, f"learned embedding is not a single-frequency circle (max CKA {best:.3f})"
