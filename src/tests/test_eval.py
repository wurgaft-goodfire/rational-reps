"""Unit tests for the Evaluator: fixed-batch determinism, scalar loss, and the collator
dropping eval-only keys. Uses the graph dataset with a fake model."""
import numpy as np
import torch
from omegaconf import OmegaConf

from src.code.config import Config
from src.code.data import KernelDataset
from src.code.eval import Evaluator, cca
from src.code.train_utils import PreBatchedDataCollator


def _config():
    cfg = OmegaConf.create({
        "setting": "graph-cooccurrence", "structure": "ring", "structure_kwargs": {"n": 8},
        "gamma": 0.8, "sample_mode": "pointwise", "random_seed": 0,
        "model_kwargs": {"model_type": "linear"},
        "training_kwargs": {"batch_size": 4, "context_length": 8},
        "eval_kwargs": {"num_eval_batches": 2, "output_hidden_states": False},
    })
    return Config(cfg)


class _FakeModel:
    """Deterministic logits over the outcome classes; no embedding table."""
    output_key = "logits"

    def __init__(self, num_dims):
        self.num_dims = num_dims

    def __call__(self, input_ids, output_hidden_states=False):
        b, length = input_ids.shape
        g = torch.Generator().manual_seed(0)
        return {"logits": torch.randn(b, length, self.num_dims, generator=g)}


def test_evaluate_returns_only_loss_when_model_has_no_embedding():
    config = _config()
    dataset = KernelDataset(config)
    metrics, extras = Evaluator(config, dataset).evaluate(_FakeModel(dataset.num_dims))
    # rep metrics are gated on the model exposing an embedding table; the fake model has none
    assert set(metrics) == {"loss"}
    assert all(isinstance(v, float) for v in metrics.values())
    assert "all_loss" in extras


def test_fixed_batches_make_evaluate_deterministic():
    config = _config()
    dataset = KernelDataset(config)
    evaluator = Evaluator(config, dataset)
    model = _FakeModel(dataset.num_dims)
    assert evaluator.evaluate(model)[0] == evaluator.evaluate(model)[0]


def test_collator_unwraps_the_prebatched_dict():
    batch = {"input_ids": torch.zeros(2, 3), "labels": torch.zeros(2, 3)}
    assert PreBatchedDataCollator()([batch]) is batch


def test_cca_ignores_float32_scale_junk_directions():
    # features computed in float32 with an exact affine constraint (e.g. probabilities summing
    # to 1) carry a junk singular direction at ~1e-6 of the top; the basis tolerance must reject
    # it, or the mean canonical correlation is silently dragged down by a noise dimension
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((500, 7))
    W = np.exp(np.concatenate([Z, np.zeros((500, 1))], 1))
    W = W / W.sum(1, keepdims=True)                                # exact rank 7
    H = np.tanh(Z @ rng.standard_normal((7, 32))) + 0.05 * rng.standard_normal((500, 32))
    s0 = np.linalg.svd(W - W.mean(0), compute_uv=False)[0]
    u = rng.standard_normal(500); u /= np.linalg.norm(u)
    v = rng.standard_normal(8); v /= np.linalg.norm(v)
    W_noisy = W + 1.5e-6 * s0 * np.outer(u, v)                     # junk direction at float32 scale
    assert abs(cca(H, W) - cca(H, W_noisy)) < 0.02
