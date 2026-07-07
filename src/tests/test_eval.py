"""Unit tests for the Evaluator: fixed-batch determinism, scalar loss, and the collator
dropping eval-only keys. Uses the graph dataset with a fake model."""
import torch
from omegaconf import OmegaConf

from src.code.config import Config
from src.code.data import GraphCooccurrenceDataset
from src.code.eval import Evaluator
from src.code.train_utils import PreBatchedDataCollator


def _config():
    cfg = OmegaConf.create({
        "setting": "graph-cooccurrence", "structure": "ring", "structure_kwargs": {"n": 8},
        "gamma": 0.8, "sample_mode": "pointwise", "random_seed": 0,
        "model_kwargs": {"model_type": "klinear"},
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
    dataset = GraphCooccurrenceDataset(config)
    metrics, extras = Evaluator(config, dataset).evaluate(_FakeModel(dataset.num_dims))
    # rep metrics are gated on the model exposing an embedding table; the fake model has none
    assert set(metrics) == {"loss"}
    assert all(isinstance(v, float) for v in metrics.values())
    assert "all_loss" in extras


def test_fixed_batches_make_evaluate_deterministic():
    config = _config()
    dataset = GraphCooccurrenceDataset(config)
    evaluator = Evaluator(config, dataset)
    model = _FakeModel(dataset.num_dims)
    assert evaluator.evaluate(model)[0] == evaluator.evaluate(model)[0]


def test_collator_drops_eval_only_keys():
    batch = {"input_ids": torch.zeros(2, 3), "labels": torch.zeros(2, 3),
             "eval_labels": torch.zeros(2, 3, 3)}
    collated = PreBatchedDataCollator()([batch])
    assert set(collated) == {"input_ids", "labels"}
