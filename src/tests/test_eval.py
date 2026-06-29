"""Unit tests for the eval rework: fixed-batch determinism, mode coverage, and
the collator dropping eval-only keys."""
import torch
from omegaconf import OmegaConf

from src.code.config import Config
from src.code.data import BallsAndUrnsDataset
from src.code.eval import Evaluator
from src.code.train_utils import PreBatchedDataCollator


def _config():
    cfg = OmegaConf.create({
        "setting": "balls-and-urns",
        "num_dims": 3,
        "num_tasks": "inf",
        "task_prior": "uniform",
        "random_seed": 0,
        "model_kwargs": {"model_type": "small_lm"},
        "training_kwargs": {"batch_size": 4, "context_length": 8},
        "eval_kwargs": {"num_eval_batches": 2, "output_hidden_states": False},
    })
    return Config(cfg)


class _FakeModel:
    """Returns deterministic logits over the task vocabulary."""
    output_key = "logits"

    def __init__(self, num_dims):
        self.num_dims = num_dims

    def __call__(self, input_ids, output_hidden_states=False):
        b, length = input_ids.shape
        g = torch.Generator().manual_seed(0)
        logits = torch.randn(b, length, self.num_dims, generator=g)
        out = {"logits": logits}
        if output_hidden_states:
            out["hidden_states"] = [torch.zeros(b, length, 2) for _ in range(2)]
        return out


def test_evaluate_returns_a_scalar_loss_per_mode():
    config = _config()
    dataset = BallsAndUrnsDataset(config)
    evaluator = Evaluator(config, dataset)
    metrics, extras = evaluator.evaluate(_FakeModel(config.num_dims))
    assert set(metrics) == {"loss"}
    assert all(isinstance(v, float) for v in metrics.values())
    # heavy per-example tensors are returned separately, not logged
    assert "all_loss" in extras


def test_fixed_batches_make_evaluate_deterministic():
    config = _config()
    evaluator = Evaluator(config, BallsAndUrnsDataset(config))
    model = _FakeModel(config.num_dims)
    first, _ = evaluator.evaluate(model)
    second, _ = evaluator.evaluate(model)
    assert first == second


def test_finite_task_dataset_builds_and_caches(tmp_path):
    cfg = _config().cfg
    cfg.num_tasks = 5
    cfg.data_path = str(tmp_path / "data.pt")
    dataset = BallsAndUrnsDataset(Config(cfg))
    assert dataset.num_tasks == 5
    assert dataset.__getitem__()["input_ids"].shape[0] == cfg.training_kwargs.batch_size


def test_collator_drops_eval_only_keys():
    batch = {"input_ids": torch.zeros(2, 3), "labels": torch.zeros(2, 3),
             "eval_labels": torch.zeros(2, 3, 3)}
    collated = PreBatchedDataCollator()([batch])
    assert set(collated) == {"input_ids", "labels"}
