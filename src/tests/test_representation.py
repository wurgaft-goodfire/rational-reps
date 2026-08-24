"""The representation-comparison path in the Evaluator, exercised deterministically (no training).

A model whose embedding *is* the predicted geometry must score CKA/RSA ~ 1; a shuffled embedding
must score lower. Also guards the gating: datasets without a `.conditional` emit no rep metrics.
"""
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.code.config import Config
from src.code.data import KernelDataset
from src.code.eval import Evaluator, cka, rsa


def _config(n=16):
    cfg = OmegaConf.create({
        "setting": "graph-cooccurrence", "structure": "ring", "structure_kwargs": {"n": n},
        "gamma": 0.8, "sample_mode": "pointwise", "random_seed": 0,
        "model_kwargs": {"model_type": "linear"},
        "training_kwargs": {"batch_size": 8, "context_length": 6},
        "eval_kwargs": {"num_eval_batches": 2, "output_hidden_states": False},
    })
    return Config(cfg)


class _EmbModel:
    """Fake model exposing a fixed embedding table + uniform logits over outcomes."""
    output_key = "logits"

    def __init__(self, weight, num_dims):
        self._emb = nn.Embedding.from_pretrained(torch.tensor(weight, dtype=torch.float64))
        self.num_dims = num_dims

    def get_input_embeddings(self):
        return self._emb

    def __call__(self, input_ids, output_hidden_states=False):
        return {"logits": torch.zeros(input_ids.shape[0], self.num_dims)}


def test_metrics_helpers_are_invariant_and_bounded():
    X = np.random.default_rng(0).standard_normal((20, 3))
    assert np.isclose(cka(X, X), 1.0)
    assert np.isclose(rsa(X, X), 1.0)
    assert cka(X, X @ np.linalg.qr(np.random.default_rng(1).standard_normal((3, 3)))[0]) > 0.99


def test_evaluator_scores_matching_embedding_near_one_and_shuffled_lower():
    config = _config(n=16)
    dataset = KernelDataset(config)
    evaluator = Evaluator(config, dataset)
    E = evaluator.predicted_E                                # (16, k)

    vocab, d = dataset.vocab_size, E.shape[1]
    weight = np.zeros((vocab, d)); weight[dataset.item_tokens.numpy()] = E
    metrics, _ = evaluator.evaluate(_EmbModel(weight, dataset.num_dims))
    assert metrics["rsa_vs_predicted"] > 0.99
    assert metrics["cka_vs_predicted"] > 0.99

    shuffled = weight.copy()
    perm = np.random.default_rng(0).permutation(16)
    shuffled[dataset.item_tokens.numpy()] = E[perm]
    m2, _ = evaluator.evaluate(_EmbModel(shuffled, dataset.num_dims))
    assert m2["rsa_vs_predicted"] < metrics["rsa_vs_predicted"]


def test_no_rep_metrics_when_model_lacks_embedding():
    config = _config(n=16)
    dataset = KernelDataset(config)
    evaluator = Evaluator(config, dataset)

    class _NoEmb:
        output_key = "logits"
        def __call__(self, input_ids, output_hidden_states=False):
            return {"logits": torch.zeros(input_ids.shape[0], dataset.num_dims)}

    metrics, _ = evaluator.evaluate(_NoEmb())
    assert "rsa_vs_predicted" not in metrics
