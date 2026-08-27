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

    def __init__(self, weight, num_dims, device="cpu"):
        self._emb = nn.Embedding.from_pretrained(
            torch.tensor(weight, dtype=torch.float64)).to(device)   # where the dataset's ids live
        self.num_dims = num_dims

    def get_input_embeddings(self):
        return self._emb

    def input_reps(self, input_ids):
        return self._emb(input_ids[:, -1])                   # the item's embedding, pre-decode

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
    metrics, _ = evaluator.evaluate(_EmbModel(weight, dataset.num_dims, dataset.device))
    assert metrics["rsa_vs_predicted"] > 0.99
    assert metrics["cka_vs_predicted"] > 0.99

    shuffled = weight.copy()
    perm = np.random.default_rng(0).permutation(16)
    shuffled[dataset.item_tokens.numpy()] = E[perm]
    m2, _ = evaluator.evaluate(_EmbModel(shuffled, dataset.num_dims, dataset.device))
    assert m2["rsa_vs_predicted"] < metrics["rsa_vs_predicted"]


def test_no_rep_metrics_when_model_exposes_no_code():
    config = _config(n=16)
    dataset = KernelDataset(config)
    evaluator = Evaluator(config, dataset)

    class _NoCode:                                           # no input_reps -> no geometry to score
        output_key = "logits"
        def __call__(self, input_ids, output_hidden_states=False):
            return {"logits": torch.zeros(input_ids.shape[0], dataset.num_dims)}

    metrics, _ = evaluator.evaluate(_NoCode())
    assert "rsa_vs_predicted" not in metrics


def test_cca_is_blind_to_an_invertible_map_where_cka_is_not():
    """The loss pins the code only up to X -> XA, W -> WA^-T. CCA cannot see that map at all, so it
    tests subspace agreement only; CKA does see it, which is what makes it the measure of the
    balanced-sqrt(S) split."""
    from src.code.eval import cca

    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 4))
    A = np.diag([8.0, 1.0, 0.4, 0.05]) @ np.linalg.qr(rng.standard_normal((4, 4)))[0]

    assert cca(X, X @ A) > 0.999                       # same subspace -> invariant
    assert cka(X, X @ A) < 0.9                         # different metric -> penalised
    assert cca(X, rng.standard_normal((60, 4))) < 0.3  # an unrelated subspace scores low


def test_cca_is_not_dominated_by_directions_that_carry_no_variance():
    """An almost-null direction of the target carries almost no loss, so nothing constrains it.
    CCA weights every direction equally, so it needs a variance cutoff to stay meaningful."""
    from src.code.eval import cca

    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 3))
    Y = np.c_[X[:, :2], 1e-3 * rng.standard_normal(60)]      # two shared dims + a near-empty one

    assert cca(X, Y, keep=0.99) > 0.99                       # the near-empty direction is cut
    assert cca(X, Y, keep=1.0) < 0.85                        # keeping it drags the score down


def test_logit_r2_is_one_when_the_model_realises_the_log_conditional():
    """The premise behind 'low loss pins the geometry': a model whose readout reproduces the
    row-centred log-conditional scores 1, and CCA against E* is then 1 by construction."""
    import torch
    from src.code.analysis_utils import logit_r2, input_code
    from src.code.geometry import factor_logcond, predicted_geometry
    from src.code.eval import cca

    dataset = KernelDataset(_config(n=16))
    L = np.asarray(dataset.conditional, float)
    _, E, D, _ = factor_logcond(L)

    class _Exact:                                            # logits = E @ D = the centred target
        readout = torch.tensor(D.T, dtype=torch.float64)
        def input_reps(self, input_ids):
            return torch.tensor(E, dtype=torch.float64)

    model = _Exact()
    assert logit_r2(model, dataset) > 0.999
    assert cca(input_code(model, dataset), predicted_geometry(L)) > 0.999
