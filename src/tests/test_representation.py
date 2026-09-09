"""The representation the Evaluator saves, compared with the predicted geometry outside it.

A model whose representation *is* the predicted geometry must score CKA/RSA ~ 1 against it; a shuffled
representation must score lower; a readout that reads no vector yields none.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.code.config import Config
from src.code.data import KernelDataset
from src.code.analysis_utils import cca, cka, cka_null, cka_shuffle_null, rsa
from src.code.eval import Evaluator
from src.code.geometry import factor_logcond, predicted_geometry


def _config(n=16):
    cfg = OmegaConf.create({
        "setting": "graph-cooccurrence", "structure": "ring", "structure_kwargs": {"n": n},
        "gamma": 0.8, "sample_mode": "pointwise", "random_seed": 0,
        "model_kwargs": {"model_type": "linear"},
        "training_kwargs": {"batch_size": 8, "context_length": 6},
    })
    return Config(cfg)


class _EmbModel:
    """Fake model in the ladder's contract: a fixed embedding table read by a fixed readout."""
    output_key = "logits"

    def __init__(self, weight, readout, device="cpu"):
        self._emb = nn.Embedding.from_pretrained(torch.tensor(weight, dtype=torch.float32)).to(device)
        self.readout_matrix = torch.tensor(readout, dtype=torch.float32)

    def __call__(self, input_ids, output_hidden_states=False):
        h = self._emb(input_ids[:, -1:])                                  # (B, 1, d)
        out = {"logits": h[:, -1] @ self.readout_matrix.T.to(h.device)}
        if output_hidden_states:
            out["hidden_states"] = (h,)
        return out


def test_metrics_helpers_are_invariant_and_bounded():
    X = np.random.default_rng(0).standard_normal((20, 3))
    assert np.isclose(cka(X, X), 1.0)
    assert np.isclose(rsa(X, X), 1.0)
    assert cka(X, X @ np.linalg.qr(np.random.default_rng(1).standard_normal((3, 3)))[0]) > 0.99


def test_saved_representation_of_the_predicted_geometry_scores_one_and_shuffled_lower():
    config = _config(n=16)
    dataset = KernelDataset(config)
    evaluator = Evaluator(config, dataset)
    _, E, D, _ = factor_logcond(np.asarray(dataset.conditional, float))
    E_pred = predicted_geometry(np.asarray(dataset.conditional, float))

    vocab, d = dataset.vocab_size, E.shape[1]
    weight = np.zeros((vocab, d)); weight[dataset.item_tokens.numpy()] = E
    ev = evaluator.evaluate(_EmbModel(weight, D.T, dataset.device))
    assert ev["loss"] < 1e-6                                            # the exact factorization
    assert cka(ev["hidden_states"][-1][:, 0], E_pred) > 0.99 and rsa(ev["hidden_states"][-1][:, 0], E_pred) > 0.99

    shuffled = weight.copy()
    shuffled[dataset.item_tokens.numpy()] = E[np.random.default_rng(0).permutation(16)]
    ev2 = evaluator.evaluate(_EmbModel(shuffled, D.T, dataset.device))
    assert rsa(ev2["hidden_states"][-1][:, 0], E_pred) < 0.99


def test_no_representation_is_saved_when_the_readout_reads_none():
    config = _config(n=16)
    dataset = KernelDataset(config)

    class _NoCode:                                           # no readout_matrix -> no vector representation
        output_key = "logits"
        def __call__(self, input_ids, output_hidden_states=False):
            return {"logits": torch.zeros(input_ids.shape[0], dataset.num_dims)}

    ev = Evaluator(config, dataset).evaluate(_NoCode())
    assert ev["hidden_states"] is None and ev["readout"] is None and ev["logits"].shape == (16, dataset.num_dims)


def test_cca_is_blind_to_an_invertible_map_where_cka_is_not():
    """The loss pins the code only up to X -> XA, W -> WA^-T. CCA cannot see that map at all, so it
    tests subspace agreement only; CKA does see it, which is what makes it the measure of the
    balanced-sqrt(S) split."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 4))
    A = np.diag([8.0, 1.0, 0.4, 0.05]) @ np.linalg.qr(rng.standard_normal((4, 4)))[0]

    assert cca(X, X @ A) > 0.999                       # same subspace -> invariant
    assert cka(X, X @ A) < 0.9                         # different metric -> penalised
    assert cca(X, rng.standard_normal((60, 4))) < 0.3  # an unrelated subspace scores low


def test_cca_is_not_dominated_by_directions_that_carry_no_variance():
    """An almost-null direction of the target carries almost no loss, so nothing constrains it.
    CCA weights every direction equally, so it needs a variance cutoff to stay meaningful."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 3))
    Y = np.c_[X[:, :2], 1e-3 * rng.standard_normal(60)]      # two shared dims + a near-empty one

    assert cca(X, Y, keep=0.99) > 0.99                       # the near-empty direction is cut
    assert cca(X, Y, keep=1.0) < 0.85                        # keeping it drags the score down


def test_cka_null_is_the_floor_a_score_has_to_clear_and_rises_with_fewer_rows():
    """A random code is not at zero: with only a handful of rows it already matches well, so every
    score needs its own null beside it. Four rows sit far above thirty."""
    rng = np.random.default_rng(0)
    few, many = rng.standard_normal((4, 3)), rng.standard_normal((60, 3))
    assert cka_null(few) > 3 * cka_null(many)
    assert cka_null(many) < 0.2
    assert cka_null(many, seed=1) == pytest.approx(cka_null(many, seed=1))    # deterministic


def test_cka_shuffle_null_keeps_the_spectrum_and_destroys_only_the_correspondence():
    """Shuffling the oracle's rows is the leakage control: same Gram spectrum, wrong assignment.
    A regular simplex is permutation-symmetric, so its shuffle null is exactly 1 (the geometry
    alone scores perfectly); a line's collapses, because which row sits where is the content."""
    simplex = np.eye(8) - 1 / 8                                # 8 equidistant points, centered
    assert cka_shuffle_null(simplex) == pytest.approx(1.0)

    line = np.arange(60, dtype=float)[:, None]
    assert cka_shuffle_null(line) < 0.2
    assert cka_shuffle_null(line, seed=1) == pytest.approx(cka_shuffle_null(line, seed=1))


def test_cka_matches_the_gram_form_and_scales_with_rows_not_rows_squared():
    """Linear CKA is <Kx, Ky> / (|Kx| |Ky|) over N x N Grams; the same number comes from the d x d
    cross-covariance, so a 12k-row code costs O(N d^2) instead of a gigabyte of Grams."""
    from src.code.analysis_utils import cka
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 5)); Y = X @ rng.normal(size=(5, 7)) + 0.3 * rng.normal(size=(300, 7))
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    Kx, Ky = Xc @ Xc.T, Yc @ Yc.T
    gram = (Kx * Ky).sum() / (np.linalg.norm(Kx) * np.linalg.norm(Ky))
    assert abs(cka(X, Y) - gram) < 1e-12
    import tracemalloc
    tracemalloc.start(); cka(rng.normal(size=(20000, 3)), rng.normal(size=(20000, 4))); _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    assert peak < 50e6                                           # a 20k x 20k Gram alone would be 3.2 GB
