"""Smoke sweep: quick in-process training across structures x architectures (and the distribution
/ relational settings) — not a full grid, just enough to catch anything broadly broken. Slow."""
import math

import pytest

from src.code.train import train_inline

pytestmark = pytest.mark.slow

_QUICK = ["training_kwargs.train_steps=300", "training_kwargs.eval_steps=150"]


def _graph_overrides(structure, kw):
    ov = ["setting=graph", f"structure={structure}", "~structure_kwargs"]
    return ov + [f"+structure_kwargs.{k}={v}" for k, v in kw.items()]


def _check(history, dataset, model):
    assert {"cka_vs_predicted", "rsa_vs_predicted"}.issubset(history.columns)
    assert len(history) >= 1
    rsa = history.iloc[-1]["rsa_vs_predicted"]
    assert isinstance(rsa, float) and not math.isnan(rsa)      # produced a real geometry score
    # the embedding table must cover the item tokens (distinct from outcome classes when rectangular)
    assert model.get_input_embeddings().weight.shape[0] >= len(dataset.item_tokens)


# a few structure x architecture combos (each architecture and several structures represented)
@pytest.mark.parametrize("structure,kw,arch", [
    ("ring", {"n": 12}, "transformer"),
    ("chain", {"n": 12}, "linear"),
    ("grid", {"n": 4, "m": 4}, "mlp"),
    ("complete", {"n": 8}, "rnn"),
    ("sphere", {"n": 40, "k": 6}, "linear"),
])
def test_graph_training_runs(structure, kw, arch, tmp_path):
    model, dataset, evaluator, history, _ = train_inline(
        _graph_overrides(structure, kw) + [f"runner={arch}"] + _QUICK,
        name=f"grid_{structure}_{arch}", cache_dir=str(tmp_path))
    _check(history, dataset, model)


def test_features_training_runs(tmp_path):
    model, dataset, evaluator, history, _ = train_inline(
        ["setting=feature_cube", "runner=linear"] + _QUICK,
        name="grid_features", cache_dir=str(tmp_path))
    _check(history, dataset, model)


def test_relational_training_runs(tmp_path):
    # group op: rsa vs the (degenerate) factor_tensor is low, so we only check it runs + scores
    model, dataset, evaluator, history, _ = train_inline(
        ["setting=relational", "operation=modular_add", "operation_kwargs.p=7", "runner=linear"] + _QUICK,
        name="grid_relational", cache_dir=str(tmp_path))
    _check(history, dataset, model)
