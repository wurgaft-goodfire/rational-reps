"""In-process notebook training: train_inline composes the CLI config and returns the live model
plus an in-memory geometry-metric history. Marked slow (runs the real Trainer path)."""
from pathlib import Path

import pytest

from src.code.train import train_inline

pytestmark = pytest.mark.slow

# A tiny transformer on a small ring — enough to learn the circle, cheap to train.
TINY = ["runner=transformer", "setting=graph", "structure_kwargs.n=12",
        "training_kwargs.train_steps=200", "training_kwargs.eval_steps=100",
        "training_kwargs.batch_size=128", "training_kwargs.context_length=24",
        "training_kwargs.learning_rate=0.003", "model_kwargs.model_config.hidden_size=32",
        "model_kwargs.model_config.num_hidden_layers=2", "model_kwargs.model_config.num_attention_heads=2",
        "eval_kwargs.num_eval_batches=2"]


def test_train_inline_returns_live_model_and_geometry_history(tmp_path):
    model, dataset, evaluator, history, config = train_inline(TINY, name="nb_test", cache_dir=str(tmp_path))

    # live model, usable in-process without reloading from disk
    assert model.get_input_embeddings().weight.shape[0] == dataset.vocab_size
    # in-memory history with the geometry metrics logged across training
    assert {"step", "loss", "cka_vs_predicted", "rsa_vs_predicted"}.issubset(history.columns)
    assert len(history) >= 2
    # the live model can be re-scored through the returned evaluator
    metrics, _ = evaluator.evaluate(model)
    assert metrics["rsa_vs_predicted"] > 0.6                    # learned the circle


def test_train_inline_is_config_keyed_and_reuses_without_retraining(tmp_path):
    m1, ds1, ev1, h1, cfg1 = train_inline(TINY, name="reuse", cache_dir=str(tmp_path))

    # run dir is config-keyed like the CLI (setting/name/override_dirname), not a hardcoded "notebook"
    assert Path(cfg1.run_dir).name != "notebook"
    ckpt = Path(cfg1.output_dir)
    model_files = sorted(ckpt.glob("*/model/*"))
    assert model_files                                          # checkpoints saved per the runner schedule
    assert (Path(cfg1.run_dir) / "history.csv").is_file()      # history persisted for reuse
    mtimes = {p: p.stat().st_mtime_ns for p in model_files}

    # second identical call: overwrite=false -> load the existing run, do NOT retrain
    m2, ds2, ev2, h2, cfg2 = train_inline(TINY, name="reuse", cache_dir=str(tmp_path))

    assert cfg2.run_dir == cfg1.run_dir
    assert {p: p.stat().st_mtime_ns for p in model_files} == mtimes   # checkpoints untouched -> reused
    assert list(h2["step"]) == list(h1["step"])                       # identical history returned
    # the loaded model is the trained final checkpoint: scores match the last logged step
    metrics2, _ = ev2.evaluate(m2)
    assert metrics2["rsa_vs_predicted"] == pytest.approx(h1["rsa_vs_predicted"].iloc[-1], abs=1e-4)
