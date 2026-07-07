"""In-process notebook training: train_inline composes the CLI config and returns the live model
plus an in-memory geometry-metric history. Marked slow (runs the real Trainer path)."""
import pytest

from src.code.train import train_inline

pytestmark = pytest.mark.slow


def test_train_inline_returns_live_model_and_geometry_history(tmp_path):
    model, dataset, evaluator, history = train_inline(
        ["runner=transformer", "setting=graph", "structure_kwargs.n=12",
         "training_kwargs.train_steps=200", "training_kwargs.eval_steps=100",
         "training_kwargs.batch_size=128", "training_kwargs.context_length=24",
         "training_kwargs.learning_rate=0.003", "model_kwargs.model_config.hidden_size=32",
         "model_kwargs.model_config.num_hidden_layers=2", "model_kwargs.model_config.num_attention_heads=2",
         "eval_kwargs.num_eval_batches=2"],
        name="nb_test", cache_dir=str(tmp_path))

    # live model, usable in-process without reloading from disk
    assert model.get_input_embeddings().weight.shape[0] == dataset.vocab_size
    # in-memory history with the geometry metrics logged across training
    assert {"step", "loss", "cka_vs_predicted", "rsa_vs_predicted"}.issubset(history.columns)
    assert len(history) >= 2
    # the live model can be re-scored through the returned evaluator
    metrics, _ = evaluator.evaluate(model)
    assert metrics["rsa_vs_predicted"] > 0.6                    # learned the circle
