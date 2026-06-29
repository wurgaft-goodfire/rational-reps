"""End-to-end smoke test: train a tiny small_lm for a few steps (wandb off) and
then evaluate the saved checkpoint. Marked slow — it runs the real Hydra +
HuggingFace Trainer path in a subprocess."""
import os
import subprocess
import sys

import pytest
import torch
from pyprojroot import here

pytestmark = pytest.mark.slow


def _run(module, cache_dir, *overrides):
    cmd = [
        sys.executable, "-m", module,
        "runner=train_small_lm", "name=ci_smoke", "wandb.enabled=false",
        "training_kwargs.train_steps=2", "training_kwargs.eval_steps=2",
        "training_kwargs.save_steps_method=end_only", "training_kwargs.batch_size=4",
        "training_kwargs.context_length=8", "model_kwargs.model_config.hidden_size=16",
        "model_kwargs.model_config.num_hidden_layers=1", "model_kwargs.model_config.num_attention_heads=1",
        "eval_kwargs.num_eval_batches=1", "overwrite=true",
        "hydra.sweep.subdir=single", *overrides,
    ]
    env = {**os.environ, "CACHE_DIR": str(cache_dir), "WANDB_MODE": "disabled"}
    result = subprocess.run(cmd, cwd=str(here()), env=env, capture_output=True, text=True)
    assert result.returncode == 0, f"{module} failed:\n{result.stdout}\n{result.stderr}"
    return result


def test_train_then_eval_smoke(tmp_path):
    _run("src.code.train", tmp_path)
    checkpoint_dir = tmp_path / "balls-and-urns" / "ci_smoke" / "single" / "checkpoints"
    assert (checkpoint_dir / "2" / "model").is_dir(), "final checkpoint was not saved"

    _run("src.code.eval", tmp_path)
    metrics = torch.load(checkpoint_dir / "2" / "metrics.pt", weights_only=False)
    assert "loss" in metrics["metrics"]
