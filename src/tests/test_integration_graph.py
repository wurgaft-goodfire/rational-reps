"""Slice-1 acceptance test: a transformer trained on ring co-occurrence converges to the
predicted circle. Marked slow — runs the real Hydra + HuggingFace Trainer path in a subprocess."""
import glob
import os

import numpy as np
import subprocess
import sys

import pytest
import torch
from pyprojroot import here

pytestmark = pytest.mark.slow


def _run(module, cache_dir, *overrides):
    cmd = [
        sys.executable, "-m", module,
        "runner=transformer", "setting=graph", "name=ci_graph", "wandb.enabled=false",
        "structure_kwargs.n=12", "training_kwargs.train_steps=200",
        "training_kwargs.eval_steps=100", "training_kwargs.save_steps_method=every_eval",
        "training_kwargs.save_online_eval=true", "training_kwargs.batch_size=128",
        "training_kwargs.context_length=24", "training_kwargs.learning_rate=0.003",
        "model_kwargs.model_config.hidden_size=32", "model_kwargs.model_config.num_hidden_layers=2",
        "model_kwargs.model_config.num_attention_heads=2", "model_kwargs.model_config.intermediate_size=64",
        "overwrite=true", "hydra.sweep.subdir=single", *overrides,
    ]
    env = {**os.environ, "CACHE_DIR": str(cache_dir), "WANDB_MODE": "disabled"}
    result = subprocess.run(cmd, cwd=str(here()), env=env, capture_output=True, text=True)
    assert result.returncode == 0, f"{module} failed:\n{result.stdout}\n{result.stderr}"
    return result


def test_ring_transformer_converges_to_predicted_circle(tmp_path):
    _run("src.code.train", tmp_path)
    ckpts = tmp_path / "graph-cooccurrence" / "single" / "checkpoints"
    saved = torch.load(ckpts / "200" / "metrics.pt", weights_only=False)
    # the saved representation (last hidden state per item) should match the closed-form circle
    from src.code.eval import cka, rsa
    from src.code.geometry import predicted_geometry
    from src.code.train import init_dataset, resolve_config
    dataset = init_dataset(resolve_config(["setting=graph", "structure_kwargs.n=12"]))
    E_pred = predicted_geometry(np.asarray(dataset.conditional, float))
    rep = saved["extras"]["representation"].numpy()
    assert rsa(rep, E_pred) > 0.7 and cka(rep, E_pred) > 0.7

    # the standalone eval path writes the same artifacts
    _run("src.code.eval", tmp_path)
    reval = torch.load(ckpts / "200" / "metrics.pt", weights_only=False)
    assert set(reval["extras"]) == {"logits", "representation", "readout"}
