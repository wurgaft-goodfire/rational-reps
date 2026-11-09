"""The optimum as a run: `runner=optimum` solves the rank-d CE optimum of a feature-cube
cell and stores it as a linear model, so it is written, reused and read exactly like a trained
class. Marked slow (a real solve and the run pipeline)."""
import os
from pathlib import Path

import numpy as np
import pytest

from src.code.analysis_utils import load_run
from src.code.geometry import modes, profiled_fit
from src.code.train import resolve_config, train_inline

pytestmark = pytest.mark.slow

CELL = ["setting=superposition", "structure_kwargs.N=256", "structure_kwargs.m=8",
        "model_kwargs.model_config.embed_dim=3", "random_seed=1"]
OPTIMUM = CELL + ["runner=optimum"]


def test_optimum_runner_stores_the_rank_constrained_geometry_as_a_linear_run(tmp_path):
    model, dataset, evaluator, history, config = train_inline(OPTIMUM, cache_dir=str(tmp_path))
    run = load_run(OPTIMUM, cache_dir=str(tmp_path), project=False)

    # the saved representation and readout are the optimum's own, the saved logits their product.
    # Tolerance: a second solve of the same cell reproduces the first to ~1e-3 of scale.
    _, X = modes(dataset.info["alpha"], dataset.features, 0.0)
    H, D = profiled_fit(X, dataset.P.cpu().numpy().astype(float), 3, gtol=config.cfg.solver.gtol, max_iter=config.cfg.solver.max_iter)
    tol = 1e-3 * np.abs(H @ D.T).max()
    assert np.allclose(run["representation"], H, atol=tol), np.abs(run["representation"] - H).max()
    assert np.allclose(run["readout"], D, atol=tol), np.abs(run["readout"] - D).max()
    assert np.allclose(run["logits"], H @ D.T, atol=tol)
    assert run["step"] == 0 and run["loss"] == history["loss"].iloc[0] > 0

    # a sibling of the trained classes in the cell directory, filled once
    linear = resolve_config(CELL + ["runner=linear"], cache_dir=str(tmp_path))
    assert Path(config.run_dir).parent == Path(linear.run_dir).parent and Path(config.run_dir).name == "optimum"
    model_files = sorted(Path(config.output_dir).glob("0/model/*"))
    mtimes = {p: p.stat().st_mtime_ns for p in model_files}
    train_inline(OPTIMUM, cache_dir=str(tmp_path))
    assert {p: p.stat().st_mtime_ns for p in model_files} == mtimes

    # the solver settings are part of the run's identity: a run solved under other settings is refused
    from omegaconf import OmegaConf
    resolved = Path(config.run_dir) / "config_resolved.yaml"
    stored = OmegaConf.load(resolved)
    stored.solver.gtol = 1e-3
    OmegaConf.save(stored, resolved)
    with pytest.raises(RuntimeError, match="resolved config or task"):
        load_run(OPTIMUM, cache_dir=str(tmp_path))
