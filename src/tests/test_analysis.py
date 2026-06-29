"""Unit test for the offline (local) eval-results loader."""
import torch
from omegaconf import OmegaConf

from src.code.analysis_utils import load_eval_results


def _write_run(root, override, steps):
    run_dir = root / "balls-and-urns" / "my_run" / override
    run_dir.mkdir(parents=True)
    OmegaConf.save(OmegaConf.create({"random_seed": 0, "name": "my_run"}),
                   run_dir / "config_resolved.yaml")
    for step in steps:
        ckpt = run_dir / "checkpoints" / str(step)
        ckpt.mkdir(parents=True)
        torch.save({"metrics": {"loss": 0.1 * step}, "extras": {}, "step": step},
                   ckpt / "metrics.pt")


def test_load_local_spreads_metrics_and_sweep_params(tmp_path, monkeypatch):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path))
    _write_run(tmp_path, "num_tasks=5", steps=[10, 20])

    df = load_eval_results("balls-and-urns", "my_run", source="local")

    assert len(df) == 2
    assert set(["num_tasks", "step", "config", "loss"]).issubset(df.columns)
    assert df["num_tasks"].tolist() == [5.0, 5.0]
    assert sorted(df["step"].tolist()) == [10, 20]
    assert df.sort_values("step")["loss"].tolist() == [1.0, 2.0]
