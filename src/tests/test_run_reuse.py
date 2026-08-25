"""A cached run is reused only while its recorded resolved config still matches: run dirs are
keyed by the *override string*, so editing a yaml default changes the task but not the path —
reuse must then retrain, never silently load artifacts trained on a different task."""
import os

from omegaconf import OmegaConf

from src.code.config import Config
from src.code.train import _run_exists


def _cfg(tmp_path, sigma, cache_dir=None):
    return Config(OmegaConf.create({
        "random_seed": 0, "overwrite": False, "emission_sigma": sigma,
        "cache_dir": cache_dir or str(tmp_path), "run_dir": str(tmp_path),
        "output_dir": str(tmp_path / "checkpoints"),
    }), make_run_dirs=True)


def test_stale_run_with_changed_config_is_wiped_not_reused(tmp_path):
    cfg = _cfg(tmp_path, sigma=2.5)
    assert not _run_exists(cfg)                                    # nothing on disk yet
    os.makedirs(os.path.join(cfg.cfg.output_dir, "100"))           # a 'finished' checkpoint
    assert _run_exists(cfg)                                        # same config -> reuse

    changed = _cfg(tmp_path, sigma=1.0)
    assert not _run_exists(changed)                                # config changed -> retrain
    assert os.listdir(changed.cfg.output_dir) == []                # ... and the stale run is wiped

    os.makedirs(os.path.join(changed.cfg.output_dir, "100"))
    assert _run_exists(changed)                                    # the new config now caches


def test_machine_local_keys_do_not_invalidate_a_run(tmp_path):
    cfg = _cfg(tmp_path, sigma=2.5)
    _run_exists(cfg)
    os.makedirs(os.path.join(cfg.cfg.output_dir, "100"))
    assert _run_exists(_cfg(tmp_path, sigma=2.5, cache_dir="/somewhere/else"))


def _cfg_eval(tmp_path, eval_steps):
    return Config(OmegaConf.create({
        "random_seed": 0, "overwrite": False, "emission_sigma": 2.5,
        "training_kwargs": {"eval_steps": eval_steps},
        "cache_dir": str(tmp_path), "run_dir": str(tmp_path),
        "output_dir": str(tmp_path / "checkpoints"),
    }), make_run_dirs=True)


def test_eval_cadence_does_not_invalidate_a_run(tmp_path):
    """Online eval reads fixed pre-sampled batches and draws no RNG, so how often it runs
    cannot change the trained weights — only the density of the history curve."""
    cfg = _cfg_eval(tmp_path, eval_steps=150)
    _run_exists(cfg)
    os.makedirs(os.path.join(cfg.cfg.output_dir, "100"))
    assert _run_exists(_cfg_eval(tmp_path, eval_steps=None))
