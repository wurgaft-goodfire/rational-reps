"""A cached run is reused only while its recorded resolved config still matches: run dirs are
keyed by the *override string*, so editing a yaml default changes the task but not the path —
reuse must then retrain, never silently load artifacts trained on a different task."""
import os
from dataclasses import dataclass

import numpy as np
from omegaconf import OmegaConf

from src.code.config import Config
from src.code.train import _run_exists


@dataclass
class _DS:
    """The only thing reuse needs from a dataset: the task tensor it will train on."""
    conditional: np.ndarray

    @property
    def P(self):
        return self.conditional


def _cfg(tmp_path, sigma, cache_dir=None):
    return Config(OmegaConf.create({
        "random_seed": 0, "overwrite": False, "emission_sigma": sigma,
        "cache_dir": cache_dir or str(tmp_path), "run_dir": str(tmp_path),
        "output_dir": str(tmp_path / "checkpoints"),
    }), make_run_dirs=True)


def test_stale_run_with_changed_config_is_wiped_not_reused(tmp_path):
    cfg = _cfg(tmp_path, sigma=2.5)
    assert not _run_exists(cfg, _DS(np.eye(3)))                                    # nothing on disk yet
    os.makedirs(os.path.join(cfg.cfg.output_dir, "100"))           # a 'finished' checkpoint
    assert _run_exists(cfg, _DS(np.eye(3)))                                        # same config -> reuse

    changed = _cfg(tmp_path, sigma=1.0)
    assert not _run_exists(changed, _DS(np.eye(3)))                                # config changed -> retrain
    assert os.listdir(changed.cfg.output_dir) == []                # ... and the stale run is wiped

    os.makedirs(os.path.join(changed.cfg.output_dir, "100"))
    assert _run_exists(changed, _DS(np.eye(3)))                                    # the new config now caches


def test_machine_local_keys_do_not_invalidate_a_run(tmp_path):
    cfg = _cfg(tmp_path, sigma=2.5)
    _run_exists(cfg, _DS(np.eye(3)))
    os.makedirs(os.path.join(cfg.cfg.output_dir, "100"))
    assert _run_exists(_cfg(tmp_path, sigma=2.5, cache_dir="/somewhere/else"), _DS(np.eye(3)))


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
    _run_exists(cfg, _DS(np.eye(3)))
    os.makedirs(os.path.join(cfg.cfg.output_dir, "100"))
    assert _run_exists(_cfg_eval(tmp_path, eval_steps=None), _DS(np.eye(3)))


def test_a_changed_task_tensor_invalidates_a_run(tmp_path):
    """The config string cannot see a code change to the task builder: a run whose conditional
    differs from the one on disk must retrain, not load weights fit to the old target."""
    cfg = _cfg(tmp_path, sigma=2.5)
    assert not _run_exists(cfg, _DS(np.eye(3)))
    os.makedirs(os.path.join(cfg.cfg.output_dir, "100"))           # a 'finished' checkpoint
    assert _run_exists(cfg, _DS(np.eye(3)))                        # same target -> reuse

    assert not _run_exists(cfg, _DS(np.ones((3, 3)) / 3))          # builder changed -> retrain
    assert os.listdir(cfg.cfg.output_dir) == []                    # ... and the stale run is wiped


def test_the_digest_tracks_the_training_target_not_the_oracle_s_view():
    """Runs train on the outcome distribution ``P``; ``conditional`` is the log-space object the
    geometry oracle factors. Changing how the oracle reads the task must not invalidate weights
    fit to an identical target."""
    from src.code.data import kernel_hash

    @dataclass
    class _DS2:
        conditional: np.ndarray
        P: np.ndarray

    P = np.array([[0.5, 0.5], [0.25, 0.75]])
    same_task_new_oracle = _DS2(np.log(P) * 3.0, P)          # oracle rescaled, target untouched
    assert kernel_hash(_DS2(np.log(P), P)) == kernel_hash(same_task_new_oracle)
    assert kernel_hash(_DS2(np.log(P), P)) != kernel_hash(_DS2(np.log(P), P[::-1]))
