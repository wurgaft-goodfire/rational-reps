"""Unit tests for the Config wrapper and save-step scheduling."""
import random
import numpy as np
import torch
from omegaconf import OmegaConf

from src.code.config import Config, set_seed, clean_dirname


def test_clean_dirname_tidies_the_override_string():
    # the raw graph_overrides string is noisy: deletions, +/structure_kwargs. prefixes, redundant
    # setting=, and verbose factor dicts. Clean it to a readable, config-keyed run/figure dir name.
    od = ("+structure_kwargs.factors=[{structure:chain,n:15},{structure:ring,n:15}],"
          "setting=graph,structure=product,~structure_kwargs")
    assert clean_dirname(od) == "structure=product,factors=[chain(n=15),ring(n=15)]"
    assert clean_dirname("+structure_kwargs.n=12,setting=graph,structure=ring,~structure_kwargs") == \
        "structure=ring,n=12"
    assert clean_dirname("") == "single_run"           # no distinguishing overrides
    assert clean_dirname("setting=graph,~structure_kwargs") == "single_run"


def test_graph_overrides_carries_kernel_params_in_structure_kwargs():
    # The co-occurrence kernel (gamma / cooccurrence) is part of the structure spec, so it rides into
    # structure_kwargs alongside the shape params — graph_conditional reads it off the spec. A spec
    # asking for the heat kernel (e.g. a hypercube, for analogies) flows through unchanged.
    from src.code.train import graph_overrides
    ov = graph_overrides({"structure": "hypercube", "n": 3, "cooccurrence": "heat", "gamma": 0.5})
    assert "structure=hypercube" in ov
    assert {"+structure_kwargs.n=3", "+structure_kwargs.cooccurrence=heat",
            "+structure_kwargs.gamma=0.5"}.issubset(ov)


def test_set_seed_makes_all_rngs_deterministic():
    set_seed(123)
    a = (random.random(), np.random.rand(), torch.rand(1).item())
    set_seed(123)
    b = (random.random(), np.random.rand(), torch.rand(1).item())
    assert a == b


def _cfg(**training_kwargs):
    base = {"train_steps": 1000, "eval_steps": 100, "num_save_checkpoints": 4, "save_steps": None}
    base.update(training_kwargs)
    return Config(OmegaConf.create({"random_seed": 0, "training_kwargs": base}))


def test_every_eval_schedule():
    cfg = _cfg(save_steps_method="every_eval")
    cfg.process_save_steps()
    assert cfg.training_kwargs.save_steps == list(range(100, 1001, 100))


def test_end_only_schedule():
    cfg = _cfg(save_steps_method="end_only")
    cfg.process_save_steps()
    assert cfg.training_kwargs.save_steps == [1000]


def test_specific_steps_schedule():
    cfg = _cfg(save_steps_method="specific_steps", save_steps=[10, 500])
    cfg.process_save_steps()
    assert cfg.training_kwargs.save_steps == [10, 500]


def test_zero_checkpoints_does_not_crash():
    cfg = _cfg(save_steps_method="sqrt_schedule", num_save_checkpoints=0)
    cfg.process_save_steps()
    assert cfg.training_kwargs.save_steps == []


def test_get_supports_nested_paths_and_default():
    cfg = _cfg(save_steps_method="end_only")
    assert cfg.get("training_kwargs.eval_steps") == 100
    assert cfg.get("does.not.exist", default=7) == 7
