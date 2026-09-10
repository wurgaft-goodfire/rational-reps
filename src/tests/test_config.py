"""Unit tests for the Config wrapper and save-step scheduling."""
import random
import numpy as np
import torch
from omegaconf import OmegaConf

from src.code.config import Config, set_seed, clean_dirname


def test_clean_dirname_tidies_the_override_string():
    # the raw graph_overrides string is noisy: deletions, +/structure_kwargs. prefixes, redundant
    # setting=, and verbose factor dicts. Clean it to a readable, config-keyed {task}/{overrides}
    # path, the *_kwargs. prefixes dropped since the keys are distinct anyway.
    od = ("+structure_kwargs.factors=[{structure:chain,n:15},{structure:ring,n:15}],"
          "setting=graph,structure=product,~structure_kwargs")
    assert clean_dirname(od) == "product/factors=[chain(n=15),ring(n=15)]"
    assert clean_dirname("+structure_kwargs.n=12,setting=graph,structure=ring,~structure_kwargs") == "ring/n=12"
    assert clean_dirname("structure=ring,model_kwargs.model_config.embed_dim=3,training_kwargs.train_steps=8000,"
                         "structure_kwargs.p=0.3,random_seed=0") == "ring/embed_dim=3,p=0.3,random_seed=0,train_steps=8000"
    assert clean_dirname("") == "default"                # no distinguishing overrides, no task
    assert clean_dirname("setting=graph,~structure_kwargs") == "default"


def test_clean_dirname_carries_the_structure_even_when_it_is_not_overridden():
    # the structure usually comes from the setting yaml, and several settings share one `setting:`
    # value — so it must key the dir, or their default runs collide.
    assert clean_dirname("setting=feature_cube", "feature_cube") == "feature_cube/default"
    assert clean_dirname("setting=hmm,emission_sigma=2.5", "ring") == "ring/emission_sigma=2.5"
    assert clean_dirname("setting=graph,structure=chain", "chain") == "chain/default"   # no duplicate
    assert clean_dirname("", None) == "default"                        # settings without a structure


def test_graph_overrides_carries_kernel_params_in_structure_kwargs():
    # The co-occurrence kernel (gamma / cooccurrence) is part of the structure spec, so it rides into
    # structure_kwargs alongside the shape params — graph_conditional reads it off the spec. A spec
    # asking for the heat kernel (e.g. a hypercube, for analogies) flows through unchanged.
    from src.code.train import graph_overrides
    ov = graph_overrides({"structure": "hypercube", "n": 3, "cooccurrence": "heat", "gamma": 0.5})
    assert "structure=hypercube" in ov
    assert {"+structure_kwargs.n=3", "+structure_kwargs.cooccurrence=heat",
            "+structure_kwargs.gamma=0.5"}.issubset(ov)


def test_graph_overrides_selects_the_setting():
    # The same spec builds a hidden-Markov run: only the setting override changes.
    from src.code.train import graph_overrides
    ov = graph_overrides({"structure": "ring", "n": 24, "directed": True, "cooccurrence": "adjacency"}, setting="hmm")
    assert ov[0] == "setting=hmm" and "+structure_kwargs.directed=True" in ov
    assert graph_overrides({"structure": "ring", "n": 24})[0] == "setting=graph"


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


def test_every_eval_without_a_cadence_saves_the_final_step():
    """The default cadence scores only the final step, so "every eval" is one save, at the end."""
    cfg = _cfg(save_steps_method="every_eval", eval_steps=None)
    cfg.process_save_steps()
    assert cfg.training_kwargs.save_steps == [1000]


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


def test_clean_dirname_names_the_relation_without_being_asked():
    """Several settings share ``setting: relational``, so the run dir must carry the relation the
    way it carries a graph's structure — injected from the resolved config, not from remembering
    to pass an override, or every relation collides on one directory."""
    from src.code.config import clean_dirname

    assert clean_dirname("model_kwargs.model_config.hidden_size=128", relation="modular_add") \
        == "modular_add/hidden_size=128"
    assert clean_dirname("relation=maximum,x=1", relation="maximum") == "maximum/x=1"
    assert clean_dirname("gamma=0.8", structure="ring") == "ring/gamma=0.8"


def test_clean_dirname_canonicalises_boolean_overrides():
    """The CLI and ``compose()`` disagree on bool casing — ``pair_inputs=true`` against
    ``pair_inputs=True`` — which gives one run two directories, so a notebook silently retrains
    what the CLI already trained. Canonicalise the value so both paths land in one place."""
    from src.code.config import clean_dirname

    assert clean_dirname("pair_inputs=true") == "pair_inputs=True"
    assert clean_dirname("pair_inputs=True") == "pair_inputs=True"
    assert clean_dirname("infer_latent=false") == "infer_latent=False"
    assert clean_dirname("name=trueish") == "name=trueish"        # only whole values, not prefixes
