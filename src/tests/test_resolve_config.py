"""A notebook that trains nothing still gets a config-keyed figure directory: `resolve_config`
composes the same run_dir `train_inline` would, so `figure_path` is the one path rule in the repo.
The key is the *override* string, so a parameter left un-overridden (a swept one) stays out of it."""
import os
from pathlib import Path

from src.code.plot_utils import figure_path
from src.code.train import resolve_config

FEATURE_CUBE = ["setting=feature_cube", "structure_kwargs.N=256", "structure_kwargs.m=8",
                "structure_kwargs.decay=0.9", "structure_kwargs.scale=8.0"]


def test_figure_dir_is_keyed_by_the_overrides(tmp_path):
    config = resolve_config(FEATURE_CUBE, cache_dir=str(tmp_path))
    path = Path(figure_path(config, "fig1_excess_sweep"))

    # {cache_dir}/{setting}/{structure}/{overrides}/figures/ — setting and structure come from the yaml
    assert path == tmp_path / "graph-cooccurrence" / "feature_cube" / "N=256,decay=0.9,m=8,scale=8.0" / \
        "figures" / "fig1_excess_sweep.png"


def test_changing_a_parameter_separates_the_figures(tmp_path):
    a = figure_path(resolve_config(FEATURE_CUBE, cache_dir=str(tmp_path)), "fig1")
    b = figure_path(resolve_config([o.replace("N=256", "N=512") for o in FEATURE_CUBE],
                                   cache_dir=str(tmp_path)), "fig1")
    assert a != b


def test_a_swept_parameter_is_absent_from_the_key(tmp_path):
    # p is the sweep axis: the notebook never overrides it, so it cannot enter the directory name
    config = resolve_config(FEATURE_CUBE, cache_dir=str(tmp_path))
    assert "p=" not in Path(figure_path(config, "fig1")).parent.parent.name


def test_same_overrides_give_the_run_dir_train_inline_would_use(tmp_path):
    # the path rule is shared: resolve_config is exactly train_inline's path half
    overrides = FEATURE_CUBE + ["runner=linear"]
    config = resolve_config(overrides, cache_dir=str(tmp_path))
    assert Path(config.cfg.run_dir) == tmp_path / "graph-cooccurrence" / "feature_cube" / \
        "N=256,decay=0.9,m=8,scale=8.0" / "linear"


def test_resolving_a_config_creates_no_run_directories(tmp_path):
    resolve_config(FEATURE_CUBE, cache_dir=str(tmp_path))
    assert not tmp_path.exists() or list(tmp_path.iterdir()) == []


def test_cache_dir_defaults_to_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "scratch"))
    config = resolve_config(FEATURE_CUBE)
    assert Path(figure_path(config, "fig1")).is_relative_to(tmp_path / "scratch")


def test_structure_distinguishes_settings_that_share_a_setting_name(tmp_path):
    # conf/setting/{graph,feature_cube,hypercube}.yaml all resolve to setting=graph-cooccurrence, so
    # the *structure* is what separates them. It comes from the yaml, not the override string, so the
    # key must carry it whether or not the caller overrode it.
    dirs = {s: figure_path(resolve_config([f"setting={s}"], cache_dir=str(tmp_path)), "f")
            for s in ("feature_cube", "graph", "hypercube")}
    assert len(set(dirs.values())) == 3
    assert "/feature_cube/default/" in dirs["feature_cube"] and "/ring/default/" in dirs["graph"]


def test_evaluation_settings_do_not_invalidate_a_trained_run():
    # eval draws no RNG and touches no parameter, so changing how a run is scored must not mark
    # its weights stale (which would silently retrain every cached run)
    from src.code.config import run_key
    from src.code.train import resolve_config
    base = resolve_config(["setting=graph", "structure_kwargs.n=8"])
    other = resolve_config(["setting=graph", "structure_kwargs.n=8",
                            "+eval_kwargs.output_hidden_states=true"])
    key = lambda c: run_key(c.cfg, base.cfg.run_key_exclude)
    assert key(base) == key(other)
    changed = resolve_config(["setting=graph", "structure_kwargs.n=9"])
    assert key(base) != key(changed)


def test_only_keys_that_reach_the_weights_are_in_the_run_key():
    # conf/config.yaml's run_key_exclude: logging, checkpoint cadence, compilation and the artifact
    # names never touch a parameter, so editing them must not wipe a cached run; the
    # optimizer-facing keys and the seed must
    from src.code.config import run_key
    from src.code.train import resolve_config
    base = resolve_config(["setting=graph"])
    key = lambda c: run_key(c.cfg, base.cfg.run_key_exclude)
    for override in ("training_kwargs.logging_steps=7", "training_kwargs.save_steps_method=every_eval",
                     "training_kwargs.num_save_checkpoints=3", "training_kwargs.save_online_eval=true",
                     "training_kwargs.compile=true", "training_kwargs.eval_steps=50", "eval_file=other.npz",
                     "overwrite=true"):
        assert key(base) == key(resolve_config(["setting=graph", override])), override
    for override in ("training_kwargs.learning_rate=0.5", "training_kwargs.train_steps=7",
                     "training_kwargs.weight_decay=0.1", "random_seed=3",
                     "model_kwargs.model_config.hidden_size=8"):
        assert key(base) != key(resolve_config(["setting=graph", override])), override


def test_a_repeated_override_keys_the_run_like_the_cli_last_value_wins(tmp_path):
    """The CLI dedups a key given twice (last wins) before naming the run dir; a notebook that appends
    per-world overrides to a shared list must land on the same directory."""
    base = ["setting=graph", "structure=ring", "training_kwargs.train_steps=30000",
            "model_kwargs.model_config.num_hidden_layers=1"]
    extra = ["model_kwargs.model_config.num_hidden_layers=2", "training_kwargs.train_steps=60000"]
    twice = resolve_config(base + extra, cache_dir=str(tmp_path))
    once = resolve_config(base[:2] + extra, cache_dir=str(tmp_path))
    assert twice.run_dir == once.run_dir
    assert twice.model_kwargs.model_config.num_hidden_layers == 2 and twice.training_kwargs.train_steps == 60000
