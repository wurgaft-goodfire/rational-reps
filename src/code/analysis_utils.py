#########################
# imports
##########################

import os
import sys

import pandas as pd
import torch
from pyprojroot import here

# add project root to path
sys.path.append(str(here()))

# load environment variables
from dotenv import load_dotenv
load_dotenv()

from .config import Config


#########################
# Loading eval results
#########################

def load_eval_results(setting, experiment_name, source="local", entity=None, project=None):
    """Load eval results into a tidy DataFrame, from disk or from wandb.

    wandb is optional in this template, so neither path is privileged: `source`
    chooses where to read from, and both return the same wide schema — one row
    per (run, step) with a `step` column, a `config` column, and one column per
    flat metric (e.g. ``loss``).

    - ``source="local"`` (default): walk ``$CACHE_DIR/{setting}/{name}/`` and read
      the ``metrics.pt`` files written by ``eval.py`` / the online-eval callback.
      No wandb required.
    - ``source="wandb"``: pull logged history for the experiment's runs via the
      wandb API. Requires ``entity``/``project`` (or env defaults).
    """
    if source == "local":
        return _load_local(setting, experiment_name)
    if source == "wandb":
        return _load_wandb(experiment_name, entity=entity, project=project)
    raise ValueError(f"Unknown source {source!r}; expected 'local' or 'wandb'.")


def _parse_sweep_params(subdir: str) -> dict:
    """Parse Hydra's override_dirname (``key=val,key2=val2``) into a dict, keeping
    the leaf of dotted keys and coercing numeric values."""
    params = {}
    for part in subdir.split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.split(".")[-1]
        try:
            value = float(value)
        except (TypeError, ValueError):
            pass
        params[key] = value
    return params


def _read_metrics_file(path: str) -> dict:
    """Read a metrics.pt ({metrics, extras, step}) and return its flat metrics."""
    if not os.path.exists(path):
        return {}
    return torch.load(path, weights_only=False).get("metrics", {})


def _load_local(setting, experiment_name) -> pd.DataFrame:
    experiment_dir = os.path.join(os.getenv("CACHE_DIR", "./outputs"), setting, experiment_name)
    rows = []
    for subdir in sorted(os.listdir(experiment_dir)):
        subdir_path = os.path.join(experiment_dir, subdir)
        config_path = os.path.join(subdir_path, "config_resolved.yaml")
        if not os.path.isdir(subdir_path) or not os.path.exists(config_path):
            continue
        config = Config.from_yaml(config_path)
        sweep_params = _parse_sweep_params(subdir)

        checkpoints_dir = os.path.join(subdir_path, "checkpoints")
        if os.path.isdir(checkpoints_dir):
            # Trained model: one row per saved checkpoint.
            for step in sorted(int(s) for s in os.listdir(checkpoints_dir)):
                metrics = _read_metrics_file(os.path.join(checkpoints_dir, str(step), "metrics.pt"))
                rows.append({**sweep_params, "config": config, "step": step, **metrics})
        else:
            # Single-run eval (LLM / Bayesian): metrics at the run root.
            metrics = _read_metrics_file(os.path.join(subdir_path, "metrics.pt"))
            rows.append({**sweep_params, "config": config, "step": None, **metrics})

    return pd.DataFrame(rows).reset_index(drop=True)


def _load_wandb(experiment_name, entity=None, project=None) -> pd.DataFrame:
    import wandb

    project = project or os.getenv("WANDB_PROJECT", "template-lm-training")
    entity = entity or os.getenv("WANDB_ENTITY")
    path = f"{entity}/{project}" if entity else project

    rows = []
    for run in wandb.Api().runs(path, filters={"config.name": experiment_name}):
        history = run.history()
        if history.empty:
            continue
        eval_cols = [c for c in history.columns if c.startswith("eval/")]
        for _, record in history.iterrows():
            metrics = {c.removeprefix("eval/"): record[c] for c in eval_cols if pd.notna(record[c])}
            if not metrics:
                continue
            rows.append({"run_name": run.name, "config": dict(run.config),
                         "step": record.get("_step"), **metrics})

    return pd.DataFrame(rows).reset_index(drop=True)
