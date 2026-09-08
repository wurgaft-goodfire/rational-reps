import hashlib
import os
import random
import re
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf

def run_key(cfg):
    """The config content that identifies a run's trained weights: the resolved config minus
    machine-local bookkeeping. Evaluation draws no RNG and touches no parameter, so its cadence
    and settings are excluded too."""
    c = OmegaConf.to_container(cfg, resolve=True)
    for k in ("cache_dir", "run_dir", "output_dir", "data_path", "eval_file", "overwrite", "wandb",
              "eval_kwargs", "eval_cache_dir"):
        c.pop(k, None)
    c.get("training_kwargs", {}).pop("eval_steps", None)
    return c


def clean_dirname(override_dirname, structure=None, relation=None):
    """Turn Hydra's raw ``override_dirname`` into a tidy, config-keyed run/figure dir name: drops
    bookkeeping noise, compresses factor dicts, puts ``structure`` first, hashes overlong names.
    ``structure`` / ``relation`` are the resolved values: several settings share one ``setting:``,
    so the name carries the task even when it came from the yaml rather than an override.
    Empty -> ``default_run``."""
    # split on top-level commas only — factor values like [{structure:ring,n:15},...] carry their own
    items, depth, cur = [], 0, ""
    for ch in str(override_dirname):
        depth += (ch in "[{(") - (ch in "]})")
        if ch == "," and depth == 0:
            items.append(cur); cur = ""
        else:
            cur += ch
    items.append(cur)

    parts = []
    for item in items:
        if not item or item.startswith("~") or item.startswith("setting="):
            continue
        item = item.lstrip("+").replace("structure_kwargs.", "")
        # the CLI and compose() disagree on bool casing (pair_inputs=true vs =True), which would
        # give one run two directories; canonicalise so either path resolves to the same one
        key, _, val = item.partition("=")
        if val.lower() in ("true", "false"):
            item = f"{key}={val.lower() == 'true'}"
        parts.append(item)
    for _key, _val in (("structure", structure), ("relation", relation)):
        if _val is not None and not any(p.startswith(f"{_key}=") for p in parts):
            parts.append(f"{_key}={_val}")
    if not parts:
        return "default_run"
    parts.sort(key=lambda s: (not s.startswith(("structure=", "relation=")), s))   # the task first
    name = re.sub(r"\{structure:([^,}]+)(?:,([^}]*))?\}",
                  lambda m: m.group(1) + (f"({m.group(2)})" if m.group(2) else ""), ",".join(parts))
    name = name.replace(":", "=")                                     # factor kwargs n:15 -> n=15
    if len(name) > 200:                                              # guard the 255-byte name limit
        name = f"{name[:180]}-{hashlib.md5(name.encode()).hexdigest()[:8]}"
    return name


# Used by conf/config.yaml's run/sweep dir templates so the CLI writes the same tidy paths that
# train_inline builds. replace=True keeps re-imports (tests, reloads) from raising.
OmegaConf.register_new_resolver("clean_dirname", clean_dirname, replace=True)


def set_seed(seed: int):
    """Seed all RNGs for reproducibility (python, numpy, torch, cuda)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Config:
    def __init__(self, cfg: DictConfig, make_run_dirs=False):
        """
        Initialize configuration from Hydra DictConfig.
        """
        # Store the resolved config
        self.cfg = cfg

        # Seed all RNGs and set the torch generator
        set_seed(cfg.random_seed)
        self.rng = torch.Generator()
        self.rng.manual_seed(cfg.random_seed)

        # Create directories if requested
        if make_run_dirs:
            self._make_directories()

    def _make_directories(self):
        """Create necessary directories based on config paths."""
        os.makedirs(self.cfg.run_dir, exist_ok=True)
        os.makedirs(self.cfg.output_dir, exist_ok=True)

    def process_save_steps(self):
        """Process save_steps configuration with smart scheduling."""
        training_kwargs = self.cfg.training_kwargs
        save_steps_method = training_kwargs.get("save_steps_method", None)
        if save_steps_method in [None, "every_eval"]:
            # eval_steps=None scores the final step only, so that is the one step worth saving.
            every = training_kwargs.eval_steps or training_kwargs.train_steps
            training_kwargs.save_steps = np.arange(every, training_kwargs.train_steps + 1, every, dtype=int).tolist()

        elif save_steps_method == "linear_schedule":
            num_checkpoints = training_kwargs.num_save_checkpoints
            if num_checkpoints == 0:
                training_kwargs.save_steps = []
            else:
                # Generate `num_checkpoints` evenly spaced points in linear space
                train_steps = training_kwargs.train_steps
                step_size = train_steps // num_checkpoints
                training_kwargs.save_steps = list(range(step_size, train_steps + 1, step_size))

        elif save_steps_method == "sqrt_schedule":
            num_checkpoints = training_kwargs.num_save_checkpoints
            if num_checkpoints == 0:
                training_kwargs.save_steps = []
            else:
                # Generate `num_checkpoints` evenly spaced points in square root space
                sqrt_space = np.linspace(0, np.sqrt(training_kwargs.train_steps), num_checkpoints)
                training_kwargs.save_steps = [int(step) for step in (sqrt_space**2)]

        elif save_steps_method == "specific_steps":
            # Use the save_steps list specified directly in the config
            training_kwargs.save_steps = list(training_kwargs.save_steps) if training_kwargs.save_steps is not None else []

        elif save_steps_method == "end_only":
            training_kwargs.save_steps = [training_kwargs.train_steps]

    def __getattr__(self, name):
        """Allow direct access to config parameters."""
        if hasattr(self.cfg, name):
            return getattr(self.cfg, name)
        raise AttributeError(f"Config has no attribute '{name}'")

    def get(self, name, default=None):
        """Like dict.get(), with support for nested paths (e.g. 'training_kwargs.batch_size')."""
        return OmegaConf.select(self.cfg, name, default=default)

    def to_yaml(self, filepath):
        """Convert config to YAML file."""
        OmegaConf.save(self.cfg, filepath, resolve=True)

    @classmethod
    def from_yaml(cls, yaml_filepath, make_run_dirs=False):
        """Create Config from YAML file."""
        return cls(cfg=OmegaConf.load(yaml_filepath), make_run_dirs=make_run_dirs)
