import os
import random
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf


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
            training_kwargs.save_steps = np.arange(training_kwargs.eval_steps, training_kwargs.train_steps + 1, training_kwargs.eval_steps, dtype=int).tolist()

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
