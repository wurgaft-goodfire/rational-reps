import os
import torch
from transformers import TrainerCallback
import wandb

from .eval import Evaluator

# Keys returned by the dataset that are for evaluation only and must not be
# forwarded to the model (they are not valid forward() arguments).
EVAL_ONLY_KEYS = {"eval_labels"}


#########################
# helpers for training
#########################

class PreBatchedDataCollator:
    """Datasets here already return a full batch from __getitem__, so the
    DataLoader hands us a single-element list. Unwrap it and drop eval-only keys
    that would otherwise be passed to model.forward()."""

    def __call__(self, features):
        return {k: v for k, v in features[0].items() if k not in EVAL_ONLY_KEYS}


#########################
# Step metrics callback
#########################

class StepMetricsCallback(TrainerCallback):
    """Runs evaluation every `step_interval` training steps and logs to wandb."""

    def __init__(self, step_interval, trainer, config):
        self.trainer = trainer
        self.config = config
        self.step_interval = step_interval
        self.use_wandb = config.cfg.wandb.enabled
        self.history = []  # in-memory {step, **metrics} log, returned for notebook use
        # Build the evaluator once (fixed batches), reusing a cached eval if configured.
        cache_dir = config.get("eval_cache_dir")
        if cache_dir:
            self.evaluator = Evaluator.from_cache(config, trainer.train_dataset, cache_dir)
        else:
            self.evaluator = Evaluator(config, trainer.train_dataset)
            # Persist the fixed batches so a later standalone eval (eval_cache_dir)
            # can score checkpoints on the exact same batches.
            self.evaluator.c.save(os.path.join(config.run_dir, "eval_batches.pt"))

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.step_interval != 0:
            return
        metrics, extras = self.evaluator.evaluate(self.trainer.model)
        self.history.append({"step": state.global_step, **metrics})
        if self.use_wandb:
            wandb.log({f"eval/{k}": v for k, v in metrics.items()}, step=state.global_step)

        if self.config.training_kwargs.save_online_eval:
            checkpoint_save_dir = os.path.join(self.config.output_dir, str(state.global_step))
            os.makedirs(checkpoint_save_dir, exist_ok=True)
            torch.save({"metrics": metrics, "extras": extras, "step": state.global_step},
                       os.path.join(checkpoint_save_dir, "metrics.pt"))

        self.trainer.model.train()


#########################
# Checkpoint saving callback
#########################

class CustomSaveCallback(TrainerCallback):
    def __init__(self, steps_to_save, trainer):
        self.steps_to_save = set(steps_to_save)
        self.trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        """Save the model at the configured steps (and at the final step, if any
        steps are scheduled at all)."""
        if (
            state.global_step in self.steps_to_save
            or (self.steps_to_save and state.global_step == args.max_steps)
        ):
            checkpoint_save_dir = f"{args.output_dir}/{state.global_step}"
            os.makedirs(checkpoint_save_dir, exist_ok=True)
            self.trainer.model.save_pretrained(os.path.join(checkpoint_save_dir, "model"))
