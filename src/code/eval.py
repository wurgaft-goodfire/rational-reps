import os
import sys
from dataclasses import dataclass

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from pyprojroot import here
from tqdm import tqdm

sys.path.append(str(here()))

# load environment variables
from dotenv import load_dotenv
load_dotenv()

from .config import Config
from .models import load_model
from .data import init_dataset

# Config fields that must match for a cached eval to be reusable. These determine
# the fixed eval batches (which task distributions are sampled and how). Keep this
# in sync with anything EvalComponents.build reads off the dataset/config.
_CACHE_MATCH_KEYS = ("setting", "num_dims", "num_tasks", "task_prior", "random_seed")


def _find_cached_eval_cell(cache_dir: str, config) -> str:
    """Locate the eval-cache subdir whose resolved config matches `config` on
    ``_CACHE_MATCH_KEYS``. `cache_dir` may be a single run (self-match) or a sweep
    root with per-cell subdirs. Fails loudly if nothing matches."""
    if not os.path.isdir(cache_dir):
        raise FileNotFoundError(f"Eval cache dir does not exist: {cache_dir}")
    candidates = []
    if os.path.isfile(os.path.join(cache_dir, "eval_batches.pt")):
        candidates.append(cache_dir)
    else:
        for entry in sorted(os.listdir(cache_dir)):
            sub = os.path.join(cache_dir, entry)
            if os.path.isfile(os.path.join(sub, "eval_batches.pt")):
                candidates.append(sub)
    live = {k: config.cfg.get(k) for k in _CACHE_MATCH_KEYS}
    for cell in candidates:
        cached = OmegaConf.load(os.path.join(cell, "config_resolved.yaml"))
        if all(cached.get(k) == live[k] for k in _CACHE_MATCH_KEYS):
            return cell
    raise FileNotFoundError(f"No eval cache cell in {cache_dir} matches {live}")


#########################
# EvalComponents: the deterministic, model-independent eval setup, built once.
#########################

@dataclass
class EvalComponents:
    batches: list  # fixed pre-sampled eval batches (on CPU)
    device: torch.device

    @classmethod
    def build(cls, config, dataset) -> "EvalComponents":
        """Pre-sample a fixed set of eval batches once. Sampling is deterministic
        given the seed, so every checkpoint/step is scored on the same batches."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batches = [
            {k: v.cpu() for k, v in dataset.__getitem__().items()}
            for _ in range(config.eval_kwargs.num_eval_batches)
        ]
        return cls(batches=batches, device=device)

    def save(self, path: str):
        torch.save({"batches": self.batches}, path)


#########################
# Evaluator: the only public interface for evaluation.
#########################

class Evaluator:
    """Owns all evaluation logic. Build once; call evaluate(model) per checkpoint/step.

    Works for any model exposing a dict output with `.output_key` logits — trained
    small LMs, pretrained LLMs, and the Bayesian baseline alike.
    """

    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset
        self.c = EvalComponents.build(config, dataset)

    @classmethod
    def from_cache(cls, config, dataset, cache_dir):
        """Reuse a prior eval run's fixed batches from `cache_dir` instead of
        re-sampling, so online and standalone eval score on identical batches.
        `cache_dir` may be a single run or a sweep root (config-matched per cell)."""
        cell = _find_cached_eval_cell(cache_dir, config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cached = torch.load(os.path.join(cell, "eval_batches.pt"), weights_only=False)
        self = cls.__new__(cls)
        self.config = config
        self.dataset = dataset
        self.c = EvalComponents(batches=cached["batches"], device=device)
        return self

    # ── public interface ──────────────────────────────────────────────────

    def evaluate(self, model, output_hidden_states=None) -> tuple[dict, dict]:
        """Evaluate `model` on the fixed batches.

        Returns (metrics, extras): `metrics` is a flat {key: scalar} dict ready
        for wandb; `extras` holds the heavy per-example tensors (all_loss,
        hidden_states) for optional disk saving.
        """
        if output_hidden_states is None:
            output_hidden_states = self.config.eval_kwargs.output_hidden_states
        # The Bayesian baseline has no hidden states.
        if "bayes" in self.config.model_kwargs.model_type:
            output_hidden_states = False
        if hasattr(model, "eval"):
            model.eval()

        aggregated = self._eval_batches(model, output_hidden_states)
        metrics = {k: v for k, v in aggregated.items() if isinstance(v, (int, float))}
        extras = {k: v for k, v in aggregated.items() if not isinstance(v, (int, float))}

        if hasattr(model, "train"):
            model.train()
        return metrics, extras

    # ── private helpers ───────────────────────────────────────────────────

    def _eval_batches(self, model, output_hidden_states) -> dict:
        per_batch = []
        with torch.no_grad():
            for batch in self.c.batches:
                outputs, eval_labels, hidden_states = self._eval_batch(model, batch, output_hidden_states)
                batch_metrics = dict(self.dataset.compute_metrics((outputs, eval_labels), return_all=True))
                if hidden_states is not None:
                    batch_metrics["hidden_states"] = hidden_states
                per_batch.append(batch_metrics)
        return self._aggregate_metrics(per_batch)

    def _eval_batch(self, model, batch, output_hidden_states):
        all_outputs = model(batch["input_ids"].to(self.c.device), output_hidden_states=output_hidden_states)
        outputs = all_outputs[model.output_key].cpu()
        # Map a full-vocab LLM distribution down to the task's item tokens.
        if outputs.shape[-1] > 1 and outputs.shape[-1] != self.dataset.num_dims:
            outputs = outputs[..., self.dataset.item_tokens]
        eval_labels = batch.get("eval_labels", batch["labels"]).cpu()
        hidden_states = (
            torch.stack([h.cpu() for h in all_outputs["hidden_states"]], dim=1)
            if output_hidden_states else None
        )
        return outputs, eval_labels, hidden_states

    @staticmethod
    def _aggregate_metrics(per_batch: list) -> dict:
        """Average scalars and concatenate tensors/lists across batches."""
        aggregated = {}
        for key in per_batch[0].keys():
            first = per_batch[0][key]
            if isinstance(first, (int, float)):
                aggregated[key] = sum(b[key] for b in per_batch) / len(per_batch)
            elif isinstance(first, (list, torch.Tensor)):
                aggregated[key] = torch.cat([b[key] for b in per_batch], dim=0)
            elif isinstance(first, dict):
                aggregated[key] = Evaluator._aggregate_metrics([b[key] for b in per_batch])
        return aggregated


#########################
# Evaluation entry point
#########################

@hydra.main(version_base=None, config_path=str(here("conf")), config_name="config")
def run_evaluation(cfg: DictConfig) -> None:
    """Standalone evaluation.

    For trained small LMs, iterates every checkpoint under ``output_dir`` and
    writes per-checkpoint ``metrics.pt``. For pretrained LLMs / the Bayesian
    baseline, runs a single evaluation and writes ``metrics.pt`` to ``output_dir``.

    Usage:
      python -m src.code.eval runner=train_small_lm name=my_run
      python -m src.code.eval runner=eval_llm model_kwargs.model_path=google/gemma-2-2b
      python -m src.code.eval -m runner=eval_llm model_kwargs.model_path=m1,m2
    """
    config = Config(cfg, make_run_dirs=True)
    model_type = config.model_kwargs.model_type
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def build_evaluator(dataset):
        cache_dir = config.get("eval_cache_dir")
        if cache_dir:
            return Evaluator.from_cache(config, dataset, cache_dir)
        return Evaluator(config, dataset)

    if model_type == "small_lm":
        # Trained small LM: evaluate every saved checkpoint on shared fixed batches.
        dataset = init_dataset(config, overwrite_data=cfg.overwrite)
        evaluator = build_evaluator(dataset)
        checkpoints = sorted(int(item) for item in os.listdir(config.output_dir))
        for checkpoint in tqdm(checkpoints, desc="Evaluating checkpoints"):
            checkpoint_dir = os.path.join(config.output_dir, str(checkpoint))
            metrics_path = os.path.join(checkpoint_dir, "metrics.pt")
            if os.path.exists(metrics_path) and not cfg.overwrite:
                print(f"{metrics_path} exists; set overwrite=true to re-evaluate.")
                continue
            model, _ = load_model(config, os.path.join(checkpoint_dir, "model"), device=device)
            metrics, extras = evaluator.evaluate(model)
            torch.save({"metrics": metrics, "extras": extras, "step": checkpoint}, metrics_path)
    else:
        # Pretrained LLM or Bayesian baseline: a single evaluation run.
        if os.path.exists(config.output_dir) and any(
            f.endswith(".pt") for f in os.listdir(config.output_dir)
        ) and not cfg.overwrite:
            print(f"Results exist in {config.output_dir}; set overwrite=true to re-evaluate.")
            return
        config.to_yaml(os.path.join(config.run_dir, "config_resolved.yaml"))
        model_path = config.get("model_kwargs.model_path")
        model, tokenizer = load_model(config, model_path, device=device)
        if tokenizer is not None:
            config.tokenizer = tokenizer  # instance attr; data.py reads it via hasattr
        dataset = init_dataset(config, overwrite_data=cfg.overwrite)
        evaluator = build_evaluator(dataset)
        evaluator.c.save(os.path.join(config.run_dir, "eval_batches.pt"))
        metrics, extras = evaluator.evaluate(model)
        torch.save({"metrics": metrics, "extras": extras, "step": None},
                   os.path.join(config.output_dir, "metrics.pt"))
        print(f"Metrics saved to {os.path.join(config.output_dir, 'metrics.pt')}")


if __name__ == "__main__":
    run_evaluation()
