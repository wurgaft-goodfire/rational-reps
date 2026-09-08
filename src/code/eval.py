import os
import sys

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from pyprojroot import here
from tqdm import tqdm

sys.path.append(str(here()))

# load environment variables
from dotenv import load_dotenv
load_dotenv()

from .config import Config
from .models import load_model, MODEL_CLASSES
from .data import init_dataset, kernel_hash

EVAL_FILE = "eval.npz"          # one file per checkpoint: loss, logits, representation, readout


def decoder_weights(model, dataset):
    """The readout rows over the outcome tokens, (C, width): ``readout_matrix`` restricted to the
    dataset's outcome tokens when the head is vocabulary-wide (the causal LMs)."""
    W = model.readout_matrix
    if W.shape[0] == getattr(dataset, "vocab_size", None) and len(dataset.label_tokens) < W.shape[0]:
        W = W[dataset.label_tokens]
    return W.detach().float().cpu().numpy()


#########################
# Evaluator: the only forward pass analysis ever depends on.
#########################

class Evaluator:
    """Evaluate a model on every input the task defines, against the exact (Bayes) outcome
    distribution of each. Build once; call evaluate(model) per checkpoint/step. Works for any
    model exposing a dict output with `.output_key` logits and the ladder's ``hidden_states``
    contract (the last hidden state at the last position is what the readout reads)."""

    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset
        self.kernel = kernel_hash(dataset)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.inputs = dataset.model_inputs().cpu()                 # (n, arity [+1]), every input once
        # the exact target after the input: appended dummy token, then the last predicted position
        padded = torch.cat([self.inputs, torch.zeros_like(self.inputs[:, :1])], dim=1)
        targets = dataset.optimal_predictions(padded.to(dataset.device))
        self.targets = (targets[:, -1] if targets.ndim == 3 else targets).float().cpu()   # (n, C)

    def evaluate(self, model) -> dict:
        """The model's own objects over every input, as numpy: ``loss`` KL(model || true),
        ``logits`` (n, C) at the outcome slot, ``representation`` (n, width) the last hidden state
        at the last position and ``readout`` (C, width) the rows that read it; both None for a
        readout that reads no vector."""
        if hasattr(model, "eval"):
            model.eval()
        with torch.no_grad():
            out = model(self.inputs.to(self.device), output_hidden_states=True)
        logits = out[model.output_key].float().cpu()
        if logits.ndim == 3:
            logits = logits[:, -1]                                 # the position predicting the outcome
        if logits.shape[-1] == self.dataset.vocab_size:            # vocab-wide head -> outcome tokens
            logits = logits[:, self.dataset.label_tokens]
        loss = self.dataset.compute_metrics((logits, self.targets), return_all=False)["loss"]
        readout = getattr(model, "readout_matrix", None)
        rep = out["hidden_states"][-1][:, -1].float().cpu().numpy() if readout is not None and "hidden_states" in out else None
        if hasattr(model, "train"):
            model.train()
        return {"loss": float(loss), "logits": logits.numpy(), "representation": rep,
                "readout": None if readout is None else decoder_weights(model, self.dataset)}

    def save(self, result, checkpoint_dir, step):
        """Write ``evaluate``'s result to ``checkpoint_dir/eval.npz`` with the step and the task's
        kernel hash; absent arrays are omitted. Returns the path."""
        path = os.path.join(checkpoint_dir, EVAL_FILE)
        np.savez(path, step=step, kernel=self.kernel,
                 **{k: v for k, v in result.items() if v is not None})
        return path


#########################
# Evaluation entry point
#########################

@hydra.main(version_base=None, config_path=str(here("conf")), config_name="config")
def run_evaluation(cfg: DictConfig) -> None:
    """Standalone evaluation. Models trained from scratch: iterates every checkpoint under
    ``output_dir``, writing per-checkpoint ``eval.npz``. Pretrained LLMs: a single evaluation,
    writing ``eval.npz`` to ``output_dir``."""
    config = Config(cfg, make_run_dirs=True)
    model_type = config.model_kwargs.model_type
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_type in MODEL_CLASSES:
        # Trained from scratch: evaluate every saved checkpoint.
        dataset = init_dataset(config, overwrite_data=cfg.overwrite)
        evaluator = Evaluator(config, dataset)
        checkpoints = sorted(int(item) for item in os.listdir(config.output_dir))
        for checkpoint in tqdm(checkpoints, desc="Evaluating checkpoints"):
            checkpoint_dir = os.path.join(config.output_dir, str(checkpoint))
            if os.path.exists(os.path.join(checkpoint_dir, EVAL_FILE)) and not cfg.overwrite:
                print(f"{checkpoint_dir}/{EVAL_FILE} exists; set overwrite=true to re-evaluate.")
                continue
            model, _ = load_model(config, os.path.join(checkpoint_dir, "model"), device=device)
            evaluator.save(evaluator.evaluate(model), checkpoint_dir, step=checkpoint)
    else:
        # Pretrained LLM: a single evaluation run.
        if os.path.exists(os.path.join(config.output_dir, EVAL_FILE)) and not cfg.overwrite:
            print(f"Results exist in {config.output_dir}; set overwrite=true to re-evaluate.")
            return
        config.to_yaml(os.path.join(config.run_dir, "config_resolved.yaml"))
        model_path = config.get("model_kwargs.model_path")
        model, tokenizer = load_model(config, model_path, device=device)
        if tokenizer is not None:
            config.tokenizer = tokenizer  # instance attr; data.py reads it via hasattr
        dataset = init_dataset(config, overwrite_data=cfg.overwrite)
        evaluator = Evaluator(config, dataset)
        path = evaluator.save(evaluator.evaluate(model), config.output_dir, step=-1)
        print(f"Evaluation saved to {path}")


if __name__ == "__main__":
    run_evaluation()
