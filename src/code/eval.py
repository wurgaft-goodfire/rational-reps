import os
import sys

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from pyprojroot import here
from tqdm import tqdm

sys.path.append(str(here()))

# load environment variables
from dotenv import load_dotenv
load_dotenv()

from .config import Config
from .models import load_model, MODEL_CLASSES
from .data import init_dataset
from .analysis_utils import decoder_weights, representation


#########################
# Representational-similarity metrics (learned embedding vs predicted geometry)
#########################

def cka(X, Y):
    """Linear CKA between two representations (rows = items). Invariant to rotation/scale."""
    X = np.asarray(X, dtype=np.float64); Y = np.asarray(Y, dtype=np.float64)
    X = X - X.mean(0); Y = Y - Y.mean(0)
    Kx, Ky = X @ X.T, Y @ Y.T
    denom = np.linalg.norm(Kx) * np.linalg.norm(Ky)
    return float((Kx * Ky).sum() / denom) if denom > 0 else 0.0


def cca(X, Y, tol=1e-5, keep=None):
    """Mean canonical correlation between X and Y (rows = samples). Invariant to any invertible
    linear map of each side — the right similarity when representations match up to a linear
    transform, and blind to the metric within the shared subspace. ``tol`` sets the basis rank
    cutoff (above float32 numerical noise); ``keep`` additionally truncates at that fraction of
    the variance, which matters because CCA weights every direction equally while a near-empty one
    carries almost no loss and so is barely constrained."""
    def basis(A):
        A = np.asarray(A, np.float64); A = A - A.mean(0)
        U, S, _ = np.linalg.svd(A, full_matrices=False)
        k = int((S > tol * S[0]).sum())
        if keep is not None:
            k = min(k, 1 + int(np.searchsorted(np.cumsum(S ** 2) / (S ** 2).sum(), keep)))
        return U[:, :k]
    Bx, By = basis(X), basis(Y)
    if Bx.shape[1] == 0 or By.shape[1] == 0:
        return 0.0
    return float(np.linalg.svd(Bx.T @ By, compute_uv=False).mean())


def cka_null(oracle, repeats=20, seed=0):
    """CKA of a random code against ``oracle``: the floor a score has to clear. It depends on how
    many rows the code has — with a handful of items a random code already scores well — so every
    reported score needs its own null beside it."""
    oracle = np.asarray(oracle, float)
    rng = np.random.default_rng(seed)
    return float(np.mean([cka(rng.normal(size=oracle.shape), oracle) for _ in range(repeats)]))


def cka_shuffle_null(oracle, repeats=20, seed=0):
    """CKA of ``oracle`` against a row-shuffled copy of itself: the leakage floor. Shuffling keeps
    the spectrum and destroys only the row correspondence, so clearing this floor (and not just
    ``cka_null``) means the assignment of rows is right, not merely the shape."""
    oracle = np.asarray(oracle, float)
    rng = np.random.default_rng(seed)
    return float(np.mean([cka(oracle[rng.permutation(len(oracle))], oracle)
                          for _ in range(repeats)]))


def _rankdata(a):
    """Average ranks of a 1-D array (tied values share their mean rank). Values within a small
    relative tolerance are treated as tied (exact ties up to float noise)."""
    a = np.round(np.asarray(a, np.float64) / (np.abs(a).max() + 1e-12), 6)   # merge ties; bands differ far more
    order = a.argsort(); sv = a[order]
    ranks = np.empty(len(a), dtype=np.float64)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j)             # mean rank of the tied block
        i = j + 1
    return ranks


def rsa(X, Y):
    """RSA: Spearman correlation of the two Euclidean RDMs (upper triangle). No scipy dependency."""
    X = np.asarray(X, dtype=np.float64); Y = np.asarray(Y, dtype=np.float64)
    Dx = ((X[:, None] - X[None]) ** 2).sum(-1)
    Dy = ((Y[:, None] - Y[None]) ** 2).sum(-1)
    iu = np.triu_indices(len(X), 1)
    rx, ry = _rankdata(Dx[iu]), _rankdata(Dy[iu])
    rx = rx - rx.mean(); ry = ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else 0.0

#########################
# Evaluator: the only public interface for evaluation.
#########################

class Evaluator:
    """Evaluate a model on every input the task defines, against the exact (Bayes) outcome
    distribution of each. Build once; call evaluate(model) per checkpoint/step. Works for any
    model exposing a dict output with `.output_key` logits and the ladder's ``hidden_states``
    contract (the last hidden state at the last position is what the readout reads)."""

    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.inputs = dataset.model_inputs().cpu()                 # (n, arity [+1]), every input once
        # the exact target after the input: appended dummy token, then the last predicted position
        padded = torch.cat([self.inputs, torch.zeros_like(self.inputs[:, :1])], dim=1)
        targets = dataset.optimal_predictions(padded.to(dataset.device))
        self.targets = (targets[:, -1] if targets.ndim == 3 else targets).float().cpu()   # (n, C)

    def evaluate(self, model) -> tuple[dict, dict]:
        """Returns (metrics, extras): ``metrics`` is {"loss": KL(model || true) over the inputs};
        ``extras`` holds what analysis reads instead of the model: ``logits`` (n, C) at the
        outcome slot, ``representation`` (n, width) the last hidden state at the last position
        (None for a readout that reads no vector), ``readout`` (C, width)."""
        if hasattr(model, "eval"):
            model.eval()
        with torch.no_grad():
            out = model(self.inputs.to(self.device), output_hidden_states=True)
        logits = out[model.output_key].float().cpu()
        if logits.ndim == 3:
            logits = logits[:, -1]                                 # the position predicting the outcome
        if logits.shape[-1] == self.dataset.vocab_size:            # vocab-wide head -> outcome tokens
            logits = logits[:, self.dataset.label_tokens]
        metrics = self.dataset.compute_metrics((logits, self.targets), return_all=False)
        readout = getattr(model, "readout_matrix", None)
        rep = out["hidden_states"][-1][:, -1].float().cpu() if readout is not None and "hidden_states" in out else None
        extras = {"logits": logits, "representation": rep,
                  "readout": None if readout is None else decoder_weights(model, self.dataset)}
        if hasattr(model, "train"):
            model.train()
        return metrics, extras

    def representation(self, model, project=True):
        """The representation of every input (`analysis_utils.representation`); None without one."""
        return representation(model, self.dataset, project=project)


#########################
# Evaluation entry point
#########################

@hydra.main(version_base=None, config_path=str(here("conf")), config_name="config")
def run_evaluation(cfg: DictConfig) -> None:
    """Standalone evaluation. Models trained from scratch: iterates every checkpoint under
    ``output_dir``, writing per-checkpoint ``metrics.pt``. Pretrained LLMs: a single evaluation,
    writing ``metrics.pt`` to ``output_dir``."""
    config = Config(cfg, make_run_dirs=True)
    model_type = config.model_kwargs.model_type
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def build_evaluator(dataset):
        return Evaluator(config, dataset)

    if model_type in MODEL_CLASSES:
        # Trained from scratch: evaluate every saved checkpoint.
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
        # Pretrained LLM: a single evaluation run.
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
        evaluator = Evaluator(config, dataset)
        metrics, extras = evaluator.evaluate(model)
        torch.save({"metrics": metrics, "extras": extras, "step": None},
                   os.path.join(config.output_dir, "metrics.pt"))
        print(f"Metrics saved to {os.path.join(config.output_dir, 'metrics.pt')}")


if __name__ == "__main__":
    run_evaluation()
