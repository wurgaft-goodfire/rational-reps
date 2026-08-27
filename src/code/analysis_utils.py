#########################
# imports
##########################

import os
import sys

import numpy as np
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
# Reading a trained model's code
#########################

def decoder_weights(model, dataset):
    """The readout rows over the outcome tokens, whatever the architecture calls them. (C, d)."""
    if hasattr(model, "readout"):                              # linear
        W = model.readout
    elif hasattr(model, "mlp"):                                # mlp: the final layer
        W = model.mlp[-1].weight
    elif hasattr(model, "head"):                               # rnn
        W = model.head.weight[dataset.label_tokens]
    else:                                                      # causal LM: the unembedding
        W = model.get_output_embeddings().weight[dataset.label_tokens]
    return W.detach().float().cpu().numpy()


def input_code(model, dataset, project=False):
    """Pre-decode code of every input the task defines, ``(n_inputs, d)``. ``project`` re-expresses
    it in the readout's row space, dropping the directions the decoder cannot read. ``None`` when
    the architecture has no vector code (a bilinear readout reads an outer product)."""
    with torch.no_grad():
        X = model.input_reps(dataset.model_inputs())
    if X is None:
        return None
    X = X.detach().float().cpu().numpy()
    return X @ readout_basis(model, dataset).T if project else X


def logit_r2(model, dataset):
    """How much of the task's row-centred log-conditional the model's readout actually realises,
    at the best global scale (a softmax ignores per-row constants, CKA ignores overall scale).
    1.0 means the logits *are* the target, which is the premise under which low loss pins the
    geometry; a target with zero-probability outcomes does not require it."""
    L = np.asarray(dataset.conditional, float)
    Lc = L - L.mean(1, keepdims=True)
    lg = input_code(model, dataset) @ decoder_weights(model, dataset).T
    lg = lg - lg.mean(1, keepdims=True)
    scale = (lg * Lc).sum() / (Lc ** 2).sum()
    return float(1 - ((lg - scale * Lc) ** 2).sum() / ((scale * Lc) ** 2).sum())


def readout_basis(model, dataset, tol=1e-4):
    """Orthonormal rows spanning the centered decoder's row space. A code component orthogonal to
    it cannot change the output, so it is not part of the geometry the task pins down."""
    W = decoder_weights(model, dataset)
    W = W - W.mean(0)
    return np.linalg.svd(W, full_matrices=False)[2][:np.linalg.matrix_rank(W, tol=tol)]


#########################
# Loading eval results
#########################

def load_eval_results(setting, experiment_name, source="local", entity=None, project=None):
    """Load eval results into a tidy DataFrame: one row per (run, step) with a `step` column, a
    `config` column, and one column per flat metric. ``source="local"`` (default) reads the
    ``metrics.pt`` files under ``$CACHE_DIR/{setting}/{override_dirname}/{name}/``;
    ``source="wandb"`` pulls run history via the wandb API (needs ``entity``/``project`` or env)."""
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
    # runs live at {setting}/{override_dirname}/{name}; collect every sweep cell of `experiment_name`
    setting_dir = os.path.join(os.getenv("CACHE_DIR", "./outputs"), setting)
    rows = []
    for subdir in sorted(os.listdir(setting_dir)):
        subdir_path = os.path.join(setting_dir, subdir, experiment_name)
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
            # Single-run eval (pretrained LLM): metrics at the run root.
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


#########################
# Representation-analysis helpers (notebook-side: alignment, held-out decoding, controls)
#########################

def align_representations(X, ref, method="rotation", rank=None):
    """Align ``X`` to ``ref``; returns ``(aligned, R2)`` with ``aligned`` in ``ref``'s space/units.
    ``"linear"``: affine least squares (the fit any invertible map achieves). ``"rotation"``:
    PCA-truncate ``X`` to ``rank`` (default: ref dim) + orthogonal Procrustes — basis-level agreement."""
    from scipy.linalg import orthogonal_procrustes
    X = np.asarray(X, dtype=np.float64); ref = np.asarray(ref, dtype=np.float64)
    if method == "linear":
        Xa = np.c_[X, np.ones(len(X))]
        pred = Xa @ np.linalg.lstsq(Xa, ref, rcond=None)[0]
        r2 = 1 - ((pred - ref) ** 2).sum() / ((ref - ref.mean(0)) ** 2).sum()
        return pred, float(r2)
    if method == "rotation":
        refc = ref - ref.mean(0); refn = refc / np.linalg.norm(refc)
        Xc = X - X.mean(0)
        Xr = Xc @ np.linalg.svd(Xc, full_matrices=False)[2][:rank or ref.shape[1]].T
        Xr = Xr / np.linalg.norm(Xr)
        R, _ = orthogonal_procrustes(Xr, refn)
        r2 = 1 - np.linalg.norm(Xr @ R - refn) ** 2 / np.linalg.norm(refn) ** 2
        return (Xr @ R) * np.linalg.norm(refc) + ref.mean(0), float(r2)
    raise ValueError(f"unknown method {method!r} (expected 'linear' or 'rotation')")


def _ridge_fit_r2(Xtr, Ytr, Xte, Yte, alpha):
    """Ridge-regress ``Y`` on standardized ``X`` (bias unregularized); return R² on the test split.
    Units with no usable variance on the train split are dropped before standardizing."""
    sd0 = Xtr.std(0)
    keep = sd0 > 1e-6 * sd0.max()
    Xtr, Xte = Xtr[:, keep], Xte[:, keep]
    mu, sd = Xtr.mean(0), Xtr.std(0)
    Xtr = np.c_[(Xtr - mu) / sd, np.ones(len(Xtr))]
    Xte = np.c_[(Xte - mu) / sd, np.ones(len(Xte))]
    A = Xtr.T @ Xtr + alpha * np.eye(Xtr.shape[1]); A[-1, -1] -= alpha
    W = np.linalg.solve(A, Xtr.T @ Ytr)
    ss_res = ((Yte - Xte @ W) ** 2).sum()
    ss_tot = ((Yte - Yte.mean(0)) ** 2).sum()
    return float(1 - ss_res / ss_tot)


def heldout_decode_r2(H, F, groups, train_frac=0.6, seed=0, alpha=1.0, train_mask=None):
    """R² of a linear map ``H -> F`` fit on a train split of *distinct prefix groups* and evaluated
    on held-out groups, so test prefixes are wholly unseen. Pass ``train_mask`` for a custom split
    (e.g. train on short prefixes, test on long)."""
    H = np.asarray(H, float); F = np.asarray(F, float); groups = np.asarray(groups)
    if train_mask is None:
        ids = np.unique(groups)
        rng = np.random.default_rng(seed); rng.shuffle(ids)
        train_ids = set(ids[:int(round(train_frac * len(ids)))].tolist())
        train_mask = np.array([g in train_ids for g in groups])
    train_mask = np.asarray(train_mask, bool)
    return _ridge_fit_r2(H[train_mask], F[train_mask], H[~train_mask], F[~train_mask], alpha)


def untrained_twin(config, dataset):
    """A fresh model with the run's architecture but random (pre-training) weights. If the predictive
    state were merely 'linearly fittable' by a wide hidden, this random model would match the trained
    one under CCA; it does not — that gap is what makes CCA≈1 a real result, not a capacity artefact."""
    from omegaconf import open_dict
    from .models import init_model
    mc = config.cfg.model_kwargs.model_config
    with open_dict(mc):
        mc.vocab_size = dataset.vocab_size
        mc.max_position_embeddings = dataset.seq_len
    return init_model(config)
