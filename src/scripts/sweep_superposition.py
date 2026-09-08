"""Sweep the feature-cube setting over sparsity p, ladder slope beta, correlation rho, rank
d and seed: train the networks per cell (GPU) and solve the theory per cell (CPU),
one npz per cell in ``SWEEP_DIR``. Correlated cells are solved and measured on the
predictive modes (`modes`). Usage: ``python -m src.scripts.sweep_superposition --train
--jobs 8 [--runners linear,additive_features,paired_features]`` and, in another process,
``--theory --jobs 12``. Each run's final evaluation is saved by training, so the notebook
reads logit tables from disk without constructing models."""
import argparse
import itertools
import pathlib

import numpy as np

N, M, SCALE, STEPS = 2048, 64, 8.0, 8000
GRID = dict(p=[0.05, 0.15, 0.3, 0.5], beta=[0.25, 0.5, 1.0], rho=[0.0, 0.3, 0.6], d=[3, 6, 12],
            seed=[0, 1, 2, 3, 4])
SWEEP_DIR = pathlib.Path("outputs/graph-cooccurrence/superposition_sweep")


def cells():
    """Every (p, beta, rho, d, seed) of the grid, in order."""
    return list(itertools.product(GRID["p"], GRID["beta"], GRID["rho"], GRID["d"], GRID["seed"]))


def task(p, beta, rho):
    return ["setting=feature_cube", f"structure_kwargs.N={N}", f"structure_kwargs.m={M}",
            f"structure_kwargs.scale={SCALE}", "+structure_kwargs.spectrum=power",
            f"+structure_kwargs.beta={beta}", f"structure_kwargs.p={p}"] + (
        [f"+structure_kwargs.rho={rho}"] if rho else [])


RUNNERS = dict(linear="sp", additive_features="addf", paired_features="pairf")


def run_name(p, beta, rho, d, seed, runner="linear"):
    return f"{RUNNERS[runner]}_p{p}_b{beta}_d{d}_s{seed}" + (f"_r{rho}" if rho else "")


def cell_path(p, beta, rho, d, seed):
    return SWEEP_DIR / (f"geom_p{p}_b{beta}_d{d}_s{seed}" + (f"_r{rho}" if rho else "") + ".npz")


def features(p, beta, rho, seed):
    from src.code.data import feature_cube
    return feature_cube(N, M, p, scale=SCALE, seed=seed, spectrum="power", beta=beta,
                        rho=rho)[1]["Z"]


def modes(alpha, Z, rho):
    """The coordinates the theory and the measurement use: the features themselves when
    independent, else the predictive modes with the Bayes map expressed on them.
    Returns (alpha_or_W, X)."""
    from src.code.geometry import whiten
    if not rho:
        return alpha, Z
    U, C_half = whiten(Z)
    return np.diag(alpha) @ C_half.T, U


def overrides(p, beta, rho, d, seed, runner="linear"):
    """The Hydra overrides of one cell's run (the run directory is keyed by them)."""
    return task(p, beta, rho) + [f"runner={runner}", f"model_kwargs.model_config.embed_dim={d}",
                                 f"training_kwargs.train_steps={STEPS}", f"random_seed={seed}",
                                 f"structure_kwargs.seed={seed}"]


def cached(p, beta, rho, d, seed, runner="linear"):
    """Whether one cell's run of `runner` is on disk with its saved final evaluation."""
    import os
    from src.code.config import EVAL_FILE
    from src.code.train import resolve_config
    config = resolve_config(overrides(p, beta, rho, d, seed, runner),
                            name=run_name(p, beta, rho, d, seed, runner))
    steps = [int(x) for x in os.listdir(config.output_dir) if x.isdigit()] if os.path.isdir(config.output_dir) else []
    return bool(steps) and os.path.exists(os.path.join(config.output_dir, str(max(steps)), EVAL_FILE))


def train_cell(p, beta, rho, d, seed, runner="linear"):
    """One cell's network (the linear network or a reference class, `RUNNERS`): trains it if
    absent, then returns the saved logit table over states (N, M) — never the model."""
    from src.code.analysis_utils import load_run
    from src.code.train import train_inline
    if not cached(p, beta, rho, d, seed, runner):
        train_inline(overrides(p, beta, rho, d, seed, runner),
                     name=run_name(p, beta, rho, d, seed, runner))
    return load_run(overrides(p, beta, rho, d, seed, runner),
                    name=run_name(p, beta, rho, d, seed, runner))["logits"]


def theory_cell(p, beta, rho, d, seed, n_frames=8):
    """Solve the theory of one cell on its coordinates and cache it (skips cells on disk)."""
    from src.code.geometry import predict_geometry
    out = cell_path(p, beta, rho, d, seed)
    if out.exists():
        return out
    alpha = SCALE * (1.0 + np.arange(M)) ** -beta
    W, X = modes(alpha, features(p, beta, rho, seed), rho)
    o = predict_geometry(W, X, d, n_frames=n_frames, seed=seed)
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(out, **{k: np.asarray(v) for k, v in o.items() if not isinstance(v, dict)})
    return out



def sweep_cell(p, beta, rho, d, runners=("linear",)):
    """One cell of the sweep for the notebook: per seed, the theory and each trained runner
    measured on the cell's coordinates (`analysis_utils.measure`), plus the theory's solution
    ('o'). Reads each run's saved evaluation, so no model is constructed."""
    from src.code.analysis_utils import measure
    from src.code.geometry import theory_table
    alpha = SCALE * (1.0 + np.arange(M)) ** -beta
    rows = []
    for seed in GRID["seed"]:
        _, X = modes(alpha, features(p, beta, rho, seed), rho)
        o = dict(np.load(cell_path(p, beta, rho, d, seed)))
        row = dict(th=measure(theory_table(o, X), X, d), o=o)
        for runner in runners:
            row[runner] = measure(train_cell(p, beta, rho, d, seed, runner), X, d)
        rows.append(row)
    return rows


if __name__ == "__main__":
    from joblib import Parallel, delayed
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--theory", action="store_true")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--runners", default="linear", help="comma-separated keys of RUNNERS")
    args = ap.parse_args()
    if args.train:
        runners = args.runners.split(",")
        Parallel(n_jobs=args.jobs)(delayed(train_cell)(*c, r) for r in runners for c in cells())
        print("trained", len(cells()), "cells x", runners, flush=True)
    if args.theory:
        Parallel(n_jobs=args.jobs)(delayed(theory_cell)(*c) for c in cells())
        print("solved", len(cells()), "cells", flush=True)
