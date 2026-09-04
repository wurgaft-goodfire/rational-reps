"""Sweep the feature-cube setting over sparsity p, ladder slope beta, rank d and seed: train
the linear network per cell (GPU) and solve the theory per cell (CPU), one npz per cell in
``SWEEP_DIR``. Usage: ``python -m src.code.sweep_superposition --train --jobs 8`` and, in
another process, ``--theory --jobs 12``."""
import argparse
import itertools
import pathlib

import numpy as np

N, M, SCALE, STEPS = 2048, 64, 8.0, 8000
GRID = dict(p=[0.05, 0.15, 0.3, 0.5], beta=[0.25, 0.5, 1.0], d=[3, 6, 12], seed=[0, 1, 2, 3, 4])
SWEEP_DIR = pathlib.Path("outputs/graph-cooccurrence/superposition_sweep")


def cells():
    """Every (p, beta, d, seed) of the grid, in order."""
    return list(itertools.product(GRID["p"], GRID["beta"], GRID["d"], GRID["seed"]))


def task(p, beta):
    return ["setting=feature_cube", f"structure_kwargs.N={N}", f"structure_kwargs.m={M}",
            f"structure_kwargs.scale={SCALE}", "+structure_kwargs.spectrum=power",
            f"+structure_kwargs.beta={beta}", f"structure_kwargs.p={p}"]


def run_name(p, beta, d, seed):
    return f"sp_p{p}_b{beta}_d{d}_s{seed}"


def cell_path(p, beta, d, seed):
    return SWEEP_DIR / f"geom_p{p}_b{beta}_d{d}_s{seed}.npz"


def features(p, beta, seed):
    from .data import feature_cube
    return feature_cube(N, M, p, scale=SCALE, seed=seed, spectrum="power", beta=beta)[1]["Z"]


def train_cell(p, beta, d, seed):
    """Train (or load) the linear network of one cell; returns its logit table over states (N, M)."""
    from .train import train_inline
    model, _ds, ev, *_ = train_inline(
        task(p, beta) + ["runner=linear", f"model_kwargs.model_config.embed_dim={d}",
                         f"training_kwargs.train_steps={STEPS}", f"random_seed={seed}",
                         f"structure_kwargs.seed={seed}"], name=run_name(p, beta, d, seed))
    code = np.asarray(ev.input_code(model, project=False), float)
    return code @ model.readout.detach().cpu().numpy()[:M].T


def theory_cell(p, beta, d, seed, n_frames=8):
    """Solve the theory of one cell and cache it (skips cells already on disk)."""
    from .geometry import predict_geometry
    out = cell_path(p, beta, d, seed)
    if out.exists():
        return out
    alpha = SCALE * (1.0 + np.arange(M)) ** -beta
    o = predict_geometry(alpha, features(p, beta, seed), d, n_frames=n_frames, seed=seed)
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(out, **{k: np.asarray(v) for k, v in o.items() if not isinstance(v, dict)})
    return out


if __name__ == "__main__":
    from joblib import Parallel, delayed
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--theory", action="store_true")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    if args.train:
        Parallel(n_jobs=args.jobs)(delayed(train_cell)(*c) for c in cells())
        print("trained", len(cells()), "cells", flush=True)
    if args.theory:
        Parallel(n_jobs=args.jobs)(delayed(theory_cell)(*c) for c in cells())
        print("solved", len(cells()), "cells", flush=True)
