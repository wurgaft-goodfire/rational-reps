"""The array side of analysis: read a run's saved evaluation, transform the arrays explicitly,
and compare them with a predicted geometry. Nothing here constructs a model; `eval.Evaluator`
wrote every array this module reads."""
import os

import numpy as np
from omegaconf import OmegaConf, open_dict

from .config import run_key


#########################
# Reading a run
#########################

def representation(hidden_states, static):
    """What the readout reads: the last layer at the last position for a static task, (n, width),
    the position that predicts the outcome; at every position for a sequence task, (B, T, width).
    None for a model without a vector representation."""
    if hidden_states is None:
        return None
    return hidden_states[-1][:, -1] if static else hidden_states[-1]


def load_evaluation(path):
    """One saved evaluation as a dict of arrays (see `Evaluator.evaluate`): ``step``, ``loss``,
    ``inputs``, ``targets``, ``logits``, ``hidden_states``, ``readout``, plus the derived
    ``representation``; ``hidden_states``, ``readout`` and ``representation`` are None when the
    model has no vector representation."""
    with np.load(path) as z:
        saved = {k: z[k] for k in z.files}
    saved["step"], saved["loss"] = int(saved["step"]), float(saved["loss"])
    saved.setdefault("hidden_states", None)
    saved.setdefault("readout", None)
    saved["representation"] = representation(saved["hidden_states"], static=saved["targets"].ndim == 2)
    return saved


def _evaluation_path(config):
    """The final checkpoint's saved evaluation of a run, or None when there is none."""
    steps = [int(d) for d in os.listdir(config.output_dir) if d.isdigit()] if os.path.isdir(config.output_dir) else []
    path = os.path.join(config.output_dir, str(max(steps)), config.eval_file) if steps else None
    return path if path is not None and os.path.exists(path) else None


def has_run(overrides, name=None, cache_dir=None):
    """Whether `load_run` would find a saved evaluation for these overrides."""
    from .train import resolve_config
    return _evaluation_path(resolve_config(overrides, name=name, cache_dir=cache_dir)) is not None


def load_run(overrides, name=None, cache_dir=None, project=True):
    """The run a list of Hydra overrides names, read from disk: its ``config``, ``dataset`` (the
    task) and the final checkpoint's `load_evaluation`. Never trains or loads a model: a missing
    run raises FileNotFoundError with the training command, and a run whose recorded resolved
    config or task kernel differs from what the overrides now build raises RuntimeError.

    ``project`` (the default) expresses ``hidden_states``, ``representation`` and ``readout`` in
    the coordinates of `readout_basis`: the directions of the width-dimensional space the readout
    reads, hence the ones the loss constrains. The logits are unchanged by it. ``project=False``
    returns the arrays as the model produced them."""
    from .data import init_dataset, kernel_hash
    from .train import resolve_config
    config = resolve_config(overrides, name=name, cache_dir=cache_dir)
    command = "uv run python -m src.code.train " + " ".join(list(overrides) + ([f"name={name}"] if name else []))
    path = _evaluation_path(config)
    if path is None:
        raise FileNotFoundError(f"no saved evaluation under {config.run_dir}; train it with\n  {command}")
    dataset = init_dataset(config)
    with open_dict(config.cfg):
        config.cfg.kernel = kernel_hash(dataset)
    exclude = config.cfg.run_key_exclude
    if run_key(OmegaConf.load(os.path.join(config.run_dir, "config_resolved.yaml")), exclude) != run_key(config.cfg, exclude):
        raise RuntimeError(f"{config.run_dir} was trained under a different resolved config or task "
                           f"than the overrides now build; retrain it with\n  {command} overwrite=true")
    run = dict(config=config, dataset=dataset, **load_evaluation(path))
    if project and run["readout"] is not None:
        B = readout_basis(run["readout"])
        for key in ("hidden_states", "representation", "readout"):
            run[key] = run[key] @ B.T
    return run


#########################
# Transforms of the representation (each an explicit choice a notebook makes)
#########################

def readout_basis(D, tol=1e-4):
    """Orthonormal rows spanning the centered readout's row space, (rank, width). A code
    component orthogonal to it cannot change the output."""
    Dc = np.asarray(D, float)
    Dc = Dc - Dc.mean(0)
    return np.linalg.svd(Dc, full_matrices=False)[2][:np.linalg.matrix_rank(Dc, tol=tol)]


def feature_vectors(H, Z):
    """Least-squares additive fit H ~ Z V + 1 v0: the vector each feature writes into the
    code. Returns (V (m, d), v0 (d,))."""
    Z = np.asarray(Z, float)
    X = np.column_stack([Z, np.ones(len(Z))])
    V = np.linalg.lstsq(X, np.asarray(H, float), rcond=None)[0]
    return V[:-1], V[-1]


def logit_map(L, Z):
    """Additive fit of the row-centred logits, L_c ~ Z R + 1 b^T: the gauge-invariant
    feature -> output map. Returns (R (m, m), b (m,), r2)."""
    L = np.asarray(L, float)
    L = L - L.mean(1, keepdims=True)
    X = np.column_stack([np.asarray(Z, float), np.ones(len(L))])
    C = np.linalg.lstsq(X, L, rcond=None)[0]
    r2 = 1.0 - ((L - X @ C) ** 2).sum() / (L ** 2).sum()
    return C[:-1], C[-1], float(r2)


def leverage(W):
    """Hat matrix of the rows of W: P = W (W^T W)^+ W^T. Invariant to any invertible
    transform of the columns; diag in [0, 1] sums to rank(W). Returns (diag (m,), P)."""
    W = np.asarray(W, float)
    P = W @ np.linalg.pinv(W.T @ W) @ W.T
    return np.diag(P).copy(), P


#########################
# Similarity between a measured and a predicted geometry
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
