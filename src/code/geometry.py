"""The predicted-geometry oracle.

Given a task's *conditional* — the log-conditional matrix ``L[theta, x] = log p(x|theta)``
for order-1 tasks, or the conditional tensor ``T[c, a, b, ...]`` for order-2+ — factor it
into a code ``E`` (the geometry we plot / compare against) and a decoder ``D`` (the readout).

This is the closed-form prediction the whole project rests on: the geometry a network *should*
learn, read straight off the structure of the task. See ``src/notebooks/demo_graphs.ipynb``
(the canonical notebook) for the derivations these functions port.
"""
import numpy as np


def factor_logcond(L, drop_base_measure=False):
    """Correspondence-analysis SVD of an order-1 log-conditional ``L[theta, x]``.

    Always column-center (remove the per-context row offset / log-partition a softmax ignores).
    ``drop_base_measure=True`` additionally removes the weighted per-outcome mean (the x-only
    base measure) — the demo_clean factorization for *distribution* families, which makes the
    decoder come out as the family's sufficient statistic. It leaves the *code* geometry
    unchanged (the base-measure mode is item-constant, so it cancels in RDM/CKA comparisons),
    so it is off by default (the demo_graphs co-occurrence factorization).

    Returns ``(Lc, E, D, S, nu)``: ``E`` the code (singular-value-scaled) and ``D`` the decoder.
    """
    L = np.asarray(L, dtype=np.float64)
    N, _ = L.shape
    w = np.full(N, 1.0 / N)                                   # uniform row (item) measure
    pp = np.exp(L - L.max(1, keepdims=True)); pp /= pp.sum(1, keepdims=True)
    nu = pp.mean(0); nu /= nu.sum()                           # natural column (outcome) measure
    Lc = L
    if drop_base_measure:
        Lc = Lc - (w @ Lc)[None, :]                           # drop x-only base measure
    Lc = Lc - (Lc @ nu)[:, None]                              # drop per-row log-partition
    U, S, Vt = np.linalg.svd(np.sqrt(w)[:, None] * Lc * np.sqrt(nu)[None, :], full_matrices=False)
    E = (U / np.sqrt(w)[:, None]) * S
    D = Vt / np.sqrt(nu)[None, :]
    return Lc, E, D, S, nu


def factor_tensor(T):
    """Closed-form factorization of an order-2+ conditional tensor.

    The first axis indexes the items (the shared code); reshape the rest into columns,
    mean-center rows, and SVD. Returns ``(E, S)`` with ``E`` the singular-value-scaled code.
    Groups are degenerate (every frequency represents the operation equally well), so this
    returns one representative of the family — matching the demo_graphs order-2 gallery.
    """
    T = np.asarray(T, dtype=np.float64)
    M = T.reshape(T.shape[0], -1)
    U, S, _ = np.linalg.svd(M - M.mean(0), full_matrices=False)
    return U * S, S


def predicted_geometry(conditional):
    """Predicted item code ``E`` for a dataset's conditional.

    Dispatches on rank: a 2-D log-conditional -> ``factor_logcond``; a 3-D+ conditional
    tensor -> ``factor_tensor``. This is the single entry point the ``Evaluator`` compares
    trained representations against.
    """
    conditional = np.asarray(conditional, dtype=np.float64)
    if conditional.ndim == 2:
        return factor_logcond(conditional)[1]
    return factor_tensor(conditional)[0]
