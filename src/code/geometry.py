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

    Returns ``(Lc, E, D, S, nu)``: the code ``E`` and decoder ``D``, each carrying the *balanced*
    (symmetric) share of the scale ``sqrt(S)``, so ``E @ D = Lc`` still holds. This matches the
    factorization gradient descent actually learns — from small init the two factors grow together
    (the ``codeᵀcode - readoutᵀreadout`` invariant stays ~0), so each ends up with ``sqrt(S)`` rather
    than the SVD's arbitrary all-scale-in-the-code convention (which leaves the decoder ``D = V`` flat).
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
    rootS = np.sqrt(S)                                        # split the scale symmetrically (balanced)
    E = (U / np.sqrt(w)[:, None]) * rootS
    D = (rootS[:, None] * Vt) / np.sqrt(nu)[None, :]
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


def spectral_hmm(pi, T, Em):
    """Hsu-Kakade-Zhang spectral learning of an HMM from its exact observable moments — the
    closed-form, observables-only belief oracle. Inputs: ``pi`` (n,), ``T`` (n,n) row-stochastic
    ``p(s'|s)``, ``Em`` (n,M) ``p(x|s)``. Returns ``(U, B, b1, b_inf)``: ``U`` (M,n) the observable
    subspace (top-n left singular vectors of the bigram co-occurrence — a closed-form belief
    geometry whose rank is the number of hidden states), ``B`` (M,n,n) the observable operators
    (the belief update is linear: ``b <- normalize(B[x] @ b)``), and ``b1``/``b_inf`` the initial
    state / normalizer. The spectral belief equals the forward-algorithm belief up to an invertible
    linear map, so predictions are identical.
    """
    pi = np.asarray(pi, np.float64); T = np.asarray(T, np.float64); Em = np.asarray(Em, np.float64)
    n, _ = Em.shape
    P1 = pi @ Em                                                      # p(x1)
    P21 = np.einsum("s,sj,st,ti->ij", pi, Em, T, Em)                 # p(x2=i, x1=j)
    P3 = np.einsum("s,sj,st,tx,tu,ui->xij", pi, Em, T, Em, T, Em)    # p(x3=i, x2=x, x1=j)
    U = np.linalg.svd(P21)[0][:, :n]                                 # observable subspace
    UP21_pinv = np.linalg.pinv(U.T @ P21)
    B = np.stack([(U.T @ P3[x]) @ UP21_pinv for x in range(len(P3))])
    return U, B, U.T @ P1, UP21_pinv.T @ P1


def spectral_beliefs(B, b1, b_inf, obs):
    """Per-position spectral belief states for ``obs`` (batch, L); each equals the forward-algorithm
    belief up to the fixed linear map from ``spectral_hmm``. Normalized every step to stay bounded."""
    obs = np.asarray(obs); out = np.zeros((*obs.shape, len(b1)))
    for r in range(obs.shape[0]):
        st = b1.copy()
        for t in range(obs.shape[1]):
            st = B[obs[r, t]] @ st; st = st / (b_inf @ st)
            out[r, t] = st
    return out


def hankel_forward_states(g, prefixes, suffixes, rank=None, tol=1e-9):
    """Spectral (Hankel) predictive-state basis for a PCFG's *string* distribution — the left-to-right
    analogue of ``spectral_hmm`` (which is the length-1 special case). Build the Hankel matrix
    ``H[u,v] = p(string = u·v)`` over prefix set ``prefixes`` and suffix set ``suffixes`` (each a
    sequence of terminal ids), using the exact inside chart. Its rank is the number of distinct
    predictive states; the SVD gives their canonical basis.

    Returns ``(forward_state, info)`` where ``forward_state(w)`` maps any prefix ``w`` to its rank-``r``
    canonical predictive state, and ``info`` holds ``H``, the singular values ``S``, and ``rank``.
    The forward state is the minimal sufficient statistic of the future given the past — i.e. the
    exact left-to-right belief, in the data-determined observable basis (up to rotation)."""
    from .data import pcfg_inside_chart
    _cache = {}

    def sprob(w):
        w = tuple(w)
        if len(w) == 0:
            return 0.0                                               # our grammars emit non-empty strings
        if w not in _cache:
            In = pcfg_inside_chart(np.asarray(w, dtype=int), g)
            _cache[w] = float(g.pi @ In[(0, len(w) - 1)])
        return _cache[w]

    suf = [list(v) for v in suffixes]
    H = np.array([[sprob(list(u) + v) for v in suf] for u in prefixes])
    U, S, Vt = np.linalg.svd(H, full_matrices=False)
    r = rank or int((S > tol * S[0]).sum())
    proj = Vt[:r].T @ np.diag(1.0 / np.sqrt(S[:r]))                 # future-vector -> canonical state

    def forward_state(w):
        rw = np.array([sprob(list(w) + v) for v in suf])
        return rw @ proj

    return forward_state, {"H": H, "S": S, "rank": r}
