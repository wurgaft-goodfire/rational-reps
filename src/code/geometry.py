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


def hankel_forward_states(g, prefixes, suffixes, rank=None, tol=1e-9, w=None, nu=None):
    """Spectral (Hankel) predictive-state basis for a PCFG's *string* distribution — the left-to-right
    analogue of ``spectral_hmm`` (which is the length-1 special case). Build the Hankel matrix
    ``H[u,v] = p(string = u·v)`` over prefix set ``prefixes`` and suffix set ``suffixes`` (each a
    sequence of terminal ids), using the exact inside chart. Its rank is the number of distinct
    predictive states; the SVD gives their canonical basis.

    ``w`` (prefix measure) and ``nu`` (suffix measure) weight the factorization correspondence-analysis
    style (``√`` on each side of ``H``, as in ``factor_logcond``): the SVD then minimises a
    ``w ⊗ nu``-weighted least-squares fit of the Hankel, concentrating the basis on the frequent
    prefixes/suffixes the model's loss actually weights. Both default to uniform ``1/n``; pass
    empirical frequencies to weight by how often each prefix/suffix occurs (e.g. PCFGs with non-uniform
    sentence frequencies). The row weight ``w`` shapes which directions are principal but cancels from
    the forward-state formula; ``nu`` enters the projection.

    Returns ``(forward_state, info)`` where ``forward_state(w)`` maps any prefix to its rank-``r``
    canonical predictive state, and ``info`` holds ``H``, the singular values ``S``, and ``rank``.
    The forward state is the minimal sufficient statistic of the future given the past — i.e. the
    exact left-to-right belief, in the data-determined observable basis (up to rotation)."""
    from .data import pcfg_inside_chart
    _cache = {}

    def sprob(pref):
        pref = tuple(pref)
        if len(pref) == 0:
            return 0.0                                               # our grammars emit non-empty strings
        if pref not in _cache:
            In = pcfg_inside_chart(np.asarray(pref, dtype=int), g)
            _cache[pref] = float(g.pi @ In[(0, len(pref) - 1)])
        return _cache[pref]

    suf = [list(v) for v in suffixes]
    H = np.array([[sprob(list(u) + v) for v in suf] for u in prefixes])
    nP, nS = H.shape
    w = np.full(nP, 1.0 / nP) if w is None else np.asarray(w, float) / np.sum(w)      # prefix measure
    nu = np.full(nS, 1.0 / nS) if nu is None else np.asarray(nu, float) / np.sum(nu)  # suffix measure
    M = np.sqrt(w)[:, None] * H * np.sqrt(nu)[None, :]              # correspondence-analysis weighting
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    r = rank or int((S > tol * S[0]).sum())
    proj = (np.sqrt(nu)[:, None] * Vt[:r].T) @ np.diag(1.0 / np.sqrt(S[:r]))   # future -> canonical state

    def forward_state(pref):
        rw = np.array([sprob(list(pref) + v) for v in suf])
        return rw @ proj

    return forward_state, {"H": H, "S": S, "rank": r}


def pcfg_predictive_states(g, seqs, test_len=4, max_tests=60):
    """Exact predictive state at every *distinct* prefix of a corpus, as ``(prefixes, F, info)``.

    The predictive state is the vector of predictions for a set of **tests** (the PSR object):
    ``F[u, v] = p(the continuation begins with v | prefix u)``, plus a final column for the test
    "the sentence ends here". Two prefixes share a predictive state exactly when their test
    predictions agree, and the next-token distribution is literally the length-1 test columns — so
    a model that predicts optimally *and* cannot re-read the prefix must carry this vector.

    This is a strictly stronger object than the next-emitter belief, which marginalizes over one
    latent node and so forgets the rest of the stack (for ``anbn`` that belief is rank 2). Unlike
    the whitened Hankel basis of ``hankel_forward_states`` it needs no rank/tolerance choice: no
    ``1/√S`` factor to amplify near-null directions. Tests are the ``max_tests`` most frequent
    observed continuation-prefixes of length ``1..test_len``; ``info["rank"]`` is the numeric rank
    of ``F``, which saturates in ``test_len`` once the tests separate every predictive state.
    ``info["loc"]`` holds each prefix's first ``(sequence, position)`` occurrence, so a caller can
    index model hidden states in the same row order as ``F``."""
    from collections import Counter
    from .data import pcfg_inside_chart, pcfg_prefix_chart
    _cache = {}

    def charts(x):
        x = tuple(x)
        if x not in _cache:
            In = pcfg_inside_chart(np.asarray(x, dtype=int), g)
            PIn = pcfg_prefix_chart(np.asarray(x, dtype=int), g, In)
            _cache[x] = (float(g.pi @ PIn[(0, len(x) - 1)]), float(g.pi @ In[(0, len(x) - 1)]))
        return _cache[x]                                             # (p(prefix x), p(sentence == x))

    first = {}
    for b, s in enumerate(seqs):
        for t in range(len(s) + 1):
            first.setdefault(tuple(s[:t]), (b, t))
    prefixes = sorted(first)
    counts = Counter(tuple(s[t:t + k]) for s in seqs for t in range(len(s))
                     for k in range(1, test_len + 1) if t + k <= len(s))
    tests = [v for v, _ in counts.most_common(max_tests)]
    F = np.empty((len(prefixes), len(tests) + 1))
    for i, u in enumerate(prefixes):
        Zu = 1.0 if not u else charts(u)[0]
        F[i, :-1] = [charts(u + v)[0] / Zu for v in tests]
        F[i, -1] = (charts(u)[1] / Zu) if u else 0.0                 # sentences are non-empty
    return prefixes, F, {"tests": tests, "rank": int(np.linalg.matrix_rank(F)),
                         "loc": np.array([first[u] for u in prefixes])}


##### Budgeted CE-optimal codes (rungs 3'/4'): truncation vs superposition #####

def support_targets(Z, Lf):
    """Target conditionals p(x|z) = softmax(Z @ Lf) per code row. (N, n_x)."""
    from scipy.special import softmax
    return softmax(Z @ Lf, axis=1)


def _psd_sqrt(M):
    lam, V = np.linalg.eigh(M)
    lam = np.clip(lam, 0.0, None)
    return (V * np.sqrt(lam)) @ V.T


def factor_features(Lf, Z, d, w=None, fit_bias=True):
    """Closed-form rank-d feature code under the Fisher surrogate (rung 3').

    Weighted fit of logits = z^T A^T D (+ b) to the exact z^T Lf, whitened by the feature
    second moment and Fbar^{1/2} (outcome Fisher at the marginal), truncated in whitened
    space, unwhitened with a balanced sqrt(S) split. With ``fit_bias`` the bias absorbs the
    mean logit row and the code is centered first; without, the fit is raw (b = 0), matching
    bias-free readouts. Returns (A (d, m), D (d, n_x), b (n_x), S).
    """
    Lf = np.asarray(Lf, float); Z = np.asarray(Z, float)
    N, m = Z.shape
    w = np.full(N, 1.0 / N) if w is None else np.asarray(w, float) / np.sum(w)
    q = w @ support_targets(Z, Lf)
    Fh = _psd_sqrt(np.diag(q) - np.outer(q, q))
    zbar = w @ Z if fit_bias else np.zeros(m)
    Zc = Z - zbar
    C = (Zc * w[:, None]).T @ Zc
    Ch = _psd_sqrt(C)
    Chi, Fhi = np.linalg.pinv(Ch), np.linalg.pinv(Fh)
    U, S, Vt = np.linalg.svd(Ch @ Lf @ Fh, full_matrices=False)
    rootS = np.sqrt(S[:d])
    A = (Chi @ (U[:, :d] * rootS)).T
    D = (rootS[:, None] * Vt[:d]) @ Fhi
    b = zbar @ (Lf - A.T @ D)
    return A, D, b, S


def expected_ce(Z, Lf, A, D, b, w=None):
    """Weighted expected CE (nats) of the linear-code model, and the Bayes CE. Returns (ce, bayes)."""
    from scipy.special import logsumexp, xlogy
    Z = np.asarray(Z, float)
    N = len(Z)
    w = np.full(N, 1.0 / N) if w is None else np.asarray(w, float) / np.sum(w)
    P = support_targets(Z, Lf)
    G = Z @ A.T @ D + b
    logq = G - logsumexp(G, axis=1, keepdims=True)
    ce = -(w[:, None] * P * logq).sum()
    bayes = -(w[:, None] * xlogy(P, P)).sum()
    return ce, bayes


def als_features(Lf, Z, d, init, w=None, ridge=1e-4, n_rounds=100, tol=1e-8, fit_bias=True):
    """Exact-CE refinement (rung 4'): alternate the two convex halves (D,b) and A, each
    solved with L-BFGS. ``fit_bias=False`` freezes b at its init (for comparing against
    bias-free readouts). The closed-form warm start can be a *local* minimum (truncation) —
    add random restarts when hunting the global optimum. Returns (A, D, b, ce_path) with
    ce_path the ridge-free CE per round."""
    import torch
    Z_ = np.asarray(Z, float)
    N = len(Z_)
    w_ = np.full(N, 1.0 / N) if w is None else np.asarray(w, float) / np.sum(w)
    Pt = torch.tensor(support_targets(Z_, Lf))
    Zt, wt = torch.tensor(Z_), torch.tensor(w_)
    A0, D0, b0 = init
    At = torch.tensor(np.asarray(A0, float).T.copy(), requires_grad=True)
    D = torch.tensor(np.asarray(D0, float), requires_grad=True)
    b = torch.tensor(np.asarray(b0, float), requires_grad=fit_bias)

    def ce():
        return -(wt[:, None] * Pt * torch.log_softmax(Zt @ At @ D + b, dim=1)).sum()

    def solve(params, frozen):
        for t in frozen:                                  # don't backprop through the fixed half
            t.requires_grad_(False)
        opt = torch.optim.LBFGS(params, max_iter=200, tolerance_grad=1e-12,
                                tolerance_change=1e-14, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = ce() + ridge * (At.square().sum() + D.square().sum())
            loss.backward()
            return loss

        opt.step(closure)
        for t in frozen:
            t.requires_grad_(True)

    ce_path = [ce().item()]
    for _ in range(n_rounds):
        solve([D, b] if fit_bias else [D], [At])
        solve([At], [D, b] if fit_bias else [D])
        ce_path.append(ce().item())
        if ce_path[-2] - ce_path[-1] < tol:
            break
    return At.detach().numpy().T, D.detach().numpy(), b.detach().numpy(), ce_path


def superposition_measure(A):
    """Elhage et al.'s per-feature superposition score sum_{j != i} (a_hat_i . a_j)^2 for
    feature columns A (d, m); 0 = the feature has a dimension to itself."""
    A = np.asarray(A, float)
    nrm = np.linalg.norm(A, axis=0)
    Ahat = A / np.where(nrm > 0, nrm, 1.0)
    G = Ahat.T @ A
    return (G ** 2).sum(1) - np.diag(G) ** 2


def _row_space(M, tol=1e-9):
    U, s, _ = np.linalg.svd(np.asarray(M, float).T, full_matrices=False)
    return U[:, s > tol * s[0]]


def principal_angles(M1, M2):
    """Principal angles (radians, ascending) between the row spaces of M1 and M2."""
    s = np.linalg.svd(_row_space(M1).T @ _row_space(M2), compute_uv=False)
    return np.arccos(np.clip(s, -1.0, 1.0))
