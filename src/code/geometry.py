"""Closed-form predicted geometries: factor a task's conditional into a code ``E`` (the
geometry a network should learn) and a decoder ``D`` (the readout). Derivations live in
``src/notebooks/demo_graphs.ipynb``."""
import numpy as np


def factor_logcond(L, drop_base_measure=False):
    """Correspondence-analysis SVD of a log-conditional ``L[theta, x]``: column-center
    (drop the per-row log-partition a softmax ignores); ``drop_base_measure=True`` also drops
    the per-outcome mean. Returns ``(Lc, E, D, S, nu)`` with the scale split as ``sqrt(S)``
    on each factor (the balanced split gradient descent learns), so ``E @ D = Lc``."""
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
    """Factor an order-2+ conditional tensor: items on the first axis, remaining axes
    reshaped into columns, mean-centered, SVD. Returns ``(E, S)`` with ``E`` the
    singular-value-scaled item code."""
    T = np.asarray(T, dtype=np.float64)
    M = T.reshape(T.shape[0], -1)
    U, S, _ = np.linalg.svd(M - M.mean(0), full_matrices=False)
    return U * S, S


def predicted_geometry(conditional):
    """Predicted item code ``E`` for a dataset's conditional: 2-D log-conditional ->
    ``factor_logcond``; 3-D+ tensor -> ``factor_tensor``."""
    conditional = np.asarray(conditional, dtype=np.float64)
    if conditional.ndim == 2:
        return factor_logcond(conditional)[1]
    return factor_tensor(conditional)[0]


def spectral_hmm(pi, T, Em):
    """Hsu-Kakade-Zhang spectral HMM from exact moments ``pi`` (n,), ``T`` (n,n), ``Em`` (n,M).
    Returns ``(U, B, b1, b_inf)``: observable subspace (M,n), operators (M,n,n) with linear
    belief update ``b <- normalize(B[x] @ b)``, and initial state / normalizer. Equals the
    forward-algorithm belief up to a fixed invertible map."""
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
    """Spectral predictive-state basis for a PCFG's string distribution: SVD of the Hankel
    matrix ``H[u,v] = p(string = u·v)``, weighted correspondence-analysis style by the prefix
    measure ``w`` and suffix measure ``nu`` (uniform by default). Returns ``(forward_state,
    info)``: ``forward_state(prefix)`` -> rank-``r`` canonical state; ``info`` holds ``H``,
    ``S``, ``rank``."""
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
    """Exact predictive state at every distinct prefix of a corpus:
    ``F[u, v] = p(continuation begins with test v | prefix u)`` plus an end-of-sentence column,
    tests being the ``max_tests`` most frequent continuations of length ``1..test_len``.
    Returns ``(prefixes, F, info)`` with ``info["rank"]`` the numeric rank of ``F`` and
    ``info["loc"]`` each prefix's first ``(sequence, position)`` occurrence."""
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


##### Budgeted CE-optimal codes: truncation vs superposition #####

def support_targets(Z, Lf):
    """Target conditionals p(x|z) = softmax(Z @ Lf) per code row. (N, n_x)."""
    from scipy.special import softmax
    return softmax(Z @ Lf, axis=1)


def _psd_sqrt(M):
    lam, V = np.linalg.eigh(M)
    lam = np.clip(lam, 0.0, None)
    return (V * np.sqrt(lam)) @ V.T


def factor_features(Lf, Z, d, w=None, fit_bias=True):
    """Closed-form rank-d feature code under the Fisher surrogate: whiten by the feature
    second moment and the outcome Fisher, truncate, unwhiten with a balanced sqrt(S) split.
    With ``fit_bias`` the code is centered and b absorbs the mean logit row; without, b = 0.
    Returns (A (d, m), D (d, n_x), b (n_x), S)."""
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


def _prox_nuclear(R, t):
    """Soft-threshold the spectrum of ``R`` by ``t`` (the proximal map of t*||R||_*)."""
    U, S, Vt = np.linalg.svd(R, full_matrices=False)
    return (U * np.maximum(S - t, 0.0)) @ Vt


def _solve_penalized_ce(P, X, w, lam, n_pen, R0, max_iter=10000, tol=1e-10):
    """FISTA global minimum over R of sum_x w_x CE(P[x], softmax(X[x] @ R)) plus
    lam * ||R[:n_pen]||_* (rows past n_pen — the bias row — are unpenalized). Returns R."""
    from scipy.special import logsumexp
    step = 2.0 / np.linalg.eigvalsh(X.T @ (X * w[:, None]))[-1]   # 1/L, L = lmax(X'WX)/2

    def objective(R):
        G = X @ R
        ce = (w * (logsumexp(G, axis=1) - (P * G).sum(1))).sum()
        return ce + lam * np.linalg.svd(R[:n_pen], compute_uv=False).sum()

    R, Y, t, f_prev = R0.copy(), R0.copy(), 1.0, np.inf
    for it in range(max_iter):
        G = X @ Y
        Q = np.exp(G - logsumexp(G, axis=1, keepdims=True))
        Rn = Y - step * (X.T @ ((Q - P) * w[:, None]))
        Rn[:n_pen] = _prox_nuclear(Rn[:n_pen], step * lam)
        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        Y = Rn + ((t - 1.0) / t_next) * (Rn - R)
        R, t = Rn, t_next
        if it % 20 == 19:
            f = objective(R)
            if f > f_prev:                                        # restart the momentum
                Y, t = R.copy(), 1.0
            elif f_prev - f < tol * max(1.0, abs(f)):
                break
            f_prev = f
    return R


def _polish_features(A0, D0, b0, Z, P, w, fit_bias, maxiter=2000):
    """Joint L-BFGS on exact weighted CE from a fixed init (deterministic basin descent).
    Returns (A, D, b, ce)."""
    from scipy.special import logsumexp
    from scipy.optimize import minimize
    (d, m), nx = A0.shape, P.shape[1]

    def unpack(theta):
        A = theta[:d * m].reshape(d, m)
        D = theta[d * m:d * m + d * nx].reshape(d, nx)
        b = theta[d * m + d * nx:] if fit_bias else np.zeros(nx)
        return A, D, b

    def fg(theta):
        A, D, b = unpack(theta)
        G = Z @ A.T @ D + b
        Q = np.exp(G - logsumexp(G, 1, keepdims=True))
        ce = (w * (logsumexp(G, 1) - (P * G).sum(1))).sum()
        E = (Q - P) * w[:, None]
        grads = [(Z.T @ E @ D.T).T.ravel(), (A @ Z.T @ E).ravel()]
        if fit_bias:
            grads.append(E.sum(0))
        return ce, np.concatenate(grads)

    theta0 = np.concatenate([A0.ravel(), D0.ravel()] + ([b0] if fit_bias else []))
    r = minimize(fg, theta0, jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter))
    A, D, b = unpack(r.x)
    return A, D, b, r.fun


def nuclear_features(Lf, Z, d, w=None, fit_bias=True, path=(0.5, 0.2, 0.08, 0.03, 0.01, 0.003)):
    """Rank-d code via the nuclear-norm regularization path of the exact-CE fit: the convex
    program CE(Z R + b) + lam*||R||_* is solved globally along a decreasing lam ladder
    (warm-started continuation), each solution is truncated to rank d and polished by joint
    L-BFGS — the path proposes the basin, the polish converges within it; everything is
    deterministic. Best exact CE wins; the scale is split balanced.
    Returns (A (d, m), D (d, n_x), b (n_x), S)."""
    Lf = np.asarray(Lf, float); Z = np.asarray(Z, float)
    N, m = Z.shape
    w = np.full(N, 1.0 / N) if w is None else np.asarray(w, float) / np.sum(w)
    P = support_targets(Z, Lf)
    nx = P.shape[1]
    X = np.column_stack([Z, np.ones(N)]) if fit_bias else Z
    q = w @ P                                                     # bias-only optimum
    Q0 = np.tile(q if fit_bias else np.full(nx, 1.0 / nx), (N, 1))
    lam_hi = np.linalg.norm(Z.T @ ((Q0 - P) * w[:, None]), 2)     # above: R = 0

    R, best = np.zeros((X.shape[1], nx)), None
    for frac in path:
        R = _solve_penalized_ce(P, X, w, frac * lam_hi, m, R)
        U, S, Vt = np.linalg.svd(R[:m])
        rootS = np.sqrt(S[:d])
        b0 = R[m] if fit_bias else np.zeros(nx)
        fit = _polish_features((U[:, :d] * rootS).T, rootS[:, None] * Vt[:d], b0,
                               Z, P, w, fit_bias)
        if best is None or fit[3] < best[3]:
            best = fit
    A, D, b, _ = best
    U2, S2, V2t = np.linalg.svd(A.T @ D)                          # plain balanced split
    rootS = np.sqrt(S2[:d])
    return (U2[:, :d] * rootS).T, rootS[:, None] * V2t[:d], b, S2[:d]


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
