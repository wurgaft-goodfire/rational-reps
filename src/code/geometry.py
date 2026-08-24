"""Predicted geometries: factor a task's conditional into a code ``E`` (the
geometry a network should learn) and a decoder ``D`` (the readout)."""
import numpy as np


def factor_logcond(L, drop_base_measure=False, w=None):
    """Correspondence-analysis SVD of a log-conditional ``L[theta, x]``: column-center
    (drop the per-row log-partition a softmax ignores); ``drop_base_measure=True`` also drops
    the per-outcome mean. ``w`` is the row (item) measure, uniform by default. Returns
    ``(Lc, E, D, S, nu)`` with the scale split as ``sqrt(S)`` on each factor (the balanced
    split gradient descent learns), so ``E @ D = Lc``."""
    L = np.asarray(L, dtype=np.float64)
    N, _ = L.shape
    w = np.full(N, 1.0 / N) if w is None else np.asarray(w, float) / np.sum(w)
    pp = np.exp(L - L.max(1, keepdims=True)); pp /= pp.sum(1, keepdims=True)
    nu = w @ pp; nu /= nu.sum()                               # natural column (outcome) measure
    Lc = L
    if drop_base_measure:
        Lc = Lc - (w @ Lc)[None, :]                           # drop x-only base measure
    Lc = Lc - (Lc @ nu)[:, None]                              # drop per-row log-partition
    U, S, Vt = np.linalg.svd(np.sqrt(w)[:, None] * Lc * np.sqrt(nu)[None, :], full_matrices=False)
    rootS = np.sqrt(S)                                        # split the scale symmetrically (balanced)
    E = (U / np.sqrt(w)[:, None]) * rootS
    D = (rootS[:, None] * Vt) / np.sqrt(nu)[None, :]
    return Lc, E, D, S, nu


def truncation_features(L, d, w=None):
    """Rank-d prefix of the correspondence-analysis factorization of ``L``: the truncation
    oracle at budget d. Returns (A (d, N), D (d, n_x)) for the identity lift."""
    _, E, D, _, _ = factor_logcond(L, w=w)
    return E[:, :d].T, D[:d]


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

##### Budgeted CE-optimal codes: truncation vs superposition #####

def support_targets(Z, Lf):
    """Target conditionals p(x|z) = softmax(Z @ Lf) per code row. (N, n_x)."""
    from scipy.special import softmax
    return softmax(Z @ Lf, axis=1)


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
    L-BFGS, plus one polish from a small fixed init (the basin training's small-init
    dynamics reach, which the path's truncations can miss); everything is deterministic.
    Best exact CE wins; the scale is split balanced. Returns (A (d, m), D (d, n_x), b (n_x), S)."""
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
    rng = np.random.default_rng(0)
    fit = _polish_features(0.01 * rng.normal(size=(d, m)), 0.01 * rng.normal(size=(d, nx)),
                           np.zeros(nx), Z, P, w, fit_bias)
    if fit[3] < best[3]:
        best = fit
    A, D, b, _ = best
    U2, S2, V2t = np.linalg.svd(A.T @ D)                          # plain balanced split
    rootS = np.sqrt(S2[:d])
    return (U2[:, :d] * rootS).T, rootS[:, None] * V2t[:d], b, S2[:d]


def feature_vectors(H, Z, w=None):
    """Weighted least-squares additive fit ``H ~ Z V + 1 v0``: effective feature vectors and
    the constant (mean-code) component. Returns (V (m, d), v0 (d,))."""
    H = np.asarray(H, float); Z = np.asarray(Z, float)
    N = len(Z)
    w = np.full(N, 1.0 / N) if w is None else np.asarray(w, float) / np.sum(w)
    X = np.column_stack([Z, np.ones(N)]) * np.sqrt(w)[:, None]
    V = np.linalg.lstsq(X, np.sqrt(w)[:, None] * H, rcond=None)[0]
    return V[:-1], V[-1]


def attribute_stats(G, Z):
    """Gauge-free per-attribute swing and intercept of logits ``G`` (N, m) on 0/1 features
    ``Z`` (N, m): least squares G[x, i] ~ a_i Z[x, i] + b_i + c_x, intercepts pinned by
    mean(c) = 0. Returns (a_hat (m,), b_hat (m,))."""
    G = np.asarray(G, float); Z = np.asarray(Z, float)
    N, m = Z.shape
    per_attribute = np.tile(np.eye(m), (N, 1))                    # one indicator column per attribute
    X = np.hstack([Z.ravel()[:, None] * per_attribute, per_attribute, np.repeat(np.eye(N), m, 0)])
    theta = np.linalg.lstsq(X, G.ravel(), rcond=None)[0]
    return theta[:m], theta[m:2 * m] + theta[2 * m:].mean()


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