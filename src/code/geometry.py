"""Predicted geometries: factor a task's conditional into a code ``E`` (the
geometry a network should learn) and a decoder ``D`` (the readout)."""
import numpy as np

# A zero-probability outcome has no finite log. Every conditional is floored here before it is
# factored, so the oracle sees one floor everywhere instead of a per-caller magic number.
LOG_FLOOR = -20.0


def log_conditional(P):
    """``log p(outcome | input)``, with *impossible* outcomes set to ``LOG_FLOOR``.

    Only exact zeros are floored. Clipping small-but-real probabilities would flatten a
    distribution's tails onto a constant and invent structure the task does not have — a Gaussian
    response kernel reaching exp(-32) is rank 3, but rank 19 once its tail is clipped."""
    P = np.asarray(P, dtype=np.float64)
    return np.where(P > 0, np.log(np.where(P > 0, P, 1.0)), LOG_FLOOR)


def factor_logcond(L):
    """SVD of the row-centered log-conditional ``L[theta, x]`` (dropping the per-row
    log-partition a softmax ignores). Returns ``(Lc, E, D, S)``, the scale split as
    ``sqrt(S)`` on each factor (the balanced split gradient descent learns), so ``E @ D = Lc``."""
    L = np.asarray(L, dtype=np.float64)
    Lc = L - L.mean(1, keepdims=True)
    U, S, Vt = np.linalg.svd(Lc, full_matrices=False)
    rootS = np.sqrt(S)                                        # split the scale symmetrically
    return Lc, U * rootS, rootS[:, None] * Vt, S


def truncation_features(L, d):
    """Rank-d prefix of the factorization of ``L``: the truncation oracle at budget d.
    Returns (A (d, N), D (d, n_x)) for the identity lift."""
    _, E, D, _ = factor_logcond(L)
    return E[:, :d].T, D[:d]


def fourier_basis(p):
    """Real DFT modes of ``Z_p`` as rows: cos/sin at each frequency, Nyquist once when p is even.
    The canonical gauge for group operations (modular add/mult), whose factorization is degenerate
    across frequencies."""
    x = 2 * np.pi * np.arange(p) / p
    rows = [np.ones(p)]
    for k in range(1, p // 2 + 1):
        rows.append(np.cos(k * x))
        if 2 * k != p:                                        # Nyquist has no sine partner
            rows.append(np.sin(k * x))
    return np.array(rows[1:])                                 # drop the constant: centering killed it


def canonical_gauge(E, D, basis, tol=1e-6):
    """Re-express a factorization in ``basis`` within each degenerate singular block, leaving
    ``E @ D`` unchanged. Blocks are runs of rows of ``D`` with (nearly) equal norm — exactly the
    directions the SVD leaves arbitrary, so any orthonormal set spanning the block is as valid.
    Each block takes the ``basis`` rows that live in its span (greedily, best-aligned first),
    orthonormalized. Returns ``(E R, R.T @ D)``."""
    E, D, basis = (np.asarray(a, float) for a in (E, D, basis))
    norms = np.linalg.norm(D, axis=1)
    R = np.eye(len(D))
    start = 0
    for stop in list(np.flatnonzero(np.diff(norms) < -tol * max(norms.max(), 1e-12)) + 1) + [len(D)]:
        blk, start = slice(start, stop), stop
        k = stop - blk.start
        if k < 2:
            continue                                          # a lone direction is already pinned
        Q = D[blk] / np.linalg.norm(D[blk], axis=1, keepdims=True)   # orthonormal rows of the block
        coords = basis @ Q.T                                  # each target mode in block coordinates
        keep = np.linalg.norm(coords, axis=1) > 1 - tol       # the modes that lie inside the span
        chosen, taken = [], np.zeros((0, k))
        for c in coords[keep]:                                # greedy Gram-Schmidt over those modes
            r = c - taken.T @ (taken @ c) if len(taken) else c
            if np.linalg.norm(r) > tol:
                r /= np.linalg.norm(r)
                chosen.append(r); taken = np.vstack([taken, r])
        if len(chosen) == k:
            R[blk, blk] = np.array(chosen).T
    return E @ R, R.T @ D


def class_means(X, labels, n=None):
    """Mean code per label and the label counts: ``((n, d), (n,))``."""
    n = n if n is not None else int(np.max(labels)) + 1
    M = (np.asarray(labels)[:, None] == np.arange(n)).astype(float)
    count = M.sum(0)
    return M.T @ X / count[:, None], count


def group_fraction(X, labels, n=None):
    """Share of code variance explained by ``labels`` alone: 1 when the code is a function of the
    label, about 1/n for a code unrelated to it. ``X`` (rows = items) is centered first."""
    X = np.asarray(X, float)
    X = X - X.mean(0)
    m, count = class_means(X, labels, n)
    return float((count[:, None] * m ** 2).sum() / (X ** 2).sum())


def frequency_plane(X, labels, k, n=None):
    """Codes projected onto the plane that frequency ``k`` of a cyclic ``labels`` spans, with that
    frequency's share of the class-mean energy and its circularity (sigma_2 / sigma_1 of the plane,
    1 for a true circle). Returns ``(xy (len(X), 2), energy, circularity)``."""
    X = np.asarray(X, float)
    X = X - X.mean(0)
    m, _ = class_means(X, labels, n)
    ang = 2 * np.pi * k * np.arange(len(m)) / len(m)
    B = np.c_[np.cos(ang), np.sin(ang)].T @ m                 # the two mode coordinates, (2, d)
    sv = np.linalg.svd(B, compute_uv=False)
    energy = (2 / len(m)) * (B ** 2).sum() / (m ** 2).sum()
    return X @ np.linalg.qr(B.T)[0], float(energy), float(sv[1] / max(sv[0], 1e-12))


def circulant_fraction(X, labels, n=None):
    """Share of the class-mean Gram explained by its translation-invariant part: 1 when the Gram
    depends only on the label difference (a ring), lower when it does not. Radii may differ across
    frequencies without cost, so this tests the symmetry, not the spectrum."""
    m, _ = class_means(np.asarray(X, float) - np.mean(X, 0), labels, n)
    m = m / np.linalg.norm(m, axis=1, keepdims=True).mean()
    G = m @ m.T
    off = (np.arange(len(m))[:, None] - np.arange(len(m))[None, :]) % len(m)
    C = np.zeros_like(G)
    for o in range(len(m)):                                   # average each diagonal offset
        C[off == o] = G[off == o].mean()
    return float(1 - ((G - C) ** 2).sum() / (G ** 2).sum())


def predicted_geometry(log_cond):
    """Predicted code ``E``, one row per input, for a log-conditional ``L[input, outcome]``."""
    return factor_logcond(log_cond)[1]

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


def capacity_allocation(A, D, d):
    """Per-outcome share of the d code dimensions: ``s`` (n_x,) summing to d. Centres the
    softmax row-shift out of ``G = A.T @ D``, rebalances to the gauge gradient descent picks,
    and splits the budget by squared decoder-column norm."""
    G = np.asarray(A, float).T @ np.asarray(D, float)
    G = G - G.mean(1, keepdims=True)                          # kill the softmax shift gauge
    _, S, Vt = np.linalg.svd(G, full_matrices=False)
    n2 = np.linalg.norm(np.sqrt(S[:d])[:, None] * Vt[:d], axis=0) ** 2
    return d * n2 / n2.sum()


def predicted_allocation(alpha, p, d):
    """Weak-coupling closed form for the capacity share ``s`` (m,), summing to d, of a
    feature cube with importances ``alpha`` and Bernoulli(p) features. Small logits make CE
    a Frobenius loss, so the rank-d code is the top-d PCA of the row-centred log-conditional,
    whose second moment is ``p^2 H aa' H + p(1-p) H diag(a^2) H`` with H the centring
    projector. The balanced decoder carries sqrt(S), hence the sqrt(lambda) weighting."""
    alpha = np.asarray(alpha, float)[:, None]
    m = len(alpha)
    H = np.eye(m) - 1.0 / m
    Ha = H @ alpha
    Mmat = p * p * (Ha @ Ha.T) + p * (1 - p) * (H @ np.diag(alpha[:, 0] ** 2) @ H)
    lam, V = np.linalg.eigh(Mmat)
    lam, V = np.maximum(lam[::-1][:d], 0.0), V[:, ::-1][:, :d]
    n2 = (V ** 2 * np.sqrt(lam)[None, :]).sum(1)
    return d * n2 / n2.sum()


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


def feature_stats(G, Z):
    """Gauge-free per-feature swing and intercept of logits ``G`` (N, m) on 0/1 features
    ``Z`` (N, m): least squares G[x, i] ~ a_i Z[x, i] + b_i + c_x, intercepts pinned by
    mean(c) = 0. Returns (a_hat (m,), b_hat (m,))."""
    G = np.asarray(G, float); Z = np.asarray(Z, float)
    N, m = Z.shape
    per_attribute = np.tile(np.eye(m), (N, 1))                    # one indicator column per feature
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