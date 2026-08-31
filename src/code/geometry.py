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


def predicted_geometry(log_cond):
    """Predicted code ``E``, one row per input, for a log-conditional ``L[input, outcome]``."""
    return factor_logcond(log_cond)[1]


##### Superposition: predicted geometry of a rank-limited softmax (notes/superposition.tex) #####

def feature_objective(W, alpha, p):
    """The capacity surrogate: J = 1/2 sum_i (||E_i||^2 - alpha_i)^2
    + p/2 sum_{i != j} <E_i, E_j>^2, with rows of W as the E_i."""
    G = np.asarray(W, float) @ np.asarray(W, float).T
    s = np.diag(G)
    return float(0.5 * ((s - np.asarray(alpha, float)) ** 2).sum()
                 + 0.5 * p * ((G ** 2).sum() - (s ** 2).sum()))


def feature_sizes(alpha, p, d):
    """Squared strengths ||E_i||^2 at budget d and the capacity level alpha_star.
    One scalar sets three regimes: alpha_i > alpha_star gets a dedicated dimension at
    s = alpha_i; p alpha_star < alpha_i <= alpha_star shares at s = (alpha_i - p alpha_star)
    / (1 - p); weaker features get nothing. alpha_star spends the budget exactly
    (sum of clipped dimension fractions = d) and is found by scanning the set breakpoints:
    within fixed dedicated/active counts the budget is linear in alpha_star.
    Returns (sizes (m,), alpha_star)."""
    alpha = np.asarray(alpha, float)
    m = len(alpha)
    if m <= d:                                        # capacity does not bind
        return alpha.copy(), 0.0
    a = np.sort(alpha)[::-1]
    csum = np.concatenate([[0.0], np.cumsum(a)])
    for n_ded in range(d):                            # dedicated count; < d leaves shared room
        for n_act in range(n_ded + 1, m + 1):         # dedicated + shared count
            astar = (csum[n_act] - csum[n_ded]) / ((1 - p) * (d - n_ded) + p * (n_act - n_ded))
            ok_ded = (n_ded == 0 or a[n_ded - 1] > astar) and a[n_ded] <= astar
            ok_act = a[n_act - 1] > p * astar and (n_act == m or a[n_act] <= p * astar)
            if ok_ded and ok_act:
                sizes = np.where(alpha > astar, alpha,
                                 np.maximum(alpha - p * astar, 0.0) / (1 - p))
                return sizes, astar
    raise RuntimeError("no consistent capacity level; degenerate importance spectrum")


def _tight_frame(s, k):
    """Vectors with squared norms ``s`` spread evenly over k dimensions:
    V (n, k) with V^T V = (sum s / k) I. Feasible iff max s <= sum s / k.

    Chan-Li construction: keep a pool of mutually orthogonal vectors (initially sqrt(c) axes
    and zeros, outer-product sum c I). Serve the largest remaining target from the two pool
    vectors straddling it: rotating an orthogonal pair reaches any norm between theirs
    exactly, and replaces them by one leftover, preserving both the outer-product sum and
    the pool's orthogonality."""
    s = np.asarray(s, float)
    n, c = len(s), s.sum() / k
    pool = [np.sqrt(c) * np.eye(k)[i] for i in range(k)] + [np.zeros(k)] * (n - k)
    out = np.zeros((n, k))
    order = np.argsort(-s, kind="stable")
    for idx in order[:-1]:
        t = s[idx]
        norms = np.array([v @ v for v in pool])
        above, below = norms >= t, norms <= t
        assert above.any() and below.any(), "no straddling pair; majorization violated"
        i = int(np.flatnonzero(above)[np.argmin(norms[above])])   # smallest norm >= t
        j = int(np.flatnonzero(below)[np.argmax(norms[below])])   # largest norm <= t
        if i == j or norms[i] == norms[j]:                        # exact hit, no rotation
            out[idx] = pool.pop(i)
            continue
        u, v = pool[i], pool[j]
        co = np.sqrt((t - norms[j]) / (norms[i] - norms[j]))
        si = np.sqrt(1.0 - co ** 2)
        out[idx] = co * u + si * v
        pool = [w for l, w in enumerate(pool) if l not in (i, j)] + [-si * u + co * v]
    out[order[-1]] = pool[0]
    return out


def feature_directions(sizes, own_axis, d):
    """Vectors W (m, d) realizing the sizes: dedicated features on orthonormal axes, the
    rest an even spread over the remaining dimensions. Zero rows for dropped features."""
    sizes = np.asarray(sizes, float)
    own = np.flatnonzero(own_axis)
    shared = np.flatnonzero(~np.asarray(own_axis, bool) & (sizes > 0))
    W = np.zeros((len(sizes), d))
    W[own, np.arange(len(own))] = np.sqrt(sizes[own])
    if len(shared):
        W[shared, len(own):] = _tight_frame(sizes[shared], d - len(own))
    return W


def predicted_features(alpha, p, d):
    """The full prediction, no training: capacity level, sizes, directions.
    Returns (W (m, d), sizes (m,), own_axis (m,), alpha_star)."""
    alpha = np.asarray(alpha, float)
    sizes, astar = feature_sizes(alpha, p, d)
    own_axis = alpha > astar
    return feature_directions(sizes, own_axis, d), sizes, own_axis, astar


def fitted_features(alpha, p, d, n_starts=6, seed=0):
    """Best W found by minimizing feature_objective directly (L-BFGS, several starts).
    A brute-force reference for the closed form, not a prediction."""
    from scipy.optimize import minimize
    alpha = np.asarray(alpha, float)
    m = len(alpha)

    def fg(f):
        W = f.reshape(m, d)
        G = W @ W.T
        s = np.diag(G).copy()
        off = G - np.diag(s)
        J = 0.5 * ((s - alpha) ** 2).sum() + 0.5 * p * (off ** 2).sum()
        grad = 2.0 * (s - alpha)[:, None] * W + 2.0 * p * (off @ W)
        return J, grad.ravel()

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_starts):
        r = minimize(fg, 0.5 * rng.normal(size=m * d), jac=True, method="L-BFGS-B",
                     options=dict(maxiter=20000, maxfun=30000, ftol=1e-15, gtol=1e-12))
        if best is None or r.fun < best.fun:
            best = r
    return best.x.reshape(m, d)


def feature_vectors(H, Z):
    """Least-squares additive fit H ~ Z V + 1 v0: the vector each feature writes into the
    code. Returns (V (m, d), v0 (d,))."""
    Z = np.asarray(Z, float)
    X = np.column_stack([Z, np.ones(len(Z))])
    V = np.linalg.lstsq(X, np.asarray(H, float), rcond=None)[0]
    return V[:-1], V[-1]


def balanced_code(E, readout):
    """Gauge-fix a trained (code, readout) pair: row-centre the logits (the softmax shift
    gauge) and split the SVD scale evenly (the balance gradient descent finds).
    Returns (H (N, d), D (m, d)) with H @ D^T the centred logits."""
    G = np.asarray(E, float) @ np.asarray(readout, float).T
    G = G - G.mean(1, keepdims=True)
    d = E.shape[1]
    U, S, Vt = np.linalg.svd(G, full_matrices=False)
    root = np.sqrt(S[:d])
    return U[:, :d] * root, (root[:, None] * Vt[:d]).T


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


def _fantope_project(A, d):
    """Nearest symmetric H to A with 0 <= H <= I and tr H = d (eigenvalue water-filling)."""
    lam, V = np.linalg.eigh(A)
    lo, hi = lam.min() - 1.0, lam.max()
    for _ in range(100):
        nu = 0.5 * (lo + hi)
        lo, hi = (lo, nu) if np.clip(lam - nu, 0.0, 1.0).sum() < d else (nu, hi)
    return (V * np.clip(lam - 0.5 * (lo + hi), 0.0, 1.0)) @ V.T


def fantope_leverage(alpha, p, d, n_samples=65536, iters=2000, step=0.05, seed=0):
    """Convex capacity allocation: minimize E_z KL(p*(.|z) || softmax(H J (alpha o z)))
    over the Fantope {0 <= H <= I, tr H = d, H 1 = 0}, by projected gradient in a basis
    of 1-perp. Returns (h (m,) = diag H*, H* (m, m))."""
    from scipy.special import logsumexp
    alpha = np.asarray(alpha, float)
    m = len(alpha)
    rng = np.random.default_rng(seed)
    Z = (rng.random((n_samples, m)) < p).astype(float)
    G = Z * alpha
    P = np.exp(G - logsumexp(G, axis=1, keepdims=True))
    Q = np.linalg.svd(np.eye(m) - 1.0 / m)[0][:, :m - 1]
    LQ = (G - G.mean(1, keepdims=True)) @ Q
    H = (d / (m - 1)) * np.eye(m - 1)
    f_prev = np.inf
    for _ in range(iters):
        L = LQ @ H @ Q.T
        f = (logsumexp(L, axis=1) - (P * L).sum(1)).mean()
        if f > f_prev + 1e-12:
            step *= 0.5
        f_prev = f
        q = np.exp(L - logsumexp(L, axis=1, keepdims=True))
        grad = LQ.T @ ((q - P) @ Q) / n_samples
        H = _fantope_project(H - step * 0.5 * (grad + grad.T), d)
    Hf = Q @ H @ Q.T
    return np.diag(Hf).copy(), Hf


def predicted_map(alpha, p, d, n_samples=16384, n_starts=3, seed=0, maxiter=8000):
    """The theory's predicted feature -> output map: minimize exact CE over the
    tied-direction class (writer W_i, reader kappa_i W_i, bias) on a fixed-seed Bernoulli
    sample of the task. Returns (R (m, m), b (m,)) with R = W (kappa * W)^T."""
    import torch
    from scipy.optimize import minimize
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)                # tiny matmuls: multithreading only thrashes
    alpha = np.asarray(alpha, float)
    m = len(alpha)
    rng = np.random.default_rng(seed)
    Z = torch.tensor((rng.random((n_samples, m)) < p).astype(float))
    P = torch.softmax(Z * torch.tensor(alpha), 1)

    def fg(f):
        t = torch.tensor(f, requires_grad=True)
        W = t[:m * d].reshape(m, d); k = t[m * d:m * d + m]; b = t[m * d + m:]
        L = (Z @ W) @ (k[:, None] * W).T + b
        J = (torch.logsumexp(L, 1) - (P * L).sum(1)).mean()
        J.backward()
        return float(J.detach()), t.grad.numpy()

    best = None
    for _ in range(n_starts):
        x0 = np.concatenate([0.3 * rng.normal(size=m * d), np.ones(m), np.zeros(m)])
        r = minimize(fg, x0, jac=True, method="L-BFGS-B",
                     options=dict(maxiter=maxiter, maxfun=2 * maxiter, ftol=1e-13, gtol=1e-9))
        if best is None or r.fun < best.fun:
            best = r
    W = best.x[:m * d].reshape(m, d)
    kap = best.x[m * d:m * d + m]
    return W @ (kap[:, None] * W).T, best.x[m * d + m:]


def logit_features(L, Z, d):
    """Measured feature vectors of a logit matrix: balanced SVD of the row-centred
    logits, additive regression of the code on Z, rescaled to balanced units.
    Returns E (m, d) with sizes ||E_i||^2 comparable to alpha."""
    L = np.asarray(L, float)
    Z = np.asarray(Z, float)
    Lc = L - L.mean(1, keepdims=True)
    U, S, _ = np.linalg.svd(Lc, full_matrices=False)
    H = U[:, :d] * np.sqrt(S[:d])
    scale = np.sqrt(len(Z) * Z.mean() * (1 - Z.mean()))
    return np.sqrt(scale) * feature_vectors(H, Z)[0]


def pair_features(Z, n_top=18):
    """The extended dictionary: [z; z_a z_b for a < b among the first n_top features].
    Returns the raw (unstandardized) matrix (n, m + n_top(n_top-1)/2)."""
    Z = np.asarray(Z, float)
    pairs = [Z[:, a] * Z[:, b] for a in range(n_top) for b in range(a + 1, n_top)]
    return np.column_stack([Z] + pairs)


def dc_map(alpha, p, d, n_top=18, n_samples=8192, seed=0,
           lam_path=(0.02, 0.05, 0.1, 0.2, 0.5), inner=400, dc_steps=6):
    """Certified predictor of the extended feature -> output map: truncated-nuclear DC
    (each step a convex program, monotone, rank collapses to d without shrinking the kept
    spectrum) followed by a convex coefficient polish in the found output subspace.
    Returns (R (q, m), b (m,), mu (q,), sd (q,)): logits of words Zw are
    ((pair_features(Zw, n_top) - mu) / sd) @ R + b."""
    from scipy.optimize import minimize
    alpha = np.asarray(alpha, float)
    m = len(alpha)
    rng = np.random.default_rng(seed)
    Z = (rng.random((n_samples, m)) < p).astype(float)
    raw = pair_features(Z, n_top)
    mu, sd = raw.mean(0), raw.std(0)
    Phi = (raw - mu) / sd
    G = Z * alpha
    from scipy.special import logsumexp
    P = np.exp(G - logsumexp(G, axis=1, keepdims=True))
    lip = np.linalg.norm(Phi, 2) ** 2 / (2 * n_samples)
    step = 1.0 / lip

    def ce_grad(R, b):
        L = Phi @ R + b
        Ls = L - L.max(1, keepdims=True)
        q = np.exp(Ls); q /= q.sum(1, keepdims=True)
        ce = float(np.mean(np.log(np.exp(Ls).sum(1)) + L.max(1) - (P * L).sum(1)))
        return ce, Phi.T @ (q - P) / n_samples, (q - P).mean(0)

    R = np.zeros((Phi.shape[1], m)); b = np.zeros(m)
    for lam in lam_path:
        for _ in range(dc_steps):
            U, s, Vt = np.linalg.svd(R, full_matrices=False)
            W = U[:, :d] @ Vt[:d]
            for _ in range(inner):
                _, gR, gb = ce_grad(R, b)
                U, s, Vt = np.linalg.svd(R - step * (gR - lam * W), full_matrices=False)
                R = (U * np.maximum(s - step * lam, 0.0)) @ Vt
                b = b - step * gb
    Uout = np.linalg.svd(R, full_matrices=False)[2][:d].T   # (m, d)

    def fg(x):
        C = x[:Phi.shape[1] * d].reshape(Phi.shape[1], d); bb = x[Phi.shape[1] * d:]
        L = (Phi @ C) @ Uout.T + bb
        Ls = L - L.max(1, keepdims=True)
        q = np.exp(Ls); q /= q.sum(1, keepdims=True)
        J = float(np.mean(np.log(np.exp(Ls).sum(1)) + L.max(1) - (P * L).sum(1)))
        return J, np.concatenate([(Phi.T @ ((q - P) @ Uout) / n_samples).ravel(),
                                  (q - P).mean(0)])

    x0 = np.concatenate([(R @ Uout).ravel(), b])
    r = minimize(fg, x0, jac=True, method="L-BFGS-B",
                 options=dict(maxiter=8000, maxfun=16000, ftol=1e-13, gtol=1e-9))
    C = r.x[:Phi.shape[1] * d].reshape(Phi.shape[1], d)
    return C @ Uout.T, r.x[Phi.shape[1] * d:], mu, sd


def waterfill_h(alpha, p, d, outer=60, nodes=20):
    """The allocation formula: water-filling h solving g_i(h_i) = nu, sum h = d, with the
    pressure g_i = alpha_i p (E[p*_i|z_i=1] - E[q_i|z_i=1]) evaluated in closed form
    (lognormal moment-matched log-denominator, fixed Gauss-Hermite quadrature of the
    logistic). No samples, no matrix optimization. Returns (h (m,), nu)."""
    alpha = np.asarray(alpha, float)
    m = len(alpha)
    t, w = np.polynomial.hermite_e.hermegauss(nodes)
    w = w / w.sum()

    def cond_means(gains, b):
        e1, e0 = np.exp(gains + b), np.exp(b)
        mean_t = e0 + p * (e1 - e0)
        var_t = p * (1 - p) * (e1 - e0) ** 2
        mu_s = mean_t.sum() - mean_t
        s2 = np.log1p((var_t.sum() - var_t) / mu_s ** 2)
        mu = np.log(mu_s) - 0.5 * s2
        def emean(a):
            x = (a - mu)[:, None] - np.sqrt(s2)[:, None] * t
            return (1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))) @ w
        return emean(gains + b), emean(b)

    def project(x):
        lo, hi = x.min() - 1.0, x.max()
        for _ in range(100):
            nu = 0.5 * (lo + hi)
            lo, hi = (lo, nu) if np.clip(x - nu, 0, 1).sum() < d else (nu, hi)
        return np.clip(x - 0.5 * (lo + hi), 0, 1)

    m1, m0 = cond_means(alpha, np.zeros(m))
    target = p * m1 + (1 - p) * m0
    h = np.full(m, d / m)
    b = np.zeros(m)
    for _ in range(outer):
        for _ in range(40):                              # calibrate biases E[q] = E[p*]
            mu1, mu0 = cond_means(h * alpha, b)
            b = b + 0.8 * (np.log(target) - np.log(np.maximum(p * mu1 + (1 - p) * mu0,
                                                              1e-12)))
            b = b - b.mean()
        mu1, _ = cond_means(h * alpha, b)
        g = alpha * p * (m1 - mu1)
        inner = (h > 1e-3) & (h < 1 - 1e-3)
        nu = g[inner].mean() if inner.any() else g.mean()
        h = 0.5 * h + 0.5 * project(h + 2.0 * (g - nu))
    mu1, _ = cond_means(h * alpha, b)
    g = alpha * p * (m1 - mu1)
    rep = h > 1e-3
    return h, float(g[rep].mean())
