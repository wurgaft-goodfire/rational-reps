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
    log-partition a softmax ignores). Returns ``(Lc, E, D, S)``, the scale split as ``sqrt(S)``
    on each factor (the balanced split gradient descent learns), so ``E @ D = Lc``; slice the
    leading columns of ``E`` and rows of ``D`` to show the top modes."""
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
    """The representation the loss pins down when capacity suffices: ``E``, one row per input,
    the balanced factor of the log-conditional ``L[input, outcome]``. For the rank-constrained
    case see `predicted_rank_constrained_geometry`."""
    return factor_logcond(log_cond)[1]


##### Superposition: first-order geometry of a rank-limited softmax (notes/superposition.tex) #####

def whiten(Z, tol=1e-10):
    """Predictive modes of binary features, u = z C^{-1/2} with C the covariance of z and
    the symmetric (pseudo-inverse) root, so mode k stays closest to feature k. Returns
    (U (n, m), C_half (m, m)) with Z = U @ C_half; the Bayes map on the modes is
    diag(alpha) @ C_half.T. Exactly collinear features share a mode: merge them first."""
    Z = np.asarray(Z, float)
    lam, V = np.linalg.eigh(np.cov(Z.T, bias=True))
    keep = lam > tol * lam.max()
    root = np.where(keep, np.sqrt(np.where(keep, lam, 1.0)), 0.0)
    inv_root = np.where(keep, 1.0 / np.where(keep, root, 1.0), 0.0)
    return Z @ (V * inv_root) @ V.T, (V * root) @ V.T


def modes(alpha, Z, correlated):
    """The coordinates the theory and the measurement share: the features themselves when
    independent, else the predictive modes with the Bayes map expressed on them (`whiten`).
    Returns (alpha (m,) or W (m, m), X (n, m))."""
    if not correlated:
        return np.asarray(alpha, float), np.asarray(Z, float)
    U, C_half = whiten(Z)
    return np.diag(alpha) @ C_half.T, U


def _logit_map(alpha, Z):
    """The Bayes logit map W (outcomes, modes): a vector alpha is diag(alpha) (logits
    alpha_i z_i); an (m, r) matrix is used as is (logits z W^T)."""
    alpha = np.asarray(alpha, float)
    return np.diag(alpha) if alpha.ndim == 1 else alpha


def _softmax_stats(alpha, Z):
    """Target probabilities and centred target logits for a dataset. Returns (G, P, Gc)."""
    from scipy.special import logsumexp
    G = np.asarray(Z, float) @ _logit_map(alpha, Z).T
    P = np.exp(G - logsumexp(G, axis=1, keepdims=True))
    return G, P, G - G.mean(1, keepdims=True)


def _field(W, Z, q, P):
    """Gradient of the mean CE with respect to the per-mode leverages at logits
    (z d) W^T: mean_n z_k ((q - P) W)_k. Returns (r,)."""
    return (Z * ((q - P) @ W)).mean(0)


def _excess_ce(L, G, P):
    """Excess CE of logits L against the target (G, P) on the same dataset."""
    from scipy.special import logsumexp
    bayes = float((logsumexp(G, axis=1) - (P * G).sum(1)).mean())
    return float((logsumexp(L, axis=1) - (P * L).sum(1)).mean()) - bayes


def _box_simplex(v, d):
    """Euclidean projection onto {0 <= x <= 1, sum x = d} by bisection on the shift."""
    lo, hi = v.min() - 1.0, v.max()
    for _ in range(100):
        t = 0.5 * (lo + hi)
        lo, hi = (lo, t) if np.clip(v - t, 0.0, 1.0).sum() < d else (t, hi)
    return np.clip(v - 0.5 * (lo + hi), 0.0, 1.0)


def allocate_d(alpha, Z, d, iters=2000, step=2.0):
    """The convex dimensional allocation on a dataset: minimize the CE of the
    fractional predictor softmax((z d) W^T) over {0 <= d_k <= 1, sum d_k = d} by
    projected gradient with early stop; W = diag(alpha) for a vector alpha. Returns
    (di (r,), delta), delta the interior marginal-utility level."""
    from scipy.special import logsumexp
    W = _logit_map(alpha, Z)
    Z = np.asarray(Z, float)
    m = Z.shape[1]
    G, P, _ = _softmax_stats(W, Z)
    di = np.full(m, d / m)
    f_prev, stall = np.inf, 0

    def project(v):
        return _box_simplex(v, d)

    for _ in range(iters):
        L = (Z * di) @ W.T
        f = float((logsumexp(L, axis=1) - (P * L).sum(1)).mean())
        if f > f_prev - 1e-12:
            step *= 0.5
            stall += 1
            if stall > 40:
                break
        else:
            stall = 0
        f_prev = f
        q = np.exp(L - logsumexp(L, axis=1, keepdims=True))
        di = project(di - step * _field(W, Z, q, P))
    L = (Z * di) @ W.T
    q = np.exp(L - logsumexp(L, axis=1, keepdims=True))
    u = -_field(W, Z, q, P)
    interior = (di > 1e-3) & (di < 1 - 1e-3)
    return di, float(u[interior].mean())


def tight_frame(di, seed=0, iters=4000, init=None):
    """A Parseval tight frame with prescribed row norms ||E_i||^2 = d_i (Schur-Horn):
    alternating projections between the rank-d projectors and the prescribed
    diagonal, then the projector's eigenbasis. Starts from a random projector, or
    from `init` (m, m) — e.g. a Fantope solution — for the fixed-diagonal
    realization nearest to it. Returns E1 (m, d)."""
    di = np.asarray(di, float)
    d = int(round(di.sum()))
    s = np.flatnonzero(di > 1e-8)
    if init is None:
        rng = np.random.default_rng(seed)
        V = np.linalg.qr(rng.normal(size=(len(s), d)))[0]
        A = V @ V.T
    else:
        A = 0.5 * (init[np.ix_(s, s)] + init[np.ix_(s, s)].T)
    for _ in range(iters):
        A[np.diag_indices_from(A)] = di[s]
        lam, U = np.linalg.eigh(A)
        A = U[:, -d:] @ U[:, -d:].T
    E1 = np.zeros((len(di), d))
    E1[s] = np.linalg.eigh(A)[1][:, -d:]
    return E1


def decoder_map(alpha, Z, E1, maxiter=4000):
    """The fixed-geometry decoder solve on a dataset: min over D (d, m) and
    mean-embedding offset e0 (d,) of the CE of softmax((z^T E1 + e0^T) D).
    Convex in each block, solved jointly from the tied decoder E1^T W^T.
    Returns (L1 (r, m) row-centred, o (m,))."""
    from scipy.optimize import minimize
    from scipy.special import logsumexp
    W = _logit_map(alpha, Z)
    Z = np.asarray(Z, float)
    m = W.shape[0]
    d = E1.shape[1]
    G, P, _ = _softmax_stats(W, Z)
    ZE = Z @ E1
    n = len(Z)

    def fg(x):
        D = x[:d * m].reshape(d, m)
        e0 = x[d * m:]
        H = ZE + e0
        L = H @ D
        lse = logsumexp(L, axis=1)
        q = np.exp(L - lse[:, None])
        return (float((lse - (P * L).sum(1)).mean()),
                np.concatenate([(H.T @ (q - P) / n).ravel(), (q - P).mean(0) @ D.T]))

    x0 = np.concatenate([(E1.T @ W.T).ravel(), np.zeros(d)])
    r = minimize(fg, x0, jac=True, method="L-BFGS-B",
                 options=dict(maxiter=maxiter, maxfun=2 * maxiter, ftol=1e-10,
                              gtol=1e-6))
    D = r.x[:d * m].reshape(d, m)
    L1 = E1 @ D
    return L1 - L1.mean(1, keepdims=True), r.x[d * m:] @ D


def signed_pressure(alpha, Z, di):
    """The first-order CE pressure on each signed overlap <E_i, E_j> at the
    fractional predictor softmax((z d) W^T): a_kl = Cov(z_k, ((q - p*) W)_l)
    + Cov(z_l, ((q - p*) W)_k), i.e. alpha_j Cov(z_i, q_j - p*_j) + (i <-> j) for a
    diagonal map. a_kl > 0 means a negative overlap lowers the CE. Returns a (r, r),
    zero diagonal."""
    from scipy.special import logsumexp
    W = _logit_map(alpha, Z)
    Z = np.asarray(Z, float)
    G, P, _ = _softmax_stats(W, Z)
    L = (Z * np.asarray(di, float)) @ W.T
    q = np.exp(L - logsumexp(L, axis=1, keepdims=True))
    half = (Z - Z.mean(0)).T @ ((q - P) @ W) / len(Z)
    a = half + half.T
    np.fill_diagonal(a, 0.0)
    return a


def _fantope_project(P, d):
    """Projection onto the Fantope {0 <= P <= I, tr P = d}: clip the spectrum."""
    lam, U = np.linalg.eigh(0.5 * (P + P.T))
    return (U * _box_simplex(lam, d)) @ U.T


def tied_map_ce(alpha, Z, P):
    """The exact CE of the tied map — logits z P W^T, the Bayes readout on the projected
    input — and its gradient in P. Returns (ce, G (r, r)) with G = Z^T (q - p*) W / n;
    at a diagonal P the off-diagonal part of G + G^T is the signed pressure."""
    from scipy.special import logsumexp
    W = _logit_map(alpha, Z)
    Z = np.asarray(Z, float)
    _, Pt, _ = _softmax_stats(W, Z)
    L = (Z @ P) @ W.T
    lse = logsumexp(L, axis=1)
    q = np.exp(L - lse[:, None])
    return float((lse - (Pt * L).sum(1)).mean()), Z.T @ ((q - Pt) @ W) / len(Z)


def _fantope_objective(alpha, Z, P, a):
    """C(diag P) + a^T g with g the upper-triangle off-diagonals of P."""
    from scipy.special import logsumexp
    W = _logit_map(alpha, Z)
    G, Pt, _ = _softmax_stats(W, Z)
    L = (np.asarray(Z, float) * np.clip(np.diag(P), 0.0, 1.0)) @ W.T
    C = float((logsumexp(L, axis=1) - (Pt * L).sum(1)).mean())
    iu = np.triu_indices(len(P), 1)
    return C + float(a[iu] @ P[iu])


def fantope_geometry(alpha, Z, d, a=None, iters=1000, tol=1e-10):
    """The first-order geometry as one convex program on the Fantope {0 <= P <= I,
    tr P = d}, by projected gradient with backtracking. With `a`: the exact CE of diag P plus
    a^T g in the off-diagonals g — the linearised program whose first-order term is the signed
    pressure; its optimum stays near the projectors. Without `a`: the exact tied-map CE
    E[CE(softmax(A P z))] over the whole P, whose KKT conditions are water filling on the
    eigenvalues of P; convex too, but its optimum exploits the relaxation (eta ~ 1) and
    realizes poorly. Returns (P (r, r), eta = tr(P - P^2), the projector-relaxation gap)."""
    from scipy.special import logsumexp
    W = _logit_map(alpha, Z)
    Z = np.asarray(Z, float)
    m = Z.shape[1]
    G, Pt, _ = _softmax_stats(W, Z)
    iu = np.triu_indices(m, 1)

    def value(P):
        return tied_map_ce(W, Z, P)[0] if a is None else _fantope_objective(W, Z, P, a)

    def gradient(P):
        if a is None:
            Gd = tied_map_ce(W, Z, P)[1]
            return 0.5 * (Gd + Gd.T)
        di = np.clip(np.diag(P), 0.0, 1.0)
        L = (Z * di) @ W.T
        q = np.exp(L - logsumexp(L, axis=1, keepdims=True))
        grad = np.zeros((m, m))
        grad[iu] = 0.5 * a[iu]
        grad = grad + grad.T
        grad[np.diag_indices(m)] = _field(W, Z, q, Pt)
        return grad

    d0, _ = allocate_d(W, Z, d)
    P = np.diag(d0)
    f = value(P)
    step = 1.0
    for _ in range(iters):
        grad = gradient(P)
        while True:
            P_new = _fantope_project(P - step * grad, d)
            f_new = value(P_new)
            if f_new <= f or step < 1e-12:
                break
            step *= 0.5
        if f - f_new < tol * max(abs(f), 1.0) and step >= 1e-12:
            P, f = P_new, f_new
            break
        P, f = P_new, f_new
        step *= 1.5
    return P, float(np.trace(P - P @ P))


def predicted_rank_constrained_geometry(alpha, Z, d, n_top=18):
    """The representation the loss pins down at rank d < the task's rank (notes/superposition.tex):
    the optimum of the softmax cross-entropy under the rank constraint, not an SVD truncation.
    Returns the representation of every state H = z E* + phi(z) E2 + e0, (n, d), and the readout
    D, (m, d), so the logits are H D^T. Allocation: water filling (`allocate_d`). Orientation: the linearised Fantope
    program on the signed pressure with the diagonal free (`fantope_geometry`), realized as the
    fixed-diagonal frame E* (`tight_frame`). Completion: decoder and pair block solved at E* by
    convex block alternation (`repair`). Pass the coordinates `modes` gives the task."""
    Z = np.asarray(Z, float)
    d_wf, _ = allocate_d(alpha, Z, d)
    P_star, _ = fantope_geometry(alpha, Z, d, signed_pressure(alpha, Z, d_wf))
    E_star = tight_frame(np.diag(P_star), init=P_star)
    rep = repair(alpha, Z, E_star, n_top=n_top)
    return Z @ E_star + pair_features(Z, n_top) @ rep["E2"] + rep["e0"], rep["D"].T


def pair_features(Z, n_top):
    """Pair products (z_i - p_i)(z_j - p_j) for i < j < min(n_top, m), residualised on [1, z]
    so the block is exactly orthogonal to the first-order dictionary (the second Walsh
    level when the features are independent), then standardized. Returns (n, K)."""
    Z = np.asarray(Z, float)
    Zc = Z - Z.mean(0)
    i, j = np.triu_indices(min(n_top, Z.shape[1]), 1)
    phi = Zc[:, i] * Zc[:, j]
    X = np.column_stack([np.ones(len(Z)), Zc])
    phi = phi - X @ np.linalg.lstsq(X, phi, rcond=None)[0]
    return phi / phi.std(0)


def repair(alpha, Z, E1, n_top=18, rounds=30, tol=1e-5):
    """Stage two at a fixed frame: the additive-plus-pairs class by block-coordinate
    descent. Code z^T E1 + phi(z)^T E2 + e0^T through one decoder D; alternate the
    convex solves for E2 (decoder fixed) and (D, e0) (code fixed), starting from the
    additive decoder, until the relative CE change per round is below tol or `rounds`
    is reached. Returns dict(E2 (K, d), L1 (m, m) row-centred first-order map E1 D,
    D (d, m), e0 (d,), ce (excess CE per round), ce_additive)."""
    from scipy.optimize import minimize
    from scipy.special import logsumexp
    Z = np.asarray(Z, float)
    G, P, _ = _softmax_stats(alpha, Z)
    m, d, n = G.shape[1], E1.shape[1], len(Z)
    bayes = float((logsumexp(G, axis=1) - (P * G).sum(1)).mean())
    phi = pair_features(Z, n_top)
    L1, o = decoder_map(alpha, Z, E1)
    D = E1.T @ L1
    e0 = np.linalg.lstsq(D.T, o, rcond=None)[0]           # offset row o = e0^T D
    H1 = Z @ E1
    E2 = np.zeros((phi.shape[1], d))
    ce = [_excess_ce(Z @ L1 + o, G, P)]
    for _ in range(rounds):
        def fg2(x):
            E2_ = x.reshape(-1, d)
            L = (H1 + phi @ E2_ + e0) @ D
            lse = logsumexp(L, axis=1)
            q = np.exp(L - lse[:, None])
            return (float((lse - (P * L).sum(1)).mean()),
                    (phi.T @ ((q - P) @ D.T) / n).ravel())
        E2 = minimize(fg2, E2.ravel(), jac=True, method="L-BFGS-B",
                      options=dict(maxiter=2000, ftol=1e-10, gtol=1e-6)).x.reshape(-1, d)
        Hc = H1 + phi @ E2

        def fgD(x):
            D_ = x[:d * m].reshape(d, m)
            e0_ = x[d * m:]
            Hh = Hc + e0_
            L = Hh @ D_
            lse = logsumexp(L, axis=1)
            q = np.exp(L - lse[:, None])
            return (float((lse - (P * L).sum(1)).mean()),
                    np.concatenate([(Hh.T @ (q - P) / n).ravel(), (q - P).mean(0) @ D_.T]))
        x = minimize(fgD, np.concatenate([D.ravel(), e0]), jac=True, method="L-BFGS-B",
                     options=dict(maxiter=4000, ftol=1e-10, gtol=1e-6)).x
        D, e0 = x[:d * m].reshape(d, m), x[d * m:]
        L = (Hc + e0) @ D
        ce.append(float((logsumexp(L, axis=1) - (P * L).sum(1)).mean()) - bayes)
        if ce[-2] - ce[-1] < tol * ce[-2]:
            break
    L1 = E1 @ D
    return dict(E2=E2, L1=L1 - L1.mean(1, keepdims=True), D=D, e0=e0,
                ce=np.array(ce), ce_additive=ce[0])


