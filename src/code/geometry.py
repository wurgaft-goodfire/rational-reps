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


def factor_logcond(L, rank=None):
    """SVD of the row-centered log-conditional ``L[theta, x]`` (dropping the per-row
    log-partition a softmax ignores), truncated to ``rank`` modes if given. Returns
    ``(Lc, E, D, S)``, the scale split as ``sqrt(S)`` on each factor (the balanced split
    gradient descent learns), so ``E @ D = Lc`` (at full rank)."""
    L = np.asarray(L, dtype=np.float64)
    Lc = L - L.mean(1, keepdims=True)
    U, S, Vt = np.linalg.svd(Lc, full_matrices=False)
    U, S, Vt = U[:, :rank], S[:rank], Vt[:rank]
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


def predicted_geometry(log_cond, rank=None):
    """Predicted code ``E``, one row per input, for a log-conditional ``L[input, outcome]``;
    ``rank`` truncates to the leading modes."""
    return factor_logcond(log_cond, rank)[1]


##### Superposition: first-order geometry of a rank-limited softmax (notes/superposition.tex) #####

def _softmax_stats(alpha, Z):
    """Target probabilities and centred target logits for a dataset. Returns (G, P, Gc)."""
    from scipy.special import logsumexp
    G = np.asarray(Z, float) * np.asarray(alpha, float)
    P = np.exp(G - logsumexp(G, axis=1, keepdims=True))
    return G, P, G - G.mean(1, keepdims=True)


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
    fractional predictor softmax(alpha_i d_i z_i) over {0 <= d_i <= 1, sum d_i = d}
    by projected gradient with early stop. Returns (di (m,), delta), delta the
    interior marginal-utility level."""
    from scipy.special import logsumexp
    alpha = np.asarray(alpha, float)
    Z = np.asarray(Z, float)
    m = len(alpha)
    G, P, _ = _softmax_stats(alpha, Z)
    di = np.full(m, d / m)
    f_prev, stall = np.inf, 0

    def project(v):
        return _box_simplex(v, d)

    for _ in range(iters):
        L = Z * (alpha * di)
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
        di = project(di - step * alpha * (Z * (q - P)).mean(0))
    L = Z * (alpha * di)
    q = np.exp(L - logsumexp(L, axis=1, keepdims=True))
    u = -alpha * (Z * (q - P)).mean(0)
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
    Convex in each block, solved jointly from the tied decoder E1^T diag(alpha).
    Returns (L1 (m, m) row-centred, o (m,))."""
    from scipy.optimize import minimize
    from scipy.special import logsumexp
    alpha = np.asarray(alpha, float)
    Z = np.asarray(Z, float)
    m = len(alpha)
    d = E1.shape[1]
    G, P, _ = _softmax_stats(alpha, Z)
    ZE = Z @ E1
    n = len(Z)

    def fg(x):
        C = x[:d * m].reshape(d, m)
        e0 = x[d * m:]
        H = ZE + e0
        L = H @ C
        lse = logsumexp(L, axis=1)
        q = np.exp(L - lse[:, None])
        return (float((lse - (P * L).sum(1)).mean()),
                np.concatenate([(H.T @ (q - P) / n).ravel(), (q - P).mean(0) @ C.T]))

    x0 = np.concatenate([(E1.T * alpha).ravel(), np.zeros(d)])
    r = minimize(fg, x0, jac=True, method="L-BFGS-B",
                 options=dict(maxiter=maxiter, maxfun=2 * maxiter, ftol=1e-10,
                              gtol=1e-6))
    C = r.x[:d * m].reshape(d, m)
    L1 = E1 @ C
    return L1 - L1.mean(1, keepdims=True), r.x[d * m:] @ C


def signed_pressure(alpha, Z, di):
    """The first-order CE pressure on each signed overlap <E_i, E_j> at the
    fractional predictor softmax(alpha_i d_i z_i): a_ij = alpha_j E[(z_i - p)(q_j - p*_j)]
    + alpha_i E[(z_j - p)(q_i - p*_i)]. a_ij > 0 means a negative overlap lowers the
    CE (feature i should suppress outcome j). Returns a (m, m), zero diagonal."""
    from scipy.special import logsumexp
    alpha = np.asarray(alpha, float)
    Z = np.asarray(Z, float)
    G, P, _ = _softmax_stats(alpha, Z)
    L = Z * (alpha * np.asarray(di, float))
    q = np.exp(L - logsumexp(L, axis=1, keepdims=True))
    half = alpha[None, :] * ((Z - Z.mean(0)).T @ (q - P)) / len(Z)
    a = half + half.T
    np.fill_diagonal(a, 0.0)
    return a


def interference_curvature(alpha, Z, di):
    """Fisher curvature of the tied-map CE in the off-diagonal coordinates g_p,
    p = (i, j) over the upper triangle, at the fractional predictor: H[p, q] =
    E[dl_p^T (diag q - q q^T) dl_q], where g_p moves logit j by alpha_j (z_i - p)
    and logit i by alpha_i (z_j - p). Returns H (m(m-1)/2, m(m-1)/2), PSD."""
    from scipy.special import logsumexp
    alpha = np.asarray(alpha, float)
    Z = np.asarray(Z, float)
    n, m = Z.shape
    L = Z * (alpha * np.asarray(di, float))
    q = np.exp(L - logsumexp(L, axis=1, keepdims=True))
    X = Z - Z.mean(0)
    I, J = np.triu_indices(m, 1)
    npairs = len(I)
    # S[:, p] = q . dl_p ; the rank-one Fisher part is -S^T S / n
    S = q[:, J] * X[:, I] * alpha[J] + q[:, I] * X[:, J] * alpha[I]
    H = -S.T @ S / n
    # the diagonal Fisher part groups pairs by the outcome they hit
    pair_index = np.full((m, m), -1)
    pair_index[I, J] = pair_index[J, I] = np.arange(npairs)
    for k in range(m):
        others = np.delete(np.arange(m), k)
        idx = pair_index[k, others]
        W = X[:, others] * alpha[k]                          # weight on logit k
        H[np.ix_(idx, idx)] += (W * q[:, [k]]).T @ W / n
    return H


def _fantope_project(P, d):
    """Projection onto the Fantope {0 <= P <= I, tr P = d}: clip the spectrum."""
    lam, U = np.linalg.eigh(0.5 * (P + P.T))
    return (U * _box_simplex(lam, d)) @ U.T


def _fantope_objective(alpha, Z, P, a, H):
    """C(diag P) + a^T g + g^T H g / 2 with g the upper-triangle off-diagonals of P."""
    from scipy.special import logsumexp
    G, Pt, _ = _softmax_stats(alpha, Z)
    L = Z * (alpha * np.clip(np.diag(P), 0.0, 1.0))
    C = float((logsumexp(L, axis=1) - (Pt * L).sum(1)).mean())
    iu = np.triu_indices(len(P), 1)
    g = P[iu]
    return C + float(a[iu] @ g) + (0.0 if H is None else 0.5 * float(g @ H @ g))


def fantope_geometry(alpha, Z, d, a, H=None, iters=1000, tol=1e-10):
    """The first-order geometry as one convex program on the Fantope: minimize the
    exact fractional-predictor CE of the diagonal plus the local expansion of the
    interference in the off-diagonals, a^T g (+ g^T H g / 2), over 0 <= P <= I,
    tr P = d. Projected gradient with backtracking. Returns (P (m, m),
    eta = tr(P - P^2), the projector-relaxation gap)."""
    from scipy.special import logsumexp
    alpha = np.asarray(alpha, float)
    Z = np.asarray(Z, float)
    m = len(alpha)
    G, Pt, _ = _softmax_stats(alpha, Z)
    iu = np.triu_indices(m, 1)
    d0, _ = allocate_d(alpha, Z, d)
    P = np.diag(d0)
    f = _fantope_objective(alpha, Z, P, a, H)
    step = 1.0
    for _ in range(iters):
        di = np.clip(np.diag(P), 0.0, 1.0)
        L = Z * (alpha * di)
        q = np.exp(L - logsumexp(L, axis=1, keepdims=True))
        g = P[iu]
        goff = a[iu] + (0.0 if H is None else H @ g)
        grad = np.zeros((m, m))
        grad[iu] = 0.5 * goff
        grad = grad + grad.T
        grad[np.diag_indices(m)] = alpha * (Z * (q - Pt)).mean(0)
        while True:
            P_new = _fantope_project(P - step * grad, d)
            f_new = _fantope_objective(alpha, Z, P_new, a, H)
            if f_new <= f or step < 1e-12:
                break
            step *= 0.5
        if f - f_new < tol * max(abs(f), 1.0) and step >= 1e-12:
            P, f = P_new, f_new
            break
        P, f = P_new, f_new
        step *= 1.5
    return P, float(np.trace(P - P @ P))


def top_eigenspace(P, d):
    """Realization A: the top-d eigenvectors of P, an orthonormal (m, d) basis."""
    lam, U = np.linalg.eigh(0.5 * (P + P.T))
    return U[:, -d:]


def predict_geometry(alpha, Z, d, n_frames=32, n_top=18, seed=0):
    """The geometry on a dataset (notes/superposition.tex). Allocation: water filling.
    Orientation: the linear Fantope program on the signed pressure with the diagonal
    free, realized as the fixed-diagonal frame E* (the *_star outputs; eta is the
    projector-relaxation gap). Completion: decoder and pair block solved at E* by
    convex block alternation (the *_rep outputs: E2_rep (K, d), e0_rep (d,), C_rep (d, m),
    and map_rep (m + K + 1, m), the completed model's logit map over its dictionary
    [z; phi(z); 1], whose `predicted_geometry` at rank d is the dictionary in balanced
    coordinates: feature rows, pair rows, offset). Reference band:
    n_frames random compatible frames at the water-filling diagonal (the *_mean/_sd
    outputs)."""
    from joblib import Parallel, delayed, parallel_config
    alpha = np.asarray(alpha, float)
    Z = np.asarray(Z, float)
    m = len(alpha)
    G, Pt, _ = _softmax_stats(alpha, Z)
    d_wf, delta = allocate_d(alpha, Z, d)
    a = signed_pressure(alpha, Z, d_wf)
    P_star, eta = fantope_geometry(alpha, Z, d, a)
    d_star = np.diag(P_star)
    E_star = tight_frame(d_star, init=P_star)
    ce_star, gram_star, spectra_star, gramD_star, offset_star = _realize_frame(alpha, Z, E_star, d)
    rep = repair(alpha, Z, E_star, n_top=n_top)
    map_rep = np.vstack([E_star, rep["E2"], rep["e0"]]) @ rep["D"]
    with parallel_config(backend="loky", inner_max_num_threads=1):
        frames = Parallel(n_jobs=min(n_frames, 32))(
            delayed(_realize_frame)(alpha, Z, tight_frame(d_wf, seed=seed + 101 * k + 1), d)
            for k in range(n_frames))
    frame_ce = np.array([f[0] for f in frames])
    grams = np.array([f[1] for f in frames])
    spectra = np.array([f[2] for f in frames])
    gramsD = np.array([f[3] for f in frames])
    sizes = grams[:, np.arange(m), np.arange(m)]
    return dict(d_wf=d_wf, d_star=d_star, delta=delta, a=a, P_star=P_star, eta=eta,
                E_star=E_star, ce_star=ce_star, gram_star=gram_star,
                sizes_star=np.diag(gram_star), spectra_star=spectra_star,
                gramD_star=gramD_star, offset_star=offset_star,
                E2_rep=rep["E2"], e0_rep=rep["e0"], C_rep=rep["D"], map_rep=map_rep,
                ce_rep=rep["ce"][-1],
                ce_add=rep["ce_additive"], ce_trace_rep=rep["ce"],
                frame_ce=frame_ce,
                gram_mean=grams.mean(0), gram_sd=grams.std(0),
                gramD_mean=gramsD.mean(0), gramD_sd=gramsD.std(0),
                sizes_mean=sizes.mean(0), sizes_sd=sizes.std(0),
                spectra_mean=spectra.mean(0), spectra_sd=spectra.std(0))


def _realize_frame(alpha, Z, E1, d):
    """A compatible tight frame with its decoder: (excess CE, feature Gram
    U Sigma U^T of L1, top-d spectrum, decoder Gram V Sigma V^T, offset row o)."""
    G, P, _ = _softmax_stats(alpha, Z)
    L1, o = decoder_map(alpha, Z, E1)
    Uf, Sf, Vt = np.linalg.svd(L1, full_matrices=False)
    return (_excess_ce(Z @ L1 + o, G, P),
            (Uf[:, :d] * Sf[:d]) @ Uf[:, :d].T, Sf[:d],
            (Vt[:d].T * Sf[:d]) @ Vt[:d], o)


def pair_features(Z, n_top):
    """Standardized centered pair products (z_i - p_i)(z_j - p_j) for i < j < n_top:
    the second Walsh level, population-orthogonal to the features. Returns (n, K)."""
    Zc = np.asarray(Z, float) - np.asarray(Z, float).mean(0)
    i, j = np.triu_indices(n_top, 1)
    phi = Zc[:, i] * Zc[:, j]
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
    alpha = np.asarray(alpha, float)
    Z = np.asarray(Z, float)
    m, d, n = len(alpha), E1.shape[1], len(Z)
    G, P, _ = _softmax_stats(alpha, Z)
    bayes = float((logsumexp(G, axis=1) - (P * G).sum(1)).mean())
    phi = pair_features(Z, n_top)
    L1, o = decoder_map(alpha, Z, E1)
    C = E1.T @ L1
    e0 = np.linalg.lstsq(C.T, o, rcond=None)[0]           # offset row o = e0^T C
    H1 = Z @ E1
    E2 = np.zeros((phi.shape[1], d))
    ce = [_excess_ce(Z @ L1 + o, G, P)]
    for _ in range(rounds):
        def fg2(x):
            E2_ = x.reshape(-1, d)
            L = (H1 + phi @ E2_ + e0) @ C
            lse = logsumexp(L, axis=1)
            q = np.exp(L - lse[:, None])
            return (float((lse - (P * L).sum(1)).mean()),
                    (phi.T @ ((q - P) @ C.T) / n).ravel())
        E2 = minimize(fg2, E2.ravel(), jac=True, method="L-BFGS-B",
                      options=dict(maxiter=2000, ftol=1e-10, gtol=1e-6)).x.reshape(-1, d)
        Hc = H1 + phi @ E2

        def fgD(x):
            C_ = x[:d * m].reshape(d, m)
            e0_ = x[d * m:]
            Hh = Hc + e0_
            L = Hh @ C_
            lse = logsumexp(L, axis=1)
            q = np.exp(L - lse[:, None])
            return (float((lse - (P * L).sum(1)).mean()),
                    np.concatenate([(Hh.T @ (q - P) / n).ravel(), (q - P).mean(0) @ C_.T]))
        x = minimize(fgD, np.concatenate([C.ravel(), e0]), jac=True, method="L-BFGS-B",
                     options=dict(maxiter=4000, ftol=1e-10, gtol=1e-6)).x
        C, e0 = x[:d * m].reshape(d, m), x[d * m:]
        L = (Hc + e0) @ C
        ce.append(float((logsumexp(L, axis=1) - (P * L).sum(1)).mean()) - bayes)
        if ce[-2] - ce[-1] < tol * ce[-2]:
            break
    L1 = E1 @ C
    return dict(E2=E2, L1=L1 - L1.mean(1, keepdims=True), D=C, e0=e0,
                ce=np.array(ce), ce_additive=ce[0])


def biconvex_pairs(alpha, Z, d, n_top=18, rounds=10, seed=0, init=None):
    """The additive-plus-pairs class at its exact-CE optimum by alternating convex
    block solves: logits = psi(z)^T B D with psi = [1, z, phi(z)], B ((1 + m + K), d),
    D (d, m). Starts from the additive solution at a random compatible frame (or
    `init` = (B, D)). Unregularized, as the SGD pairs runner. Returns dict(B, D, ce,
    ce_trace) with the excess CE after every block solve."""
    from scipy.optimize import minimize
    from scipy.special import logsumexp
    alpha = np.asarray(alpha, float)
    Z = np.asarray(Z, float)
    n, m = Z.shape
    G, P, _ = _softmax_stats(alpha, Z)
    bayes = float((logsumexp(G, axis=1) - (P * G).sum(1)).mean())
    psi = np.column_stack([np.ones(n), Z, pair_features(Z, n_top)])
    if init is None:
        di, _ = allocate_d(alpha, Z, d)
        E1 = tight_frame(di, seed=seed)
        L1, o = decoder_map(alpha, Z, E1)
        C = E1.T @ L1
        B = np.zeros((psi.shape[1], d))
        B[1:1 + m] = E1
        B[0] = np.linalg.lstsq(C.T, o, rcond=None)[0]
        D = C
    else:
        B, D = (np.array(x, float) for x in init)

    def excess(B_, D_):
        L = (psi @ B_) @ D_
        return float((logsumexp(L, axis=1) - (P * L).sum(1)).mean()) - bayes

    def solve(fg, x0):
        return minimize(fg, x0, jac=True, method="L-BFGS-B",
                        options=dict(maxiter=4000, ftol=1e-11, gtol=1e-7)).x

    trace = [excess(B, D)]
    for _ in range(rounds):
        def fgB(x):
            B_ = x.reshape(-1, d)
            H = psi @ B_
            L = H @ D
            lse = logsumexp(L, axis=1)
            q = np.exp(L - lse[:, None])
            return (float((lse - (P * L).sum(1)).mean()),
                    (psi.T @ ((q - P) @ D.T) / n).ravel())
        B = solve(fgB, B.ravel()).reshape(-1, d)
        trace.append(excess(B, D))
        H = psi @ B

        def fgD(x):
            D_ = x.reshape(d, m)
            L = H @ D_
            lse = logsumexp(L, axis=1)
            q = np.exp(L - lse[:, None])
            return (float((lse - (P * L).sum(1)).mean()),
                    (H.T @ (q - P) / n).ravel())
        D = solve(fgD, D.ravel()).reshape(d, m)
        trace.append(excess(B, D))
    return dict(B=B, D=D, ce=trace[-1], ce_trace=np.array(trace))


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


