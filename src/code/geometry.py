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


def class_means(X, labels, n=None):
    """Mean code per label and the label counts: ``((n, d), (n,))``."""
    n = n if n is not None else int(np.max(labels)) + 1
    M = (np.asarray(labels)[:, None] == np.arange(n)).astype(float)
    count = M.sum(0)
    return M.T @ X / count[:, None], count


def predicted_geometry(log_cond):
    """The representation the loss pins down when capacity suffices: ``E``, one row per input,
    the balanced factor of the log-conditional ``L[input, outcome]``. For the rank-constrained
    case see `rank_constrained_geometry`."""
    return factor_logcond(log_cond)[1]


def factor_planes(D, blocks):
    """Per outcome block of a decoder ``D`` (d, C): the block's mean column (its presence
    direction, (d,)) and an orthonormal basis of its token-centred columns (the plane the factor's
    positions live in, (d, rank)). Returns a list of ``(mean, basis)``."""
    D = np.asarray(D, dtype=np.float64)
    out = []
    for b in blocks:
        m = D[:, b].mean(1)
        U, S, _ = np.linalg.svd(D[:, b] - m[:, None], full_matrices=False)
        out.append((m, U[:, S > 1e-6 * S[0]]))
    return out


def predicted_inference_geometry(log_cond, D):
    """Representation of each state in the frame of a fitted decoder: rows of ``L[state, outcome]``
    centered over outcomes, times ``pinv(D)``. ``D`` (d, C) from ``factor_logcond`` on sampled
    histories; returns ``(n_states, d)``."""
    L = np.asarray(log_cond, dtype=np.float64)
    return (L - L.mean(1, keepdims=True)) @ np.linalg.pinv(D)


##### Superposition: the rank-d optimum of a softmax readout (notes/superposition_new.tex) #####

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
    """The coordinates every model is measured in, and the optimum's moment frame is built on: the
    features themselves when independent, else the predictive modes with the Bayes map expressed
    on them (`whiten`).
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


def softmax_stats(alpha, Z):
    """Target probabilities and centred target logits for a dataset. Returns (G, P, Gc)."""
    from scipy.special import logsumexp
    G = np.asarray(Z, float) @ _logit_map(alpha, Z).T
    P = np.exp(G - logsumexp(G, axis=1, keepdims=True))
    return G, P, G - G.mean(1, keepdims=True)


def excess_ce(L, G, P):
    """Excess CE of logits L against the target (G, P) on the same dataset."""
    from scipy.special import logsumexp
    bayes = float((logsumexp(G, axis=1) - (P * G).sum(1)).mean())
    return float((logsumexp(L, axis=1) - (P * L).sum(1)).mean()) - bayes


def _moment_frame(X, P, d, tilt=0.05):
    """Top-d right singular vectors of Cov(standardized X, P), the decoder subspace the modes
    predict best linearly, each tilted by ``tilt`` toward one fixed centred direction over the
    outputs so that no output starts with a zero footprint (a hard cut is a stationary point of
    the profiled CE that the descent could not leave). (d, m), rows orthonormal and orthogonal
    to the ones vector."""
    X = np.asarray(X, float)
    Xs = (X - X.mean(0)) / X.std(0)
    M = Xs.T @ (P - P.mean(0)) / len(X)
    V = np.linalg.svd(M, full_matrices=False)[2][:d]
    w = np.arange(1, P.shape[1] + 1, dtype=float)
    w -= w.mean()
    V = V + tilt * w / np.linalg.norm(w)
    U, _, Vt = np.linalg.svd(V - V.mean(1, keepdims=True), full_matrices=False)
    return U @ Vt


def moment_match(D, P, E0=None, iters=40, tol=1e-8):
    """E (n, d) with D softmax(E D) = D P for every state: the I-projection of each target onto
    the exponential family with sufficient statistics D (d, m), by damped Newton warm-started at
    E0, backtracking only on the states whose full step fails. Returns (E, Q), Q the matched
    predictions."""
    from scipy.special import logsumexp
    d = D.shape[0]
    E = np.zeros((len(P), d)) if E0 is None else np.array(E0, float)

    def ce(E, P):
        L = E @ D
        return logsumexp(L, axis=1) - (L * P).sum(1)
    for _ in range(iters):
        L = E @ D
        Q = np.exp(L - logsumexp(L, axis=1, keepdims=True))
        r = (Q - P) @ D.T
        if np.abs(r).max() < tol:
            break
        QD = Q @ D.T
        Hs = np.einsum("nk,ki,kj->nij", Q, D.T, D.T) - QD[:, :, None] * QD[:, None, :]
        step = np.linalg.solve(Hs + 1e-10 * np.eye(d), r[..., None])[..., 0]
        f0, slope, t = ce(E, P), (r * step).sum(1), np.ones(len(P))
        E1 = E - step
        f1 = ce(E1, P)
        idx = np.flatnonzero(f1 > f0 - 1e-4 * slope)
        while len(idx) and t[idx].min() > 1e-12:
            t[idx] *= 0.5
            E1[idx] = E[idx] - t[idx, None] * step[idx]
            f1[idx] = ce(E1[idx], P[idx])
            idx = idx[f1[idx] > f0[idx] - 1e-4 * t[idx] * slope[idx]]
        E = E1
    L = E @ D
    return E, np.exp(L - logsumexp(L, axis=1, keepdims=True))


def profiled_fit(X, P, d, gtol=1e-7, max_iter=600, memory=10):
    """The rank-d optimum of the mean CE against the targets P (n, m): Riemannian L-BFGS on the
    decoder subspace for the profiled CE L(D) = mean KL(p || I-projection onto D), the
    representation profiled out by `moment_match`, the envelope gradient mean E (q - p)^T
    projected onto the Grassmann tangent, curvature pairs transported by projection, Armijo
    backtracking and a polar retraction, from `_moment_frame`; stops when the largest entry of
    the Riemannian gradient is below gtol. Returns (H (n, d), D (m, d)) with logits H D^T."""
    from scipy.special import logsumexp

    def retract(D):
        D = D - D.mean(1, keepdims=True)
        U, _, Vt = np.linalg.svd(D, full_matrices=False)
        return U @ Vt

    def tangent(V, D):
        return V - (V @ D.T) @ D

    def profiled(D, E0):
        E, Q = moment_match(D, P, E0)
        L = E @ D
        return float((logsumexp(L, axis=1) - (L * P).sum(1)).mean()), E.T @ (Q - P) / len(P), E
    D = retract(_moment_frame(X, P, d))
    f, grad, E = profiled(D, None)
    g = tangent(grad, D)
    pairs = []
    for _ in range(max_iter):
        if np.abs(g).max() < gtol:
            break
        q, alphas = g.copy(), []                          # two-loop recursion in the tangent space
        for s, y in reversed(pairs):
            a = (s * q).sum() / (y * s).sum()
            q -= a * y
            alphas.append(a)
        gamma = (pairs[-1][0] * pairs[-1][1]).sum() / (pairs[-1][1] ** 2).sum() if pairs else 0.1 / np.sqrt((g ** 2).sum())
        r = gamma * q
        for (s, y), a in zip(pairs, reversed(alphas)):
            r += (a - (y * r).sum() / (y * s).sum()) * s
        p, slope = -r, -(g * r).sum()
        if slope >= 0:                                    # not a descent direction: restart from the gradient
            pairs, p, slope = [], -g, -(g ** 2).sum()
        t = 1.0
        while True:
            Dn = retract(D + t * p)
            fn, gradn, En = profiled(Dn, E)
            if fn <= f + 1e-4 * t * slope or t < 1e-10:
                break
            t *= 0.5
        gn = tangent(gradn, Dn)
        s, y = tangent(t * p, Dn), gn - tangent(g, Dn)
        pairs = [(tangent(s_, Dn), tangent(y_, Dn)) for s_, y_ in pairs][-(memory - 1):]
        if (s * y).sum() > 1e-10 * np.sqrt((s ** 2).sum() * (y ** 2).sum()):
            pairs.append((s, y))
        D, f, grad, E, g = Dn, fn, gradn, En, gn
    E, _ = moment_match(D, P, E, iters=100, tol=1e-10)
    return E, D.T


def rank_constrained_geometry(alpha, Z, d, gtol=1e-7, max_iter=600):
    """The representation the loss pins down at rank d < the task's rank
    (notes/superposition_new.tex): `profiled_fit` on the task's target. Pass the coordinates
    `modes` gives the task. Returns (H (n, d), D (m, d)) with logits H D^T."""
    Z = np.asarray(Z, float)
    return profiled_fit(Z, softmax_stats(alpha, Z)[1], d, gtol=gtol, max_iter=max_iter)


def frequency_threshold(alpha, m):
    """rho_c(alpha, m): for signed singletons of common strength alpha over m features, a retained
    feature i and a dropped feature j start sharing a direction when pi_j / pi_i > rho_c."""
    return alpha ** 2 * (m - 1 + np.cosh(alpha)) / (m * np.sinh(alpha) ** 2)
