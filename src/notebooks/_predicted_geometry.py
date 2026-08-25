"""Closed-form prediction of the learned code and decoder (see notes/head_tail_derivation.md).

Staging area for the notebook: everything here is derived in the note but not yet promoted to
``src/code``. Task parameters in, geometry out, no training.
"""
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize

# finite-capacity correction, A = k^2 sqrt(Npq) measured per budget (note §4.4)
A_OF_D = {2: 2.46, 3: 2.25, 4: 2.37, 6: 2.04, 8: 1.79, 12: 1.67}


def fisher_ratio(Z, alpha, mu, P):
    """Qbar/Q1 over head features: how much less probable outcome i is on average than
    when its own feature fires. Task-only; approaches p when inactive features are negligible."""
    head = np.where(alpha > mu)[0]
    head = head if len(head) else np.array([0])
    return float(np.mean([P[:, i].mean() / P[Z[:, i] == 1, i].mean() for i in head]))


def water_level(Z, alpha, P, N, p, d, A=None):
    """Solve mu = sqrt(Npq) (Qbar/Q1) B/d with B = sum_i A (alpha_i - mu)+/sqrt(Npq).
    Returns (mu, ||w_i||^2)."""
    A = A_OF_D.get(d, 2.0) if A is None else A
    rt = np.sqrt(N * p * (1 - p))
    mu = 0.3 * alpha.max()
    for _ in range(80):                                   # damped fixed point
        B = (A * np.maximum(alpha - mu, 0) / rt).sum()
        mu = 0.5 * mu + 0.5 * (rt * fisher_ratio(Z, alpha, mu, P) * B / d)
    return mu, A * np.maximum(alpha - mu, 0) / rt


def min_interference_dirs(n2, d, seed=0):
    """Directions minimising sum_{i!=k} <w_i,w_k>^2 at fixed norms. Returns W (m, d)."""
    nrm = np.sqrt(n2)
    act = np.where(nrm > 1e-9)[0]
    n = nrm[act]
    rng = np.random.default_rng(seed)

    def fg(f):
        U = f.reshape(len(act), d)
        U = U / np.linalg.norm(U, axis=1, keepdims=True)
        G = (n[:, None] * U) @ (n[:, None] * U).T
        np.fill_diagonal(G, 0.0)
        return float((G ** 2).sum())

    r = minimize(fg, rng.normal(size=(len(act), d)).ravel(), method="L-BFGS-B",
                 options=dict(maxiter=4000))
    U = r.x.reshape(len(act), d)
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    W = np.zeros((len(nrm), d))
    W[act] = n[:, None] * U
    return W


def decoder_from_code(A_code, P, m, d):
    """Joint convex solve for all decoders given the code. Returns (D, excess CE)."""
    n = A_code.shape[0]

    def fg(f):
        D = f.reshape(m, d)
        G = A_code @ D.T
        Q = np.exp(G - logsumexp(G, 1, keepdims=True))
        ce = float((logsumexp(G, 1) - (P * G).sum(1)).mean())
        return ce, (((Q - P).T @ A_code) / n).ravel()

    r = minimize(fg, np.zeros(m * d), jac=True, method="L-BFGS-B",
                 options=dict(maxiter=8000, maxfun=12000))
    return r.x.reshape(m, d), r.fun


def predict(Z, alpha, N, p, d, n_dirs=3, A=None):
    """Full pipeline. Returns dict with mu, norms, code, decoder, and the CE achieved."""
    logits = Z * alpha
    P = np.exp(logits - logsumexp(logits, 1, keepdims=True))
    m = len(alpha)
    mu, n2 = water_level(Z, alpha, P, N, p, d, A)
    Zc = Z - Z.mean(0)
    best = (np.inf, None, None)
    for s in range(n_dirs):                               # candidates, selected by CE
        W = min_interference_dirs(n2, d, seed=s)
        code = Zc @ W
        D, ce = decoder_from_code(code, P, m, d)
        if ce < best[0]:
            best = (ce, code, D)
    bayes = -(P * np.log(np.clip(P, 1e-300, None))).sum(1).mean()
    return dict(mu=mu, n2=n2, code=best[1], decoder=best[2], excess_ce=best[0] - bayes)


def gauge(E, Wr, d):
    """Balanced, row-centred factorisation of a trained model. Returns (code, decoder)."""
    G = np.asarray(E) @ np.asarray(Wr).T
    G = G - G.mean(1, keepdims=True)
    U, S, Vt = np.linalg.svd(G, full_matrices=False)
    return U[:, :d] * np.sqrt(S[:d]), (np.sqrt(S[:d])[:, None] * Vt[:d]).T


def encoding_weights(code, Z):
    """||w_i|| by regressing the code on the feature indicators."""
    Zc = Z - Z.mean(0)
    w = np.linalg.lstsq(Zc, code - code.mean(0), rcond=None)[0]
    return np.linalg.norm(w, axis=1)
