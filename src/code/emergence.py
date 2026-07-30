"""Spectral emergence of a latent category from cooccurrence counts.

The gender/noun-class DGP and its measurements. Nouns carry a hidden binary class revealed only
through a pair of *marked* (agreement) contexts — seen with probability ``p`` per token, biased
by ``delta`` toward the class's preferred form — amid class-neutral contexts. The class is a
planted rank-1 spike in the correspondence-analysis residuals of the noun x context count matrix:
its population singular value is ``sqrt(N * p) * delta`` against a Marchenko-Pastur noise bulk,
so it becomes *detectable* (BBP transition) at ``N* = sqrt(V * C) / (p * delta^2)``. Crossing N*
is what buys the abstraction's payoff: one-shot classification of a held-out noun from a single
marked token, and rare nouns classified without per-noun attestation.
"""
import numpy as np


def class_context_joint(n_nouns, n_neutral, p, delta, n_marked=4, zipf_a=1.2, seed=0):
    """Population joint ``Q[noun, context]`` for the noun-class DGP.

    The first ``2 * n_marked`` contexts are marked agreement *paradigms* — pairs ``(2k, 2k+1)``
    of alternating forms (la/el, -a/-o, ...), each carrying an equal share of the marked mass
    ``p``; a class-0 noun spends fraction ``(1+delta)/2`` of a pair's mass on its even member
    (class 1 mirrored). The remaining ``n_neutral`` contexts are class-neutral. The planted
    spike's singular value is ``delta * sqrt(p)`` regardless of ``n_marked`` (the per-pair signal
    shrinks as the pairs multiply), but recovering the *sign pattern across pairs* — what
    ``one_shot_accuracy`` needs — requires the direction, not one lucky bit. Noun frequencies
    are Zipf(``zipf_a``), assigned independently of class (classes exactly balanced).
    Returns ``(Q, cls, f)`` with ``Q`` summing to 1.
    """
    rng = np.random.default_rng(seed)
    cls = rng.permutation(np.arange(n_nouns) % 2)
    f = np.arange(1, n_nouns + 1, dtype=np.float64) ** -zipf_a
    f = rng.permutation(f / f.sum())
    ctx = np.empty((n_nouns, 2 * n_marked + n_neutral))
    even = p / n_marked * np.where(cls == 0, (1 + delta) / 2, (1 - delta) / 2)
    ctx[:, 0:2 * n_marked:2] = even[:, None]
    ctx[:, 1:2 * n_marked:2] = p / n_marked - even[:, None]
    ctx[:, 2 * n_marked:] = (1 - p) / n_neutral
    return f[:, None] * ctx, cls, f


def sample_counts(Q, N, rng):
    """A corpus of ``N`` tokens: multinomial counts under the joint ``Q``, shape ``Q.shape``."""
    q = np.asarray(Q, np.float64).ravel()
    return rng.multinomial(N, q / q.sum()).reshape(Q.shape)


def ca_residuals(M):
    """``sqrt(N)``-scaled correspondence-analysis residuals of a count matrix.

    ``R = sqrt(N) * (P - r c^T) / sqrt(r c^T)`` with ``P = M/N`` and ``r``/``c`` its margins.
    Under row-column independence the entries are ~N(0,1), so the noise singular values form a
    Marchenko-Pastur bulk with edge ``mp_edge``; a planted class direction rides on top as a
    spike. Rows/columns with zero margin carry no evidence and get zero residuals (their expected
    counts are also zero).
    """
    M = np.asarray(M, np.float64)
    N = M.sum()
    P = M / N
    E = np.outer(P.sum(1), P.sum(0))
    return np.sqrt(N) * np.where(E > 0, (P - E) / np.sqrt(np.where(E > 0, E, 1.0)), 0.0)


def mp_edge(shape):
    """Marchenko-Pastur edge (largest noise singular value) for unit-variance entries."""
    V, C = shape
    return np.sqrt(V) + np.sqrt(C)


def bbp_threshold(n_nouns, n_contexts, p, delta):
    """Corpus size at which the class spike becomes detectable: the BBP condition
    ``N p delta^2 > sqrt(V C)`` solved for N."""
    return np.sqrt(n_nouns * n_contexts) / (p * delta ** 2)


def permutation_null(M, n_perm, rng):
    """Top residual singular values of ``n_perm`` corpora resampled under independence with
    ``M``'s margins — the finite-sample null band for spike detection."""
    M = np.asarray(M, np.float64)
    q = np.outer(M.sum(1), M.sum(0)).ravel()
    N = int(M.sum())
    return np.array([np.linalg.svd(ca_residuals(rng.multinomial(N, q / q.sum()).reshape(M.shape)),
                                   compute_uv=False)[0] for _ in range(n_perm)])


def class_axis(M):
    """Top singular triple ``(u, v, S)`` of the residuals: ``u`` the per-noun class scores,
    ``v`` the context axis, ``S`` the full spectrum."""
    U, S, Vt = np.linalg.svd(ca_residuals(M), full_matrices=False)
    return U[:, 0], Vt[0], S


def identification(M, cls):
    """BBP order parameter: |corr| between the empirical class scores ``u`` and the *planted*
    singular direction — class signs weighted by ``sqrt`` of the (empirical) noun margins, since
    a noun's residual row scales with the root of its frequency. ~0 below the detection
    threshold, rising toward 1 above it; reduces to plain sign-correlation for uniform
    frequencies. Zipf frequencies smear the global transition (frequent nouns' rows align
    first) — the per-noun staircase ``per_noun_correct`` resolves."""
    u = class_axis(M)[0]
    ustar = (1.0 - 2.0 * np.asarray(cls)) * np.sqrt(np.asarray(M, np.float64).sum(1))
    return float(abs(np.corrcoef(u, ustar)[0, 1]))


def per_noun_correct(M, cls):
    """Whether each noun falls on its class's side of the empirical class axis (boolean per noun).
    True labels fix only the axis' sign — the gauge the |corr| in ``identification`` also mods out."""
    u = class_axis(M)[0]
    s = 1.0 - 2.0 * np.asarray(cls)
    return np.sign(u @ s) * u * s > 0


def one_shot_accuracy(M_train, cls_train, cls_test, delta, n_marked, rng):
    """The abstraction's payoff: classify *held-out* nouns from a single marked token each.

    The context axis ``v`` is estimated from training nouns only (its one global sign fixed by
    their labels — the gauge ``identification`` also mods out); each test noun draws one marked
    paradigm uniformly and one form from its class-conditional ``(1±delta)/2``, and is classified
    by which side of the axis that form falls on within its pair. The pairs' sign pattern is
    direction information, so below the detection threshold this is chance *on average* over
    corpora; above it, it approaches the Bayes ceiling ``(1+delta)/2``.
    """
    u, v, _ = class_axis(M_train)
    orient = np.sign(u @ (1.0 - 2.0 * np.asarray(cls_train)))
    d = orient * (v[0:2 * n_marked:2] - v[1:2 * n_marked:2])  # d[k] > 0: even form of pair k <-> class 0
    cls_test = np.asarray(cls_test)
    pair = rng.integers(n_marked, size=len(cls_test))
    p_even = np.where(cls_test == 0, (1 + delta) / 2, (1 - delta) / 2)
    even = rng.random(len(cls_test)) < p_even                 # one marked token per test noun
    pred = np.where(even == (d[pair] > 0), 0, 1)
    return float((pred == cls_test).mean())


def identification_curve(Q, cls, Ns, n_seeds=4, seed=0):
    """Identification vs corpus size for nested (cumulative) corpora: ``(n_seeds, len(Ns))``."""
    Ns = np.asarray(Ns)
    out = np.empty((n_seeds, len(Ns)))
    for s in range(n_seeds):
        rng = np.random.default_rng(seed * 1000 + s)
        M = np.zeros(Q.shape)
        for j, N in enumerate(Ns):
            M += sample_counts(Q, int(N) - (int(Ns[j - 1]) if j else 0), rng)
            out[s, j] = identification(M, cls)
    return out


def payoff_curve(Q, cls, Ns, delta, n_marked, n_seeds=4, seed=0):
    """One-shot accuracy vs corpus size for nested corpora, ``(n_seeds, len(Ns))``. The context
    axis is estimated on the even-index nouns; the odd-index nouns are the held-out test set."""
    cls = np.asarray(cls)
    train, test = np.arange(len(cls))[::2], np.arange(len(cls))[1::2]
    Ns = np.asarray(Ns)
    out = np.empty((n_seeds, len(Ns)))
    for s in range(n_seeds):
        rng = np.random.default_rng(seed * 1000 + s)
        M = np.zeros(Q.shape)
        for j, N in enumerate(Ns):
            M += sample_counts(Q, int(N) - (int(Ns[j - 1]) if j else 0), rng)
            out[s, j] = one_shot_accuracy(M[train], cls[train], cls[test], delta, n_marked, rng)
    return out


def acquisition_N(Q, cls, Ns, seed=0):
    """Per-noun age of acquisition on one growing corpus: the smallest ``Ns[j]`` from which the
    noun is correctly classified at *every* subsequent grid point (``inf`` if never)."""
    rng = np.random.default_rng(seed)
    Ns = np.asarray(Ns)
    M = np.zeros(Q.shape)
    correct = np.empty((len(Ns), Q.shape[0]), dtype=bool)
    for j, N in enumerate(Ns):
        M += sample_counts(Q, int(N) - (int(Ns[j - 1]) if j else 0), rng)
        correct[j] = per_noun_correct(M, cls)
    stays = np.logical_and.accumulate(correct[::-1], axis=0)[::-1]   # correct from j onward
    aoa = np.where(stays.any(0), Ns[stays.argmax(0)], np.inf)
    return aoa.astype(np.float64)


def threshold_N(Ns, curves, level=0.5):
    """Measured emergence threshold: the corpus size where the median identification curve first
    crosses ``level``, log-interpolated between grid points."""
    Ns = np.asarray(Ns, np.float64)
    med = np.median(curves, axis=0)
    (above,) = np.nonzero(med >= level)
    if len(above) == 0:
        raise ValueError(f"identification never reaches {level}: max {med.max():.3f}")
    i = above[0]
    if i == 0:
        return float(Ns[0])
    t = (level - med[i - 1]) / (med[i] - med[i - 1])
    return float(np.exp(np.log(Ns[i - 1]) + t * (np.log(Ns[i]) - np.log(Ns[i - 1]))))
