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


def spectral_lpcfg(g):
    """Spectral learning of a single-nonterminal latent-variable PCFG (Cohen et al. 2012), from the
    grammar's *exact* observable moments — the tree analogue of ``spectral_hmm``.

    The observable features are: inside tree -> its first terminal (``I[x,h] = p(first terminal x |
    state h)``); outside tree -> the terminal following the span, plus an END marker (``J``, solved
    from an exact outside linear system). ``Ω = I diag(γ) Jᵀ`` (Lemma 1); its top-``m`` left singular
    vectors ``U`` fix the *data-determined* observable basis ``G = Uᵀ I``. The returned observable
    operators (Theorem 2) drive the tensor inside algorithm (``spectral_inside``); by Lemma 2 the
    inside vectors it computes equal the true inside terms up to the single invertible map ``G⁻¹`` —
    so the representation is pinned to the observable basis, not an arbitrary latent parameterization.

    Returns a dict with ``C`` (rule operator, a function ``y -> (m,m)``), ``c_inf`` (a function
    ``terminal -> (m,)``), ``c1`` (root, ``(m,)``), and ``G`` (the observable basis map, for tests).
    Requires the first-terminal feature to have rank ``m`` (holds when the alphabet is at least as
    expressive as the state space, e.g. identity emissions with ``V = m``)."""
    pi = np.asarray(g.pi, float); pb = np.asarray(g.p_branch, float); e = 1.0 - pb
    S = np.asarray(g.S, float); Tt = np.asarray(g.Tt, float); Em = np.asarray(g.Em, float)
    m = len(pi); V = Em.shape[1]
    R = np.linalg.inv(np.eye(m) - pb[:, None] * S)                   # left-spine descent resolvent
    LT = R @ (e[:, None] * Em)                                       # (m,V)  LT[h,x]=p(first terminal x|h)
    I = LT.T                                                         # (V,m)  inside features
    n = pi @ np.linalg.inv(np.eye(m) - pb[:, None] * (S + Tt))       # expected #nodes per state
    gamma = n / n.sum()                                             # state dist at a random node
    # outside feature J[:,h] = dist of the terminal following a state-h node (+ END at index V):
    # a node is a left child (following = right sibling's first terminal), a right child (following =
    # parent's following terminal — a fixed point), or the root (END). Weighted by expected counts.
    W_R = (n * pb)[:, None] * Tt                                     # right-child edge weights
    W_L = (n * pb)[:, None] * S                                      # left-child edge weights
    Src = np.zeros((V + 1, m))
    Src[:V, :] = (Tt @ LT).T @ W_L                                   # left child -> sibling first terminal
    Src[V, :] = pi                                                   # root -> END
    J = Src @ np.linalg.inv(np.diag(n) - W_R)                        # (V+1,m)
    Omega = I @ (gamma[:, None] * J.T)                              # (V,V+1) = I diag(γ) Jᵀ
    U = np.linalg.svd(Omega, full_matrices=False)[0][:, :m]          # observable subspace (rank m)
    G = U.T @ I                                                     # (m,m) invertible basis map
    Ginv = np.linalg.inv(G); GTtT = G @ Tt.T
    def C(y):                                                       # observable rule operator (Thm 2)
        d = (y @ G) @ S.T
        return GTtT @ ((d * pb)[:, None] * Ginv)
    def c_inf(x):                                                   # observable emission operator
        return (e * Em[:, x]) @ Ginv
    return {"C": C, "c_inf": c_inf, "c1": G @ pi, "G": G}


def spectral_inside(ops, tree):
    """Tensor inside algorithm (Cohen et al. Fig 2) on a skeletal ``tree`` (no latent states):
    a leaf is ``("leaf", terminal)``, an internal node ``("node", left, right)``. Returns
    ``(f_root, p)``: the root spectral inside vector (= true inside up to ``G⁻¹``) and ``p(tree)``."""
    C, c_inf = ops["C"], ops["c_inf"]
    def f(t):
        if t[0] == "leaf":
            return c_inf(t[1])
        return f(t[2]) @ C(f(t[1]))                                  # f_i = f_right · C(f_left)
    fr = f(tree)
    return fr, float(fr @ ops["c1"])
