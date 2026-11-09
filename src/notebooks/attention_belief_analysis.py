import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The belief after attention, and every stage predicted

    An exchangeable stream: $z$ is drawn once, then $x_{1:t}$ i.i.d. $p(x \mid z)$. The Bayes state is the count
    vector $n$, and the posterior over $z$ is log-linear in it,
    $\log p(z \mid x_{1:t}) = \log p(z) + \sum_x n_x \log p(x \mid z)$ — each token adds its likelihood vector.
    A 1-layer transformer trained on next-token prediction must route this through one attention block and one MLP.

    Causal attention averages: right after it the residual can hold the frequencies $n / t$ (and, through a start
    token taking a share of the weight, $(c + n)/(c' + t)$), not the counts. So the posterior over $z$ is linear in the
    post-attention residual *at each position* (a frequency times $t$), and needs $t$ to be read by one map across
    positions. The next-token predictive is a mixture over $z$, a nonlinear function of $n$ left to the MLP. Two
    worlds: the line and the week of the inference notebook.

    The measure throughout is held-out ridge regression (`heldout_decode_r2`: standardized states, held-out streams
    wholly unseen). Sections 1–2 decode Bayes targets from the states at a site — the embedding, the residual after
    attention, and the readout's input after the MLP. Section 3 runs the other way: each stage's representation is
    predicted from Bayes alone and the prediction mapped onto the trained states.
    """)
    return


@app.cell
def _(mo):
    import os, sys
    import numpy as np

    ROOT = mo.notebook_dir().parent.parent    # src/notebooks -> <repo>
    os.chdir(ROOT); sys.path.insert(0, str(ROOT))
    from src.code.analysis_utils import heldout_decode_r2, load_run, readout_basis
    from src.code.geometry import factor_logcond
    from src.code.train import graph_overrides
    from src.code.plot_utils import FIG, save_figure
    import matplotlib.pyplot as plt

    from pathlib import Path
    FIGURES = Path(os.environ.get("CACHE_DIR", ROOT / "outputs")) / "figures"
    FIGURES.mkdir(parents=True, exist_ok=True)
    return (
        FIG,
        FIGURES,
        factor_logcond,
        graph_overrides,
        heldout_decode_r2,
        load_run,
        np,
        plt,
        readout_basis,
        save_figure,
    )


@app.cell
def _(graph_overrides, load_run):
    # the hidden worlds of the inference notebook (13 temperatures read through 5 adjectives, 28 times of week through
    # 7 days); trained by src/scripts/train_next_token.sh. The Dirichlet-categorical is left out: with 3 tokens and 2
    # evidence modes any value vectors fit its evidence, so the token-level test below cannot fail there
    WORLDS = {"line": graph_overrides({"structure": "chain", "cooccurrence": "gaussian", "n": 13, "m": 5, "gamma": 0.02}),
              "week": graph_overrides({"structure": "ring", "cooccurrence": "gaussian", "n": 28, "m": 7, "gamma": 0.8})}
    T = 12
    COMMON = ["runner=transformer", "sample_mode=sequence", "++infer_latent=false",
              f"training_kwargs.context_length={T}", "training_kwargs.batch_size=256",
              "training_kwargs.train_steps=30000", "model_kwargs.model_config.hidden_size=64",
              "model_kwargs.model_config.num_hidden_layers=1"]
    runs = {_name: load_run(_ov + COMMON, project=False) for _name, _ov in WORLDS.items()}
    for _name, _run in runs.items():
        print(f"{_name}: KL(Bayes || model) {_run['loss']:.4f} at step {_run['step']}")
    return COMMON, T, runs


@app.cell
def _(T, np, runs):
    def targets(run):
        """Per state (the position after x_1:t, t = 0..T-1): the counts, the z-centred log-posterior over z
        (log prior plus summed log-likelihoods), and the Bayes next-token predictive and its log. Each (B, T, .)."""
        ds = run["dataset"]
        obs = run["inputs"][:, 1:]
        n = np.concatenate([np.zeros((len(obs), 1, ds.n_outcomes)),
                            np.eye(ds.n_outcomes)[obs].cumsum(1)[:, :-1]], 1)
        ll = ds.conditional                                              # (n_z, K): log p(x | z), float64
        lp = n @ ll.T + np.log(ds.prior.cpu().numpy())
        pred = run["targets"].astype(float)
        return {"counts": n, "log-posterior over z": lp - lp.mean(-1, keepdims=True),
                "predictive": pred, "log-predictive": np.log(pred)}

    def sites(run):
        """States at each site, (B, T, width), positions 0..T-1 (the last position predicts past the context)."""
        return {"embedding": run["hidden_states"][0][:, :T], "after attention": run["attention_states"][0][:, :T],
                "after MLP": run["hidden_states"][-1][:, :T]}

    data = {_name: (targets(_run), sites(_run)) for _name, _run in runs.items()}
    return data, sites, targets


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. One ridge map across positions

    Each cell: held-out $R^2$ of a target from every state of a site, all positions pooled into one map.
    """)
    return


@app.cell
def _(data, heldout_decode_r2, np):
    def pooled(Y, H):
        B, T = H.shape[:2]
        return heldout_decode_r2(H.reshape(B * T, -1), Y.reshape(B * T, -1), np.repeat(np.arange(B), T))

    pooled_r2 = {_name: {_k: {_s: pooled(_Y, _H) for _s, _H in _sites.items()} for _k, _Y in _targets.items()}
                 for _name, (_targets, _sites) in data.items()}
    for _name, _rows in pooled_r2.items():
        print(_name)
        for _k, _row in _rows.items():
            print(f"   {_k:22s}" + "".join(f"  {_s} {_v:.3f}" for _s, _v in _row.items()))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. One ridge map per position

    The same decoding fitted and scored separately at each position $t$ (from $t = 1$: at $t = 0$ every stream sits at
    the start token). Solid: the log-posterior over $z$; dashed: the next-token predictive. Attention writes a mean, so
    the posterior over $z$ should be read exactly here even where the pooled map falls short.
    """)
    return


@app.cell
def _(FIG, FIGURES, data, heldout_decode_r2, np, plt, save_figure):
    def per_position(Y, H):
        return np.array([heldout_decode_r2(H[:, t], Y[:, t], np.arange(len(H))) for t in range(1, H.shape[1])])

    SITE_COLORS = {"embedding": FIG.colors["green"], "after attention": FIG.colors["blue"],
                   "after MLP": FIG.colors["orange"]}
    fig_pos, _axes = plt.subplots(1, len(data), figsize=(5 * len(data), 4), sharey=True)
    for _ax, (_name, (_targets, _sites)) in zip(_axes, data.items()):
        print(_name)
        for _s, _H in _sites.items():
            _t = np.arange(1, _H.shape[1])
            _post, _pred = per_position(_targets["log-posterior over z"], _H), per_position(_targets["predictive"], _H)
            print(f"   {_s:16s} log-posterior over z {np.round(_post, 3)}\n   {'':16s} predictive           {np.round(_pred, 3)}")
            _ax.plot(_t, _post, "-", color=SITE_COLORS[_s], label=_s)
            _ax.plot(_t, _pred, "--", color=SITE_COLORS[_s])
        _ax.set_title(_name); _ax.set_xlabel("position $t$"); _ax.set_ylim(0, 1.02)
        FIG.style_axes(_ax)
    _axes[0].set_ylabel("held-out ridge $R^2$")
    _axes[0].legend(frameon=False, loc="lower right")
    save_figure(fig_pos, FIGURES / "attention_belief_per_position.png")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Predicting every stage

    Everything below comes from Bayes; only a linear map onto the trained states is fitted.

    * **Evidence vectors (what attention reads out of each token).** The log-likelihood, centred over $z$ and over
      tokens, factors as $\ell[z, x] = E_z(z)^\top D(x)$: token $x$ adds $D(x) \in \mathbb{R}^r$ to the state. What is
      left after centring over tokens alone is a mode common to every token, $c(z)$, which a stream of $t$ tokens
      contributes $t\,c(z)$ of: a count, not evidence about which token was seen. Tested directly on what the attention
      block reads out of each token, its value vector $v(x) = W_O W_V \mathrm{LN}(e(x))$ (position-free: rotary touches
      queries and keys only): $v(x)$ should be an affine image of $D(x)$. An affine map from $\mathbb{R}^r$ fits any $r + 1$
      points, so this can fail only with more tokens than that (line: 5 tokens, $r = 1$; week: 7 tokens, $r = 2$). The
      null is every reassignment of the $D(x)$ to the tokens; the symmetries of the structure (the line's reversal, the
      week's rotations and reflections) tie with the true order, so the smallest attainable $p$ is their share.
    * **After attention.** Uniform causal attention over the start token and $x_{1:t}$ writes the mean,
      $\sum_i D(x_i) / (1 + t)$, and the start's share $1/(1 + t)$ (which carries $t$, hence the common mode); the
      residual adds the current token. The posterior over $z$ is these times $1 + t$:
      $\log p(z \mid x_{1:t}) = (1 + t)\,E_z(z)^\top \bar{D}_t + t\,c(z) + \mathrm{const}$.
    * **Before the readout.** The factored log-predictive of the sampled prefixes, $\Phi$ (as in the inference notebook).
    * **Readout.** The predictive's decoder $D_x$, carried into the trained coordinates through the ridge map of the
      trained states onto $\Phi$, compared with the trained readout in the metric of the states it reads (raw weight
      entries are dominated by directions the states barely visit, which the loss leaves free). Given near-Bayes
      logits and states that are an affine image of $\Phi$, this one is implied rather than independent.

    Controls: for the evidence, the same prediction with one mode fewer and with every token's frequency (the most
    a mean over tokens could carry); for the state after attention, the posterior's own coordinates, $\sum_i D(x_i)$
    and the count $t$ — the sum the derivation writes, rather than the mean attention computes.
    """)
    return


@app.cell
def _(T, data, factor_logcond, heldout_decode_r2, np, readout_basis, runs):
    def evidence(ds, tol=1e-3):
        """Token evidence vectors: the log-likelihood centred over z and over tokens, factored; (K, r) rows D(x)
        for the r modes above ``tol`` of the leading one."""
        ll = ds.conditional.T                                            # (K, n_z): rows tokens
        Lcc = ll - ll.mean(1, keepdims=True) - ll.mean(0, keepdims=True) + ll.mean()
        U, S, _ = np.linalg.svd(Lcc, full_matrices=False)
        r = int((S > tol * S[0]).sum())
        return U[:, :r] * np.sqrt(S[:r])

    def predicted_after_attention(run, n, D):
        """[current token, mean evidence over the start and x_1:t, the start's share] per state, (B, T, .)."""
        tok = run["inputs"][:, :T]
        share = 1.0 / (1 + n.sum(-1, keepdims=True))
        return np.concatenate([np.eye(run["dataset"].n_outcomes + 1)[tok], n @ D * share, share], -1)

    def affine_r2(X, Y):
        """In-sample R^2 of the least-squares affine map X -> Y."""
        X1 = np.c_[X, np.ones(len(X))]
        res = Y - X1 @ np.linalg.lstsq(X1, Y, rcond=None)[0]
        return float(1 - (res ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())

    def value_fit(run, D):
        """R^2 of the outcome tokens' value vectors from their evidence vectors, and its permutation p: the share of
        token reassignments (all K! of them) fitting at least as well."""
        from itertools import permutations
        V = run["value_vectors"][:len(D)]                                # outcome tokens come first in the vocabulary
        r2 = affine_r2(D, V)
        null = np.array([affine_r2(D[list(_p)], V) for _p in permutations(range(len(D)))])
        return r2, float((null >= r2 - 1e-9).mean())

    def predicted_readout(H, Phi, Dx):
        """The predictive's decoder in the trained state coordinates: Phi ~ H A + a, so logits Phi Dx = H (A Dx).
        Returns (K, width), centred over outputs."""
        A = np.linalg.lstsq(np.c_[H, np.ones(len(H))], Phi, rcond=None)[0][:-1]
        W = (A @ Dx).T
        return W - W.mean(0)

    def state_metric_r2(W, Wp, H):
        """R^2 of a predicted readout in the metric of the states it reads: both times C^(1/2), C = cov(H)."""
        lam, V = np.linalg.eigh(np.cov(H.T))
        C_half = (V * np.sqrt(np.clip(lam, 0, None))) @ V.T
        a, b = W @ C_half, Wp @ C_half
        return float(1 - ((a - b) ** 2).sum() / (a ** 2).sum())

    def stage_scores(run, targets, sites, tol=1e-3):
        ds, n = run["dataset"], targets["counts"]
        B, K = n.shape[0], ds.n_outcomes
        g = np.repeat(np.arange(B), T)
        fit = lambda X, H: heldout_decode_r2(X.reshape(B * T, -1), H.reshape(B * T, -1), g)   # predicted -> trained
        D = evidence(ds, tol)
        Ha = sites["after attention"]
        v_r2, v_p = value_fit(run, D)
        out = {"evidence rank r": D.shape[1],
               "value vectors: D(x) -> W_O W_V LN(e(x))": v_r2,
               "  permutation p": v_p,
               "after attention: mean of D (rank r)": fit(predicted_after_attention(run, n, D), Ha),
               "  control: rank r - 1": fit(predicted_after_attention(run, n, D[:, :-1]), Ha),
               "  control: every token's frequency": fit(predicted_after_attention(run, n, np.eye(K)), Ha),
               "  control: sum of D and t (log-posterior coords)": fit(np.concatenate(
                   [np.eye(K + 1)[run["inputs"][:, :T]], n @ D, n.sum(-1, keepdims=True)], -1), Ha)}
        _, Phi, Dx, S = factor_logcond(targets["log-predictive"].reshape(B * T, K))
        rx = int((S > tol * S[0]).sum())
        Rb = readout_basis(run["readout"])
        Hf = (sites["after MLP"] @ Rb.T).reshape(B * T, -1)               # what the readout reads
        out["predictive rank"] = rx
        out["before readout: Phi"] = fit(Phi[:, :rx], Hf)
        W = run["readout"] @ Rb.T
        out["readout (state metric)"] = state_metric_r2(W - W.mean(0), predicted_readout(Hf, Phi[:, :rx], Dx[:rx]), Hf)
        return out

    stages = {_name: stage_scores(runs[_name], *data[_name]) for _name in runs}
    _rows = list(next(iter(stages.values())))
    print(f"{'':50s}" + "".join(f"{_name:>11s}" for _name in stages))
    for _row in _rows:
        print(f"{_row:50s}" + "".join(f"{_s[_row]:>11.4f}" if isinstance(_s[_row], float) else f"{_s[_row]:>11d}"
                                      for _s in stages.values()))
    return affine_r2, evidence, predicted_after_attention, stages


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Log or probability coordinates, and what attention keeps

    **Which per-token vector.** Here Bayes is a sum of log-likelihoods, so each token should write $D(x)$. The
    constrained-belief account (Piotrowski et al. 2025, for HMMs) has each token write a probability displacement,
    $p(z \mid x) - \pi$; with no transitions, its Eq. 5 becomes exactly that. Both are vectors over $z$, both centred
    over tokens and cut to their top $k$ modes, then fitted affinely to the value vectors. An affine map fits any
    $k + 1$ points, so the contest is decided only by tokens beyond that: line and week have them, Mess3 (3 tokens, 2
    modes) does not. The same contest is repeated on the post-attention state, with the probability displacement in
    place of $D$ inside the mean. It turns out undecidable here too: the two accounts' top-$r$ token subspaces
    coincide (cosines printed below). The week's is forced: a map commuting with the ring's rotations keeps its
    Fourier modes, so both lead with the first harmonic at any sharpness. The line's differs only at sharp
    likelihoods (a scratch sweep over $\gamma$: the probability modes from the log modes $R^2$ 0.9999 at
    $\gamma = 0.02$, 0.97 at $\gamma = 1$, the sharpest float32 holds). Deciding it needs a world without the symmetry.

    **What attention keeps.** The frequency vector splits into its part in $\mathrm{span}(1, D)$ (the evidence) and the
    rest, $n\,Q$, which says which tokens were seen without bearing on $z$. If attention compresses to the belief, the
    state after it holds no more of $n\,Q$ than the predicted state does. The predicted state is not zero on it:
    tokens covary through $z$. Per position, held-out $R^2$ of $n\,Q$ from the predicted state, from the trained
    state, and from both; trained adding nothing over predicted means the rest is discarded.
    """)
    return


@app.cell
def _(T, affine_r2, data, evidence, heldout_decode_r2, np, predicted_after_attention, runs):
    from itertools import permutations

    def top_modes(M, k):
        """Rows of M (tokens) centred over tokens, cut to their top k modes: (K, k)."""
        U, S, _ = np.linalg.svd(M - M.mean(0), full_matrices=False)
        return U[:, :k] * np.sqrt(S[:k])

    def token_vectors(ds):
        """Per token, over z: the log-likelihood (z-centred) and the probability displacement p(z | x) - pi."""
        ll, prior = ds.conditional.T, ds.prior.double().cpu().numpy()   # (K, n_z), (n_z,)
        post = np.exp(ll) * prior
        return {"log-likelihood D(x)": ll - ll.mean(1, keepdims=True),
                "probability p(z|x) - pi": post / post.sum(1, keepdims=True) - prior}

    def contest(run, k):
        """Per account: value-vector R^2 at k modes and its permutation p."""
        V = run["value_vectors"][:run["dataset"].n_outcomes]
        out = {}
        for _name, M in token_vectors(run["dataset"]).items():
            X = top_modes(M, k)
            r2 = affine_r2(X, V)
            null = np.array([affine_r2(X[list(_p)], V) for _p in permutations(range(len(X)))])
            out[_name] = (r2, float((null >= r2 - 1e-9).mean()))
        return out

    def kept(run, targets, sites):
        """Per position t >= 1: held-out R^2 of n Q from the predicted state, the trained state, and both."""
        n, Ha = targets["counts"], sites["after attention"]
        D = evidence(run["dataset"])
        K, r = n.shape[-1], D.shape[1]
        Q = np.linalg.svd(np.c_[np.ones(K), D])[0][:, r + 1:]          # token directions outside span(1, D)
        Y, X0 = n @ Q, predicted_after_attention(run, n, D)
        g = np.arange(len(n))
        dec = lambda X, t: heldout_decode_r2(X[:, t], Y[:, t], g)
        out = {_k: np.array([dec(_X, t) for t in range(1, T)])
               for _k, _X in {"predicted": X0, "trained": Ha, "both": np.concatenate([X0, Ha], -1)}.items()}
        # how much of the state it is: the share of the state's variance n Q adds over the predicted state
        var = lambda X, t: heldout_decode_r2(X[:, t], Ha[:, t], g)
        out["state variance added"] = np.array([var(np.concatenate([X0, Y], -1), t) - var(X0, t) for t in range(1, T)])
        return out

    def overlap(ds, k):
        """Cosines of the principal angles between the two accounts' top-k token subspaces."""
        A, B = (np.linalg.qr(top_modes(_M, k))[0] for _M in token_vectors(ds).values())
        return np.linalg.svd(A.T @ B, compute_uv=False)

    coords, keeps = {}, {}
    for _name, _run in runs.items():
        _targets, _sites = data[_name]
        K, r = _run["dataset"].n_outcomes, evidence(_run["dataset"]).shape[1]
        coords[_name] = {k: contest(_run, k) for k in range(1, K - 1)}
        _n = _targets["counts"]
        _B = len(_n)
        _g = np.repeat(np.arange(_B), T)
        _fit = lambda X: heldout_decode_r2(X.reshape(_B * T, -1), _sites["after attention"].reshape(_B * T, -1), _g)
        coords[_name]["state"] = {_acc: _fit(predicted_after_attention(_run, _n, top_modes(_M, r)))
                                  for _acc, _M in token_vectors(_run["dataset"]).items()}
        keeps[_name] = kept(_run, _targets, _sites)
        print(f"{_name} (evidence rank r = {r})")
        for k in range(1, K - 1):
            print(f"   value vectors, k = {k}" + "".join(f"   {_acc}: {_v[0]:.4f} (p {_v[1]:.3f})"
                                                      for _acc, _v in coords[_name][k].items()))
        print("   after attention, k = r" + "".join(f"   {_acc}: {_v:.4f}" for _acc, _v in coords[_name]["state"].items()))
        print("   subspace cosines, k = 1..K-2: " + "  ".join(str(np.round(overlap(_run["dataset"], k), 4))
                                                       for k in range(1, K - 1)))
        for _k, _v in keeps[_name].items():
            print(f"   n Q {_k:22s} {np.round(_v, 4)}  mean {_v.mean():.4f}")
    return coords, keeps, overlap


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Line and week together

    Every test above, one row each. $n\,Q$ rows are means over positions $t \geq 1$.
    """)
    return


@app.cell
def _(coords, keeps, overlap, runs, stages):
    def summary(name):
        run, s, c, kp = runs[name], stages[name], coords[name], keeps[name]
        r = s["evidence rank r"]
        ll, pr = "log-likelihood D(x)", "probability p(z|x) - pi"
        return {"KL(Bayes || model)": run["loss"],
                "evidence rank r / predictive rank": f"{r} / {s['predictive rank']}",
                "value vectors <- D(x)  [perm p]": f"{c[r][ll][0]:.4f} [{c[r][ll][1]:.3f}]",
                "value vectors <- p(z|x) - pi  [perm p]": f"{c[r][pr][0]:.4f} [{c[r][pr][1]:.3f}]",
                "  the two accounts' top-r subspaces, min cosine": overlap(run["dataset"], r).min(),
                "after attention <- mean of D": s["after attention: mean of D (rank r)"],
                "after attention <- mean of p(z|x) - pi": c["state"][pr],
                "  control: rank r - 1": s["  control: rank r - 1"],
                "  control: sum of D and t": s["  control: sum of D and t (log-posterior coords)"],
                "  control: every token's frequency": s["  control: every token's frequency"],
                "n Q decoded from predicted state": kp["predicted"].mean(),
                "n Q decoded from trained state": kp["trained"].mean(),
                "n Q share of trained-state variance": kp["state variance added"].mean(),
                "before readout <- Phi": s["before readout: Phi"],
                "readout (state metric)": s["readout (state metric)"]}

    table = {_name: summary(_name) for _name in runs}
    print(f"{'':50s}" + "".join(f"{_name:>18s}" for _name in table))
    for _row in next(iter(table.values())):
        print(f"{_row:50s}" + "".join(f"{_v:>18.4f}" if isinstance(_v, float) else f"{_v:>18s}"
                                      for _v in (table[_n][_row] for _n in table)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. When no belief is needed: the Dirichlet-categorical

    A belief distinct from the predictive is needed when the predictive is not a linear image of what attention can
    aggregate, even in probability coordinates — marginalizing over $z$ is the case of line and week. The
    Dirichlet-categorical is the control: its predictive $(1 + n_x)/(K + t)$ is itself a mean, so attention can write it
    directly, the start token taking the prior's share $K/(K + t)$. The MLP is left only the softmax's log, the same
    for every coordinate. Three tests on all four worlds, each one ridge map per position (mean over $t \geq 1$):

    1. the predictive probabilities from the state after attention;
    2. that same linear readout, then a pointwise log, against the log-predictive — what is left to the MLP if all it
       does is the link;
    3. the state after attention from the current token and the predictive probabilities alone — nothing more held.

    Predicted: all three $\approx 1$ for the Dirichlet, short of 1 where the predictive is a marginal over $z$ (line,
    week, and Tenenbaum's hypothesis spaces — intervals, subtrees, keys, rectangles — where the marginal is over
    sets under the size principle and the belief is of full rank $K - 1$, as the Dirichlet's). Per position, not across: attention averages uniformly over the start and $x_{1:t}$ in every world, the
    start counting as $c \approx 1$ pseudo-count rather than the prior's $K$ (last row), so it writes $[n, 1]/(c + t)$;
    the rescaling by $t$ is the MLP's, alongside the log, and costs it nothing.
    """)
    return


@app.cell
def _(COMMON, T, data, graph_overrides, heldout_decode_r2, load_run, np, runs, sites, targets):
    # the controls, and Tenenbaum's hypothesis spaces (size principle; evidence and predictive both of rank K - 1, so
    # the frequencies are already minimal and the worlds differ from the Dirichlet only past attention)
    controls = {"Dirichlet K=3": load_run(["setting=dirichlet"] + COMMON, project=False),
                "Dirichlet K=7": load_run(["setting=dirichlet", "n_outcomes=7", "lattice=14"] + COMMON, project=False),
                **{_s: load_run(graph_overrides({"structure": _s}) + [_o.replace("train_steps=30000", "train_steps=150000")
                                                                      for _o in COMMON], project=False)
                   for _s in ("intervals", "subtrees", "keys", "rectangles")}}       # these need 5x the steps
    for _name, _run in controls.items():
        print(f"{_name}: KL(Bayes || model) {_run['loss']:.5f} at step {_run['step']}")

    def at_position(run, targets, sites, t, seed=0):
        """The three tests at position t, one map each, scored on held-out streams."""
        Ha, p, lp = sites["after attention"][:, t], targets["predictive"][:, t], targets["log-predictive"][:, t]
        B = len(Ha)
        tok = np.eye(run["dataset"].n_outcomes + 1)[run["inputs"][:, t]]
        train = np.isin(np.arange(B), np.random.default_rng(seed).permutation(B)[:int(0.6 * B)])
        X1 = np.c_[Ha, np.ones(B)]
        p_hat = np.clip(X1[~train] @ np.linalg.lstsq(X1[train], p[train], rcond=None)[0], 1e-4, None)
        centre = lambda L: L - L.mean(1, keepdims=True)
        res, lc = centre(np.log(p_hat)) - centre(lp[~train]), centre(lp[~train])
        return [heldout_decode_r2(Ha, p, np.arange(B)),
                float(1 - (res ** 2).sum() / ((lc - lc.mean(0)) ** 2).sum()),
                heldout_decode_r2(np.c_[tok, p], Ha, np.arange(B))]

    def interface(run, targets, sites):
        per_t = np.array([at_position(run, targets, sites, t) for t in range(1, T)]).mean(0)
        # why one map across positions fails: attention writes [n, 1] / (c + t), the start as c pseudo-counts
        B, n = len(targets["counts"]), targets["counts"]
        flat = lambda X: X.reshape(B * T, -1)
        tok = flat(np.eye(run["dataset"].n_outcomes + 1)[run["inputs"][:, :T]])
        sink = lambda c: flat(np.concatenate([n, np.ones_like(n[..., :1])], -1) / (c + n.sum(-1, keepdims=True)))
        cs = np.geomspace(0.25, 32, 22)
        fits = [heldout_decode_r2(np.c_[tok, sink(c)], flat(sites["after attention"]), np.repeat(np.arange(B), T))
                for c in cs]
        return {"1. predictive <- after attention": per_t[0],
                "2. log of that readout vs log-predictive": per_t[1],
                "3. after attention <- [token, predictive]": per_t[2],
                "start's pseudo-count c (pooled fit; Bayes: K)": float(cs[int(np.argmax(fits))])}

    _all = {**{_n: (runs[_n], *data[_n]) for _n in runs},
            **{_n: (_r, targets(_r), sites(_r)) for _n, _r in controls.items()}}
    interfaces = {_n: interface(*_v) for _n, _v in _all.items()}
    print(f"{'':46s}" + "".join(f"{_n:>14s}" for _n in interfaces))
    for _row in next(iter(interfaces.values())):
        print(f"{_row:46s}" + "".join(f"{_s[_row]:>14.4f}" for _s in interfaces.values()))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Figure: the decomposition, stage by stage (week)

    In the inference notebook's style: per stage, the Bayes prediction beside the transformer's states, rotated (no
    stretch) onto it, with the Grams of the mean path per true latent and the CKA over every state. A strip on top
    marks the stage. (a) what each token writes: $D(x)$ against the value vectors; (b) after attention: the mean of
    $D$ over the start and $x_{1:t}$ — the log-posterior, up to the count — each state coloured by the Bayes posterior
    over $z$ (on the trained side, the same: colour is the target, position the test); (c) after the MLP: the
    log-predictive's top modes, coloured by the predictive over tokens.
    """)
    return


@app.cell
def _(FIG, FIGURES, T, data, evidence, factor_logcond, np, plt, runs, save_figure):
    import io
    import matplotlib.patheffects as pe
    from matplotlib.patches import FancyBboxPatch
    from src.code.analysis_utils import cka, rigid_map
    from src.code.plot_utils import gram_heatmap, merge_panels

    FONT = 18
    NODES = [r"$x_{1:t}$", "attention", "MLP", "readout"]

    def stage_strip(ax, active, equation):
        """The computation graph, the active stage filled: a node index, or (i, i + 1) for the edge between."""
        xs = np.linspace(0.11, 0.89, len(NODES))
        for _i, (_x, _label) in enumerate(zip(xs, NODES)):
            on = active == _i
            ax.add_patch(FancyBboxPatch((_x - 0.085, 0.52), 0.17, 0.3, boxstyle="round,pad=0.01,rounding_size=0.04",
                                        fc=FIG.colors["blue"] if on else "white", ec="0.25", lw=1.2, clip_on=False))
            ax.text(_x, 0.67, _label, ha="center", va="center", fontsize=FONT - 4, color="white" if on else "0.15")
            if _i:
                edge = active == (_i - 1, _i)
                ax.annotate("", (_x - 0.09, 0.67), (xs[_i - 1] + 0.09, 0.67),
                            arrowprops=dict(arrowstyle="-|>", lw=3.2 if edge else 1.2,
                                            color=FIG.colors["blue"] if edge else "0.35"))
        ax.text(0.5, 0.2, equation, ha="center", va="center", fontsize=FONT - 3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    def gauge(dirs):
        """Rotation/reflection putting the first token's direction up and the rest clockwise. dirs (K, 2)."""
        a = np.pi / 2 - np.arctan2(dirs[0, 1], dirs[0, 0])
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        d = dirs @ R.T
        return (np.diag([-1.0, 1.0]) @ R) if d[0, 0] * d[1, 1] - d[0, 1] * d[1, 0] > 0 else R

    def sides(pred, trained, dirs):
        """Both codes (N, r) and (N, k): the trained one rotated onto the prediction in the full space, then
        both shown in the prediction's top-2 modes, in the gauge of ``dirs`` (K, 2) (token directions there)."""
        X = trained @ rigid_map(trained, pred)
        G = gauge(dirs)
        return pred[:, :2] @ G.T, X[:, :2] @ G.T

    STREAM_STYLES = [("0.1", "Greys"), ("#8C3B0E", "Oranges")]          # (line, dots light -> dark with t) per stream

    def panel(ax, title, pts, rgb, paths=(), dot=34):
        ax.scatter(*pts.T, s=dot, color=rgb, alpha=0.6, lw=0, zorder=2)
        for path, (line, dots) in zip(paths, STREAM_STYLES):
            ax.plot(*path.T, color=line, lw=1.8, zorder=4,
                    path_effects=[pe.Stroke(linewidth=4.0, foreground="white"), pe.Normal()])
            ax.scatter(*path.T, c=np.linspace(0.25, 1.0, len(path)), cmap=dots, vmin=0, vmax=1,
                       s=70, zorder=5, edgecolor=line, lw=0.9)
        lo, hi = pts.min(0), pts.max(0); pad = 0.08 * (hi - lo).max()
        ax.set_xlim(lo[0] - pad, hi[0] + pad); ax.set_ylim(lo[1] - pad, hi[1] + pad)
        ax.set_aspect("equal"); FIG.clean(ax); ax.set_title(title, fontsize=FONT, pad=8)

    def block(active, equation, pred, trained, rgb, reps, score, paths=(), dot=34):
        """One stage: strip, the two planes, the two Grams. ``pred``/``trained`` (N, 2) shown; ``reps``
        (pred, trained) the Gram rows; ``paths`` (index arrays into the rows) the highlighted streams."""
        fig = plt.figure(figsize=(7.6, 10.4))
        gs = fig.add_gridspec(3, 2, height_ratios=[0.32, 1.15, 1], hspace=0.22, wspace=0.25)
        stage_strip(fig.add_subplot(gs[0, :]), active, equation)
        for _j, (_label, _pts) in enumerate([("predicted", pred), ("transformer", trained)]):
            panel(fig.add_subplot(gs[1, _j]), _label, _pts, rgb, [_pts[_p] for _p in paths], dot)
            _ax = fig.add_subplot(gs[2, _j])
            gram_heatmap(_ax, reps[_j], title=r"$E^{*}E^{*\top}$" if _j == 0 else r"$EE^{\top}$" + f"   CKA {score:.3f}")
            _ax.title.set_fontsize(FONT)
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=300, bbox_inches="tight"); plt.close(fig)
        buf.seek(0); return buf

    def ring_rgb(p):
        """Colour of a belief over points evenly spaced on a ring (N, K): hue its circular mean on the cyclic
        map, saturation its concentration (resultant length over its 99th percentile), white when flat."""
        c = p @ np.exp(2j * np.pi * np.arange(p.shape[1]) / p.shape[1])
        hue = plt.cm.twilight((np.angle(c) / (2 * np.pi)) % 1)[:, :3]
        sat = np.clip(np.abs(c) / np.quantile(np.abs(c), 0.99), 0, 1)[:, None]
        return 1 - sat * (1 - hue)

    def latent_means(code, z):
        """Mean code per true latent at every position, (n_z T, d): the Gram's rows."""
        return np.stack([code[z == _k].mean(0) for _k in range(int(z.max()) + 1)]).reshape(-1, code.shape[-1])

    def _figure(name):
        run = runs[name]
        targets_w, sites_w = data[name]
        ds = run["dataset"]
        z = run["latents"][:, 0]
        B, K = len(z), ds.n_outcomes
        tok_rgb = plt.cm.twilight(np.arange(K) / K)[:, :3]
        lat_rgb = plt.cm.twilight(np.arange(ds.n_items) / ds.n_items)[:, :3]
        flat = lambda X: X.reshape(B * T, -1)

        # (a) what each token writes
        D = evidence(ds)                                                     # (K, 2)
        V = run["value_vectors"][:K]
        Da, Va = sides(D - D.mean(0), V - V.mean(0), D)
        blk_a = block((0, 1), r"$v(x) \;\propto\; D(x)$, the token's log-likelihood over $z$",
                      Da, Va, tok_rgb, (D - D.mean(0), V - V.mean(0)), cka(V, D), dot=320)

        # (b) after attention: the mean of D over the start and x_1:t, coloured by the Bayes posterior over z
        n = targets_w["counts"]
        share = 1.0 / (1 + n.sum(-1, keepdims=True))
        # what attention writes, centred per position: the start's share is the same for every stream at a
        # position, so this drops its offset (whose size the theory leaves free) and keeps the evidence
        centre_t = lambda X: X - X.mean(0, keepdims=True)
        Eb = centre_t(n @ D * share)                                         # (B, T, 2)
        Hb = centre_t(sites_w["after attention"] - sites_w["embedding"])
        post = np.exp(targets_w["log-posterior over z"]); post /= post.sum(-1, keepdims=True)
        Pb, Tb = sides(flat(Eb), flat(Hb), D)
        _end = Pb.reshape(B, T, 2)[:, -1]; _r = np.linalg.norm(_end, axis=1)
        _far = np.flatnonzero((_r > np.quantile(_r, 0.85)) & (_r < np.quantile(_r, 0.95)))
        _angle = lambda idx, a: idx[np.argmin(np.abs(np.angle(np.exp(1j * (np.angle(_end[idx] @ [1, 1j]) - a)))))]
        b_far = int(_angle(_far, 3 * np.pi / 4))                          # an agreeing stream, out to the rim, heading up-left
        # a typical stream: ends at the median radius (its tokens partly disagree), heading down-right
        _mid = np.flatnonzero(np.abs(_r - np.median(_r)) < 0.02 * np.ptp(_r))
        b_mid = int(_angle(_mid, -np.pi / 4))
        paths = [np.arange(_b * T, (_b + 1) * T) for _b in (b_far, b_mid)]
        blk_b = block((1, 2),r"$\bar D_t = \frac{1}{1+t}\sum_{i \leq t} D(x_i)$:  $\log p(z \mid x_{1:t})$ up to $t$",
                      Pb, Tb, flat(post) @ lat_rgb, (latent_means(Eb, z), latent_means(Hb, z)), cka(flat(Hb), flat(Eb)), paths)

        # (c) after the MLP: the log-predictive's top modes, coloured by the predictive over tokens
        _, Ec, Dx, _ = factor_logcond(flat(targets_w["log-predictive"]))
        Ec, Dx = Ec[:, :K - 1], Dx[:K - 1]
        Hc = flat(sites_w["after MLP"])
        Pc, Tc = sides(Ec, Hc, Dx[:2].T)
        blk_c = block((2, 3),r"$\log p(x \mid x_{1:t}) = \log \sum_z p(x \mid z)\, p(z \mid x_{1:t})$",
                      Pc, Tc, ring_rgb(flat(targets_w["predictive"])),
                      (latent_means(Ec.reshape(B, T, -1), z), latent_means(Hc.reshape(B, T, -1), z)), cka(Hc, Ec), paths)

        fig_stages = merge_panels([blk_a, blk_b, blk_c], ncols=3, label_size=30)
        save_figure(fig_stages, FIGURES / f"decomposition_{name}.png", dpi=200)

    _figure("week")


if __name__ == "__main__":
    app.run()
