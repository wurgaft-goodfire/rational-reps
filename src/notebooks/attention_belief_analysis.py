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
    return T, runs


@app.cell
def _(T, np, runs):
    def targets(run):
        """Per state (the position after x_1:t, t = 0..T-1): the counts, the z-centred log-posterior over z
        (log prior plus summed log-likelihoods), and the Bayes next-token predictive and its log. Each (B, T, .)."""
        ds = run["dataset"]
        obs = run["inputs"][:, 1:]
        n = np.concatenate([np.zeros((len(obs), 1, ds.n_outcomes)),
                            np.eye(ds.n_outcomes)[obs].cumsum(1)[:, :-1]], 1)
        ll = np.log(ds.P.double().cpu().numpy())                        # (n_z, K): log p(x | z)
        lp = n @ ll.T + np.log(ds.prior.cpu().numpy())
        pred = run["targets"].astype(float)
        return {"counts": n, "log-posterior over z": lp - lp.mean(-1, keepdims=True),
                "predictive": pred, "log-predictive": np.log(pred)}

    def sites(run):
        """States at each site, (B, T, width), positions 0..T-1 (the last position predicts past the context)."""
        return {"embedding": run["hidden_states"][0][:, :T], "after attention": run["attention_states"][0][:, :T],
                "after MLP": run["hidden_states"][-1][:, :T]}

    data = {_name: (targets(_run), sites(_run)) for _name, _run in runs.items()}
    return (data,)


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
        ll = np.log(ds.P.double().cpu().numpy()).T                       # (K, n_z): rows tokens
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
    return


if __name__ == "__main__":
    app.run()
