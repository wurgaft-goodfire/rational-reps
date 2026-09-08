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
    # Inference: the same factorization over histories

    $z$ is hidden; the model reads $x_{1:t} \sim p(x \mid z)$. Bayes rows of sampled prefixes are
    stacked and factored; each state's representation is $E(s_t) = L_c(s_t,\cdot)\,D^{+}$.
    """)
    return


@app.cell
def _(mo):
    import os, sys
    import numpy as np
    from scipy.special import logsumexp

    ROOT = mo.notebook_dir().parent.parent    # src/notebooks -> <repo>
    os.chdir(ROOT); sys.path.insert(0, str(ROOT))
    from src.code.train import graph_overrides
    from src.code.analysis_utils import load_run
    from src.code.geometry import factor_logcond, predicted_inference_geometry
    from src.code import plot_utils
    import matplotlib.pyplot as plt

    from pathlib import Path
    FIGURES = Path(os.environ.get("CACHE_DIR", ROOT / "outputs")) / "figures"
    FIGURES.mkdir(parents=True, exist_ok=True)
    return (
        FIGURES,
        factor_logcond,
        graph_overrides,
        load_run,
        logsumexp,
        np,
        plot_utils,
        plt,
        predicted_inference_geometry,
    )


@app.cell
def _(graph_overrides, load_run, logsumexp, np, plot_utils, plt):
    # the two hidden-input worlds of the concepts notebook: a fine latent (13 temperatures, 28 times
    # of week) emitting coarse mentions (5 adjectives, 7 days), bipartite so the input is hidden
    # kernels calibrated so the log-posterior spans about 8 nats at the end of the context: every
    # latent's logit stays in the range the loss scores
    WORLDS = {"binary": {"structure": "feature_cube", "m": 2, "decay": 1.0, "scale": 1.25,
                         "paired": True, "all_bundles": True},
              "line": {"structure": "chain", "cooccurrence": "gaussian", "n": 13, "m": 5, "gamma": 0.03},
              "week": {"structure": "ring", "cooccurrence": "gaussian", "n": 28, "m": 7, "gamma": 0.8}}
    T, B = 12, 256                                                     # context length, streams per batch

    def overrides(spec, infer):
        return graph_overrides(spec) + [
            "runner=transformer", "sample_mode=sequence", f"++infer_latent={str(infer).lower()}",
            f"training_kwargs.context_length={T}", f"training_kwargs.batch_size={B}",
            "training_kwargs.train_steps=30000", "model_kwargs.model_config.hidden_size=64",
            "model_kwargs.model_config.num_hidden_layers=1"]

    def world(spec, infer, extra=()):
        """The saved run of a world: its dataset, the streams the evaluator scored (`inputs`) and
        the forward pass on them, hidden states in the readout's row space."""
        return load_run(overrides(spec, infer) + list(extra))

    def log_posteriors(ds, obs):
        """Normalized log p(z | x_{1:t}) for t = 0..L, in float64 log space. obs (B, L) -> (B, L+1, n_z)."""
        ll = np.asarray(ds.conditional)                                # (n_z, m): log p(x | z)
        lp0 = np.log(ds.prior.cpu().numpy())
        s = lp0 + np.concatenate([np.zeros((len(obs), 1, len(lp0))),
                                  np.cumsum(ll[:, obs].transpose(1, 2, 0), 1)], 1)
        return s - logsumexp(s, -1, keepdims=True)

    def frame(ax, title, xlim, ylim, pad=8):
        """Square panel; the axes are SVD modes, so their relative scale is not a shape claim."""
        ax.set_title(title, fontsize=13, pad=pad)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_box_aspect(1); plot_utils.FIG.clean(ax)

    def argmax_regions(Dcols, r_x, r_y, res=500):
        """Argmax-output index over the plane. Dcols (2, n) -> (res, res) image + extent."""
        _gx, _gy = np.linspace(*r_x, res), np.linspace(*r_y, res)
        C = np.stack(np.meshgrid(_gx, _gy), -1)                        # [i, j] = (x_j, y_i)
        return (C @ Dcols).argmax(-1), (*r_x, *r_y)

    def region_bg(ax, win, ext, rgb, fade=0.5, aspect=None):
        """Decision regions as a white-blended background image (rgb: (n, 3-or-4) region colors)."""
        ax.imshow(1 - fade * (1 - np.asarray(rgb)[win][..., :3]), origin="lower", extent=ext,
                  zorder=0, aspect=aspect)

    def latent_colors(name, n):
        """One color per latent: ordered on the line, cyclic on the week, categorical for words."""
        if name == "line":
            return plt.get_cmap(plot_utils.FIG.diverging_cmap)(np.linspace(0.05, 0.95, n))
        if name == "week":
            return plt.cm.twilight(np.arange(n) / n)
        return plt.cm.Set2(np.arange(n))
    return (
        B,
        T,
        WORLDS,
        argmax_regions,
        frame,
        latent_colors,
        log_posteriors,
        overrides,
        region_bg,
        world,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Predicted geometry from sampled histories (latent target)

    Top: states in the top two modes over the latent readout's cones, one stream highlighted.
    Bottom: the binary stream's log-odds per feature (each moves only on its own feature's tokens);
    mode coefficients against $t$ for line and week (mean ± sd over streams; a count mode has no spread).
    """)
    return


@app.cell
def _(WORLDS, factor_logcond, log_posteriors, np, world):
    hist = {}
    for _name, _spec in WORLDS.items():
        _run = world(_spec, infer=True)
        _ds, _obs = _run["dataset"], _run["inputs"][:, 1:]           # the evaluator's streams: (B, T) observations
        _LP = log_posteriors(_ds, _obs)                               # (B, T+1, n_z)
        _Lc, _E, _D, _S = factor_logcond(_LP.reshape(-1, _LP.shape[-1]))
        _E, _D, _S = _E[:, :3], _D[:3], _S[:3]                        # the top three modes
        hist[_name] = dict(run=_run, ds=_ds, obs=_obs, LP=_LP, E=_E.reshape(*_LP.shape[:2], -1), D=_D, S=_S)
        print(_name, "history spectrum / top:", np.round(_S / _S[0], 4))
    return (hist,)


@app.cell
def _(FIGURES, T, argmax_regions, frame, hist, latent_colors, np, plot_utils, plt, region_bg):
    fig_pred, _axs = plt.subplots(2, 3, figsize=(15, 9.5), height_ratios=[1.35, 1])
    for _j, (_name, _h) in enumerate(hist.items()):
        _E2, _D2 = _h["E"][..., :2], _h["D"][:2]                       # (B, T+1, 2), (2, n_z)
        _a = _axs[0, _j]
        _lo, _hi = _E2.reshape(-1, 2).min(0), _E2.reshape(-1, 2).max(0)  # the cloud's box, padded
        _pad = 0.12 * (_hi - _lo).max()
        _xr, _yr = (_lo[0] - _pad, _hi[0] + _pad), (_lo[1] - _pad, _hi[1] + _pad)
        _n_z = _D2.shape[1]
        region_bg(_a, *argmax_regions(_D2, _xr, _yr), latent_colors(_name, _n_z), fade=0.4, aspect="auto")
        _a.scatter(*_E2.reshape(-1, 2).T, s=4, color="0.45", alpha=0.25, lw=0, zorder=2)
        _b = 0                                                         # one stream
        _a.plot(*_E2[_b].T, color="0.2", lw=0.9, zorder=3)
        _a.scatter(*_E2[_b].T, c=np.arange(T + 1), cmap="viridis", s=26, zorder=4, edgecolor="none")
        _a.annotate("$t=0$", _E2[_b, 0], _E2[_b, 0] + _pad * np.array([-1.8, 0.3]), fontsize=11, color="0.3")
        frame(_a, f"{_name}: states over the latent cones", _xr, _yr)
        _a.set(xlabel="mode 1", ylabel="mode 2")

        _a = _axs[1, _j]
        _t = np.arange(T + 1)
        if _name == "binary":                                          # the chord: log-odds per feature, one stream
            _Z = _h["ds"].features; _LPb = _h["LP"][_b]                # (n_words, m), (T+1, n_words)
            from scipy.special import logsumexp as _lse
            for _i, (_col, _lab) in enumerate(zip([plot_utils.FIG.colors["blue"], plot_utils.FIG.colors["orange"]],
                                                  ["feature 1", "feature 2"])):
                _lo = _lse(_LPb[:, _Z[:, _i] == 1], -1) - _lse(_LPb[:, _Z[:, _i] == 0], -1)
                _a.plot(_t, _lo, "o-", color=_col, lw=1.6, ms=4, label=f"{_lab} log-odds")
                _hit = np.flatnonzero(_h["obs"][_b] % _Z.shape[1] == _i) + 1   # steps whose token is about feature i
                _a.scatter(_hit, _lo[_hit], s=70, facecolor="none", edgecolor=_col, lw=1.4, zorder=5)
            _a.axhline(0, color="0.7", lw=0.8)
            _a.set(xlabel="observations  $t$", ylabel="log-odds")
            _a.set_title("binary: one stream, circles mark that feature's tokens", fontsize=12)
            _a.legend(frameon=False); plot_utils.FIG.style_axes(_a)
            continue
        for _k, _col in enumerate([plot_utils.FIG.colors["blue"], plot_utils.FIG.colors["orange"]]):
            _c = _h["E"][..., _k]                                      # (B, T+1)
            _a.plot(_t, _c.mean(0), color=_col, lw=1.8, label=f"mode {_k + 1}")
            _a.fill_between(_t, _c.mean(0) - _c.std(0), _c.mean(0) + _c.std(0), color=_col, alpha=0.2, lw=0)
        _a.axhline(0, color="0.7", lw=0.8)
        _a.set(xlabel="observations  $t$", ylabel="coefficient (mean ± sd over streams)")
        _a.set_title(f"{_name}: mode coefficients", fontsize=13)
        _a.legend(frameon=False)
        plot_utils.FIG.style_axes(_a)
    fig_pred.tight_layout()
    plot_utils.save_figure(fig_pred, FIGURES / "inference_predicted.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Trained model vs prediction (latent target)

    A transformer per world reads the same streams and predicts $z$ at every position. Its hidden
    state, projected on the readout's row space, is fit affinely from the predicted representation
    $E(s_t)$; positions $0..T{-}1$ are the scored ones. Left: the model's top-2 readout plane with
    its argmax cones, one stream's hidden trajectory and the predicted trajectory mapped into the
    plane. Right: KL(Bayes $\|$ model) per position.
    """)
    return


@app.cell
def _(B, T, hist, logsumexp, np, predicted_inference_geometry):
    from src.code.analysis_utils import align_representations, cka, cka_null

    fits = {}
    for _name, _h in hist.items():
        _run, _ds = _h["run"], _h["ds"]
        _Hall = _run["hidden_states"][:, :, :T]                         # (layers+1, B, T, k): every layer, in the readout's row space
        _Hp = _Hall[-1]                                                 # (B, T, k): what the readout reads
        _LP = _h["LP"][:, :T]                                           # Bayes rows at scored positions
        _Ep = predicted_inference_geometry(_LP.reshape(-1, _LP.shape[-1]), _h["D"][:2]).reshape(B, T, 2)
        _fit, _r2 = align_representations(_Ep.reshape(-1, 2), _Hp.reshape(-1, _Hp.shape[-1]), "linear")
        _lg = _run["logits"][:, :T].astype(float)
        _lq = _lg - logsumexp(_lg, -1, keepdims=True)
        _lp0 = np.log(_ds.prior.cpu().numpy())
        fits[_name] = dict(H=_Hp, Hall=_Hall, E=_Ep, Hfit=_fit.reshape(B, T, -1), r2=_r2,
                           kl_prior=(np.exp(_LP) * (_LP - _lp0)).sum(-1).mean(0),
                           cka=cka(_Hp.reshape(-1, _Hp.shape[-1]), _Ep.reshape(-1, 2)),
                           null=cka_null(_Ep.reshape(-1, 2)),
                           W=_run["readout"],                            # readout rows in the k-space
                           kl=(np.exp(_LP) * (_LP - _lq)).sum(-1).mean(0))
        print(f"{_name}: readout rank {_run['readout'].shape[1]} | affine R2 predicted->hidden {_r2:.3f} | "
              f"CKA {fits[_name]['cka']:.3f} (null {fits[_name]['null']:.3f}) | KL(Bayes||model) mean {fits[_name]['kl'].mean():.4f}")
    return (fits,)


@app.cell
def _(FIGURES, T, argmax_regions, fits, frame, latent_colors, np, plot_utils, plt, region_bg):
    fig_learned, _axs = plt.subplots(2, 3, figsize=(15, 9.5), height_ratios=[1.35, 1])
    for _i, (_name, _f) in enumerate(fits.items()):
        _H2, _W2, _P2 = _f["H"][..., :2], _f["W"][:, :2], _f["Hfit"][..., :2]   # top-2 readout plane
        _a = _axs[0, _i]
        _pts = np.concatenate([_H2.reshape(-1, 2), _P2.reshape(-1, 2)])
        _lo, _hi = _pts.min(0), _pts.max(0); _pad = 0.12 * (_hi - _lo).max()
        _xr, _yr = (_lo[0] - _pad, _hi[0] + _pad), (_lo[1] - _pad, _hi[1] + _pad)
        _n_z = len(_W2)
        region_bg(_a, *argmax_regions(_W2.T, _xr, _yr), latent_colors(_name, _n_z), fade=0.4, aspect="auto")
        _a.scatter(*_H2.reshape(-1, 2).T, s=4, color="0.45", alpha=0.2, lw=0, zorder=2)
        _b = 0
        _a.plot(*_P2[_b].T, color="0.1", lw=1.6, zorder=3, label="predicted (affine-mapped)")
        _a.plot(*_H2[_b].T, color="0.5", lw=0.8, zorder=3)
        _a.scatter(*_H2[_b].T, c=np.arange(T), cmap="viridis", s=26, zorder=4, edgecolor="none", label="hidden state")
        _a.legend(frameon=False, loc="lower left", fontsize=10)
        frame(_a, f"{_name}: hidden states, model's readout plane", _xr, _yr)
        _a.set(xlabel="readout direction 1", ylabel="readout direction 2")

        _a = _axs[1, _i]
        _a.plot(np.arange(T), _f["kl_prior"], color="0.6", lw=1.2, ls="--", label="prior")
        _a.plot(np.arange(T), _f["kl"], color="0.15", lw=1.8, label="model")
        _a.set_yscale("log"); _a.legend(frameon=False)
        _a.set(xlabel="observations  $t$", ylabel=r"KL(Bayes $\|$ model)  [nats]", ylim=(0, None))
        _a.set_title(f"{_name}: affine $R^2$ = {_f['r2']:.3f}, CKA {_f['cka']:.2f} (null {_f['null']:.2f})", fontsize=12)
        plot_utils.FIG.style_axes(_a)
    fig_learned.tight_layout()
    plot_utils.save_figure(fig_learned, FIGURES / "inference_learned.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Additivity

    Left: $R^2$ of an affine fit from the observation histogram to the hidden state, per layer and
    in the readout space. Right: the fitted per-token increment against the predicted message
    (each token's centered log-likelihood column in $D$'s frame), CKA with its null.
    """)
    return


@app.cell
def _(B, FIGURES, T, fits, hist, np, plot_utils, plt, predicted_inference_geometry):
    from src.code.analysis_utils import align_representations as _align
    from src.code.analysis_utils import cka as _cka, cka_null as _cka_null

    fig_add, (_a, _a2) = plt.subplots(1, 2, figsize=(12, 4.2))
    _cols = [plot_utils.FIG.colors[c] for c in ("blue", "orange", "green")]
    for _j, (_name, _f) in enumerate(fits.items()):
        _h = hist[_name]; _obs, _m = _h["obs"], _h["ds"].n_outcomes
        _counts = np.cumsum(np.eye(_m)[_obs], axis=1)                   # (B, T, m) counts after t obs
        _counts = np.concatenate([np.zeros((B, 1, _m)), _counts[:, :-1]], 1).reshape(-1, _m)   # position t: x_1..x_t
        _r2 = [_align(_counts, _H.reshape(B * T, -1), "linear")[1] for _H in _f["Hall"]]
        _Hr = _f["H"].reshape(B * T, -1)
        _r2_read = _align(_counts, _Hr, "linear")[1]
        _a.plot(range(len(_r2)), _r2, "o-", color=_cols[_j], lw=1.6, label=f"{_name}: raw layers")
        _a.plot([len(_r2) - 1], [_r2_read], marker="*", ms=14, color=_cols[_j], ls="none",
                label=f"{_name}: readout space ({_r2_read:.3f})")
        # the fitted increment per token vs the predicted message (log-likelihood column in D's frame)
        _W = np.linalg.lstsq(np.c_[_counts, np.ones(len(_counts))], _Hr, rcond=None)[0][:_m]
        _msg = predicted_inference_geometry(np.asarray(_h["ds"].conditional).T, _h["D"][:2])
        _c, _n = _cka(_W, _msg), _cka_null(_msg)
        _a2.bar([_j - 0.18, _j + 0.18], [_c, _n], width=0.36, color=[_cols[_j], "0.8"])
        print(f"{_name}: histogram -> hidden R2 per layer {np.round(_r2, 3)} | readout space {_r2_read:.3f} | "
              f"increment vs message CKA {_c:.3f} (null {_n:.2f})")
    _a.set(xlabel="layer (0 = embeddings)", ylabel=r"$R^2$ histogram $\to$ hidden state", ylim=(0, 1.02))
    _a.legend(frameon=False, fontsize=9)
    _a2.set(xticks=range(len(fits)), xticklabels=list(fits), ylabel="CKA increment vs predicted message", ylim=(0, 1.02))
    _a2.set_title("colored: trained; grey: random-code null", fontsize=11)
    for _ax in (_a, _a2):
        plot_utils.FIG.style_axes(_ax)
    fig_add.tight_layout()
    plot_utils.save_figure(fig_add, FIGURES / "inference_additivity.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Next-token target

    Same worlds, the model predicts $x_{t+1}$. The history kernel is the centered log-predictive;
    its factorization gives the candidate representation. Top: affine $R^2$ and CKA of the hidden
    state against three candidates: the log-predictive code, the log-posterior code (section 1's
    frame), and the belief simplex. Bottom: static vs history-kernel spectra.
    """)
    return


@app.cell
def _(B, T, WORLDS, factor_logcond, hist, log_posteriors, logsumexp, np, predicted_inference_geometry, world):
    from src.code.analysis_utils import align_representations as _align
    from src.code.analysis_utils import cka as _cka

    fits_next = {}
    for _name, _spec in WORLDS.items():
        _run = world(_spec, infer=False)
        _ds, _obs = _run["dataset"], _run["inputs"][:, 1:]
        _LP = log_posteriors(_ds, _obs)                                 # (B, T+1, n_z)
        _ll = np.asarray(_ds.conditional)                               # (n_z, m)
        _LX = logsumexp(_LP[..., None] + _ll[None, None], axis=2)       # (B, T+1, m): log p(x | x_{1:t})
        _m = _LX.shape[-1]
        _Lc, _E, _D, _S = factor_logcond(_LX.reshape(-1, _m))
        _E, _D, _S = _E[:, :_m - 1], _D[:_m - 1], _S[:_m - 1]
        _S_static = factor_logcond(_ll)[3]
        _H = _run["hidden_states"][-1][:, :T].reshape(B * T, -1)       # the readout's input, in its row space
        _cands = {"log-predictive code": predicted_inference_geometry(_LX[:, :T].reshape(-1, _m), _D),
                  "log-posterior code": predicted_inference_geometry(_LP[:, :T].reshape(-1, _LP.shape[-1]), hist[_name]["D"][:2]),
                  "belief simplex": np.exp(_LP[:, :T]).reshape(B * T, -1)}
        _scores = {_k: (_align(_v, _H, "linear")[1], _cka(_H, _v)) for _k, _v in _cands.items()}
        _lg = _run["logits"][:, :T].astype(float)
        _lq = _lg - logsumexp(_lg, -1, keepdims=True)
        _kl = (np.exp(_LX[:, :T]) * (_LX[:, :T] - _lq)).sum(-1).mean()
        fits_next[_name] = dict(scores=_scores, S=_S / _S[0], S_static=_S_static / _S_static[0], kl=_kl)
        print(f"{_name}: KL(Bayes||model) {_kl:.4f} | history sv {np.round(_S / _S[0], 3)} | static sv {np.round(_S_static / _S_static[0], 3)}")
        for _k, (_r2, _c) in _scores.items():
            print(f"   {_k:22s} affine R2 {_r2:.3f}  CKA {_c:.3f}")
    return (fits_next,)


@app.cell
def _(FIGURES, fits_next, np, plot_utils, plt):
    fig_next, _axs = plt.subplots(2, 3, figsize=(15, 8))
    _cands = list(next(iter(fits_next.values()))["scores"])
    for _j, (_name, _f) in enumerate(fits_next.items()):
        _a = _axs[0, _j]
        _r2 = [_f["scores"][_k][0] for _k in _cands]; _ck = [_f["scores"][_k][1] for _k in _cands]
        _x = np.arange(len(_cands))
        _a.bar(_x - 0.18, _r2, width=0.36, color=plot_utils.FIG.colors["blue"], label="affine $R^2$")
        _a.bar(_x + 0.18, _ck, width=0.36, color=plot_utils.FIG.colors["orange"], label="CKA")
        _a.set(xticks=_x, xticklabels=_cands, ylim=(0, 1.15), yticks=np.linspace(0, 1, 6))
        _a.set_title(f"{_name}: KL(Bayes$\\|$model) {_f['kl']:.4f}", fontsize=12)
        _a.tick_params(axis="x", labelsize=9)
        if _j == 0:
            _a.legend(frameon=False, loc="upper left", ncol=2, fontsize=10)
        plot_utils.FIG.style_axes(_a)
        _a = _axs[1, _j]
        _k = np.arange(len(_f["S"]))
        _a.bar(_k - 0.18, np.r_[_f["S_static"], np.zeros(len(_k))][:len(_k)], width=0.36, color="0.5", label="static kernel")
        _a.bar(_k + 0.18, _f["S"], width=0.36, color=plot_utils.FIG.colors["green"], label="history kernel")
        _a.set(xlabel="mode", ylabel="singular value / top", xticks=_k, ylim=(0, 1.15), yticks=np.linspace(0, 1, 6))
        if _j == 0:
            _a.legend(frameon=False, loc="upper right", fontsize=10)
        plot_utils.FIG.style_axes(_a)
    fig_next.tight_layout()
    plot_utils.save_figure(fig_next, FIGURES / "inference_next_token.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Hypothesis spaces

    Bayesian generalization on a line and a grid (Tenenbaum & Griffiths 2001), a taxonomy
    (Xu & Tenenbaum 2007), and key finding (Krumhansl & Schmuckler). The first three predict the
    next example (the belief over hypotheses is near full rank); keys infer the key. Top: states in
    the top two modes over the readout's cones, one stream highlighted. Bottom: static vs
    history-kernel spectra.
    """)
    return


@app.cell
def _(factor_logcond, log_posteriors, logsumexp, np, world):
    HYP = {"intervals": ({"structure": "intervals", "n": 10}, False),
           "rectangles": ({"structure": "rectangles", "n": 5}, False),    # 225 hypotheses over 25 cells
           "subtrees": ({"structure": "subtrees", "n": 3, "k": 3}, False),
           "keys": ({"structure": "keys", "beta": 1.5}, True)}
    HYP_EXTRA = {"rectangles": ["model_kwargs.model_config.num_hidden_layers=2",       # two axes to bound
                                "training_kwargs.train_steps=60000"]}

    def target_rows(ds, obs, infer):
        """Bayes rows per prefix: log-posterior over hypotheses, or log-predictive over items."""
        _LP = log_posteriors(ds, obs)
        if infer:
            return _LP
        return logsumexp(_LP[..., None] + np.asarray(ds.conditional)[None, None], axis=2)

    hyp = {}
    for _name, (_spec, _infer) in HYP.items():
        _run = world(_spec, _infer, HYP_EXTRA.get(_name, []))
        _ds, _obs = _run["dataset"], _run["inputs"][:, 1:]
        _R = target_rows(_ds, _obs, _infer)                            # (B, T+1, C)
        _S_all = factor_logcond(_R.reshape(-1, _R.shape[-1]))[3]
        _rank = int((_S_all > 1e-8 * _S_all[0]).sum())
        _Lc, _E, _D, _S = factor_logcond(_R.reshape(-1, _R.shape[-1]))
        _E, _D, _S = _E[:, :_rank], _D[:_rank], _S[:_rank]
        _S_static = factor_logcond(np.asarray(_ds.conditional))[3]
        hyp[_name] = dict(run=_run, ds=_ds, obs=_obs, R=_R, E=_E.reshape(*_R.shape[:2], -1), D=_D, S=_S,
                          S_static=_S_static, infer=_infer)
        print(f"{_name}: rows {_R.shape[-1]} | rank {_rank} | history sv {np.round(_S[:6] / _S[0], 3)} | "
              f"static sv {np.round(_S_static[:6] / _S_static[0], 3)}")
    return HYP, HYP_EXTRA, hyp, target_rows


@app.cell
def _(FIGURES, T, argmax_regions, frame, hyp, np, plot_utils, plt, region_bg):
    fig_hyp, _axs = plt.subplots(2, 4, figsize=(19, 9.5), height_ratios=[1.35, 1])
    for _j, (_name, _h) in enumerate(hyp.items()):
        _E2, _D2 = _h["E"][..., :2], _h["D"][:2]
        _a = _axs[0, _j]
        _flat = _E2.reshape(-1, 2)
        _lo, _hi = _flat.min(0), _flat.max(0); _pad = 0.12 * (_hi - _lo).max()
        _xr, _yr = (_lo[0] - _pad, _hi[0] + _pad), (_lo[1] - _pad, _hi[1] + _pad)
        _n_c = _D2.shape[1]
        region_bg(_a, *argmax_regions(_D2, _xr, _yr), plt.cm.twilight(np.arange(_n_c) / _n_c) if _name == "keys"
                  else plt.get_cmap(plot_utils.FIG.diverging_cmap)(np.linspace(0.05, 0.95, _n_c)), fade=0.35, aspect="auto")
        _a.scatter(*_flat.T, c=np.tile(np.arange(T + 1), len(_E2)), cmap="viridis", s=5, alpha=0.25, lw=0, zorder=2)
        _b = 0
        _a.plot(*_E2[_b].T, color="0.15", lw=1.0, zorder=3)
        _a.scatter(*_E2[_b].T, c=np.arange(T + 1), cmap="viridis", s=28, zorder=4, edgecolor="k", lw=0.4)
        frame(_a, f"{_name}: {'posterior' if _h['infer'] else 'predictive'} states", _xr, _yr)
        _a.set(xlabel="mode 1", ylabel="mode 2")
        _a = _axs[1, _j]
        _k = np.arange(min(10, len(_h["S"])))
        _st = np.r_[_h["S_static"] / _h["S_static"][0], np.zeros(len(_k))][:len(_k)]
        _a.bar(_k - 0.18, _st, width=0.36, color="0.5", label="static kernel")
        _a.bar(_k + 0.18, (_h["S"] / _h["S"][0])[:len(_k)], width=0.36, color=plot_utils.FIG.colors["green"], label="history kernel")
        _a.set(xlabel="mode", ylabel="singular value / top", xticks=_k, ylim=(0, 1.15), yticks=np.linspace(0, 1, 6))
        if _j == 0:
            _a.legend(frameon=False, fontsize=10)
        plot_utils.FIG.style_axes(_a)
    fig_hyp.tight_layout()
    plot_utils.save_figure(fig_hyp, FIGURES / "inference_hypotheses_predicted.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5b. Trained models

    One transformer per world. KL(Bayes $\|$ model) per position with the prior baseline; the
    title gives the affine $R^2$ and CKA of the hidden state against the predicted representation.
    """)
    return


@app.cell
def _(B, FIGURES, T, hyp, logsumexp, np, plot_utils, plt, predicted_inference_geometry):
    from src.code.analysis_utils import align_representations as _align
    from src.code.analysis_utils import cka as _cka, cka_null as _cka_null

    fits_hyp = {}
    fig_hyp_learned, _axs = plt.subplots(1, 4, figsize=(19, 4.2))
    for _j, (_name, _h) in enumerate(hyp.items()):
        _run, _ds = _h["run"], _h["ds"]
        _H = _run["hidden_states"][-1][:, :T].reshape(B * T, -1)       # the readout's input, in its row space
        _R = _h["R"][:, :T]
        _Ep = predicted_inference_geometry(_R.reshape(-1, _R.shape[-1]), _h["D"])
        _r2 = _align(_Ep, _H, "linear")[1]; _c, _n = _cka(_H, _Ep), _cka_null(_Ep)
        _lg = _run["logits"][:, :T].astype(float)
        _lq = _lg - logsumexp(_lg, -1, keepdims=True)
        _kl = (np.exp(_R) * (_R - _lq)).sum(-1).mean(0)
        _lp0 = np.log(_ds.prior.cpu().numpy())
        _base = _lp0 if _h["infer"] else logsumexp(_lp0[:, None] + np.asarray(_ds.conditional), axis=0)
        _kl0 = (np.exp(_R) * (_R - _base)).sum(-1).mean(0)
        fits_hyp[_name] = dict(r2=_r2, cka=_c, null=_n, kl=_kl, ds=_ds)
        print(f"{_name}: readout rank {_run['readout'].shape[1]} | affine R2 {_r2:.3f} | CKA {_c:.3f} (null {_n:.3f}) | KL mean {_kl.mean():.4f}")
        _a = _axs[_j]
        _a.plot(np.arange(T), _kl0, color="0.6", lw=1.2, ls="--", label="prior")
        _a.plot(np.arange(T), _kl, color="0.15", lw=1.8, label="model")
        _a.set_yscale("log"); _a.legend(frameon=False)
        _a.set(xlabel="observations  $t$", ylabel=r"KL(Bayes $\|$ ·)  [nats]")
        _a.set_title(f"{_name}: affine $R^2$ {_r2:.3f}, CKA {_c:.2f} (null {_n:.2f})", fontsize=12)
        plot_utils.FIG.style_axes(_a)
    fig_hyp_learned.tight_layout()
    plot_utils.save_figure(fig_hyp_learned, FIGURES / "inference_hypotheses_learned.png")
    return (fits_hyp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5c. The size principle

    Examples cycling through a fixed span: the predictive over items after $n$ = 2, 4, 8, 12 of
    them, model (solid) against Bayes (dashed). More examples with the same span should tighten
    generalization to the span.
    """)
    return


@app.cell
def _(FIGURES, HYP, HYP_EXTRA, hyp, logsumexp, np, overrides, plot_utils, plt, target_rows):
    # a handcrafted probe stream, which the evaluator's fixed batch cannot hold: the one place the
    # live model is read (train_inline loads the saved run)
    import torch
    from src.code.train import train_inline

    fig_size, _axs = plt.subplots(1, 2, figsize=(13, 4.4))
    _ns = [2, 4, 8, 12]
    _cols = plt.cm.viridis(np.linspace(0.1, 0.9, len(_ns)))
    _spans = {"intervals": [3, 6, 4, 5], "rectangles": [(2, 2), (4, 4), (2, 4), (4, 2), (3, 3), (2, 3), (4, 3), (3, 2), (3, 4)]}
    for _a, _name, _pair in zip(_axs, ["intervals", "rectangles"], [(3, 6), ((2, 2), (4, 4))]):
        _spec, _infer = HYP[_name]
        _model, _ds, *_ = train_inline(overrides(_spec, _infer) + HYP_EXTRA.get(_name, []))
        _n = hyp[_name]["ds"].info["hypotheses"].max()               # grid side / line length
        _tok = lambda it: it - 1 if _name == "intervals" else (it[0] - 1) * _n + (it[1] - 1)
        _cyc = _spans[_name]
        _stream = np.array([_tok(_cyc[i % len(_cyc)]) for i in range(12)])   # the span's items, in turn
        _ids = torch.tensor([[int(_ds.start_token), *_stream]], device=_ds.device)
        with torch.no_grad():
            _lg = _model(_ids)["logits"][0][..., _ds.label_tokens].double().cpu().numpy()
        _lq = _lg - logsumexp(_lg, -1, keepdims=True)
        _bayes = target_rows(_ds, _stream[None], infer=False)[0]      # (13, items), row n = after n examples
        for _c, _k in zip(_cols, _ns):
            if _name == "intervals":
                _x = np.arange(1, _n + 1)
                _a.plot(_x, np.exp(_bayes[_k]), ls="--", color=_c, lw=1.4)
                _a.plot(_x, np.exp(_lq[_k]), color=_c, lw=1.8, label=f"$n$ = {_k}")
            else:                                                     # a diagonal slice of the grid
                _diag = [(i - 1) * _n + (i - 1) for i in range(1, _n + 1)]
                _a.plot(np.arange(1, _n + 1), np.exp(_bayes[_k])[_diag], ls="--", color=_c, lw=1.4)
                _a.plot(np.arange(1, _n + 1), np.exp(_lq[_k])[_diag], color=_c, lw=1.8, label=f"$n$ = {_k}")
        for _e in (_pair if _name == "intervals" else [p[0] for p in _pair]):
            _a.axvline(_e, color="0.8", lw=0.8, zorder=0)
        _a.set(xlabel="item" if _name == "intervals" else "diagonal cell", ylabel=r"$p(y \mid X)$",
               xticks=np.arange(1, _n + 1))
        _a.set_title(f"{_name}: span {_pair}; solid model, dashed Bayes", fontsize=12)
        _a.legend(frameon=False)
        plot_utils.FIG.style_axes(_a)
    fig_size.tight_layout()
    plot_utils.save_figure(fig_size, FIGURES / "inference_size_principle.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5d. Key geometry

    The 24 keys' decoder vectors in the top two modes of the key-finding kernel, joined in
    circle-of-fifths order; majors and their relative minors should interleave.
    """)
    return


@app.cell
def _(FIGURES, hyp, np, plot_utils, plt):
    _h = hyp["keys"]
    _keys = _h["ds"].info["keys"]
    _W = _h["D"][:2].T                                                 # (24, 2) decoder vectors
    _names = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
    fig_keys, _a = plt.subplots(figsize=(6.5, 6.5))
    for _mode, _col, _mk in (("major", plot_utils.FIG.colors["blue"], "o"), ("minor", plot_utils.FIG.colors["orange"], "s")):
        _idx = [i for i, (t, m) in enumerate(_keys) if m == _mode]
        _fifths = sorted(_idx, key=lambda i: (_keys[i][0] * 7) % 12)  # circle-of-fifths order
        _a.plot(*_W[_fifths + _fifths[:1]].T, color=_col, lw=1.0, alpha=0.6)
        _a.scatter(*_W[_idx].T, color=_col, marker=_mk, s=70, label=_mode, zorder=3)
        for _i in _idx:
            _t = _keys[_i][0]
            _a.annotate(_names[_t] + ("" if _mode == "major" else "m"), _W[_i], fontsize=9,
                        xytext=(4, 4), textcoords="offset points", color="0.2")
    _a.legend(frameon=False)
    _a.set_aspect("equal"); plot_utils.FIG.clean(_a)
    _a.set_title("key finding: decoder vectors, joined by fifths", fontsize=13)
    plot_utils.save_figure(fig_keys, FIGURES / "inference_keys.png")
    return


if __name__ == "__main__":
    app.run()
