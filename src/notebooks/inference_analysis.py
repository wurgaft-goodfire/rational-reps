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
    T, B_TRAIN = 12, 256                                               # context length, training batch

    def overrides(spec, infer):
        return graph_overrides(spec) + [
            "runner=transformer", "sample_mode=sequence", f"++infer_latent={str(infer).lower()}",
            f"training_kwargs.context_length={T}", f"training_kwargs.batch_size={B_TRAIN}",
            "training_kwargs.train_steps=30000", "model_kwargs.model_config.hidden_size=64",
            "model_kwargs.model_config.num_hidden_layers=1"]

    def world(spec, infer, extra=()):
        """The saved run of a world: its dataset, the streams the evaluator scored (`inputs`) and
        the forward pass on them, hidden states in the readout's row space. When the latent is the
        target its labels name it: ``latent`` (B,) is the true z behind each stream."""
        run = load_run(overrides(spec, infer) + list(extra))
        if infer:
            run["latent"] = run["labels"][:, 1] - run["dataset"].n_outcomes - 1
        return run

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
        T,
        WORLDS,
        argmax_regions,
        frame,
        latent_colors,
        log_posteriors,
        region_bg,
        world,
    )


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Latent target: predicted and trained states

    Top: the Bayes rows of sampled prefixes, factored; states in the top two modes. Bottom: the
    trained model's hidden states in their own top-2 plane, rotated (no stretch) onto the prediction,
    with the readout rows $w_z$ projected the same way. One stream is highlighted in both rows; the
    title gives the affine $R^2$ and CKA of hidden state against prediction.

    **Rays.** Streams are grouped by the true latent behind them (the run's labels); each ray runs
    from the origin through the group's mean state at the last scored position $t = T{-}1$: the
    direction a $z$-stream heads on average. Cones are the readout's argmax regions.
    """)
    return


@app.cell
def _(T, hist, logsumexp, np, predicted_inference_geometry):
    from src.code.analysis_utils import align_representations, cka, cka_null

    fits = {}
    for _name, _h in hist.items():
        _run, _ds = _h["run"], _h["ds"]
        _Hall = _run["hidden_states"][:, :, :T]                         # (layers+1, B, T, k): every layer, in the readout's row space
        _Hp = _Hall[-1]                                                 # (B, T, k): what the readout reads
        _LP = _h["LP"][:, :T]                                           # Bayes rows at scored positions
        _Ep = predicted_inference_geometry(_LP.reshape(-1, _LP.shape[-1]), _h["D"][:2]).reshape(-1, T, 2)
        _fit, _r2 = align_representations(_Ep.reshape(-1, 2), _Hp.reshape(-1, _Hp.shape[-1]), "linear")
        _lg = _run["logits"][:, :T].astype(float)
        _lq = _lg - logsumexp(_lg, -1, keepdims=True)
        _lp0 = np.log(_ds.prior.cpu().numpy())
        fits[_name] = dict(H=_Hp, Hall=_Hall, E=_Ep, Hfit=_fit.reshape(-1, T, _fit.shape[-1]), r2=_r2,
                           kl_prior=(np.exp(_LP) * (_LP - _lp0)).sum(-1).mean(0),
                           cka=cka(_Hp.reshape(-1, _Hp.shape[-1]), _Ep.reshape(-1, 2)),
                           null=cka_null(_Ep.reshape(-1, 2)),
                           W=_run["readout"],                            # readout rows in the k-space
                           kl=(np.exp(_LP) * (_LP - _lq)).sum(-1).mean(0))
        print(f"{_name}: readout rank {_run['readout'].shape[1]} | affine R2 predicted->hidden {_r2:.3f} | "
              f"CKA {fits[_name]['cka']:.3f} (null {fits[_name]['null']:.3f}) | KL(Bayes||model) mean {fits[_name]['kl'].mean():.4f}")
    return (fits,)


@app.cell
def _(T, fits, hist, logsumexp, np):
    def rotate_onto(X, Y):
        """Orthogonal 2x2 map (rotation or reflection) minimizing |X R - Y|: a gauge, not a fit."""
        _U, _, _Vt = np.linalg.svd(X.T @ Y)
        return _U @ _Vt

    def expected_rows(ds, ts):
        """Expected Bayes log-posterior rows for each true latent after t observations, t in ts:
        (n_z, len(ts), n_z), from log p0(z') - t KL(z || z')."""
        _ll = np.asarray(ds.conditional); _lp0 = np.log(ds.prior.cpu().numpy())
        _K = np.einsum("zm,zwm->zw", np.exp(_ll), _ll[:, None, :] - _ll[None, :, :])   # KL(z || z')
        _rows = _lp0 - np.asarray(ts)[None, :, None] * _K[:, None, :]
        return _rows - logsumexp(_rows, -1, keepdims=True)

    def latent_means(pts, z, n):
        """Mean state per true latent at every position: pts (B, T, d), z (B,) -> (n, T, d)."""
        return np.stack([pts[z == _k].mean(0) for _k in range(n)])

    planes = {}
    for _name, _h in hist.items():
        _f = fits[_name]; _n = _h["D"].shape[1]; _z = _h["run"]["latent"]
        _E2 = _h["E"][:, :T, :2]                                       # predicted states at the scored positions
        _H = _f["H"].reshape(-1, _f["H"].shape[-1])                    # (B T, k) hidden states, readout row space
        _P = np.linalg.svd(_H, full_matrices=False)[2][:2]             # top-2 plane through the origin
        _Q = rotate_onto(_H @ _P.T, _E2.reshape(-1, 2)).T @ _P          # rotated plane basis (2, k)
        _X = (_H @ _Q.T).reshape(-1, T, 2)
        _rows = expected_rows(_h["ds"], [T - 1])[:, 0]                  # Bayes' expected row per latent at the last scored position
        _pz = np.exp(_rows[np.arange(_n), np.arange(_n)])              # expected posterior on the true latent
        planes[_name] = dict(pred=dict(pts=_E2, cones=_h["D"][:2], mean=latent_means(_E2, _z, _n)),
                             trained=dict(pts=_X, cones=_Q @ _f["W"].T, mean=latent_means(_X, _z, _n)),
                             count=np.bincount(_z, minlength=_n))
        print(f"{_name}: streams per latent {planes[_name]['count'].min()}-{planes[_name]['count'].max()} | "
              f"expected p(z | {T - 1} obs) median {np.median(_pz):.3f}")
    return (planes,)


@app.cell
def _(
    FIGURES,
    argmax_regions,
    fits,
    frame,
    hist,
    latent_colors,
    np,
    planes,
    plot_utils,
    plt,
    region_bg,
):
    def _state_panel(ax, title, pl, colors, b, fade=0.25):
        """One plane: cones, sampled states, a ray per true latent through its mean end state, one highlighted stream."""
        _flat = pl["pts"].reshape(-1, 2)
        _lo, _hi = _flat.min(0), _flat.max(0); _pad = 0.12 * (_hi - _lo).max()
        _xr, _yr = (_lo[0] - _pad, _hi[0] + _pad), (_lo[1] - _pad, _hi[1] + _pad)
        region_bg(ax, *argmax_regions(pl["cones"], _xr, _yr), colors, fade=fade, aspect="auto")
        ax.scatter(*pl["pts"].reshape(-1, 2).T, s=4, color="0.45", alpha=0.25, lw=0, zorder=2)
        for _z, (_dx, _dy) in enumerate(pl["mean"][:, -1]):             # ray through the mean end state of each true latent
            _s = min(((_xr[1] if _dx > 0 else _xr[0]) / _dx) if _dx else np.inf,
                     ((_yr[1] if _dy > 0 else _yr[0]) / _dy) if _dy else np.inf)
            ax.plot([0, _s * _dx], [0, _s * _dy], color=np.asarray(colors[_z][:3]) * 0.8,
                    lw=0.8 if len(colors) <= 12 else 0.5, alpha=0.6, zorder=3)
        ax.plot(*pl["pts"][b].T, color="0.2", lw=0.9, zorder=4)
        ax.scatter(*pl["pts"][b].T, c=np.arange(pl["pts"].shape[1]), cmap="viridis", s=26, zorder=5, edgecolor="none")
        frame(ax, title, _xr, _yr)

    fig_latent, _axs = plt.subplots(2, 3, figsize=(15, 10))
    _b = 0
    for _j, (_name, _pl) in enumerate(planes.items()):
        _cols = latent_colors(_name, hist[_name]["D"].shape[1])
        _state_panel(_axs[0, _j], f"{_name}: predicted", _pl["pred"], _cols, _b)
        _axs[0, _j].set(xlabel="mode 1", ylabel="mode 2")
        _f = fits[_name]
        _state_panel(_axs[1, _j], f"{_name}: trained, affine $R^2$ {_f['r2']:.3f}, CKA {_f['cka']:.2f}", _pl["trained"], _cols, _b)
        _axs[1, _j].set(xlabel="hidden PC 1 (rotated)", ylabel="hidden PC 2 (rotated)")
    fig_latent.tight_layout()
    plot_utils.save_figure(fig_latent, FIGURES / "inference_latent.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Next-token target

    Same worlds, the model predicts $x_{t+1}$. The history kernel is the centered log-predictive;
    its factorization gives the candidate representation. Printed: affine $R^2$ and CKA of the
    hidden state against three candidates (the log-predictive code, the log-posterior code in
    section 1's frame, and the belief simplex), and the static vs history-kernel spectra.
    """)
    return


@app.cell
def _(
    T,
    WORLDS,
    factor_logcond,
    hist,
    log_posteriors,
    logsumexp,
    np,
    predicted_inference_geometry,
    world,
):
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
        _H = _run["hidden_states"][-1][:, :T].reshape(-1, _run["hidden_states"].shape[-1])       # the readout's input, in its row space
        _cands = {"log-predictive code": predicted_inference_geometry(_LX[:, :T].reshape(-1, _m), _D),
                  "log-posterior code": predicted_inference_geometry(_LP[:, :T].reshape(-1, _LP.shape[-1]), hist[_name]["D"][:2]),
                  "belief simplex": np.exp(_LP[:, :T]).reshape(-1, _LP.shape[-1])}
        _scores = {_k: (_align(_v, _H, "linear")[1], _cka(_H, _v)) for _k, _v in _cands.items()}
        _lg = _run["logits"][:, :T].astype(float)
        _lq = _lg - logsumexp(_lg, -1, keepdims=True)
        _kl = (np.exp(_LX[:, :T]) * (_LX[:, :T] - _lq)).sum(-1).mean()
        fits_next[_name] = dict(scores=_scores, S=_S / _S[0], S_static=_S_static / _S_static[0], kl=_kl)
        print(f"{_name}: KL(Bayes||model) {_kl:.4f} | history sv {np.round(_S / _S[0], 3)} | static sv {np.round(_S_static / _S_static[0], 3)}")
        for _k, (_r2, _c) in _scores.items():
            print(f"   {_k:22s} affine R2 {_r2:.3f}  CKA {_c:.3f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Hypothesis spaces

    Bayesian generalization on a line and a grid (Tenenbaum & Griffiths 2001), a taxonomy
    (Xu & Tenenbaum 2007), and key finding (Krumhansl & Schmuckler). The first three predict the
    next example (the belief over hypotheses is near full rank); keys infer the key. States in the
    top two modes over the readout's cones, one stream highlighted.
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
    return (hyp,)


@app.cell
def _(FIGURES, T, argmax_regions, frame, hyp, np, plot_utils, plt, region_bg):
    fig_hyp, _axs = plt.subplots(1, 4, figsize=(19, 5.4))
    for _j, (_name, _h) in enumerate(hyp.items()):
        _E2, _D2 = _h["E"][..., :2], _h["D"][:2]
        _a = _axs[_j]
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
    fig_hyp.tight_layout()
    plot_utils.save_figure(fig_hyp, FIGURES / "inference_hypotheses_predicted.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Trained models

    One transformer per world. Printed: the affine $R^2$ and CKA of the hidden state against the
    predicted representation, with the CKA null, and the mean KL(Bayes $\|$ model).
    """)
    return


@app.cell
def _(T, hyp, logsumexp, np, predicted_inference_geometry):
    from src.code.analysis_utils import align_representations as _align
    from src.code.analysis_utils import cka as _cka, cka_null as _cka_null

    fits_hyp = {}
    for _name, _h in hyp.items():
        _run, _ds = _h["run"], _h["ds"]
        _H = _run["hidden_states"][-1][:, :T].reshape(-1, _run["hidden_states"].shape[-1])       # the readout's input, in its row space
        _R = _h["R"][:, :T]
        _Ep = predicted_inference_geometry(_R.reshape(-1, _R.shape[-1]), _h["D"])
        _r2 = _align(_Ep, _H, "linear")[1]; _c, _n = _cka(_H, _Ep), _cka_null(_Ep)
        _lg = _run["logits"][:, :T].astype(float)
        _lq = _lg - logsumexp(_lg, -1, keepdims=True)
        _kl = (np.exp(_R) * (_R - _lq)).sum(-1).mean(0)
        fits_hyp[_name] = dict(r2=_r2, cka=_c, null=_n, kl=_kl, ds=_ds)
        print(f"{_name}: readout rank {_run['readout'].shape[1]} | affine R2 {_r2:.3f} | CKA {_c:.3f} (null {_n:.3f}) | KL mean {_kl.mean():.4f}")
    return


if __name__ == "__main__":
    app.run()
