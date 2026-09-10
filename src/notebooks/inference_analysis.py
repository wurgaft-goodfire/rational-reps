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
    import io, os, sys
    import numpy as np
    import torch
    from scipy.special import logsumexp

    ROOT = mo.notebook_dir().parent.parent    # src/notebooks -> <repo>
    os.chdir(ROOT); sys.path.insert(0, str(ROOT))
    from src.code.train import graph_overrides
    from src.code.analysis_utils import load_run, rigid_map
    from src.code.geometry import factor_logcond, predicted_inference_geometry
    from src.code import plot_utils
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.colors import to_rgb

    from pathlib import Path
    FIGURES = Path(os.environ.get("CACHE_DIR", ROOT / "outputs")) / "figures"
    FIGURES.mkdir(parents=True, exist_ok=True)
    return (
        FIGURES,
        factor_logcond,
        graph_overrides,
        io,
        load_run,
        logsumexp,
        np,
        pe,
        plot_utils,
        plt,
        predicted_inference_geometry,
        rigid_map,
        to_rgb,
        torch,
    )


@app.cell
def _(graph_overrides, load_run, logsumexp, np, plot_utils, plt, to_rgb):
    # the worlds of the concepts notebook, hidden: a fine latent (13 temperatures, 28 times of week)
    # emitting coarse mentions (5 adjectives, 7 days), bipartite so the input is hidden. Each kernel is
    # the concepts one divided by a temperature (gamma and scale are exact temperatures: same centred
    # log-conditional up to a factor, so the same geometry), cooled so that a whole context of T
    # observations keeps every latent's logit in the range the loss scores (line: median end-of-context
    # span 4 nats, 90th percentile 8; at gamma 0.03 the far latents fell to -13 and the trained states
    # bent away from the affine image of the Bayes code there)
    WORLDS = {"binary": {"structure": "feature_cube", "m": 2, "decay": 1.0, "scale": 1.25,
                         "paired": True, "all_bundles": True},
              "line": {"structure": "chain", "cooccurrence": "gaussian", "n": 13, "m": 5, "gamma": 0.02},
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
        the forward pass on them, hidden states in the readout's row space; ``latent`` (B,) is the
        true z behind each stream."""
        run = load_run(overrides(spec, infer) + list(extra))
        run["latent"] = run["latents"][:, 0]
        return run

    def log_posteriors(ds, obs):
        """Normalized log p(z | x_{1:t}) for t = 0..L, in float64 log space. obs (B, L) -> (B, L+1, n_z)."""
        ll = np.asarray(ds.conditional)                                # (n_z, m): log p(x | z)
        lp0 = np.log(ds.prior.cpu().numpy())
        s = lp0 + np.concatenate([np.zeros((len(obs), 1, len(lp0))),
                                  np.cumsum(ll[:, obs].transpose(1, 2, 0), 1)], 1)
        return s - logsumexp(s, -1, keepdims=True)

    def frame(ax, title, xlim, ylim, pad=8, fontsize=13):
        """Square panel; the axes are SVD modes, so their relative scale is not a shape claim."""
        ax.set_title(title, fontsize=fontsize, pad=pad)
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
        """One color per latent: ordered on the line, cyclic on the week; for the words, royal as hue
        (orange no, blue yes, the concepts figure's pair) and female as lightness (dark no, light yes),
        in cube_codes order [man, king, woman, queen]."""
        if name == "line":
            return plt.get_cmap(plot_utils.FIG.diverging_cmap)(np.linspace(0.05, 0.95, n))
        if name == "week":
            return plt.cm.twilight(np.arange(n) / n)
        _no, _yes = (np.array(to_rgb(plot_utils.FIG.colors[_k])) for _k in ("orange", "blue"))
        return np.array([0.8 * _no, 0.8 * _yes, 1 - 0.55 * (1 - _no), 1 - 0.55 * (1 - _yes)])

    return T, WORLDS, frame, latent_colors, log_posteriors, world


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
    trained model's hidden states in their own top-2 plane, rotated (no stretch) onto the prediction.
    One stream is highlighted in both rows; the title gives the affine $R^2$ and CKA of hidden state
    against prediction.

    **Colour.** Each state is the belief-weighted mix of the latent colours: the Bayes posterior on
    the predicted side, the model's own readout $\mathrm{softmax}(h \cdot W)$ on the trained side, so
    matching colours across the pair are a check in themselves. Certainty is not painted separately;
    it is saturation, growing with distance from the origin, where every world sits at the prior's mix.
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
    return align_representations, cka, fits


@app.cell
def _(
    T,
    factor_logcond,
    fits,
    frame,
    hist,
    io,
    np,
    pe,
    plot_utils,
    plt,
    rigid_map,
):
    def latent_means(pts, z, n):
        """Mean state per true latent at every position: pts (B, T, d), z (B,) fixed or (B, T) moving -> (n, T, d)."""
        _Z = np.broadcast_to(np.asarray(z).reshape(len(z), -1), pts.shape[:2])
        return np.stack([np.stack([pts[:, _t][_Z[:, _t] == _k].mean(0) for _t in range(pts.shape[1])]) for _k in range(n)])

    def canonical_gauge(name, Dc, tokens=False):
        """The 2x2 rotation/reflection that puts a world's plane in the concepts notebook's position, from
        its readout directions Dc (2, n), one per latent (or per token, ``tokens``): binary, royal on +x
        and female up; line, the middle one up and cold to hot left to right; week and ring, the first up
        and the rest clockwise; any other world keeps its SVD frame. The loss is blind to these isometries."""
        _pairs = {"binary": (([0], [2]), ([1], [3])) if tokens else (([1, 3], [0, 2]), ([2, 3], [0, 1])),   # (royal, female) as (yes, no) index sets
                  "binary clock": (([1, 2], [0, 3]), ([2, 3], [0, 1]))}      # tokens [1+, 2+, 1-, 2-]; latents [man, king, woman, queen]; cycle [man, king, queen, woman]
        if name not in (*_pairs, "line", "week", "ring"):
            return np.eye(2)
        if name in _pairs:
            _royal, _female = _pairs[name]
            v, up = Dc[:, _royal[0]].sum(1) - Dc[:, _royal[1]].sum(1), False
        elif name == "line":
            v, up = Dc[:, Dc.shape[1] // 2], True
        else:
            v, up = Dc[:, 0], True
        _a = -np.arctan2(v[1], v[0]) + (np.pi / 2 if up else 0.0)
        R = np.array([[np.cos(_a), -np.sin(_a)], [np.sin(_a), np.cos(_a)]])
        Dr = R @ Dc
        flip_x = (name == "line" and Dr[0, -1] < Dr[0, 0]) or \
                 (name in ("week", "ring") and Dr[0, 0] * Dr[1, 1] - Dr[1, 0] * Dr[0, 1] > 0)
        flip_y = name in _pairs and (Dr[:, _female[0]].sum(1) - Dr[:, _female[1]].sum(1))[1] < 0
        return np.diag([-1.0 if flip_x else 1.0, -1.0 if flip_y else 1.0]) @ R

    def planes_for(name, E, D, H, W, z, tokens=False):
        """The two planes of one world, in the canonical gauge. Predicted: the code's top-2 modes. Trained:
        the hidden states rotated (not stretched) onto the predicted code in the full space, then the same two modes.
        Each side carries its states (B, T, 2),
        the readout columns in the plane (2, C), the mean state per true latent (n, T, 2) and the
        representative rows for its Gram: the mean path per latent, ordered (latent, t)."""
        _n = int(z.max()) + 1
        _G = canonical_gauge(name, D[:2], tokens)                                  # the same isometry on both sides
        _E2 = E[..., :2] @ _G.T
        _Hf = H.reshape(-1, H.shape[-1])
        _R = rigid_map(_Hf, E.reshape(-1, E.shape[-1]))                            # (k, r): hidden -> code coordinates, an isometry
        _Q = _G @ _R[:, :2].T                                                      # the predicted top-2 modes, read from the hidden state
        _X = (_Hf @ _Q.T).reshape(*H.shape[:2], 2)
        _rep = lambda pts: latent_means(pts, z, _n).reshape(_n * pts.shape[1], -1)
        return dict(pred=dict(pts=_E2, cones=_G @ D[:2], mean=latent_means(_E2, z, _n), rep=_rep(E)),
                    trained=dict(pts=_X, cones=_Q @ W.T, mean=latent_means(_X, z, _n), rep=_rep(H)))

    def belief_bg(ax, Dcols, r_x, r_y, colors, fade=0.3, res=500):
        """Background from the readout's belief at each plane point, softmax(p . D): the posterior-weighted
        mean of the outcome colours."""
        _gx, _gy = np.linspace(*r_x, res), np.linspace(*r_y, res)
        _C = np.stack(np.meshgrid(_gx, _gy), -1)                                   # [i, j] = (x_j, y_i)
        _L = _C @ Dcols; _p = np.exp(_L - _L.max(-1, keepdims=True)); _p /= _p.sum(-1, keepdims=True)
        _rgb = _p @ np.asarray(colors)[:, :3]
        ax.imshow(1 - fade * (1 - _rgb), origin="lower", extent=(*r_x, *r_y), zorder=0, aspect="auto")

    def semantic_planes(E, H, z, targets):
        """A semantic gauge for one world: each code is projected, by rotation only, onto the plane of the
        directions that best predict two Bayes quantities (``targets``: {name: (B, T)}). Each side carries
        its states, mean paths, Gram rows, and the decoding R^2 of the two targets."""
        _n = int(z.max()) + 1
        _Y = np.stack([_v.reshape(-1) for _v in targets.values()], 1)             # (B T, 2)
        _Yc = _Y - _Y.mean(0)
        def _side(code):
            _C = code.reshape(-1, code.shape[-1])
            _W = np.linalg.lstsq(_C, _Yc, rcond=None)[0]                          # directions predicting the targets
            _Q = np.linalg.qr(_W)[0]                                              # orthonormal: a rotation within the code
            _pts = (_C @ _Q).reshape(*code.shape[:2], 2)
            _r2 = 1 - ((_C @ _W - _Yc) ** 2).sum(0) / (_Yc ** 2).sum(0)
            return dict(pts=_pts, cones=None, mean=latent_means(_pts, z, _n),
                        rep=latent_means(code, z, _n).reshape(_n * code.shape[1], -1), r2=_r2)
        return dict(pred=_side(E), trained=_side(H), axes=list(targets))

    def state_panel(ax, title, pl, colors, b, rays=True, background=True, cloud=None, belief=None, dot=34, path=True):
        """One plane: the readout's belief as background (optional), sampled states (grey; coloured by
        ``cloud`` (B, T) on plasma, dark at its min, yellow at its max; or by ``belief`` (B, T, n), each
        state the belief-weighted mix of the latent colours), optionally a ray per true latent through
        its mean end state, optionally ``latents`` (n, 2): the kernel's own point per latent, an outlined marker; and one highlighted stream
        (light to dark with t)."""
        _flat = pl["pts"].reshape(-1, 2)
        _lo, _hi = _flat.min(0), _flat.max(0); _pad = 0.06 * (_hi - _lo).max()
        _xr, _yr = (_lo[0] - _pad, _hi[0] + _pad), (_lo[1] - _pad, _hi[1] + _pad)
        if background:
            belief_bg(ax, pl["cones"], _xr, _yr, colors)
        if belief is not None:
            _rgb = belief.reshape(-1, belief.shape[-1]) @ np.asarray(colors)[:, :3]
            ax.scatter(*_flat.T, s=dot, color=_rgb, alpha=0.6, lw=0, zorder=2)
        elif cloud is None:
            ax.scatter(*_flat.T, s=7, color="0.55", alpha=0.18, lw=0, zorder=2)
        else:
            _u = (cloud.ravel() - cloud.min()) / np.ptp(cloud)
            ax.scatter(*_flat.T, s=7, color=plt.cm.plasma(0.05 + 0.9 * _u), alpha=0.45, lw=0, zorder=2)
        if rays:
            for _z, (_dx, _dy) in enumerate(pl["mean"][:, -1]):
                _s = min(((_xr[1] if _dx > 0 else _xr[0]) / _dx) if _dx else np.inf,
                         ((_yr[1] if _dy > 0 else _yr[0]) / _dy) if _dy else np.inf)
                ax.plot([0, _s * _dx], [0, _s * _dy], color=np.asarray(colors[_z][:3]) * 0.8,
                        lw=1.1 if len(colors) <= 12 else 0.7, alpha=0.6, zorder=3)
        if path:                                                                            # one highlighted stream, light to dark with t
            ax.plot(*pl["pts"][b].T, color="0.1", lw=1.8, zorder=4,
                    path_effects=[pe.Stroke(linewidth=4.0, foreground="white"), pe.Normal()])   # a halo lifts the path off the cloud
            ax.scatter(*pl["pts"][b].T, c=np.linspace(0.25, 1.0, pl["pts"].shape[1]), cmap="Greys", vmin=0, vmax=1,
                       s=70, zorder=5, edgecolor="0.1", lw=0.9)
        frame(ax, title, _xr, _yr, fontsize=FONT); ax.set_box_aspect(None)          # fill the cell
        if "axes" in pl:
            ax.set_xlabel(pl["axes"][0], fontsize=FONT - 4); ax.set_ylabel(pl["axes"][1], fontsize=FONT - 4)

    def pick_stream(pts, toward=None, lowest=None, lo=0.85, hi=0.95):
        """A far-reaching stream (end radius between the lo and hi quantiles): with ``toward`` (an angle),
        the one heading closest to it; with ``lowest`` (B, T), the one ending at its smallest value; else
        the one nearest the middle of that band."""
        _end = pts[:, -1]; _r = np.linalg.norm(_end, axis=1)
        _far = np.flatnonzero((_r > np.quantile(_r, lo)) & (_r < np.quantile(_r, hi)))
        if toward is not None:
            return int(_far[np.argmin(np.abs(np.angle(_end[_far] @ [1, 1j]) - toward))])
        if lowest is not None:
            return int(_far[np.argmin(lowest[_far, -1])])
        return int(_far[np.argmin(np.abs(_r[_far] - np.quantile(_r, (lo + hi) / 2)))])

    FONT = 18                                                                     # panel-title size inside a block

    def plane_block(pl, colors, b, cka_all, rays=True, background=True, cloud=None, belief=None, dot=34, path=True, sketch=None):
        """One world's block, laid out like `plot_embedding_panels`: predicted beside transformer (states
        above; below, Grams of the mean path per true latent with the CKA over every state). Rendered to an
        in-memory png for `merge_panels`; only the merged figure is saved. ``belief``: {side: (B, T, n)}
        colours each side's states by its own belief over the latents. ``sketch(ax)`` draws the world's
        data-generating process (its latent graph) in a strip above the panels.
        """
        _fig = plt.figure(figsize=(7.6, 9.0 if sketch is None else 10.0))          # large in inches: merged fonts match the geometry figure
        if sketch is None:
            _gs = _fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.2, wspace=0.25)
        else:
            _outer = _fig.add_gridspec(2, 1, height_ratios=[0.32, 2.15], hspace=0.12)
            sketch(_fig.add_subplot(_outer[0]))
            _gs = _outer[1].subgridspec(2, 2, height_ratios=[1.15, 1], hspace=0.2, wspace=0.25)
        for _j, (_side, _label) in enumerate([("pred", "predicted"), ("trained", "transformer")]):
            if "axes" in pl:                                                       # semantic gauge: name the axes, report decoding R^2
                pl[_side]["axes"] = pl["axes"]
                _label += "   $R^2$ " + ", ".join(f"{_r:.2f}" for _r in pl[_side]["r2"])
            state_panel(_fig.add_subplot(_gs[0, _j]), _label, pl[_side], colors, b, rays=rays, background=background, cloud=cloud,
                        belief=None if belief is None else belief[_side], dot=dot, path=path)
            _ax = _fig.add_subplot(_gs[1, _j])
            plot_utils.gram_heatmap(_ax, pl[_side]["rep"],
                                    title=r"$E^{*}E^{*\top}$" if _j == 0 else r"$EE^{\top}$" + f"   CKA {cka_all:.3f}")
            _ax.title.set_fontsize(FONT)
        _buf = io.BytesIO(); _fig.savefig(_buf, format="png", dpi=600, bbox_inches="tight"); plt.close(_fig)
        _buf.seek(0); return _buf

    planes = {_name: planes_for(_name, factor_logcond(_h["LP"].reshape(-1, _h["D"].shape[1]))[1].reshape(*_h["LP"].shape[:2], -1)[:, :T],
                                _h["D"], fits[_name]["H"], fits[_name]["W"], _h["run"]["latent"])
              for _name, _h in hist.items()}
    for _name, _h in hist.items():
        print(f"{_name}: streams per latent {np.bincount(_h['run']['latent']).min()}-{np.bincount(_h['run']['latent']).max()}")
    return FONT, pick_stream, plane_block, planes, planes_for


@app.cell
def _(
    FIGURES,
    T,
    fits,
    hist,
    latent_colors,
    logsumexp,
    np,
    pick_stream,
    plane_block,
    planes,
    plot_utils,
):
    _blocks = []
    for _name, _pl in planes.items():
        _b = pick_stream(_pl["pred"]["pts"], toward=3 * np.pi / 4) if _name == "week" else 0   # the ring: a far stream heading up-left
        _lg = hist[_name]["run"]["logits"][:, :T].astype(float)                 # the model's belief over z, per state
        _belief = dict(pred=np.exp(hist[_name]["LP"][:, :T]), trained=np.exp(_lg - logsumexp(_lg, -1, keepdims=True)))
        _blocks.append(plane_block(_pl, latent_colors(_name, hist[_name]["D"].shape[1]), _b, fits[_name]["cka"],
                                   rays=False, background=False, belief=_belief))
    fig_latent = plot_utils.merge_panels(_blocks, ncols=3, label_size=30)
    plot_utils.save_figure(fig_latent, FIGURES / "inference_latent.png", dpi=200)
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
        fits_next[_name] = dict(scores=_scores, S=_S / _S[0], S_static=_S_static / _S_static[0], kl=_kl,
                                E=_E.reshape(*_LX.shape[:2], -1)[:, :T], D=_D, H=_run["hidden_states"][-1][:, :T], W=_run["readout"],
                                belief=dict(pred=np.exp(_LX[:, :T]), trained=np.exp(_lq)), latent=_run["latent"])
        print(f"{_name}: KL(Bayes||model) {_kl:.4f} | history sv {np.round(_S / _S[0], 3)} | static sv {np.round(_S_static / _S_static[0], 3)}")
        for _k, (_r2, _c) in _scores.items():
            print(f"   {_k:22s} affine R2 {_r2:.3f}  CKA {_c:.3f}")
    return (fits_next,)


@app.cell
def _(mo):
    mo.md(r"""
    Figure 1 again for the next-token model: the predictive code's top two modes against the hidden
    states, each state coloured by its belief over the next token (Bayes predictive on the predicted
    side, the model's softmax on the trained side); the binary world's tokens are royal (blue yes,
    orange no) and female (purple yes, green no).
    """)
    return


@app.cell
def _(
    FIGURES,
    fits_next,
    np,
    pick_stream,
    plane_block,
    planes_for,
    plot_utils,
    plt,
    to_rgb,
):
    def token_colors(name, m):
        """One colour per token: the binary pairs royal (blue/orange) and female (purple/green) in token
        order [1+, 2+, 1-, 2-]; the adjectives on the diverging map; the days on the cyclic map."""
        if name == "line":
            return plt.get_cmap(plot_utils.FIG.diverging_cmap)(np.linspace(0.05, 0.95, m))
        if name == "week":
            return plt.cm.twilight(np.arange(m) / m)
        return np.array([to_rgb(plot_utils.FIG.colors[_k]) for _k in ("blue", "purple", "orange", "green")])

    planes_next = {_name: planes_for(_name, _f["E"], _f["D"], _f["H"], _f["W"], _f["latent"], tokens=True)
                   for _name, _f in fits_next.items()}
    _blocks = []
    for _name, _pl in planes_next.items():
        _b = pick_stream(_pl["pred"]["pts"], toward=3 * np.pi / 4) if _name == "week" else 0
        _blocks.append(plane_block(_pl, token_colors(_name, fits_next[_name]["D"].shape[1]), _b, fits_next[_name]["scores"]["log-predictive code"][1],
                                   rays=False, background=False, belief=fits_next[_name]["belief"]))
    fig_next = plot_utils.merge_panels(_blocks, ncols=3, label_size=30)
    plot_utils.save_figure(fig_next, FIGURES / "inference_next.png", dpi=200)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Hypothesis spaces

    Bayesian generalization on a line and a grid (Tenenbaum & Griffiths 2001) and a taxonomy
    (Xu & Tenenbaum 2007): the model predicts the next example, so the rows are log-predictives over
    items and the belief over hypotheses is near full rank. Same figure as section 1, one block per
    world: predicted beside transformer, states in the top two modes coloured by generalization
    breadth, the posterior's expected predictive entropy (yellow: broad, dark: specific), one far-reaching
    stream; below, Grams of the mean path per true hypothesis, rows ordered (hypothesis, $t$), with the
    CKA over every state. No rays or belief background: with 55 to 225 hypotheses the rays would only
    clutter, and the predictive over 10 to 27 items is broad wherever states sit. Printed above: affine
    $R^2$, CKA with its null, and the mean KL(Bayes $\|$ model).
    """)
    return


@app.cell
def _(factor_logcond, log_posteriors, logsumexp, np, world):
    HYP = {"intervals": ({"structure": "intervals", "n": 10}, False),
           "rectangles": ({"structure": "rectangles", "n": 4}, False),    # 100 hypotheses over 16 cells
           "subtrees": ({"structure": "subtrees", "n": 3, "k": 3}, False)}
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
    return (fits_hyp,)


@app.cell
def _(
    FIGURES,
    T,
    fits_hyp,
    hyp,
    log_posteriors,
    np,
    pick_stream,
    plane_block,
    planes_for,
    plot_utils,
    plt,
):
    def breadth(ds, obs):
        """Generalization breadth per state: the posterior's expected log effective size of the hypothesis,
        exp of its predictive entropy (about the span length, rectangle area, subtree leaf count; the lapse
        rate keeps every item possible, so a hard support would be the whole space). obs (B, L) -> (B, L+1):
        state t has seen obs[:, :t]."""
        _ll = np.asarray(ds.conditional)
        _logsize = -(np.exp(_ll) * _ll).sum(1)                         # entropy of p(x | h) = log effective size
        return np.exp(log_posteriors(ds, obs)) @ _logsize

    # the same figure for the hypothesis spaces: no belief background (the predictive over 10-27 items is
    # broad wherever states sit) and no rays (55-225 hypotheses); the cloud is coloured by the posterior's
    # expected breadth, yellow = broad, dark = specific; Grams of the mean path per true hypothesis
    planes_hyp = {_name: planes_for(_name, _h["E"][:, :T], _h["D"], _h["run"]["hidden_states"][-1][:, :T], _h["run"]["readout"],
                                    _h["run"]["latent"]) for _name, _h in hyp.items()}
    _blocks = []
    for _name, _pl in planes_hyp.items():
        _C = hyp[_name]["D"].shape[1]
        _br = breadth(hyp[_name]["ds"], hyp[_name]["obs"])[:, :T]
        _blocks.append(plane_block(_pl, plt.get_cmap(plot_utils.FIG.diverging_cmap)(np.linspace(0.05, 0.95, _C)),
                                   pick_stream(_pl["pred"]["pts"], lowest=_br),   # a far stream that ends specific
                                   fits_hyp[_name]["cka"], rays=False, background=False, cloud=_br))
    fig_hyp = plot_utils.merge_panels(_blocks, ncols=3, label_size=30)
    plot_utils.save_figure(fig_hyp, FIGURES / "inference_hypotheses.png", dpi=200)
    return (breadth,)


@app.cell
def _(T, breadth, fits_hyp, hyp, mo, np, pick_stream, rigid_map):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    def states_3d(name):
        """Interactive top-3 view of one hypothesis world: predicted beside transformer (rotated onto the
        prediction), states coloured by generalization breadth (dark = specific), one far stream drawn."""
        _h = hyp[name]
        _E3 = _h["E"][:, :T, :3]
        _H = _h["run"]["hidden_states"][-1][:, :T]; _Hf = _H.reshape(-1, _H.shape[-1])
        _R = rigid_map(_Hf, _h["E"][:, :T].reshape(-1, _h["E"].shape[-1]))       # full-space alignment, then the top-3 modes
        _X3 = (_Hf @ _R[:, :3]).reshape(*_H.shape[:2], 3)
        _br = breadth(_h["ds"], _h["obs"])[:, :T]
        _b = pick_stream(_E3[..., :2], lowest=_br)
        _fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
                             subplot_titles=("predicted", f"transformer   CKA {fits_hyp[name]['cka']:.3f}"))
        for _j, _pts in enumerate([_E3, _X3], start=1):
            _f = _pts.reshape(-1, 3)
            _fig.add_trace(go.Scatter3d(x=_f[:, 0], y=_f[:, 1], z=_f[:, 2], mode="markers", showlegend=False,
                                        marker=dict(size=1.6, color=_br.ravel(), colorscale="Plasma",
                                                    opacity=0.5, cmin=_br.min(), cmax=_br.max())), row=1, col=_j)
            _fig.add_trace(go.Scatter3d(x=_pts[_b, :, 0], y=_pts[_b, :, 1], z=_pts[_b, :, 2], mode="lines+markers", showlegend=False,
                                        line=dict(color="black", width=3),
                                        marker=dict(size=4, color=np.linspace(0.2, 1, T), colorscale="Greys", cmin=0, cmax=1,
                                                    line=dict(color="white", width=1))), row=1, col=_j)
        _fig.update_layout(height=520, width=1100, margin=dict(l=0, r=0, t=40, b=0))
        _fig.update_scenes(xaxis_title="mode 1", yaxis_title="mode 2", zaxis_title="mode 3", aspectmode="data")
        return _fig

    mo.ui.tabs({_name: states_3d(_name) for _name in hyp})
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. A moving latent: hidden Markov worlds

    The latent now walks its structure and is read through a few coarse, sharp sensors, so every
    observation is one of a handful of contraction maps on the belief and the reachable beliefs form a
    self-similar set. Three worlds: Mess3 (three states, sticky, one noisy symbol per state), a random
    walk on a 12-ring with four sensors at the quarter marks, and a random walk on a 5 × 5 grid with four
    sensors at the corners. Figure 1's layout: Bayes (the one-step predictive belief, the model's
    target), factored, beside the transformer; states coloured by belief; one stream highlighted.
    """)
    return


@app.cell
def _(
    align_representations,
    cka,
    factor_logcond,
    graph_overrides,
    load_run,
    logsumexp,
    np,
    predicted_inference_geometry,
    torch,
):
    HMMS = {"mess3": dict(spec={"structure": "complete", "n": 3, "gamma": 0.1062, "cooccurrence": "heat"},          # stay 0.82
                          sigma=0.45, emission_floor=0.0, transition_floor=0.0, sensors=None),                            # symbol accuracy 0.86
            "ring": dict(spec={"structure": "ring", "n": 12, "gamma": 0.5, "cooccurrence": "diffusion"},
                         sigma=0.8, emission_floor=0.15, transition_floor=0.05, sensors=[0, 3, 6, 9]),
            "grid": dict(spec={"structure": "grid", "n": 5, "m": 5, "gamma": 0.6, "cooccurrence": "diffusion"},
                         sigma=1.5, emission_floor=0.15, transition_floor=0.05, sensors=[0, 4, 20, 24])}
    T_HMM = 24

    def hmm_run(w):
        """The saved run of a hidden-Markov world: the hmm_transformer preset with the latent as target."""
        _sensors = "null" if w["sensors"] is None else str(w["sensors"]).replace(" ", "")
        return load_run(graph_overrides(w["spec"], "hmm") + [
            "runner=hmm_transformer", "++infer_latent=true", f"training_kwargs.context_length={T_HMM}", "training_kwargs.train_steps=40000",
            f"emission_sigma={w['sigma']}", f"emission_floor={w['emission_floor']}", f"transition_floor={w['transition_floor']}", f"sensors={_sensors}"])

    hmms = {}
    for _name, _w in HMMS.items():
        _run = hmm_run(_w)
        _ds, _obs = _run["dataset"], _run["inputs"][:, 1:]
        # the label at position i is the state emitting the next symbol, so Bayes at position i is the one-step
        # predictive belief: the filtered belief after i observations pushed through the transition (the prior at i = 0)
        _b = _ds.forward_algorithm(torch.as_tensor(_obs, device=_ds.device)).cpu().numpy()        # filtered, after t + 1 observations
        _prior = np.broadcast_to(_ds.prior.cpu().numpy(), (len(_obs), 1, _ds.n_states))
        _LP = np.log(np.concatenate([_prior, _b[:, :-1] @ _ds.transition.cpu().numpy()], 1))     # (B, T, n)
        _Z = _run["latents"]                                                                         # (B, T): the state each position predicts
        _Lc, _Efull, _D, _S = factor_logcond(_LP.reshape(-1, _LP.shape[-1]))
        _H = _run["hidden_states"][-1][:, :T_HMM]
        _Ep = predicted_inference_geometry(_LP.reshape(-1, _LP.shape[-1]), _D[:2]).reshape(-1, T_HMM, 2)
        _lg = _run["logits"][:, :T_HMM].astype(float)
        hmms[_name] = dict(run=_run, ds=_ds, LP=_LP, E=_Efull.reshape(*_LP.shape[:2], -1), D=_D[:3], S=_S[:3], H=_H, W=_run["readout"], Z=_Z,
                           belief=dict(pred=np.exp(_LP), trained=np.exp(_lg - logsumexp(_lg, -1, keepdims=True))),
                           r2=align_representations(_Ep.reshape(-1, 2), _H.reshape(-1, _H.shape[-1]), "linear")[1],
                           cka=cka(_H.reshape(-1, _H.shape[-1]), _Ep.reshape(-1, 2)))
        _kl = (np.exp(_LP) * (_LP - np.log(hmms[_name]["belief"]["trained"]))).sum(-1).mean()
        print(f"{_name}: belief spectrum {np.round(_S[:4] / _S[0], 3)} | affine R2 predicted->hidden {hmms[_name]['r2']:.3f} | CKA {hmms[_name]['cka']:.3f} | KL(Bayes||model) {_kl:.4f}")
    return HMMS, hmms


@app.cell
def _(FIGURES, FONT, HMMS, hmms, np, plane_block, planes_for, plot_utils, plt):
    def hmm_colors(name, ds):
        """State colours: Mess3's three states pure red/green/blue (so the belief mixes RGB, as in Shai et
        al.); the ring on the cyclic map; the grid by column on a stark red-blue (coolwarm) darkened by row,
        so its sheet reads differently from the ring."""
        n = ds.n_states
        if name == "mess3":
            return np.eye(3)                                                         # RGB corners of the belief simplex
        if name == "ring":
            return plt.cm.twilight(np.arange(n) / n)[:, :3]
        _rows, _cols = np.divmod(np.arange(n), 5)
        _hue = plt.cm.coolwarm(_cols / 4)[:, :3]                                     # starker than vlag: crisper columns
        return _hue * (1 - 0.5 * _rows / 4)[:, None]

    from src.code.data import structure_adjacency
    from matplotlib.collections import LineCollection

    def dgp_sketch(name, colors):
        """Draw world ``name``'s latent graph: light edges, nodes in their state colours, the sensor
        states ringed in black. Reads positions and adjacency from the structure gallery."""
        _spec = HMMS[name]["spec"]; _A = structure_adjacency(_spec)
        _pos = plot_utils._shape_positions(_spec["structure"], _spec)
        _sensors = list(range(len(_pos))) if HMMS[name]["sensors"] is None else HMMS[name]["sensors"]
        def _draw(ax):
            _ii, _jj = np.where(np.triu(_A, 1) > 0)
            ax.add_collection(LineCollection([[_pos[a], _pos[b]] for a, b in zip(_ii, _jj)], colors="0.8", linewidths=0.6, zorder=0))
            ax.scatter(*_pos.T, c=colors, s=26, zorder=1, edgecolor="none")
            ax.scatter(*_pos[_sensors].T, s=95, facecolor="none", edgecolor="0.1", lw=1.3, zorder=2)   # the sensors
            ax.set_aspect("equal"); ax.set_axis_off()
            ax.set_title(f"latent graph, {len(_sensors)} sensor{'s' if len(_sensors) > 1 else ''}", fontsize=FONT - 2)
        return _draw

    planes_hmm = {_name: planes_for(_name, _c["E"], _c["D"], _c["H"], _c["W"], _c["Z"]) for _name, _c in hmms.items()}
    _blocks = []
    for _name, _pl in planes_hmm.items():
        _cols = hmm_colors(_name, hmms[_name]["ds"])
        _blocks.append(plane_block(_pl, _cols, 0, hmms[_name]["cka"],
                                   rays=False, background=False, belief=hmms[_name]["belief"], dot=3, path=False, sketch=dgp_sketch(_name, _cols)))
    fig_hmm = plot_utils.merge_panels(_blocks, ncols=3, label_size=30)
    plot_utils.save_figure(fig_hmm, FIGURES / "inference_hmm.png", dpi=200)
    return


if __name__ == "__main__":
    app.run()
