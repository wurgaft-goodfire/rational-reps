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
    # Feature geometry of a rank-limited softmax

    Trained networks (blue), the additive and additive-plus-pairs reference models (green,
    purple), and the theory of `notes/superposition.tex` (black dashed): water filling →
    pressure-driven orientation on the Fantope → completion by decoder and pairs at the
    predicted frame, stored as a run of its own (`runner=theory`, representation
    $z E^\star + \phi(z) E_2 + e_0$ and readout $D$). Every model is read from its saved
    representation $h(z)$ and readout $D$ and measured by the first-order vectors $E_i$ of the
    least-squares fit $h(z) \approx \bar h + \sum_i E_i z_i$; correlated cells use the predictive
    modes $z\,C^{-1/2}$. Leverage, the spectrum and the leverage projector are gauge-invariant;
    the Grams are in each model's own gauge (the loss is flat along $E \to EA$, $D \to DA^{-\top}$,
    and AdamW does not balance the two). Five runs per cell, task and training seeds coupled.
    Each figure has its own setting selectors; results and open items in
    `notes/superposition_results.md`.
    """)
    return


@app.cell
def _(mo):
    import os, sys
    import numpy as np

    ROOT = mo.notebook_dir().parent.parent    # src/notebooks -> <repo>
    os.chdir(ROOT); sys.path.insert(0, str(ROOT))
    from hydra.core.override_parser.overrides_parser import OverridesParser
    from omegaconf import OmegaConf
    from src.code.analysis_utils import feature_vectors, has_run, leverage, load_run, rigid_map
    from src.code.geometry import factor_logcond, modes, pair_features
    from src.code.train import resolve_config
    from src.code import plot_utils
    import matplotlib.pyplot as plt

    # the sweep is the setting's: conf/setting/superposition.yaml declares it under
    # hydra.sweeper.params, and `python -m src.code.train -m runner=<class> setting=superposition`
    # trains it; here the same params name the cells to read
    SETTING = "superposition"
    _params = OmegaConf.load(ROOT / "conf" / "setting" / f"{SETTING}.yaml").hydra.sweeper.params
    GRID = {_o.key_or_group: list(_o.sweep_string_iterator()) if _o.is_sweep_override() else [str(_o.value())]
            for _o in OverridesParser.create().parse_overrides([f"{_k}={_v}" for _k, _v in _params.items()])}
    P_KEY, B_KEY, R_KEY = "structure_kwargs.p", "structure_kwargs.beta", "structure_kwargs.rho"
    D_KEY, S_KEY = "model_kwargs.model_config.embed_dim", "random_seed"
    D_GRID, SEEDS = GRID[D_KEY], GRID[S_KEY]

    def cell_overrides(runner, p, b, r, d, s):
        """One cell's run: the setting, the model class and one value per swept key."""
        _picks = {P_KEY: p, B_KEY: b, R_KEY: r, D_KEY: d, S_KEY: s}
        return [f"setting={SETTING}", f"runner={runner}"] + [f"{_k}={_picks.get(_k, _v[0])}" for _k, _v in GRID.items()]

    CONFIG = resolve_config(cell_overrides("linear", "0.3", "0.5", "0.0", D_GRID[0], SEEDS[0]))   # the figure cache lives under the reference cell
    COL = plot_utils.FIG.colors
    RUNNERS = {"linear": ("trained", COL["blue"]), "paired_features": ("pairs model", COL["purple"]),
               "additive_features": ("additive model", COL["green"])}
    CLASSES = list(RUNNERS) + ["theory"]                 # every model class a cell can hold
    return (
        B_KEY,
        CLASSES,
        COL,
        CONFIG,
        D_GRID,
        GRID,
        P_KEY,
        RUNNERS,
        R_KEY,
        SEEDS,
        cell_overrides,
        factor_logcond,
        feature_vectors,
        has_run,
        leverage,
        load_run,
        modes,
        np,
        pair_features,
        plot_utils,
        plt,
        rigid_map,
    )


@app.cell
def _(
    CLASSES,
    SEEDS,
    cell_overrides,
    factor_logcond,
    feature_vectors,
    has_run,
    load_run,
    modes,
    np,
    rigid_map,
):
    # loaded cells, filled on demand the first time a figure asks for one
    SW = {}

    def sweep_rows(p, b, r, d, classes):
        """One cell, per seed: the cell's coordinates X (features, or predictive modes), the
        ladder alpha, and for every model class present (the theory included) its saved
        representation and readout ('model') and the first-order vectors E_i of the least-squares
        fit h(z) ~ h_bar + sum_i E_i x_i ('E')."""
        _rows = []
        for _s in SEEDS:
            _runs = {_c: load_run(cell_overrides(_c, p, b, r, d, _s)) for _c in classes}
            _ds = _runs["linear"]["dataset"]
            _, _X = modes(_ds.info["alpha"], _ds.features, float(r))
            _model = {_c: (_runs[_c]["representation"], _runs[_c]["readout"]) for _c in classes}
            _rows.append(dict(X=_X, alpha=_ds.info["alpha"], model=_model,
                              E={_c: feature_vectors(_H, _X)[0] for _c, (_H, _D) in _model.items()}))
        return _rows

    HEAD = 3                                          # figure 4: the first three features by alpha are the head ...
    CUT = 0.2                                         # ... the rest with leverage >= 1/5 are the superposed band; below is not drawn

    def whiten(V):
        """V (V'V)^{-1/2} up to a rotation: rows are the whitened vectors, squared lengths the leverage."""
        _U, _, _Vt = np.linalg.svd(np.asarray(V, float), full_matrices=False)
        return _U @ _Vt

    def aligned_mean(mats, ref):
        """The shared geometry of a set of frames: whiten each, rotate it rigidly onto ``ref`` (orthogonal
        Procrustes over all rows), average. Seed-specific wobble cancels; what every seed agrees on stays."""
        return np.mean([whiten(_M) @ rigid_map(whiten(_M), ref) for _M in mats], 0)

    def projector(V):
        """Whitened Gram of the rows of V: V (V'V)^+ V', the leverage projector. Gauge-free (V -> V A leaves
        it fixed); its diagonal is the leverage, its entries the inner products of the whitened vectors."""
        _V = np.asarray(V, float)
        return _V @ np.linalg.pinv(_V.T @ _V) @ _V.T

    def balanced_factors(H, D, d):
        """A model's (code, readout) in the balanced gauge: `factor_logcond` of its logit table H D' (row-
        centred, SVD scale split evenly) at rank d. The code is scaled per state, E'E / N = diag(sigma_k /
        sqrt N) as in figure 4, so its sizes compare to alpha; the readout's rows are centred over outcomes
        by construction. Returns (E (N, d), D (m, d)). The gauge GD from small init learns."""
        _, _E, _D, _ = factor_logcond(np.asarray(H, float) @ np.asarray(D, float).T)
        return len(H) ** 0.25 * _E[:, :d], _D[:d].T

    def load_cell(p, b, r, d):
        """One loaded cell, cached in SW: every model class whose seeds are all saved; None
        until the trained linear network and the theory are."""
        if (p, b, r, d) not in SW:
            _have = [_c for _c in CLASSES if all(has_run(cell_overrides(_c, p, b, r, d, _s)) for _s in SEEDS)]
            SW[(p, b, r, d)] = sweep_rows(p, b, r, d, _have) if {"linear", "theory"} <= set(_have) else None
        return SW[(p, b, r, d)]

    return CUT, HEAD, aligned_mean, balanced_factors, load_cell, whiten


@app.cell
def _(B_KEY, D_GRID, GRID, P_KEY, R_KEY, load_cell, mo):
    def setting_ui():
        """Selectors for one figure: sparsity p, ladder slope beta, correlation rho."""
        return mo.ui.dictionary({
            "p": mo.ui.dropdown(options=GRID[P_KEY], value="0.3", label="sparsity p"),
            "beta": mo.ui.dropdown(options=GRID[B_KEY], value="0.5", label="ladder slope β"),
            "rho": mo.ui.dropdown(options=GRID[R_KEY], value="0.0", label="correlation ρ"),
        })

    def rows_for(sel):
        """The sweep rows of a selected setting: dict(rows {d: rows} or None, title, p, b, rho,
        model classes present in every d)."""
        _p, _b, _r = sel["p"], sel["beta"], sel["rho"]
        _rows = {_d: load_cell(_p, _b, _r, _d) for _d in D_GRID}
        if any(_v is None for _v in _rows.values()):
            _rows = None
        _runs = set.intersection(*[set(_rows[_d][0]["model"]) - {"theory"} for _d in D_GRID]) if _rows else set()
        return dict(rows=_rows, title=rf"$p$ = {_p}, $\beta$ = {_b}, $\rho$ = {_r}",
                    p=float(_p), b=float(_b), rho=float(_r), runners=_runs)

    NOT_YET = mo.md("this setting is not trained yet")

    def selector_row(sel):
        return mo.hstack([sel["p"], sel["beta"], sel["rho"]], justify="start", gap=2)

    return NOT_YET, rows_for, selector_row, setting_ui


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 1 — allocation
    Leverage $d_i$ of the first-order vectors (diagonal of the hat matrix $E (E^\top E)^{+} E^\top$,
    gauge-invariant): per-seed trained dots, reference means $\pm$ se, and the predicted
    allocation (water filling plus the soft tail).
    """)
    return


@app.cell(hide_code=True)
def _(setting_ui):
    SEL_F1 = setting_ui()
    return (SEL_F1,)


@app.cell
def _(
    COL,
    CONFIG,
    D_GRID,
    NOT_YET,
    RUNNERS,
    SEL_F1,
    leverage,
    mo,
    np,
    plot_utils,
    plt,
    rows_for,
    selector_row,
):
    _sel = rows_for(SEL_F1.value)
    mo.stop(_sel["rows"] is None, NOT_YET)
    _lev = lambda _row, _c: leverage(_row["E"][_c])[0][:24]
    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for _ax, _d in zip(_axes, D_GRID):
        _x = np.arange(24)
        for _row in _sel["rows"][_d]:
            _ax.plot(_x, _lev(_row, "linear"), "o", ms=4, alpha=0.45, color=COL["blue"], label="trained" if _row is _sel["rows"][_d][0] else None)
        for _run, (_lbl, _col) in RUNNERS.items():
            if _run not in _sel["runners"]:
                continue
            _rm = np.array([_lev(_row, _run) for _row in _sel["rows"][_d]])
            _mu, _se = _rm.mean(0), _rm.std(0) / np.sqrt(len(_rm))
            if _run != "linear":
                _ax.plot(_x, _mu, "-", lw=1.8, color=_col, label=_lbl)
                _ax.fill_between(_x, _mu - _se, _mu + _se, color=_col, alpha=0.25, lw=0)
        _pr = np.array([_lev(_row, "theory") for _row in _sel["rows"][_d]])
        _m, _s = _pr.mean(0), _pr.std(0) / np.sqrt(len(_pr))
        _ax.plot(_x, _m, "--", lw=2.0, color="black", label="theory")
        _ax.fill_between(_x, _m - _s, _m + _s, color="0.3", alpha=0.18, lw=0)
        _ax.set_title(f"d = {_d}")
        _ax.set_xlabel("feature $i$" if _sel["rho"] == 0.0 else "mode $k$")
        plot_utils.FIG.style_axes(_ax)
    _axes[0].set_ylabel(r"leverage $d_i$")
    _axes[0].legend(frameon=False, fontsize=9)
    _fig.suptitle(_sel["title"], y=1.02)
    _out = plot_utils.save_figure(_fig, plot_utils.figure_path(CONFIG, "fig1_allocation"))
    mo.vstack([selector_row(SEL_F1), _out])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 2 — feature sizes
    $\|E_i\|^2$ of the first-order vectors, every model (theory included) in the balanced gauge of its
    per-state logit table: row-centred $h(z)D^\top$, SVD scale split evenly at rank $d$, $E^\top E/N =
    \mathrm{diag}(\sigma_k/\sqrt N)$. The gauge GD from small init learns; the head's excess over
    $\alpha_i$ is the amplification of the notes, a property of this gauge (the map's diagonal is
    $\bar\alpha$ exactly).
    """)
    return


@app.cell(hide_code=True)
def _(setting_ui):
    SEL_F2 = setting_ui()
    return (SEL_F2,)


@app.cell
def _(
    CONFIG,
    D_GRID,
    NOT_YET,
    RUNNERS,
    SEL_F2,
    balanced_factors,
    feature_vectors,
    mo,
    np,
    plot_utils,
    plt,
    rows_for,
    selector_row,
):
    _sel = rows_for(SEL_F2.value)
    mo.stop(_sel["rows"] is None, NOT_YET)
    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for _ax, _d in zip(_axes, D_GRID):
        _sizes = lambda _row, _c: (feature_vectors(balanced_factors(*_row["model"][_c], int(_d))[0], _row["X"])[0] ** 2).sum(1)[:24]
        _x = np.arange(24)
        for _run, (_lbl, _col) in RUNNERS.items():
            if _run not in _sel["runners"]:
                continue
            _rm = np.array([_sizes(_row, _run) for _row in _sel["rows"][_d]])
            _mu, _se = _rm.mean(0), _rm.std(0) / np.sqrt(len(_rm))
            _ax.plot(_x, _mu, "o-" if _run == "linear" else "-", ms=4, lw=1.2 if _run == "linear" else 1.8, color=_col, label=_lbl)
            _ax.fill_between(_x, _mu - _se, _mu + _se, color=_col, alpha=0.25, lw=0)
        _pr = np.array([_sizes(_row, "theory") for _row in _sel["rows"][_d]])
        _m, _s = _pr.mean(0), _pr.std(0) / np.sqrt(len(_pr))
        _ax.plot(_x, _m, "--", lw=2.0, color="black", label="theory")
        _ax.fill_between(_x, _m - _s, _m + _s, color="0.3", alpha=0.18, lw=0)
        _ax.set_title(f"d = {_d}")
        _ax.set_xlabel("feature $i$" if _sel["rho"] == 0.0 else "mode $k$")
        plot_utils.FIG.style_axes(_ax)
    _axes[0].set_ylabel(r"$\|E_i\|^2$")
    _axes[0].legend(frameon=False, fontsize=9)
    _fig.suptitle(_sel["title"], y=1.02)
    _out = plot_utils.save_figure(_fig, plot_utils.figure_path(CONFIG, "fig2_sizes"))
    mo.vstack([selector_row(SEL_F2), _out])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 3 — feature Grams
    Gram of the shared frame: the first-order vectors whitened, $E(E^\top E)^{-1/2}$, every seed rotated
    rigidly onto the first predicted frame, averaged, then $WW^\top$ over the top 16 features. Averaging the
    frames before the Gram cancels seed-specific wobble, so this is the geometry every seed shares (the mean
    of per-seed Grams would keep each seed's own overlaps). Gauge-free; the diagonal is the leverage of the
    mean vectors.
    """)
    return


@app.cell(hide_code=True)
def _(setting_ui):
    SEL_F3 = setting_ui()
    return (SEL_F3,)


@app.cell
def _(
    CONFIG,
    D_GRID,
    NOT_YET,
    RUNNERS,
    SEL_F3,
    aligned_mean,
    mo,
    np,
    plot_utils,
    plt,
    rows_for,
    selector_row,
    whiten,
):
    _sel = rows_for(SEL_F3.value)
    mo.stop(_sel["rows"] is None, NOT_YET)
    _K = 16
    _gram = lambda _rows, _c, _ref: (lambda _W: _W @ _W.T)(aligned_mean([_row["E"][_c] for _row in _rows], _ref))[:_K, :_K]
    _refs = [_run for _run in ("paired_features", "additive_features") if _run in _sel["runners"]]
    _fig, _axes = plt.subplots(2 + len(_refs), 3, figsize=(13.2, 4.1 * (2 + len(_refs))), squeeze=False, layout="constrained")
    _grid = {}
    for _c, _d in enumerate(D_GRID):
        _ref = whiten(_sel["rows"][_d][0]["E"]["theory"])                       # every frame is aligned to the first predicted one
        _panels = [(_gram(_sel["rows"][_d], "linear", _ref), "trained (5-seed mean)"),
                   (_gram(_sel["rows"][_d], "theory", _ref), "theory (5-draw mean)")]
        _panels += [(_gram(_sel["rows"][_d], _run, _ref), f"{RUNNERS[_run][0]} (5-seed mean)") for _run in _refs]
        _grid[_d] = _panels
    _v = max(np.abs(_Gm - np.diag(np.diag(_Gm))).max() for _panels in _grid.values() for _Gm, _lbl in _panels)
    for _c, (_d, _panels) in enumerate(_grid.items()):                                              # one scale for every panel, trained and theory: the
        for _r, (_Gm, _lbl) in enumerate(_panels):                                 # largest off-diagonal entry; the diagonal saturates
            _ax = _axes[_r, _c]
            _im = _ax.imshow(_Gm, cmap="vlag", vmin=-_v, vmax=_v)
            _ax.set_title(f"d = {_d} — {_lbl}", fontsize=11)
            _ax.set_xticks([]); _ax.set_yticks([])
    _fig.colorbar(_im, ax=_axes.ravel().tolist(), shrink=0.5, pad=0.02)
    _fig.suptitle(_sel["title"])
    _out = plot_utils.save_figure(_fig, plot_utils.figure_path(CONFIG, "fig3_grams"))
    mo.vstack([selector_row(SEL_F3), _out])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 4 — the frame, feature by feature
    Whitened first-order vectors $E(E^\top E)^{-1/2}$ (squared length the leverage), in three dimensions:
    the top three principal directions of the theory group's span (completed from the whole frame when
    the group has fewer than three vectors). Predicted beside the trained linear network; the trained frame is
    carried onto the predicted one by orthogonal Procrustes in the full $d$-dimensional space first, then
    projected, as in the inference figures; every seed's frame is aligned to the first seed's predicted frame
    before averaging.
    Rows are three groups, each in its own frame so the later ones are not squashed against the head's
    axes: the top three by leverage (Gram-Schmidt frame with the head along $z$) and the rest with
    $d_i > 0.2$ (the features that share dimensions, in their own PCA frame). Features below are not drawn. Colour: $\alpha_i$ on a log scale.
    """)
    return


@app.cell
def _(CUT, HEAD, np, plot_utils, plt):
    # the frame drawing, shared by figures 4 and 7
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d.proj3d import proj_transform

    class Arrow3D(FancyArrowPatch):
        """A FancyArrowPatch from the origin to a 3-D point: a filled head, unlike quiver's three lines."""
        def __init__(self, xyz, **kw):
            super().__init__((0, 0), (0, 0), **kw); self._xyz = xyz
        def do_3d_projection(self, renderer=None):
            _x0, _y0, _z0 = proj_transform(0, 0, 0, self.axes.M)
            _x1, _y1, _z1 = proj_transform(*self._xyz, self.axes.M)
            self.set_positions((_x0, _y0), (_x1, _y1))
            return min(_z0, _z1)

    def band(lev):
        """The superposed band: features with leverage above CUT, the head excluded."""
        return np.flatnonzero(lev > CUT)[HEAD:]

    def basis3(group, frame, tripod):
        """A view of the group. The head: Gram-Schmidt on its three vectors, the first along z, the next along
        x, the third along y. A superposed set: its top-3 principal directions, which spread it. Completed
        from the whole frame when the group has fewer than three vectors."""
        if tripod:
            _Qm, _R = np.linalg.qr(group[:3].T)
            _Q = (_Qm * np.sign(np.diag(_R))).T                                         # each axis points along its own vector
        else:
            _Q = np.linalg.svd(group, full_matrices=False)[2][:3]
        if len(_Q) < 3:
            _rest = frame - frame @ _Q.T @ _Q                                           # the frame outside the group's span
            _Q = np.vstack([_Q, np.linalg.svd(_rest, full_matrices=False)[2][:3 - len(_Q)]])
        return _Q[[1, 2, 0]]                                                            # x <- second, y <- third, z <- head

    # the house diverging map with its pale middle cut out: blue -> light blue | light red -> red over log alpha,
    # so no arrow washes out on the white page
    _vlag = plt.get_cmap(plot_utils.FIG.diverging_cmap)
    ALPHA_CMAP = plt.matplotlib.colors.ListedColormap(np.vstack([_vlag(np.linspace(0.0, 0.36, 128)), _vlag(np.linspace(0.64, 1.0, 128))]), name="vlag_cut")

    def frame_panel(ax, W, idx, B, title, alpha, norm, zoom=1.6, box=True):
        """The vectors W[idx] in the 3-D view B (3, d), each an arrow coloured by alpha; ``box`` draws the axes' box."""
        _xyz = W[idx] @ B.T
        for _k, _v in zip(idx, _xyz):
            ax.add_artist(Arrow3D(_v, arrowstyle="-|>", mutation_scale=16, lw=2.4, color=ALPHA_CMAP(norm(alpha[_k])), shrinkA=0, shrinkB=0))
        ax.set_xlim(-0.85, 0.85); ax.set_ylim(-0.85, 0.85); ax.set_zlim(-0.85, 0.85); ax.set_box_aspect((1, 1, 1), zoom=zoom)
        ax.set_proj_type("ortho"); ax.view_init(elev=24, azim=-45)                     # no perspective, x and y foreshortened alike
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        for _pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            _pane.pane.set_facecolor((1, 1, 1, 0))
        if not box:
            ax.set_axis_off()
        ax.set_title(title, fontsize=11)

    return ALPHA_CMAP, band, basis3, frame_panel


@app.cell(hide_code=True)
def _(setting_ui):
    SEL_F4 = setting_ui()
    return (SEL_F4,)


@app.cell
def _(
    ALPHA_CMAP,
    CONFIG,
    CUT,
    D_GRID,
    HEAD,
    NOT_YET,
    SEL_F4,
    aligned_mean,
    band,
    basis3,
    frame_panel,
    mo,
    np,
    plot_utils,
    plt,
    rows_for,
    selector_row,
    whiten,
):
    _sel = rows_for(SEL_F4.value)
    mo.stop(_sel["rows"] is None, NOT_YET)

    # two groups, each drawn in its own frame: the head (first HEAD features by alpha, a near-orthogonal
    # tripod) and the superposed band (the rest with leverage >= 1/5: where the frame's overlaps live)
    _GROUPS = [(f"Top {HEAD} $d_i$", lambda _lev: np.arange(HEAD), True),
               (f"Rest with $d_i > {CUT}$", band, False)]

    _alpha = np.asarray(_sel["rows"][D_GRID[0]][0]["alpha"], float)                # the ladder: colour by alpha_i
    _norm = plt.matplotlib.colors.LogNorm(_alpha.min(), _alpha.max())               # the ladder is a power law: log spreads it
    _fig = plt.figure(figsize=(15.5, 2.7 * len(_GROUPS)), layout="constrained")
    *_sfs, _sf_bar = _fig.subfigures(1, 4, wspace=0.12, width_ratios=[1, 1, 1, 0.12])       # gaps between d, none within
    for _c, (_d, _sf) in enumerate(zip(D_GRID, _sfs)):
        _sf.suptitle(f"d = {_d}", fontsize=15)
        _gs = _sf.add_gridspec(len(_GROUPS), 2, wspace=0, hspace=0, left=0, right=1, top=0.9, bottom=0)
        _rows = _sel["rows"][_d]
        _ref = whiten(_rows[0]["E"]["theory"])                                      # the frame every seed is aligned to
        _Wp = aligned_mean([_row["E"]["theory"] for _row in _rows], _ref)
        _Wt = aligned_mean([_row["E"]["linear"] for _row in _rows], _ref)
        _lev = (_Wp ** 2).sum(1)
        for _r, (_name, _pick, _tripod) in enumerate(_GROUPS):
            _idx = _pick(_lev)
            _B = basis3(_Wp[_idx], _Wp, _tripod) if len(_idx) else None
            for _j, (_W, _lbl) in enumerate([(_Wp, "predicted"), (_Wt, "linear")]):
                _ax = _sf.add_subplot(_gs[_r, _j], projection="3d")
                _ax.set_title(_lbl, fontsize=11)
                if _B is not None:
                    frame_panel(_ax, _W, _idx, _B, _lbl, _alpha, _norm)
                if _c == 0 and _j == 0:
                    _ax.text2D(-0.12, 0.5, _name, transform=_ax.transAxes, rotation=90, va="center", ha="center", fontsize=12)
    _fig.suptitle(_sel["title"])
    _cb = _fig.colorbar(plt.cm.ScalarMappable(norm=_norm, cmap=ALPHA_CMAP), cax=_sf_bar.add_axes([0.15, 0.25, 0.25, 0.5]), label=r"$\alpha_i$")
    _cb.set_ticks([1, 2, 4, 8]); _cb.set_ticklabels(["1", "2", "4", "8"]); _cb.ax.minorticks_off()
    _out = plot_utils.save_figure(_fig, plot_utils.figure_path(CONFIG, "fig4_frame"))
    mo.vstack([selector_row(SEL_F4), _out])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 5 — features and pairs together
    Gram of the shared frame of the regression vectors of the saved representation on $[1, z - \bar z, \phi(z)]$
    (whitened, seeds aligned to the first predicted frame, averaged; gauge-free): the top 16 features
    and the 15 pairs among the top 6, colour scale set by the pair blocks (the feature block
    saturates).
    """)
    return


@app.cell(hide_code=True)
def _(setting_ui):
    SEL_F6 = setting_ui()
    return (SEL_F6,)


@app.cell
def _(
    CONFIG,
    D_GRID,
    NOT_YET,
    SEL_F6,
    aligned_mean,
    mo,
    np,
    pair_features,
    plot_utils,
    plt,
    rows_for,
    selector_row,
    whiten,
):
    _sel = rows_for(SEL_F6.value)
    mo.stop(_sel["rows"] is None, NOT_YET)
    import itertools as _it
    _K, _TOP = 16, 6
    _pairs_all = list(_it.combinations(range(18), 2))
    _pairsel = [_pairs_all.index(p) for p in _it.combinations(range(_TOP), 2)]   # 15 pairs among the top 6

    def _ext_vectors(_H, _X):
        """Regression vectors of the representation on [1, x, phi(x)], the intercept dropped: (64 + pairs, d)."""
        _A = np.column_stack([np.ones(len(_X)), _X - _X.mean(0), pair_features(_X, 18)])
        return np.linalg.lstsq(_A, _H, rcond=None)[0][1:]

    def _ext_gram(_rows, _c, _ref):
        """Gram of the shared frame of those vectors (whitened, aligned to _ref, averaged): top-K features
        and the selected pairs."""
        _W = aligned_mean([_ext_vectors(_row["model"][_c][0], _row["X"]) for _row in _rows], _ref)
        _idx = np.r_[np.arange(_K), 64 + np.asarray(_pairsel)]
        return (_W @ _W.T)[np.ix_(_idx, _idx)]

    _has_pairs = "paired_features" in _sel["runners"]
    _fig, _axes = plt.subplots(3 if _has_pairs else 2, 3, figsize=(13.2, 12.4 if _has_pairs else 8.4), squeeze=False, layout="constrained")
    for _c, _d in enumerate(D_GRID):
        _rows = _sel["rows"][_d]
        _ref = whiten(_ext_vectors(_rows[0]["model"]["theory"][0], _rows[0]["X"]))
        _panels = [(_ext_gram(_rows, "linear", _ref), "trained (5-seed mean)"), (_ext_gram(_rows, "theory", _ref), "theory (5-draw mean)")]
        if _has_pairs:
            _panels.append((_ext_gram(_rows, "paired_features", _ref), "pairs model (5-seed mean)"))
        _v = max(np.abs(_Gm[_K:, :]).max() for _Gm, _lbl in _panels)   # scale to the pair blocks; the feature block saturates
        for _r, (_Gm, _lbl) in enumerate(_panels):
            _ax = _axes[_r, _c]
            _im = _ax.imshow(_Gm, cmap="vlag", vmin=-_v, vmax=_v)
            _ax.axhline(_K - 0.5, color="0.3", lw=0.8); _ax.axvline(_K - 0.5, color="0.3", lw=0.8)
            _ax.set_title(f"d = {_d} — {_lbl}", fontsize=11)
            _ax.set_xticks([]); _ax.set_yticks([])
        _fig.colorbar(_im, ax=_axes[:, _c].tolist(), shrink=0.5, pad=0.02)        # one bar per column: its own pair scale
    _fig.suptitle(_sel["title"])
    _out = plot_utils.save_figure(_fig, plot_utils.figure_path(CONFIG, "fig5_extended_grams"))
    mo.vstack([selector_row(SEL_F6), _out])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 6 — readout Grams
    Gram of the shared readout frame over outcomes: the balanced readout (`factor_logcond` of the logit
    table at rank $d$; row-centring the logits removes the softmax shift, a common offset of every saved
    readout row that would soak up leverage), whitened, every seed rotated onto the first predicted one,
    averaged, then the Gram, top 16; gauge-free.
    """)
    return


@app.cell(hide_code=True)
def _(setting_ui):
    SEL_F5 = setting_ui()
    return (SEL_F5,)


@app.cell
def _(
    CONFIG,
    D_GRID,
    NOT_YET,
    RUNNERS,
    SEL_F5,
    aligned_mean,
    balanced_factors,
    mo,
    np,
    plot_utils,
    plt,
    rows_for,
    selector_row,
    whiten,
):
    _sel = rows_for(SEL_F5.value)
    mo.stop(_sel["rows"] is None, NOT_YET)
    _K = 16
    # the balanced readout from the logit table: row-centring the logits removes the softmax shift (one
    # vector added to every saved readout row, 0.8 of a row's norm when trained) that would soak up leverage
    _dec = lambda _row, _c, _d: balanced_factors(*_row["model"][_c], int(_d))[1]
    _gram = lambda _rows, _c, _d, _ref: (lambda _W: _W @ _W.T)(aligned_mean([_dec(_row, _c, _d) for _row in _rows], _ref))[:_K, :_K]
    _refs = [_run for _run in ("paired_features", "additive_features") if _run in _sel["runners"]]
    _fig, _axes = plt.subplots(2 + len(_refs), 3, figsize=(13.2, 4.1 * (2 + len(_refs))), squeeze=False, layout="constrained")
    _grid = {}
    for _c, _d in enumerate(D_GRID):
        _ref = whiten(_dec(_sel["rows"][_d][0], "theory", _d))
        _panels = [(_gram(_sel["rows"][_d], "linear", _d, _ref), "trained (5-seed mean)"),
                   (_gram(_sel["rows"][_d], "theory", _d, _ref), "theory (5-draw mean)")]
        _panels += [(_gram(_sel["rows"][_d], _run, _d, _ref), f"{RUNNERS[_run][0]} (5-seed mean)") for _run in _refs]
        _grid[_d] = _panels
    _v = max(np.abs(_Gm - np.diag(np.diag(_Gm))).max() for _panels in _grid.values() for _Gm, _lbl in _panels)
    for _c, (_d, _panels) in enumerate(_grid.items()):                                              # one scale for every panel, trained and theory: the
        for _r, (_Gm, _lbl) in enumerate(_panels):                                 # largest off-diagonal entry; the diagonal saturates
            _ax = _axes[_r, _c]
            _im = _ax.imshow(_Gm, cmap="vlag", vmin=-_v, vmax=_v)
            _ax.set_title(f"d = {_d} — {_lbl}", fontsize=11)
            _ax.set_xticks([]); _ax.set_yticks([])
    _fig.colorbar(_im, ax=_axes.ravel().tolist(), shrink=0.5, pad=0.02)
    _fig.suptitle(_sel["title"])
    _out = plot_utils.save_figure(_fig, plot_utils.figure_path(CONFIG, "fig6_decoder_grams"))
    mo.vstack([selector_row(SEL_F5), _out])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Figure 7 — the sparsity sweep
    Sparsity $p$ falls left to right at one $d$. Per column, a predicted block and a trained block: the
    superposed band ($d_i > 0.2$, the second row of figure 4) in its PCA frame, the Gram of the shared frame
    (top 16 features, one colour scale over every panel, set by the largest off-diagonal entry), and the
    leverage $d_i$ of the first 24 features (seed mean).
    """)
    return


@app.cell
def _(B_KEY, D_GRID, GRID, R_KEY, mo):
    P_SWEEP = ["0.3", "0.15", "0.05"]
    SEL_F7 = mo.ui.dictionary({
        "beta": mo.ui.dropdown(options=GRID[B_KEY], value="0.5", label="ladder slope β"),
        "rho": mo.ui.dropdown(options=GRID[R_KEY], value="0.0", label="correlation ρ"),
        "d": mo.ui.dropdown(options=D_GRID, value="6", label="d"),
    })
    return P_SWEEP, SEL_F7


@app.cell
def _(
    ALPHA_CMAP,
    CONFIG,
    CUT,
    NOT_YET,
    P_SWEEP,
    SEL_F7,
    aligned_mean,
    band,
    basis3,
    frame_panel,
    leverage,
    load_cell,
    mo,
    np,
    plot_utils,
    plt,
    whiten,
):
    _b, _r, _d = SEL_F7.value["beta"], SEL_F7.value["rho"], SEL_F7.value["d"]
    _cells = {_p: load_cell(_p, _b, _r, _d) for _p in P_SWEEP}
    mo.stop(any(_rows is None for _rows in _cells.values()), NOT_YET)
    _K, _NBAR = 16, 18
    _SIDES = [("theory", "predicted"), ("linear", "trained")]

    _frames = {_p: {_c: aligned_mean([_row["E"][_c] for _row in _rows], whiten(_rows[0]["E"]["theory"])) for _c, _ in _SIDES}
               for _p, _rows in _cells.items()}
    _grams = {(_p, _c): _W[:_K] @ _W[:_K].T for _p, _fr in _frames.items() for _c, _W in _fr.items()}
    _v = max(np.abs(_Gm - np.diag(np.diag(_Gm))).max() for _Gm in _grams.values())      # one scale: the largest off-diagonal entry
    _levs = {(_p, _c): np.mean([leverage(_row["E"][_c])[0][:_NBAR] for _row in _rows], 0) for _p, _rows in _cells.items() for _c, _ in _SIDES}
    _alpha = np.asarray(_cells[P_SWEEP[0]][0]["alpha"], float)
    _norm = plt.matplotlib.colors.LogNorm(_alpha.min(), _alpha.max())

    _fig = plt.figure(figsize=(15.5, 7.2), layout="constrained")
    *_sfs, _sf_bar = _fig.subfigures(1, 4, wspace=0.22, width_ratios=[1, 1, 1, 0.12])       # gaps between p, none within
    for _col, (_p, _sf) in enumerate(zip(P_SWEEP, _sfs)):
        _sf.suptitle(f"$p$ = {_p}", fontsize=15)
        _gs = _sf.add_gridspec(3, 2, wspace=0.0, hspace=0.0, height_ratios=[0.8, 0.95, 0.85], left=0, right=1, top=0.94, bottom=0)
        _Wp = _frames[_p]["theory"]
        _idx = band((_Wp ** 2).sum(1))
        _B = basis3(_Wp[_idx], _Wp, False)
        for _j, (_c, _lbl) in enumerate(_SIDES):
            _ax3 = _sf.add_subplot(_gs[0, _j], projection="3d")
            frame_panel(_ax3, _frames[_p][_c], _idx, _B, _lbl, _alpha, _norm, zoom=2.0, box=False)
            _axg = _sf.add_subplot(_gs[1, _j])
            _im = _axg.imshow(_grams[_p, _c], cmap="vlag", vmin=-_v, vmax=_v)
            _axg.set_xticks([]); _axg.set_yticks([])
            _axb = _sf.add_subplot(_gs[2, _j])
            _axb.barh(np.arange(_NBAR), _levs[_p, _c], color="black", height=0.75)
            _axb.set_xlim(0, 1); _axb.set_ylim(_NBAR - 0.5, -0.5); _axb.set_yticks([]); _axb.set_xticks([0, 0.5, 1])
            _axb.set_xlabel("leverage $d_i$")
            plot_utils.FIG.style_axes(_axb)
            if _col == 0 and _j == 0:
                _ax3.text2D(-0.12, 0.5, f"band, $d_i > {CUT}$", transform=_ax3.transAxes, rotation=90, va="center", ha="center", fontsize=12)
                _axg.set_ylabel("Gram", fontsize=12)
                _axb.set_ylabel("feature $i$", fontsize=12)
    _fig.suptitle(rf"$d$ = {_d}, $\beta$ = {_b}, $\rho$ = {_r}")
    _cb = _fig.colorbar(plt.cm.ScalarMappable(norm=_norm, cmap=ALPHA_CMAP), cax=_sf_bar.add_axes([0.15, 0.7, 0.25, 0.25]), label=r"$\alpha_i$")
    _cb.set_ticks([1, 2, 4, 8]); _cb.set_ticklabels(["1", "2", "4", "8"]); _cb.ax.minorticks_off()
    _fig.colorbar(_im, cax=_sf_bar.add_axes([0.15, 0.36, 0.25, 0.25]))
    _out = plot_utils.save_figure(_fig, plot_utils.figure_path(CONFIG, "fig7_sparsity"))
    mo.vstack([mo.hstack([SEL_F7["beta"], SEL_F7["rho"], SEL_F7["d"]], justify="start", gap=2), _out])
    return


if __name__ == "__main__":
    app.run()
