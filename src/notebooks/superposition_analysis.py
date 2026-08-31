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
    # Superposition: the theory in four sentences

    $m$ binary features $z_i \sim \mathrm{Bernoulli}(p)$, target
    $p_\star(x{=}i \mid z) \propto \exp(\alpha_i z_i)$, rank-$d$ model. The theory
    (`notes/map_theory.md`):

    1. **The anchor law.** The feature $\to$ output map's diagonal is pinned exactly at the
       centred target, $R_{ii} = \alpha_i(1 - 1/m)$; interference is carried in negative
       couplings that cancel out of the response ($G_{ii} + pS_i = \alpha_i$ is this law's
       measurable shadow).
    2. **The allocation formula.** Capacity shares $h_i$ solve water-filling: calibrate,
       equalize the closed-form hybrid-field pressure $g_i(h_i)$ at one level $\nu$, spend
       $\sum h_i = d$ (`geometry.waterfill_h`; the convex Fantope program is its certified
       evaluator).
    3. **The mechanical map.** Everything measured — sizes, "amplified" head, spectra,
       Grams — is the balanced-SVD factorization of the predicted map; the certified
       realization is `geometry.dc_map` (truncated-nuclear difference-of-convex, every step
       a convex program).
    4. **The boundary.** Trained networks live between the additive and the additive+pairs
       class optima; the gap is training dynamics, out of scope by design.

    Degenerate directions (orientations, threshold ties, pair allocation) are unpredictable
    in principle; the theory predicts their statistics.
    """)
    return


@app.cell
def _(mo):
    import os, sys
    import numpy as np

    ROOT = mo.notebook_dir().parent.parent    # src/notebooks -> <repo>
    os.chdir(ROOT); sys.path.insert(0, str(ROOT))
    from src.code.data import feature_cube
    from src.code.geometry import (balanced_code, fantope_leverage, feature_vectors,
                                   leverage, logit_features, predicted_map)
    from src.code.train import resolve_config, train_inline
    from src.code import plot_utils
    import matplotlib.pyplot as plt

    N, M, P, SCALE, BETA = 2048, 64, 0.3, 8.0, 0.5
    D_GRID = [3, 6, 12]
    SEEDS = range(5)
    R = np.sqrt(N * P * (1 - P))                       # the one global activation scale
    ALPHA = SCALE * (1.0 + np.arange(M)) ** -BETA
    Z = feature_cube(N, M, P, scale=SCALE, seed=0, spectrum="power", beta=BETA)[1]["Z"]

    TASK = ["setting=feature_cube", f"structure_kwargs.N={N}", f"structure_kwargs.m={M}",
            f"structure_kwargs.scale={SCALE}", "+structure_kwargs.spectrum=power",
            f"+structure_kwargs.beta={BETA}", f"structure_kwargs.p={P}"]
    CONFIG = resolve_config(TASK)
    COL = plot_utils.FIG.colors

    def trained_sizes(d, seed):
        """One trained run's per-feature vectors in balanced units: (E_meas (m, d), sizes)."""
        model, _ds, ev, *_ = train_inline(
            TASK + ["runner=linear", f"model_kwargs.model_config.embed_dim={d}",
                    "training_kwargs.train_steps=8000", f"random_seed={seed}"],
            name=f"sp_d{d}_s{seed}")
        code = np.asarray(ev.input_code(model, project=False), float)
        H, _ = balanced_code(code, model.readout.detach().cpu().numpy()[:M])
        V, _ = feature_vectors(H, Z)
        E_meas = np.sqrt(R) * V
        return E_meas, (E_meas ** 2).sum(1)

    return (
        ALPHA,
        COL,
        CONFIG,
        D_GRID,
        P,
        SEEDS,
        Z,
        fantope_leverage,
        leverage,
        logit_features,
        np,
        plot_utils,
        plt,
        predicted_map,
        trained_sizes,
    )


@app.cell
def _(
    ALPHA,
    CONFIG,
    D_GRID,
    P,
    Z,
    fantope_leverage,
    logit_features,
    np,
    plot_utils,
    predicted_map,
):
    # the theory pipeline, task-only: predicted map -> measured-style predicted vectors,
    # and the Fantope allocation. Solves take ~15 min; cached in the gitignored figure
    # cache — delete the npz files to recompute.
    import pathlib as _pl

    E_PRED = {}
    H_FANT = {}
    for _d in D_GRID:
        _f = _pl.Path(plot_utils.figure_path(CONFIG, f"predictions_d{_d}", ext="npz"))
        if _f.exists():
            _c = np.load(_f)
            _R, _b, _h = _c["R"], _c["b"], _c["h"]
        else:
            _R, _b = predicted_map(ALPHA, P, _d, n_samples=16384, n_starts=3)
            _h = fantope_leverage(ALPHA, P, _d, n_samples=16384, iters=1200)[0]
            np.savez(_f, R=_R, b=_b, h=_h)
        E_PRED[_d] = logit_features(Z @ _R + _b, Z, _d)
        H_FANT[_d] = _h
    return E_PRED, H_FANT


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 2 — measured sizes: trained vs the theory pipeline

    Dots: measured $\|E_i\|^2$ (5 seeds). Line: the same measurement applied to the
    predicted map's logits — the balanced factorization of $R_{\rm pred}$.

    **Criterion**: the line runs through the dots *including the amplified head and the
    S-shaped rise* — both are properties of the balanced gauge of an asymmetric map, and
    the prediction must reproduce them without any fit to the trained runs. Known residual:
    at $d = 12$ the trained head sits above the line — the non-additive (strong-pair)
    class effect, outside the additive theory (`notes/map_theory.md`).
    """)
    return


@app.cell
def _(D_GRID, SEEDS, trained_sizes):
    RUNS = {d: [trained_sizes(d, s) for s in SEEDS] for d in D_GRID}
    return (RUNS,)


@app.cell
def _(
    ALPHA,
    COL,
    CONFIG,
    D_GRID,
    E_DC,
    E_PAIR_ALL,
    E_PRED,
    RUNS,
    np,
    plot_utils,
    plt,
):
    _fig, _axes = plt.subplots(1, len(D_GRID), figsize=(4.6 * len(D_GRID), 4.4), sharex=True)
    for _ax, _d in zip(_axes, D_GRID):
        for _k, (_E, _s) in enumerate(RUNS[_d]):
            _ax.scatter(ALPHA, _s, s=26, color=COL["blue"], alpha=0.5, edgecolors="none",
                        zorder=3, label="trained (5 seeds)" if _k == 0 else None)
        _sp = (E_PRED[_d] ** 2).sum(1)
        _spp = np.mean([( _E ** 2).sum(1) for _E in E_PAIR_ALL[_d]], axis=0)
        _o = np.argsort(ALPHA)
        _ax.plot(ALPHA[_o], _sp[_o], color=COL["orange"], lw=2.6, zorder=4,
                 label="additive theory")
        _ax.plot(ALPHA[_o], _spp[_o], color=COL["green"], lw=2.2, ls="--", zorder=5,
                 label="additive + pairs oracle (15-restart mean)")
        _sdc = (E_DC[_d] ** 2).sum(1)
        _ax.plot(ALPHA[_o], _sdc[_o], color="0.2", lw=1.8, ls=":", zorder=6,
                 label="certified theory (DC)")
        _ax.set_title(f"$d = {_d}$", fontsize=15)
        _ax.set_xlabel(r"$\alpha_i$")
        plot_utils.FIG.style_axes(_ax)
    _axes[0].set_ylabel(r"$\|E_i\|^2$")
    _axes[0].legend(frameon=False, fontsize=10, loc="upper left")
    _fig.tight_layout()
    plot_utils.save_figure(_fig, plot_utils.figure_path(CONFIG, "fig2_three_regimes"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 2b — dimensional allocation (leverage)

    Gauge-free capacity shares $h_i = P_{ii}$ (hat matrix of the measured vectors,
    invariant to any invertible transform of the code) against the Fantope prediction —
    a certified convex program on the task alone — with the closed-form water-filling formula (`waterfill_h`, dotted) on top of it.

    **Criterion**: the Fantope curve tracks the trained ramp including the sub-1
    shoulder (fractional dedication) and the transition locations; the represented
    count matches to within a couple of features.
    """)
    return


@app.cell
def _(
    ALPHA,
    COL,
    CONFIG,
    D_GRID,
    E_PAIR_ALL,
    H_FANT,
    RUNS,
    leverage,
    np,
    plot_utils,
    plt,
):
    _fig, _axes = plt.subplots(1, len(D_GRID), figsize=(4.6 * len(D_GRID), 4.2))
    for _ax, _d in zip(_axes, D_GRID):
        for _k, (_E, _s) in enumerate(RUNS[_d]):
            _ax.scatter(ALPHA, leverage(_E)[0], s=22, color=COL["blue"], alpha=0.45,
                        edgecolors="none", zorder=3,
                        label="trained (5 seeds)" if _k == 0 else None)
        _o = np.argsort(ALPHA)
        _ax.plot(ALPHA[_o], H_FANT[_d][_o], color=COL["orange"], lw=2.4, zorder=4,
                 label="Fantope (certified program)")
        import pathlib as _pl4
        _fc = np.load(_pl4.Path(plot_utils.figure_path(CONFIG, f"formula_d{_d}", ext="npz")))
        _ax.plot(ALPHA[_o], _fc["h"][_o], color="0.2", lw=1.8, ls=":", zorder=6,
                 label="water-filling formula")
        _H = np.array([leverage(_E)[0] for _E in E_PAIR_ALL[_d]])
        _hp, _hs = _H.mean(0), _H.std(0)
        _ax.plot(ALPHA[_o], _hp[_o], color=COL["green"], lw=2.0, ls="--", zorder=5,
                 label="additive + pairs oracle (15-restart mean $\pm$ sd)")
        _ax.fill_between(ALPHA[_o], (_hp - _hs)[_o], (_hp + _hs)[_o], color=COL["green"],
                         alpha=0.2, lw=0, zorder=2)
        _ax.set_ylim(-0.05, 1.1)
        _ax.set_title(f"$d = {_d}$", fontsize=15)
        _ax.set_xlabel(r"$\alpha_i$")
        plot_utils.FIG.style_axes(_ax)
    _axes[0].set_ylabel(r"leverage $h_i$")
    _axes[0].legend(frameon=False, fontsize=9, loc="center right")
    _fig.tight_layout()
    plot_utils.save_figure(_fig, plot_utils.figure_path(CONFIG, "fig2b_leverage"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 3a/b — geometry without orientation

    **(a)** Eigenvalues of the measured frame operator $\sum_i E_i E_i^\top$ (dots, 5
    seeds) against the theory pipeline's eigenvalues. **(b)** Interference load
    $\sum_{j\neq i}\langle E_i,E_j\rangle^2$ against $\|E_i\|^2$; orange open dots are
    the prediction's features.

    **Criterion**: (a) matching spectra including the head spike; (b) trained dots and
    predicted dots trace the same load curve — in particular the head carries a large
    load (it is *not* orthogonal to the rest; its extra norm cancels interference).
    """)
    return


@app.cell
def _(COL, CONFIG, D_GRID, E_DC, E_PRED, RUNS, np, plot_utils, plt):
    _fig, _axes = plt.subplots(2, len(D_GRID), figsize=(4.4 * len(D_GRID), 7.4))
    for _j, _d in enumerate(D_GRID):
        _Ep = E_PRED[_d]
        _ax = _axes[0, _j]
        _ev_pred = np.sort(np.linalg.eigvalsh(_Ep.T @ _Ep))[::-1]
        _ax.plot(np.arange(1, _d + 1), _ev_pred, "o-", color=COL["orange"], lw=2.2, ms=7,
                 zorder=4, label="additive theory")
        _ev_dc = np.sort(np.linalg.eigvalsh(E_DC[_d].T @ E_DC[_d]))[::-1]
        _ax.plot(np.arange(1, _d + 1), _ev_dc, "x:", color="0.2", lw=1.6, ms=7,
                 zorder=5, label="certified theory (DC)")
        for _k, (_E, _s) in enumerate(RUNS[_d]):
            _ev = np.sort(np.linalg.eigvalsh(_E.T @ _E))[::-1]
            _ax.scatter(np.arange(1, _d + 1), _ev, s=30, color=COL["blue"], alpha=0.6,
                        zorder=3, label="trained (5 seeds)" if _k == 0 else None)
        _ax.set_title(f"$d = {_d}$", fontsize=15)
        _ax.set_xlabel("eigenvalue index")
        _ax.set_xticks(np.arange(1, _d + 1, max(1, _d // 6)))
        plot_utils.FIG.style_axes(_ax)

        _ax = _axes[1, _j]
        _Gp = _Ep @ _Ep.T
        _sp = np.diag(_Gp)
        _lp = (_Gp ** 2).sum(1) - _sp ** 2
        _ax.scatter(_sp, _lp, s=55, facecolor="none", edgecolor=COL["orange"], lw=1.8,
                    zorder=4, label="additive theory")
        _Gdc = E_DC[_d] @ E_DC[_d].T
        _sdc = np.diag(_Gdc)
        _ax.scatter(_sdc, (_Gdc ** 2).sum(1) - _sdc ** 2, s=50, marker="x", color="0.2",
                    zorder=5, label="certified theory (DC)")
        for _k, (_E, _s) in enumerate(RUNS[_d]):
            _G = _E @ _E.T
            _load = (_G ** 2).sum(1) - np.diag(_G) ** 2
            _ax.scatter(_s, _load, s=26, color=COL["blue"], alpha=0.5, edgecolors="none",
                        zorder=3, label="trained (5 seeds)" if _k == 0 else None)
        _ax.set_xlabel(r"measured $\|E_i\|^2$")
        plot_utils.FIG.style_axes(_ax)
    _axes[0, 0].set_ylabel("frame operator eigenvalues")
    _axes[1, 0].set_ylabel(r"$\sum_{j\neq i}\langle E_i,E_j\rangle^2$")
    _axes[0, 0].legend(frameon=False, fontsize=11)
    _axes[1, 0].legend(frameon=False, fontsize=11)
    _fig.tight_layout()
    plot_utils.save_figure(_fig, plot_utils.figure_path(CONFIG, "fig3ab_frame_geometry"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 3c — the vectors themselves ($d = 3$)

    Basis from each model itself: the top feature's unit vector, and its orthogonal
    complement as the shared plane. Columns: trained (seed 0), the additive + pairs oracle
    (one restart — directions are the degenerate coordinate, so they cannot be averaged),
    and the additive theory. Top row: shared-plane arrows colored by $\alpha_i$. Bottom
    row: component along the top feature's direction ($x$) against the magnitude of the
    rest ($y$).

    **Criterion**: arrow count and lengths match trained best in the pairs column (angles
    are the free orientation); the L-shape in the bottom row is shared by trained and pairs,
    while the additive column under-populates the plane. Trained's off-L points are the
    partial pair sector that training finds.
    """)
    return


@app.cell
def _(ALPHA, CONFIG, E_DC, E_PRED, RUNS, np, plot_utils, plt):
    _E = RUNS[3][0][0]                                     # trained, seed 0
    _Ep = E_PRED[3]
    _Edc = E_DC[3]                              # one restart: directions are not averageable
    _fig, _axes = plt.subplots(2, 3, figsize=(13.5, 8.6))
    _cmap = plt.cm.viridis
    _norm = plt.Normalize(0, ALPHA[0])

    def _split(Emat):
        """(component along the top feature's direction, in-plane 2D coords)."""
        _u0 = Emat[0] / np.linalg.norm(Emat[0])
        _para = Emat @ _u0
        _resid = Emat - _para[:, None] * _u0
        _Q = np.linalg.svd(_resid, full_matrices=False)[2][:2]
        return _para, _resid @ _Q.T

    _cols = [(_E, "trained (seed 0)"), (_Edc, "certified theory (DC)"),
             (_Ep, "additive theory")]
    for _c, (_Emat, _title) in enumerate(_cols):
        _ax = _axes[0, _c]
        _, _xy = _split(_Emat)
        _rep = np.flatnonzero((_Emat ** 2).sum(1) > 0.05)[1:]   # shared, visibly represented
        for _i in _rep:
            _ax.annotate("", xy=_xy[_i], xytext=(0, 0),
                         arrowprops=dict(arrowstyle="->", lw=2.0, color=_cmap(_norm(ALPHA[_i]))))
        _lim = np.abs(_xy[_rep]).max() * 1.2
        _ax.set_xlim(-_lim, _lim); _ax.set_ylim(-_lim, _lim)
        _ax.set_aspect("equal")
        _ax.set_title(f"shared plane — {_title}", fontsize=13)
        plot_utils.FIG.clean(_ax)

        _ax = _axes[1, _c]
        _para, _xy = _split(_Emat)
        _sc = _ax.scatter(np.abs(_para), np.linalg.norm(_xy, axis=1), s=55,
                          c=ALPHA, cmap=_cmap, norm=_norm, zorder=3, edgecolors="none")
        _ax.axhline(0, color="0.8", lw=1); _ax.axvline(0, color="0.8", lw=1)
        _ax.set_xlabel("component on the top feature's direction")
        _ax.set_title(_title, fontsize=13)
        plot_utils.FIG.style_axes(_ax)
    _axes[1, 0].set_ylabel("magnitude of the rest")
    _fig.colorbar(_sc, ax=_axes[:, 2], label=r"$\alpha_i$", shrink=0.8)
    plot_utils.save_figure(_fig, plot_utils.figure_path(CONFIG, "fig3c_vectors_d3"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 3d — Gram fingerprint at every budget

    $\langle E_i, E_j\rangle$ over the top 16 features: the 5-seed mean of the trained
    Grams beside the additive-class optimum (the theory pipeline), per budget.

    **Criterion**: individual off-diagonals are seed noise (seed-to-seed relative
    distance $\approx 1.4$), so only statistics are testable: matching diagonals
    (amplified head included), all-negative haze in the represented block, and the
    haze concentrated toward the threshold features. Known residual: the seed-mean haze
    is $\approx 2\times$ stronger than the additive optimum's — the non-additive
    (strong-pair) class component, `notes/map_theory.md`.
    """)
    return


@app.cell
def _(CONFIG, D_GRID, E_DC, E_PAIR_ALL, E_PRED, RUNS, np, plot_utils, plt):
    _k = 16
    _fig, _axes = plt.subplots(len(D_GRID), 4, figsize=(16.4, 4.0 * len(D_GRID)))
    for _j, _d in enumerate(D_GRID):
        _Gm = np.mean([_E[:_k] @ _E[:_k].T for _E, _s in RUNS[_d]], axis=0)
        _Gpair = np.mean([_E[:_k] @ _E[:_k].T for _E in E_PAIR_ALL[_d]], axis=0)
        _Gdc = E_DC[_d][:_k] @ E_DC[_d][:_k].T
        _Gp = E_PRED[_d][:_k] @ E_PRED[_d][:_k].T
        _panels = [(_Gm, "trained (5-seed mean)"),
                   (_Gpair, "additive + pairs (15-restart mean)"),
                   (_Gdc, "certified theory (DC)"), (_Gp, "additive theory")]
        _lim = max(np.abs(_G).max() for _G, _t in _panels)
        for _i, (_G, _title) in enumerate(_panels):
            _ax = _axes[_j, _i]
            _im = _ax.imshow(_G, cmap=plot_utils.FIG.diverging_cmap, vmin=-_lim, vmax=_lim)
            _ax.set_title(f"$d = {_d}$ — {_title}", fontsize=11)
            _ax.set_xticks([]); _ax.set_yticks([])
        _fig.colorbar(_im, ax=_axes[_j, :], shrink=0.8, label=r"$\langle E_i, E_j\rangle$")
    plot_utils.save_figure(_fig, plot_utils.figure_path(CONFIG, "fig3d_grams"))
    return


@app.cell
def _(ALPHA, CONFIG, D_GRID, Z, logit_features, np, plot_utils):
    # additive-class oracle on the training words (the exact 2048-word sample the
    # networks see), tied-direction + kappa class; cached beside the predictions.
    import pathlib as _pl2
    import torch as _torch
    from scipy.optimize import minimize as _minimize

    _torch.set_default_dtype(_torch.float64)
    _torch.set_num_threads(1)
    E_ORC = {}
    for _d in D_GRID:
        _f = _pl2.Path(plot_utils.figure_path(CONFIG, f"train_oracle_d{_d}", ext="npz"))
        if _f.exists():
            _c = np.load(_f)
            _R, _b = _c["R"], _c["b"]
        else:
            _Zt = _torch.tensor(Z)
            _Pt = _torch.softmax(_Zt * _torch.tensor(ALPHA), 1)
            _m = len(ALPHA)

            def _fg(_x, _d=_d, _Zt=_Zt, _Pt=_Pt, _m=_m):
                _t = _torch.tensor(_x, requires_grad=True)
                _W = _t[:_m * _d].reshape(_m, _d)
                _k = _t[_m * _d:_m * _d + _m]
                _bb = _t[_m * _d + _m:]
                _L = (_Zt @ _W) @ (_k[:, None] * _W).T + _bb
                _J = (_torch.logsumexp(_L, 1) - (_Pt * _L).sum(1)).mean()
                _J.backward()
                return float(_J.detach()), _t.grad.numpy()

            _rng = np.random.default_rng(0)
            _best = None
            for _s in range(2):
                _x0 = np.concatenate([0.3 * _rng.normal(size=_m * _d), np.ones(_m), np.zeros(_m)])
                _r = _minimize(_fg, _x0, jac=True, method="L-BFGS-B",
                               options=dict(maxiter=6000, maxfun=12000, ftol=1e-13, gtol=1e-9))
                if _best is None or _r.fun < _best.fun:
                    _best = _r
            _W = _best.x[:_m * _d].reshape(_m, _d)
            _kap = _best.x[_m * _d:_m * _d + _m]
            _R = _W @ (_kap[:, None] * _W).T
            _b = _best.x[_m * _d + _m:]
            np.savez(_f, R=_R, b=_b)
        E_ORC[_d] = logit_features(Z @ _R + _b, Z, _d)
    return


@app.cell
def _(ALPHA, CONFIG, D_GRID, Z, logit_features, np, plot_utils):
    # additive + strong-pair oracle on the training words (top-18 pair interactions,
    # code stays d-dimensional): the class that contains the trained nets. Five restarts;
    # the pair allocation is tie-degenerate at the threshold; figures show the
    # 15-restart mean with a std band.
    import pathlib as _pl3
    import torch as _torch2
    from scipy.optimize import minimize as _minimize2

    _torch2.set_default_dtype(_torch2.float64)
    _torch2.set_num_threads(1)
    _PIDX = [(_a, _b) for _a in range(18) for _b in range(_a + 1, 18)]
    _U = np.column_stack([Z[:, _a] * Z[:, _b] for _a, _b in _PIDX])
    _U = (_U - _U.mean(0)) / _U.std(0)
    E_PAIR_ALL = {}
    for _d in D_GRID:
        _sols = []
        for _seed in range(15):
            _f = _pl3.Path(plot_utils.figure_path(CONFIG, f"pairs_oracle_d{_d}_s{_seed}", ext="npz"))
            if _f.exists():
                _L = np.load(_f)["L"]
            else:
                _Zt = _torch2.tensor(Z)
                _Ut = _torch2.tensor(_U)
                _Pt = _torch2.softmax(_Zt * _torch2.tensor(ALPHA), 1)
                _m, _np_ = len(ALPHA), len(_PIDX)

                def _fg(_x, _d=_d, _Zt=_Zt, _Ut=_Ut, _Pt=_Pt, _m=_m, _np_=_np_):
                    _t = _torch2.tensor(_x, requires_grad=True)
                    _A = _t[:_m * _d].reshape(_m, _d)
                    _Cp = _t[_m * _d:_m * _d + _np_ * _d].reshape(_np_, _d)
                    _B = _t[_m * _d + _np_ * _d:_m * _d + _np_ * _d + _m * _d].reshape(_m, _d)
                    _bb = _t[_m * _d + _np_ * _d + _m * _d:]
                    _Lg = (_Zt @ _A + _Ut @ _Cp) @ _B.T + _bb
                    _J = (_torch2.logsumexp(_Lg, 1) - (_Pt * _Lg).sum(1)).mean()
                    _J.backward()
                    return float(_J.detach()), _t.grad.numpy()

                _rng = np.random.default_rng(_seed)
                _x0 = np.concatenate([0.3 * _rng.normal(size=_m * _d),
                                      0.05 * _rng.normal(size=_np_ * _d),
                                      0.3 * _rng.normal(size=_m * _d), np.zeros(_m)])
                _r = _minimize2(_fg, _x0, jac=True, method="L-BFGS-B",
                                options=dict(maxiter=6000, maxfun=12000, ftol=1e-13, gtol=1e-9))
                _t = _r.x
                _A = _t[:_m * _d].reshape(_m, _d)
                _Cp = _t[_m * _d:_m * _d + _np_ * _d].reshape(_np_, _d)
                _B = _t[_m * _d + _np_ * _d:_m * _d + _np_ * _d + _m * _d].reshape(_m, _d)
                _L = (Z @ _A + _U @ _Cp) @ _B.T + _t[_m * _d + _np_ * _d + _m * _d:]
                np.savez(_f, L=_L)
            _sols.append(logit_features(_L, Z, _d))
        E_PAIR_ALL[_d] = _sols
        _S = np.array([( _E ** 2).sum(1) for _E in _sols])
        print(f"d={_d}: size-curve restart spread (mean over features of std) = "
              f"{_S.std(0).mean():.3f}")
    return (E_PAIR_ALL,)


@app.cell
def _(CONFIG, D_GRID, Z, logit_features, np, plot_utils):
    # the certified theory predictor (dc_map, truncated-nuclear DC + convex polish),
    # loaded from the figure cache; measured through the same pipeline as everything else.
    import pathlib as _pl5
    E_DC = {}
    for _d in D_GRID:
        _L = np.load(_pl5.Path(plot_utils.figure_path(CONFIG, f"dc_d{_d}", ext="npz")))["L"]
        E_DC[_d] = logit_features(_L, Z, _d)

    return (E_DC,)


if __name__ == "__main__":
    app.run()
