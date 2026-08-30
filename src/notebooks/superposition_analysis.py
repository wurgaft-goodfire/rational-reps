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
    # Superposition: the map, the gauge, and the measured geometry

    $m$ binary features $z_i \sim \mathrm{Bernoulli}(p)$, target
    $p_\star(x{=}i \mid z) \propto \exp(\alpha_i z_i)$ with $\alpha_i = 8\,(i{+}1)^{-1/2}$,
    rank-$d$ additive model. The theory (`notes/map_theory.md`) predicts in layers:

    1. **The map** $R$ (feature $\to$ output, gauge-invariant): diagonal pinned at the
       centred target, $R_{ii} = \alpha_i(1 - 1/m)$, off-diagonals negative and
       asymmetric (reader $= \kappa_i \times$ writer, directions tied).
    2. **Allocation**: dimensional capacity $h_i$ from the convex Fantope program.
    3. **Measured geometry**: the balanced-SVD factorization of $R$ — this, not the map,
       is where the "amplified" head norms come from.

    The predicted map is `geometry.predicted_map` (task-only, no training data); measured
    feature vectors of any logit matrix come from `geometry.logit_features`, applied
    identically to trained models and to the prediction.
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
    ALPHA_C = ALPHA * (1 - 1.0 / M)                    # centred target diagonal
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
        ALPHA_C,
        COL,
        CONFIG,
        D_GRID,
        M,
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
def _(ALPHA, D_GRID, P, Z, fantope_leverage, logit_features, np, predicted_map):
    # the theory pipeline, task-only: predicted map -> measured-style predicted vectors,
    # and the Fantope allocation. (A few minutes of deterministic optimization per budget.)
    PRED_MAP = {}
    E_PRED = {}
    H_FANT = {}
    for _d in D_GRID:
        _R, _b = predicted_map(ALPHA, P, _d, n_samples=16384, n_starts=3)
        PRED_MAP[_d] = (_R, _b)
        E_PRED[_d] = logit_features(Z @ _R + _b, Z, _d)
        H_FANT[_d] = fantope_leverage(ALPHA, P, _d, n_samples=16384, iters=1200)[0]
    return E_PRED, H_FANT, PRED_MAP


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 1 — the map anchor (no training)

    Top: diagonal of the predicted centred map against $\alpha_i(1 - 1/m)$; the identity
    line is the theory's exact anchor. Bottom: each represented feature's off-diagonal
    row sum $S_i = \sum_{j \ne i} R_{ij}$.

    **Criterion**: represented diagonals on the identity line (sub-percent away from the
    representation threshold); every represented row sum strictly negative. A diagonal
    far above the line would mean the map itself amplifies — the old misreading.
    """)
    return


@app.cell
def _(ALPHA, ALPHA_C, COL, CONFIG, D_GRID, PRED_MAP, np, plot_utils, plt):
    _fig, _axes = plt.subplots(2, len(D_GRID), figsize=(4.4 * len(D_GRID), 7.2))
    for _j, _d in enumerate(D_GRID):
        _R = PRED_MAP[_d][0]
        _Rc = _R - _R.mean(1, keepdims=True)
        _dg = np.diag(_Rc)
        _rep = _dg > 0.15

        _ax = _axes[0, _j]
        _ax.plot([0, ALPHA_C[0]], [0, ALPHA_C[0]], color="0.55", lw=1.2, ls="--", zorder=2)
        _ax.scatter(ALPHA_C, _dg, s=30, color=COL["blue"], zorder=3, edgecolors="none")
        _ax.set_title(f"$d = {_d}$", fontsize=15)
        _ax.set_xlabel(r"$\alpha_i (1 - 1/m)$")
        plot_utils.FIG.style_axes(_ax)

        _ax = _axes[1, _j]
        _S = (_Rc - np.diag(_dg)).sum(1)
        _ax.axhline(0, color="0.75", lw=1)
        _ax.scatter(ALPHA[_rep], _S[_rep], s=30, color=COL["blue"], zorder=3,
                    edgecolors="none")
        _ax.set_xlabel(r"$\alpha_i$ (represented)")
        plot_utils.FIG.style_axes(_ax)
    _axes[0, 0].set_ylabel(r"predicted $R_{ii}$")
    _axes[1, 0].set_ylabel(r"row sum $S_i$")
    _fig.tight_layout()
    plot_utils.save_figure(_fig, plot_utils.figure_path(CONFIG, "fig1_theory_check"))
    return


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
def _(ALPHA, COL, CONFIG, D_GRID, E_PRED, RUNS, np, plot_utils, plt):
    _fig, _axes = plt.subplots(1, len(D_GRID), figsize=(4.6 * len(D_GRID), 4.4), sharex=True)
    for _ax, _d in zip(_axes, D_GRID):
        for _k, (_E, _s) in enumerate(RUNS[_d]):
            _ax.scatter(ALPHA, _s, s=26, color=COL["blue"], alpha=0.5, edgecolors="none",
                        zorder=3, label="trained (5 seeds)" if _k == 0 else None)
        _sp = (E_PRED[_d] ** 2).sum(1)
        _o = np.argsort(ALPHA)
        _ax.plot(ALPHA[_o], _sp[_o], color=COL["orange"], lw=2.6, zorder=4,
                 label="theory pipeline")
        _ax.set_title(f"$d = {_d}$", fontsize=15)
        _ax.set_xlabel(r"$\alpha_i$")
        plot_utils.FIG.style_axes(_ax)
    _axes[0].set_ylabel(r"$\|E_i\|^2$")
    _axes[0].legend(frameon=False, fontsize=12, loc="upper left")
    _fig.tight_layout()
    plot_utils.save_figure(_fig, plot_utils.figure_path(CONFIG, "fig2_three_regimes"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 2b — dimensional allocation (leverage)

    Gauge-free capacity shares $h_i = P_{ii}$ (hat matrix of the measured vectors,
    invariant to any invertible transform of the code) against the Fantope prediction —
    a certified convex program on the task alone.

    **Criterion**: the Fantope curve tracks the trained ramp including the sub-1
    shoulder (fractional dedication) and the transition locations; the represented
    count matches to within a couple of features.
    """)
    return


@app.cell
def _(ALPHA, COL, CONFIG, D_GRID, H_FANT, RUNS, leverage, np, plot_utils, plt):
    _fig, _axes = plt.subplots(1, len(D_GRID), figsize=(4.6 * len(D_GRID), 4.2))
    for _ax, _d in zip(_axes, D_GRID):
        for _k, (_E, _s) in enumerate(RUNS[_d]):
            _ax.scatter(ALPHA, leverage(_E)[0], s=22, color=COL["blue"], alpha=0.45,
                        edgecolors="none", zorder=3,
                        label="trained (5 seeds)" if _k == 0 else None)
        _o = np.argsort(ALPHA)
        _ax.plot(ALPHA[_o], H_FANT[_d][_o], color=COL["orange"], lw=2.4, zorder=4,
                 label="Fantope")
        _ax.set_ylim(-0.05, 1.1)
        _ax.set_title(f"$d = {_d}$", fontsize=15)
        _ax.set_xlabel(r"$\alpha_i$")
        plot_utils.FIG.style_axes(_ax)
    _axes[0].set_ylabel(r"leverage $h_i$")
    _axes[0].legend(frameon=False, fontsize=11, loc="center right")
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
def _(CONFIG, D_GRID, E_PRED, RUNS, np, plot_utils, plt, COL):
    _fig, _axes = plt.subplots(2, len(D_GRID), figsize=(4.4 * len(D_GRID), 7.4))
    for _j, _d in enumerate(D_GRID):
        _Ep = E_PRED[_d]
        _ax = _axes[0, _j]
        _ev_pred = np.sort(np.linalg.eigvalsh(_Ep.T @ _Ep))[::-1]
        _ax.plot(np.arange(1, _d + 1), _ev_pred, "o-", color=COL["orange"], lw=2.2, ms=7,
                 zorder=4, label="theory pipeline")
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
                    zorder=4, label="theory pipeline")
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
    complement as the shared plane. Top row: shared-plane arrows, trained (seed 0) beside
    the theory pipeline, colored by $\alpha_i$. Bottom row: component along the top
    feature's direction ($x$) against the magnitude of the rest ($y$).

    **Criterion**: arrow lengths match left-right (angles are the free orientation); in
    the bottom row both panels show the same shape — including the head's *nonzero*
    in-plane component (the anti-aligned interference budget).
    """)
    return


@app.cell
def _(ALPHA, CONFIG, E_PRED, RUNS, np, plot_utils, plt):
    _E = RUNS[3][0][0]                                     # trained, seed 0
    _Ep = E_PRED[3]
    _fig, _axes = plt.subplots(2, 2, figsize=(9.2, 8.6))
    _cmap = plt.cm.viridis
    _norm = plt.Normalize(0, ALPHA[0])

    def _split(Emat):
        """(component along the top feature's direction, in-plane 2D coords)."""
        _u0 = Emat[0] / np.linalg.norm(Emat[0])
        _para = Emat @ _u0
        _resid = Emat - _para[:, None] * _u0
        _Q = np.linalg.svd(_resid, full_matrices=False)[2][:2]
        return _para, _resid @ _Q.T

    for _ax, _Emat, _title in [(_axes[0, 0], _E, "trained (seed 0)"),
                               (_axes[0, 1], _Ep, "theory pipeline")]:
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

    for _ax, _Emat, _title in [(_axes[1, 0], _E, "trained (seed 0)"),
                               (_axes[1, 1], _Ep, "theory pipeline")]:
        _para, _xy = _split(_Emat)
        _sc = _ax.scatter(np.abs(_para), np.linalg.norm(_xy, axis=1), s=55,
                          c=ALPHA, cmap=_cmap, norm=_norm, zorder=3, edgecolors="none")
        _ax.axhline(0, color="0.8", lw=1); _ax.axvline(0, color="0.8", lw=1)
        _ax.set_xlabel("component on the top feature's direction")
        _ax.set_title(_title, fontsize=13)
        plot_utils.FIG.style_axes(_ax)
    _axes[1, 0].set_ylabel("magnitude of the rest")
    _fig.colorbar(_sc, ax=_axes[:, 1], label=r"$\alpha_i$", shrink=0.8)
    plot_utils.save_figure(_fig, plot_utils.figure_path(CONFIG, "fig3c_vectors_d3"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 3d — Gram fingerprint at every budget

    $\langle E_i, E_j\rangle$ over the top 16 features, trained (seed 0) beside the
    theory pipeline, per budget.

    **Criterion**: matching diagonals (including the amplified head), negative haze in
    the represented block on both sides. Individual off-diagonals need not match — only
    the structure.
    """)
    return


@app.cell
def _(CONFIG, D_GRID, E_PRED, RUNS, np, plot_utils, plt):
    _k = 16
    _fig, _axes = plt.subplots(len(D_GRID), 2, figsize=(8.4, 4.0 * len(D_GRID)))
    for _j, _d in enumerate(D_GRID):
        _pairs = [(RUNS[_d][0][0], "trained (seed 0)"), (E_PRED[_d], "theory pipeline")]
        _lim = max(np.abs(_E[:_k] @ _E[:_k].T).max() for _E, _ in _pairs)
        for _i, (_E, _title) in enumerate(_pairs):
            _ax = _axes[_j, _i]
            _im = _ax.imshow(_E[:_k] @ _E[:_k].T, cmap=plot_utils.FIG.diverging_cmap,
                             vmin=-_lim, vmax=_lim)
            _ax.set_title(f"$d = {_d}$ — {_title}", fontsize=12)
            _ax.set_xticks([]); _ax.set_yticks([])
        _fig.colorbar(_im, ax=_axes[_j, :], shrink=0.8, label=r"$\langle E_i, E_j\rangle$")
    plot_utils.save_figure(_fig, plot_utils.figure_path(CONFIG, "fig3d_grams"))
    return


if __name__ == "__main__":
    app.run()
