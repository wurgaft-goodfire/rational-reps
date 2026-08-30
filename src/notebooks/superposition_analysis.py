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
    # Superposition: the predicted geometry of a rank-limited softmax

    $m$ binary features $z_i \sim \mathrm{Bernoulli}(p)$, target
    $p_\star(x{=}i \mid z) \propto \exp(\alpha_i z_i)$ with $\alpha_i = 8\,(i{+}1)^{-1/2}$.
    A rank-$d$ model embeds each feature as $E_i \in \mathbb{R}^d$. The theory
    (`notes/superposition.tex`) says the learned representation minimizes

    $$\mathcal J(E) = \tfrac12\sum_i\big(\|E_i\|^2-\alpha_i\big)^2
      + \tfrac{p}{2}\sum_{i\neq j}\langle E_i,E_j\rangle^2,$$

    whose solution is set by one capacity level $\alpha_\star$ spending the budget exactly:

    - $\alpha_i > \alpha_\star$: **dedicated** orthogonal direction, $\|E_i\|^2 = \alpha_i$;
    - $p\alpha_\star < \alpha_i \le \alpha_\star$: **shared** tight frame,
      $\|E_i\|^2 = (\alpha_i - p\alpha_\star)/(1-p)$;
    - $\alpha_i \le p\alpha_\star$: **dropped**, $E_i = 0$.

    Trained sizes are measured by gauge-fixing the model (`balanced_code`), regressing the
    code on the features (`feature_vectors`), and rescaling to balanced units:
    $\|E_i\|^2 = \sqrt{Np(1-p)}\,\|V_i\|^2$.
    """)
    return


@app.cell
def _(mo):
    import os, sys
    import numpy as np

    ROOT = mo.notebook_dir().parent.parent    # src/notebooks -> <repo>
    os.chdir(ROOT); sys.path.insert(0, str(ROOT))
    from src.code.data import feature_cube
    from src.code.geometry import (balanced_code, feature_vectors, fitted_features,
                                   leverage, predicted_features)
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

    PRED = {d: predicted_features(ALPHA, P, d) for d in D_GRID}    # (W, sizes, own, astar)

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
        PRED,
        SEEDS,
        fitted_features,
        leverage,
        np,
        plot_utils,
        plt,
        trained_sizes,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 1 — the closed form solves the objective (no training)

    Left row: $\|E_i\|^2$ against $\alpha_i$; line = closed form, dots = L-BFGS run directly
    on $\mathcal J$ from random starts. Right row: eigenvalues of $E^\top E$; line = predicted
    $\{\alpha_i\ \text{dedicated}\} \cup \{\alpha_\star\ \text{repeated}\}$, open dots = L-BFGS.

    **Criterion**: dots on the line everywhere, spike-then-flat spectrum. A gap falsifies the
    implementation, not the theory.
    """)
    return


@app.cell
def _(
    ALPHA,
    COL,
    CONFIG,
    D_GRID,
    P,
    PRED,
    fitted_features,
    np,
    plot_utils,
    plt,
):
    _fig, _axes = plt.subplots(2, len(D_GRID), figsize=(4.4 * len(D_GRID), 7.2))
    for _j, _d in enumerate(D_GRID):
        _W, _sizes, _own, _astar = PRED[_d]
        _Wf = fitted_features(ALPHA, P, _d, n_starts=4)
        _sf = (_Wf ** 2).sum(1)

        _ax = _axes[0, _j]
        _ax.plot(ALPHA, _sizes, color=COL["orange"], lw=2.4, zorder=3, label="closed form")
        _ax.scatter(ALPHA, _sf, s=30, color=COL["blue"], zorder=4, label="L-BFGS on $\\mathcal{J}$")
        for _x, _ls in [(_astar, "--"), (P * _astar, ":")]:
            _ax.axvline(_x, color="0.55", lw=1.2, ls=_ls, zorder=2)
        _ax.text(_astar, _ax.get_ylim()[1] * 0.9, r" $\alpha_\star$", color="0.4", fontsize=12)
        _ax.text(P * _astar, _ax.get_ylim()[1] * 0.9, r" $p\alpha_\star$", color="0.4", fontsize=12)
        _ax.set_title(f"$d = {_d}$", fontsize=15)
        _ax.set_xlabel(r"$\alpha_i$")
        plot_utils.FIG.style_axes(_ax)

        _ax = _axes[1, _j]
        _ev_pred = np.sort(np.concatenate([ALPHA[_own], np.full(_d - _own.sum(), _astar)]))[::-1]
        _ev_fit = np.sort(np.linalg.eigvalsh(_Wf.T @ _Wf))[::-1]
        _ax.plot(np.arange(1, _d + 1), _ev_pred, "o-", color=COL["orange"], lw=2.2, ms=7,
                 zorder=3, label="predicted")
        _ax.scatter(np.arange(1, _d + 1), _ev_fit, s=70, facecolor="none",
                    edgecolor=COL["blue"], lw=2, zorder=4, label="L-BFGS")
        _ax.set_xlabel("eigenvalue index")
        _ax.set_xticks(np.arange(1, _d + 1, max(1, _d // 6)))
        plot_utils.FIG.style_axes(_ax)
    _axes[0, 0].set_ylabel(r"$\|E_i\|^2$")
    _axes[1, 0].set_ylabel(r"eigenvalues of $E^\top E$")
    _axes[0, 0].legend(frameon=False, fontsize=11)
    _axes[1, 0].legend(frameon=False, fontsize=11)
    _fig.tight_layout()
    plot_utils.save_figure(_fig, plot_utils.figure_path(CONFIG, "fig1_theory_check"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 2 — the three regimes in a trained network

    Dots: measured $\|E_i\|^2$ (5 seeds). Line: the piecewise prediction. Verticals at
    $p\alpha_\star$ (representation threshold) and $\alpha_\star$ (dedication threshold).

    **Criterion**: measured sizes drop to zero below $p\alpha_\star$ and rise with
    $\alpha_i$; the represented count is bracketed by the prediction.

    **Known deviation (understood)**: the head sits *above* $\alpha_1$ and the count runs
    low. Both are properties of the balanced gauge, not of the input-output map: the
    gauge-invariant map diagonal (`logit_map`) is pinned at $\bar\alpha_i$ exactly, and
    the excess measured norm is the balanced factorization of an asymmetric map (reader
    $= \kappa_i \times$ writer). See `notes/map_theory.md`.
    """)
    return


@app.cell
def _(D_GRID, SEEDS, trained_sizes):
    RUNS = {d: [trained_sizes(d, s) for s in SEEDS] for d in D_GRID}
    return (RUNS,)


@app.cell
def _(ALPHA, COL, CONFIG, D_GRID, P, PRED, RUNS, np, plot_utils, plt):
    _fig, _axes = plt.subplots(1, len(D_GRID), figsize=(4.6 * len(D_GRID), 4.4), sharex=True)
    for _ax, _d in zip(_axes, D_GRID):
        _W, _sizes, _own, _astar = PRED[_d]
        for _k, (_E, _s) in enumerate(RUNS[_d]):
            _ax.scatter(ALPHA, _s, s=26, color=COL["blue"], alpha=0.5, edgecolors="none",
                        zorder=3, label="trained (5 seeds)" if _k == 0 else None)
        _o = np.argsort(ALPHA)
        _ax.plot(ALPHA[_o], _sizes[_o], color=COL["orange"], lw=2.6, zorder=4, label="predicted")
        _ax.axvline(_astar, color="0.55", lw=1.2, ls="--", zorder=2)
        _ax.axvline(P * _astar, color="0.55", lw=1.2, ls=":", zorder=2)
        _ax.text(_astar, _ax.get_ylim()[1] * 0.88, r" $\alpha_\star$", color="0.4", fontsize=13)
        _ax.text(P * _astar, _ax.get_ylim()[1] * 0.88, r" $p\alpha_\star$", color="0.4", fontsize=13)
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

    Gauge-free capacity shares: $h_i = P_{ii}$ with $P$ the hat matrix of the measured
    feature vectors ($0 \le h_i \le 1$, $\sum_i h_i = d$; invariant to any invertible
    transform of the code). Overlay: the closed form
    $h_i = [(\alpha_i/\alpha_\star - p)/(1-p)]_0^1$.

    **Criterion**: a $0 \to$ fractional $\to 1$ ramp tracking the prediction (corr
    $\approx 0.95$), transitions shifted modestly right; full dedication ($h = 1$) only
    approached as $d$ grows. A head feature far below the ramp falsifies the allocation
    theory.
    """)
    return


@app.cell
def _(ALPHA, CONFIG, D_GRID, P, PRED, RUNS, leverage, np, plot_utils, plt):
    _fig, _axes = plt.subplots(1, len(D_GRID), figsize=(4.6 * len(D_GRID), 4.2))
    for _ax, _d in zip(_axes, D_GRID):
        _astar = PRED[_d][3]
        _hp = np.clip((ALPHA / _astar - P) / (1 - P), 0.0, 1.0)
        for _k, (_E, _s) in enumerate(RUNS[_d]):
            _ax.scatter(ALPHA, leverage(_E)[0], s=22, color=plot_utils.FIG.colors["blue"],
                        alpha=0.45, edgecolors="none", zorder=3,
                        label="trained (5 seeds)" if _k == 0 else None)
        _o = np.argsort(ALPHA)
        _ax.plot(ALPHA[_o], _hp[_o], color=plot_utils.FIG.colors["orange"], lw=2.4,
                 zorder=4, label="predicted")
        _ax.axvline(_astar, color="0.55", lw=1.2, ls="--", zorder=2)
        _ax.axvline(P * _astar, color="0.55", lw=1.2, ls=":", zorder=2)
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

    **(a)** Eigenvalues of the measured frame operator $\sum_i E_iE_i^\top$ (dots, 5 seeds)
    against the predicted $\{\alpha_i\ \text{dedicated}\} \cup \{\alpha_\star\ \text{flat}\}$.
    **(b)** Each feature's interference load $\sum_{j\neq i}\langle E_i,E_j\rangle^2$ against
    its measured $\|E_i\|^2$; the prediction is the parabola
    $\alpha_\star\|E_i\|^2 - \|E_i\|^4$, zero at both ends.

    **Criterion**: (a) spike-plus-flat-block at the predicted levels; (b) dots on the parabola,
    and dedicated features at zero load — orthogonal to everything, not merely large.
    """)
    return


@app.cell
def _(ALPHA, COL, CONFIG, D_GRID, PRED, RUNS, np, plot_utils, plt):
    _fig, _axes = plt.subplots(2, len(D_GRID), figsize=(4.4 * len(D_GRID), 7.4))
    for _j, _d in enumerate(D_GRID):
        _Wp, _sizes, _own, _astar = PRED[_d]
        _ax = _axes[0, _j]
        _ev_pred = np.sort(np.concatenate([ALPHA[_own], np.full(_d - _own.sum(), _astar)]))[::-1]
        _ax.plot(np.arange(1, _d + 1), _ev_pred, "o-", color=COL["orange"], lw=2.2, ms=7,
                 zorder=4, label="predicted")
        for _k, (_E, _s) in enumerate(RUNS[_d]):
            _ev = np.sort(np.linalg.eigvalsh(_E.T @ _E))[::-1]
            _ax.scatter(np.arange(1, _d + 1), _ev, s=30, color=COL["blue"], alpha=0.6,
                        zorder=3, label="trained (5 seeds)" if _k == 0 else None)
        _ax.set_title(f"$d = {_d}$", fontsize=15)
        _ax.set_xlabel("eigenvalue index")
        _ax.set_xticks(np.arange(1, _d + 1, max(1, _d // 6)))
        plot_utils.FIG.style_axes(_ax)

        _ax = _axes[1, _j]
        _grid = np.linspace(0, _astar, 200)
        _ax.plot(_grid, _astar * _grid - _grid ** 2, color=COL["orange"], lw=2.4, zorder=4,
                 label=r"$\alpha_\star\|E_i\|^2 - \|E_i\|^4$")
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

    Basis from the trained model itself: the dedicated direction is the top feature's unit
    vector; the shared plane is its orthogonal complement. Top row: shared-plane arrows,
    trained (seed 0) beside the predicted realization, colored by $\alpha_i$. Bottom row: each
    feature's component along the dedicated direction ($x$) against the magnitude of the rest
    ($y$).

    **Criterion**: arrow lengths match left-right (angles are free — the unpredicted
    orientation); bottom row forms an L, with no shared feature leaking onto the $x$ axis.
    """)
    return


@app.cell
def _(ALPHA, CONFIG, PRED, RUNS, np, plot_utils, plt):
    _Wp, _sizes_p, _own_p, _astar = PRED[3]
    _E = RUNS[3][0][0]                                     # trained, seed 0
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
                               (_axes[0, 1], _Wp, "predicted")]:
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
                               (_axes[1, 1], _Wp, "predicted")]:
        _para, _xy = _split(_Emat)
        _sc = _ax.scatter(np.abs(_para), np.linalg.norm(_xy, axis=1), s=55,
                          c=ALPHA, cmap=_cmap, norm=_norm, zorder=3, edgecolors="none")
        _ax.axhline(0, color="0.8", lw=1); _ax.axvline(0, color="0.8", lw=1)
        _ax.set_xlabel("component on the dedicated direction")
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

    $\langle E_i, E_j\rangle$ over the top 16 features, trained (seed 0) beside the predicted
    realization, per budget.

    **Criterion**: diagonal = the three-regime sizes; dedicated rows blank off-diagonal; the
    shared block a small even haze. Individual shared off-diagonals need not match — only the
    structure.
    """)
    return


@app.cell
def _(CONFIG, D_GRID, PRED, RUNS, np, plot_utils, plt):
    _k = 16
    _fig, _axes = plt.subplots(len(D_GRID), 2, figsize=(8.4, 4.0 * len(D_GRID)))
    for _j, _d in enumerate(D_GRID):
        _pairs = [(RUNS[_d][0][0], "trained (seed 0)"), (PRED[_d][0], "predicted")]
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
