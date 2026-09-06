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
    predicted frame. Every model is reduced to its logit table over states (trained: code
    table times readout; theory: $(z E^\star + \phi(z) E_2 + e_0)\,D$), factored at rank $d$
    into a balanced code and readout (`predicted_geometry`), and measured by the first-order
    vectors $E_i$ of the least-squares fit $h(z) \approx \bar h + \sum_i E_i z_i$ to that code;
    correlated cells use the predictive modes $z\,C^{-1/2}$. Five runs per cell, task and
    training seeds coupled. Each figure has its own setting selectors; results and open items
    in `notes/superposition_results.md`.
    """)
    return


@app.cell
def _(mo):
    import os, sys
    import numpy as np

    ROOT = mo.notebook_dir().parent.parent    # src/notebooks -> <repo>
    os.chdir(ROOT); sys.path.insert(0, str(ROOT))
    from src.code.geometry import factor_logcond, pair_features, theory_table
    from src.code.train import resolve_config
    from src.code import plot_utils
    from src.scripts import sweep_superposition as sw
    import matplotlib.pyplot as plt

    N, M = sw.N, sw.M
    D_GRID, SEEDS = sw.GRID["d"], sw.GRID["seed"]
    CONFIG = resolve_config(sw.task(0.3, 0.5, 0.0))       # the figure cache lives under the reference cell
    COL = plot_utils.FIG.colors
    RUNNERS = {"linear": ("trained", COL["blue"]), "paired_features": ("pairs model", COL["purple"]),
               "additive_features": ("additive model", COL["green"])}
    return (
        COL,
        CONFIG,
        D_GRID,
        N,
        RUNNERS,
        SEEDS,
        factor_logcond,
        np,
        pair_features,
        plot_utils,
        plt,
        sw,
        theory_table,
    )


@app.cell
def _(RUNNERS, SEEDS, mo, sw):
    import itertools
    # every cell of the sweep whose networks have finished training; each row holds the theory
    # ("th"), the theory's solution ("o") and one measured tuple per trained runner
    # measured cells, filled on demand: one saved evaluation is cheap to read, but there are
    # 8100 of them, so a setting is loaded the first time a figure asks for it
    SW = {}

    return (SW,)


@app.cell
def _(D_GRID, SW, mo, sw):
    def setting_ui():
        """Selectors for one figure: sparsity p, ladder slope beta, correlation rho."""
        return mo.ui.dictionary({
            "p": mo.ui.dropdown(options=[str(_p) for _p in sw.GRID["p"]], value="0.3", label="sparsity p"),
            "beta": mo.ui.dropdown(options=[str(_b) for _b in sw.GRID["beta"]], value="0.5", label="ladder slope β"),
            "rho": mo.ui.dropdown(options=[str(_r) for _r in sw.GRID["rho"]], value="0.0", label="correlation ρ"),
        })

    def load_cell(p, b, r, d):
        """One measured cell, cached in SW: the theory and every runner whose five seeds are saved."""
        if (p, b, r, d) not in SW:
            _have = [_run for _run in RUNNERS if all(sw.cached(p, b, r, d, _s, _run) for _s in SEEDS)]
            SW[(p, b, r, d)] = sw.sweep_cell(p, b, r, d, runners=_have) if "linear" in _have else None
        return SW[(p, b, r, d)]

    def rows_for(sel):
        """The sweep rows of a selected setting: dict(rows {d: rows} or None, title, p, b, rho,
        runners present in every d)."""
        _p, _b, _r = float(sel["p"]), float(sel["beta"]), float(sel["rho"])
        _rows = {_d: load_cell(_p, _b, _r, _d) for _d in D_GRID}
        if any(_v is None for _v in _rows.values()):
            _rows = None
        _runs = set.intersection(*[set(_rows[_d][0]) - {"th", "o"} for _d in D_GRID]) if _rows else set()
        return dict(rows=_rows, title=rf"$p$ = {_p}, $\beta$ = {_b}, $\rho$ = {_r}", p=_p, b=_b, rho=_r, runners=_runs)

    NOT_YET = mo.md("this setting is not trained yet")

    def selector_row(sel):
        return mo.hstack([sel["p"], sel["beta"], sel["rho"]], justify="start", gap=2)

    return NOT_YET, load_cell, rows_for, selector_row, setting_ui


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 1 — allocation
    Leverage $d_i = \|E_i\|^2$ of the whitened first-order frame: per-seed trained dots,
    reference means $\pm$ se, and the predicted allocation (water filling plus the soft tail).
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
    mo,
    np,
    plot_utils,
    plt,
    rows_for,
    selector_row,
):
    _sel = rows_for(SEL_F1.value)
    mo.stop(_sel["rows"] is None, NOT_YET)
    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for _ax, _d in zip(_axes, D_GRID):
        _x = np.arange(24)
        for _row in _sel["rows"][_d]:
            _ax.plot(_x, _row["linear"][0][:24], "o", ms=4, alpha=0.45, color=COL["blue"], label="trained" if _row is _sel["rows"][_d][0] else None)
        for _run, (_lbl, _col) in RUNNERS.items():
            if _run not in _sel["runners"]:
                continue
            _rm = np.array([_row[_run][0][:24] for _row in _sel["rows"][_d]])
            _mu, _se = _rm.mean(0), _rm.std(0) / np.sqrt(len(_rm))
            if _run != "linear":
                _ax.plot(_x, _mu, "-", lw=1.8, color=_col, label=_lbl)
                _ax.fill_between(_x, _mu - _se, _mu + _se, color=_col, alpha=0.25, lw=0)
        _pr = np.array([_row["th"][0][:24] for _row in _sel["rows"][_d]])
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
    $\|E_i\|^2$ of the first-order vectors in the balanced gauge of the state table; the
    dotted line is $\alpha_i$ in the same units.
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
    N,
    NOT_YET,
    RUNNERS,
    SEL_F2,
    mo,
    np,
    plot_utils,
    plt,
    rows_for,
    selector_row,
    sw,
):
    _sel = rows_for(SEL_F2.value)
    mo.stop(_sel["rows"] is None, NOT_YET)
    _alpha = sw.SCALE * (1.0 + np.arange(sw.M)) ** -_sel["b"]
    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for _ax, _d in zip(_axes, D_GRID):
        _x = np.arange(24)
        for _run, (_lbl, _col) in RUNNERS.items():
            if _run not in _sel["runners"]:
                continue
            _rm = np.array([np.diag(_row[_run][1])[:24] for _row in _sel["rows"][_d]])
            _mu, _se = _rm.mean(0), _rm.std(0) / np.sqrt(len(_rm))
            _ax.plot(_x, _mu, "o-" if _run == "linear" else "-", ms=4, lw=1.2 if _run == "linear" else 1.8, color=_col, label=_lbl)
            _ax.fill_between(_x, _mu - _se, _mu + _se, color=_col, alpha=0.25, lw=0)
        _pr = np.array([np.diag(_row["th"][1])[:24] for _row in _sel["rows"][_d]])
        _m, _s = _pr.mean(0), _pr.std(0) / np.sqrt(len(_pr))
        _ax.plot(_x, _m, "--", lw=2.0, color="black", label="theory")
        _ax.fill_between(_x, _m - _s, _m + _s, color="0.3", alpha=0.18, lw=0)
        if _sel["rho"] == 0.0:
            _ax.plot(_x, _alpha[:24] / np.sqrt(N * _sel["p"] * (1 - _sel["p"])), ":", lw=1.2, color="0.6",
                     label=r"$\alpha_i / \sqrt{Np(1-p)}$")
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
    $E E^\top$ of the first-order vectors, top 16 features, seed means.
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
    mo,
    np,
    plot_utils,
    plt,
    rows_for,
    selector_row,
):
    _sel = rows_for(SEL_F3.value)
    mo.stop(_sel["rows"] is None, NOT_YET)
    _K = 16
    _refs = [_run for _run in ("paired_features", "additive_features") if _run in _sel["runners"]]
    _fig, _axes = plt.subplots(2 + len(_refs), 3, figsize=(13.2, 4.1 * (2 + len(_refs))), squeeze=False)
    for _c, _d in enumerate(D_GRID):
        _panels = [(np.mean([_row["linear"][1][:_K, :_K] for _row in _sel["rows"][_d]], axis=0), "trained (5-seed mean)"),
                   (np.mean([_row["th"][1][:_K, :_K] for _row in _sel["rows"][_d]], axis=0), "theory (5-draw mean)")]
        _panels += [(np.mean([_row[_run][1][:_K, :_K] for _row in _sel["rows"][_d]], axis=0), f"{RUNNERS[_run][0]} (5-seed mean)") for _run in _refs]
        _v = max(np.abs(_Gm).max() for _Gm, _lbl in _panels)
        for _r, (_Gm, _lbl) in enumerate(_panels):
            _ax = _axes[_r, _c]
            _im = _ax.imshow(_Gm, cmap="vlag", vmin=-_v, vmax=_v)
            _ax.set_title(f"d = {_d} — {_lbl}", fontsize=11)
            _ax.set_xticks([]); _ax.set_yticks([])
            _fig.colorbar(_im, ax=_ax, shrink=0.8)
    _fig.suptitle(_sel["title"], y=1.0)
    _out = plot_utils.save_figure(_fig, plot_utils.figure_path(CONFIG, "fig3_grams"))
    mo.vstack([selector_row(SEL_F3), _out])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 4 — spectrum
    Top-$d$ singular values of the row-centred logit table over states, per state ($\sigma_k(\Lambda)/\sqrt{n}$).
    """)
    return


@app.cell(hide_code=True)
def _(setting_ui):
    SEL_F4 = setting_ui()
    return (SEL_F4,)


@app.cell
def _(
    CONFIG,
    D_GRID,
    NOT_YET,
    RUNNERS,
    SEL_F4,
    mo,
    np,
    plot_utils,
    plt,
    rows_for,
    selector_row,
):
    _sel = rows_for(SEL_F4.value)
    mo.stop(_sel["rows"] is None, NOT_YET)
    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4.0))
    for _ax, _d in zip(_axes, D_GRID):
        _x = 1 + np.arange(_d)
        for _run, (_lbl, _col) in RUNNERS.items():
            if _run not in _sel["runners"]:
                continue
            _rm = np.array([_row[_run][2] for _row in _sel["rows"][_d]])
            _mu, _se = _rm.mean(0), _rm.std(0) / np.sqrt(len(_rm))
            _ax.plot(_x, _mu, "o-" if _run == "linear" else "-", ms=4, lw=1.2 if _run == "linear" else 1.8, color=_col, label=_lbl)
            _ax.fill_between(_x, _mu - _se, _mu + _se, color=_col, alpha=0.25, lw=0)
        _pr = np.array([_row["th"][2] for _row in _sel["rows"][_d]])
        _m, _s = _pr.mean(0), _pr.std(0) / np.sqrt(len(_pr))
        _ax.plot(_x, _m, "--", lw=2.0, color="black", label="theory")
        _ax.fill_between(_x, _m - _s, _m + _s, color="0.3", alpha=0.18, lw=0)
        _ax.set_title(f"d = {_d}")
        _ax.set_xlabel("mode")
        plot_utils.FIG.style_axes(_ax)
    _axes[0].set_ylabel(r"$\sigma_k(\Lambda) / \sqrt{n}$")
    _axes[0].legend(frameon=False, fontsize=9)
    _fig.suptitle(_sel["title"], y=1.02)
    _out = plot_utils.save_figure(_fig, plot_utils.figure_path(CONFIG, "fig4_spectra"))
    mo.vstack([selector_row(SEL_F4), _out])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 5 — readout Grams
    $D^\top D$ of the balanced readout, top 16 outcomes, seed means.
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
    mo,
    np,
    plot_utils,
    plt,
    rows_for,
    selector_row,
):
    _sel = rows_for(SEL_F5.value)
    mo.stop(_sel["rows"] is None, NOT_YET)
    _K = 16
    _refs = [_run for _run in ("paired_features", "additive_features") if _run in _sel["runners"]]
    _fig, _axes = plt.subplots(2 + len(_refs), 3, figsize=(13.2, 4.1 * (2 + len(_refs))), squeeze=False)
    for _c, _d in enumerate(D_GRID):
        _panels = [(np.mean([_row["linear"][3][:_K, :_K] for _row in _sel["rows"][_d]], axis=0), "trained (5-seed mean)"),
                   (np.mean([_row["th"][3][:_K, :_K] for _row in _sel["rows"][_d]], axis=0), "theory (5-draw mean)")]
        _panels += [(np.mean([_row[_run][3][:_K, :_K] for _row in _sel["rows"][_d]], axis=0), f"{RUNNERS[_run][0]} (5-seed mean)") for _run in _refs]
        _v = max(np.abs(_Gm).max() for _Gm, _lbl in _panels)
        for _r, (_Gm, _lbl) in enumerate(_panels):
            _ax = _axes[_r, _c]
            _im = _ax.imshow(_Gm, cmap="vlag", vmin=-_v, vmax=_v)
            _ax.set_title(f"d = {_d} — {_lbl}", fontsize=11)
            _ax.set_xticks([]); _ax.set_yticks([])
            _fig.colorbar(_im, ax=_ax, shrink=0.8)
    _fig.suptitle(_sel["title"], y=1.0)
    _out = plot_utils.save_figure(_fig, plot_utils.figure_path(CONFIG, "fig5_decoder_grams"))
    mo.vstack([selector_row(SEL_F5), _out])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 6 — features and pairs together
    Gram of the regression vectors of the balanced code on $[1, z - \bar z, \phi(z)]$: the top 16 features
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
    SEEDS,
    SEL_F6,
    factor_logcond,
    mo,
    np,
    pair_features,
    plot_utils,
    plt,
    rows_for,
    selector_row,
    sw,
    theory_table,
):
    _sel = rows_for(SEL_F6.value)
    mo.stop(_sel["rows"] is None, NOT_YET)
    import itertools as _it
    _K, _TOP = 16, 6
    _pairs_all = list(_it.combinations(range(18), 2))
    _pairsel = [_pairs_all.index(p) for p in _it.combinations(range(_TOP), 2)]   # 15 pairs among the top 6
    _alpha = sw.SCALE * (1.0 + np.arange(sw.M)) ** -_sel["b"]

    def _ext_gram(_L, _X, _d):
        """Gram of the regression vectors of the balanced code on [1, x, phi(x)]: top-K features
        and the selected pairs."""
        _H = factor_logcond(_L, rank=_d)[1]
        _A = np.column_stack([np.ones(len(_X)), _X - _X.mean(0), pair_features(_X, 18)])
        _B = np.linalg.lstsq(_A, _H, rcond=None)[0]
        _E = np.vstack([_B[1:1 + _K], _B[1 + 64:][_pairsel]])
        return _E @ _E.T

    def _coords(_s):
        return sw.modes(_alpha, sw.features(_sel["p"], _sel["b"], _sel["rho"], _s), _sel["rho"])[1]

    _has_pairs = "paired_features" in _sel["runners"]
    _fig, _axes = plt.subplots(3 if _has_pairs else 2, 3, figsize=(13.2, 12.4 if _has_pairs else 8.4), squeeze=False)
    for _c, _d in enumerate(D_GRID):
        _panels = [(np.mean([_ext_gram(sw.train_cell(_sel["p"], _sel["b"], _sel["rho"], _d, _s), _coords(_s), _d) for _s in SEEDS], 0), "trained (5-seed mean)"),
                   (np.mean([_ext_gram(theory_table(_row["o"], _coords(_s)), _coords(_s), _d) for _s, _row in zip(SEEDS, _sel["rows"][_d])], 0), "theory (5-draw mean)")]
        if _has_pairs:
            _panels.append((np.mean([_ext_gram(sw.train_cell(_sel["p"], _sel["b"], _sel["rho"], _d, _s, "paired_features"), _coords(_s), _d) for _s in SEEDS], 0), "pairs model (5-seed mean)"))
        _v = max(np.abs(_Gm[_K:, :]).max() for _Gm, _lbl in _panels)   # scale to the pair blocks; the feature block saturates
        for _r, (_Gm, _lbl) in enumerate(_panels):
            _ax = _axes[_r, _c]
            _im = _ax.imshow(_Gm, cmap="vlag", vmin=-_v, vmax=_v)
            _ax.axhline(_K - 0.5, color="0.3", lw=0.8); _ax.axvline(_K - 0.5, color="0.3", lw=0.8)
            _ax.set_title(f"d = {_d} — {_lbl}", fontsize=11)
            _ax.set_xticks([]); _ax.set_yticks([])
            _fig.colorbar(_im, ax=_ax, shrink=0.8)
    _fig.suptitle(_sel["title"], y=1.0)
    _out = plot_utils.save_figure(_fig, plot_utils.figure_path(CONFIG, "fig6_extended_grams"))
    mo.vstack([selector_row(SEL_F6), _out])
    return


if __name__ == "__main__":
    app.run()
