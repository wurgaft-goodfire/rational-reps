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
    vectors $E_i$ of the least-squares fit $h(z) \approx \bar h + \sum_i E_i z_i$ to that code. Five
    runs per $d$, task and training seeds coupled; results and open items in
    `notes/superposition_results.md`.
    """)
    return


@app.cell
def _(mo):
    import os, sys
    import numpy as np

    ROOT = mo.notebook_dir().parent.parent    # src/notebooks -> <repo>
    os.chdir(ROOT); sys.path.insert(0, str(ROOT))
    from src.code.data import feature_cube
    from src.code.geometry import (factor_logcond, feature_vectors, leverage, pair_features,
                                   predict_geometry)
    from src.code.train import resolve_config, train_inline
    from src.code import plot_utils
    import matplotlib.pyplot as plt

    N, M, P, SCALE, BETA = 2048, 64, 0.3, 8.0, 0.5
    D_GRID = [3, 6, 12]
    SEEDS = range(5)
    ALPHA = SCALE * (1.0 + np.arange(M)) ** -BETA
    ZS = {s: feature_cube(N, M, P, scale=SCALE, seed=s, spectrum="power", beta=BETA)[1]["Z"]
          for s in SEEDS}

    TASK = ["setting=feature_cube", f"structure_kwargs.N={N}", f"structure_kwargs.m={M}",
            f"structure_kwargs.scale={SCALE}", "+structure_kwargs.spectrum=power",
            f"+structure_kwargs.beta={BETA}", f"structure_kwargs.p={P}"]
    CONFIG = resolve_config(TASK)
    COL = plot_utils.FIG.colors

    def measure(L, Zs, d):
        """One model's geometry from its logit table over states L (n, m): balanced rank-d
        code H and readout D (factor_logcond), first-order vectors E1 of H by least squares.
        Returns (leverage h (m,), feature Gram E1 E1^T (m, m), spectrum of the first-order
        map E1 D (d,), readout Gram D^T D (m, m), leverage projector P (m, m))."""
        _, H, D, _ = factor_logcond(L, rank=d)
        E1 = feature_vectors(H, Zs)[0]
        L1 = E1 @ D
        L1 = L1 - L1.mean(1, keepdims=True)
        h, P = leverage(E1)
        return h, E1 @ E1.T, np.linalg.svd(L1, compute_uv=False)[:d], D.T @ D, P

    def theory_table(o, Zs):
        """The completed model's logit table over states: [z, phi(z), 1] @ map_rep."""
        return np.column_stack([Zs, pair_features(Zs, 18), np.ones(len(Zs))]) @ o["map_rep"]

    def run_map(d, seed, runner="linear", tag="sp"):
        """A trained run's geometry, any runner, measured on its own task draw (task and
        training seeds are coupled): `measure` of its logit table code @ readout^T."""
        model, _ds, ev, *_ = train_inline(
            TASK + [f"runner={runner}", f"model_kwargs.model_config.embed_dim={d}",
                    "training_kwargs.train_steps=8000", f"random_seed={seed}",
                    f"structure_kwargs.seed={seed}"],
            name=f"{tag}_d{d}_s{seed}")
        code = np.asarray(ev.input_code(model, project=False), float)
        return measure(code @ model.readout.detach().cpu().numpy()[:M].T, ZS[seed], d)

    return (
        ALPHA,
        COL,
        CONFIG,
        D_GRID,
        SEEDS,
        TASK,
        ZS,
        factor_logcond,
        measure,
        np,
        plot_utils,
        plt,
        predict_geometry,
        run_map,
        theory_table,
        train_inline,
    )


@app.cell
def _(ALPHA, CONFIG, D_GRID, SEEDS, ZS, np, plot_utils, predict_geometry):
    # the theory pipeline, per run: each seed's prediction is solved on that seed's
    # own task draw, mirroring the coupled-seed convention of the models. Water
    # filling -> Fantope orientation (E*, the *_star series) -> exact completion at E*
    # (decoder + pairs, the *_rep series) -> random-frame reference band (*_mean/_sd).
    # Cached in the gitignored figure cache - delete the npz files to recompute.
    import pathlib as _pl

    PRED = {}
    for _d in D_GRID:
        PRED[_d] = []
        for _s in SEEDS:
            _f = _pl.Path(plot_utils.figure_path(CONFIG, f"geom_d{_d}_s{_s}", ext="npz"))
            if _f.exists():
                PRED[_d].append(dict(np.load(_f)))
            else:
                _out = predict_geometry(ALPHA, ZS[_s], _d, n_frames=32, seed=_s)
                _out = {_k: np.asarray(_v) for _k, _v in _out.items()
                        if not isinstance(_v, dict)}
                np.savez(_f, **_out)
                PRED[_d].append(_out)
    return (PRED,)


@app.cell
def _(D_GRID, PRED, SEEDS, ZS, measure, theory_table):
    # the theory measured exactly like a trained run, from its logit table over states
    TH = {d: [measure(theory_table(o, ZS[s]), ZS[s], d) for s, o in zip(SEEDS, PRED[d])]
          for d in D_GRID}
    return (TH,)


@app.cell
def _(D_GRID, SEEDS, run_map):
    TR = {d: [run_map(d, s) for s in SEEDS] for d in D_GRID}
    return (TR,)


@app.cell
def _(D_GRID, SEEDS, run_map):
    # reference classes, trained by the same pipeline as the networks (same steps,
    # same trainer, same seeds) and measured identically; per-seed runs kept for se bands.
    REF = {d: dict(additive=[run_map(d, s, "additive_features", "addf") for s in SEEDS],
                   pairs=[run_map(d, s, "paired_features", "pairf") for s in SEEDS])
           for d in D_GRID}
    return (REF,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 1 — allocation
    Leverage $d_i = \|E_i\|^2$ of the whitened first-order frame: per-seed trained dots,
    reference means $\pm$ se, and the predicted allocation (water filling plus the soft tail).
    """)
    return


@app.cell
def _(COL, CONFIG, D_GRID, PRED, REF, TR, np, plot_utils, plt):
    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for _ax, _d in zip(_axes, D_GRID):
        _x = np.arange(24)
        for _h, _G, _S, _GD, _P in TR[_d]:
            _ax.plot(_x, _h[:24], "o", ms=4, alpha=0.45, color=COL["blue"])
        for _key, _col, _lbl in [("additive", COL["green"], "additive model"),
                                 ("pairs", COL["purple"], "additive + pairs model")]:
            _rm = np.array([_r[0][:24] for _r in REF[_d][_key]])
            _rmu, _rse = _rm.mean(0), _rm.std(0) / np.sqrt(len(_rm))
            _ax.plot(_x, _rmu, "-", lw=1.8, color=_col, label=_lbl)
            _ax.fill_between(_x, _rmu - _rse, _rmu + _rse, color=_col, alpha=0.25, lw=0)
        _pm = np.array([_o["d_star"][:24] for _o in PRED[_d]])
        _pmu, _pse = _pm.mean(0), _pm.std(0) / np.sqrt(len(_pm))
        _ax.plot(_x, _pmu, "--", lw=2.0, color="black", label="theory")
        _ax.fill_between(_x, _pmu - _pse, _pmu + _pse, color="0.3", alpha=0.18, lw=0)
        _ax.set_title(f"d = {_d}")
        _ax.set_xlabel("feature $i$")
        plot_utils.FIG.style_axes(_ax)
    _axes[0].set_ylabel(r"leverage $d_i$")
    _axes[0].legend(frameon=False, fontsize=9)
    plot_utils.save_figure(_fig, plot_utils.figure_path(CONFIG, "fig1_allocation"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 2 — feature sizes
    $\|E_i\|^2$ of the first-order vectors in the balanced gauge of the state table; the
    dotted line is $\alpha_i$ in the same units.
    """)
    return


@app.cell
def _(ALPHA, COL, CONFIG, D_GRID, N, P, REF, TH, TR, np, plot_utils, plt):
    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for _ax, _d in zip(_axes, D_GRID):
        _x = np.arange(24)
        _tm = np.array([np.diag(_G)[:24] for _h, _G, _S, _GD, _P in TR[_d]])
        _mu, _se = _tm.mean(0), _tm.std(0) / np.sqrt(len(_tm))
        _ax.plot(_x, _mu, "o-", ms=4, lw=1.2, color=COL["blue"], label="trained")
        _ax.fill_between(_x, _mu - _se, _mu + _se, color=COL["blue"], alpha=0.25, lw=0)
        for _key, _col, _lbl in [("additive", COL["green"], "additive model"),
                                 ("pairs", COL["purple"], "additive + pairs model")]:
            _rm = np.array([np.diag(_r[1])[:24] for _r in REF[_d][_key]])
            _rmu, _rse = _rm.mean(0), _rm.std(0) / np.sqrt(len(_rm))
            _ax.plot(_x, _rmu, "-", lw=1.8, color=_col, label=_lbl)
            _ax.fill_between(_x, _rmu - _rse, _rmu + _rse, color=_col, alpha=0.25, lw=0)
        _pr = np.array([np.diag(_t[1])[:24] for _t in TH[_d]])
        _m, _s = _pr.mean(0), _pr.std(0) / np.sqrt(len(_pr))
        _ax.plot(_x, _m, "--", lw=2.0, color="black", label="theory")
        _ax.fill_between(_x, _m - _s, _m + _s, color="0.3", alpha=0.18, lw=0)
        _ax.plot(_x, ALPHA[:24] / np.sqrt(N * P * (1 - P)), ":", lw=1.2, color="0.6",
                 label=r"$\alpha_i / \sqrt{Np(1-p)}$")
        _ax.set_title(f"d = {_d}")
        _ax.set_xlabel("feature $i$")
        plot_utils.FIG.style_axes(_ax)
    _axes[0].set_ylabel(r"$\|E_i\|^2$")
    _axes[0].legend(frameon=False, fontsize=9)
    plot_utils.save_figure(_fig, plot_utils.figure_path(CONFIG, "fig2_sizes"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 3 — feature Grams
    $E E^\top$ of the first-order vectors, top 16 features, seed means.
    """)
    return


@app.cell
def _(CONFIG, D_GRID, REF, TH, TR, np, plot_utils, plt):
    _K = 16

    _fig, _axes = plt.subplots(4, 3, figsize=(13.2, 16.4))
    for _c, _d in enumerate(D_GRID):
        _rows = [(np.mean([_G[:_K, :_K] for _h, _G, _S, _GD, _P in TR[_d]], axis=0),
                  "trained (5-seed mean)"),
                 (np.mean([_r[1][:_K, :_K] for _r in REF[_d]["pairs"]], axis=0),
                  "additive + pairs (5-seed mean)"),
                 (np.mean([_r[1][:_K, :_K] for _r in REF[_d]["additive"]], axis=0),
                  "additive (5-seed mean)"),
                 (np.mean([_t[1][:_K, :_K] for _t in TH[_d]], axis=0),
              "theory (5-draw mean)")]
        _v = max(np.abs(_Gm).max() for _Gm, _lbl in _rows)
        for _r, (_Gm, _lbl) in enumerate(_rows):
            _ax = _axes[_r, _c]
            _im = _ax.imshow(_Gm, cmap="vlag", vmin=-_v, vmax=_v)
            _ax.set_title(f"d = {_d} — {_lbl}", fontsize=11)
            _ax.set_xticks([]); _ax.set_yticks([])
            _fig.colorbar(_im, ax=_ax, shrink=0.8)
    plot_utils.save_figure(_fig, plot_utils.figure_path(CONFIG, "fig3_grams"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 4 — spectrum
    Singular values of the first-order map $E W^\top$ (gauge-free).
    """)
    return


@app.cell
def _(COL, CONFIG, D_GRID, REF, TH, TR, np, plot_utils, plt):
    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4.0))
    for _ax, _d in zip(_axes, D_GRID):
        _x = 1 + np.arange(_d)
        _tm = np.array([_S for _h, _G, _S, _GD, _P in TR[_d]])
        _mu, _se = _tm.mean(0), _tm.std(0) / np.sqrt(len(_tm))
        _ax.plot(_x, _mu, "o-", ms=4, lw=1.2, color=COL["blue"], label="trained")
        _ax.fill_between(_x, _mu - _se, _mu + _se, color=COL["blue"], alpha=0.25, lw=0)
        for _key, _col, _lbl in [("additive", COL["green"], "additive model"),
                                 ("pairs", COL["purple"], "additive + pairs model")]:
            _rm = np.array([_r[2] for _r in REF[_d][_key]])
            _rmu, _rse = _rm.mean(0), _rm.std(0) / np.sqrt(len(_rm))
            _ax.plot(_x, _rmu, "-", lw=1.8, color=_col, label=_lbl)
            _ax.fill_between(_x, _rmu - _rse, _rmu + _rse, color=_col, alpha=0.25, lw=0)
        _pr = np.array([_t[2] for _t in TH[_d]])
        _m, _s = _pr.mean(0), _pr.std(0) / np.sqrt(len(_pr))
        _ax.plot(_x, _m, "--", lw=2.0, color="black", label="theory")
        _ax.fill_between(_x, _m - _s, _m + _s, color="0.3", alpha=0.18, lw=0)
        _ax.set_title(f"d = {_d}")
        _ax.set_xlabel("mode")
        plot_utils.FIG.style_axes(_ax)
    _axes[0].set_ylabel(r"$\sigma_k(L^{(1)})$")
    _axes[0].legend(frameon=False, fontsize=9)
    plot_utils.save_figure(_fig, plot_utils.figure_path(CONFIG, "fig4_spectra"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 5 — readout Grams
    $D^\top D$ of the balanced readout, top 16 outcomes, seed means.
    """)
    return


@app.cell
def _(CONFIG, D_GRID, REF, TH, TR, np, plot_utils, plt):
    _K = 16
    _fig, _axes = plt.subplots(4, 3, figsize=(13.2, 16.4))
    for _c, _d in enumerate(D_GRID):
        _rows = [(np.mean([_GD[:_K, :_K] for _h, _G, _S, _GD, _P in TR[_d]], axis=0),
                  "trained (5-seed mean)"),
                 (np.mean([_r[3][:_K, :_K] for _r in REF[_d]["pairs"]], axis=0),
                  "additive + pairs (5-seed mean)"),
                 (np.mean([_r[3][:_K, :_K] for _r in REF[_d]["additive"]], axis=0),
                  "additive (5-seed mean)"),
                 (np.mean([_t[3][:_K, :_K] for _t in TH[_d]], axis=0),
                  "theory (5-draw mean)")]
        _v = max(np.abs(_Gm).max() for _Gm, _lbl in _rows)
        for _r, (_Gm, _lbl) in enumerate(_rows):
            _ax = _axes[_r, _c]
            _im = _ax.imshow(_Gm, cmap="vlag", vmin=-_v, vmax=_v)
            _ax.set_title(f"d = {_d} — {_lbl}", fontsize=11)
            _ax.set_xticks([]); _ax.set_yticks([])
            _fig.colorbar(_im, ax=_ax, shrink=0.8)
    plot_utils.save_figure(_fig, plot_utils.figure_path(CONFIG, "fig5_decoder_grams"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 6 — soft tail
    Trained leverage of features water filling drops, against the closed form
    $\|a_{\cdot j}\|^2 / 4\delta^2$ (dots) and the Fantope program's own tail (crosses); log–log
    with the identity and a factor-of-two band.
    """)
    return


@app.cell
def _(COL, CONFIG, D_GRID, PRED, TR, np, plot_utils, plt):
    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for _ax, _d in zip(_axes, D_GRID):
        _xs, _ys, _fs = [], [], []
        for _o, (_h, _G, _S, _GD, _P) in zip(PRED[_d], TR[_d]):
            _rep = _o["d_wf"] > 0.05
            _drop = np.flatnonzero((~_rep) & (np.arange(len(_rep)) < 24))
            _an2 = (_o["a"][_drop][:, _rep] ** 2).sum(1)
            _xs.append(_an2 / (4 * float(_o["delta"]) ** 2))
            _ys.append(_h[_drop]); _fs.append(_o["d_star"][_drop])
        _x, _y, _f = (np.concatenate(_v) for _v in (_xs, _ys, _fs))
        _ok = (_y > 1e-4) & (_x > 1e-6)
        _ax.loglog(_x[_ok], _y[_ok], "o", ms=4, alpha=0.6, color=COL["blue"], label="trained")
        _ax.loglog(_x[_ok], np.maximum(_f[_ok], 1e-5), "x", ms=5, color="black", label="Fantope tail $d^\\star_j$")
        _lo, _hi = _x[_ok].min() * 0.5, _x[_ok].max() * 2
        _ax.plot([_lo, _hi], [_lo, _hi], "-", color="0.5", lw=1)
        _ax.fill_between([_lo, _hi], [_lo / 2, _hi / 2], [_lo * 2, _hi * 2], color="0.5", alpha=0.15, lw=0)
        _slope = np.polyfit(np.log(_x[_ok]), np.log(_y[_ok]), 1)[0]
        _ax.set_title(f"d = {_d}   (log–log slope {_slope:.2f})")
        _ax.set_xlabel(r"predicted $\|a_{\cdot j}\|^2 / 4\delta^2$")
        plot_utils.FIG.style_axes(_ax)
    _axes[0].set_ylabel("tail leverage")
    _axes[0].legend(frameon=False, fontsize=9)
    plot_utils.save_figure(_fig, plot_utils.figure_path(CONFIG, "fig6_soft_tail"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 7 — features and pairs together
    Gram of the regression vectors of the balanced code on $[1, z - \bar z, \phi(z)]$: the top 16 features
    and the 15 pairs among the top 6, colour scale set by the pair blocks (the feature block
    saturates).
    """)
    return


@app.cell
def _(
    CONFIG,
    D_GRID,
    PRED,
    SEEDS,
    TASK,
    ZS,
    np,
    plot_utils,
    plt,
    factor_logcond,
    theory_table,
    train_inline,
):
    import itertools as _it
    from src.code.geometry import pair_features as _pf
    _K, _TOP = 16, 6
    _pairs_all = list(_it.combinations(range(18), 2))
    _sel = [_pairs_all.index(p) for p in _it.combinations(range(_TOP), 2)]   # 15 pairs among the top 6

    def _ext_gram(_code, _Zs):
        _X = np.column_stack([np.ones(len(_Zs)), _Zs - _Zs.mean(0), _pf(_Zs, 18)])
        _B = np.linalg.lstsq(_X, _code, rcond=None)[0]
        _E = np.vstack([_B[1:1 + _K], _B[1 + 64:][_sel]])
        return _E @ _E.T

    def _ext_gram_trained(_d, _s, _runner, _tag):
        _model, _ds, _ev, *_ = train_inline(TASK + [f"runner={_runner}", f"model_kwargs.model_config.embed_dim={_d}",
            "training_kwargs.train_steps=8000", f"random_seed={_s}", f"structure_kwargs.seed={_s}"], name=f"{_tag}_d{_d}_s{_s}")
        _L = np.asarray(_ev.input_code(_model, project=False), float) @ _model.readout.detach().cpu().numpy()[:64].T
        return _ext_gram(factor_logcond(_L, rank=_d)[1], ZS[_s])

    def _ext_gram_theory(_o, _s, _d):
        return _ext_gram(factor_logcond(theory_table(_o, ZS[_s]), rank=_d)[1], ZS[_s])

    _fig, _axes = plt.subplots(3, 3, figsize=(13.2, 12.4))
    for _c, _d in enumerate(D_GRID):
        _rows = [(np.mean([_ext_gram_trained(_d, _s, "linear", "sp") for _s in SEEDS], 0), "trained (5-seed mean)"),
                 (np.mean([_ext_gram_trained(_d, _s, "paired_features", "pairf") for _s in SEEDS], 0), "additive + pairs (5-seed mean)"),
                 (np.mean([_ext_gram_theory(_o, _s, _d) for _s, _o in zip(SEEDS, PRED[_d])], 0), "theory (5-draw mean)")]
        _v = max(np.abs(_Gm[_K:, :]).max() for _Gm, _lbl in _rows)   # scale to the pair blocks; the feature block saturates
        for _r, (_Gm, _lbl) in enumerate(_rows):
            _ax = _axes[_r, _c]
            _im = _ax.imshow(_Gm, cmap="vlag", vmin=-_v, vmax=_v)
            _ax.axhline(_K - 0.5, color="0.3", lw=0.8); _ax.axvline(_K - 0.5, color="0.3", lw=0.8)
            _ax.set_title(f"d = {_d} — {_lbl}", fontsize=11)
            _ax.set_xticks([]); _ax.set_yticks([])
            _fig.colorbar(_im, ax=_ax, shrink=0.8)
    plot_utils.save_figure(_fig, plot_utils.figure_path(CONFIG, "fig7_extended_grams"))
    return



@app.cell
def _(mo):
    mo.md(r"""
    ## Sweep — sparsity $p$, ladder slope $\beta$, rank $d$
    Trained networks and the theory on the grid $p \in \{0.05, 0.15, 0.3, 0.5\}$,
    $\beta \in \{0.25, 0.5, 1\}$, $d \in \{3, 6, 12\}$, five seeds per cell, measured like the
    main figures (`src/code/sweep_superposition.py`).
    """)
    return


@app.cell
def _(measure, np, theory_table):
    import itertools
    from src.code import sweep_superposition as sw

    SW = {}
    for _p, _b, _d in itertools.product(sw.GRID["p"], sw.GRID["beta"], sw.GRID["d"]):
        _rows = []
        for _s in sw.GRID["seed"]:
            _Zs = sw.features(_p, _b, _s)
            _o = dict(np.load(sw.cell_path(_p, _b, _d, _s)))
            _rows.append(dict(tr=measure(sw.train_cell(_p, _b, _d, _s), _Zs, _d),
                              th=measure(theory_table(_o, _Zs), _Zs, _d), o=_o))
        SW[(_p, _b, _d)] = _rows

    def place_corr(P1, P2, k=16):
        """Placement: correlation of the top-k off-diagonal entries of the leverage projectors."""
        _iu = np.triu_indices(k, 1)
        return np.corrcoef(P1[:k, :k][_iu], P2[:k, :k][_iu])[0, 1]

    def cell_scores(rows):
        """(leverage corr, placement corr on the leverage projectors, soft-tail slope) of the theory
        against the trained 5-seed means of one cell; the slope pools the dropped features of every
        seed (nan when water filling drops none)."""
        _ht = np.mean([r["tr"][0] for r in rows], 0); _hp = np.mean([r["th"][0] for r in rows], 0)
        _Pt = np.mean([r["tr"][4] for r in rows], 0); _Pp = np.mean([r["th"][4] for r in rows], 0)
        _xs, _ys = [], []
        for r in rows:
            _o, _h = r["o"], r["tr"][0]
            _rep = _o["d_wf"] > 0.05
            _drop = np.flatnonzero(~_rep & (np.arange(len(_rep)) < 24))
            _x = (_o["a"][_drop][:, _rep] ** 2).sum(1) / (4 * float(_o["delta"]) ** 2); _y = _h[_drop]
            _ok = (_y > 1e-4) & (_x > 1e-6); _xs.append(_x[_ok]); _ys.append(_y[_ok])
        _x, _y = np.concatenate(_xs), np.concatenate(_ys)
        _slope = np.polyfit(np.log(_x), np.log(_y), 1)[0] if len(_x) > 3 else np.nan
        return np.corrcoef(_hp[:24], _ht[:24])[0, 1], place_corr(_Pp, _Pt), _slope
    return SW, cell_scores, place_corr, sw


@app.cell
def _(mo):
    mo.md(r"""
    ## Figure 8 — allocation across settings
    Leverage $d_i$ per feature: trained per-seed dots, theory dashed (5-draw mean). Left grid $p \times d$
    at $\beta = 0.5$; right grid $p \times \beta$ at $d = 6$. Titles give the leverage and placement
    (leverage-projector) correlations of the theory with the trained 5-seed means.
    """)
    return


@app.cell
def _(COL, CONFIG, SW, cell_scores, np, plot_utils, plt, sw):
    def grid_fig(_cells, _row_vals, _col_vals, _row_name, _col_name, _fname):
        _fig, _axes = plt.subplots(len(_row_vals), len(_col_vals), figsize=(3.6 * len(_col_vals), 2.9 * len(_row_vals)),
                                   sharex=True, sharey=True)
        for _i, _rv in enumerate(_row_vals):
            for _j, _cv in enumerate(_col_vals):
                _ax = _axes[_i, _j]; _rows = SW[_cells(_rv, _cv)]
                _x = np.arange(24)
                for _r in _rows:
                    _ax.plot(_x, _r["tr"][0][:24], "o", ms=3, alpha=0.4, color=COL["blue"])
                _ax.plot(_x, np.mean([_r["th"][0][:24] for _r in _rows], 0), "--", lw=1.8, color="black")
                _lev, _pl, _sl = cell_scores(_rows)
                _ax.set_title(f"{_row_name}={_rv}, {_col_name}={_cv}   lev {_lev:.2f}  place {_pl:+.2f}", fontsize=9)
                plot_utils.FIG.style_axes(_ax)
                if _i == len(_row_vals) - 1: _ax.set_xlabel("feature $i$")
                if _j == 0: _ax.set_ylabel("leverage $d_i$")
        return plot_utils.save_figure(_fig, plot_utils.figure_path(CONFIG, _fname))

    grid_fig(lambda _p, _d: (_p, 0.5, _d), sw.GRID["p"], sw.GRID["d"], "p", "d", "fig8a_sweep_p_by_d")
    return (grid_fig,)


@app.cell
def _(grid_fig, sw):
    grid_fig(lambda _p, _b: (_p, _b, 6), sw.GRID["p"], sw.GRID["beta"], "p", r"$\beta$", "fig8b_sweep_p_by_beta")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Figure 9 — theory scores across the grid
    Leverage correlation, placement correlation (top-16 off-diagonals of the leverage projectors), and
    soft-tail log–log slope of the theory against the trained 5-seed means; one $p \times d$ heatmap per
    $\beta$. A dash marks cells where water filling drops no feature.
    """)
    return


@app.cell
def _(CONFIG, SW, cell_scores, np, plot_utils, plt, sw):
    _names = ["leverage corr", "placement corr", "soft-tail slope"]
    _vr = [(0, 1), (-1, 1), (0, 3)]
    _fig, _axes = plt.subplots(3, len(sw.GRID["beta"]), figsize=(4.8 * len(sw.GRID["beta"]), 10.5),
                               constrained_layout=True)
    for _j, _b in enumerate(sw.GRID["beta"]):
        _S = np.array([[cell_scores(SW[(_p, _b, _d)]) for _d in sw.GRID["d"]] for _p in sw.GRID["p"]])   # (p, d, 3)
        for _i in range(3):
            _ax = _axes[_i, _j]
            _im = _ax.imshow(_S[:, :, _i], cmap="vlag" if _i == 1 else "Reds", vmin=_vr[_i][0], vmax=_vr[_i][1], aspect="auto")
            for _a in range(len(sw.GRID["p"])):
                for _c in range(len(sw.GRID["d"])):
                    _v = _S[_a, _c, _i]
                    _ax.text(_c, _a, "–" if np.isnan(_v) else f"{_v:.2f}", ha="center", va="center", fontsize=9,
                             color="white" if (_i != 1 and _v > 0.6 * _vr[_i][1]) else "black")
            _ax.set_xticks(range(len(sw.GRID["d"]))); _ax.set_xticklabels([f"d={_d}" for _d in sw.GRID["d"]])
            _ax.set_yticks(range(len(sw.GRID["p"]))); _ax.set_yticklabels([f"p={_p}" for _p in sw.GRID["p"]])
            _ax.set_title(rf"$\beta$ = {_b} — {_names[_i]}", fontsize=10)
            _fig.colorbar(_im, ax=_ax, shrink=0.8)
    plot_utils.save_figure(_fig, plot_utils.figure_path(CONFIG, "fig9_sweep_scores"))
    return

if __name__ == "__main__":
    app.run()
