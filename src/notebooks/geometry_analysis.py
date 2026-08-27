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
    # Representational geometry: predicted vs. learned

    The geometry a network learns is predictable in closed form from the structure of the task.
    """)
    return


@app.cell
def _(mo):
    import os, sys
    import numpy as np
    import torch

    # Project root, independent of the working directory (a headless run can start outside
    # the repo). chdir so Hydra resolves conf/, and add it to sys.path.
    ROOT = mo.notebook_dir().parent.parent    # src/notebooks -> <repo>
    os.chdir(ROOT); sys.path.insert(0, str(ROOT))

    from src.code.data import graph_conditional
    from src.code.geometry import (predicted_geometry, factor_logcond, canonical_gauge,
                                   log_conditional,
                                   fourier_basis, circulant_fraction, class_means,
                                   frequency_plane, group_fraction)
    from src.code.eval import cka, rsa
    from src.code.train import train_inline, graph_overrides
    from src.code import plot_utils           # importing this applies the house plotting style

    FIGURES = ROOT / "outputs" / "figures"     # one flat dir: {structure}_{figure}.png
    FIGURES.mkdir(parents=True, exist_ok=True)
    return (
        FIGURES,
        canonical_gauge,
        circulant_fraction,
        cka,
        class_means,
        factor_logcond,
        fourier_basis,
        frequency_plane,
        graph_conditional,
        graph_overrides,
        group_fraction,
        log_conditional,
        np,
        plot_utils,
        predicted_geometry,
        torch,
        train_inline,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Factoring a task's conditional

    The SVD of the row-centred log-conditional splits into a code $E^{*}$ and a decoder $D^{*}$.
    """)
    return


@app.cell
def _(FIGURES, graph_conditional, plot_utils, predicted_geometry):
    # cooccurrence (diffusion | geodesic | adjacency | heat) flows to both the prediction and the
    # training; a hypercube wants "heat" so its bits stay independent.nalogies).
    spec = {"structure": "ring", "cooccurrence": "diffusion", "n": 30}
    L = graph_conditional(spec)
    E_pred = predicted_geometry(L)
    fig_fact, _ = plot_utils.plot_factorization(L)
    plot_utils.save_figure(fig_fact, FIGURES / f"{spec['structure']}_factorization.png")
    return E_pred, L, spec


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. The predicted geometry

    The structure, the log-conditional it induces, and the closed-form code: for the ring, a circle.
    """)
    return


@app.cell
def _(E_pred, FIGURES, L, plot_utils, spec):
    fig_pred = plot_utils.plot_predicted_geometry(spec, L, E_pred)
    plot_utils.save_figure(fig_pred, FIGURES / f"{spec['structure']}_predicted_geometry.png")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Trained geometry across settings

    Predicted code Gram $E^{*}E^{*\top}$ beside each learned $EE^{\top}$, per architecture.
    """)
    return


@app.cell
def _(
    FIGURES,
    cka,
    graph_conditional,
    graph_overrides,
    plot_utils,
    predicted_geometry,
    train_inline,
):
    ARCHS = ["transformer", "linear", "mlp", "rnn"]

    SPECS = [
        {"structure": "ring", "cooccurrence": "diffusion", "n": 30},
        {"structure": "grid", "cooccurrence": "heat", "n": 5, "m": 5},   # heat: additive over the chain factors, so the lattice sides stay straight
        {"structure": "product", "cooccurrence": "heat", "view": "3d",   # helix: line x circle
         "factors": [{"structure": "chain", "n": 8}, {"structure": "ring", "n": 8}]},
        {"structure": "hypercube", "cooccurrence": "heat", "n": 3},
        {"structure": "complete", "cooccurrence": "diffusion", "n": 4},
        {"structure": "tree", "n": 2, "k": 3},
    ]
    SETTING_CMAP = plot_utils.FIG.item_cmap        # twilight minus its grey ends (house item map)
    figs_settings = {}
    for _sp in SPECS:
        _L = graph_conditional(_sp)
        _reps, _cfg = {}, None
        for _arch in ARCHS:
            # runner defaults: readouts pointwise on the exact rows, LMs on [start, item, outcome]
            _model, _ds, _ev, _hist, _cfg = train_inline(graph_overrides(_sp) + [f"runner={_arch}"])
            _reps[_arch] = _ev.input_code(_model)
            print(f"{_sp['structure']:10s} {_arch:12s} "
                  f"CKA={cka(_reps[_arch], predicted_geometry(_ds.conditional)):.3f}")
        _fig = plot_utils.plot_embedding_panels(predicted_geometry(_L), _reps, spec=_sp,
                                                cmap=SETTING_CMAP)
        _key = "helix" if _sp["structure"] == "product" else _sp["structure"]
        plot_utils.save_figure(_fig, FIGURES / f"{_key}_embedding_panels.png")
        figs_settings[_key] = _fig
    return SETTING_CMAP, figs_settings


@app.cell
def _(figs_settings, mo):
    mo.ui.tabs(figs_settings)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### The six settings merged
    """)
    return


@app.cell
def _(FIGURES, plot_utils):
    import matplotlib.image as mpimg

    # letters follow the columns: (a-c) ring, grid, helix; (d-f) hypercube, simplex, hierarchy
    _grid = [["ring", "hypercube"], ["grid", "complete"], ["helix", "tree"]]
    _imgs = {_n: mpimg.imread(FIGURES / "png" / f"{_n}_embedding_panels.png") for _n in sum(_grid, [])}
    _ratios = [max(_imgs[_n].shape[0] / _imgs[_n].shape[1] for _n in _row) for _row in _grid]
    fig_main = plot_utils.plt.figure(figsize=(21, 10.5 * sum(_ratios) + 2.5))
    _gs = fig_main.add_gridspec(3, 2, height_ratios=_ratios, hspace=0.22, wspace=0.08,
                                left=0.02, right=0.98, top=0.96, bottom=0.03)
    for _i, _row in enumerate(_grid):
        for _j, _n in enumerate(_row):
            _ax = fig_main.add_subplot(_gs[_i, _j])
            _ax.imshow(_imgs[_n]); _ax.axis("off")
            _ax.set_title(f"({'abcdef'[_i + 3 * _j]})", loc="left", fontsize=30, pad=14)
    plot_utils.save_figure(fig_main, FIGURES / "geometry_predictions_main.png", dpi=200)
    return (mpimg,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Relational settings

    $c = a \ast b$: the conditional is a tensor $T[a,b,c]$ and the code is read per input pair.
    Each answer is a set or a noisy magnitude, so the target is graded and its spectrum decays —
    CKA against $E^{*}$ then measures the geometry, as in the static settings.

    | relation | answer |
    | --- | --- |
    | `set_intersection` | uniform over the features shared by $a$ and $b$ (Tversky's common features) |
    | `symmetric_difference` | uniform over the features of exactly one of them (distinctive features) |
    | `entailment` | uniform over the common ancestors of two nodes of a tree |
    | `between` | uniform over the integers in $[\min(a,b), \max(a,b)]$ |
    | `common_divisors` | uniform over the divisors of $\gcd(a,b)$ |
    | `approx_add` | $a+b$ under Weber noise: Gaussian in $\log$ space |
    """)
    return


@app.cell
def _(train_inline):
    from src.code.analysis_utils import input_code

    # every knob (width, decay, input format) lives in the runner preset, so a run is named by
    # the relation and the runner alone — nothing here can drift from what the CLI trained
    RUNNERS_REL = ["rel_transformer", "rel_rnn", "rel_mlp", "rel_linear_paired", "rel_mlp_paired"]
    OPS_REL = ["set_intersection", "symmetric_difference", "entailment",
               "between", "common_divisors", "approx_add"]

    def load_run(relation, runner, seed=0):
        """A cached run's model and dataset; `input_code` reads the geometry off them."""
        _model, _ds, _, _, _ = train_inline(
            [f"runner={runner}", f"relation={relation}", f"random_seed={seed}"])
        return _model, _ds


    return OPS_REL, RUNNERS_REL, input_code, load_run


@app.cell
def _(
    FIGURES,
    OPS_REL,
    RUNNERS_REL,
    SETTING_CMAP,
    cka,
    input_code,
    load_run,
    np,
    plot_utils,
    predicted_geometry,
):
    # One panel set per relation, in the static settings\' style: the predicted pair code beside every
    # architecture\'s (top) and their Grams (bottom). These relations have decaying spectra, so CKA
    # against E* is the measure.
    _rng_null = np.random.default_rng(0)

    figs_rel, NULLS_REL = {}, {}
    for _op in OPS_REL:
        _codes, _E, _colors = {}, None, None
        for _v in RUNNERS_REL:
            _model, _ds = load_run(_op, _v)
            _codes[_v.removeprefix("rel_")] = input_code(_model, _ds, project=True)
            if _E is None:                       # the oracle factors the dataset's log-conditional
                _E = np.asarray(predicted_geometry(_ds.conditional), float)
                # colour each pair by its answer: the distinct target rows, ranked by expected outcome
                _M = _ds.P.cpu().numpy()
                _uniq, _inv = np.unique(_M, axis=0, return_inverse=True)
                _colors = np.argsort(np.argsort(_uniq @ np.arange(_M.shape[1])))[_inv]
                NULLS_REL[_op] = np.mean([cka(_rng_null.normal(size=_E.shape), _E) for _ in range(20)])
        _scores = {_n: cka(_c, _E) for _n, _c in _codes.items()}
        print(f"{_op:22s} null {NULLS_REL[_op]:.2f}  "
              + "  ".join(f"{_n} {_s:.2f}" for _n, _s in _scores.items()), flush=True)
        _fig = plot_utils.plot_embedding_panels(_E, _codes, cmap=SETTING_CMAP,
                                                colors=_colors, scores=_scores)
        plot_utils.save_figure(_fig, FIGURES / f"relational_{_op}_embedding_panels.png")
        figs_rel[_op] = _fig
    return NULLS_REL, figs_rel


@app.cell
def _(figs_rel, mo):
    mo.ui.tabs(figs_rel)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### The six relations merged
    """)
    return


@app.cell
def _(FIGURES, NULLS_REL, OPS_REL, mpimg, plot_utils):
    # letters follow the columns, as in the static merged figure; each carries its random-code null
    _grid_rel = [[OPS_REL[_i], OPS_REL[_i + 3]] for _i in range(3)]
    _imgs = {_n: mpimg.imread(FIGURES / "png" / f"relational_{_n}_embedding_panels.png")
             for _n in OPS_REL}
    _ratios = [max(_imgs[_n].shape[0] / _imgs[_n].shape[1] for _n in _row) for _row in _grid_rel]
    fig_rel_main = plot_utils.plt.figure(figsize=(21, 10.5 * sum(_ratios) + 2.5))
    _gs = fig_rel_main.add_gridspec(3, 2, height_ratios=_ratios, hspace=0.22, wspace=0.08,
                                    left=0.02, right=0.98, top=0.96, bottom=0.03)
    for _i, _row in enumerate(_grid_rel):
        for _j, _n in enumerate(_row):
            _ax = fig_rel_main.add_subplot(_gs[_i, _j])
            _ax.imshow(_imgs[_n]); _ax.axis("off")
            _ax.set_title(f"({'abcdef'[_i + 3 * _j]}) {_n}   null {NULLS_REL[_n]:.2f}",
                          loc="left", fontsize=26, pad=14)
    plot_utils.save_figure(fig_rel_main, FIGURES / "relational_predictions_main.png", dpi=200)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Appendix. Modular addition

    A group's spectrum is flat: every irrep carries the same weight, so $E^{*}$ is fixed only up to
    a rotation within the degenerate block — which CKA does not see. With a delta target one
    frequency sufficed (a 2-D code determines $s = a+b$, and scaling the readout drives KL to 0),
    so architectures truncated to an arbitrary irrep. The lapse removes that escape: the best
    rank-2 solution now bottoms out at KL 0.56, so every frequency must be carried.
    """)
    return


@app.cell
def _(FIGURES, canonical_gauge, factor_logcond, fourier_basis, log_conditional,
      np, plot_utils):
    from src.code.data import relation_tensor

    P_MOD = 11                                        # prime: every frequency has full period p
    _T_rel = relation_tensor({"relation": "modular_add", "p": P_MOD}).reshape(P_MOD**2, P_MOD)
    # log p = -inf off the diagonal; floor it. Centering makes the floor a pure rescaling.
    M_rel = log_conditional(_T_rel)
    S_REL = (np.arange(P_MOD**2) // P_MOD + np.arange(P_MOD**2) % P_MOD) % P_MOD

    # degenerate frequencies leave the SVD basis arbitrary: re-express in the DFT gauge
    _Lc, _E, _D, _S = factor_logcond(M_rel)
    E_gauge, D_gauge = canonical_gauge(_E, _D, fourier_basis(P_MOD))
    # no frequency is privileged: show the f1 cos/sin pair, the two dimensions drawing the circle
    _ROWS = slice(0, 4 * P_MOD)                       # four a-blocks: the pattern, not all 144 rows
    _fig, _ = plot_utils.plot_factorization(M_rel[_ROWS], k=2,
                                            factors=(E_gauge[_ROWS], D_gauge),
                                            item_label=r"inputs  $(a,b)$", outcome_label=r"outputs  $c$",
                                            figsize=(16, 5.6), embedding_width=0.1,
                                            decoder_height=0.15, conditional_aspect=0.5)
    plot_utils.save_figure(_fig, FIGURES / "relational_factorization.png")
    return M_rel, P_MOD, S_REL


@app.cell
def _(
    M_rel,
    P_MOD,
    RUNNERS_REL,
    S_REL,
    circulant_fraction,
    group_fraction,
    input_code,
    load_run,
    predicted_geometry,
):
    SEEDS_REL = range(5)
    ARCHS_REL = RUNNERS_REL                          # same presets, on the default relation
    KS_REL = list(range(1, P_MOD // 2 + 1))          # k and p-k span the same real plane

    E_rel = predicted_geometry(M_rel)
    rel_codes = {}                                   # (runner, seed) -> readout-projected pair code
    for _r in ARCHS_REL:
        for _s in SEEDS_REL:
            rel_codes[_r, _s] = input_code(*load_run("modular_add", _r, seed=_s), project=True)
            print(f"{_r:20s} seed {_s}  "
                  f"circulant={circulant_fraction(rel_codes[_r, _s], S_REL):.2f}  "
                  f"s-frac={group_fraction(rel_codes[_r, _s], S_REL):.2f}", flush=True)
    return ARCHS_REL, E_rel, KS_REL, SEEDS_REL, rel_codes


@app.cell
def _(
    ARCHS_REL,
    E_rel,
    FIGURES,
    KS_REL,
    SEEDS_REL,
    SETTING_CMAP,
    S_REL,
    circulant_fraction,
    cka,
    frequency_plane,
    np,
    plot_utils,
    rel_codes,
):
    # The same panel as every other setting: predicted code beside each architecture, Grams below.
    # One representative run per arch: the median seed by circulance, not the best or the worst.
    def dominant_frequency(X, tol=0.02):
        """Frequency carrying most of the energy, or 1 when every irrep carries the same."""
        _en = {k: frequency_plane(X, S_REL, k)[1] for k in KS_REL}
        return (1 if max(_en.values()) - min(_en.values()) < tol else max(_en, key=_en.get)), _en

    def median_seed(label):
        """The seed whose circulance is the median over the sweep."""
        _o = sorted(SEEDS_REL, key=lambda s: circulant_fraction(rel_codes[label, s], S_REL))
        return _o[len(_o) // 2]

    _codes = {_l.removeprefix("rel_"): rel_codes[_l, median_seed(_l)] for _l in ARCHS_REL}
    _E = np.asarray(E_rel, float)
    # A group's spectrum is flat, so the top-2 PCA plane is an arbitrary mixture of irreps and the
    # circle disappears. Scatter the f1 plane instead; the Grams below still show the whole code.
    _coords = {_n: frequency_plane(_X, S_REL, 1)[0] for _n, _X in _codes.items()}
    _coords["predicted"] = frequency_plane(_E, S_REL, 1)[0]

    fig_rel = plot_utils.plot_embedding_panels(
        _E, _codes, cmap=SETTING_CMAP, colors=S_REL, coords=_coords,
        scores={_n: cka(_X, _E) for _n, _X in _codes.items()})
    plot_utils.save_figure(fig_rel, FIGURES / "relational_merged.png")
    return


if __name__ == "__main__":
    app.run()
