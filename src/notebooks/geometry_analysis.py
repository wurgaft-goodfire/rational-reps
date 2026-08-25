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

    The project's claim: the geometry a network learns is **predictable in closed form** from the
    structure of the task.
    """)
    return


app._unparsable_cell(
    r"""
    import os, sys
    imporst numpy as np

    # Project root, independent of the working directory (a headless run can start outside
    # the repo). chdir so Hydra resolves conf/, and add it to sys.path.
    ROOT = mo.notebook_dir().parent.parent    # src/notebooks -> <repo>
    os.chdir(ROOT); sys.path.insert(0, str(ROOT))

    from src.code.data import graph_conditional
    from src.code.geometry import predicted_geometry
    from src.code.eval import cka, rsa
    from src.code.train import train_inline, graph_overrides
    from src.code import plot_utils           # importing this applies the house plotting style

    FIGURES = ROOT / "outputs" / "figures"     # one flat dir: {structure}_{figure}.png
    FIGURES.mkdir(parents=True, exist_ok=True)
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. The predicted geometry (closed form)

    Factor the ring's log-conditional. The translation symmetry gives a
    top Fourier pair, and the top-2 code lands on a circle.
    """)
    return


@app.cell
def _(FIGURES, graph_conditional, plot_utils, predicted_geometry):
    # pick a structure here — everything below follows from it. `cooccurrence` is part of the spec
    # (diffusion | geodesic | adjacency | heat) and flows to both the predicted geometry and the trained
    # models; a product diffuses, but e.g. a hypercube wants "heat" so its bits stay independent (analogies).
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

    The structure, the log-conditional it induces, and the closed-form embedding: the ring's
    top-2 code lands on a circle.
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
    ## 3. Train everything: the prediction across settings

    Loop over structures: train all four architectures per structure (hparams per arch from a
    ring sweep: the sequence models want longer training and a lower lr), then save that
    structure's embedding-panel figure into its own run directory. One shared point colormap
    (half of twilight, acyclic); matrix row = the code Gram, $E^{*}E^{*\top}$ predicted beside
    each learned $EE^{\top}$.
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
            _reps[_arch] = _ev.item_representations(_model)
            print(f"{_sp['structure']:10s} {_arch:12s} "
                  f"CKA={cka(_reps[_arch], predicted_geometry(_ds.conditional)):.3f}")
        _fig = plot_utils.plot_embedding_panels(predicted_geometry(_L), _reps, spec=_sp,
                                                cmap=SETTING_CMAP)
        _key = "helix" if _sp["structure"] == "product" else _sp["structure"]
        plot_utils.save_figure(_fig, FIGURES / f"{_key}_embedding_panels.png")
        figs_settings[_key] = _fig
    return (figs_settings,)


@app.cell
def _(figs_settings, mo):
    mo.ui.tabs(figs_settings)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Main figure: the six settings merged

    The per-setting embedding panels tiled (3, 2): ring, grid, helix down the first column;
    hypercube, simplex, hierarchy down the second.
    """)
    return


@app.cell
def _(FIGURES, plot_utils):
    import matplotlib.image as mpimg

    # letters follow the columns: (a-c) ring, grid, helix; (d-f) hypercube, simplex, hierarchy
    _grid = [["ring", "hypercube"], ["grid", "complete"], ["helix", "tree"]]
    _imgs = {_n: mpimg.imread(FIGURES / f"{_n}_embedding_panels.png") for _n in sum(_grid, [])}
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
    return


if __name__ == "__main__":
    app.run()
