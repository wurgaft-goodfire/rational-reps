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

    from src.code.data import apply_lapse, graph_conditional
    from src.code.geometry import (predicted_geometry, factor_logcond, canonical_gauge,
                                   log_conditional,
                                   fourier_basis, circulant_fraction, class_means,
                                   frequency_plane, group_fraction)
    from src.code.eval import cka, cka_null, rsa
    from src.code.train import train_inline, graph_overrides
    from src.code import plot_utils           # importing this applies the house plotting style

    FIGURES = ROOT / "outputs" / "figures"     # one flat dir: {structure}_{figure}.png
    FIGURES.mkdir(parents=True, exist_ok=True)
    return (
        FIGURES,
        apply_lapse,
        canonical_gauge,
        cka,
        cka_null,
        factor_logcond,
        fourier_basis,
        graph_conditional,
        graph_overrides,
        log_conditional,
        np,
        plot_utils,
        predicted_geometry,
        train_inline,
    )


@app.cell
def _(FIGURES, cka, cka_null, class_means, np, plot_utils, predicted_geometry,
      train_inline):
    from src.code.analysis_utils import input_code

    ITEM_CMAP = plot_utils.FIG.item_cmap           # twilight minus its grey ends (house item map)

    def geometry_panel(name, runs, spec=None, classes=None):
        """Draw the predicted code beside each run's, and score them against it.

        ``runs`` maps a column label to that run's Hydra overrides. ``classes``, when given, is the
        answer class of each row: all rows sharing an answer sit on one point of E*, so the panels
        show one point per answer while CKA is still scored over every row, where a random code has
        far less room to match.
        """
        codes, oracle = {}, None
        for label, overrides in runs.items():
            model, dataset, _, _, _ = train_inline(overrides)
            codes[label] = input_code(model, dataset)
            if oracle is None:
                oracle = np.asarray(predicted_geometry(np.asarray(dataset.conditional, float)), float)
        scores = {label: cka(code, oracle) for label, code in codes.items()}
        # printed, not drawn: the null is how to read the scores, not part of the geometry
        print(f"{name:26s} null {cka_null(oracle):.2f}  "
              + "  ".join(f"{label} {s:.2f}" for label, s in scores.items()), flush=True)

        show = (lambda X: class_means(X, classes)[0]) if classes is not None else (lambda X: X)
        fig = plot_utils.plot_embedding_panels(show(oracle), {k: show(v) for k, v in codes.items()},
                                               spec=spec, cmap=ITEM_CMAP, scores=scores)
        return plot_utils.save_figure(fig, FIGURES / f"{name}_panels.png")
    return ITEM_CMAP, geometry_panel


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Factoring a task's conditional

    The SVD of the row-centred log-conditional splits into a code $E^{*}$ and a decoder $D^{*}$.
    """)
    return


@app.cell
def _(FIGURES, graph_conditional, plot_utils, predicted_geometry):
    # cooccurrence (diffusion | geodesic | adjacency | heat | gaussian) flows to both the
    # prediction and the training; a hypercube wants "heat" so its bits stay independent.
    # Gaussian tuning on a ring is exactly von Mises, so the log-conditional is rank 2 and the
    # top pair is the whole code -- nothing is hidden in the dimensions the schematic drops.
    spec = {"structure": "ring", "cooccurrence": "gaussian", "n": 30}
    L = graph_conditional(spec)
    E_pred = predicted_geometry(L)
    fig_fact, _ = plot_utils.plot_factorization(L, k=2)
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
def _(geometry_panel, graph_overrides):
    ARCHS = ["transformer", "linear", "mlp", "rnn"]
    SPECS = {
        "ring": {"structure": "ring", "cooccurrence": "diffusion", "n": 30},
        "grid": {"structure": "grid", "cooccurrence": "heat", "n": 5, "m": 5},   # heat: additive over the chain factors, so the lattice sides stay straight
        "helix": {"structure": "product", "cooccurrence": "heat", "view": "3d",  # line x circle
                  "factors": [{"structure": "chain", "n": 8}, {"structure": "ring", "n": 8}]},
        "hypercube": {"structure": "hypercube", "cooccurrence": "heat", "n": 3},
        "complete": {"structure": "complete", "cooccurrence": "diffusion", "n": 4},
        "tree": {"structure": "tree", "n": 2, "k": 3},
    }
    # runner defaults: readouts pointwise on the exact rows, LMs on [start, item, outcome]
    figs_settings = {
        name: geometry_panel(name, {arch: graph_overrides(spec) + [f"runner={arch}"]
                                    for arch in ARCHS}, spec=spec)
        for name, spec in SPECS.items()
    }
    return (figs_settings,)


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
def _(FIGURES, SPECS, plot_utils):
    fig_main = plot_utils.merge_panels(
        [FIGURES / "png" / f"{name}_panels.png" for name in SPECS])
    plot_utils.save_figure(fig_main, FIGURES / "geometry_predictions_main.png", dpi=200)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Relational settings

    $c = a \ast b$: the conditional is a tensor $T[a,b,c]$, so the code is read per input pair.
    Every relation carries a 5% lapse rate, which makes each outcome possible and so identifies the
    whole log-conditional — without it, KL puts no weight on a zero-probability outcome and $E^{*}$
    asserts structure the loss never constrains. Panels show one point per distinct answer.

    The factorization below reads modular addition's conditional directly: degenerate frequencies
    leave the SVD basis arbitrary, so it is shown in the DFT gauge, where the frequency-1 pair is
    the circle.

    | relation | answer | predicted geometry |
    | --- | --- | --- |
    | `approx_add` | $a+b$ under Weber noise | the same operation graded — a line in $\log(a+b)$ |
    | `pitch_transpose` | the transposed note, confusable in height and in chroma | a helix: a line plus a circle |
    | `feature_taxonomy` | is a drawn feature shared, when features come in chains? | a grid, one axis per chain |
    | `set_difference` | is a drawn feature in $a$ but not $b$? | a box, asymmetric, axes scaled by salience |
    | `entailment` | a common ancestor of $a, b$ | the tree, read as an ultrametric over the leaves |
    """)
    return


@app.cell
def _(
    FIGURES,
    apply_lapse,
    canonical_gauge,
    factor_logcond,
    fourier_basis,
    log_conditional,
    np,
    plot_utils,
):
    from src.code.data import relation_tensor

    P_MOD = 11                                        # prime: every frequency has full period p
    modular = relation_tensor({"relation": "modular_add", "p": P_MOD}).reshape(P_MOD**2, P_MOD)
    modular_log = log_conditional(apply_lapse(modular))

    # degenerate frequencies leave the SVD basis arbitrary: re-express in the DFT gauge, then show
    # the f1 cos/sin pair -- the two dimensions that draw the circle. Four a-blocks, not all 121 rows.
    code, decoder = canonical_gauge(*factor_logcond(modular_log)[1:3], fourier_basis(P_MOD))
    rows = slice(0, 4 * P_MOD)
    fig_fact_rel, _ = plot_utils.plot_factorization(
        modular_log[rows], k=2, factors=(code[rows], decoder),
        item_label=r"inputs  $(a,b)$", outcome_label=r"outputs  $c$",
        figsize=(16, 5.6), embedding_width=0.1, decoder_height=0.15, conditional_aspect=0.5)
    plot_utils.save_figure(fig_fact_rel, FIGURES / "relational_factorization.png")
    return


@app.cell
def _(FIGURES, geometry_panel, plot_utils):
    from src.code.data import answer_classes

    # every knob (width, decay, input format, lapse) lives in the preset, so a run is named by the
    # relation and the runner alone and nothing here can drift from what the CLI trained
    RUNNERS = ["rel_transformer", "rel_rnn", "rel_mlp", "rel_linear_paired", "rel_mlp_paired"]
    # Only graded relations are scored: a delta answer makes every outcome equidistant, so its pair
    # code is a simplex with no geometry to predict. Modular addition is that case, and stays as
    # the factorization above, where the DFT gauge is the point.
    RELATIONS = ["approx_add", "pitch_transpose", "between",
                 "feature_taxonomy", "set_difference", "entailment"]

    figs_rel = {}
    for relation in RELATIONS:
        classes, draw_spec = answer_classes(relation)
        runs = {runner.removeprefix("rel_"): [f"runner={runner}", f"relation={relation}",
                                              "random_seed=0"] for runner in RUNNERS}
        figs_rel[relation] = geometry_panel(f"relational_{relation}", runs,
                                            spec=draw_spec, classes=classes)

    fig_rel_main = plot_utils.merge_panels(
        [FIGURES / "png" / f"relational_{relation}_panels.png" for relation in RELATIONS],
        labels=RELATIONS)
    plot_utils.save_figure(fig_rel_main, FIGURES / "relational_predictions_main.png", dpi=200)
    return (figs_rel,)


@app.cell
def _(figs_rel, mo):
    mo.ui.tabs(figs_rel)
    return


if __name__ == "__main__":
    app.run()
