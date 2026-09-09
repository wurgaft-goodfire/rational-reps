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
    import io, os, sys
    import numpy as np

    # Project root, independent of the working directory (a headless run can start outside
    # the repo). chdir so Hydra resolves conf/, and add it to sys.path.
    ROOT = mo.notebook_dir().parent.parent    # src/notebooks -> <repo>
    os.chdir(ROOT); sys.path.insert(0, str(ROOT))

    from src.code.data import apply_lapse, graph_conditional
    from src.code.geometry import predicted_geometry, factor_logcond, log_conditional, class_means
    from src.code.analysis_utils import cka, cka_null, cka_shuffle_null, load_run, rsa
    from src.code.train import graph_overrides
    from src.code import plot_utils           # importing this applies the house plotting style

    from pathlib import Path
    FIGURES = Path(os.environ.get("CACHE_DIR", ROOT / "outputs")) / "figures"   # one flat dir: {structure}_{figure}.png
    FIGURES.mkdir(parents=True, exist_ok=True)
    return (
        FIGURES,
        apply_lapse,
        cka,
        cka_null,
        cka_shuffle_null,
        class_means,
        factor_logcond,
        graph_conditional,
        graph_overrides,
        io,
        load_run,
        log_conditional,
        np,
        plot_utils,
        predicted_geometry,
    )


@app.cell
def _(
    cka,
    cka_null,
    cka_shuffle_null,
    class_means,
    io,
    load_run,
    np,
    plot_utils,
    predicted_geometry,
):
    ITEM_CMAP = plot_utils.FIG.item_cmap           # twilight minus its grey ends (house item map)

    def as_block(fig, dpi=600):
        """A figure rendered to an in-memory png, for `merge_panels`; only the merged figure is saved."""
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight"); plot_utils.plt.close(fig)
        buf.seek(0); return buf

    def geometry_panel(name, runs, spec=None, classes=None):
        """Draw the predicted code beside each run's, and score them against it; returns the block as an
        in-memory png.

        ``runs`` maps a column label to that run's Hydra overrides; each run is read from its saved
        evaluation, and its representation is the raw last hidden state. ``classes``, when given, is
        the answer class of each row: all rows sharing an answer sit on one point of E*, so the
        panels show one point per answer while CKA is still scored over every row, where a random
        code has far less room to match.
        """
        codes, oracle = {}, None
        for label, overrides in runs.items():
            run = load_run(overrides)
            codes[label] = run["representation"]
            if oracle is None:
                oracle = np.asarray(predicted_geometry(np.asarray(run["dataset"].conditional, float)), float)
        scores = {label: cka(code, oracle) for label, code in codes.items()}
        print(f"{name:26s} rand {cka_null(oracle):.2f} shuf {cka_shuffle_null(oracle):.2f}  "
              + "  ".join(f"{label} {s:.2f}" for label, s in scores.items()), flush=True)

        show = (lambda X: class_means(X, classes)[0]) if classes is not None else (lambda X: X)
        fig = plot_utils.plot_embedding_panels(show(oracle), {k: show(v) for k, v in codes.items()},
                                               spec=spec, cmap=ITEM_CMAP, scores=scores)
        return as_block(fig)

    return as_block, geometry_panel


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Factoring a task's conditional

    The SVD of the row-centred log-conditional splits into a code $E^{*}$ and a decoder $D^{*}$.
    """)
    return


@app.cell
def _(FIGURES, as_block, graph_conditional, plot_utils, predicted_geometry):
    # cooccurrence (diffusion | geodesic | adjacency | heat | gaussian) flows to both the
    # prediction and the training; a hypercube wants "heat" so its bits stay independent.
    # Gaussian tuning on a ring is exactly von Mises, so the log-conditional is rank 2 and the
    # top pair is the whole code -- nothing is hidden in the dimensions the schematic drops.
    spec = {"structure": "ring", "cooccurrence": "gaussian", "n": 30}
    L = graph_conditional(spec)
    E_pred = predicted_geometry(L)
    fig_fact, _ = plot_utils.plot_factorization(L, k=2)
    plot_utils.save_figure(fig_fact, FIGURES / f"{spec['structure']}_factorization.png")
    # the same factorization, wide, as the header of the merged geometry figure (rendered large so its
    # fonts land near the blocks' after scaling)
    header_ring = as_block(plot_utils.plot_factorization(L, k=2, figsize=(26, 5.2), font_scale=2.0)[0])
    return E_pred, L, header_ring, spec


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
def _(FIGURES, geometry_panel, graph_overrides, header_ring, plot_utils):
    ARCHS = ["transformer", "linear", "mlp", "rnn"]
    SPECS = {
        "ring": {"structure": "ring", "cooccurrence": "gaussian", "n": 30},   # von Mises: exactly rank 2, the header's factorization
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

    # Merge the panels into one figure
    fig_main = plot_utils.merge_panels(list(figs_settings.values()), header=header_ring)
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

    The factorization below reads `pitch_transpose`'s conditional directly. It is exactly rank
    four: height and the curvature that bends the height line into an arc (the Gaussian's linear
    and quadratic terms), plus the chroma pair.

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
    as_block,
    factor_logcond,
    log_conditional,
    plot_utils,
):
    from src.code.data import relation_tensor

    # the helix relation's conditional, every (note, interval) pair as a row, factored whole; the
    # first four note blocks are drawn, not all 144 rows, so the columns stay readable
    pitch = relation_tensor({"relation": "pitch_transpose"})
    pitch_log = log_conditional(apply_lapse(pitch.reshape(-1, pitch.shape[-1])))
    rows = slice(0, 4 * pitch.shape[1])
    _, E_pitch, D_pitch, _ = factor_logcond(pitch_log)                 # the whole task's factors; only the first rows are drawn
    fig_fact_rel, _ = plot_utils.plot_factorization(
        pitch_log[rows], k=4, factors=(E_pitch[rows], D_pitch), item_label=r"inputs  $(a,b)$", outcome_label=r"outputs  $c$",
        figsize=(16, 5.6), embedding_width=0.1, decoder_height=0.15, conditional_aspect=0.5)
    plot_utils.save_figure(fig_fact_rel, FIGURES / "relational_factorization.png")

    # approx_add's conditional, exactly rank 2 (the log(a+b) line and its curvature), as the header of the
    # merged relational figure: factored whole, the first four a-blocks drawn
    add = relation_tensor({"relation": "approx_add"})
    add_log = log_conditional(apply_lapse(add.reshape(-1, add.shape[-1])))
    _, E_add, D_add, _ = factor_logcond(add_log)
    header_add = as_block(plot_utils.plot_factorization(
        add_log[rows], k=2, factors=(E_add[rows], D_add), item_label=r"inputs  $(a,b)$", outcome_label=r"outputs  $c$",
        figsize=(26, 5.2), embedding_width=0.1, decoder_height=0.15, conditional_aspect=0.5, font_scale=2.0)[0])
    return (header_add,)


@app.cell
def _(FIGURES, geometry_panel, header_add, plot_utils):
    from src.code.data import answer_classes

    RUNNERS = {"rel_transformer": "transformer", "rel_rnn": "rnn", "rel_mlp": "mlp", "rel_linear_paired": "linear"}
    RELATIONS = ["approx_add", "pitch_transpose", "between",
                 "feature_taxonomy", "set_difference", "entailment"]

    figs_rel = {}
    for relation in RELATIONS:
        classes, draw_spec = answer_classes(relation)
        runs = {label: [f"runner={runner}", f"relation={relation}"] for runner, label in RUNNERS.items()}
        figs_rel[relation] = geometry_panel(f"relational_{relation}", runs,
                                            spec=draw_spec, classes=classes)

    fig_rel_main = plot_utils.merge_panels(list(figs_rel.values()), header=header_add)
    plot_utils.save_figure(fig_rel_main, FIGURES / "relational_predictions_main.png", dpi=200)
    return


if __name__ == "__main__":
    app.run()
