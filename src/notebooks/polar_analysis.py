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
    # Polar codes on a labeled tree

    A depth-3, branching-3 tree whose edges carry one of three labels (birth order). Node positions
    $p(x) = \sum_{e \in \mathrm{path}(x)} (\mathbf{e}_e + \sqrt{\alpha}\, \mathbf{u}_{\ell(e)})$ and
    $\log p(y \mid x) = -\tfrac{\gamma}{2}\lVert p(x) - p(y) \rVert^2$. Two worlds: labels uninformative
    ($\alpha = 0$) and informative ($\alpha = 1$). Four architectures per world, full width, SGD.
    """)
    return


@app.cell
def _(mo):
    import os, sys
    import numpy as np

    ROOT = mo.notebook_dir().parent.parent    # src/notebooks -> <repo>
    os.chdir(ROOT); sys.path.insert(0, str(ROOT))
    from src.code.analysis_utils import cka, load_run, rsa
    from src.code.data import labeled_tree, tree_parents
    from src.code.geometry import edge_cosines, edge_vectors, predicted_geometry
    from src.code.plot_utils import FIG, figure_path, save_figure
    import matplotlib.pyplot as plt

    return (
        FIG,
        cka,
        edge_cosines,
        edge_vectors,
        figure_path,
        labeled_tree,
        load_run,
        np,
        plt,
        predicted_geometry,
        rsa,
        save_figure,
        tree_parents,
    )


@app.cell
def _(labeled_tree, load_run, np, predicted_geometry, tree_parents):
    WORLDS = {"α = 0": ["structure_kwargs.alpha=0.0", "structure_kwargs.gamma=1.33"], "α = 1": []}
    ARCHS = {"linear": 0.5, "mlp": 0.1, "rnn": 0.5, "transformer": 0.1}      # SGD learning rate
    WIDTH = {"linear": "embed_dim", "mlp": "embed_dim", "rnn": "embed_dim", "transformer": "hidden_size"}

    def overrides(arch, world):
        return [f"runner={arch}", "setting=labeled_tree", "training_kwargs.optimizer=sgd",
                "training_kwargs.train_steps=60000", f"training_kwargs.learning_rate={ARCHS[arch]}",
                f"model_kwargs.model_config.{WIDTH[arch]}=64"] + WORLDS[world]

    PARENTS = tree_parents(n=3, k=3)
    LABELS = np.array([PARENTS[1:c].count(PARENTS[c]) for c in range(1, len(PARENTS))])   # one per edge
    ORDER = np.argsort(LABELS, kind="stable")                                           # edges grouped by label

    runs = {w: {a: load_run(overrides(a, w)) for a in ARCHS} for w in WORLDS}
    oracle = {w: predicted_geometry(labeled_tree(**dict(next(iter(runs[w].values()))["config"].cfg.structure_kwargs)))
              for w in WORLDS}
    codes = {w: {"predicted": oracle[w], **{a: r["representation"] for a, r in runs[w].items()}} for w in WORLDS}
    for _w in WORLDS:                              # full rank: every model reads as many directions as the code has
        _rank = np.linalg.matrix_rank(oracle[_w], tol=1e-6 * np.abs(oracle[_w]).max())
        for _a, _r in runs[_w].items():
            assert _r["representation"].shape[1] >= _rank, f"{_w} {_a}: width {_r['representation'].shape[1]} < rank {_rank}"
    return ARCHS, LABELS, ORDER, PARENTS, WORLDS, codes, oracle, runs


@app.cell
def _(LABELS, PARENTS, WORLDS, cka, codes, edge_cosines, edge_vectors, np, oracle, rsa, runs):
    _same = (LABELS[:, None] == LABELS[None]) & ~np.eye(len(LABELS), dtype=bool)
    _diff = LABELS[:, None] != LABELS[None]
    print(f"{'world':6s} {'model':12s} {'KL':>8s} {'RSA':>6s} {'CKA':>6s} {'same cos':>9s} {'diff cos':>9s} {'|edge| cv':>9s}")
    for _w in WORLDS:
        for _name, _E in codes[_w].items():
            _C = edge_cosines(_E, PARENTS)
            _n = np.linalg.norm(edge_vectors(_E, PARENTS), axis=1)
            _kl = runs[_w][_name]["loss"] if _name in runs[_w] else 0.0
            print(f"{_w:6s} {_name:12s} {_kl:8.1e} {rsa(_E, oracle[_w]):6.3f} {cka(_E, oracle[_w]):6.3f} "
                  f"{_C[_same].mean():9.3f} {_C[_diff].mean():9.3f} {_n.std() / _n.mean():9.3f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 1 — edge cosines

    Cosine between every pair of edge vectors (child minus parent code), edges grouped by label; rows are
    the two worlds, columns the predicted code and each trained model.
    """)
    return


@app.cell
def _(FIG, ORDER, PARENTS, WORLDS, codes, edge_cosines, figure_path, plt, runs, save_figure):
    _names = list(codes["α = 1"])
    _fig, _axes = plt.subplots(len(WORLDS), len(_names), figsize=(2.6 * len(_names), 2.9 * len(WORLDS)))
    for _i, _w in enumerate(WORLDS):
        for _j, _name in enumerate(_names):
            _ax = _axes[_i, _j]
            _ax.imshow(edge_cosines(codes[_w][_name], PARENTS)[ORDER][:, ORDER], cmap=FIG.diverging_cmap,
                       vmin=-1, vmax=1)
            _ax.set_xticks([]); _ax.set_yticks([])
            _ax.set_title(_name if _i == 0 else "")
        _axes[_i, 0].set_ylabel(_w.replace("α", r"$\alpha$"))
    save_figure(_fig, figure_path(runs["α = 1"]["linear"]["config"], "edge_cosines"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 2 — edge vectors in the label plane

    Held-out edge vectors (odd edges) projected onto the plane of the three label-mean edge vectors of the
    even edges (centered), colored by label, lengths in units of the mean edge length; same layout as Figure 1.
    """)
    return


@app.cell
def _(FIG, LABELS, PARENTS, WORLDS, codes, edge_vectors, figure_path, np, plt, runs, save_figure):
    _colors = [FIG.colors["blue"], FIG.colors["orange"], FIG.colors["green"]]
    _names = list(codes["α = 1"])
    _fig, _axes = plt.subplots(len(WORLDS), len(_names), figsize=(2.6 * len(_names), 2.9 * len(WORLDS)))
    for _i, _w in enumerate(WORLDS):
        for _j, _name in enumerate(_names):
            _V = edge_vectors(codes[_w][_name], PARENTS)
            _fit = np.arange(len(_V)) % 2 == 0                                   # plane from half the edges,
            _M = np.array([_V[_fit & (LABELS == l)].mean(0) for l in range(3)])   # drawn with the other half
            _B = np.linalg.svd(_M - _M.mean(0), full_matrices=False)[2][:2]
            _XY = _V[~_fit] @ _B.T / np.linalg.norm(_V, axis=1).mean()           # unit mean edge length
            _ax = _axes[_i, _j]
            for (_x, _y), _l in zip(_XY, LABELS[~_fit]):
                _ax.plot([0, _x], [0, _y], color=_colors[_l], lw=1.2, alpha=0.8)
            _ax.set_xlim(-1.05, 1.05); _ax.set_ylim(-1.05, 1.05); _ax.set_aspect("equal")
            _ax.set_xticks([]); _ax.set_yticks([])
            _ax.set_title(_name if _i == 0 else "")
        _axes[_i, 0].set_ylabel(_w.replace("α", r"$\alpha$"))
    save_figure(_fig, figure_path(runs["α = 1"]["linear"]["config"], "edge_label_plane"))
    return


if __name__ == "__main__":
    app.run()
