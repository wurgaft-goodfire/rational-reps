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

    Node positions $p(x) = \sum_{e \in \mathrm{path}(x)} (\mathbf{e}_e + \sqrt{\alpha}\, \mathbf{u}_{\ell(e)})$ and
    $\log p(y \mid x) = -\tfrac{\gamma}{2}\lVert p(x) - p(y) \rVert^2$. Two worlds: labels uninformative
    ($\alpha = 0$) and informative ($\alpha = 1$); every model full width, SGD. Two trees: depth 3 with three
    labels (birth order), and ego's family (links up to a parent, to a sibling, down to a child; the label
    statistic is generation, so up and down links share one direction with opposite signs).
    """)
    return


@app.cell
def _(mo):
    import os, sys
    import numpy as np

    ROOT = mo.notebook_dir().parent.parent    # src/notebooks -> <repo>
    os.chdir(ROOT); sys.path.insert(0, str(ROOT))
    from src.code.analysis_utils import align_representations, cka, load_run, rsa
    from src.code.data import labeled_tree, tree_parents
    from src.code.geometry import (class_means, edge_cosines, edge_vectors, factor_logcond, predicted_geometry,
                                   predicted_inference_geometry)
    from src.code.plot_utils import FIG, figure_path, save_figure
    import matplotlib.pyplot as plt

    return (
        FIG,
        align_representations,
        class_means,
        cka,
        edge_cosines,
        edge_vectors,
        factor_logcond,
        figure_path,
        labeled_tree,
        load_run,
        np,
        plt,
        predicted_geometry,
        predicted_inference_geometry,
        rsa,
        save_figure,
        tree_parents,
    )


@app.cell
def _(class_means, labeled_tree, load_run, np, predicted_geometry, tree_parents):
    WORLDS = {"α = 0": ["structure_kwargs.alpha=0.0", "structure_kwargs.gamma=1.33"], "α = 1": []}
    LR = {"linear": 0.5, "mlp": 0.1, "rnn": 0.5, "transformer": 0.1}          # SGD learning rate
    WIDTH = {"linear": "embed_dim", "mlp": "embed_dim", "rnn": "embed_dim", "transformer": "hidden_size"}

    def overrides(model, world, setting):
        arch, sequence = model.split(" ")[0], model.endswith("(stream)")
        return ([f"runner={arch}", f"setting={setting}", "training_kwargs.optimizer=sgd",
                 "training_kwargs.train_steps=60000", f"training_kwargs.learning_rate={LR[arch]}",
                 f"model_kwargs.model_config.{WIDTH[arch]}=64"]
                + (["sample_mode=sequence"] if sequence else []) + WORLDS[world])

    def node_code(run, n):
        """One vector per node: the representation itself (pointwise), or its mean over the stream positions
        where the node is the current token (sequence)."""
        R = run["representation"]
        if R.ndim == 2:
            return R
        tokens = np.asarray(run["inputs"])[:, -R.shape[1]:]
        keep = tokens < n                                                     # drop the start token
        return class_means(R[keep], tokens[keep], n)[0]

    def load_tree(setting, models):
        """Oracle and trained node codes for a labeled-tree setting, per world."""
        kw = dict(load_run(overrides(models[0], "α = 1", setting))["config"].cfg.structure_kwargs)
        parents = tree_parents(kw.get("n", 2), kw.get("k", 3), kw.get("parents"))
        labels = (np.array(kw["labels"][1:]) if kw.get("labels") is not None
                  else np.array([parents[1:c].count(parents[c]) for c in range(1, len(parents))]))   # one per edge
        runs = {w: {m: load_run(overrides(m, w, setting)) for m in models} for w in WORLDS}
        codes = {}
        for w in WORLDS:
            E = predicted_geometry(labeled_tree(**dict(next(iter(runs[w].values()))["config"].cfg.structure_kwargs)))
            rank = np.linalg.matrix_rank(E, tol=1e-6 * np.abs(E).max())
            codes[w] = {"predicted": E}
            for m, r in runs[w].items():                  # full rank: every model reads as many directions as the code has
                assert r["representation"].shape[-1] >= rank, f"{w} {m}: width {r['representation'].shape[-1]} < rank {rank}"
                codes[w][m] = node_code(r, len(parents))
        return dict(parents=parents, labels=labels, order=np.argsort(labels, kind="stable"), runs=runs, codes=codes,
                    config=runs["α = 1"][models[0]]["config"])

    POINTWISE = ["linear", "mlp", "rnn", "transformer"]
    three = load_tree("labeled_tree", POINTWISE)
    family = load_tree("family_tree", POINTWISE + ["rnn (stream)", "transformer (stream)"])
    return WORLDS, family, three


@app.cell
def _(WORLDS, cka, edge_cosines, edge_vectors, np, rsa):
    def scores(tree):
        lab = tree["labels"]
        same, diff = (lab[:, None] == lab[None]) & ~np.eye(len(lab), dtype=bool), lab[:, None] != lab[None]
        print(f"{'world':6s} {'model':22s} {'KL':>8s} {'RSA':>6s} {'CKA':>6s} {'same cos':>9s} {'diff cos':>9s} {'|edge| cv':>9s}")
        for w in WORLDS:
            E0 = tree["codes"][w]["predicted"]
            for name, E in tree["codes"][w].items():
                C, n = edge_cosines(E, tree["parents"]), np.linalg.norm(edge_vectors(E, tree["parents"]), axis=1)
                kl = tree["runs"][w][name]["loss"] if name in tree["runs"][w] else 0.0
                print(f"{w:6s} {name:22s} {kl:8.1e} {rsa(E, E0):6.3f} {cka(E, E0):6.3f} "
                      f"{C[same].mean():9.3f} {C[diff].mean():9.3f} {n.std() / n.mean():9.3f}")

    return (scores,)


@app.cell
def _(FIG, WORLDS, edge_cosines, edge_vectors, figure_path, np, plt, save_figure):
    def _grid(tree):
        names = list(tree["codes"]["α = 1"])
        fig, axes = plt.subplots(len(WORLDS), len(names), figsize=(2.6 * len(names), 2.9 * len(WORLDS)))
        return names, fig, axes

    def _label_rows(axes, names):
        for i, w in enumerate(WORLDS):
            axes[i, 0].set_ylabel(w.replace("α", r"$\alpha$"))
        for j, name in enumerate(names):
            axes[0, j].set_title(name)

    def plot_edge_cosines(tree, name):
        names, fig, axes = _grid(tree)
        for i, w in enumerate(WORLDS):
            for j, m in enumerate(names):
                C = edge_cosines(tree["codes"][w][m], tree["parents"])[tree["order"]][:, tree["order"]]
                axes[i, j].imshow(C, cmap=FIG.diverging_cmap, vmin=-1, vmax=1)
                axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
        _label_rows(axes, names)
        return save_figure(fig, figure_path(tree["config"], name))

    def plot_label_plane(tree, name):
        colors = [FIG.colors["blue"], FIG.colors["orange"], FIG.colors["green"]]
        lab, k = tree["labels"], tree["labels"].max() + 1
        names, fig, axes = _grid(tree)
        for i, w in enumerate(WORLDS):
            for j, m in enumerate(names):
                V = edge_vectors(tree["codes"][w][m], tree["parents"])
                fit = np.arange(len(V)) % 2 == 0                                  # plane from half the edges,
                M = np.array([V[fit & (lab == l)].mean(0) for l in range(k)])      # drawn with the other half
                B = np.linalg.svd(M if k == 2 else M - M.mean(0), full_matrices=False)[2][:2]
                XY = V[~fit] @ B.T / np.linalg.norm(V, axis=1).mean()             # unit mean edge length
                for (x, y), l in zip(XY, lab[~fit]):
                    axes[i, j].plot([0, x], [0, y], color=colors[l], lw=1.2, alpha=0.8)
                axes[i, j].set_xlim(-1.05, 1.05); axes[i, j].set_ylim(-1.05, 1.05); axes[i, j].set_aspect("equal")
                axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
        _label_rows(axes, names)
        return save_figure(fig, figure_path(tree["config"], name))

    return plot_edge_cosines, plot_label_plane


@app.cell
def _(scores, three):
    scores(three)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 1 — edge cosines, three labels

    Cosine between every pair of edge vectors (child minus parent code), edges grouped by label; rows are
    the two worlds, columns the predicted code and each trained model.
    """)
    return


@app.cell
def _(plot_edge_cosines, three):
    plot_edge_cosines(three, "edge_cosines")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 2 — edge vectors in the label plane, three labels

    Held-out edge vectors (odd edges) projected onto the plane of the three label-mean edge vectors of the
    even edges (centered), colored by label, lengths in units of the mean edge length; same layout as Figure 1.
    """)
    return


@app.cell
def _(plot_label_plane, three):
    plot_label_plane(three, "edge_label_plane")
    return


@app.cell
def _(family, scores):
    scores(family)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 3 — edge cosines, family tree

    As Figure 1 for ego's family (edges grouped into up, sibling and down links); the last two columns are
    models reading a stream of co-mentions, one code per person averaged over the positions where they are
    mentioned.
    """)
    return


@app.cell
def _(family, plot_edge_cosines):
    plot_edge_cosines(family, "edge_cosines")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 4 — family links along the generation axis

    Held-out links (odd edges) projected onto the axis from the mean down link to the mean up link of the even
    edges (x) and the top principal direction of the even edges orthogonal to it (y); blue up, orange sibling,
    green down, lengths in units of the mean link length.
    """)
    return


@app.cell
def _(FIG, WORLDS, edge_vectors, family, figure_path, np, plt, save_figure):
    _colors = [FIG.colors["blue"], FIG.colors["orange"], FIG.colors["green"]]
    _lab, _names = family["labels"], list(family["codes"]["α = 1"])
    _fig, _axes = plt.subplots(len(WORLDS), len(_names), figsize=(2.6 * len(_names), 2.9 * len(WORLDS)))
    for _i, _w in enumerate(WORLDS):
        for _j, _m in enumerate(_names):
            _V = edge_vectors(family["codes"][_w][_m], family["parents"])
            _fit = np.arange(len(_V)) % 2 == 0
            _x = _V[_fit & (_lab == 0)].mean(0) - _V[_fit & (_lab == 2)].mean(0); _x /= np.linalg.norm(_x)
            _R = _V[_fit] - np.outer(_V[_fit] @ _x, _x)
            _y = np.linalg.svd(_R, full_matrices=False)[2][0]
            _XY = np.c_[_V[~_fit] @ _x, _V[~_fit] @ _y] / np.linalg.norm(_V, axis=1).mean()
            for (_a, _b), _l in zip(_XY, _lab[~_fit]):
                _axes[_i, _j].plot([0, _a], [0, _b], color=_colors[_l], lw=1.4, alpha=0.8)
            _axes[_i, _j].set_xlim(-1.05, 1.05); _axes[_i, _j].set_ylim(-1.05, 1.05); _axes[_i, _j].set_aspect("equal")
            _axes[_i, _j].set_xticks([]); _axes[_i, _j].set_yticks([])
            _axes[0, _j].set_title(_m)
        _axes[_i, 0].set_ylabel(_w.replace("α", r"$\alpha$"))
    save_figure(_fig, figure_path(family["config"], "generation_axis"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Scope-stack programs

    Streams of code-like tokens: `for` / `def` / `if` / `with` open a block that binds a fresh name, `end` closes
    it, and every open scope adds its effect to the next-token scores. Kinds inert: a block's kind changes no
    token; kinds act: `for` raises break / continue, `def` return / yield, `if` pass. Width-64 transformer and GRU, next-token prediction, SGD; the predicted code is the factored exact next-token rows.
    """)
    return


@app.cell
def _(align_representations, cka, factor_logcond, load_run, np, predicted_inference_geometry, rsa):
    SCOPE_WORLDS = {"kinds inert": "0.0", "kinds act": "1.0"}
    SCOPE_MODELS = {"transformer": ("0.1", "hidden_size"), "rnn": ("0.5", "embed_dim")}

    def scope_overrides(arch, world):
        lr, width = SCOPE_MODELS[arch]
        return [f"runner={arch}", "setting=scopes", "training_kwargs.context_length=64", "training_kwargs.optimizer=sgd",
                "training_kwargs.train_steps=60000", f"training_kwargs.learning_rate={lr}",
                f"model_kwargs.model_config.{width}=64", f"scope_kwargs.kind_strength={SCOPE_WORLDS[world]}"]

    scopes = {}
    for _w in SCOPE_WORLDS:
        _runs = {a: load_run(scope_overrides(a, _w)) for a in SCOPE_MODELS}
        _r0 = _runs["transformer"]
        _P = _r0["targets"].astype(np.float64)                                   # (B, T, V) exact next-token rows
        _R = np.log(_P).reshape(-1, _P.shape[-1])
        _, _, _D, _S = factor_logcond(_R)
        _rank = int((_S > 1e-8 * _S[0]).sum())
        _codes = {"predicted": predicted_inference_geometry(_R, _D[:_rank]).reshape(_P.shape[0], _P.shape[1], _rank)}
        for _a, _r in _runs.items():
            assert _r["representation"].shape[-1] >= _rank, f"{_w} {_a}: width < rank {_rank}"
            _codes[_a] = _r["representation"][:, :_P.shape[1]]                   # position t predicts token t+1
        scopes[_w] = dict(runs=_runs, codes=_codes, dataset=_r0["dataset"], inputs=_r0["inputs"])

    _rng = np.random.default_rng(0)
    print(f"{'world':12s} {'model':12s} {'KL':>8s} {'RSA':>6s} {'CKA':>6s} {'affine R2':>9s} | "
          f"{'same kind':>9s} {'diff kind':>9s} {'same name':>9s} {'|step| cv':>9s}")
    for _w, _s in scopes.items():
        _E0 = _s["codes"]["predicted"].reshape(-1, _s["codes"]["predicted"].shape[-1])
        _idx = _rng.choice(len(_E0), 3000, replace=False)
        for _m, _H in _s["codes"].items():
            _X = _H.reshape(-1, _H.shape[-1])
            _ent = _s["dataset"].block_entries(_s["inputs"], _H)
            _k = np.array([e[0] for e in _ent]); _n = np.array([e[1] for e in _ent])
            _U = np.array([e[3] for e in _ent]); _len = np.linalg.norm(_U, axis=1); _U = _U / _len[:, None]
            _C = _U @ _U.T
            _sk, _sn = _k[:, None] == _k[None], _n[:, None] == _n[None]
            _kl = _s["runs"][_m]["loss"] if _m in _s["runs"] else 0.0
            print(f"{_w:12s} {_m:12s} {_kl:8.1e} {rsa(_X[_idx], _E0[_idx]):6.3f} {cka(_X[_idx], _E0[_idx]):6.3f} "
                  f"{align_representations(_E0[_idx], _X[_idx], 'linear')[1]:9.3f} | {_C[_sk & ~_sn].mean():9.3f} "
                  f"{_C[~_sk & ~_sn].mean():9.3f} {_C[~_sk & _sn].mean():9.3f} {_len.std() / _len.mean():9.3f}")
    return SCOPE_WORLDS, scopes


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 5 — block-entry steps

    Cosine between the mean block-entry steps (code after the bound name minus code before the kind token), one
    per (kind, name), sorted by kind then name; rows are the two worlds, columns the predicted code and each
    trained model.
    """)
    return


@app.cell
def _(FIG, SCOPE_WORLDS, figure_path, np, plt, save_figure, scopes):
    _names = list(scopes["kinds act"]["codes"])
    _fig, _axes = plt.subplots(len(SCOPE_WORLDS), len(_names), figsize=(2.9 * len(_names), 3.1 * len(SCOPE_WORLDS)))
    for _i, _w in enumerate(SCOPE_WORLDS):
        _ds = scopes[_w]["dataset"]
        _keys = [(k, n) for k in _ds.kinds for n in _ds.names]
        for _j, _m in enumerate(_names):
            _ent = _ds.block_entries(scopes[_w]["inputs"], scopes[_w]["codes"][_m])
            _steps = {}
            for _k, _n, _, _v in _ent:
                _steps.setdefault((_k, _n), []).append(_v)
            _M = np.array([np.mean(_steps[key], 0) if key in _steps else np.full(len(_ent[0][3]), np.nan) for key in _keys])
            _M = _M / np.linalg.norm(_M, axis=1, keepdims=True)
            _ax = _axes[_i, _j]
            _ax.imshow(_M @ _M.T, cmap=FIG.diverging_cmap, vmin=-1, vmax=1)
            _ticks = np.arange(len(_ds.kinds)) * len(_ds.names) + len(_ds.names) / 2 - 0.5
            _ax.set_xticks(_ticks, _ds.kinds if _i == len(SCOPE_WORLDS) - 1 else [])
            _ax.set_yticks(_ticks, _ds.kinds if _j == 0 else [])
            _axes[0, _j].set_title(_m)
        _axes[_i, 0].set_ylabel(_w)
    save_figure(_fig, figure_path(scopes["kinds act"]["runs"]["transformer"]["config"], "block_entry_cosines"))
    return


if __name__ == "__main__":
    app.run()
