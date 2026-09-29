import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Polar codes: a family tree

    Ego's family as links up to a parent, to a sibling and down to a child. When a person is mentioned, the
    companion is drawn with $\log p(y \mid x) = -\tfrac{\gamma}{2}\left[d_{\mathrm{kin}}(x, y) + \alpha\,
    (g(x) - g(y))^2\right]$, $g$ the generation (up minus down links), $\alpha = 1$. Static models read a person
    and output the companion; the stream transformer reads a sequence of co-mentions. Full width, SGD.
    """)
    return


@app.cell
def _(class_means, labeled_tree, load_run, np, predicted_geometry, tree_parents):
    LR = {"linear": 0.5, "mlp": 0.1, "rnn": 0.5, "transformer": 0.1}          # SGD learning rate
    WIDTH = {"linear": "embed_dim", "mlp": "embed_dim", "rnn": "embed_dim", "transformer": "hidden_size"}

    def family_overrides(model):
        arch = model.split(" ")[0]
        return ([f"runner={arch}", "setting=family_tree", "training_kwargs.optimizer=sgd",
                 "training_kwargs.train_steps=60000", f"training_kwargs.learning_rate={LR[arch]}",
                 f"model_kwargs.model_config.{WIDTH[arch]}=64"] + (["sample_mode=sequence"] if model.endswith("(stream)") else []))

    def node_code(run, n):
        """One vector per person: the representation (static), or its mean over the stream positions where the
        person is the current token."""
        R = run["representation"]
        if R.ndim == 2:
            return R
        tokens = np.asarray(run["inputs"])[:, -R.shape[1]:]
        keep = tokens < n                                                     # drop the start token
        return class_means(R[keep], tokens[keep], n)[0]

    FAMILY_MODELS = ["linear", "mlp", "rnn", "transformer", "transformer (stream)"]
    family_runs = {m: load_run(family_overrides(m)) for m in FAMILY_MODELS}
    _kw = dict(family_runs["linear"]["config"].cfg.structure_kwargs)
    PARENTS = tree_parents(parents=_kw["parents"])
    LINKS = np.array(_kw["labels"][1:])                                       # 0 up, 1 sibling, 2 down; one per edge
    family = {"predicted": predicted_geometry(labeled_tree(**_kw))}
    _rank = np.linalg.matrix_rank(family["predicted"], tol=1e-6 * np.abs(family["predicted"]).max())
    for _m, _r in family_runs.items():                # full rank: every model reads as many directions as the code has
        assert _r["representation"].shape[-1] >= _rank, f"{_m}: width {_r['representation'].shape[-1]} < rank {_rank}"
        family[_m] = node_code(_r, len(PARENTS))
    return LINKS, PARENTS, family, family_runs


@app.cell
def _(LINKS, PARENTS, cka, edge_cosines, edge_vectors, family, family_runs, np, rsa):
    _pairs = {"up-up": (0, 0), "down-down": (2, 2), "up-down": (0, 2), "sib-sib": (1, 1), "sib-up": (1, 0)}
    print(f"{'model':22s} {'KL':>8s} {'RSA':>6s} {'CKA':>6s} " + " ".join(f"{k:>9s}" for k in _pairs) + f" {'|link| cv':>9s}")
    for _m, _E in family.items():
        _C = edge_cosines(_E, PARENTS)
        _n = np.linalg.norm(edge_vectors(_E, PARENTS), axis=1)
        _cos = []
        for _a, _b in _pairs.values():
            _blk = _C[np.ix_(LINKS == _a, LINKS == _b)]
            _cos.append(_blk[~np.eye(len(_blk), dtype=bool)].mean() if _a == _b else _blk.mean())
        _kl = family_runs[_m]["loss"] if _m in family_runs else 0.0
        print(f"{_m:22s} {_kl:8.1e} {rsa(_E, family['predicted']):6.3f} {cka(_E, family['predicted']):6.3f} "
              + " ".join(f"{c:9.3f}" for c in _cos) + f" {_n.std() / _n.mean():9.3f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 1 — link cosines

    Cosine between every pair of link vectors (the code of the farther person minus the nearer one), links
    grouped into up, sibling and down; the predicted code and each trained model, the stream transformer's code
    one mean per person.
    """)
    return


@app.cell
def _(FIG, LINKS, PARENTS, edge_cosines, family, family_runs, figure_path, np, plt, save_figure):
    _order = np.argsort(LINKS, kind="stable")
    _fig, _axes = plt.subplots(1, len(family), figsize=(2.6 * len(family), 3.0))
    for _ax, (_m, _E) in zip(_axes, family.items()):
        _ax.imshow(edge_cosines(_E, PARENTS)[_order][:, _order], cmap=FIG.diverging_cmap, vmin=-1, vmax=1)
        _ax.set_xticks([]); _ax.set_yticks([]); _ax.set_title(_m)
    save_figure(_fig, figure_path(family_runs["linear"]["config"], "link_cosines"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 2 — links along the generation axis

    Held-out links (odd edges) projected onto the axis from the mean down link to the mean up link of the even
    edges (x) and the top principal direction of the even edges orthogonal to it (y); blue up, orange sibling,
    green down, lengths in units of the mean link length.
    """)
    return


@app.cell
def _(FIG, LINKS, PARENTS, edge_vectors, family, family_runs, figure_path, np, plt, save_figure):
    _colors = [FIG.colors["blue"], FIG.colors["orange"], FIG.colors["green"]]
    _fig, _axes = plt.subplots(1, len(family), figsize=(2.6 * len(family), 3.0))
    for _ax, (_m, _E) in zip(_axes, family.items()):
        _V = edge_vectors(_E, PARENTS)
        _fit = np.arange(len(_V)) % 2 == 0
        _x = _V[_fit & (LINKS == 0)].mean(0) - _V[_fit & (LINKS == 2)].mean(0); _x /= np.linalg.norm(_x)
        _y = np.linalg.svd(_V[_fit] - np.outer(_V[_fit] @ _x, _x), full_matrices=False)[2][0]
        _XY = np.c_[_V[~_fit] @ _x, _V[~_fit] @ _y] / np.linalg.norm(_V, axis=1).mean()
        for (_a, _b), _l in zip(_XY, LINKS[~_fit]):
            _ax.plot([0, _a], [0, _b], color=_colors[_l], lw=1.4, alpha=0.8)
        _ax.set_xlim(-1.05, 1.05); _ax.set_ylim(-1.05, 1.05); _ax.set_aspect("equal")
        _ax.set_xticks([]); _ax.set_yticks([]); _ax.set_title(_m)
    save_figure(_fig, figure_path(family_runs["linear"]["config"], "generation_axis"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Scope-stack programs

    Streams of code-like tokens: `for` / `def` / `if` / `with` open a block that binds a fresh name, `end` closes
    it, and every open scope adds its effect to the next-token scores: its name is used, `for` raises break /
    continue, `def` return / yield, `if` pass. Width-64 transformer, next-token prediction, SGD; the predicted
    code is the factored exact next-token rows.
    """)
    return


@app.cell
def _(align_representations, cka, factor_logcond, load_run, np, predicted_inference_geometry, rsa):
    scope_run = load_run(["runner=transformer", "setting=scopes", "training_kwargs.context_length=64",
                          "training_kwargs.optimizer=sgd", "training_kwargs.train_steps=60000",
                          "training_kwargs.learning_rate=0.1", "model_kwargs.model_config.hidden_size=64",
                          "scope_kwargs.kind_strength=1.0"])
    _P = scope_run["targets"].astype(np.float64)                                 # (B, T, V) exact next-token rows
    _R = np.log(_P).reshape(-1, _P.shape[-1])
    _, _, _D, _S = factor_logcond(_R)
    _rank = int((_S > 1e-8 * _S[0]).sum())
    assert scope_run["representation"].shape[-1] >= _rank
    scope_codes = {"predicted": predicted_inference_geometry(_R, _D[:_rank]).reshape(_P.shape[0], _P.shape[1], _rank),
                   "transformer": scope_run["representation"][:, :_P.shape[1]]}  # position t predicts token t+1

    _E0 = scope_codes["predicted"].reshape(-1, _rank)
    _idx = np.random.default_rng(0).choice(len(_E0), 3000, replace=False)
    print(f"KL {scope_run['loss']:.1e}")
    print(f"{'code':12s} {'RSA':>6s} {'CKA':>6s} {'affine R2':>9s} | {'same kind':>9s} {'diff kind':>9s} {'same name':>9s} {'|step| cv':>9s}")
    for _m, _H in scope_codes.items():
        _X = _H.reshape(-1, _H.shape[-1])
        _ent = scope_run["dataset"].block_entries(scope_run["inputs"], _H)
        _k = np.array([e[0] for e in _ent]); _n = np.array([e[1] for e in _ent])
        _U = np.array([e[3] for e in _ent]); _len = np.linalg.norm(_U, axis=1); _U = _U / _len[:, None]
        _C = _U @ _U.T
        _sk, _sn = _k[:, None] == _k[None], _n[:, None] == _n[None]
        print(f"{_m:12s} {rsa(_X[_idx], _E0[_idx]):6.3f} {cka(_X[_idx], _E0[_idx]):6.3f} "
              f"{align_representations(_E0[_idx], _X[_idx], 'linear')[1]:9.3f} | {_C[_sk & ~_sn].mean():9.3f} "
              f"{_C[~_sk & ~_sn].mean():9.3f} {_C[~_sk & _sn].mean():9.3f} {_len.std() / _len.mean():9.3f}")
    return scope_codes, scope_run


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 3 — block-entry steps

    Cosine between the mean block-entry steps (code after the bound name minus code before the kind token), one
    per (kind, name), sorted by kind then name; the predicted code and the trained transformer.
    """)
    return


@app.cell
def _(FIG, figure_path, np, plt, save_figure, scope_codes, scope_run):
    _ds = scope_run["dataset"]
    _keys = [(k, n) for k in _ds.kinds for n in _ds.names]
    _fig, _axes = plt.subplots(1, len(scope_codes), figsize=(3.2 * len(scope_codes), 3.3))
    for _ax, (_m, _H) in zip(_axes, scope_codes.items()):
        _steps = {}
        for _k, _n, _, _v in _ds.block_entries(scope_run["inputs"], _H):
            _steps.setdefault((_k, _n), []).append(_v)
        _M = np.array([np.mean(_steps[key], 0) for key in _keys])
        _M = _M / np.linalg.norm(_M, axis=1, keepdims=True)
        _ax.imshow(_M @ _M.T, cmap=FIG.diverging_cmap, vmin=-1, vmax=1)
        _ticks = np.arange(len(_ds.kinds)) * len(_ds.names) + len(_ds.names) / 2 - 0.5
        _ax.set_xticks(_ticks, _ds.kinds); _ax.set_yticks(_ticks, _ds.kinds); _ax.set_title(_m)
    save_figure(_fig, figure_path(scope_run["config"], "block_entry_cosines"))
    return


if __name__ == "__main__":
    app.run()
