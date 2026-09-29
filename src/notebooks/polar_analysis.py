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
        cka,
        class_means,
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
    and output the companion; the in-context transformer reads a sequence of co-mentions. Full width, SGD.
    """)
    return


@app.cell
def _(
    class_means,
    labeled_tree,
    load_run,
    np,
    predicted_geometry,
    tree_parents,
):
    LR = {"linear": 0.5, "mlp": 0.1, "rnn": 0.5, "transformer": 0.1}          # SGD learning rate
    WIDTH = {"linear": "embed_dim", "mlp": "embed_dim", "rnn": "embed_dim", "transformer": "hidden_size"}
    STATIC = ["linear", "mlp", "rnn", "transformer"]
    IN_CONTEXT = "in-context transformer"                                     # reads a sequence of co-mentions
    PEOPLE = ["me", "sibling", "mom", "grandma M", "grandpa M", "aunt M", "cousin M1", "uncle M", "cousin M2",
              "dad", "grandma D", "grandpa D", "aunt D", "cousin D1", "uncle D", "cousin D2"]   # M mom's side, D dad's

    def family_overrides(model):
        arch = "transformer" if model == IN_CONTEXT else model
        return ([f"runner={arch}", "setting=family_tree", "training_kwargs.optimizer=sgd",
                 "training_kwargs.train_steps=60000", f"training_kwargs.learning_rate={LR[arch]}",
                 f"model_kwargs.model_config.{WIDTH[arch]}=64"] + (["sample_mode=sequence"] if model == IN_CONTEXT else []))

    def node_code(run, n):
        """One vector per person: the representation (static), or its mean over the positions of a sequence where
        the person is the current token."""
        R = run["representation"]
        if R.ndim == 2:
            return R
        tokens = np.asarray(run["inputs"])[:, -R.shape[1]:]
        keep = tokens < n                                                     # drop the start token
        return class_means(R[keep], tokens[keep], n)[0]

    family_runs = {m: load_run(family_overrides(m)) for m in STATIC + [IN_CONTEXT]}
    _kw = dict(family_runs["linear"]["config"].cfg.structure_kwargs)
    PARENTS = tree_parents(parents=_kw["parents"])
    LINKS = np.array(_kw["labels"][1:])                                       # 0 up, 1 sibling, 2 down; one per edge
    family = {"predicted": predicted_geometry(labeled_tree(**_kw))}
    _rank = np.linalg.matrix_rank(family["predicted"], tol=1e-6 * np.abs(family["predicted"]).max())
    for _m, _r in family_runs.items():                # full rank: every model reads as many directions as the code has
        assert _r["representation"].shape[-1] >= _rank, f"{_m}: width {_r['representation'].shape[-1]} < rank {_rank}"
        family[_m] = node_code(_r, len(PARENTS))
    return IN_CONTEXT, LINKS, PARENTS, PEOPLE, STATIC, family, family_runs


@app.cell
def _(
    LINKS,
    PARENTS,
    cka,
    edge_cosines,
    edge_vectors,
    family,
    family_runs,
    np,
    rsa,
):
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
    ## Figure 1 — link directions (static models)

    Top: every link vector (the farther person's code minus the nearer one's) as an arrow from the origin, on the
    generation axis (mean up link minus mean down link) and the sibling axis (mean sibling link, orthogonal to it);
    each link is projected on axes fit to the other fourteen. Thick arrows are the per-type means, lengths in units of
    the mean link length. Bottom: cosine between every pair of link vectors, grouped into up, sibling and down.
    """)
    return


@app.cell
def _(
    FIG,
    LINKS,
    PARENTS,
    STATIC,
    edge_cosines,
    edge_vectors,
    family,
    family_runs,
    figure_path,
    np,
    plt,
    save_figure,
):
    LINK_COLORS = [FIG.colors["blue"], FIG.colors["orange"], FIG.colors["green"]]
    LINK_TYPES = ["up (to a parent)", "sibling", "down (to a child)"]

    def link_directions(ax, E):
        """King-style direction plot: each link projected on generation / sibling axes fit without it."""
        V = edge_vectors(E, PARENTS)
        V = V / np.linalg.norm(V, axis=1).mean()
        XY = np.zeros((len(V), 2))
        for i in range(len(V)):
            rest = np.arange(len(V)) != i
            x = V[rest & (LINKS == 0)].mean(0) - V[rest & (LINKS == 2)].mean(0); x /= np.linalg.norm(x)
            y = V[rest & (LINKS == 1)].mean(0); y -= (y @ x) * x; y /= np.linalg.norm(y)
            XY[i] = V[i] @ x, V[i] @ y
        for (a, b), l in zip(XY, LINKS):
            ax.annotate("", (a, b), (0, 0), arrowprops=dict(arrowstyle="-|>", color=LINK_COLORS[l], lw=0.9, alpha=0.45))
        for l in range(3):
            a, b = XY[LINKS == l].mean(0)
            ax.annotate("", (a, b), (0, 0), arrowprops=dict(arrowstyle="-|>", color=LINK_COLORS[l], lw=3, mutation_scale=16))
        ax.axhline(0, color="0.85", lw=0.6, zorder=0); ax.axvline(0, color="0.85", lw=0.6, zorder=0)
        ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    def link_gram(ax, E, ylabels=True):
        order = np.argsort(LINKS, kind="stable")
        ax.imshow(edge_cosines(E, PARENTS)[order][:, order], cmap=FIG.diverging_cmap, vmin=-1, vmax=1)
        edges = np.cumsum(np.bincount(LINKS))
        ax.set_xticks(edges - np.bincount(LINKS) / 2 - 0.5, ["up", "sib", "down"])
        ax.set_yticks(edges - np.bincount(LINKS) / 2 - 0.5, ["up", "sib", "down"] if ylabels else [])

    _names = ["predicted"] + STATIC
    _fig, _axes = plt.subplots(2, len(_names), figsize=(2.9 * len(_names), 6.2), gridspec_kw=dict(hspace=0.15))
    for _j, _m in enumerate(_names):
        link_directions(_axes[0, _j], family[_m]); _axes[0, _j].set_title(_m)
        link_gram(_axes[1, _j], family[_m], ylabels=_j == 0)
    _axes[0, 0].set_ylabel("sibling axis"); _axes[0, 0].set_xlabel("generation axis")
    _fig.legend([plt.Line2D([], [], color=c, lw=3) for c in LINK_COLORS], LINK_TYPES, loc="upper center", ncol=3,
                frameon=False, bbox_to_anchor=(0.5, 1.02))
    save_figure(_fig, figure_path(family_runs["linear"]["config"], "link_directions"))
    return LINK_COLORS, LINK_TYPES, link_gram


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 2 — one sequence parsed by the in-context transformer

    Top: the transformer's representation at every mention in one test sequence (small points), on the top two
    principal directions of that sequence's representations; labels at each person's mean, and the family links among
    the people mentioned drawn between them, colored by type. Bottom: link cosines of its per-person codes (means over
    all test sequences), as in Figure 1.
    """)
    return


@app.cell
def _(
    IN_CONTEXT,
    LINKS,
    LINK_COLORS,
    LINK_TYPES,
    PARENTS,
    PEOPLE,
    family,
    family_runs,
    figure_path,
    link_gram,
    np,
    plt,
    save_figure,
):
    _run = family_runs[IN_CONTEXT]
    _n = len(PARENTS)
    _tok = np.asarray(_run["inputs"])
    _R = _run["representation"]
    _b = int(np.argmax([len(set(_s[_s < _n])) for _s in _tok]))              # the sequence mentioning the most people
    _keep = _tok[_b] < _n
    _X, _t = _R[_b][_keep], _tok[_b][_keep]
    _Xc = _X - _X.mean(0)
    _XY = _Xc @ np.linalg.svd(_Xc, full_matrices=False)[2][:2].T
    _fig, (_ax, _axg) = plt.subplots(2, 1, figsize=(5.2, 9.4), gridspec_kw=dict(height_ratios=[1.35, 1], hspace=0.2))
    _mean = {p: _XY[_t == p].mean(0) for p in set(_t.tolist())}
    for _c, _p in enumerate(PARENTS):
        if _p >= 0 and _c in _mean and _p in _mean:
            _ax.plot(*zip(_mean[_p], _mean[_c]), color=LINK_COLORS[LINKS[_c - 1]], lw=2.2, zorder=1)
    _ax.scatter(*_XY.T, s=8, color="0.6", alpha=0.5, zorder=2)
    for _p, _xy in _mean.items():
        _ax.scatter(*_xy, s=40, color="k", zorder=3)
        _ax.annotate(PEOPLE[_p], _xy, textcoords="offset points", xytext=(4, 4), fontsize=9)
    _ax.set_xticks([]); _ax.set_yticks([]); _ax.set_title(f"one sequence, {len(_t)} mentions", fontsize=12)
    _ax.set_xlabel(" · ".join(PEOPLE[i] for i in _t[:10]) + " · …", fontsize=8)
    link_gram(_axg, family[IN_CONTEXT])
    _fig.legend([plt.Line2D([], [], color=c, lw=3) for c in LINK_COLORS], LINK_TYPES, loc="upper center", ncol=3,
                frameon=False, bbox_to_anchor=(0.5, 0.97), fontsize=8)
    save_figure(_fig, figure_path(_run["config"], "in_context_parse"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Scope-stack programs

    Streams of code-like tokens: `for` / `def` / `if` / `with` open a block that binds a fresh name, `end` closes
    it, and every open scope adds its effect to the next-token scores: its name is used, `for` raises break /
    continue, `def` return / yield, `if` pass. Width-64 transformer, next-token prediction, AdamW; the predicted
    code is the factored exact next-token rows.
    """)
    return


@app.cell
def _(
    align_representations,
    cka,
    factor_logcond,
    load_run,
    np,
    predicted_inference_geometry,
    rsa,
):
    scope_run = load_run(["runner=transformer", "setting=scopes", "training_kwargs.context_length=64",
                          "training_kwargs.optimizer=adamw_torch", "training_kwargs.train_steps=15000", "training_kwargs.batch_size=1024",
                          "training_kwargs.learning_rate=0.001", "model_kwargs.model_config.hidden_size=64",
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
