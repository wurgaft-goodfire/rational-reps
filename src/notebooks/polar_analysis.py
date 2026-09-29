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
    from src.code.geometry import (class_means, decode_tree, edge_cosines, edge_vectors, factor_logcond,
                                   predicted_geometry, predicted_inference_geometry, tree_scores)
    from src.code.plot_utils import FIG, figure_path, save_figure
    import matplotlib.pyplot as plt

    return (
        FIG,
        align_representations,
        cka,
        class_means,
        decode_tree,
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
        tree_scores,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Polar codes: a family tree

    Ego's family as links up to a parent, to a sibling and down to a child. When a person is mentioned, the
    companion is drawn with $\log p(y \mid x) = -\tfrac{\gamma}{2}\left[d_{\mathrm{kin}}(x, y) + \alpha
    \lVert c(x) - c(y) \rVert^2\right]$, $c = $ (generation, sibling links) counted from ego, $\alpha = 1/2$ (a link, squared distance $1 + \alpha$, stays closer than two people sharing a parent and link type, $2$): close
    kin, one generation and one line (direct or collateral) are mentioned together. Static models read a person and
    output the companion; the in-context transformer reads a sequence of co-mentions. Full width, SGD.
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
                 "training_kwargs.train_steps=20000", f"training_kwargs.learning_rate={LR[arch]}",
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
    ## Figure 1 — the tree read off each code (static models)

    Top: the family tree decoded from the code alone, as the Polar Probe does: links are the minimum spanning tree of
    the distances between people, each link's type and direction the nearest link-type prototype; drawn on a family chart (generations as rows, mom's side left, dad's
    right), links colored by decoded type (dashed gray: a wrong link or label). Middle: every link
    vector as an arrow on the generation and sibling axes (fit to the other fourteen links; thick = per-type mean).
    Bottom: cosine between every pair of link vectors, grouped into up, sibling and down.
    """)
    return


@app.cell
def _(
    FIG,
    LINKS,
    PARENTS,
    PEOPLE,
    STATIC,
    decode_tree,
    edge_cosines,
    edge_vectors,
    family,
    family_runs,
    figure_path,
    np,
    plt,
    save_figure,
    tree_scores,
):
    LINK_COLORS = [FIG.colors["blue"], FIG.colors["orange"], FIG.colors["green"]]
    LINK_TYPES = ["up (to a parent)", "sibling", "down (to a child)"]
    REVERSE = {0: 2, 2: 0}                                                    # a down link read backwards is an up link
    CHART = np.array([(-0.6, 0), (0.6, 0), (-1.6, 1), (-2.6, 2), (-0.9, 2), (-3.2, 1), (-3.2, 0), (-4.8, 1), (-4.8, 0),
                      (1.6, 1), (0.9, 2), (2.6, 2), (3.2, 1), (3.2, 0), (4.8, 1), (4.8, 0)], float)   # family-chart layout

    def draw_decoded(ax, XY, decoded, parents, labels, names):
        """Decoded links between people placed at XY, colored by decoded type; wrong ones dashed gray."""
        true = {(p, c): labels[c] for c, p in enumerate(parents) if p >= 0}
        for p, c, l in decoded:
            right = true.get((p, c)) == l or (l in REVERSE and true.get((c, p)) == REVERSE[l])
            ax.plot(*zip(XY[p], XY[c]), color=LINK_COLORS[l] if right else "0.6", ls="-" if right else "--",
                    lw=2.2, zorder=1)
        ax.scatter(*XY.T, s=22, color="k", zorder=2)
        for i, name in enumerate(names):
            ax.annotate(name, XY[i], textcoords="offset points", xytext=(0, -8), ha="center", va="top", fontsize=6)
        ax.set_xlim(-5.6, 5.6); ax.set_ylim(-0.6, 2.4); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    def top2(X):
        Xc = X - X.mean(0)
        return Xc @ np.linalg.svd(Xc, full_matrices=False)[2][:2].T

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

    _labels = [-1] + list(LINKS)
    _names = ["predicted"] + STATIC
    _fig, _axes = plt.subplots(3, len(_names), figsize=(3.3 * len(_names), 9.4), gridspec_kw=dict(hspace=0.22, height_ratios=[0.8, 1, 1]))
    for _j, _m in enumerate(_names):
        _dec = decode_tree(family[_m], PARENTS, _labels)
        _u, _l = tree_scores(_dec, PARENTS, _labels, REVERSE)
        draw_decoded(_axes[0, _j], CHART, _dec, PARENTS, _labels, PEOPLE)
        _axes[0, _j].set_title(f"{_m}\nlinks {_u:.0%}, labels {_l:.0%}", fontsize=11)
        link_directions(_axes[1, _j], family[_m])
        link_gram(_axes[2, _j], family[_m], ylabels=_j == 0)
    _axes[1, 0].set_ylabel("sibling axis"); _axes[1, 0].set_xlabel("generation axis")
    _fig.legend([plt.Line2D([], [], color=c, lw=3) for c in LINK_COLORS], LINK_TYPES, loc="upper center", ncol=3,
                frameon=False, bbox_to_anchor=(0.5, 1.0))
    save_figure(_fig, figure_path(family_runs["linear"]["config"], "family_tree_codes"))
    return CHART, LINK_COLORS, LINK_TYPES, REVERSE, draw_decoded, link_gram


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 2 — one sequence parsed by the in-context transformer

    Top: the tree decoded from one test sequence alone, on the family chart of Figure 1 (each person's mean representation in it;
    link-type prototypes from the model's codes), over the people it mentions. The sequence is the most complete one
    whose people form a connected family tree. Bottom: link cosines of its per-person codes (means over all test
    sequences), as in Figure 1.
    """)
    return


@app.cell
def _(
    CHART,
    IN_CONTEXT,
    LINKS,
    LINK_COLORS,
    LINK_TYPES,
    PARENTS,
    PEOPLE,
    REVERSE,
    decode_tree,
    draw_decoded,
    edge_vectors,
    family,
    family_runs,
    figure_path,
    link_gram,
    np,
    plt,
    save_figure,
    tree_scores,
):
    _run = family_runs[IN_CONTEXT]
    _n = len(PARENTS)
    _labels = [-1] + list(LINKS)
    _tok = np.asarray(_run["inputs"])
    _R = _run["representation"]
    _sets = [set(_s[_s < _n].tolist()) for _s in _tok]
    _ok = [b for b, S in enumerate(_sets) if 0 in S and all(PARENTS[c] in S for c in S if c != 0)]
    _b = max(_ok, key=lambda b: len(_sets[b]))                                # most people, connected through ego
    _keep = _tok[_b] < _n
    _X, _t = _R[_b][_keep], _tok[_b][_keep]
    _people = sorted(_sets[_b]); _ix = {p: i for i, p in enumerate(_people)}
    _par = [-1 if p == 0 else _ix[PARENTS[p]] for p in _people]
    _lab = [-1 if p == 0 else _labels[p] for p in _people]
    _E = np.array([_X[_t == p].mean(0) for p in _people])
    _V = edge_vectors(family[IN_CONTEXT], PARENTS)
    _protos = np.array([_V[LINKS == l].mean(0) for l in range(3)])
    _dec = decode_tree(_E, _par, _lab, protos=_protos)
    _u, _l = tree_scores(_dec, _par, _lab, REVERSE)
    _fig, (_ax, _axg) = plt.subplots(2, 1, figsize=(6.2, 8.4), gridspec_kw=dict(height_ratios=[0.8, 1], hspace=0.3))
    draw_decoded(_ax, CHART[_people], _dec, _par, _lab, [PEOPLE[p] for p in _people])
    _ax.set_title(f"one sequence, {len(_t)} mentions of {len(_people)} people\nlinks {_u:.0%}, labels {_l:.0%}", fontsize=11)
    _ax.text(0.5, -0.12, " · ".join(PEOPLE[i] for i in _t[:9]) + " · …", transform=_ax.transAxes, ha="center", fontsize=8)
    link_gram(_axg, family[IN_CONTEXT])
    _fig.legend([plt.Line2D([], [], color=c, lw=3) for c in LINK_COLORS], LINK_TYPES, loc="upper center", ncol=3,
                frameon=False, bbox_to_anchor=(0.5, 0.97), fontsize=8)
    save_figure(_fig, figure_path(_run["config"], "in_context_parse"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Scope-stack programs

    Streams of code-like tokens: `for` / `def` / `if` open a block that binds a fresh name, `end` closes
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
    ## Figure 3 — the scope tree read off the code

    Top: the nesting of one test program decoded from the representation alone: nodes are the module and each block
    (the state after its bound name), links the minimum spanning tree of their distances, each link's kind the nearest
    kind prototype (mean block-entry step); drawn like indented code, depth down and blocks in opening order, links
    colored by decoded kind (dashed gray: wrong); titles give this program's scores and the mean over all test
    programs with at least two blocks. Middle: block-entry steps (code after the bound name minus code before
    the kind token; 300 drawn) on the plane of the three kind means, centered on their average; thick = per-kind mean. Bottom: cosine
    between the mean block-entry steps, one per (kind, name), sorted by kind then name.
    """)
    return


@app.cell
def _(
    FIG,
    decode_tree,
    figure_path,
    np,
    plt,
    save_figure,
    scope_codes,
    scope_run,
    tree_scores,
):
    _ds = scope_run["dataset"]
    _inp = np.asarray(scope_run["inputs"])
    _KIND_COLORS = [FIG.colors["blue"], FIG.colors["orange"], FIG.colors["green"]]
    _T = scope_codes["predicted"].shape[1]
    _trees = [_ds.blocks(_s) for _s in _inp]
    _depth = lambda tree, i: 0 if tree[i][1] < 0 else 1 + _depth(tree, tree[i][1])
    _ok = [b for b, tr in enumerate(_trees) if all(p < _T for p, *_ in tr)]
    _b = max(_ok, key=lambda b: (max(_depth(_trees[b], i) for i in range(len(_trees[b]))), len(_trees[b])))
    _tree = _trees[_b]
    _parents = [p for _, p, _, _ in _tree]
    _labels = [-1] + [_ds.kinds.index(k) for _, _, k, _ in _tree[1:]]
    _XY = np.array([(i, -_depth(_tree, i)) for i in range(len(_tree))], float)
    _names = ["module"] + [f"{k} {n}" for _, _, k, n in _tree[1:]]
    _keys = [(k, n) for k in _ds.kinds for n in _ds.names]
    _fig, _axes = plt.subplots(3, len(scope_codes), figsize=(4.6 * len(scope_codes), 12.5),
                               gridspec_kw=dict(height_ratios=[0.8, 1, 1], hspace=0.28))
    for _j, (_m, _H) in enumerate(scope_codes.items()):
        _ent = _ds.block_entries(_inp, _H)
        _k = np.array([_ds.kinds.index(e[0]) for e in _ent]); _V = np.array([e[3] for e in _ent])
        _protos = np.array([_V[_k == l].mean(0) for l in range(len(_ds.kinds))])
        _E = np.array([_H[_b, p] for p, *_ in _tree])
        _dec = decode_tree(_E, _parents, _labels, protos=_protos)
        _u, _l = tree_scores(_dec, _parents, _labels)
        _ax = _axes[0, _j]
        _true = {(p, c): _labels[c] for c, p in enumerate(_parents) if p >= 0}
        for _p, _c, _lab in _dec:
            _right = _true.get((_p, _c)) == _lab
            _ax.plot(*zip(_XY[_p], _XY[_c]), color=_KIND_COLORS[_lab] if _right else "0.6", ls="-" if _right else "--", lw=2.2, zorder=1)
        _ax.scatter(*_XY.T, s=22, color="k", zorder=2)
        for _i, _nm in enumerate(_names):
            _ax.annotate(_nm, _XY[_i], textcoords="offset points", xytext=(0, -8), ha="center", va="top", fontsize=8)
        _ax.set_xlim(-0.6, len(_tree) - 0.4); _ax.set_ylim(_XY[:, 1].min() - 0.6, 0.4)
        _ax.set_xticks([]); _ax.set_yticks([]); [_s.set_visible(False) for _s in _ax.spines.values()]
        _all = [tree_scores(decode_tree(np.array([_H[b, p] for p, *_ in _trees[b]]), [p for _, p, _, _ in _trees[b]],
                                        [-1] + [_ds.kinds.index(k) for _, _, k, _ in _trees[b][1:]], protos=_protos),
                            [p for _, p, _, _ in _trees[b]], [-1] + [_ds.kinds.index(k) for _, _, k, _ in _trees[b][1:]])
                for b in _ok if len(_trees[b]) > 2]
        _mu, _ml = np.mean(_all, 0)
        _ax.set_title(f"{_m}\nthis program: links {_u:.0%}, kinds {_l:.0%}\n{len(_all)} programs: links {_mu:.0%}, kinds {_ml:.0%}", fontsize=10)
        _Mc = _protos - _protos.mean(0)                                       # three kind means span a plane
        _B = np.linalg.svd(_Mc, full_matrices=False)[2][:2]
        _scale = np.linalg.norm(_Mc, axis=1).max()
        _ax = _axes[1, _j]
        for _i in np.random.default_rng(0).choice(len(_V), 300, replace=False):
            _a, _c2 = (_V[_i] - _protos.mean(0)) @ _B.T / _scale
            _ax.annotate("", (_a, _c2), (0, 0), arrowprops=dict(arrowstyle="-|>", color=_KIND_COLORS[_k[_i]], lw=0.6, alpha=0.3))
        for _l2 in range(len(_ds.kinds)):
            _a, _c2 = _Mc[_l2] @ _B.T / _scale
            _ax.annotate("", (_a, _c2), (0, 0), arrowprops=dict(arrowstyle="-|>", color=_KIND_COLORS[_l2], lw=3, mutation_scale=16))
        _ax.axhline(0, color="0.85", lw=0.6, zorder=0); _ax.axvline(0, color="0.85", lw=0.6, zorder=0)
        _ax.set_xlim(-1.5, 1.5); _ax.set_ylim(-1.5, 1.5); _ax.set_aspect("equal"); _ax.set_xticks([]); _ax.set_yticks([])
        _steps = {}
        for _kk, _n, _, _v in _ent:
            _steps.setdefault((_kk, _n), []).append(_v)
        _M = np.array([np.mean(_steps[key], 0) for key in _keys])
        _M = _M / np.linalg.norm(_M, axis=1, keepdims=True)
        _ax = _axes[2, _j]
        _ax.imshow(_M @ _M.T, cmap=FIG.diverging_cmap, vmin=-1, vmax=1)
        _ticks = np.arange(len(_ds.kinds)) * len(_ds.names) + len(_ds.names) / 2 - 0.5
        _ax.set_xticks(_ticks, _ds.kinds); _ax.set_yticks(_ticks, _ds.kinds if _j == 0 else [])
    _fig.legend([plt.Line2D([], [], color=c, lw=3) for c in _KIND_COLORS], _ds.kinds, loc="upper center", ncol=3,
                frameon=False, bbox_to_anchor=(0.5, 0.995))
    save_figure(_fig, figure_path(scope_run["config"], "scope_tree_codes"))
    return


if __name__ == "__main__":
    app.run()
