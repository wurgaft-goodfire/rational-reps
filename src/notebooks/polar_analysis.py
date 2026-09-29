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
    return LINK_COLORS, LINK_TYPES, link_gram


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 2 — one sequence in the in-context transformer

    Left: the transformer's representation at every mention in one test sequence (gray), on that sequence's top two
    principal directions; black points are each person's mean, joined by their family links (colored by type). The
    sequence is the most complete one whose people form a connected family tree. Middle: the same links as vectors
    (farther person's mean minus the nearer one's) on the generation and sibling axes of the model's per-person codes.
    Right: link cosines of its per-person codes over all test sequences, as in Figure 1.
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
    edge_vectors,
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
    _labels = [-1] + list(LINKS)
    _tok = np.asarray(_run["inputs"])
    _R = _run["representation"]
    _sets = [set(_s[_s < _n].tolist()) for _s in _tok]
    _ok = [b for b, S in enumerate(_sets) if 0 in S and all(PARENTS[c] in S for c in S if c != 0)]
    _b = max(_ok, key=lambda b: len(_sets[b]))                                # most people, connected through ego
    _keep = _tok[_b] < _n
    _X, _t = _R[_b][_keep], _tok[_b][_keep]
    _people = sorted(_sets[_b])
    _E = {p: _X[_t == p].mean(0) for p in _people}                           # each person's mean state in this sequence
    _Xc = _X - _X.mean(0)
    _P2 = np.linalg.svd(_Xc, full_matrices=False)[2][:2]
    _XY = _Xc @ _P2.T
    _mean_xy = {p: (_E[p] - _X.mean(0)) @ _P2.T for p in _people}
    _fig, (_axm, _axa, _axg) = plt.subplots(1, 3, figsize=(16, 5.4), gridspec_kw=dict(width_ratios=[1.5, 1, 1], wspace=0.25))
    _axm.scatter(*_XY.T, s=10, color="0.75", zorder=0)
    for _c in _people:
        _p = PARENTS[_c]
        if _p >= 0:
            _axm.plot(*zip(_mean_xy[_p], _mean_xy[_c]), color=LINK_COLORS[LINKS[_c - 1]], lw=2.2, zorder=1)
    for _p in _people:
        _axm.scatter(*_mean_xy[_p], s=36, color="k", zorder=2)
        _axm.annotate(PEOPLE[_p], _mean_xy[_p], textcoords="offset points", xytext=(4, 3), fontsize=8)
    _axm.set_xticks([]); _axm.set_yticks([])
    _axm.set_title(f"representations in one sequence ({len(_t)} mentions, top two directions)", fontsize=11)
    _axm.set_xlabel(" · ".join(PEOPLE[i] for i in _t[:9]) + " · …", fontsize=8)
    _V = edge_vectors(family[IN_CONTEXT], PARENTS)                           # axes from the model's per-person codes
    _x = _V[LINKS == 0].mean(0) - _V[LINKS == 2].mean(0); _x /= np.linalg.norm(_x)
    _y = _V[LINKS == 1].mean(0); _y -= (_y @ _x) * _x; _y /= np.linalg.norm(_y)
    _links = [(PARENTS[c], c) for c in _people if PARENTS[c] >= 0]
    _W = np.array([[(_E[c] - _E[p]) @ _x, (_E[c] - _E[p]) @ _y] for p, c in _links])
    _W /= np.linalg.norm([_E[c] - _E[p] for p, c in _links], axis=1).mean()
    for (_a, _bb), (_p, _c) in zip(_W, _links):
        _axa.annotate("", (_a, _bb), (0, 0), arrowprops=dict(arrowstyle="-|>", color=LINK_COLORS[LINKS[_c - 1]], lw=1.6, alpha=0.8))
    _axa.axhline(0, color="0.85", lw=0.6, zorder=0); _axa.axvline(0, color="0.85", lw=0.6, zorder=0)
    _axa.set_xlim(-1.2, 1.2); _axa.set_ylim(-1.2, 1.2); _axa.set_aspect("equal"); _axa.set_xticks([]); _axa.set_yticks([])
    _axa.set_xlabel("generation axis"); _axa.set_ylabel("sibling axis")
    _axa.set_title("this sequence's links", fontsize=11)
    link_gram(_axg, family[IN_CONTEXT]); _axg.set_title("link cosines, all test sequences", fontsize=11)
    _fig.legend([plt.Line2D([], [], color=c, lw=3) for c in LINK_COLORS], LINK_TYPES, loc="upper center", ncol=3,
                frameon=False, bbox_to_anchor=(0.5, 1.04))
    save_figure(_fig, figure_path(_run["config"], "in_context_parse"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Scope-stack programs

    Streams of code-like tokens: `for` / `def` / `if` open a block that binds a fresh name, `end` closes
    it, and every open scope adds its effect to the next-token scores: its name is used, `for` raises break /
    continue, `def` return / yield, `if` pass (kind effects at half strength, so same-kind sibling blocks stay farther
    apart than parent and child). Width-64 transformer, next-token prediction, AdamW; the predicted
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
                          "training_kwargs.optimizer=adamw_torch", "training_kwargs.train_steps=45000", "training_kwargs.batch_size=1024",
                          "training_kwargs.learning_rate=0.001", "model_kwargs.model_config.hidden_size=64",
                          "scope_kwargs.kind_strength=0.5"])
    _P = scope_run["targets"].astype(np.float64)                                 # (B, T, V) exact next-token rows
    _R = np.log(_P).reshape(-1, _P.shape[-1])
    _, _, _D, _S = factor_logcond(_R)
    _rank = int((_S > 1e-8 * _S[0]).sum())
    assert scope_run["representation"].shape[-1] >= _rank
    scope_codes = {"predicted": predicted_inference_geometry(_R, _D[:_rank]).reshape(_P.shape[0], _P.shape[1], _rank),
                   "transformer": scope_run["representation"][:, :_P.shape[1]]}  # position t predicts token t+1

    _E0 = scope_codes["predicted"].reshape(-1, _rank)
    _block = ~np.isin(np.asarray(scope_run["inputs"])[:, :_P.shape[1]], range(3)).reshape(-1)   # no name pending: a used
    _idx = np.random.default_rng(0).choice(np.flatnonzero(_block), 3000, replace=False)       # name's -30 is never scored
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
    ## Figure 3 — scope programs in the code

    Left: the state at every position of one test program where no bound name is pending (gray trajectory, drawn at random among programs with six or
    more blocks, three deep), on that program's top two principal directions; black points are the module and each block
    (the state after its bound name), joined to the block that contains them, colored by kind. Titles give how often the
    tree is decoded from the code alone (spanning tree of block-state distances, nearest kind prototype) over all test
    programs. Middle: block-entry steps (code after the bound name minus code before the kind token; 300 drawn) on the
    plane of the three kind means, centered on their average; thick = per-kind mean. Right: cosine between the mean
    block-entry steps, one per (kind, name), sorted by kind then name.
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
    _rich = [b for b in _ok if len(_trees[b]) >= 6 and max(_depth(_trees[b], i) for i in range(len(_trees[b]))) >= 3]
    _b = int(np.random.default_rng(0).choice(_rich))                        # a random program with some nesting
    _tree = _trees[_b]
    _parents = [p for _, p, _, _ in _tree]
    _labels = [-1] + [_ds.kinds.index(k) for _, _, k, _ in _tree[1:]]
    _names = ["module"] + [f"{k} {n}" for _, _, k, n in _tree[1:]]
    _keys = [(k, n) for k in _ds.kinds for n in _ds.names]
    _fig, _axes = plt.subplots(len(scope_codes), 3, figsize=(16, 5.2 * len(scope_codes)),
                               gridspec_kw=dict(width_ratios=[1.5, 1, 1], wspace=0.25, hspace=0.3))
    for _i, (_m, _H) in enumerate(scope_codes.items()):
        _ent = _ds.block_entries(_inp, _H)
        _k = np.array([_ds.kinds.index(e[0]) for e in _ent]); _V = np.array([e[3] for e in _ent])
        _protos = np.array([_V[_k == l].mean(0) for l in range(len(_ds.kinds))])
        _all = [tree_scores(decode_tree(np.array([_H[b, p] for p, *_ in _trees[b]]), [p for _, p, _, _ in _trees[b]],
                                        [-1] + [_ds.kinds.index(k) for _, _, k, _ in _trees[b][1:]], protos=_protos),
                            [p for _, p, _, _ in _trees[b]], [-1] + [_ds.kinds.index(k) for _, _, k, _ in _trees[b][1:]])
                for b in _ok if len(_trees[b]) > 2]
        _mu, _ml = np.mean(_all, 0)
        # the program's trajectory: every position's state on the program's top two directions
        _settled = ~np.isin(_inp[_b, :_T], range(len(_ds.kinds)))           # skip states with a name pending
        _Hb = _H[_b]
        _mu_b = _Hb[_settled].mean(0)
        _P2 = np.linalg.svd(_Hb[_settled] - _mu_b, full_matrices=False)[2][:2]
        _Y = (_Hb - _mu_b) @ _P2.T
        _ax = _axes[_i, 0]
        _ax.plot(*_Y[_settled].T, color="0.8", lw=0.8, zorder=0)
        _ax.scatter(*_Y[_settled].T, s=8, color="0.7", zorder=0)
        _N = np.array([_Y[p] for p, *_ in _tree])
        for _c in range(1, len(_tree)):
            _ax.plot(*zip(_N[_parents[_c]], _N[_c]), color=_KIND_COLORS[_labels[_c]], lw=2.4, zorder=1)
        _ax.scatter(*_N.T, s=36, color="k", zorder=2)
        for _j, _nm in enumerate(_names):
            _ax.annotate(_nm, _N[_j], textcoords="offset points", xytext=(4, 3), fontsize=8)
        _ax.set_xticks([]); _ax.set_yticks([])
        _ax.set_title(f"{_m}: one program's states (top two directions)\n{len(_all)} programs: tree decoded "
                      f"{_mu:.0%} of links, {_ml:.0%} of kinds", fontsize=10)
        # block-entry steps on the plane of the three kind means
        _Mc = _protos - _protos.mean(0)
        _B = np.linalg.svd(_Mc, full_matrices=False)[2][:2]
        _scale = np.linalg.norm(_Mc, axis=1).max()
        _ax = _axes[_i, 1]
        for _r in np.random.default_rng(0).choice(len(_V), 300, replace=False):
            _a, _c2 = (_V[_r] - _protos.mean(0)) @ _B.T / _scale
            _ax.annotate("", (_a, _c2), (0, 0), arrowprops=dict(arrowstyle="-|>", color=_KIND_COLORS[_k[_r]], lw=0.6, alpha=0.3))
        for _l2 in range(len(_ds.kinds)):
            _a, _c2 = _Mc[_l2] @ _B.T / _scale
            _ax.annotate("", (_a, _c2), (0, 0), arrowprops=dict(arrowstyle="-|>", color=_KIND_COLORS[_l2], lw=3, mutation_scale=16))
        _ax.axhline(0, color="0.85", lw=0.6, zorder=0); _ax.axvline(0, color="0.85", lw=0.6, zorder=0)
        _ax.set_xlim(-1.5, 1.5); _ax.set_ylim(-1.5, 1.5); _ax.set_aspect("equal"); _ax.set_xticks([]); _ax.set_yticks([])
        _ax.set_title("block-entry steps, all programs", fontsize=10)
        _steps = {}
        for _kk, _n, _, _v in _ent:
            _steps.setdefault((_kk, _n), []).append(_v)
        _M = np.array([np.mean(_steps[key], 0) for key in _keys])
        _M = _M / np.linalg.norm(_M, axis=1, keepdims=True)
        _ax = _axes[_i, 2]
        _ax.imshow(_M @ _M.T, cmap=FIG.diverging_cmap, vmin=-1, vmax=1)
        _ticks = np.arange(len(_ds.kinds)) * len(_ds.names) + len(_ds.names) / 2 - 0.5
        _ax.set_xticks(_ticks, _ds.kinds); _ax.set_yticks(_ticks, _ds.kinds)
        _ax.set_title("cosine of mean steps per (kind, name)", fontsize=10)
    _fig.legend([plt.Line2D([], [], color=c, lw=3) for c in _KIND_COLORS], _ds.kinds, loc="upper center", ncol=3,
                frameon=False, bbox_to_anchor=(0.5, 0.97))
    save_figure(_fig, figure_path(scope_run["config"], "scope_tree_codes"))
    return


if __name__ == "__main__":
    app.run()
