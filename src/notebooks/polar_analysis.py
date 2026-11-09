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
    output the companion; the in-context transformer reads a sequence of co-mentions. Full width; static models SGD, the in-context transformer AdamW (15k steps, batch 1024).
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
        if model == IN_CONTEXT:                                               # a stream model: AdamW, large batches
            return ["runner=transformer", "setting=family_tree", "sample_mode=sequence", "training_kwargs.optimizer=adamw_torch",
                    "training_kwargs.learning_rate=0.001", "training_kwargs.batch_size=1024", "training_kwargs.train_steps=15000",
                    "model_kwargs.model_config.hidden_size=64"]
        return [f"runner={model}", "setting=family_tree", "training_kwargs.optimizer=sgd", "training_kwargs.train_steps=20000",
                f"training_kwargs.learning_rate={LR[model]}", f"model_kwargs.model_config.{WIDTH[model]}=64"]

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
    ## Figure 2 — one sequence read by the in-context transformer

    Both panels on one plane of the transformer's raw representation: the generation axis (mean up link minus mean down
    link of its per-person codes) and the sibling axis orthogonal to it. Left: every link of its per-person codes (thick =
    per-type mean). Middle: its representation as one test sequence is read, an arrow from each mention to the next,
    for mentions of directly related people, colored by the relation (gray dots: every mention); labels at each person's mean position. Right:
    link cosines of its per-person codes over all test sequences.
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
    _tok = np.asarray(_run["inputs"])
    _R = _run["representation"]
    _V = edge_vectors(family[IN_CONTEXT], PARENTS)                           # the model's links, one per family link
    _x = _V[LINKS == 0].mean(0) - _V[LINKS == 2].mean(0); _x /= np.linalg.norm(_x)
    _y = _V[LINKS == 1].mean(0); _y -= (_y @ _x) * _x; _y /= np.linalg.norm(_y)
    _A = np.stack([_x, _y])                                                  # one projection for both panels
    _unit = np.linalg.norm(_V, axis=1).mean()
    _rel = {}                                                                # (a, b) -> type of the move from a to b
    for _c, _p in enumerate(PARENTS):
        if _p >= 0:
            _l = LINKS[_c - 1]
            _rel[(_p, _c)] = _l
            _rel[(_c, _p)] = {0: 2, 2: 0, 1: 1}[_l]                          # read backwards: up <-> down
    _b = max(range(len(_tok)), key=lambda b: sum((int(_tok[b, t]), int(_tok[b, t + 1])) in _rel for t in range(1, _tok.shape[1] - 1)))
    _keep = _tok[_b] < _n
    _X, _t = _R[_b][_keep], _tok[_b][_keep]
    _Z = (_X - _X[0]) @ _A.T / _unit
    _fig, (_axd, _axp, _axg) = plt.subplots(1, 3, figsize=(16.5, 5.4), gridspec_kw=dict(width_ratios=[1, 1.5, 1], wspace=0.25))
    _W = _V @ _A.T / _unit
    for (_a, _bb), _l in zip(_W, LINKS):
        _axd.annotate("", (_a, _bb), (0, 0), arrowprops=dict(arrowstyle="-|>", color=LINK_COLORS[_l], lw=0.9, alpha=0.45))
    for _l in range(3):
        _a, _bb = _W[LINKS == _l].mean(0)
        _axd.annotate("", (_a, _bb), (0, 0), arrowprops=dict(arrowstyle="-|>", color=LINK_COLORS[_l], lw=3, mutation_scale=16))
    _axd.axhline(0, color="0.85", lw=0.6, zorder=0); _axd.axvline(0, color="0.85", lw=0.6, zorder=0)
    _axd.set_xlim(-1.2, 1.2); _axd.set_ylim(-1.2, 1.2); _axd.set_aspect("equal"); _axd.set_xticks([]); _axd.set_yticks([])
    _axd.set_xlabel("generation axis"); _axd.set_ylabel("sibling axis"); _axd.set_title("link directions (per-person codes)", fontsize=11)
    for _s in range(len(_t) - 1):                                           # moves between directly related people
        _l = _rel.get((int(_t[_s]), int(_t[_s + 1])))
        if _l is not None:
            _axp.annotate("", _Z[_s + 1], _Z[_s], arrowprops=dict(arrowstyle="-|>", lw=2.2, color=LINK_COLORS[_l]))
    _axp.scatter(*_Z.T, s=10, color="0.6", zorder=0)
    for _p in sorted(set(_t.tolist())):
        _axp.annotate(PEOPLE[_p], _Z[_t == _p].mean(0), textcoords="offset points", xytext=(5, -3), fontsize=7.5, color="0.3")
    _lo, _hi = _Z.min(0), _Z.max(0); _pad = 0.15 * (_hi - _lo).max()
    _axp.set_xlim(_lo[0] - _pad, _hi[0] + _pad); _axp.set_ylim(_lo[1] - _pad, _hi[1] + _pad)
    _axp.set_aspect("equal"); _axp.set_xticks([]); _axp.set_yticks([])
    _axp.set_title(f"one sequence read mention by mention ({len(_t)} mentions)", fontsize=11)
    _axp.set_xlabel(" · ".join(PEOPLE[i] for i in _t[:9]) + " · …", fontsize=8)
    link_gram(_axg, family[IN_CONTEXT]); _axg.set_title("link cosines, all test sequences", fontsize=11)
    _fig.legend([plt.Line2D([], [], color=c, lw=3) for c in LINK_COLORS], LINK_TYPES, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04))
    save_figure(_fig, figure_path(_run["config"], "in_context_parse"))
    return


if __name__ == "__main__":
    app.run()
