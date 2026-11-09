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
    # Theory tests on the syntactic polar code

    If what a word predicts about its tree neighbours is a sum of one tied term per relation along its path,
    its code is linear in the path: a step per relation, plus terms for its own relation and its parent's
    (the observations above a word see the path with its last steps removed). So an edge's direction is
    $v_t - w_q$: its type $t$, shifted by its parent's relation $q$, the same shift for every child type.
    Probe: the polar probe (all types) at Llama-3.1-8B's best layer, 14, read on the EWT test split.

    1. **Parent effect.** Variance of edge vectors explained by type and by type x parent relation, against
       the parent relation shuffled within each type; and whether the shift a parent relation induces is
       shared across child types.
    2. **Angles between types.** Each relation's predicted direction from the EWT context kernel (rows: tree
       positions; columns: the part of speech at the parent, grandparent and first child by each common
       relation), against the probe's prototype cosines; control: the same from part-of-speech pairs alone.
    """)
    return


@app.cell
def _(mo):
    import os, sys
    from collections import Counter
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    ROOT = mo.notebook_dir().parent.parent    # src/notebooks -> <repo>
    os.chdir(ROOT); sys.path.insert(0, str(ROOT))
    from src.code.treebank import context_counts, parse_conllu, root_paths
    from src.code.syntax_probe import prototypes, sentence_index, train_probe
    from src.code.geometry import predicted_geometry
    from src.code.plot_utils import save_figure

    CACHE = Path(os.environ.get("CACHE_DIR", ROOT / "outputs"))
    FIGURES = CACHE / "figures"; FIGURES.mkdir(parents=True, exist_ok=True)
    return (
        CACHE,
        Counter,
        FIGURES,
        context_counts,
        np,
        parse_conllu,
        plt,
        predicted_geometry,
        prototypes,
        root_paths,
        save_figure,
        sentence_index,
        spearmanr,
        train_probe,
    )


@app.cell
def _(CACHE, Counter, np, parse_conllu, prototypes, train_probe):
    LAYERS = [0, 4, 8, 12, 14, 16, 18, 20, 24, 28, 32]; LAYER = 14
    tr = parse_conllu(CACHE / "ud_ewt/en_ewt-ud-train.conllu"); te = parse_conllu(CACHE / "ud_ewt/en_ewt-ud-test.conllu")
    _f = lambda s: CACHE / f"syntax/llama-3.1-8b/{s}_layers{'-'.join(map(str, LAYERS))}.npy"
    Xtr = np.asarray(np.load(_f("train"), mmap_mode="r")[LAYERS.index(LAYER)], dtype=np.float32)
    Xte = np.asarray(np.load(_f("test"), mmap_mode="r")[LAYERS.index(LAYER)], dtype=np.float32)
    _counts = Counter(r for s in tr for r in s["rels"] if r != "root")
    TYPES = {r: i for i, (r, _) in enumerate(_counts.most_common())}
    COMMON = [r for r, _ in _counts.most_common(14) if r != "punct"]          # the relations analysed
    probe = train_probe(Xtr, tr, TYPES, rank=128, epochs=30, lr=5e-3, lam=10.0, batch_size=200, seed=0)
    P = prototypes(probe, Xtr, tr)
    print("relations analysed:", COMMON)
    return COMMON, P, TYPES, Xte, probe, te, tr


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Parent effect

    Edge vectors $s = z_{head} - z_{dependent}$ in probe space. $R^2$ of the type means, of the (type, parent
    relation) means, and of the latter with parent relations shuffled within each type (100 shuffles): the gain
    over the shuffle is the parent effect. Then the offsets $m_{t,q} - m_t$ of each child type $t$ under parent
    relation $q$: the theory says they are one vector $-w_q$ for every $t$, so offsets of two child types under
    the same parent should align (mean cosine), unlike offsets under different parents.
    """)
    return


@app.cell
def _(COMMON, Xte, np, probe, sentence_index, te):
    _Z = ((Xte - probe["mu"]) / probe["sd"]) @ probe["B"].T
    _rows = [(_Z[k + h] - _Z[k + j], r, s["rels"][h]) for k, s in zip(sentence_index(te), te)
             for j, (h, r) in enumerate(zip(s["heads"], s["rels"])) if h >= 0 and r in COMMON and s["rels"][h] in COMMON + ["root"]]
    S = np.array([v for v, _, _ in _rows]); t_lab = np.array([r for _, r, _ in _rows]); q_lab = np.array([q for _, _, q in _rows])


    def r2(V, g):
        V = V - V.mean(0); res = 0.0
        for x in np.unique(g):
            m = g == x; res += ((V[m] - V[m].mean(0)) ** 2).sum()
        return 1 - res / (V ** 2).sum()


    _rng = np.random.default_rng(0)
    def _shuffle_within(t, q):
        q = q.copy()
        for x in np.unique(t):
            m = np.flatnonzero(t == x); q[m] = q[_rng.permutation(m)]
        return q
    _tq = np.char.add(np.char.add(t_lab, "|"), q_lab)
    _null = [r2(S, np.char.add(np.char.add(t_lab, "|"), _shuffle_within(t_lab, q_lab))) for _ in range(100)]
    print(f"R2 type {r2(S, t_lab):.3f} | type x parent {r2(S, _tq):.3f} | parent shuffled within type {np.mean(_null):.3f} "
          f"(97.5% {np.percentile(_null, 97.5):.3f})")

    # offsets m_{t,q} - m_t, for (t, q) cells with >= 30 edges
    _cells = {}
    for t in np.unique(t_lab):
        mt = S[t_lab == t].mean(0)
        for q in np.unique(q_lab[t_lab == t]):
            m = (t_lab == t) & (q_lab == q)
            if m.sum() >= 30:
                _cells[t, q] = S[m].mean(0) - mt
    _keys = list(_cells); _U = {k: v / np.linalg.norm(v) for k, v in _cells.items()}
    same = [_U[a] @ _U[b] for i, a in enumerate(_keys) for b in _keys[i + 1:] if a[1] == b[1] and a[0] != b[0]]
    diff = [_U[a] @ _U[b] for i, a in enumerate(_keys) for b in _keys[i + 1:] if a[1] != b[1] and a[0] != b[0]]
    print(f"parent offsets of different child types: same parent relation cos {np.mean(same):+.3f} (n={len(same)}), "
          f"different parent relations {np.mean(diff):+.3f} (n={len(diff)})")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Angles between relation types

    The EWT context kernel, row-centred per column block (each block a distribution over parts of speech) and
    factored, $E = U\sqrt{S}$: one predicted code per tree position (positions seen at least 30 times). A
    relation's predicted direction is the mean, over positions $p$ with both $p$ and $p + t$ kept, of
    $E[p] - E[p + t]$ (head minus dependent, as the probe's edges). Control: the same mean of the dependent's
    minus the head's part-of-speech one-hot over training edges. Scored: Spearman correlation, over pairs of the
    analysed relations, of the predicted cosines with the probe prototypes' cosines.
    """)
    return


@app.cell
def _(
    COMMON,
    FIGURES,
    P,
    TYPES,
    context_counts,
    np,
    plt,
    predicted_geometry,
    save_figure,
    spearmanr,
    tr,
):
    UPOS = sorted({u for s in tr for u in s["upos"]})
    _C, _rows, _cols = context_counts(tr, COMMON[:8], UPOS)
    _B = len(UPOS) + 1
    _keep = [i for i, r in enumerate(_rows) if _C[i, :_B].sum() >= 30]
    _C = _C[_keep]; rows = [_rows[i] for i in _keep]
    _L = np.log(_C + 1.0)
    _L = np.concatenate([b - b.mean(1, keepdims=True) for b in np.split(_L, _C.shape[1] // _B, axis=1)], 1)
    E = predicted_geometry(_L)
    _ix = {r: i for i, r in enumerate(rows)}


    def kernel_direction(t):
        d = [E[_ix[p]] - E[_ix[p + (t,)]] for p in rows if p + (t,) in _ix]
        return np.mean(d, 0) if d else None


    _pos = {}
    for s in tr:
        for j, (h, r) in enumerate(zip(s["heads"], s["rels"])):
            if h >= 0 and r in COMMON:
                v = np.zeros(len(UPOS)); v[UPOS.index(s["upos"][h])] += 1; v[UPOS.index(s["upos"][j])] -= 1
                _pos.setdefault(r, []).append(v)
    kern = {t: kernel_direction(t) for t in COMMON}
    types = [t for t in COMMON if kern[t] is not None]
    cos = lambda V: (V / np.linalg.norm(V, axis=1, keepdims=True)) @ (V / np.linalg.norm(V, axis=1, keepdims=True)).T
    G_llm = P[[TYPES[t] for t in types]] @ P[[TYPES[t] for t in types]].T
    G_kern = cos(np.array([kern[t] for t in types]))
    G_pos = cos(np.array([np.mean(_pos[t], 0) for t in types]))
    _iu = np.triu_indices(len(types), 1)
    print(f"{len(rows)} tree positions; {len(types)} relations: {types}")
    print(f"Spearman over relation pairs with the probe's prototype cosines: kernel {spearmanr(G_kern[_iu], G_llm[_iu])[0]:.2f} | "
          f"part of speech only {spearmanr(G_pos[_iu], G_llm[_iu])[0]:.2f} | kernel beyond part of speech (partial) "
          f"{spearmanr(G_kern[_iu] - np.polyval(np.polyfit(G_pos[_iu], G_kern[_iu], 1), G_pos[_iu]), G_llm[_iu] - np.polyval(np.polyfit(G_pos[_iu], G_llm[_iu], 1), G_pos[_iu]))[0]:.2f}")
    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for _ax, (_t, _G) in zip(_axes, {"probe prototypes (Llama layer 14)": G_llm, "predicted: context kernel": G_kern,
                                     "control: part of speech of head and dependent": G_pos}.items()):
        _ax.imshow(_G, cmap="RdBu_r", vmin=-1, vmax=1); _ax.set_title(_t, fontsize=10)
        _ax.set_xticks(range(len(types)), types, rotation=90, fontsize=7); _ax.set_yticks(range(len(types)), types, fontsize=7)
    _fig.tight_layout()
    save_figure(_fig, FIGURES / "syntax_type_angles.png")
    return


if __name__ == "__main__":
    app.run()
