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
    # Concept geometry from corpus co-occurrence

    Full 50k x 50k window matrices built by `build_corpus.py` (window 16; lowercased Wikipedia 2023-11,
    BabyLM 100M and Project Gutenberg). A pair $d$ positions apart adds a weight $f(d)$: flat $1$, linear
    $17 - d$ (word2vec's expected window), or $1/d$ (GloVe). The skip-gram log-conditional
    $\log P(j \mid i) = \log C_{ij} / r_i$, with $r_i$ the row's total weight (an empty cell counts as one
    co-occurrence), is factored into the predicted code $E = U\sqrt{S}$ after row centring, as in every
    other setting. "Block" is the concept x concept sub-matrix; "rows" the concept x vocabulary rows.
    """)
    return


@app.cell
def _(mo):
    import os, sys
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt

    ROOT = mo.notebook_dir().parent.parent    # src/notebooks -> <repo>
    os.chdir(ROOT); sys.path.insert(0, str(ROOT))
    from src.code.corpus import load_corpus, log_conditional
    from src.code.geometry import predicted_geometry
    from src.code import plot_utils

    CACHE = Path(os.environ.get("CACHE_DIR", ROOT / "outputs"))
    FIGURES = CACHE / "figures"; FIGURES.mkdir(parents=True, exist_ok=True)
    return CACHE, FIGURES, load_corpus, log_conditional, np, plot_utils, plt, predicted_geometry


@app.cell
def _(mo):
    view_weighting = mo.ui.dropdown(options=["flat", "linear", "1/d"], value="flat", label="window weighting")
    view_weighting
    return (view_weighting,)


@app.cell
def _(CACHE, load_corpus, view_weighting):
    TAGS = {"wikipedia": "wikipedia_20231101_en", "babylm": "babylm_100M", "gutenberg": "gutenberg_en"}
    weighting = view_weighting.value
    tag_w = weighting.replace("/", "")
    corpora = {c: load_corpus(CACHE / "corpus" / tag, weighting=weighting) for c, tag in TAGS.items()
               if (CACHE / "corpus" / tag / f"C_w16_{tag_w}.npy").exists()}           # (C, vocab, freq, row sums)
    index = {c: {w: i for i, w in enumerate(vocab)} for c, (_, vocab, _, _) in corpora.items()}
    for _c, (_C, _vocab, _freq, _r) in corpora.items():
        print(f"{_c:10s} {len(_vocab):,} words, {int(_freq.sum()):>13,d} tokens, matrix weight {_r.sum():.3e}")
    print(f"weighting {weighting}; not built:", sorted(set(TAGS) - set(corpora)) or "none")
    return corpora, index, tag_w, weighting


@app.cell
def _():
    HEX = {"red": "#e60000", "orange": "#ff8c00", "yellow": "#f2d600", "green": "#008000", "blue": "#0000ff",
           "purple": "#800080", "pink": "#ff69b4", "brown": "#8b4513", "black": "#000000", "white": "#d0d0d0",
           "gray": "#808080"}                                     # the 11 basic colour terms, hue order then achromatic
    MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
              "november", "december"]
    DOMAINS = {"months": {"words": MONTHS, "spec": {"structure": "ring", "n": 12}},
               "colors": {"words": list(HEX), "spec": None},
               "numbers": {"words": [str(n) for n in range(1, 501)], "spec": None}}
    return DOMAINS, HEX


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Blocks

    Each row of the block is a concept's conditional over the other concepts; the counts are symmetric,
    the row totals are not. The diagonal is a concept near another copy of itself.
    """)
    return


@app.cell
def _(DOMAINS, HEX, corpora, index, log_conditional, np, predicted_geometry):
    from matplotlib.colors import ListedColormap


    def item_colors(name, concepts):
        """Hex colour per item: the colour words in their own colour, else None (list-order colouring)."""
        return [HEX[c] for c in concepts] if name == "colors" else None


    def concept_ids(name, drop=(), corpus="wikipedia"):
        words = [w for w in DOMAINS[name]["words"] if w not in drop]
        return [index[corpus][w] for w in words], words


    def block(name, drop=(), corpus="wikipedia"):
        """The concept x concept block of a domain as log P(j | i), and the concept labels."""
        C, _, _, r = corpora[corpus]
        ids, words = concept_ids(name, drop, corpus)
        return log_conditional(C[np.ix_(ids, ids)], r[ids]), words


    def rows(name, drop=(), corpus="wikipedia"):
        """The concept x vocabulary rows of a domain (all 50k context words) as log P(j | i)."""
        C, _, _, r = corpora[corpus]
        ids, _ = concept_ids(name, drop, corpus)
        return log_conditional(C[ids], r[ids])


    def predicted(name, drop=(), corpus="wikipedia"):
        """``(E_block, E_rows, concepts)``: the codes factored from the block and from the rows."""
        M, concepts = block(name, drop, corpus)
        return predicted_geometry(M), predicted_geometry(rows(name, drop, corpus)), concepts


    return ListedColormap, block, item_colors, predicted


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Months

    Left: the PMI block itself. Right: the predicted code's top two PCs and its Gram. Once
    with all twelve months, once without "may" (the modal verb shares the token).
    """)
    return


@app.cell
def _(FIGURES, block, plot_utils, plt, tag_w, weighting):
    def block_heatmap(name, drop=(), corpus="wikipedia"):
        M, concepts = block(name, drop, corpus)
        M = M - M.mean(1, keepdims=True)                          # row-centred, as the factorisation sees it
        fig, ax = plt.subplots(figsize=(5.5, 5))
        lim = abs(M).max()
        im = ax.imshow(M, cmap=plot_utils.FIG.diverging_cmap, vmin=-lim, vmax=lim)
        ax.set_xticks(range(len(concepts))); ax.set_yticks(range(len(concepts)))
        ax.set_xticklabels(concepts, rotation=90, fontsize=7); ax.set_yticklabels(concepts, fontsize=7)
        ax.set_title(f"{corpus} {name}: row-centred log P(j | i), {weighting}"); fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        return plot_utils.save_figure(fig, FIGURES / f"block_{corpus}_{name}{'_drop' if drop else ''}_logcond_{tag_w}.png")

    block_heatmap("months")
    return (block_heatmap,)


@app.cell
def _(DOMAINS, FIGURES, ListedColormap, item_colors, plot_utils, predicted, tag_w, weighting):
    def code_panel(name, drop=(), corpus="wikipedia"):
        E_block, E_rows, concepts = predicted(name, drop, corpus)
        hexes = item_colors(name, concepts)
        fig = plot_utils.plot_embedding_panels(E_block, {"rows (50k contexts)": E_rows}, DOMAINS[name]["spec"],
                                               cmap=ListedColormap(hexes) if hexes else None)
        fig.suptitle(f"{corpus} {name}, {weighting}: code from the block (left) and from the rows (right)"
                     + (f", without {', '.join(drop)}" if drop else ""), y=1.02)
        return plot_utils.save_figure(fig, FIGURES / f"block_{corpus}_{name}{'_drop' if drop else ''}_code_{tag_w}.png")

    code_panel("months")

    return (code_panel,)


@app.cell
def _(code_panel):
    code_panel("months", drop=("may",))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Colors and numbers

    Colors in hue order (red, orange, yellow, green, blue, purple, pink) then brown, black, white,
    gray; numbers 1 to 500 coloured by magnitude.
    """)
    return


@app.cell
def _(block_heatmap):
    block_heatmap("colors")
    return


@app.cell
def _(code_panel):
    code_panel("colors")
    return


@app.cell
def _(block_heatmap):
    block_heatmap("numbers")
    return


@app.cell
def _(code_panel):
    code_panel("numbers")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Gutenberg colours

    Fiction names colours far more often than Wikipedia does; the same block and rows codes from the
    Gutenberg matrix.
    """)
    return


@app.cell
def _(block_heatmap):
    block_heatmap("colors", corpus="gutenberg")
    return


@app.cell
def _(code_panel):
    code_panel("colors", corpus="gutenberg")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interactive 3-D view

    Pick the corpus, the domain and any three of the first six PCs. Block code on the left, rows code
    on the right, rotatable; hover for the word. Points are coloured in list order (calendar order,
    hue order, magnitude).
    """)
    return


@app.cell
def _(item_colors, mo, np, plot_utils, predicted, weighting):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from plotly.colors import cyclical


    def code_3d(name, drop=(), pcs=(0, 1, 2), corpus="wikipedia"):
        """Three chosen PCs (0-based) of the block code and of the rows code, interactive; hover shows the word."""
        E_block, E_rows, concepts = predicted(name, drop, corpus)
        hexes = item_colors(name, concepts)
        fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
                            subplot_titles=("block", "rows (50k contexts)"))
        for col, E in ((1, E_block), (2, E_rows)):
            P = plot_utils._pca(E, max(pcs) + 1)[:, list(pcs)]
            fig.add_trace(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers", text=concepts, hoverinfo="text",
                                       marker=dict(size=5, color=hexes or np.arange(len(concepts)), colorscale=cyclical.Twilight,
                                                   showscale=False)), row=1, col=col)
        fig.update_layout(height=600, autosize=True, showlegend=False,
                          title=f"{corpus} {name}, {weighting}: PCs {', '.join(str(p + 1) for p in pcs)}" + (f", without {', '.join(drop)}" if drop else ""))
        for scene in ("scene", "scene2"):
            fig.layout[scene].update(xaxis_title=f"PC{pcs[0] + 1}", yaxis_title=f"PC{pcs[1] + 1}", zaxis_title=f"PC{pcs[2] + 1}")
        return mo.ui.plotly(fig)



    return (code_3d,)


@app.cell
def _(DOMAINS, corpora, mo):
    _pcs = {f"PC{i}": i - 1 for i in range(1, 7)}
    view_corpus = mo.ui.dropdown(options=list(corpora), value="wikipedia", label="corpus")
    view_domain = mo.ui.dropdown(options=list(DOMAINS), value="months", label="domain")
    view_x = mo.ui.dropdown(options=_pcs, value="PC1", label="x")
    view_y = mo.ui.dropdown(options=_pcs, value="PC2", label="y")
    view_z = mo.ui.dropdown(options=_pcs, value="PC3", label="z")
    view_drop_may = mo.ui.checkbox(value=True, label="drop may (months)")
    mo.hstack([view_corpus, view_domain, view_x, view_y, view_z, view_drop_may], justify="start")

    return view_corpus, view_domain, view_drop_may, view_x, view_y, view_z


@app.cell
def _(
    code_3d,
    view_corpus,
    view_domain,
    view_drop_may,
    view_x,
    view_y,
    view_z,
):
    code_3d(view_domain.value,
            drop=("may",) if view_drop_may.value and view_domain.value == "months" else (),
            pcs=(view_x.value, view_y.value, view_z.value), corpus=view_corpus.value)

    return


if __name__ == "__main__":
    app.run()
