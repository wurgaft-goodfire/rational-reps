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
    # Concept geometry from Wikipedia co-occurrence: the concept × concept block

    Counts from `count_wikipedia.py`: English Wikipedia (2023-11, articles of at least 200 words,
    lowercased), every occurrence of a concept word counted against the tokens within 16 positions.
    Here only the block among the concepts themselves is used: how often each month occurs near each
    other month, each color near each other color, each number near each other number. The block is
    normalised as Karkada et al.'s $M^\star_{ij} = (P_{ij} - P_i P_j) / \tfrac12 (P_{ij} + P_i P_j)$
    and factored into the predicted code $E = U\sqrt{S}$ (row-centred).
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
    from src.code.geometry import predicted_geometry
    from src.code import plot_utils

    CACHE = Path(os.environ.get("CACHE_DIR", ROOT / "outputs"))
    FIGURES = CACHE / "figures"; FIGURES.mkdir(parents=True, exist_ok=True)
    return CACHE, FIGURES, np, plot_utils, plt, predicted_geometry


@app.cell
def _(CACHE, np):
    WINDOW = 16
    CORPORA = {"wikipedia": "wikipedia_20231101_en", "babylm": "babylm_100M"}       # corpus -> table tag
    DOMAINS = {"months": {"spec": {"structure": "ring", "n": 12}},
               "colors": {"spec": None},
               "numbers": {"spec": None}}
    tables = {corpus: {name: dict(np.load(CACHE / "corpus" / f"{tag}_{name}_w{WINDOW}.npz")) for name in DOMAINS}
              for corpus, tag in CORPORA.items()}
    for _corpus, _ts in tables.items():
        for _name, _t in _ts.items():
            print(f"{_corpus:10s} {_name:8s} occurrences {int(_t['concept_counts'].min()):>9,d} to {int(_t['concept_counts'].max()):>10,d}; tokens {int(_t['total']):,}")

    return CORPORA, DOMAINS, WINDOW, tables


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Blocks

    $P_{ij}$ is the pair rate per window position, $P_i$ and $P_j$ the concepts' token rates. The
    block is symmetric up to counting noise; its diagonal is a concept near another copy of itself.
    """)
    return


@app.cell
def _(WINDOW, np, predicted_geometry, tables):
    from matplotlib.colors import ListedColormap

    HEX = {"red": "#e60000", "scarlet": "#ff2400", "crimson": "#dc143c", "maroon": "#800000", "orange": "#ff8c00",
           "beige": "#d9c9a3", "yellow": "#f2d600", "lime": "#32cd32", "olive": "#808000", "green": "#008000",
           "teal": "#008080", "turquoise": "#40e0d0", "cyan": "#00b7eb", "blue": "#0000ff", "indigo": "#4b0082",
           "violet": "#8f00ff", "purple": "#800080", "magenta": "#ff00ff", "pink": "#ff69b4", "brown": "#8b4513",
           "black": "#000000", "white": "#d0d0d0", "gray": "#808080"}


    def item_colors(name, concepts):
        """Hex colour per item: the colour words in their own colour, else None (list-order colouring)."""
        return [HEX[c] for c in concepts] if name == "colors" else None


    def block(name, drop=(), corpus="wikipedia"):
        """The concept x concept block of a domain's table as bounded PMI, and the concept labels."""
        t = tables[corpus][name]
        keep = [i for i, c in enumerate(t["concepts"]) if c not in drop]
        concepts = [str(t["concepts"][i]) for i in keep]
        C = t["block"][np.ix_(keep, keep)]                       # concept x concept counts, saved by the pass
        P_ij = C / (2 * WINDOW * t["total"])
        P_i = t["concept_counts"][keep][:, None] / t["total"]
        return (P_ij - P_i * P_i.T) / (0.5 * (P_ij + P_i * P_i.T)), concepts


    def rows(name, drop=(), corpus="wikipedia"):
        """The concept x vocabulary rows of a domain's table (all 25k context tokens) as bounded PMI."""
        t = tables[corpus][name]
        keep = [i for i, c in enumerate(t["concepts"]) if c not in drop]
        C = t["C"][keep].astype(np.float64)
        P_ij = C / (2 * WINDOW * t["total"])
        P_i, P_j = t["concept_counts"][keep][:, None] / t["total"], t["word_counts"][None] / t["total"]
        return (P_ij - P_i * P_j) / (0.5 * (P_ij + P_i * P_j))


    def predicted(name, drop=(), corpus="wikipedia"):
        """``(E_block, E_rows, concepts)``: the codes factored from the block and from the rows."""
        M, concepts = block(name, drop, corpus)
        return predicted_geometry(M), predicted_geometry(rows(name, drop, corpus)), concepts


    return ListedColormap, block, item_colors, predicted


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Months

    Left: the block $M^\star$ itself. Right: the predicted code's top two PCs and its Gram. Once
    with all twelve months, once without "may" (the modal verb shares the token).
    """)
    return


@app.cell
def _(FIGURES, block, plot_utils, plt):
    def block_heatmap(name, drop=(), corpus="wikipedia"):
        M, concepts = block(name, drop, corpus)
        fig, ax = plt.subplots(figsize=(5.5, 5))
        lim = abs(M).max()
        im = ax.imshow(M, cmap=plot_utils.FIG.diverging_cmap, vmin=-lim, vmax=lim)
        ax.set_xticks(range(len(concepts))); ax.set_yticks(range(len(concepts)))
        ax.set_xticklabels(concepts, rotation=90, fontsize=7); ax.set_yticklabels(concepts, fontsize=7)
        ax.set_title(rf"{corpus} {name}: $M^\star$ block"); fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        return plot_utils.save_figure(fig, FIGURES / f"block_{corpus}_{name}{'_drop' if drop else ''}_mstar.png")

    block_heatmap("months")
    return (block_heatmap,)


@app.cell
def _(DOMAINS, FIGURES, ListedColormap, item_colors, plot_utils, predicted):
    def code_panel(name, drop=(), corpus="wikipedia"):
        E_block, E_rows, concepts = predicted(name, drop, corpus)
        hexes = item_colors(name, concepts)
        fig = plot_utils.plot_embedding_panels(E_block, {"rows (25k contexts)": E_rows}, DOMAINS[name]["spec"],
                                               cmap=ListedColormap(hexes) if hexes else None)
        fig.suptitle(f"{corpus} {name}: code from the block (left) and from the rows (right)"
                     + (f", without {', '.join(drop)}" if drop else ""), y=1.02)
        return plot_utils.save_figure(fig, FIGURES / f"block_{corpus}_{name}{'_drop' if drop else ''}_code.png")

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
    gray; numbers 1 to 100 coloured by magnitude.
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
    ## Interactive 3-D view

    Pick the domain and any three of the first six PCs. Block code on the left, rows code (25k contexts)
    on the right, rotatable; hover for the word. Points are coloured in list order (calendar order, hue
    order, magnitude).
    """)
    return


@app.cell
def _(item_colors, mo, np, plot_utils, predicted):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from plotly.colors import cyclical


    def code_3d(name, drop=(), pcs=(0, 1, 2), corpus="wikipedia"):
        """Three chosen PCs (0-based) of the block code and of the rows code, interactive; hover shows the word."""
        E_block, E_rows, concepts = predicted(name, drop, corpus)
        hexes = item_colors(name, concepts)
        fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
                            subplot_titles=("block", "rows (25k contexts)"))
        for col, E in ((1, E_block), (2, E_rows)):
            P = plot_utils._pca(E, max(pcs) + 1)[:, list(pcs)]
            fig.add_trace(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers", text=concepts, hoverinfo="text",
                                       marker=dict(size=5, color=hexes or np.arange(len(concepts)), colorscale=cyclical.Twilight,
                                                   showscale=False)), row=1, col=col)
        fig.update_layout(height=600, autosize=True, showlegend=False,
                          title=f"{corpus} {name}: PCs {', '.join(str(p + 1) for p in pcs)}" + (f", without {', '.join(drop)}" if drop else ""))
        for scene in ("scene", "scene2"):
            fig.layout[scene].update(xaxis_title=f"PC{pcs[0] + 1}", yaxis_title=f"PC{pcs[1] + 1}", zaxis_title=f"PC{pcs[2] + 1}")
        return mo.ui.plotly(fig)



    return (code_3d,)


@app.cell
def _(CORPORA, DOMAINS, mo):
    _pcs = {f"PC{i}": i - 1 for i in range(1, 7)}
    view_corpus = mo.ui.dropdown(options=list(CORPORA), value="wikipedia", label="corpus")
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
