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
    other setting. "Block" is the concept x concept sub-matrix; its code is set beside the representations
    Llama-3.1-8B gives the same words.
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
    from src.code.analysis_utils import cka
    from src.code.llm import input_embeddings, whitened_unembeddings, word_means
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    CACHE = Path(os.environ.get("CACHE_DIR", ROOT / "outputs"))
    FIGURES = CACHE / "figures"; FIGURES.mkdir(parents=True, exist_ok=True)
    return (
        AutoModelForCausalLM,
        AutoTokenizer,
        CACHE,
        FIGURES,
        cka,
        input_embeddings,
        load_corpus,
        log_conditional,
        np,
        plot_utils,
        plt,
        predicted_geometry,
        torch,
        whitened_unembeddings,
        word_means,
    )


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
    ## Llama-3.1-8B representations

    Three representations per word: the input embedding of its token, the residual at the word's last
    token averaged over six neutral templates (all layers), and the unembedding row in Park et al.'s
    causal inner product (whitened over the vocabulary). Cached per domain; a changed word list recomputes.
    """)
    return


@app.cell
def llm_config(CACHE, DOMAINS):
    import hashlib

    LLM_PATH, LLM_NAME = "meta-llama/Llama-3.1-8B", "llama-3.1-8b"
    TEMPLATES = ["", "The word", "She said", "I think of", "Consider", "It was"]   # prefix text; the word follows
    LLM_DIR = CACHE / "corpus" / "llm" / LLM_NAME; LLM_DIR.mkdir(parents=True, exist_ok=True)
    CASE = {"months": str.capitalize}                                              # the corpus token as written in text


    def llm_form(name, word):
        """The word as it appears in running text, with its leading space."""
        return " " + CASE.get(name, str)(word)


    def llm_cache(name):
        """``(path, words)``: the cache file for a domain's LLM representations, keyed by its word list."""
        words = [llm_form(name, w) for w in DOMAINS[name]["words"]]
        key = hashlib.md5(" ".join(words).encode()).hexdigest()[:8]
        return LLM_DIR / f"{name}_{key}.npz", words


    return LLM_PATH, TEMPLATES, llm_cache


@app.cell
def llm_reps(
    AutoModelForCausalLM,
    AutoTokenizer,
    DOMAINS,
    LLM_PATH,
    TEMPLATES,
    input_embeddings,
    llm_cache,
    np,
    torch,
    whitened_unembeddings,
    word_means,
):
    _missing = [n for n in DOMAINS if not llm_cache(n)[0].exists()]
    if _missing:                                        # the expensive step, cached raw per domain
        _tok = AutoTokenizer.from_pretrained(LLM_PATH)
        _model = AutoModelForCausalLM.from_pretrained(LLM_PATH, dtype=torch.bfloat16, device_map="cuda").eval()
        _templates = [([_tok.bos_token_id] + _tok.encode(t, add_special_tokens=False), []) for t in TEMPLATES]
        for _n in _missing:
            _path, _words = llm_cache(_n)
            _ids = {w: _tok.encode(w, add_special_tokens=False) for w in _words}
            np.savez(_path, words=np.array(_words), n_tokens=np.array([len(v) for v in _ids.values()]),
                     residual=word_means(_model, _ids, _templates), input=input_embeddings(_model, _ids),
                     unembedding=whitened_unembeddings(_model, _ids))
            print(f"{_n}: {len(_words)} words cached")
        del _model; torch.cuda.empty_cache()
    llm = {n: dict(np.load(llm_cache(n)[0])) for n in DOMAINS}
    for _n, _d in llm.items():
        print(f"{_n:8s} residual {_d['residual'].shape}, input {_d['input'].shape}, unembedding {_d['unembedding'].shape}, "
              f"multi-token words {int((_d['n_tokens'] > 1).sum())}")

    return (llm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Corpus code against Llama

    Linear CKA between each domain's block code and Llama's representation of the same words, per
    residual layer, with the input embedding (square) and the whitened unembedding (triangle) beside it.
    Months are scored without "may".
    """)
    return


@app.cell
def llm_scores(DOMAINS, block, cka, corpora, llm, np, predicted_geometry):
    LAYERS = np.arange(llm["months"]["residual"].shape[0])
    DROP = {"months": ("may",)}                                        # scored without the modal-verb token


    def llm_reps(name, drop=()):
        """The LLM representations of a domain's words, minus ``drop``: {kind: (n, d) or (layers, n, d)}."""
        keep = [i for i, w in enumerate(DOMAINS[name]["words"]) if w not in drop]
        R = llm[name]
        return {"residual": R["residual"][:, keep], "input": R["input"][keep], "unembedding": R["unembedding"][keep]}


    scores = {}
    for _c in corpora:
        for _n in DOMAINS:
            _E = predicted_geometry(block(_n, DROP.get(_n, ()), _c)[0])
            _R = llm_reps(_n, DROP.get(_n, ()))
            scores[_c, _n] = {"residual": np.array([cka(_E, _R["residual"][l]) for l in LAYERS]),
                              "input": cka(_E, _R["input"]), "unembedding": cka(_E, _R["unembedding"])}


    def best_layer(name, corpus):
        return int(scores[corpus, name]["residual"].argmax())


    print(f"{'corpus':10s} {'domain':8s} {'best layer':>10s} {'CKA':>6s} {'input':>6s} {'unemb':>6s}")
    for (_c, _n), _s in scores.items():
        _l = best_layer(_n, _c)
        print(f"{_c:10s} {_n:8s} {_l:>10d} {_s['residual'][_l]:6.2f} {_s['input']:6.2f} {_s['unembedding']:6.2f}")

    return LAYERS, best_layer, llm_reps, scores


@app.cell
def llm_cka_figure(
    DOMAINS,
    FIGURES,
    LAYERS,
    corpora,
    plot_utils,
    plt,
    scores,
    tag_w,
    weighting,
):
    def cka_by_layer():
        fig, axes = plt.subplots(1, len(DOMAINS), figsize=(4 * len(DOMAINS), 3.2), sharey=True)
        for ax, name in zip(axes, DOMAINS):
            for c in corpora:
                s = scores[c, name]
                line, = ax.plot(LAYERS, s["residual"], label=c)
                ax.scatter([0], [s["input"]], marker="s", color=line.get_color())
                ax.scatter([LAYERS[-1]], [s["unembedding"]], marker="^", color=line.get_color())
            ax.set_title(name); ax.set_xlabel("layer"); ax.set_ylim(0, 1)
        axes[0].set_ylabel("CKA(corpus block code, Llama)"); axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"square: input embedding; triangle: whitened unembedding; {weighting} window", y=1.02)
        return plot_utils.save_figure(fig, FIGURES / f"corpus_llm_cka_{tag_w}.png")


    cka_by_layer()

    return


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


    def predicted(name, drop=(), corpus="wikipedia"):
        """``(E_block, concepts)``: the code factored from the block."""
        M, concepts = block(name, drop, corpus)
        return predicted_geometry(M), concepts


    return ListedColormap, block, item_colors, predicted


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Months

    Left: the PMI block itself. Right: the block code's top two PCs beside Llama's, Grams below with CKA. Once
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
def _(
    DOMAINS,
    FIGURES,
    ListedColormap,
    best_layer,
    cka,
    item_colors,
    llm_reps,
    plot_utils,
    predicted,
    tag_w,
    weighting,
):
    def code_panel(name, drop=(), corpus="wikipedia"):
        E_block, concepts = predicted(name, drop, corpus)
        L = best_layer(name, corpus)
        R = llm_reps(name, drop)
        reps = {f"Llama L{L}": R["residual"][L], "input": R["input"], "unembedding": R["unembedding"]}
        sc = {k: cka(E_block, v) for k, v in reps.items()}
        hexes = item_colors(name, concepts)
        fig = plot_utils.plot_embedding_panels(E_block, reps, DOMAINS[name]["spec"],
                                               cmap=ListedColormap(hexes) if hexes else None, scores=sc)
        fig.suptitle(f"{corpus} {name}, {weighting}: corpus block code beside Llama-3.1-8B"
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
    code_panel("colors", corpus="wikipedia")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interactive 3-D view

    Pick the corpus, the domain and any three of the first six PCs. Block code on the left, Llama's best
    layer on the right, rotatable; hover for the word. Points are coloured in list order (calendar order,
    hue order, magnitude).
    """)
    return


@app.cell
def _(
    best_layer,
    item_colors,
    llm_reps,
    mo,
    np,
    plot_utils,
    predicted,
    weighting,
):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from plotly.colors import cyclical


    def code_3d(name, drop=(), pcs=(0, 1, 2), corpus="wikipedia"):
        """Three chosen PCs (0-based) of the block code and of Llama's best layer, interactive; hover shows the word."""
        E_block, concepts = predicted(name, drop, corpus)
        L = best_layer(name, corpus)
        hexes = item_colors(name, concepts)
        fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
                            subplot_titles=("corpus block", f"Llama layer {L}"))
        for col, E in ((1, E_block), (2, llm_reps(name, drop)["residual"][L])):
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
