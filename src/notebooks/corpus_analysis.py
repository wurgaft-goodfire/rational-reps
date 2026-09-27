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
    other setting. "Block" is the concept x concept sub-matrix. Per domain, Llama-3.1-8B's representation of the
    words (the layer that agrees best with the corpora) is shown beside the code predicted from each corpus.
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
    from src.code.analysis_utils import rsa
    from src.code.llm import word_means
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    CACHE = Path(os.environ.get("CACHE_DIR", ROOT / "outputs"))
    FIGURES = CACHE / "figures"; FIGURES.mkdir(parents=True, exist_ok=True)
    return (
        AutoModelForCausalLM,
        AutoTokenizer,
        CACHE,
        FIGURES,
        load_corpus,
        log_conditional,
        np,
        plot_utils,
        plt,
        predicted_geometry,
        rsa,
        torch,
        word_means,
    )


@app.cell
def _(mo):
    view_weighting = mo.ui.dropdown(options=["flat", "linear", "1/d"], value="flat", label="window weighting")
    view_weighting
    return (view_weighting,)


@app.cell
def _(CACHE, load_corpus, view_weighting):
    TAGS = {"wikipedia": "wikipedia_20231101_en", "gutenberg": "gutenberg_en", "fineweb": "fineweb_edu_10bt"}
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

    Two reads per word, at every layer (layer 0 is the input embedding). **Prompt end**: three domain
    prompts name the word and the residual is taken at the prompt's final token, e.g. "The color I am
    thinking about is red." **Word token**: six neutral prefixes and the residual at the word's own last
    token. Each is averaged over its prompts; cached per domain and prompt set.
    """)
    return


@app.cell
def llm_config(CACHE, DOMAINS):
    import hashlib

    LLM_PATH, LLM_NAME = "meta-llama/Llama-3.1-8B", "llama-3.1-8b"
    LLM_DIR = CACHE / "corpus" / "llm" / LLM_NAME; LLM_DIR.mkdir(parents=True, exist_ok=True)
    CASE = {"months": str.capitalize}                                              # the corpus token as written in text
    NEUTRAL = [(t, "") for t in ["", "The word", "She said", "I think of", "Consider", "It was"]]   # read at the word
    PROMPTS = {                                                                     # (prefix, suffix), read at the prompt's end
        "months": [("The month I am thinking about is", "."), ("The month of the year is", ". What month is it?"),
                   ("The event took place in", ".")],
        "colors": [("The color I am thinking about is", "."), ("The color of the object is", ". What color is the object?"),
                   ("My favorite color is", ".")],
        "numbers": [("The number I am thinking about is", "."), ("The number of items is", ". How many items are there?"),
                    ("The answer is", ".")]}
    READS = {"end": lambda name: (PROMPTS[name], True), "word": lambda name: (NEUTRAL, False)}   # -> (templates, at_end)


    def llm_form(name, word):
        """The word as it appears in running text, with its leading space."""
        return " " + CASE.get(name, str)(word)


    def llm_cache(name, read):
        """``(path, words, templates, at_end)`` for one domain and read; the file is keyed by words and prompts."""
        words = [llm_form(name, w) for w in DOMAINS[name]["words"]]
        templates, at_end = READS[read](name)
        key = hashlib.md5(" ".join(words + [f"{p}|{s}" for p, s in templates] + [str(at_end)]).encode()).hexdigest()[:8]
        return LLM_DIR / f"{name}_{read}_{key}.npz", words, templates, at_end


    return LLM_PATH, READS, llm_cache


@app.cell
def llm_reps(
    AutoModelForCausalLM,
    AutoTokenizer,
    DOMAINS,
    LLM_PATH,
    READS,
    llm_cache,
    np,
    torch,
    word_means,
):
    _missing = [(n, r) for n in DOMAINS for r in READS if not llm_cache(n, r)[0].exists()]
    if _missing:                                        # the expensive step, cached raw per domain and read
        _tok = AutoTokenizer.from_pretrained(LLM_PATH)
        _model = AutoModelForCausalLM.from_pretrained(LLM_PATH, dtype=torch.bfloat16, device_map="cuda").eval()
        for _n, _r in _missing:
            _path, _words, _templates, _at_end = llm_cache(_n, _r)
            _ids = {w: _tok.encode(w, add_special_tokens=False) for w in _words}
            _t = [([_tok.bos_token_id] + _tok.encode(p, add_special_tokens=False), _tok.encode(s, add_special_tokens=False))
                  for p, s in _templates]
            np.savez(_path, words=np.array(_words), residual=word_means(_model, _ids, _t, at_end=_at_end))
            print(f"{_n} / {_r}: {len(_words)} words cached")
        del _model; torch.cuda.empty_cache()
    llm = {n: {r: np.load(llm_cache(n, r)[0])["residual"] for r in READS} for n in DOMAINS}     # (layers, n, d)
    for _n, _d in llm.items():
        print(f"{_n:8s} " + ", ".join(f"{r} {R.shape}" for r, R in _d.items()))

    return (llm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Agreement by layer

    RSA (Spearman correlation of the Euclidean distance matrices) between each corpus's block code and
    Llama's representation of the same words, per layer. Solid: prompt-end read; dashed: word-token read.
    The panels use the prompt-end read at the layer with the best mean over corpora.
    """)
    return


@app.cell
def llm_scores(DOMAINS, READS, corpora, llm, np, predicted, rsa):
    READ = "end"                                                                    # the read the panels use
    LAYERS = np.arange(llm["months"][READ].shape[0])
    codes = {(n, c): predicted(n, c)[0] for n in DOMAINS for c in corpora}                  # corpus block codes
    scores = {(n, c, r): np.array([rsa(codes[n, c], llm[n][r][l]) for l in LAYERS])
              for n in DOMAINS for c in corpora for r in READS}


    def best_layer(name, read=READ):
        return int(np.mean([scores[name, c, read] for c in corpora], 0).argmax())


    print(f"{'domain':8s} {'read':5s} {'best layer':>10s}  " + "  ".join(f"{c:>10s}" for c in corpora))
    for _n in DOMAINS:
        for _r in READS:
            _l = best_layer(_n, _r)
            print(f"{_n:8s} {_r:5s} {_l:>10d}  " + "  ".join(f"{scores[_n, c, _r][_l]:10.2f}" for c in corpora))

    return LAYERS, READ, best_layer, codes, scores


@app.cell
def rsa_figure(
    DOMAINS,
    FIGURES,
    LAYERS,
    best_layer,
    corpora,
    plot_utils,
    plt,
    scores,
    tag_w,
    weighting,
):
    def rsa_by_layer():
        fig, axes = plt.subplots(1, len(DOMAINS), figsize=(4 * len(DOMAINS), 3.2), sharey=True)
        for ax, name in zip(axes, DOMAINS):
            for c in corpora:
                line, = ax.plot(LAYERS, scores[name, c, "end"], label=c)
                ax.plot(LAYERS, scores[name, c, "word"], ls="--", lw=1, color=line.get_color())
            ax.axvline(best_layer(name), color="0.8", lw=0.8)
            ax.axhline(0, color="0.7", lw=0.8)
            ax.set_title(name); ax.set_xlabel("layer"); ax.set_ylim(-0.3, 1)
        axes[0].set_ylabel("RSA(corpus block code, Llama)"); axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"{weighting} window; solid: prompt end, dashed: word token; layer 0 = input embedding", y=1.02)
        return plot_utils.save_figure(fig, FIGURES / f"corpus_llm_rsa_{tag_w}.png")


    rsa_by_layer()

    return


@app.cell
def panel_functions(
    DOMAINS,
    FIGURES,
    ListedColormap,
    READ,
    best_layer,
    block,
    codes,
    corpora,
    item_colors,
    llm,
    np,
    plot_utils,
    plt,
    scores,
    tag_w,
    weighting,
):
    def block_heatmaps(name):
        """One row: each corpus's row-centred log P(j | i) block for the domain."""
        fig, axes = plt.subplots(1, len(corpora), figsize=(4.6 * len(corpora), 4.4))
        for ax, c in zip(np.atleast_1d(axes), corpora):
            M, concepts = block(name, (), c)
            M = M - M.mean(1, keepdims=True)
            lim = abs(M).max()
            im = ax.imshow(M, cmap=plot_utils.FIG.diverging_cmap, vmin=-lim, vmax=lim)
            ax.set_xticks(range(len(concepts))); ax.set_yticks(range(len(concepts)))
            ax.set_xticklabels(concepts, rotation=90, fontsize=6); ax.set_yticklabels(concepts, fontsize=6)
            ax.set_title(f"{c}"); fig.colorbar(im, ax=ax, shrink=0.75)
        fig.suptitle(f"{name}: row-centred log P(j | i), {weighting} window", y=1.0)
        fig.tight_layout()
        return plot_utils.save_figure(fig, FIGURES / f"blocks_{name}_{tag_w}.png")


    def domain_panel(name):
        """Llama's best layer on the left, each corpus's predicted code beside it, Grams below with RSA."""
        L = best_layer(name)
        concepts = DOMAINS[name]["words"]
        hexes = item_colors(name, concepts)
        fig = plot_utils.plot_embedding_panels(llm[name][READ][L], {c: codes[name, c] for c in corpora}, DOMAINS[name]["spec"],
                                               cmap=ListedColormap(hexes) if hexes else None,
                                               scores={c: scores[name, c, READ][L] for c in corpora}, score_label="RSA",
                                               first_name=f"Llama L{L}", starred="rest")
        fig.suptitle(f"{name}, {weighting} window: Llama-3.1-8B beside the corpus-predicted codes", y=1.02)
        return plot_utils.save_figure(fig, FIGURES / f"domain_{name}_{tag_w}.png")


    return block_heatmaps, domain_panel


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Months

    Twelve months, calendar order; "may" kept.
    """)
    return


@app.cell
def _(block_heatmaps):
    block_heatmaps("months")
    return


@app.cell
def _(domain_panel):
    domain_panel("months")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Colours

    The 11 basic terms in hue order, then brown, black, white, gray.
    """)
    return


@app.cell
def _(block_heatmaps):
    block_heatmaps("colors")
    return


@app.cell
def _(domain_panel):
    domain_panel("colors")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Numbers

    1 to 500, coloured by magnitude.
    """)
    return


@app.cell
def _(block_heatmaps):
    block_heatmaps("numbers")
    return


@app.cell
def _(domain_panel):
    domain_panel("numbers")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interactive 3-D view

    Pick the domain, a corpus and any three of the first six PCs. Llama's best layer on the left, the
    corpus-predicted code on the right, rotatable; hover for the word.
    """)
    return


@app.cell
def view3d(
    DOMAINS,
    READ,
    best_layer,
    codes,
    item_colors,
    llm,
    mo,
    np,
    plot_utils,
    weighting,
):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from plotly.colors import cyclical


    def code_3d(name, corpus, pcs=(0, 1, 2)):
        """Three chosen PCs (0-based) of Llama's best layer and of a corpus's block code; hover shows the word."""
        L = best_layer(name)
        concepts = DOMAINS[name]["words"]
        hexes = item_colors(name, concepts)
        fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
                            subplot_titles=(f"Llama L{L}", f"{corpus} block code"))
        for col, E in ((1, llm[name][READ][L]), (2, codes[name, corpus])):
            P = plot_utils._pca(E, max(pcs) + 1)[:, list(pcs)]
            fig.add_trace(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers", text=concepts, hoverinfo="text",
                                       marker=dict(size=5, color=hexes or np.arange(len(concepts)), colorscale=cyclical.Twilight,
                                                   showscale=False)), row=1, col=col)
        fig.update_layout(height=600, autosize=True, showlegend=False,
                          title=f"{name}, {weighting}: PCs {', '.join(str(p + 1) for p in pcs)}")
        for scene in ("scene", "scene2"):
            fig.layout[scene].update(xaxis_title=f"PC{pcs[0] + 1}", yaxis_title=f"PC{pcs[1] + 1}", zaxis_title=f"PC{pcs[2] + 1}")
        return mo.ui.plotly(fig)


    return (code_3d,)


@app.cell
def _(DOMAINS, corpora, mo):
    _pcs = {f"PC{i}": i - 1 for i in range(1, 7)}
    view_domain = mo.ui.dropdown(options=list(DOMAINS), value="months", label="domain")
    view_corpus = mo.ui.dropdown(options=list(corpora), value=next(iter(corpora)), label="corpus")
    view_x = mo.ui.dropdown(options=_pcs, value="PC1", label="x")
    view_y = mo.ui.dropdown(options=_pcs, value="PC2", label="y")
    view_z = mo.ui.dropdown(options=_pcs, value="PC3", label="z")
    mo.hstack([view_domain, view_corpus, view_x, view_y, view_z], justify="start")

    return view_corpus, view_domain, view_x, view_y, view_z


@app.cell
def _(code_3d, view_corpus, view_domain, view_x, view_y, view_z):
    code_3d(view_domain.value, view_corpus.value, pcs=(view_x.value, view_y.value, view_z.value))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Blocks and codes

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


    def block(name, drop=(), corpus="wikipedia"):
        """The concept x concept block of a domain as log P(j | i), and the concept labels."""
        C, _, _, r = corpora[corpus]
        words = [w for w in DOMAINS[name]["words"] if w not in drop]
        ids = [index[corpus][w] for w in words]
        return log_conditional(C[np.ix_(ids, ids)], r[ids]), words


    def predicted(name, corpus):
        """``(E_block, concepts)``: the code factored from a corpus's block."""
        M, concepts = block(name, (), corpus)
        return predicted_geometry(M), concepts


    return ListedColormap, block, item_colors, predicted


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interactive 3-D view

    Pick the corpus, the domain and any three of the first six PCs. Block code on the left, Llama's best
    layer on the right, rotatable; hover for the word. Points are coloured in list order (calendar order,
    hue order, magnitude).
    """)
    return


if __name__ == "__main__":
    app.run()
