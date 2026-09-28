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
    DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    DOMAINS = {"months": {"words": MONTHS, "spec": {"structure": "ring", "n": 12}, "cmap": None},
               "days": {"words": DAYS, "spec": {"structure": "ring", "n": 7}, "cmap": None},
               "colors": {"words": list(HEX), "spec": None, "cmap": None},
               "numbers": {"words": [str(n) for n in range(1, 501)], "spec": None, "cmap": "magma"}}
    # cmap: matplotlib map for list-order colouring (None = the house cyclic map); the colour words use their own hexes

    return DOMAINS, HEX


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
    ## Llama-3.1-8B representations

    The residual at the word's last token, at every layer (layer 0 is the input embedding), averaged over six
    sentences that name the category and end with the word, in Karkada et al.'s form: "The month of the year
    is May", "The color of the object is red", "The number is 42". Cached per domain and prompt set.
    """)
    return


@app.cell
def llm_config(CACHE, DOMAINS):
    import hashlib

    LLM_PATH, LLM_NAME = "meta-llama/Llama-3.1-8B", "llama-3.1-8b"
    LLM_DIR = CACHE / "corpus" / "llm" / LLM_NAME; LLM_DIR.mkdir(parents=True, exist_ok=True)
    CASE = {"months": str.capitalize, "days": str.capitalize}                                              # the corpus token as written in text
    PROMPTS = {                                                                     # the word follows; read at its last token
        "months": ["The month of the year is", "It happened in the month of", "She was born in the month of",
                   "The calendar month is", "The event is scheduled for the month of", "The month written on the form is"],
        "colors": ["The color of the object is", "The color she chose is", "The wall is painted the color",
                   "The color of the car is", "Her favorite color is", "The color of the shirt is"],
        "numbers": ["The number is", "The number of items is", "The number she picked is", "The count came to the number",
                    "The number written on the card is", "The winning number is"],
        "days": ["The day of the week is", "It happened on", "She was born on a", "The meeting is on",
                 "The day written on the form is", "The weekday is"]}
    READS = {"category": lambda name, word: PROMPTS[name]}                          # -> prefixes for one word


    def llm_form(name, word):
        """The word as it appears in running text, with its leading space."""
        return " " + CASE.get(name, str)(word)


    def llm_cache(name, read):
        """``(path, words)`` for one domain and read; the file is keyed by the words and the prompts."""
        words = DOMAINS[name]["words"]
        key = hashlib.md5(" ".join([llm_form(name, w) for w in words] + [read] + PROMPTS[name]).encode()).hexdigest()[:8]
        return LLM_DIR / f"{name}_{read}_{key}.npz", words


    return LLM_PATH, READS, llm_cache, llm_form


@app.cell
def llm_reps(
    AutoModelForCausalLM,
    AutoTokenizer,
    DOMAINS,
    LLM_PATH,
    READS,
    llm_cache,
    llm_form,
    np,
    torch,
    word_means,
):
    _missing = [(n, r) for n in DOMAINS for r in READS if not llm_cache(n, r)[0].exists()]
    if _missing:                                        # the expensive step, cached raw per domain and read
        _tok = AutoTokenizer.from_pretrained(LLM_PATH)
        _model = AutoModelForCausalLM.from_pretrained(LLM_PATH, dtype=torch.bfloat16, device_map="cuda").eval()
        for _n, _r in _missing:
            _path, _words = llm_cache(_n, _r)
            _reps = []
            for _w in _words:                           # prefixes may depend on the word, so one call per word
                _t = [([_tok.bos_token_id] + _tok.encode(p, add_special_tokens=False), []) for p in READS[_r](_n, _w)]
                _reps.append(word_means(_model, {_w: _tok.encode(llm_form(_n, _w), add_special_tokens=False)}, _t)[:, 0])
            np.savez(_path, words=np.array(_words), residual=np.stack(_reps, 1))
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
    Llama's representation of the same words, per layer. Each panel below opens at the layer with the best
    mean over corpora; its selector picks any other.
    """)
    return


@app.cell
def llm_scores(DOMAINS, READS, corpora, llm, np, predicted, rsa):
    READ = "category"                                                                    # the read the panels use
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
    READ,
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
                ax.plot(LAYERS, scores[name, c, READ], label=c)
            ax.axvline(best_layer(name), color="0.8", lw=0.8)
            ax.axhline(0, color="0.7", lw=0.8)
            ax.set_title(name); ax.set_xlabel("layer"); ax.set_ylim(-0.3, 1)
        axes[0].set_ylabel("RSA(corpus block code, Llama)"); axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"{weighting} window; layer 0 = input embedding", y=1.02)
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
            if len(concepts) <= 31:
                ax.set_xticks(range(len(concepts))); ax.set_yticks(range(len(concepts)))
                ax.set_xticklabels(concepts, rotation=90, fontsize=6); ax.set_yticklabels(concepts, fontsize=6)
            else:
                ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{c}"); fig.colorbar(im, ax=ax, shrink=0.75)
        fig.suptitle(f"{name}: row-centred log P(j | i), {weighting} window", y=1.0)
        fig.tight_layout()
        return plot_utils.save_figure(fig, FIGURES / f"blocks_{name}_{tag_w}.png")


    def domain_panel(name, layer="best"):
        """Llama's layer (the best over corpora, or a chosen one) on the left, each corpus's predicted code
        beside it, Grams below with RSA."""
        L = best_layer(name) if layer == "best" else int(layer)
        concepts = DOMAINS[name]["words"]
        hexes = item_colors(name, concepts)
        fig = plot_utils.plot_embedding_panels(llm[name][READ][L], {c: codes[name, c] for c in corpora}, DOMAINS[name]["spec"],
                                               cmap=ListedColormap(hexes) if hexes else DOMAINS[name]["cmap"],
                                               scores={c: scores[name, c, READ][L] for c in corpora}, score_label="RSA",
                                               first_name=f"Llama L{L}", starred="rest")
        fig.suptitle(f"{name}, {weighting} window: Llama-3.1-8B beside the corpus-predicted codes", y=1.02)
        return plot_utils.save_figure(fig, FIGURES / f"domain_{name}_{tag_w}.png")


    return block_heatmaps, domain_panel


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Panels

    Pick a domain and a Llama layer. First the corpora's row-centred blocks side by side, then Llama's
    representation beside the code predicted from each corpus, Grams below with RSA.
    """)
    return


@app.cell
def panel_controls(DOMAINS, LAYERS, mo):
    view_domain = mo.ui.dropdown(options=list(DOMAINS), value="months", label="domain")
    view_layer = mo.ui.dropdown(options=["best"] + [str(l) for l in LAYERS], value="best", label="Llama layer")
    mo.hstack([view_domain, view_layer], justify="start")

    return view_domain, view_layer


@app.cell
def _(block_heatmaps, view_domain):
    block_heatmaps(view_domain.value)
    return


@app.cell
def _(domain_panel, view_domain, view_layer):
    domain_panel(view_domain.value, view_layer.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interactive 3-D view

    The domain and layer chosen above; pick a corpus and any three of the first six PCs. Llama on the left,
    the corpus-predicted code on the right, rotatable; hover for the word.
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
    plt,
    weighting,
):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from matplotlib.colors import to_hex


    def domain_hex(name):
        """One hex colour per item, the same map the panels use: the colour words' own colours, else the
        domain's colormap (or the house cyclic map) sampled in list order."""
        words = DOMAINS[name]["words"]
        hexes = item_colors(name, words)
        if hexes:
            return hexes
        cmap = plt.get_cmap(DOMAINS[name]["cmap"] or plot_utils.FIG.cyclic_cmap)
        return [to_hex(cmap(x)) for x in np.linspace(0, 1, len(words))]


    def code_3d(name, corpus, pcs=(0, 1, 2), layer="best"):
        """Three chosen PCs (0-based) of a Llama layer and of a corpus's block code; hover shows the word."""
        L = best_layer(name) if layer == "best" else int(layer)
        concepts = DOMAINS[name]["words"]
        fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
                            subplot_titles=(f"Llama L{L}", f"{corpus} block code"))
        for col, E in ((1, llm[name][READ][L]), (2, codes[name, corpus])):
            P = plot_utils._pca(E, max(pcs) + 1)[:, list(pcs)]
            fig.add_trace(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers", text=concepts, hoverinfo="text",
                                       marker=dict(size=5, color=domain_hex(name))), row=1, col=col)
        fig.update_layout(height=600, autosize=True, showlegend=False,
                          title=f"{name}, {weighting}: PCs {', '.join(str(p + 1) for p in pcs)}")
        for scene in ("scene", "scene2"):
            fig.layout[scene].update(xaxis_title=f"PC{pcs[0] + 1}", yaxis_title=f"PC{pcs[1] + 1}", zaxis_title=f"PC{pcs[2] + 1}")
        return mo.ui.plotly(fig)


    return (code_3d,)


@app.cell
def _(corpora, mo):
    _pcs = {f"PC{i}": i - 1 for i in range(1, 7)}
    view_corpus = mo.ui.dropdown(options=list(corpora), value=next(iter(corpora)), label="corpus")
    view_x = mo.ui.dropdown(options=_pcs, value="PC1", label="x")
    view_y = mo.ui.dropdown(options=_pcs, value="PC2", label="y")
    view_z = mo.ui.dropdown(options=_pcs, value="PC3", label="z")
    mo.hstack([view_corpus, view_x, view_y, view_z], justify="start")

    return view_corpus, view_x, view_y, view_z


@app.cell
def _(code_3d, view_corpus, view_domain, view_layer, view_x, view_y, view_z):
    code_3d(view_domain.value, view_corpus.value, pcs=(view_x.value, view_y.value, view_z.value), layer=view_layer.value)
    return


if __name__ == "__main__":
    app.run()
