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
    from src.code.corpus import (analogy_accuracy, binary_subtrees, coarse_to_fine, collapse_phrases, concept_rows,
                                 load_corpus, log_conditional, orthogonality, read_analogies, wordnet_hierarchy)
    from src.code.geometry import predicted_geometry
    from src.code import plot_utils
    from src.code.analysis_utils import cka, cka_null, cka_shuffle_null, rsa
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
        analogy_accuracy,
        binary_subtrees,
        cka,
        cka_null,
        cka_shuffle_null,
        coarse_to_fine,
        collapse_phrases,
        concept_rows,
        load_corpus,
        log_conditional,
        np,
        orthogonality,
        plot_utils,
        plt,
        predicted_geometry,
        read_analogies,
        rsa,
        torch,
        word_means,
        wordnet_hierarchy,
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
    return TAGS, corpora, index, tag_w, weighting


@app.cell
def _():
    HEX = {"red": "#e60000", "orange": "#ff8c00", "yellow": "#f2d600", "green": "#008000", "blue": "#0000ff",
           "purple": "#800080", "pink": "#ff69b4", "brown": "#8b4513", "black": "#000000", "white": "#d0d0d0",
           "gray": "#808080"}                                     # the 11 basic colour terms, hue order then achromatic
    MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
              "november", "december"]
    DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    YEARS = [str(y) for y in range(1700, 1974)]                    # Karkada's historical years, cut where Gutenberg's vocabulary ends
    STATES = ["alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut", "delaware", "florida",
              "georgia", "hawaii", "idaho", "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana", "maine",
              "maryland", "massachusetts", "michigan", "minnesota", "mississippi", "missouri", "montana", "nebraska",
              "nevada", "new hampshire", "new jersey", "new mexico", "new york", "north carolina", "north dakota", "ohio",
              "oklahoma", "oregon", "pennsylvania", "rhode island", "south carolina", "south dakota", "tennessee", "texas",
              "utah", "vermont", "virginia", "washington", "west virginia", "wisconsin", "wyoming"]
    # all 50 states; the two-word names are counted as phrases in the id stream (flat window only)
    DOMAINS = {"months": {"words": MONTHS, "spec": {"structure": "ring", "n": 12}, "cmap": None},
               "days": {"words": DAYS, "spec": {"structure": "ring", "n": 7}, "cmap": None},
               "colors": {"words": list(HEX), "spec": None, "cmap": None},
               "numbers": {"words": [str(n) for n in range(1, 501)], "spec": None, "cmap": "magma"},
               "years": {"words": YEARS, "spec": None, "cmap": "magma"},
               "states": {"words": STATES, "spec": None, "cmap": None}}
    # cmap: matplotlib map for list-order colouring (None = the house cyclic map); the colour words use their own hexes

    return DOMAINS, HEX


@app.cell
def _(CACHE, DOMAINS, TAGS, collapse_phrases, concept_rows, corpora, index, np):
    def concept_rows_for(name, corpus):
        """``(R, cols)`` for a domain with multi-word items: flat window co-occurrence rows over the vocabulary
        plus one id per phrase, counted from the corpus's id stream and cached beside its matrices; ``cols``
        are the items' own ids, so ``R[:, cols]`` is the block."""
        tag = TAGS[corpus]; path = CACHE / "corpus" / tag / f"rows_{name}.npz"
        words = DOMAINS[name]["words"]
        if not path.exists():
            ix = index[corpus]; V = len(corpora[corpus][1])
            phrases = [[ix[t] for t in w.split()] for w in words if " " in w]
            lengths = np.load(CACHE / "corpus" / tag / "doc_lengths.npy")
            ids, docs = collapse_phrases(np.fromfile(CACHE / "corpus" / tag / "ids.bin", dtype=np.uint16),
                                         np.repeat(np.arange(len(lengths), dtype=np.int32), lengths), phrases, V)
            cols = [V + phrases.index([ix[t] for t in w.split()]) if " " in w else ix[w] for w in words]
            np.savez(path, rows=concept_rows(ids, docs, cols, V + len(phrases)), cols=np.array(cols))
        d = np.load(path)
        return d["rows"], d["cols"]


    return (concept_rows_for,)


@app.cell
def _(DOMAINS, HEX, concept_rows_for, corpora, index, log_conditional, np, predicted_geometry):
    from matplotlib.colors import ListedColormap


    def item_colors(name, concepts):
        """Hex colour per item: the colour words in their own colour, else None (list-order colouring)."""
        return [HEX[c] for c in concepts] if name == "colors" else None


    def block(name, drop=(), corpus="wikipedia"):
        """The concept x concept block of a domain as log P(j | i), and the concept labels."""
        words = [w for w in DOMAINS[name]["words"] if w not in drop]
        if any(" " in w for w in words):                          # multi-word items: rows counted from the id stream
            R, cols = concept_rows_for(name, corpus)
            return log_conditional(R[:, cols], R.sum(1)), words
        C, _, _, r = corpora[corpus]
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
    CASE = {"months": str.capitalize, "days": str.capitalize, "states": str.title}                         # the corpus token as written in text
    PROMPTS = {                                                                     # the word follows; read at its last token
        "months": ["The month of the year is", "It happened in the month of", "She was born in the month of",
                   "The calendar month is", "The event is scheduled for the month of", "The month written on the form is"],
        "colors": ["The color of the object is", "The color she chose is", "The wall is painted the color",
                   "The color of the car is", "Her favorite color is", "The color of the shirt is"],
        "numbers": ["The number is", "The number of items is", "The number she picked is", "The count came to the number",
                    "The number written on the card is", "The winning number is"],
        "days": ["The day of the week is", "It happened on", "She was born on a", "The meeting is on",
                 "The day written on the form is", "The weekday is"],
        "years": ["In the year", "It happened in the year", "She was born in the year", "The year is",
                  "The event took place in the year", "The year written on the document is"],
        "states": ["The location of the US state", "She lives in the US state of", "The US state is",
                   "They moved to the state of", "The state written on the form is", "It is located in the state of"]}
    READS = {"category": lambda name, word: PROMPTS[name]}                          # -> prefixes for one word


    def llm_form(name, word):
        """The word as it appears in running text, with its leading space."""
        return " " + CASE.get(name, str)(word)


    def llm_cache(name, read):
        """``(path, words)`` for one domain and read; the file is keyed by the words and the prompts."""
        words = DOMAINS[name]["words"]
        key = hashlib.md5(" ".join([llm_form(name, w) for w in words] + [read] + PROMPTS[name]).encode()).hexdigest()[:8]
        return LLM_DIR / f"{name}_{read}_{key}.npz", words


    return LLM_DIR, LLM_PATH, READS, llm_cache, llm_form


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
def llm_scores(DOMAINS, READS, cka, cka_null, cka_shuffle_null, corpora, llm, np, predicted, rsa):
    READ = "category"                                                                    # the read the panels use
    LAYERS = np.arange(llm["months"][READ].shape[0])
    codes = {(n, c): predicted(n, c)[0] for n in DOMAINS for c in corpora}                  # corpus block codes
    scores = {(n, c, r): np.array([rsa(codes[n, c], llm[n][r][l]) for l in LAYERS])
              for n in DOMAINS for c in corpora for r in READS}                            # RSA picks the layer
    scores_cka = {(n, c, r): np.array([cka(codes[n, c], llm[n][r][l]) for l in LAYERS])
                  for n in DOMAINS for c in corpora for r in READS}


    def best_layer(name, read=READ):
        return int(np.mean([scores[name, c, read] for c in corpora], 0).argmax())


    print(f"{'domain':8s} {'best layer':>10s}  " + "  ".join(f"{c + ' RSA/CKA':>16s}" for c in corpora) + "   CKA floor random/shuffled")
    for _n in DOMAINS:
        for _r in READS:
            _l = best_layer(_n, _r)
            _L = llm[_n][_r][_l]
            print(f"{_n:8s} {_l:>10d}  " + "  ".join(f"{scores[_n, c, _r][_l]:10.2f}/{scores_cka[_n, c, _r][_l]:.2f}" for c in corpora)
                  + f"   {cka_null(_L):.2f}/{cka_shuffle_null(_L):.2f}")

    return LAYERS, READ, best_layer, codes, scores, scores_cka


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
    scores_cka,
    tag_w,
    weighting,
):
    def rsa_by_layer():
        fig, axes = plt.subplots(1, len(DOMAINS), figsize=(4 * len(DOMAINS), 3.2), sharey=True)
        for ax, name in zip(axes, DOMAINS):
            for c in corpora:
                line, = ax.plot(LAYERS, scores[name, c, READ], label=c)
                ax.plot(LAYERS, scores_cka[name, c, READ], ls="--", lw=0.9, color=line.get_color())
            ax.axvline(best_layer(name), color="0.8", lw=0.8)
            ax.axhline(0, color="0.7", lw=0.8)
            ax.set_title(name); ax.set_xlabel("layer"); ax.set_ylim(-0.3, 1)
        axes[0].set_ylabel("corpus block code vs Llama\nRSA (solid), CKA (dashed)"); axes[0].legend(frameon=False, fontsize=8)
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
    scores_cka,
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
        beside it, Grams below with RSA and CKA."""
        L = best_layer(name) if layer == "best" else int(layer)
        concepts = DOMAINS[name]["words"]
        hexes = item_colors(name, concepts)
        fig = plot_utils.plot_embedding_panels(llm[name][READ][L], {c: codes[name, c] for c in corpora}, DOMAINS[name]["spec"],
                                               cmap=ListedColormap(hexes) if hexes else DOMAINS[name]["cmap"],
                                               scores={c: {"RSA": scores[name, c, READ][L], "CKA": scores_cka[name, c, READ][L]} for c in corpora},
                                               first_name=f"Llama L{L}", starred="rest")
        fig.suptitle(f"{name}, {weighting} window: Llama-3.1-8B beside the corpus-predicted codes", y=1.02)
        return plot_utils.save_figure(fig, FIGURES / f"domain_{name}_{tag_w}.png")


    return block_heatmaps, domain_panel


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Panels

    Pick a domain and a Llama layer. First the corpora's row-centred blocks side by side, then Llama's
    representation beside the code predicted from each corpus, Grams below with RSA and CKA.
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Hierarchy: WordNet subtrees

    Nava and Wyart's setup: WordNet's noun hyponymy contracted to the words every corpus has (a word names a
    sense only when that sense holds most of its WordNet tags, at least one; a few words whose WordNet sense is far from
    their use are excluded), then perfect binary subtrees of depth 3 under a root, 15 words each, drawn uniformly over the
    valid structures. Roots: organism, artifact (WordNet's instrumentality, whose word is "instrumentation"), event. The corpus block code of the 15 words
    against Llama's representation of them, per tree; then two tree measures on both codes against shuffled node order:
    coarse-to-fine alignment, how far the top k PCs span the k coarsest splits (k = 1, 3, 7), and parent-child
    orthogonality, the mean |cos| between a child's offset from its parent and the parent's vector.
    """)
    return


@app.cell
def _(binary_subtrees, corpora, wordnet_hierarchy):
    from nltk.corpus import wordnet as wn
    ROOTS = {"organism": "organism.n.01", "artifact": "instrumentality.n.03", "event": "event.n.01"}   # artifact's word is "instrumentation"
    N_TREES, DEPTH = 40, 3
    EXCLUDE = {"cricket", "queen", "nobody", "gallant", "widower", "beau", "coon", "buff",         # WordNet senses far from the word's use
               "opener", "shear", "crunch", "dealings", "scourge", "gig", "transit", "gourd", "binder", "trapping"}
    _vocabs = [v[1].tolist() for v in corpora.values()]
    _others = [set(v) for v in _vocabs[1:]]
    _shared = [w for w in _vocabs[0] if all(w in s for s in _others) and w not in EXCLUDE]   # words every corpus has, by frequency
    _children, _word = wordnet_hierarchy(_shared)
    trees = {r: [[[_word[s] for s in level] for level in t] for t in binary_subtrees(_children.get, wn.synset(name), DEPTH, N_TREES, seed=0)]
             for r, name in ROOTS.items()}
    tree_words = {r: sorted({w for t in ts for level in t for w in level}) for r, ts in trees.items()}
    for _r, _ts in trees.items():
        print(f"{_r:10s} {len(_ts)} trees, {len(tree_words[_r])} distinct words; e.g. {_ts[0]}")
    return DEPTH, ROOTS, tree_words, trees


@app.cell
def _(AutoModelForCausalLM, AutoTokenizer, LLM_DIR, LLM_PATH, np, torch, word_means):
    import hashlib as _hashlib

    NEUTRAL = ["", "The word", "She said", "I think of", "Consider", "It was"]        # no category for generic words


    def neutral_reps(name, words):
        """{word: (layers, d)}: Llama's residual at each word's last token, averaged over the NEUTRAL prefixes;
        cached as ``{name}_{md5}.npz``."""
        path = LLM_DIR / f"{name}_{_hashlib.md5(' '.join(words + NEUTRAL).encode()).hexdigest()[:8]}.npz"
        if not path.exists():
            tok = AutoTokenizer.from_pretrained(LLM_PATH)
            model = AutoModelForCausalLM.from_pretrained(LLM_PATH, dtype=torch.bfloat16, device_map="cuda").eval()
            templates = [([tok.bos_token_id] + tok.encode(p, add_special_tokens=False), []) for p in NEUTRAL]
            np.savez(path, words=np.array(words),
                     residual=word_means(model, {w: tok.encode(" " + w, add_special_tokens=False) for w in words}, templates))
            del model; torch.cuda.empty_cache()
            print(f"{name}: {len(words)} words cached")
        return dict(zip(words, np.moveaxis(np.load(path)["residual"], 1, 0)))
    return (neutral_reps,)


@app.cell
def _(neutral_reps, tree_words):
    tree_llm = {r: neutral_reps(f"tree_{r}", tree_words[r]) for r in tree_words}       # root -> word -> (layers, d)
    return (tree_llm,)


@app.cell
def _(cka, cka_null, cka_shuffle_null, coarse_to_fine, corpora, index, log_conditional, np, orthogonality, predicted_geometry, rsa, tree_llm, trees):
    def tree_nodes(tree):
        return [w for level in tree for w in level]


    def tree_code(tree, corpus):
        """The corpus block code of a tree's 15 words, in level order."""
        C, _, _, r = corpora[corpus]
        ids = [index[corpus][w] for w in tree_nodes(tree)]
        return predicted_geometry(log_conditional(C[np.ix_(ids, ids)], r[ids]))


    def tree_llm_code(tree, root, layer):
        return np.stack([tree_llm[root][w][layer] for w in tree_nodes(tree)])


    TREE_LAYERS = np.arange(next(iter(tree_llm["organism"].values())).shape[0])


    def tree_scores(metric):
        """root -> (layers, corpora): the metric between corpus and Llama tree codes, mean over the root's trees."""
        return {r: np.array([[np.mean([metric(tree_code(t, c), tree_llm_code(t, r, l)) for t in ts]) for c in corpora]
                             for l in TREE_LAYERS]) for r, ts in trees.items()}


    tree_rsa, tree_cka = tree_scores(rsa), tree_scores(cka)
    TREE_LAYER = int(np.mean([v.mean(1) for v in tree_rsa.values()], 0).argmax())                 # RSA picks the layer


    def tree_measures(root, seed=0):
        """Per code (Llama at TREE_LAYER, each corpus, and a shuffled Llama): coarse-to-fine alignment at k = 1, 3,
        7 and parent-child |cos|, averaged over the root's trees."""
        rng = np.random.default_rng(seed)
        out = {}
        for name, code in [("Llama", lambda t: tree_llm_code(t, root, TREE_LAYER)), *[(c, (lambda c: lambda t: tree_code(t, c))(c)) for c in corpora],
                           ("Llama shuffled", lambda t: tree_llm_code(t, root, TREE_LAYER)[rng.permutation(15)])]:
            rows = [[coarse_to_fine(code(t), t, k) for k in (1, 3, 7)] + [orthogonality(code(t), t)] for t in trees[root]]
            out[name] = np.mean(rows, 0)
        return out


    print(f"Llama layer {TREE_LAYER} (best mean RSA over trees and corpora)")
    for _r in trees:
        _floor = np.mean([[cka_null(tree_llm_code(t, _r, TREE_LAYER)), cka_shuffle_null(tree_llm_code(t, _r, TREE_LAYER))] for t in trees[_r]], 0)
        print(f"{_r:10s} corpus code vs Llama, RSA/CKA " + "  ".join(f"{c} {v:.2f}/{w:.2f}" for c, v, w in zip(corpora, tree_rsa[_r][TREE_LAYER], tree_cka[_r][TREE_LAYER]))
              + f"   CKA floor random/shuffled {_floor[0]:.2f}/{_floor[1]:.2f}")
        for _name, _m in tree_measures(_r).items():
            print(f"{'':10s} {_name:16s} coarse-to-fine k=1 {_m[0]:.2f}  k=3 {_m[1]:.2f}  k=7 {_m[2]:.2f}   parent-child |cos| {_m[3]:.2f}")
    return TREE_LAYER, TREE_LAYERS, tree_cka, tree_code, tree_llm_code, tree_nodes, tree_rsa


@app.cell
def _(FIGURES, TREE_LAYER, TREE_LAYERS, corpora, plot_utils, plt, tag_w, tree_cka, tree_rsa, trees):
    def tree_rsa_by_layer():
        fig, axes = plt.subplots(1, len(trees), figsize=(4 * len(trees), 3.2), sharey=True)
        for ax, root in zip(axes, trees):
            for j, c in enumerate(corpora):
                line, = ax.plot(TREE_LAYERS, tree_rsa[root][:, j], label=c)
                ax.plot(TREE_LAYERS, tree_cka[root][:, j], ls="--", lw=0.9, color=line.get_color())
            ax.axvline(TREE_LAYER, color="0.8", lw=0.8); ax.axhline(0, color="0.7", lw=0.8)
            ax.set_title(root); ax.set_xlabel("layer"); ax.set_ylim(-0.3, 1)
        axes[0].set_ylabel("corpus tree code vs Llama\nRSA (solid), CKA (dashed)"); axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"{len(next(iter(trees.values())))} depth-3 WordNet subtrees per root, mean over trees; {tag_w} window", y=1.02)
        return plot_utils.save_figure(fig, FIGURES / f"tree_llm_rsa_{tag_w}.png")


    tree_rsa_by_layer()
    return


@app.cell
def _(mo, trees):
    view_root = mo.ui.dropdown(options=list(trees), value="organism", label="root")
    view_tree = mo.ui.dropdown(options=[str(i) for i in range(len(trees["organism"]))], value="0", label="tree")
    mo.hstack([view_root, view_tree], justify="start")
    return view_root, view_tree


@app.cell
def _(FIGURES, ListedColormap, TREE_LAYER, cka, corpora, mo, plot_utils, rsa, tag_w, tree_code, tree_llm_code, trees, view_root, view_tree):
    LEAF_COLORS = ["#1f4e9c"] * 2 + ["#7fb2e5"] * 2 + ["#b22222"] * 2 + ["#f4a582"] * 2   # leaf pairs; hue = side of the root


    def tree_panel(root, i):
        """Llama's tree code beside each corpus's, as ultrametric dendrograms of the 8 leaves (coloured by their
        parent, hue by the root's side), Grams below."""
        tree = trees[root][i]
        parents = [-1] + [(u - 1) // 2 for u in range(1, 15)]                       # level order: children of u at 2u+1, 2u+2
        spec = {"structure": "tree", "parents": parents, "view": "ultrametric"}
        E = tree_llm_code(tree, root, TREE_LAYER)
        codes_t = {c: tree_code(tree, c) for c in corpora}
        fig = plot_utils.plot_embedding_panels(E, codes_t, spec, cmap=ListedColormap(["0.5"] * 7 + LEAF_COLORS),
                                               scores={c: {"RSA": rsa(E, v), "CKA": cka(E, v)} for c, v in codes_t.items()},
                                               first_name=f"Llama L{TREE_LAYER}", starred="rest")
        fig.suptitle(f"{root} tree {i}: " + " / ".join(", ".join(level) for level in tree), y=1.04, fontsize=9)
        return plot_utils.save_figure(fig, FIGURES / f"tree_{root}_{tag_w}.png")


    mo.vstack([mo.md("  \n".join(f"level {l}: " + ", ".join(level) for l, level in enumerate(trees[view_root.value][int(view_tree.value)]))),
               tree_panel(view_root.value, int(view_tree.value))])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Analogies: the Google set

    Korchinski et al.'s test as a corpus vs Llama comparison: the 14 families of Mikolov's questions-words set,
    kept to the questions whose four words every corpus has. Corpus code = block code of those words at rank K
    (the leading K columns of the balanced factor); Llama = the same words after the neutral prefixes, per
    layer. Score = top-1 3CosAdd accuracy per family (nearest analogy word by cosine to b − a + c, excluding
    a, b, c), and RSA between the block code and Llama over the same words.
    """)
    return


@app.cell
def _(CACHE, corpora, read_analogies):
    _q = read_analogies(CACHE / "corpus/analogies/questions-words.txt")
    _shared = set.intersection(*[set(v[1].tolist()) for v in corpora.values()])
    analogy_form = {w.lower(): w for qs in _q.values() for t in qs for w in t}       # lowercase -> the file's form, for Llama
    analogies = {f: [tuple(w.lower() for w in t) for t in qs if all(w.lower() in _shared for w in t)] for f, qs in _q.items()}
    analogy_words = sorted({w for qs in analogies.values() for t in qs for w in t})
    for _f, _qs in analogies.items():
        print(f"{_f:28s} {len(_qs):5d} of {len(_q[_f]):5d} questions")
    print(len(analogy_words), "analogy words")
    return analogies, analogy_form, analogy_words


@app.cell
def _(analogy_form, analogy_words, corpora, index, log_conditional, neutral_reps, np, predicted_geometry):
    analogy_index = {w: i for i, w in enumerate(analogy_words)}


    def analogy_code(corpus):
        """Full-rank balanced factor of the analogy words' block; rank K = its first K columns."""
        C, _, _, r = corpora[corpus]
        ids = [index[corpus][w] for w in analogy_words]
        return predicted_geometry(log_conditional(C[np.ix_(ids, ids)], r[ids]))


    analogy_codes = {c: analogy_code(c) for c in corpora}
    _reps = neutral_reps("analogies", [analogy_form[w] for w in analogy_words])
    analogy_llm = np.stack([_reps[analogy_form[w]] for w in analogy_words], 1)         # (layers, words, d)
    return analogy_codes, analogy_index, analogy_llm


@app.cell
def _(analogies, analogy_accuracy, analogy_codes, analogy_index, analogy_llm, cka, cka_null, cka_shuffle_null, corpora, np, rsa):
    RANKS = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    acc_corpus = {c: np.array([[analogy_accuracy(E[:, :K], analogy_index, qs) for qs in analogies.values()] for K in RANKS])
                  for c, E in analogy_codes.items()}                                              # corpus -> (ranks, families)
    acc_llm = np.array([[analogy_accuracy(analogy_llm[l], analogy_index, qs) for qs in analogies.values()]
                        for l in range(analogy_llm.shape[0])])                                     # (layers, families)
    analogy_rsa = np.array([[rsa(E, analogy_llm[l]) for E in analogy_codes.values()] for l in range(analogy_llm.shape[0])])
    analogy_cka = np.array([[cka(E, analogy_llm[l]) for E in analogy_codes.values()] for l in range(analogy_llm.shape[0])])
    ANALOGY_LAYER = int(acc_llm.mean(1).argmax())
    BEST_RANK = {c: RANKS[int(a.mean(1).argmax())] for c, a in acc_corpus.items()}
    print(f"Llama best layer {ANALOGY_LAYER} (mean accuracy {acc_llm[ANALOGY_LAYER].mean():.2f}); best ranks {BEST_RANK}")
    print(f"{'family':28s} {'Llama':>6s}  " + "  ".join(f"{c:>10s}" for c in corpora) + "   (corpus at its best rank)")
    for _j, _f in enumerate(analogies):
        print(f"{_f:28s} {acc_llm[ANALOGY_LAYER, _j]:6.2f}  " + "  ".join(f"{acc_corpus[c][RANKS.index(BEST_RANK[c]), _j]:10.2f}" for c in corpora))
    print(f"{'mean':28s} {acc_llm[ANALOGY_LAYER].mean():6.2f}  " + "  ".join(f"{acc_corpus[c][RANKS.index(BEST_RANK[c])].mean():10.2f}" for c in corpora))
    print("block code vs Llama at the best layer, RSA/CKA: " + ", ".join(f"{c} {v:.2f}/{w:.2f}" for c, v, w in zip(corpora, analogy_rsa[ANALOGY_LAYER], analogy_cka[ANALOGY_LAYER]))
          + f"; CKA floor random/shuffled {cka_null(analogy_llm[ANALOGY_LAYER], repeats=5):.2f}/{cka_shuffle_null(analogy_llm[ANALOGY_LAYER], repeats=5):.2f}")
    return ANALOGY_LAYER, BEST_RANK, RANKS, acc_corpus, acc_llm, analogy_cka, analogy_rsa


@app.cell
def _(ANALOGY_LAYER, FIGURES, RANKS, acc_corpus, acc_llm, analogy_cka, analogy_rsa, corpora, plot_utils, plt, tag_w):
    def analogy_figure():
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.3))
        for c, a in acc_corpus.items():
            axes[0].plot(RANKS, a.mean(1), marker="o", ms=3, label=c)
            axes[0].fill_between(RANKS, a.mean(1) - a.std(1), a.mean(1) + a.std(1), alpha=0.12)
        axes[0].axhline(acc_llm[ANALOGY_LAYER].mean(), color="0.3", ls="--", lw=0.8, label=f"Llama L{ANALOGY_LAYER}")
        axes[0].set_xscale("log", base=2); axes[0].set_xlabel("rank K of the block code"); axes[0].set_ylabel("3CosAdd accuracy")
        axes[0].set_title("corpus codes"); axes[0].legend(frameon=False, fontsize=8, loc="lower right")
        layers = range(acc_llm.shape[0])
        axes[1].plot(layers, acc_llm.mean(1), color="0.3")
        axes[1].fill_between(layers, acc_llm.mean(1) - acc_llm.std(1), acc_llm.mean(1) + acc_llm.std(1), color="0.3", alpha=0.12)
        axes[1].set_xlabel("layer"); axes[1].set_title("Llama-3.1-8B")
        for j, c in enumerate(corpora):
            line, = axes[2].plot(layers, analogy_rsa[:, j], label=c)
            axes[2].plot(layers, analogy_cka[:, j], ls="--", lw=0.9, color=line.get_color())
        axes[2].axhline(0, color="0.7", lw=0.8); axes[2].set_xlabel("layer"); axes[2].set_ylabel("RSA (solid), CKA (dashed)")
        axes[2].set_title("block code vs Llama, all analogy words")
        for ax in axes[:2]:
            ax.set_ylim(0, 1)
        fig.suptitle(f"Google analogies, mean over 14 families (band: sd across families); {tag_w} window", y=1.03)
        return plot_utils.save_figure(fig, FIGURES / f"analogies_{tag_w}.png")


    analogy_figure()
    return


@app.cell
def _(ANALOGY_LAYER, BEST_RANK, FIGURES, RANKS, acc_corpus, acc_llm, analogies, corpora, np, plot_utils, plt, tag_w):
    def analogy_families():
        """Accuracy per family: Llama at its best layer beside each corpus at its best rank."""
        fig, ax = plt.subplots(figsize=(12, 3.4))
        x = np.arange(len(analogies)); w = 0.8 / (len(corpora) + 1)
        ax.bar(x, acc_llm[ANALOGY_LAYER], w, color="0.3", label=f"Llama L{ANALOGY_LAYER}")
        for j, c in enumerate(corpora, 1):
            ax.bar(x + j * w, acc_corpus[c][RANKS.index(BEST_RANK[c])], w, label=f"{c} K={BEST_RANK[c]}")
        ax.set_xticks(x + 0.4 - w / 2); ax.set_xticklabels(list(analogies), rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("3CosAdd accuracy"); ax.set_ylim(0, 1)
        ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
        return plot_utils.save_figure(fig, FIGURES / f"analogy_families_{tag_w}.png")


    analogy_families()
    return


if __name__ == "__main__":
    app.run()
