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
    # Predicting an LLM's word geometry from corpus statistics

    For a set of concept words, the corpus conditional $p(w \mid \text{concept})$ over the 25k most
    frequent context words is counted as windowed co-occurrence (both within 16 tokens) over the
    FineWeb 10B-token sample, with each occurrence anchored to the intended sense (months next to a
    day or year or after a month preposition; colors within reach of a colour cue). Two predicted
    codes per domain: the balanced factor $E = U\sqrt{S}$ of the row-centred, floored
    $\log p(w \mid \text{concept})$, and of Karkada et al.'s bounded PMI
    $(P_{ij} - P_i P_j) / \tfrac12 (P_{ij} + P_i P_j)$. The LLM's representation is the residual at
    the word's last token inside a few neutral templates, averaged. Known-answer references: a
    circle for months, a line / log line / decade helix for numbers 1–999, and for MIDI note numbers
    (a prompt-only probe read at the prompt's last token) a line, the chroma circle and their helix.
    """)
    return


@app.cell
def _(mo):
    import os, sys
    from pathlib import Path
    import numpy as np
    import torch
    import matplotlib.pyplot as plt

    ROOT = mo.notebook_dir().parent.parent    # src/notebooks -> <repo>
    os.chdir(ROOT); sys.path.insert(0, str(ROOT))
    from src.code.corpus import window_counts, cooccurrence_table, hub_texts, date_anchor, NearCue
    from src.code.geometry import predicted_geometry, log_conditional
    from src.code.analysis_utils import cka, rsa, align_representations
    from src.code.llm import word_means
    from src.code import plot_utils

    CACHE = Path(os.environ.get("CACHE_DIR", ROOT / "outputs"))
    FIGURES = CACHE / "figures"; FIGURES.mkdir(parents=True, exist_ok=True)
    return (
        CACHE,
        FIGURES,
        NearCue,
        align_representations,
        cka,
        cooccurrence_table,
        date_anchor,
        hub_texts,
        log_conditional,
        np,
        plot_utils,
        plt,
        predicted_geometry,
        rsa,
        torch,
        window_counts,
        word_means,
    )


@app.cell
def _(NearCue, date_anchor, np):
    CORPUS = ("HuggingFaceFW/fineweb", "sample/10BT/")                # the plain FineWeb 10B-token sample, all shards
    TAG = "_".join(CORPUS).replace("/", "_").strip("_")
    WINDOW, K = 16, 25000                                              # co-occurrence window, context words
    MODEL = "meta-llama/Llama-3.1-8B"
    COLOR_CUES = {"color", "colour", "colors", "colours", "colored", "coloured", "paint", "painted", "shade",
                  "shades", "hue", "dye", "dyed", "pigment"}

    def circle(x, period):
        return np.c_[np.cos(2 * np.pi * x / period), np.sin(2 * np.pi * x / period)]

    _n, _midi = np.arange(1, 1000), np.arange(21, 109)                  # single Llama-3 tokens; the piano's MIDI range
    DOMAINS = {
        "months": {"words": ["January", "February", "March", "April", "May", "June", "July", "August",
                             "September", "October", "November", "December"],
                   "templates": [("It was", "."), ("She thought about", " all day."), ("The topic was", ".")],
                   "anchor": date_anchor, "corpus": True, "at_end": False, "spec": {"structure": "ring", "n": 12},
                   "refs": {"circle": circle(np.arange(12), 12)}},
        "colors": {"words": ["red", "orange", "yellow", "green", "blue", "purple", "pink",     # hue order first
                             "brown", "black", "white", "gray"],
                   "templates": [("It was", "."), ("She thought about", " all day."), ("The topic was", ".")],
                   "anchor": NearCue(COLOR_CUES, WINDOW), "corpus": True, "at_end": False, "spec": None, "refs": {}},
        "numbers": {"words": [str(n) for n in _n],
                    "templates": [("The number is", "."), ("It was", "."), ("The answer is", ".")],
                    "anchor": None, "corpus": True, "at_end": False, "spec": None,
                    "refs": {"line": _n[:, None] / 100, "log line": np.log(_n)[:, None],
                             "log line + last digit": np.c_[np.log(_n), circle(_n, 10)]}},
        "midi": {"words": [str(n) for n in _midi],
                 "templates": [("The note I am playing is MIDI", ".")],
                 "anchor": None, "corpus": False, "at_end": True, "spec": None,
                 "refs": {"line": _midi[:, None] / 12, "chroma circle": circle(_midi, 12),
                          "helix": np.c_[_midi / 12, circle(_midi, 12)]}},
    }
    return CORPUS, DOMAINS, K, MODEL, TAG, WINDOW


@app.cell
def _(CACHE, CORPUS, DOMAINS, K, TAG, WINDOW, cooccurrence_table, hub_texts, np, window_counts):
    # the corpus pass per domain, cached as the count table over the top-K context words
    def table_path(name):
        return CACHE / "corpus" / f"{TAG}_{name}_w{WINDOW}.npz"

    for _name, _d in DOMAINS.items():
        if _d["corpus"] and not table_path(_name).exists():
            _anchors = {w: _d["anchor"] for w in _d["words"]} if _d["anchor"] else None
            _counts, _freq = window_counts(hub_texts(*CORPUS), _d["words"], window=WINDOW, workers=40, chunk=20000,
                                           anchors=_anchors)
            _C, _vocab = cooccurrence_table(_counts, _freq, _d["words"], K)
            np.savez(table_path(_name), C=_C.astype(np.float32), vocab=np.array(_vocab),
                     concept_counts=np.array([sum(_counts[w].values()) / (2 * WINDOW) for w in _d["words"]]),
                     word_counts=np.array([_freq[w] for w in _vocab]), total=sum(_freq.values()))
    tables = {name: dict(np.load(table_path(name))) for name, d in DOMAINS.items() if d["corpus"]}
    print({name: (int(t["concept_counts"].min()), int(t["concept_counts"].max())) for name, t in tables.items()})  # occurrences
    return (tables,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Predicted codes

    Below: the four frequent context words each concept is most over-represented with under the
    bounded PMI (centred across concepts), a check that the counts carry meaning and not frequency.
    """)
    return


@app.cell
def _(DOMAINS, WINDOW, log_conditional, np, predicted_geometry, tables):
    def bounded_pmi(t):
        """Karkada et al.'s normalised co-occurrence: ``(P_ij - P_i P_j) / (½ (P_ij + P_i P_j))`` with
        ``P_ij`` the pair rate per window position, ``P_i`` the concept rate, ``P_j`` the word rate."""
        P_ij = t["C"].astype(np.float64) / (2 * WINDOW * t["total"])
        P_i, P_j = t["concept_counts"][:, None] / t["total"], t["word_counts"][None] / t["total"]
        return (P_ij - P_i * P_j) / (0.5 * (P_ij + P_i * P_j))

    predicted = {name: {"log-conditional": predicted_geometry(log_conditional(t["C"] / t["C"].sum(1, keepdims=True))),
                        "bounded PMI": predicted_geometry(bounded_pmi(t))} for name, t in tables.items()}
    _show = {"months": DOMAINS["months"]["words"], "colors": DOMAINS["colors"]["words"],
             "numbers": ["1", "2", "7", "10", "12", "24", "60", "100", "365", "500", "999"]}
    for _name, _words in _show.items():
        _M = bounded_pmi(tables[_name])[:, :2000]; _Mc = _M - _M.mean(0)
        for _w in _words:
            _row = _Mc[DOMAINS[_name]["words"].index(_w)]
            print(f"{_name:8s} {_w:10s}", ", ".join(tables[_name]["vocab"][j] for j in np.argsort(-_row)[:4]))
    return (predicted,)


@app.cell
def _(CACHE, DOMAINS, MODEL, np, torch, word_means):
    # the LLM pass, cached raw: per-layer word representations averaged over the templates
    def _cache(name):
        return CACHE / "corpus" / f"{name}_word_means.npz"

    _todo = [n for n in DOMAINS if not _cache(n).exists()]
    if _todo:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(MODEL)
        _llm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto").eval()
        for _n in _todo:
            _d = DOMAINS[_n]
            _templates = [(_tok.encode(pre), _tok.encode(post, add_special_tokens=False)) for pre, post in _d["templates"]]
            _words = {w: _tok.encode(" " + w, add_special_tokens=False) for w in _d["words"]}
            np.savez(_cache(_n), means=word_means(_llm, _words, _templates, at_end=_d["at_end"]), words=list(_words))
        del _llm; torch.cuda.empty_cache()
    means = {n: np.load(_cache(n))["means"] for n in DOMAINS}               # (layers, n_words, d)
    return (means,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fit

    CKA and RSA per layer between the LLM word representations and each corpus prediction and each
    known-answer reference. The Grams here are close to diagonal (each word has contexts of its own),
    which inflates CKA; RSA (Spearman over the off-diagonal distances) scores the structure between
    words. Printed: how circular / linear each corpus prediction itself is (RSA to the references).
    """)
    return


@app.cell
def _(DOMAINS, cka, means, np, predicted, rsa):
    METRICS = {"CKA": cka, "RSA": rsa}

    def by_layer(name, E):
        return {metric: np.array([f(m, E) for m in means[name]]) for metric, f in METRICS.items()}

    codes = {name: {**{f"corpus: {k}": E for k, E in predicted.get(name, {}).items()},
                    **{f"ref: {k}": R for k, R in DOMAINS[name]["refs"].items()}} for name in DOMAINS}
    scores = {name: {label: by_layer(name, E) for label, E in cs.items()} for name, cs in codes.items()}
    for _name in predicted:
        for _k, _E in predicted[_name].items():
            for _r, _R in DOMAINS[_name]["refs"].items():
                print(f"{_name}: corpus {_k} vs ref {_r}: RSA {rsa(_E, _R):.2f}, CKA {cka(_E, _R):.2f}")

    def best(name):
        """The corpus prediction (else the reference) with the highest RSA to the LLM, and its layer."""
        pool = [l for l in codes[name] if l.startswith("corpus")] or list(codes[name])
        label = max(pool, key=lambda l: scores[name][l]["RSA"][1:].max())
        return label, 1 + int(scores[name][label]["RSA"][1:].argmax())
    return METRICS, best, codes, scores


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figures

    Curves: CKA and RSA by layer, solid for corpus predictions, dashed for references. Panels: the
    best code (by RSA) beside Llama at its best layer, top two PCs with the LLM's rotated
    (Procrustes) onto the code's frame, Grams below; items are coloured in list order. MIDI: the
    LLM's top PC against note number, the next two PCs coloured by pitch class, and RSA by layer.
    """)
    return


@app.cell
def _(DOMAINS, FIGURES, METRICS, align_representations, best, codes, means, plot_utils, plt, scores):
    def curves(name):
        label, layer = best(name)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for ax, metric in zip(axes, METRICS):
            for i, (l, s) in enumerate(scores[name].items()):
                ax.plot(s[metric], "-" if l.startswith("corpus") else "--", color=f"C{i}", label=l)
            ax.set_xlabel("layer"); ax.set_ylabel(metric); ax.set_ylim(0, 1); ax.set_title(f"{name}: {metric}")
        axes[0].legend(fontsize=9)
        fig.tight_layout()
        print(f"{name}: best {label} at layer {layer}: " + ", ".join(f"{m} {scores[name][label][m][layer]:.3f}" for m in METRICS))
        return plot_utils.save_figure(fig, FIGURES / f"corpus_{name}_curves.png")

    def panels(name):
        label, layer = best(name)
        llm, E, M = f"Llama L{layer}", codes[name][label], means[name][layer]
        ref = plot_utils._pca(E, 2)                                            # the code's own PC frame
        coords = {"predicted": ref, llm: align_representations(M, ref, method="rotation", rank=2)[0]}
        fig = plot_utils.plot_embedding_panels(E, {llm: M}, DOMAINS[name]["spec"],
                                               scores={llm: scores[name][label]["CKA"][layer]}, coords=coords)
        fig.suptitle(f"{name}: {label}, RSA {scores[name][label]['RSA'][layer]:.2f}", y=1.02)
        return plot_utils.save_figure(fig, FIGURES / f"corpus_{name}_panels.png")
    return curves, panels


@app.cell
def _(curves):
    curves("months")
    return


@app.cell
def _(panels):
    panels("months")
    return


@app.cell
def _(curves):
    curves("colors")
    return


@app.cell
def _(panels):
    panels("colors")
    return


@app.cell
def _(curves):
    curves("numbers")
    return


@app.cell
def _(panels):
    panels("numbers")
    return


@app.cell
def _(curves):
    curves("midi")
    return


@app.cell
def _(FIGURES, means, np, plot_utils, plt, scores):
    def midi_figure():
        n = np.arange(21, 109)
        layer = 1 + int(scores["midi"]["ref: helix"]["RSA"][1:].argmax())
        pcs = plot_utils._pca(means["midi"][layer], 3)
        fig, (a, b, c) = plt.subplots(1, 3, figsize=(15, 4.2))
        a.scatter(n, pcs[:, 0], c=n % 12, cmap="twilight", s=25); a.set_xlabel("MIDI note"); a.set_ylabel("PC1")
        a.set_title(f"Llama layer {layer}: height")
        b.scatter(pcs[:, 1], pcs[:, 2], c=n % 12, cmap="twilight", s=25); b.set_xlabel("PC2"); b.set_ylabel("PC3")
        b.set_title("PC2–PC3, coloured by pitch class"); b.set_aspect("equal")
        for l, s in scores["midi"].items():
            c.plot(s["RSA"], "--", label=l.removeprefix("ref: "))
        c.set_xlabel("layer"); c.set_ylabel("RSA"); c.set_ylim(0, 1); c.legend(); c.set_title("MIDI: RSA to references")
        fig.tight_layout()
        return plot_utils.save_figure(fig, FIGURES / "corpus_midi_helix.png")

    midi_figure()
    return


if __name__ == "__main__":
    app.run()
