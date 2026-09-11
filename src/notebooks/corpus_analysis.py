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

    For a set of concept words, the corpus conditional $p(w \mid \text{concept})$ over the 2000 most
    frequent context words is counted as windowed co-occurrence (both within 16 tokens) over the
    FineWeb 10B-token sample. The predicted code is the balanced factor
    $E = U\sqrt{S}$ of the row-centred, floored $\log p(w \mid \text{concept})$. The LLM's word
    representation is the residual at the word's last token inside a few neutral templates,
    averaged. Months are the known answer (a calendar circle); colors follow.
    """)
    return


@app.cell
def _(mo):
    import os, sys, json
    from pathlib import Path
    from collections import Counter
    import numpy as np
    import torch
    import matplotlib.pyplot as plt

    ROOT = mo.notebook_dir().parent.parent    # src/notebooks -> <repo>
    os.chdir(ROOT); sys.path.insert(0, str(ROOT))
    from src.code.corpus import window_counts, cooccurrence_table, hub_texts
    from src.code.data import graph_conditional
    from src.code.geometry import predicted_geometry, log_conditional
    from src.code.analysis_utils import cka, rsa, align_representations
    from src.code.llm import word_means
    from src.code import plot_utils

    CACHE = Path(os.environ.get("CACHE_DIR", ROOT / "outputs"))
    FIGURES = CACHE / "figures"; FIGURES.mkdir(parents=True, exist_ok=True)
    return (
        CACHE,
        Counter,
        FIGURES,
        align_representations,
        cka,
        cooccurrence_table,
        graph_conditional,
        hub_texts,
        json,
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
def _():
    CORPUS = ("HuggingFaceFW/fineweb", "sample/10BT/")                # the plain FineWeb 10B-token sample, all shards
    WINDOW, K = 16, 25000                                              # co-occurrence window, context words
    MODEL = "meta-llama/Llama-3.1-8B"
    DOMAINS = {"months": {"words": ["January", "February", "March", "April", "May", "June", "July", "August",
                                    "September", "October", "November", "December"],
                          "spec": {"structure": "ring", "n": 12}},         # the known answer: a calendar circle
               "colors": {"words": ["red", "orange", "yellow", "green", "blue", "purple", "pink",   # hue order first
                                    "brown", "black", "white", "gray"],
                          "spec": None}}
    TEMPLATES = [("It was", "."), ("She thought about", " all day."), ("The topic was", ".")]
    return CORPUS, DOMAINS, K, MODEL, TEMPLATES, WINDOW


@app.cell
def _(CACHE, CORPUS, Counter, DOMAINS, K, WINDOW, cooccurrence_table, hub_texts, json, window_counts):
    # the corpus pass, cached raw: per-concept context counters and the word frequencies of the sample
    _path = CACHE / "corpus" / f"{'_'.join(CORPUS).replace('/', '_')}w{WINDOW}_v2.json"
    _concepts = [w for d in DOMAINS.values() for w in d["words"]]
    if not _path.exists():
        _counts, _freq = window_counts(hub_texts(*CORPUS), _concepts, window=WINDOW, workers=40, chunk=20000)
        _path.parent.mkdir(parents=True, exist_ok=True)
        _path.write_text(json.dumps({"counts": {c: dict(v.most_common(50000)) for c, v in _counts.items()},
                                     "freq": dict(_freq.most_common(100000)), "total": sum(_freq.values())}))
    _raw = json.loads(_path.read_text())
    counts, freq, total = {c: Counter(v) for c, v in _raw["counts"].items()}, Counter(_raw["freq"]), _raw["total"]
    tables = {name: cooccurrence_table(counts, freq, d["words"], K) for name, d in DOMAINS.items()}   # (C, vocab)
    print({c: sum(v.values()) // (2 * WINDOW) for c, v in counts.items()})                          # occurrences per concept
    return counts, freq, tables, total


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Predicted codes

    Two normalisations of the same counts, each row-centred and factored: the floored
    $\log p(w \mid \text{concept})$ (the repo's oracle) and Karkada et al.'s bounded PMI
    $M^\star_{ij} = (P_{ij} - P_i P_j) / \tfrac12 (P_{ij} + P_i P_j)$, which caps the influence of
    rare cells. Below: the three context words each concept is most over-represented with
    (centred log-odds), a check that the counts carry meaning and not just frequency.
    """)
    return


@app.cell
def _(DOMAINS, WINDOW, freq, log_conditional, np, predicted_geometry, tables, total):
    def bounded_pmi(C, vocab, concept_counts):
        """Karkada et al.'s normalised co-occurrence: ``(P_ij - P_i P_j) / (½ (P_ij + P_i P_j))`` with
        ``P_ij`` the pair rate per window position, ``P_i`` the concept rate, ``P_j`` the word rate."""
        N = total
        P_ij = C / (2 * WINDOW * N)
        P_i, P_j = concept_counts[:, None] / N, np.array([freq[w] for w in vocab])[None] / N
        return (P_ij - P_i * P_j) / (0.5 * (P_ij + P_i * P_j))

    logcond = {name: log_conditional(C / C.sum(1, keepdims=True)) for name, (C, _) in tables.items()}
    predicted = {name: {"log-conditional": predicted_geometry(L),
                        "bounded PMI": predicted_geometry(bounded_pmi(*tables[name], tables[name][0].sum(1) / (2 * WINDOW)))}
                 for name, L in logcond.items()}
    for _name, _L in logcond.items():
        _Lc = _L - _L.mean(0)                                                  # over-representation vs the other concepts
        for _w, _row in zip(DOMAINS[_name]["words"], _Lc):
            print(f"{_name:7s} {_w:10s}", ", ".join(tables[_name][1][j] for j in np.argsort(-_row)[:3]))
    return logcond, predicted


@app.cell
def _(CACHE, DOMAINS, MODEL, TEMPLATES, np, torch, word_means):
    # the LLM pass, cached raw: per-layer word representations averaged over the templates
    def _cache(name):
        return CACHE / "corpus" / f"{name}_word_means.npz"

    _todo = [n for n in DOMAINS if not _cache(n).exists()]
    if _todo:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(MODEL)
        _llm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto").eval()
        _templates = [(_tok.encode(pre), _tok.encode(post, add_special_tokens=False)) for pre, post in TEMPLATES]
        for _n in _todo:
            _words = {w: _tok.encode(" " + w, add_special_tokens=False) for w in DOMAINS[_n]["words"]}
            np.savez(_cache(_n), means=word_means(_llm, _words, _templates), words=list(_words))
        del _llm; torch.cuda.empty_cache()
    means = {n: np.load(_cache(n))["means"] for n in DOMAINS}               # (layers, n_words, d)
    return (means,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fit

    CKA and RSA per layer between the LLM word representations and each corpus prediction. The Grams
    here are close to diagonal (each word has contexts of its own), which inflates CKA; RSA
    (Spearman over the off-diagonal distances) scores the structure between words. For months, both
    sides are also scored against an ideal circle (the diffusion ring code on 12 nodes).
    """)
    return


@app.cell
def _(cka, graph_conditional, means, np, predicted, predicted_geometry, rsa):
    ideal = predicted_geometry(graph_conditional({"structure": "ring", "n": 12, "cooccurrence": "diffusion", "gamma": 0.8}))
    METRICS = {"CKA": cka, "RSA": rsa}
    scores = {name: {metric: {norm: np.array([f(m, E) for m in M]) for norm, E in predicted[name].items()}
                     for metric, f in METRICS.items()} for name, M in means.items()}
    ring_scores = {metric: {"llm": np.array([f(m, ideal) for m in means["months"]]),
                            **{norm: f(E, ideal) for norm, E in predicted["months"].items()}}
                   for metric, f in METRICS.items()}

    def best(name):
        """The (normalisation, layer) with the highest RSA to the LLM, the embedding layer excluded."""
        norm = max(predicted[name], key=lambda n: scores[name]["RSA"][n][1:].max())
        return norm, 1 + int(scores[name]["RSA"][norm][1:].argmax())
    return METRICS, best, ideal, ring_scores, scores


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figures

    Curves: CKA and RSA by layer to each corpus prediction (and, for months, the LLM against the
    ideal circle; dotted: each prediction against it). Panels: the better prediction (by RSA)
    beside Llama at its best layer, top two PCs with the LLM's rotated (Procrustes) onto the
    prediction's frame, Grams below. Items are coloured in list order, so a ring in calendar or hue
    order reads as a smooth colour cycle.
    """)
    return


@app.cell
def _(DOMAINS, FIGURES, METRICS, align_representations, best, means, plot_utils, plt, predicted, ring_scores, scores):
    def curves(name):
        norm, layer = best(name)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for ax, metric in zip(axes, METRICS):
            for i, n in enumerate(predicted[name]):
                ax.plot(scores[name][metric][n], color=f"C{i}", label=n)
                if name == "months":
                    ax.axhline(ring_scores[metric][n], ls=":", color=f"C{i}")
            if name == "months":
                ax.plot(ring_scores[metric]["llm"], "--", color="k", label="LLM vs ideal circle")
            ax.set_xlabel("layer"); ax.set_ylabel(metric); ax.set_ylim(0, 1); ax.set_title(f"{name}: {metric}")
        axes[0].legend()
        fig.tight_layout()
        print(f"{name}: best {norm} at layer {layer}: " + ", ".join(
            f"{m} {scores[name][m][norm][layer]:.3f}" for m in METRICS)
            + (" | vs ideal circle: " + ", ".join(f"{m} corpus {ring_scores[m][norm]:.2f} / LLM {ring_scores[m]['llm'][layer]:.2f}"
                                                  for m in METRICS) if name == "months" else ""))
        return plot_utils.save_figure(fig, FIGURES / f"corpus_{name}_curves.png")

    def panels(name):
        norm, layer = best(name)
        label, E, M = f"Llama L{layer}", predicted[name][norm], means[name][layer]
        ref = plot_utils._pca(E, 2)                                            # the prediction's PC frame
        coords = {"predicted": ref, label: align_representations(M, ref, method="rotation", rank=2)[0]}
        fig = plot_utils.plot_embedding_panels(E, {label: M}, DOMAINS[name]["spec"],
                                               scores={label: scores[name]["CKA"][norm][layer]}, coords=coords)
        fig.suptitle(f"{name}: corpus prediction ({norm}), RSA {scores[name]['RSA'][norm][layer]:.2f}", y=1.02)
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


if __name__ == "__main__":
    app.run()
