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
    # Replicating the syntactic Polar Probe (Diego-Simon et al., NeurIPS 2024)

    English Web Treebank (UD English-EWT r2.14), sentences with an e-mail or web address dropped, relation
    subtypes merged (obl:tmod -> obl). Llama-3.1-8B reads each sentence alone; a word's code is its residual
    averaged over its sub-tokens. Their probe: one linear map to 128 dimensions, Hewitt & Manning's structural
    loss (tree distance against squared probe distance, L1) plus 10 x an angular loss (probed edges
    head - dependent of one type collinear, of different types orthogonal); Adam lr 5e-3, batches of 200
    sentences, 30 epochs. Scored on the test split without punctuation: **UUAS** (gold edges in the minimum
    spanning tree of squared probe distances), **type** and **direction** accuracy on gold edges (nearest
    per-type prototype by |cosine|, head by its sign), and **LAS** (all three right). Their Llama-2-7B reaches
    LAS 62.9 at layer 16.
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
    from src.code.treebank import parse_conllu, read_words
    from src.code.syntax_probe import evaluate, prototypes, train_probe
    from src.code.plot_utils import save_figure

    CACHE = Path(os.environ.get("CACHE_DIR", ROOT / "outputs"))
    FIGURES = CACHE / "figures"; FIGURES.mkdir(parents=True, exist_ok=True)
    return CACHE, FIGURES, evaluate, np, parse_conllu, plt, prototypes, read_words, save_figure, torch, train_probe


@app.cell
def _(CACHE, parse_conllu):
    from collections import Counter

    UD = CACHE / "ud_ewt"
    splits = {s: parse_conllu(UD / f"en_ewt-ud-{s}.conllu") for s in ("train", "dev", "test")}
    _counts = Counter(r for s in splits["train"] for r in s["rels"] if r != "root")
    TYPES = {r: i for i, (r, _) in enumerate(_counts.most_common())}           # every relation seen in training
    print({k: len(v) for k, v in splits.items()}, f"{len(TYPES)} relation types; most common:", list(TYPES)[:8])
    return TYPES, splits


@app.cell
def _(CACHE, np, read_words, splits, torch):
    LLM_PATH, LLM_NAME = "meta-llama/Llama-3.1-8B", "llama-3.1-8b"
    LAYERS = [0, 4, 8, 12, 14, 16, 18, 20, 24, 28, 32]
    READ_DIR = CACHE / "syntax" / LLM_NAME; READ_DIR.mkdir(parents=True, exist_ok=True)
    _path = lambda s: READ_DIR / f"{s}_layers{'-'.join(map(str, LAYERS))}.npy"
    _missing = [s for s in ("train", "test") if not _path(s).exists()]
    if _missing:                                                     # the expensive step, cached per split
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(LLM_PATH)
        _model = AutoModelForCausalLM.from_pretrained(LLM_PATH, dtype=torch.bfloat16, device_map="cuda").eval()
        for _s in _missing:
            np.save(_path(_s), read_words(_model, _tok, splits[_s], LAYERS))
        del _model; torch.cuda.empty_cache()
    words = {s: np.load(_path(s), mmap_mode="r") for s in ("train", "test")}    # (layers, words, d)
    print({s: w.shape for s, w in words.items()})
    return LAYERS, words


@app.cell
def _(LAYERS, TYPES, evaluate, np, prototypes, splits, train_probe, words):
    results, probes = [], []
    for _k, _l in enumerate(LAYERS):
        _Xtr = np.asarray(words["train"][_k], dtype=np.float32)
        _probe = train_probe(_Xtr, splits["train"], TYPES, rank=128, epochs=30, lr=5e-3, lam=10.0, batch_size=200, seed=0)
        _P = prototypes(_probe, _Xtr, splits["train"])
        _r = evaluate(_probe, _P, np.asarray(words["test"][_k], dtype=np.float32), splits["test"])
        results.append(_r); probes.append((_probe, _P))
        print(f"layer {_l:2d}: " + ", ".join(f"{k} {v:.3f}" for k, v in _r.items()), flush=True)
    return probes, results


@app.cell
def _(FIGURES, LAYERS, plt, results, save_figure):
    _fig, _ax = plt.subplots(figsize=(5.5, 3.4))
    for _k in ("uuas", "type", "direction", "las"):
        _ax.plot(LAYERS, [r[_k] for r in results], marker="o", label=_k.upper() if _k in ("uuas", "las") else _k)
    _ax.set_xlabel("layer"); _ax.set_ylabel("test score"); _ax.set_ylim(0, 1); _ax.legend(frameon=False, fontsize=8)
    _ax.set_title("Polar Probe on EWT, Llama-3.1-8B")
    save_figure(_fig, FIGURES / "syntax_replication_by_layer.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Probe space at the best layer

    Their Figure 2: test edges (head minus dependent, in probe space) of the three commonest types projected on
    their first two PCs, one arrow per edge from the origin; and the cosines between all type prototypes, the
    commonest first.
    """)
    return


@app.cell
def _(FIGURES, LAYERS, TYPES, np, plt, probes, results, save_figure, splits, words):
    from src.code.syntax_probe import sentence_index

    _b = int(np.argmax([r["las"] for r in results])); _probe, _P = probes[_b]
    _Z = ((np.asarray(words["test"][_b], dtype=np.float32) - _probe["mu"]) / _probe["sd"]) @ _probe["B"].T
    _show = ["nsubj", "obj", "det"]
    _edges = [(_Z[st + h] - _Z[st + j], r) for st, s in zip(sentence_index(splits["test"]), splits["test"])
              for j, (h, r) in enumerate(zip(s["heads"], s["rels"])) if h >= 0 and r in _show]
    _V = np.array([v for v, _ in _edges]); _lab = np.array([r for _, r in _edges])
    _U, _S, _Wt = np.linalg.svd(_V - _V.mean(0), full_matrices=False); _pc = _V @ _Wt[:2].T
    _fig, (_ax, _axg) = plt.subplots(1, 2, figsize=(11, 4.6))
    for _r, _c in zip(_show, ("C0", "C1", "C2")):
        _sel = np.flatnonzero(_lab == _r)[:300]
        _ax.scatter(_pc[_sel, 0], _pc[_sel, 1], s=6, color=_c, alpha=0.5, label=_r)
    _ax.axhline(0, color="0.8", lw=0.6); _ax.axvline(0, color="0.8", lw=0.6)
    _ax.set_title(f"layer {LAYERS[_b]}: probed edges (head - dependent), PCs 1-2"); _ax.legend(frameon=False, markerscale=3)
    _top = list(TYPES)[:12]; _ix = [TYPES[r] for r in _top]
    _im = _axg.imshow(_P[_ix] @ _P[_ix].T, cmap="RdBu_r", vmin=-1, vmax=1)
    _axg.set_xticks(range(len(_top)), _top, rotation=90, fontsize=8); _axg.set_yticks(range(len(_top)), _top, fontsize=8)
    _axg.set_title("cosines between type prototypes"); _fig.colorbar(_im, ax=_axg, shrink=0.8)
    _fig.tight_layout()
    save_figure(_fig, FIGURES / "syntax_probe_space.png")
    return


if __name__ == "__main__":
    app.run()
