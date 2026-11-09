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
    # Relational structures described in text, read by Llama-3.1-8B

    King et al.'s format: an intro line, every adjacent fact once in random order and direction, then
    "Who are ...?" listing the entities in a fixed order; an entity's representation is the residual at its
    token in that list, averaged over a structure's descriptions, and indexed by its position along the
    structure.

    ```
    I am going to describe how some variables compare.
    d is greater than b. a is less than b. e is greater than d. c is less than a.
    Who are c, a, b, d, e?
    ```

    Two predictions per domain, each the balanced factor of a table the structure fixes. **Answers**: yes / no
    questions the structure settles (chain: is i greater than j; loop: is j within 2 stops after i), +1 / -1.
    **Next**: which entity comes right after i, a single 1 per row (a simplex).
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
    from src.code.relations import comparison_table, describe, next_table, read_entities, sample_order, within_table
    from src.code.geometry import predicted_geometry
    from src.code.analysis_utils import cka
    from src.code.plot_utils import save_figure

    CACHE = Path(os.environ.get("CACHE_DIR", ROOT / "outputs"))
    FIGURES = CACHE / "figures"; FIGURES.mkdir(parents=True, exist_ok=True)
    return (
        CACHE,
        FIGURES,

        comparison_table,
        describe,
        next_table,
        np,
        plt,
        predicted_geometry,
        read_entities,
        sample_order,
        save_figure,
        torch,
        within_table,
    )


@app.cell
def _(describe, np, sample_order):
    N, GRAPHS, DESCRIPTIONS, SEED = 7, 40, 20, 0          # entities per structure, structures, descriptions each
    LLM_PATH, LLM_NAME = "meta-llama/Llama-3.1-8B", "llama-3.1-8b"


    def make_graphs(domain):
        """[(order, [(text, listing), ...])]: each structure's list order is drawn once and kept for all its descriptions."""
        rng = np.random.default_rng([SEED, ["chain", "cycle"].index(domain)])
        out = []
        for _ in range(GRAPHS):
            order = sample_order(domain, N, rng)
            listing = [order[i] for i in rng.permutation(N)]
            out.append((order, [(describe(domain, order, listing, rng), listing) for _ in range(DESCRIPTIONS)]))
        return out


    graphs = {d: make_graphs(d) for d in ("chain", "cycle")}
    print(graphs["chain"][0][1][0][0])
    return DESCRIPTIONS, GRAPHS, LLM_NAME, LLM_PATH, N, SEED, graphs


@app.cell
def _(
    CACHE,
    DESCRIPTIONS,
    GRAPHS,
    LLM_NAME,
    LLM_PATH,
    N,
    SEED,
    graphs,
    np,
    read_entities,
    torch,
):
    READ_DIR = CACHE / "relations" / LLM_NAME; READ_DIR.mkdir(parents=True, exist_ok=True)
    _path = lambda d: READ_DIR / f"{d}_n{N}_g{GRAPHS}_d{DESCRIPTIONS}_s{SEED}.npy"
    _missing = [d for d in graphs if not _path(d).exists()]
    if _missing:                                                     # the expensive step, cached per domain
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(LLM_PATH)
        _model = AutoModelForCausalLM.from_pretrained(LLM_PATH, dtype=torch.bfloat16, device_map="cuda").eval()
        for _d in _missing:
            np.save(_path(_d), read_entities(_model, _tok, graphs[_d]))
        del _model; torch.cuda.empty_cache()
    reads = {d: np.load(_path(d)) for d in graphs}                  # (layers, graphs, N, d) by position
    print({d: r.shape for d, r in reads.items()})
    return (reads,)


@app.cell
def _(N, comparison_table, next_table, predicted_geometry, within_table):
    PREDICTIONS = {
        "chain": {"answers": predicted_geometry(comparison_table(N)), "next": predicted_geometry(next_table(N, cyclic=False))},
        "cycle": {"answers": predicted_geometry(within_table(N, 2)), "next": predicted_geometry(next_table(N, cyclic=True))},
    }
    return (PREDICTIONS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Agreement by layer

    Raw codes are dominated by each name's own token and by its slot in the "Who are" list, both random with
    respect to position, so both are removed first: per layer, the mean code of every name and of every list
    slot over all structures is subtracted (alternating, to convergence). The remaining codes give a centred
    Gram per structure, averaged over structures; score = RSA (Spearman correlation of pairwise distances)
    with the "answers" prediction. Gray: 95% band with the positions shuffled. The chain is scored on its
    interior positions only: its two ends appear in one fact, the rest in two, which on its own sets
    distances. The "next" prediction is a simplex, the same under any shuffle, so RSA cannot score it.
    """)
    return


@app.cell
def _(FIGURES, N, PREDICTIONS, graphs, np, plt, reads, save_figure):
    from scipy.stats import spearmanr


    def residual(H, gs):
        """``H`` (graphs, N, d) by position with each name's and each list slot's mean code removed."""
        names = {w: i for i, w in enumerate(sorted({w for o, _ in gs for w in o}))}
        labels = [np.array([[names[w] for w in o] for o, _ in gs]),
                  np.array([[descs[0][1].index(w) for w in o] for o, descs in gs])]
        R = H - H.mean((0, 1))
        for _ in range(5):
            for lab in labels:
                M = np.zeros((lab.max() + 1, R.shape[-1])); np.add.at(M, lab.ravel(), R.reshape(-1, R.shape[-1]))
                R = R - (M / np.bincount(lab.ravel())[:, None])[lab]
        return R


    def mean_gram(R, keep):
        X = R[:, keep] - R[:, keep].mean(1, keepdims=True)
        return np.einsum("gid,gjd->ij", X, X) / len(X)


    def distances(G):
        return np.diag(G)[:, None] + np.diag(G)[None] - 2 * G


    def rsa_gram(G, P):
        iu = np.triu_indices(len(G), 1)
        return spearmanr(distances(G)[iu], distances(P)[iu])[0]


    KEEP = {"chain": list(range(1, N - 1)), "cycle": list(range(N))}
    residuals = {d: np.stack([residual(H, graphs[d]) for H in reads[d]]) for d in reads}     # (layers, graphs, N, d)


    def layer_scores(domain, shuffles=100):
        keep = KEEP[domain]; E = PREDICTIONS[domain]["answers"]; E = E - E.mean(0); P = (E @ E.T)[np.ix_(keep, keep)]
        rng = np.random.default_rng(0)
        s, band = [], []
        for R in residuals[domain]:
            s.append(rsa_gram(mean_gram(R, keep), P))
            band.append(np.percentile([rsa_gram(mean_gram(R[:, rng.permutation(N)], keep), P) for _ in range(shuffles)], [2.5, 97.5]))
        return np.array(s), np.array(band)


    scores = {d: layer_scores(d) for d in reads}
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for _ax, (_d, (_s, _band)) in zip(_axes, scores.items()):
        _L = np.arange(len(_s))
        _ax.fill_between(_L, _band[:, 0], _band[:, 1], color="0.85", label="shuffled positions")
        _ax.plot(_L, _s, label="answers prediction"); _ax.axhline(0, color="0.6", lw=0.8)
        _ax.set_title(f"{_d} (positions {KEEP[_d][0]}-{KEEP[_d][-1]})"); _ax.set_xlabel("layer"); _ax.set_ylim(-1, 1)
        print(f"{_d}: best layer {int(np.nanargmax(_s))}, RSA {np.nanmax(_s):.2f}, shuffled 97.5% {_band[int(np.nanargmax(_s)), 1]:.2f}")
    _axes[0].set_ylabel("RSA with the prediction"); _axes[0].legend(frameon=False, fontsize=8)
    save_figure(_fig, FIGURES / "relations_rsa_by_layer.png")
    return KEEP, mean_gram, residuals, scores


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Gram matrices at the best layer

    Llama's centred Gram over positions after removing names and list slots (mean over structures) beside
    each prediction's; then the mean residual code by position on its top two PCs, positions in order.
    """)
    return


@app.cell
def _(FIGURES, N, PREDICTIONS, mean_gram, np, plt, reads, residuals, save_figure, scores):
    def gram(X):
        X = X - X.mean(0)
        G = X @ X.T
        return G / np.abs(G).max()


    _fig, _axes = plt.subplots(2, 4, figsize=(15, 7.2))
    for _row, _d in zip(_axes, reads):
        _l = int(np.nanargmax(scores[_d][0]))
        _H = residuals[_d][_l]
        _G = mean_gram(_H, list(range(N)))
        _panels = {f"Llama layer {_l}": _G / np.abs(_G).max(),
                   **{f"predicted: {k}": gram(E) for k, E in PREDICTIONS[_d].items()}}
        for _ax, (_t, _G) in zip(_row[:3], _panels.items()):
            _ax.imshow(_G, cmap="RdBu_r", vmin=-1, vmax=1); _ax.set_title(f"{_d}: {_t}", fontsize=10)
            _ax.set_xticks(range(len(_G))); _ax.set_yticks(range(len(_G)))
        _M = _H.mean(0); _M = _M - _M.mean(0)
        _U, _S, _ = np.linalg.svd(_M, full_matrices=False); _P = _U[:, :2] * _S[:2]
        _closed = list(range(len(_P))) + ([0] if _d == "cycle" else [])
        _row[3].plot(_P[_closed, 0], _P[_closed, 1], "-", color="0.7")
        _row[3].scatter(_P[:, 0], _P[:, 1], c=np.arange(len(_P)), cmap="viridis", zorder=3)
        for _i, (_x, _y) in enumerate(_P):
            _row[3].annotate(str(_i), (_x, _y), textcoords="offset points", xytext=(4, 4), fontsize=8)
        _row[3].set_title(f"{_d}: mean code by position, PCs 1-2", fontsize=10); _row[3].set_aspect("equal")
    _fig.tight_layout()
    save_figure(_fig, FIGURES / "relations_grams.png")
    return


if __name__ == "__main__":
    app.run()
