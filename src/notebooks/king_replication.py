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
    # Replicating the Polar Probe on semantic structures (Diego-Simon et al., COLM 2026)

    Their Euclidean domains first: **variable ordinality** (letters on a line, greater / less than) and
    **spatial layout** (objects on a square grid, left / right and on top / below). Graphs by their
    Algorithm 1; their headers, entity names and relations (Table 1); every relation once, in random order and
    phrasing; then "Who are a, b, and c?" in vocabulary order, read at those entity tokens.
    30 graphs x 20 descriptions to train, 50 x 30 to test; Llama-3.1-8B and the same architecture with random
    weights. Probe per layer as theirs: rank 512, structural (1 - soft Spearman) + 5 x angular loss, Adam
    lr 1e-5, 100 epochs. Not stated in the paper, chosen here: 7 entities (ordinality) and 5 (spatial), the
    exact fact sentences and list wording, a sigmoid soft rank, and standardised features.
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
    from src.code.relations import KING, graph_distances, incidence, king_description, king_graph, read_listed
    from src.code.polar_probe import evaluate, train_probes
    from src.code.plot_utils import save_figure

    CACHE = Path(os.environ.get("CACHE_DIR", ROOT / "outputs"))
    FIGURES = CACHE / "figures"; FIGURES.mkdir(parents=True, exist_ok=True)
    return (
        CACHE,
        FIGURES,
        KING,
        evaluate,
        graph_distances,
        incidence,
        king_description,
        king_graph,
        np,
        plt,
        read_listed,
        save_figure,
        torch,
        train_probes,
    )


@app.cell
def _(KING, graph_distances, incidence, king_description, king_graph, np):
    import os as _os
    N_ENTITIES = {"ordinality": 7, "spatial": 5}
    _train = tuple(int(x) for x in _os.environ.get("KING_TRAIN", "30x20").split("x"))   # their 30 x 20 by default
    SPLITS = {"train": _train, "test": (50, 30)}                       # (graphs, descriptions per graph)
    SEED = 0


    def make_split(domain, split):
        """[(text, listing)] and the matching [(D, M)] per description."""
        G, K = SPLITS[split]
        rng = np.random.default_rng([SEED, list(N_ENTITIES).index(domain), list(SPLITS).index(split)])
        prompts, gold = [], []
        for _ in range(G):
            names, _, edges = king_graph(domain, N_ENTITIES[domain], rng)
            n = len(names); D, M = graph_distances(n, edges), incidence(n, edges, KING[domain]["dims"])
            for _ in range(K):
                prompts.append(king_description(domain, names, edges, rng)); gold.append((D, M))
        return prompts, gold


    data = {(d, s): make_split(d, s) for d in N_ENTITIES for s in SPLITS}
    print(data["ordinality", "train"][0][0][0]); print(); print(data["spatial", "train"][0][0][0])
    return N_ENTITIES, SEED, SPLITS, data


@app.cell
def _(CACHE, N_ENTITIES, SEED, SPLITS, data, np, read_listed, torch):
    LLM_PATH, LLM_NAME = "meta-llama/Llama-3.1-8B", "llama-3.1-8b"
    MODELS = ("trained", "random")
    READ_DIR = CACHE / "king" / LLM_NAME; READ_DIR.mkdir(parents=True, exist_ok=True)
    _path = lambda m, d, s: READ_DIR / f"{m}_{d}_{s}_n{N_ENTITIES[d]}_g{SPLITS[s][0]}x{SPLITS[s][1]}_s{SEED}.npy"
    for _m in MODELS:
        _missing = [k for k in data if not _path(_m, *k).exists()]
        if _missing:                                                   # the expensive step, cached per model, domain, split
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
            _tok = AutoTokenizer.from_pretrained(LLM_PATH)
            if _m == "trained":
                _model = AutoModelForCausalLM.from_pretrained(LLM_PATH, dtype=torch.bfloat16, device_map="cuda")
            else:
                torch.manual_seed(SEED)
                _model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(LLM_PATH), dtype=torch.bfloat16).cuda()
            _model.eval()
            for _k in _missing:
                np.save(_path(_m, *_k), read_listed(_model, _tok, data[_k][0]))
            del _model; torch.cuda.empty_cache()
    reads = {(m, *k): np.load(_path(m, *k), mmap_mode="r") for m in MODELS for k in data}     # (layers, prompts, n, d)
    print({k: v.shape for k, v in reads.items() if k[0] == "trained"})
    return MODELS, reads


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Existence and type by layer

    Test-set Spearman per description, averaged: **existence**, probe distances against graph distances over
    pairs; **type**, the probed incidence (cosine of each linked pair's difference with each prototype)
    against the gold incidence over all ordered pairs. Dotted: the type score of the exact code (true
    lattice positions read with the true axes), the ceiling, below 1 because unlinked pairs (diagonal
    neighbours, two steps apart) have non-zero cosines with the prototypes where the gold incidence is 0.
    """)
    return


@app.cell
def _(KING, MODELS, N_ENTITIES, data, evaluate, np, reads, train_probes):
    import json
    from src.code.relations import grid_graph


    def layer_scores(model, domain):
        """Probes for every layer trained together, then scored on the test set: (layers, 2) existence, type."""
        tr, te = reads[model, domain, "train"], reads[model, domain, "test"]
        D, M = (np.stack([g[k] for g in data[domain, "train"][1]]) for k in range(2))
        probes = train_probes(np.asarray(tr, np.float32), D, M, rank=512, epochs=100, lr=1e-5, lam=5.0, seed=0)
        return np.array([evaluate(p, [(np.asarray(te[l, i], np.float32), *g) for i, g in enumerate(data[domain, "test"][1])])
                         for l, p in enumerate(probes)])


    def ceiling(domain):
        """The exact code's scores: true lattice positions, true axes as prototypes."""
        rng = np.random.default_rng(1); n, d = N_ENTITIES[domain], KING[domain]["dims"]
        from src.code.relations import graph_distances, incidence
        S = []
        for _ in range(300):
            pos, e = grid_graph(n, d, rng); S.append((pos.astype(float), graph_distances(n, e), incidence(n, e, d)))
        return evaluate({"B": np.eye(d), "P": np.eye(d), "mu": np.zeros(d), "sd": np.ones(d)}, S)


    scores = {(m, d): layer_scores(m, d) for m in MODELS for d in N_ENTITIES}
    ceilings = {d: ceiling(d) for d in N_ENTITIES}
    for (_m, _d), _s in scores.items():
        _b = int(_s[:, 0].argmax())
        print(f"{_m:8s} {_d:10s} best layer {_b:2d}: existence {_s[_b, 0]:.2f}, type {_s[_b, 1]:.2f}  (exact-code type ceiling {ceilings[_d][1]:.2f})")
    return ceilings, scores


@app.cell
def _(FIGURES, N_ENTITIES, SPLITS, ceilings, np, plt, save_figure, scores):
    _fig, _axes = plt.subplots(1, len(N_ENTITIES), figsize=(10, 3.4), sharey=True)
    for _ax, _d in zip(_axes, N_ENTITIES):
        for _k, _c in ((0, "C0"), (1, "C1")):
            for _m, _ls in (("trained", "-"), ("random", "--")):
                _ax.plot(scores[_m, _d][:, _k], _ls, color=_c, label=f"{['existence', 'type'][_k]} ({_m} LLM)")
        _ax.axhline(ceilings[_d][1], color="C1", ls=":", lw=1, label="type, exact code")
        _ax.set_title(_d); _ax.set_xlabel("layer"); _ax.set_ylim(-0.1, 1)
    _axes[0].set_ylabel("Spearman (test)"); _axes[0].legend(frameon=False, fontsize=7)
    save_figure(_fig, FIGURES / f"king_replication_by_layer_train{SPLITS['train'][0]}x{SPLITS['train'][1]}.png")
    return


if __name__ == "__main__":
    app.run()
