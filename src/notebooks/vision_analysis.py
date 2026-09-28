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
    # Image geometry of vision classifiers from soft labels

    A classifier's pre-readout features times its readout are its log-conditional over classes, one row per
    image. The theory therefore predicts every image's representation as the balanced factor of an image x class
    table: $E^\star = $ `factor_logcond`$(L)$, $L_{iy} = \log P(y \mid \text{image } i)$. Two tables: the model's
    own logits (row-centred, they are the log-conditional exactly), and human votes per image where a relabelling
    gives them, CIFAR-10H's ~51 annotators per test image (ImageNet-ReaL has 1.15 labels per image, too sparse
    per image). Agreement with the actual features is CKA over all images and RSA over a class-sorted subsample,
    whose Grams show the class blocks.
    """)
    return


@app.cell
def _(mo):
    import json, os, sys
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt

    ROOT = mo.notebook_dir().parent.parent
    os.chdir(ROOT); sys.path.insert(0, str(ROOT))
    from src.code.analysis_utils import cka, rsa
    from src.code.geometry import predicted_geometry
    from src.code import plot_utils

    CACHE = Path(os.environ.get("CACHE_DIR", ROOT / "outputs"))
    VISION = CACHE / "vision"
    FIGURES = CACHE / "figures"; FIGURES.mkdir(parents=True, exist_ok=True)
    return FIGURES, VISION, cka, np, plot_utils, predicted_geometry, rsa


@app.cell
def _(VISION, np):
    def load_run(model, dataset):
        """Cached per-image features, logits and labels of ``model`` on ``dataset``."""
        d = dict(np.load(VISION / model / f"{dataset}.npz"))
        return {"features": d["features"].astype(np.float32), "logits": d["logits"].astype(np.float64),
                "labels": d["labels"], "index": d["index"], "C": d["logits"].shape[1],
                "top1": float((d["logits"].argmax(1) == d["labels"]).mean())}


    def human_votes(dataset, run):
        """``(T, covered)``: per-image human vote table (N, C) and the images it covers. CIFAR-10H: ~51 annotator
        counts per test image. ImageNet-ReaL raters: the "yes" votes of five raters per candidate label, on the
        24,889 validation images that were re-rated."""
        if dataset == "cifar10":
            return np.load(VISION / "labels" / "cifar10h-counts.npy")[run["index"]].astype(np.float64), np.ones(len(run["index"]), bool)
        d = np.load(VISION / "labels" / "raters.npz", allow_pickle=True)
        yes = d["tensor"][:, :, 2].sum(0)                                            # (pairs,) yes votes over raters
        image = np.array([int(f[15:23]) - 1 for f in d["info"][:, 0]])              # ILSVRC2012_val_NNNNNNNN.JPEG -> 0-based
        cand = d["info"][:, 1].astype(int)
        row = np.full(50000, -1); row[run["index"]] = np.arange(len(run["index"]))
        T = np.zeros((len(run["index"]), run["C"]))
        np.add.at(T, (row[image], cand), yes)
        covered = np.zeros(len(run["index"]), bool); covered[row[image]] = True
        return T, covered


    RUNS = {("imagenet", "resnet50"), ("imagenet", "vit_b_16"), ("cifar10", "cifar_resnet56")}
    runs = {k: load_run(k[1], k[0]) for k in RUNS if (VISION / k[1] / f"{k[0]}.npz").exists()}
    votes = {k: human_votes(k[0], r) for k, r in runs.items()}
    for (_ds, _m), _r in runs.items():
        _T, _cov = votes[_ds, _m]
        print(f"{_ds:9s} {_m:14s} top-1 {_r['top1']:.3f}  images {len(_r['labels']):,}  features {_r['features'].shape[1]}  "
              f"classes {_r['C']}  human votes: {int(_cov.sum()):,} images, {_T[_cov].sum(1).mean():.1f} votes/image")
    print("not cached:", sorted(RUNS - set(runs)) or "none")

    return runs, votes


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Predicted codes and agreement

    Per run: the features beside the codes factored from the model's logits and, for CIFAR-10, from the human
    votes per image. CKA over every image, against a floor from shuffling images; RSA over a class-balanced subsample as a
    second number. The panels draw that subsample sorted by class, so the Grams show the class blocks.
    """)
    return


@app.cell
def _(cka, np, predicted_geometry, rsa, runs, votes):
    def log_table(T, floor=1.0):
        """log P(y | x) from an (N, C) vote table, an empty cell counted as ``floor`` votes."""
        T = np.maximum(np.asarray(T, dtype=np.float64), floor)
        return np.log(T / T.sum(1, keepdims=True))


    def subsample(labels, per_class, seed=0):
        """Indices of up to ``per_class`` images of each class, sorted by class so Grams show blocks."""
        rng = np.random.default_rng(seed)
        idx = np.concatenate([rng.permutation(np.where(labels == c)[0])[:per_class] for c in range(labels.max() + 1)])
        return idx[np.argsort(labels[idx], kind="stable")]


    PER_CLASS = {"imagenet": 5, "cifar10": 500}


    def codes_for(key):
        """Per-image codes ``{name: (E, covered)}``: from the model's row-centred logits on every image, and from
        the human votes on the images they cover (rows outside are zero and excluded from every comparison)."""
        run = runs[key]; T, covered = votes[key]
        E_h = np.zeros((len(T), min(covered.sum(), run["C"]))); E_h[covered] = predicted_geometry(log_table(T[covered]))
        return {"model logits": (predicted_geometry(run["logits"]), np.ones(len(T), bool)), "human votes": (E_h, covered)}


    def scores_for(key, codes_k, seed=0):
        run = runs[key]; H = run["features"]
        valid = np.all([m for _, m in codes_k.values()], 0)                          # images every code covers
        sub = np.where(valid)[0][subsample(run["labels"][valid], PER_CLASS[key[0]], seed)]
        rng = np.random.default_rng(seed)
        out = {}
        for name, (E, m) in codes_k.items():
            out[name] = {"CKA": cka(H[m], E[m]), "CKA floor": float(np.mean([cka(H[m], E[m][rng.permutation(int(m.sum()))]) for _ in range(5)])),
                         "RSA (subsample)": rsa(H[sub], E[sub]), "images": int(m.sum())}
        return out, sub


    codes = {k: codes_for(k) for k in runs}
    scores = {k: scores_for(k, codes[k]) for k in runs}
    for _k, (_s, _sub) in scores.items():
        print(f"{_k[0]:9s} {_k[1]:14s} subsample {len(_sub):,}  " + "  |  ".join(
            f"{n} ({d['images']:,} images): " + " ".join(f"{m} {v:.2f}" for m, v in d.items() if m != "images") for n, d in _s.items()))

    return PER_CLASS, codes, scores


@app.cell
def _(FIGURES, PER_CLASS, codes, plot_utils, runs, scores):
    def run_panel(key):
        """Features beside the predicted codes on the class-sorted subsample, coloured by class; Grams below with CKA over all images."""
        run, (sc, sub) = runs[key], scores[key]
        cols = {name: E[sub] for name, (E, _m) in codes[key].items()}
        fig = plot_utils.plot_embedding_panels(run["features"][sub], cols, None, cmap="magma" if run["C"] > 20 else None,
                                               colors=run["labels"][sub], scores={n: sc[n]["CKA"] for n in cols},
                                               score_label="CKA", first_name=f"{key[1]} features", starred="rest",
                                               gram_percentile=99 if run["C"] > 20 else None)   # 1000 classes: blocks, not norms
        fig.suptitle(f"{key[0]}: {len(sub):,} images, {PER_CLASS[key[0]]} per class, sorted and coloured by class", y=1.02)
        return plot_utils.save_figure(fig, FIGURES / f"vision_{key[0]}_{key[1]}.png")


    return (run_panel,)


@app.cell
def _(mo, runs):
    view_run = mo.ui.dropdown(options={f"{d} / {m}": (d, m) for d, m in sorted(runs)}, value=f"{sorted(runs)[0][0]} / {sorted(runs)[0][1]}", label="run")
    view_run
    return (view_run,)


@app.cell
def _(run_panel, view_run):
    run_panel(view_run.value)
    return


if __name__ == "__main__":
    app.run()
