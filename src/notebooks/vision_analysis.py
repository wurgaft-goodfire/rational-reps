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
    # Class geometry of vision classifiers from soft labels

    A classifier's pre-readout class means times its readout are its class log-conditional, so the
    theory predicts the class-mean geometry from any table of $P(y \mid x)$ over classes. Two
    tables per dataset: the human relabelling (ImageNet-ReaL votes, CIFAR-10H annotator votes) and
    the model's own mean softmax per class. One-hot labels would give the equiangular frame of neural
    collapse, whose distances are all equal. Each is factored as $E^\star = $ `factor_logcond`$(\log P)$ with an empty
    cell counted as one vote, and compared with the model's class means by RSA.
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
    from src.code.analysis_utils import pair_auc, rsa
    from src.code.geometry import class_means, predicted_geometry
    from src.code.vision import real_table, soft_table
    from src.code import plot_utils

    CACHE = Path(os.environ.get("CACHE_DIR", ROOT / "outputs"))
    VISION = CACHE / "vision"
    FIGURES = CACHE / "figures"; FIGURES.mkdir(parents=True, exist_ok=True)
    return (
        FIGURES,
        VISION,
        class_means,
        json,
        np,
        pair_auc,
        plot_utils,
        predicted_geometry,
        real_table,
        rsa,
        soft_table,
    )


@app.cell
def _(VISION, class_means, json, np, real_table, soft_table):
    def load_run(model, dataset):
        """Cached features and logits of ``model`` on ``dataset``, with the class means and the mean softmax per class."""
        d = dict(np.load(VISION / model / f"{dataset}.npz"))
        C = d["logits"].shape[1]
        P = np.exp(d["logits"].astype(np.float64)); P /= P.sum(1, keepdims=True)
        return {"features": d["features"].astype(np.float32), "labels": d["labels"], "index": d["index"], "C": C,
                "means": class_means(d["features"].astype(np.float64), d["labels"], C)[0],
                "softmax": np.stack([P[d["labels"] == x].mean(0) for x in range(C)]),
                "top1": float((d["logits"].argmax(1) == d["labels"]).mean())}


    def human_table(dataset, run):
        """The human relabelling as a (C, C) table in the run's class order."""
        if dataset == "imagenet":
            real = json.load(open(VISION / "labels" / "real.json"))
            return real_table([real[i] for i in run["index"]], run["labels"], run["C"])
        counts = np.load(VISION / "labels" / "cifar10h-counts.npy")                 # ~51 annotator votes per image
        n = np.bincount(run["labels"], minlength=run["C"])[:, None]
        return soft_table(counts[run["index"]], run["labels"], run["C"]) * n       # votes per class pair


    RUNS = {("imagenet", "resnet50"), ("imagenet", "vit_b_16"), ("cifar10", "cifar_resnet56")}
    runs = {k: load_run(k[1], k[0]) for k in RUNS if (VISION / k[1] / f"{k[0]}.npz").exists()}
    tables = {k: human_table(k[0], r) for k, r in runs.items()}
    for (_ds, _m), _r in runs.items():
        _T = tables[_ds, _m]
        print(f"{_ds:9s} {_m:14s} top-1 {_r['top1']:.3f}  classes {_r['C']}  human table: off-diagonal mass "
              f"{1 - np.trace(_T) / _T.sum():.3f}, classes with any off-diagonal vote {int(((_T - np.diag(np.diag(_T))).sum(1) > 0).sum())}")
    print("not cached:", sorted(RUNS - set(runs)) or "none")
    return runs, tables


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Predicted codes and agreement

    For each run: the model's class means beside the codes factored from the human table and the model's own
    softmax table. RSA over class pairs, with the floor from shuffling classes. The
    human table is sparse, so a second test asks whether the class pairs humans ever confuse are closer
    in the model than the rest (area under the curve, 0.5 = no relation).
    """)
    return


@app.cell
def _(np, pair_auc, predicted_geometry, rsa, runs, tables):
    def log_table(T, floor=1.0):
        """log P(y | x) from a (C, C) vote table, an empty cell counted as ``floor`` votes."""
        T = np.maximum(np.asarray(T, dtype=np.float64), floor)
        return np.log(T / T.sum(1, keepdims=True))


    def codes_for(key):
        """Codes from two vote tables on the same footing: human votes, and the model's mean softmax scaled to
        votes (one per image of the class)."""
        run, T = runs[key], tables[key]
        n = np.bincount(run["labels"], minlength=run["C"])[:, None].astype(float)
        return {"human": predicted_geometry(log_table(T)),
                "model softmax": predicted_geometry(log_table(run["softmax"] * n))}


    def scores_for(key, codes, seed=0):
        rng = np.random.default_rng(seed)
        E = runs[key]["means"]
        out = {name: rsa(E, code) for name, code in codes.items()}
        out["shuffle floor"] = float(np.mean([rsa(E, codes["human"][rng.permutation(len(E))]) for _ in range(20)]))
        T = tables[key]
        confusable = (T + T.T - 2 * np.diag(np.diag(T))) > 0
        out["confusable pairs closer (AUC)"] = pair_auc(E, confusable)
        return out


    codes = {k: codes_for(k) for k in runs}
    scores = {k: scores_for(k, codes[k]) for k in runs}
    for _k, _s in scores.items():
        print(f"{_k[0]:9s} {_k[1]:14s} " + "  ".join(f"{n} {v:.2f}" for n, v in _s.items()))
    return codes, scores


@app.cell
def _(FIGURES, codes, plot_utils, runs, scores):
    def run_panel(key):
        """The model's class means beside the three predicted codes, Grams below with RSA."""
        run = runs[key]
        fig = plot_utils.plot_embedding_panels(run["means"], codes[key], None, cmap="magma" if run["C"] > 20 else None,
                                               scores={n: scores[key][n] for n in codes[key]}, score_label="RSA",
                                               first_name=f"{key[1]} class means", starred="rest")
        fig.suptitle(f"{key[0]}: class means beside the codes predicted from the two label tables", y=1.02)
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
