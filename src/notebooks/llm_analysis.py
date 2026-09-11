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
    # A pretrained LLM reads a graph walk in context

    A random walk on a graph, its nodes named by unrelated single-token nouns, is fed to
    Llama-3.1-8B as `word, word, word, ...`. Each node's representation is the residual at the
    separator after it, averaged over its appearances in the last 50 steps and over walks. The
    walk uses the one-step adjacency kernel; the prediction is the balanced factor of the
    **diffusion** log-conditional on the same graph, $E = U\sqrt{S}$ of the row-centred
    $\log p_\gamma(x' \mid x)$ with $p_\gamma \propto (I - \gamma P)^{-1}$, with $\gamma$ fitted.
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
    from src.code.train import resolve_config, graph_overrides
    from src.code.data import init_dataset, structure_adjacency, adjacency_conditional
    from src.code.geometry import predicted_geometry
    from src.code.analysis_utils import cka, align_representations
    from src.code.llm import node_tokens, node_means, bayes_floor
    from src.code import plot_utils
    return (
        Path,
        adjacency_conditional,
        align_representations,
        bayes_floor,
        cka,
        graph_overrides,
        init_dataset,
        node_means,
        node_tokens,
        np,
        plot_utils,
        plt,
        predicted_geometry,
        resolve_config,
        structure_adjacency,
        torch,
    )


@app.cell
def _(Path, graph_overrides, resolve_config):
    WINDOW, L, B = 50, 1024, 256               # steps averaged, walk length, walks
    WORLDS = {"ring": {"spec": {"structure": "ring", "n": 24, "cooccurrence": "adjacency"}, "backtrack": True},
              "grid": {"spec": {"structure": "grid", "n": 5, "cooccurrence": "adjacency"}, "backtrack": False},
              "grid, backtracking": {"spec": {"structure": "grid", "n": 5, "cooccurrence": "adjacency"}, "backtrack": True}}

    def config(world):
        w = WORLDS[world]
        return resolve_config(graph_overrides(w["spec"]) + [
            "runner=transformer", "model=llm", "name=llm", "sample_mode=sequence", "++lapse=0.0",
            f"++backtrack={str(w['backtrack']).lower()}",
            f"training_kwargs.context_length={L}", f"training_kwargs.batch_size={B}"])

    def cache_path(world):
        return Path(config(world).run_dir) / "node_means.npz"
    return B, L, WINDOW, WORLDS, cache_path, config


@app.cell
def _(WINDOW, WORLDS, cache_path, config, init_dataset, node_means, node_tokens, np, torch):
    # the expensive step, cached raw: walks, node words and the per-layer node means
    _todo = [w for w in WORLDS if not cache_path(w).exists()]
    if _todo:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _path = config(_todo[0]).model_kwargs.model_path
        _tok = AutoTokenizer.from_pretrained(_path)
        _llm = AutoModelForCausalLM.from_pretrained(_path, dtype=torch.bfloat16, device_map="cuda").eval()
        (_sep,) = _tok.encode(",", add_special_tokens=False)
        for _w in _todo:
            _cfg = config(_w); _ds = init_dataset(_cfg)
            _walks = _ds[0]["input_ids"][:, 1:]                                 # drop the start token
            _ids, _words = node_tokens(_tok, _ds.n_items, seed=_cfg.random_seed)
            _out = node_means(_llm, _walks, _ids, sep=_sep, window=WINDOW, prefix=(_tok.bos_token_id,))
            cache_path(_w).parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_path(_w), walks=_walks.cpu().numpy(), tokens=_ids, words=_words, **_out)
        del _llm; torch.cuda.empty_cache()
    runs = {w: dict(np.load(cache_path(w))) for w in WORLDS}
    return (runs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fit

    CKA between the LLM node means and the predicted code per layer, for the diffusion prediction
    at each $\gamma$ and for the adjacency prediction (the kernel the walk actually uses).
    """)
    return


@app.cell
def _(WORLDS, adjacency_conditional, cka, np, predicted_geometry, runs, structure_adjacency):
    GAMMAS = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99])

    def predicted(world, gamma=None):
        """Predicted code: the diffusion log-conditional at ``gamma``, or adjacency when None."""
        A = structure_adjacency(WORLDS[world]["spec"])
        L = (adjacency_conditional(A, cooccurrence="adjacency") if gamma is None
             else adjacency_conditional(A, gamma, "diffusion"))
        return predicted_geometry(L)

    scores = {}
    for _w in WORLDS:
        _M = runs[_w]["means"]
        scores[_w] = {"diffusion": np.array([[cka(_m, predicted(_w, g)) for g in GAMMAS] for _m in _M]),
                      "adjacency": np.array([cka(_m, predicted(_w)) for _m in _M])}

    def best_layer(world):
        """The layer (embedding excluded) and gamma index where the diffusion prediction fits best."""
        best = scores[world]["diffusion"].max(1)
        layer = 1 + int(best[1:].argmax())
        return layer, int(scores[world]["diffusion"][layer].argmax())
    return GAMMAS, best_layer, predicted, scores


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Curves

    Next-word NLL by step against the walk's Bayes floor (shaded: the averaged window); CKA by
    layer for the diffusion prediction at its best $\gamma$ and for adjacency; CKA against
    $\gamma$ at the best layer.
    """)
    return


@app.cell
def _(GAMMAS, L, WINDOW, WORLDS, bayes_floor, best_layer, config, plot_utils, plt, runs, scores, structure_adjacency):
    def curves(world):
        run, sc = runs[world], scores[world]
        layer, g = best_layer(world)
        best = sc["diffusion"].max(1)
        floor = bayes_floor(structure_adjacency(WORLDS[world]["spec"]), run["walks"], WORLDS[world]["backtrack"])
        fig, (a, b, c) = plt.subplots(1, 3, figsize=(16, 4.2), width_ratios=[2, 1, 1])
        a.plot(run["nll"], label="Llama-3.1-8B"); a.plot(floor, "--", color="k", label="Bayes floor")
        a.axvspan(L - 1 - WINDOW, L - 1, color="gray", alpha=0.15)
        a.set_xlabel("step"); a.set_ylabel("next-word NLL (nats)"); a.set_ylim(0, None); a.legend()
        a.set_title(f"{world}: learning the walk in context")
        b.plot(best, label=r"diffusion, best $\gamma$ per layer"); b.plot(sc["adjacency"], "--", label="adjacency")
        b.set_xlabel("layer"); b.set_ylabel("CKA to prediction"); b.set_ylim(0, 1); b.legend()
        c.semilogx(GAMMAS, sc["diffusion"][layer], marker="o", label=f"diffusion, layer {layer}")
        c.axhline(sc["adjacency"][layer], ls="--", color="C1", label="adjacency")
        c.set_xlabel(r"$\gamma$"); c.set_ylabel("CKA to prediction"); c.set_ylim(0, 1); c.legend()
        fig.tight_layout()
        print(f"{world}: layer {layer}, CKA diffusion {best[layer]:.3f} (γ={GAMMAS[g]}) vs adjacency "
              f"{sc['adjacency'][layer]:.3f}; final NLL {run['nll'][-WINDOW:].mean():.3f} vs floor {floor[-WINDOW:].mean():.3f}")
        return plot_utils.save_figure(fig, plot_utils.figure_path(config(world), "llm_incontext_curves"))
    return (curves,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comparison

    The predicted code beside Llama at the layer where the diffusion prediction fits best. Top: top
    two PCs, the LLM's rotated (Procrustes) onto the prediction's PC frame so the panels share one
    frame. Bottom: Grams, annotated with the CKA to the shown prediction.
    """)
    return


@app.cell
def _(GAMMAS, WORLDS, align_representations, best_layer, config, plot_utils, predicted, runs, scores):
    def panels(world):
        run, sc = runs[world], scores[world]
        layer, g = best_layer(world)
        name, E, M = f"Llama L{layer}", predicted(world, GAMMAS[g]), run["means"][layer]
        ref = plot_utils._pca(E, 2)                                             # the prediction's PC frame
        coords = {"predicted": ref, name: align_representations(M, ref, method="rotation", rank=2)[0]}
        fig = plot_utils.plot_embedding_panels(E, {name: M}, WORLDS[world]["spec"],
                                               scores={name: sc["diffusion"][layer, g]}, coords=coords)
        fig.suptitle(rf"{world}: diffusion prediction, $\gamma$={GAMMAS[g]}", y=1.02)
        return plot_utils.save_figure(fig, plot_utils.figure_path(config(world), "llm_incontext_panels"))
    return (panels,)


@app.cell
def _(curves):
    curves("ring")
    return


@app.cell
def _(panels):
    panels("ring")
    return


@app.cell
def _(curves):
    curves("grid")
    return


@app.cell
def _(panels):
    panels("grid")
    return


@app.cell
def _(curves):
    curves("grid, backtracking")
    return


@app.cell
def _(panels):
    panels("grid, backtracking")
    return


if __name__ == "__main__":
    app.run()
