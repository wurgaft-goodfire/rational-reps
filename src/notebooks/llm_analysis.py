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
    # A pretrained LLM reads a task from examples in context

    Items are named by unrelated single-token nouns and Llama-3.1-8B reads a long stream of
    examples with no instruction. Each key's representation is the residual at the separator
    after it, averaged over its appearances in the last 50 steps and over sequences, and is
    compared to the balanced factor $E = U\sqrt{S}$ of a row-centred log-conditional.

    **Walks.** `word, word, ...` on a ring and a grid under the one-step adjacency kernel; the
    prediction is the **diffusion** log-conditional $\log p_\gamma(x' \mid x)$,
    $p_\gamma \propto (I - \gamma P)^{-1}$, with $\gamma$ fitted.

    **Relations.** Triples `a b : c ,` read at the colon: `entailment` (15 tree nodes, $c$ uniform
    over the common ancestors of $a, b$) and `set_difference` (8 feature sets; six outcome words
    for "feature $f$ is in $a$ but not $b$: yes / no", asked with salience $1 : \tfrac12 : \tfrac14$).
    The prediction is the graded kernel's $(a,b) \times c$ log-conditional; the alternative drops
    the grading (lowest common ancestor only; equal salience).
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
    from src.code.data import (init_dataset, structure_adjacency, adjacency_conditional, relation_tensor,
                               answer_classes, apply_lapse)
    from src.code.geometry import predicted_geometry, log_conditional, class_means
    from src.code.analysis_utils import cka, align_representations
    from src.code.llm import node_tokens, node_means, bayes_floor, pair_means, triple_stream, relation_floor
    from src.code import plot_utils
    return (
        Path,
        adjacency_conditional,
        align_representations,
        answer_classes,
        apply_lapse,
        bayes_floor,
        cka,
        class_means,
        graph_overrides,
        init_dataset,
        log_conditional,
        node_means,
        node_tokens,
        np,
        pair_means,
        plot_utils,
        plt,
        predicted_geometry,
        relation_floor,
        relation_tensor,
        resolve_config,
        structure_adjacency,
        torch,
        triple_stream,
    )


@app.cell
def _(Path, graph_overrides, resolve_config):
    WINDOW, L, B = 50, 1024, 256               # steps averaged, walk length, sequences
    WORLDS = {"ring": {"spec": {"structure": "ring", "n": 24, "cooccurrence": "adjacency"}, "backtrack": True},
              "grid": {"spec": {"structure": "grid", "n": 5, "cooccurrence": "adjacency"}, "backtrack": True}}

    def config(world):
        w = WORLDS[world]
        return resolve_config(graph_overrides(w["spec"]) + [
            "runner=transformer", "model=llm", "name=llm", "sample_mode=sequence", "++lapse=0.0",
            f"++backtrack={str(w['backtrack']).lower()}",
            f"training_kwargs.context_length={L}", f"training_kwargs.batch_size={B}"])

    def cache_path(world):
        return Path(config(world).run_dir) / "node_means.npz"

    K = 400                                    # triples per sequence (five tokens each)
    RELATIONS = {"entailment": {"alt": {"graded": False}},
                 "set_difference": {"alt": {"salience": [1.0, 1.0, 1.0]}}}

    def rel_config(relation):
        return resolve_config(["runner=rel_transformer", f"relation={relation}", "model=llm", "name=llm",
                               "++lapse=0.0", f"training_kwargs.batch_size={B * K}"])

    def rel_cache(relation):
        return Path(rel_config(relation).run_dir) / "pair_means.npz"
    return B, K, L, RELATIONS, WINDOW, WORLDS, cache_path, config, rel_cache, rel_config


@app.cell
def _(B, K, RELATIONS, WINDOW, WORLDS, cache_path, config, init_dataset, node_means, node_tokens, np, pair_means, rel_cache, rel_config, torch, triple_stream):
    # the expensive step, cached raw: the streams, the item words and the per-layer means per key
    _walks_todo = [w for w in WORLDS if not cache_path(w).exists()]
    _rel_todo = [r for r in RELATIONS if not rel_cache(r).exists()]
    if _walks_todo or _rel_todo:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _path = config("ring").model_kwargs.model_path
        _tok = AutoTokenizer.from_pretrained(_path)
        _llm = AutoModelForCausalLM.from_pretrained(_path, dtype=torch.bfloat16, device_map="cuda").eval()
        (_sep,) = _tok.encode(",", add_special_tokens=False); (_colon,) = _tok.encode(":", add_special_tokens=False)
        for _w in _walks_todo:
            _cfg = config(_w); _ds = init_dataset(_cfg)
            _walks = _ds[0]["input_ids"][:, 1:]                                 # drop the start token
            _ids, _words = node_tokens(_tok, _ds.n_items, seed=_cfg.random_seed)
            _out = node_means(_llm, _walks, _ids, sep=_sep, window=WINDOW, prefix=(_tok.bos_token_id,))
            cache_path(_w).parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_path(_w), walks=_walks.cpu().numpy(), tokens=_ids, words=_words, **_out)
        for _r in _rel_todo:
            _cfg = rel_config(_r); _ds = init_dataset(_cfg)
            _triples = triple_stream(_ds, B, K)
            _ids, _words = node_tokens(_tok, _ds.n_items + _ds.n_outcomes * _ds.bipartite, seed=_cfg.random_seed)
            _items, _outs = _ids[:_ds.n_items], (_ids[_ds.n_items:] if _ds.bipartite else _ids)
            _out = pair_means(_llm, _triples, _items, _outs, colon=_colon, sep=_sep, window=WINDOW,
                              prefix=(_tok.bos_token_id,))
            rel_cache(_r).parent.mkdir(parents=True, exist_ok=True)
            np.savez(rel_cache(_r), triples=_triples.numpy(), tokens=_ids, words=_words, **_out)
        del _llm; torch.cuda.empty_cache()
    runs = {w: dict(np.load(cache_path(w))) for w in WORLDS}
    rel_runs = {r: dict(np.load(rel_cache(r))) for r in RELATIONS}
    return rel_runs, runs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Walks: fit

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
    ## Walks: figures

    Curves: next-word NLL by step against the walk's Bayes floor (shaded: the averaged window);
    CKA by layer for the diffusion prediction at its best $\gamma$ and for adjacency; CKA against
    $\gamma$ at the best layer. Panels: the predicted code beside Llama at that layer, top two PCs
    with the LLM's rotated (Procrustes) onto the prediction's PC frame, Grams below.
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Relations: fit

    CKA per layer between the LLM pair codes (one row per ordered pair) and the predicted pair code
    of the graded kernel and of its ungraded alternative, both with the relational setting's 5%
    lapse before the log.
    """)
    return


@app.cell
def _(RELATIONS, apply_lapse, cka, log_conditional, np, predicted_geometry, rel_runs, relation_tensor):
    def rel_predicted(relation, **kwargs):
        """Predicted pair code: the balanced factor of the lapsed (a,b) x c log-conditional."""
        T = relation_tensor({"relation": relation, **kwargs})
        return predicted_geometry(log_conditional(apply_lapse(T.reshape(-1, T.shape[-1]), 0.05)))

    rel_scores = {}
    for _r, _spec in RELATIONS.items():
        _M = rel_runs[_r]["means"]
        assert np.isfinite(_M).all(), f"{_r}: a pair never appeared in the averaged window"
        rel_scores[_r] = {"graded": np.array([cka(_m, rel_predicted(_r)) for _m in _M]),
                          "alternative": np.array([cka(_m, rel_predicted(_r, **_spec["alt"])) for _m in _M])}

    def rel_best_layer(relation):
        return 1 + int(rel_scores[relation]["graded"][1:].argmax())
    return rel_best_layer, rel_predicted, rel_scores


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Relations: figures

    Curves: NLL of $c$ at the colon by triple, split by whether the pair already appeared in the
    sequence, against the Bayes floor (the entropy of each pair's outcome row; shaded: the averaged
    window); CKA by layer for both kernels. Panels: the graded prediction beside Llama at its best
    layer, one point per answer class (class means), in the relation's own view.
    """)
    return


@app.cell
def _(K, WINDOW, np, plot_utils, plt, rel_best_layer, rel_config, rel_runs, rel_scores, relation_floor, relation_tensor):
    def rel_curves(relation, min_count=8):
        run, sc = rel_runs[relation], rel_scores[relation]
        T = relation_tensor({"relation": relation})
        floor = relation_floor(T, run["triples"][..., 0] * T.shape[0] + run["triples"][..., 1])   # (B, K)
        flat = T.reshape(-1, T.shape[-1])                                                       # pairs are uniform
        blind = -(flat * np.log(flat.mean(0))).sum(-1).mean()                                   # predict the marginal of c
        nll, seen = run["nll"], run["seen"]
        layer = rel_best_layer(relation)

        def by_step(values, mask):                     # mean over the sequences where mask holds, NaN when too few
            count = mask.sum(0)
            return np.where(count >= min_count, (values * mask).sum(0) / np.maximum(count, 1), np.nan)

        fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2), width_ratios=[2, 1])
        a.plot(by_step(nll, ~seen), color="C0", label="first appearance of the pair")
        a.plot(by_step(nll, seen), color="C2", label="repeated pair")
        a.plot(floor.mean(0), "--", color="k", label="Bayes floor")
        a.axhline(blind, ls=":", color="k", label="pair-blind: the marginal of c")
        a.axvspan(K - WINDOW, K - 1, color="gray", alpha=0.15)
        a.set_xlabel("triple"); a.set_ylabel("NLL of c (nats)"); a.set_ylim(0, None); a.legend()
        a.set_title(f"{relation}: learning the relation from examples")
        b.plot(sc["graded"], label="graded kernel"); b.plot(sc["alternative"], "--", label="ungraded alternative")
        b.set_xlabel("layer"); b.set_ylabel("CKA to prediction"); b.set_ylim(0, 1); b.legend()
        fig.tight_layout()
        tail = slice(K - WINDOW, K)
        print(f"{relation}: layer {layer}, CKA graded {sc['graded'][layer]:.3f} vs alternative "
              f"{sc['alternative'][layer]:.3f}; final NLL {nll[:, tail].mean():.3f} "
              f"(first {np.nanmean(by_step(nll, ~seen)[tail]):.3f}, repeated {np.nanmean(by_step(nll, seen)[tail]):.3f}) "
              f"vs floor {floor[:, tail].mean():.3f}, pair-blind {blind:.3f}")
        return plot_utils.save_figure(fig, plot_utils.figure_path(rel_config(relation), "llm_incontext_curves"))
    return (rel_curves,)


@app.cell
def _(answer_classes, class_means, plot_utils, rel_best_layer, rel_config, rel_predicted, rel_runs, rel_scores):
    def rel_panels(relation):
        layer = rel_best_layer(relation)
        classes, draw_spec = answer_classes(relation)
        name = f"Llama L{layer}"
        show = lambda X: class_means(X, classes)[0]                         # one point per answer
        fig = plot_utils.plot_embedding_panels(show(rel_predicted(relation)), {name: show(rel_runs[relation]["means"][layer])},
                                               spec=draw_spec, cmap=plot_utils.FIG.item_cmap,
                                               scores={name: rel_scores[relation]["graded"][layer]})
        fig.suptitle(f"{relation}: graded prediction", y=1.02)
        return plot_utils.save_figure(fig, plot_utils.figure_path(rel_config(relation), "llm_incontext_panels"))
    return (rel_panels,)


@app.cell
def _(rel_curves):
    rel_curves("entailment")
    return


@app.cell
def _(rel_panels):
    rel_panels("entailment")
    return


@app.cell
def _(rel_curves):
    rel_curves("set_difference")
    return


@app.cell
def _(rel_panels):
    rel_panels("set_difference")
    return


if __name__ == "__main__":
    app.run()
