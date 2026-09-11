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

    Items are named by unrelated single-token nouns and a Llama-3.1 model reads a long stream of
    examples. Each key's representation is the residual at the separator after it, averaged over
    its appearances in the last 50 steps and over sequences, and is compared to the balanced
    factor $E = U\sqrt{S}$ of a row-centred log-conditional.

    **Walks.** `word, word, ...` on a ring and a grid under the one-step adjacency kernel; the
    prediction is the **diffusion** log-conditional $\log p_\gamma(x' \mid x)$,
    $p_\gamma \propto (I - \gamma P)^{-1}$, with $\gamma$ fitted.

    **Relations.** Triples `a b : c ,` read at the colon: `entailment` (15 tree nodes, $c$ uniform
    over the common ancestors of $a, b$) and `set_difference` (8 feature sets; six outcome words
    for "feature $f$ is in $a$ but not $b$: yes / no", asked with salience $1 : \tfrac12 : \tfrac14$).
    Three variants: 8B from examples alone, 8B with the structure and the rule stated first, and
    70B from examples alone. The prediction is the graded kernel's $(a,b) \times c$
    log-conditional; the alternative drops the grading (lowest common ancestor only; equal salience).
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
    from src.code.llm import (node_tokens, node_means, bayes_floor, pair_means, triple_stream, relation_floor,
                              entailment_instruction, set_difference_instruction)
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
        entailment_instruction,
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
        set_difference_instruction,
        structure_adjacency,
        torch,
        triple_stream,
    )


@app.cell
def _(Path, graph_overrides, resolve_config):
    WINDOW, L, B = 50, 1024, 256               # steps averaged, walk length, sequences
    WORLDS = {"ring": {"spec": {"structure": "ring", "n": 24, "cooccurrence": "adjacency"}, "backtrack": True},
              "grid": {"spec": {"structure": "grid", "n": 5, "cooccurrence": "adjacency"}, "backtrack": True},
              "tree": {"spec": {"structure": "tree_graph", "n": 3, "k": 2, "cooccurrence": "adjacency"}, "backtrack": True,
                       "draw": {"structure": "tree", "n": 3, "k": 2}},          # the tree's ultrametric view
              "hypercube": {"spec": {"structure": "hypercube", "n": 3, "cooccurrence": "adjacency"}, "backtrack": True}}

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
    VARIANTS = {"8B, examples": {"model": "llm", "instruction": False},
                "8B, instructed": {"model": "llm", "instruction": True},
                "70B, examples": {"model": "llm_70b", "instruction": False}}
    BATCH = {"llm": 16, "llm_70b": 4}          # sequences per forward pass

    def rel_config(relation, variant):
        v = VARIANTS[variant]
        return resolve_config(["runner=rel_transformer", f"relation={relation}", f"model={v['model']}", "name=llm",
                               "++lapse=0.0", f"training_kwargs.batch_size={B * K}"]
                              + (["++instruction=true"] if v["instruction"] else []))

    def rel_cache(relation, variant):
        return Path(rel_config(relation, variant).run_dir) / "pair_means.npz"
    return B, BATCH, K, L, RELATIONS, VARIANTS, WINDOW, WORLDS, cache_path, config, rel_cache, rel_config


@app.cell
def _(answer_classes, entailment_instruction, np, relation_tensor, set_difference_instruction):
    def rel_words(relation, dataset):
        """How many nouns a relation needs beyond its items: a separate outcome alphabet, and
        feature names for the instruction."""
        n_features = int(np.log2(dataset.n_items)) if relation == "set_difference" else 0
        return dataset.n_items + dataset.n_outcomes * dataset.bipartite + n_features

    def rel_instruction(relation, words, dataset):
        """The instruction text for a relation, from its item / outcome / feature words."""
        n, m = dataset.n_items, dataset.n_outcomes
        if relation == "entailment":
            return entailment_instruction(words, answer_classes("entailment")[1]["parents"])
        T = relation_tensor({"relation": relation}); nbit = m // 2
        salience = T[0, 0, :nbit] + T[0, 0, nbit:]                     # how often each feature is asked
        return set_difference_instruction(words[:n], words[n + m:], words[n:n + m], salience)
    return rel_instruction, rel_words


@app.cell
def _(B, BATCH, K, RELATIONS, VARIANTS, WINDOW, WORLDS, cache_path, config, init_dataset, node_means, node_tokens, np, pair_means, rel_cache, rel_config, rel_instruction, rel_words, torch, triple_stream):
    # the expensive step, cached raw: the streams, the words and the per-layer means per key; one
    # model load per model that still has work
    _jobs = {}
    for _w in WORLDS:
        if not cache_path(_w).exists():
            _jobs.setdefault("llm", []).append((config(_w), "walk", _w, None))
    for _r in RELATIONS:
        for _v in VARIANTS:
            if not rel_cache(_r, _v).exists():
                _jobs.setdefault(VARIANTS[_v]["model"], []).append((rel_config(_r, _v), "relation", _r, _v))
    for _model, _todo in _jobs.items():
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _path = _todo[0][0].model_kwargs.model_path
        _tok = AutoTokenizer.from_pretrained(_path)
        _llm = AutoModelForCausalLM.from_pretrained(_path, dtype=torch.bfloat16, device_map="auto").eval()
        (_sep,) = _tok.encode(",", add_special_tokens=False); (_colon,) = _tok.encode(":", add_special_tokens=False)
        for _cfg, _kind, _key, _variant in _todo:
            _ds = init_dataset(_cfg)
            if _kind == "walk":
                _walks = _ds[0]["input_ids"][:, 1:]                             # drop the start token
                _ids, _words = node_tokens(_tok, _ds.n_items, seed=_cfg.random_seed)
                _out = node_means(_llm, _walks, _ids, sep=_sep, window=WINDOW, prefix=(_tok.bos_token_id,),
                                  batch_size=BATCH[_model])
                cache_path(_key).parent.mkdir(parents=True, exist_ok=True)
                np.savez(cache_path(_key), walks=_walks.cpu().numpy(), tokens=_ids, words=_words, **_out)
                continue
            _triples = triple_stream(_ds, B, K)
            _ids, _words = node_tokens(_tok, rel_words(_key, _ds), seed=_cfg.random_seed)
            _n, _m = _ds.n_items, _ds.n_outcomes
            _items, _outs = _ids[:_n], (_ids[_n:_n + _m] if _ds.bipartite else _ids[:_n])
            _text = rel_instruction(_key, _words, _ds) if VARIANTS[_variant]["instruction"] else ""
            _prefix = (_tok.bos_token_id, *(_tok.encode(_text + "\n", add_special_tokens=False) if _text else ()))
            _out = pair_means(_llm, _triples, _items, _outs, colon=_colon, sep=_sep, window=WINDOW,
                              prefix=_prefix, batch_size=BATCH[_model])
            rel_cache(_key, _variant).parent.mkdir(parents=True, exist_ok=True)
            np.savez(rel_cache(_key, _variant), triples=_triples.numpy(), tokens=_ids, words=_words,
                     instruction=_text, **_out)
        del _llm; torch.cuda.empty_cache()
    runs = {w: dict(np.load(cache_path(w))) for w in WORLDS}
    rel_runs = {(r, v): dict(np.load(rel_cache(r, v))) for r in RELATIONS for v in VARIANTS}
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
        """The layer (embedding excluded), kernel and gamma index where a prediction fits best;
        the gamma index is None when the adjacency prediction wins."""
        d, a = scores[world]["diffusion"].max(1), scores[world]["adjacency"]
        if a[1:].max() > d[1:].max():
            return 1 + int(a[1:].argmax()), "adjacency", None
        layer = 1 + int(d[1:].argmax())
        return layer, "diffusion", int(scores[world]["diffusion"][layer].argmax())

    def best_prediction(world):
        """``(name, E, CKA)`` of the best-fitting prediction at its best layer."""
        layer, kernel, g = best_layer(world)
        if kernel == "adjacency":
            return "adjacency", predicted(world), scores[world]["adjacency"][layer]
        return rf"diffusion, $\gamma$={GAMMAS[g]}", predicted(world, GAMMAS[g]), scores[world]["diffusion"][layer, g]
    return GAMMAS, best_layer, best_prediction, predicted, scores


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
def _(GAMMAS, L, WINDOW, WORLDS, bayes_floor, best_layer, best_prediction, config, plot_utils, plt, runs, scores, structure_adjacency):
    def curves(world):
        run, sc = runs[world], scores[world]
        layer, kernel, g = best_layer(world)
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
        print(f"{world}: best {kernel} at layer {layer}, CKA diffusion {best[layer]:.3f} (γ={GAMMAS[sc['diffusion'][layer].argmax()]}) "
              f"vs adjacency {sc['adjacency'][layer]:.3f}; final NLL {run['nll'][-WINDOW:].mean():.3f} vs floor {floor[-WINDOW:].mean():.3f}")
        return plot_utils.save_figure(fig, plot_utils.figure_path(config(world), "llm_incontext_curves"))
    return (curves,)


@app.cell
def _(WORLDS, align_representations, best_layer, best_prediction, config, plot_utils, runs):
    def panels(world):
        layer, _, _ = best_layer(world)
        label, E, score = best_prediction(world)
        name, M = f"Llama L{layer}", runs[world]["means"][layer]
        draw = WORLDS[world].get("draw", WORLDS[world]["spec"])
        k = 3 if plot_utils._view(draw) == "3d" else 2                          # the structure's own view
        ref = plot_utils._pca(E, k)                                             # the prediction's PC frame
        coords = {"predicted": ref, name: align_representations(M, ref, method="rotation", rank=k)[0]}
        fig = plot_utils.plot_embedding_panels(E, {name: M}, draw, scores={name: score}, coords=coords)
        fig.suptitle(f"{world}: {label} prediction", y=1.02)
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
    curves("tree")
    return


@app.cell
def _(panels):
    panels("tree")
    return


@app.cell
def _(curves):
    curves("hypercube")
    return


@app.cell
def _(panels):
    panels("hypercube")
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
def _(RELATIONS, VARIANTS, apply_lapse, cka, log_conditional, np, predicted_geometry, rel_runs, relation_tensor):
    def rel_predicted(relation, **kwargs):
        """Predicted pair code: the balanced factor of the lapsed (a,b) x c log-conditional."""
        T = relation_tensor({"relation": relation, **kwargs})
        return predicted_geometry(log_conditional(apply_lapse(T.reshape(-1, T.shape[-1]), 0.05)))

    rel_scores = {}
    for (_r, _v), _run in rel_runs.items():
        _M = _run["means"]
        assert np.isfinite(_M).all(), f"{_r}, {_v}: a pair never appeared in the averaged window"
        rel_scores[_r, _v] = {"graded": np.array([cka(_m, rel_predicted(_r)) for _m in _M]),
                              "alternative": np.array([cka(_m, rel_predicted(_r, **RELATIONS[_r]["alt"])) for _m in _M])}

    def rel_best_layer(relation, variant):
        return 1 + int(rel_scores[relation, variant]["graded"][1:].argmax())
    return rel_best_layer, rel_predicted, rel_scores


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Relations: figures

    Curves: NLL of $c$ at the colon by triple for each variant, against the Bayes floor (the entropy
    of each pair's outcome row) and the pair-blind marginal of $c$ (shaded: the averaged window);
    CKA by relative depth for both kernels (solid graded, dashed alternative). Panels: the graded
    prediction beside each variant at its best layer, one point per answer class (class means), in
    the relation's own view.
    """)
    return


@app.cell
def _(K, RELATIONS, VARIANTS, WINDOW, np, plot_utils, plt, rel_best_layer, rel_config, rel_runs, rel_scores, relation_floor, relation_tensor):
    def rel_curves(relation):
        T = relation_tensor({"relation": relation}); flat = T.reshape(-1, T.shape[-1])
        blind = -(flat * np.log(flat.mean(0))).sum(-1).mean()                   # predict the marginal of c
        fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2), width_ratios=[2, 1])
        for _i, _v in enumerate(VARIANTS):
            run, sc = rel_runs[relation, _v], rel_scores[relation, _v]
            floor = relation_floor(T, run["triples"][..., 0] * T.shape[0] + run["triples"][..., 1])
            layer, tail = rel_best_layer(relation, _v), slice(K - WINDOW, K)
            a.plot(run["nll"].mean(0), color=f"C{_i}", label=_v)
            depth = np.arange(len(sc["graded"])) / (len(sc["graded"]) - 1)
            b.plot(depth, sc["graded"], color=f"C{_i}", label=_v); b.plot(depth, sc["alternative"], "--", color=f"C{_i}")
            seen = run["seen"][:, tail]
            print(f"{relation}, {_v}: layer {layer}, CKA graded {sc['graded'][layer]:.3f} vs alternative "
                  f"{sc['alternative'][layer]:.3f}; final NLL {run['nll'][:, tail].mean():.3f} "
                  f"(first {run['nll'][:, tail][~seen].mean():.3f}, repeated {run['nll'][:, tail][seen].mean():.3f}) "
                  f"vs floor {floor[:, tail].mean():.3f}, pair-blind {blind:.3f}")
        a.plot(floor.mean(0), "--", color="k", label="Bayes floor")
        a.axhline(blind, ls=":", color="k", label="pair-blind: the marginal of c")
        a.axvspan(K - WINDOW, K - 1, color="gray", alpha=0.15)
        a.set_xlabel("triple"); a.set_ylabel("NLL of c (nats)"); a.set_ylim(0, None); a.legend()
        a.set_title(f"{relation}: learning the relation in context")
        b.set_xlabel("layer / depth"); b.set_ylabel("CKA to prediction"); b.set_ylim(0, 1); b.legend()
        fig.tight_layout()
        return plot_utils.save_figure(fig, plot_utils.figure_path(rel_config(relation, "8B, examples"), "llm_incontext_curves"))
    return (rel_curves,)


@app.cell
def _(VARIANTS, answer_classes, class_means, plot_utils, rel_best_layer, rel_config, rel_predicted, rel_runs, rel_scores):
    def rel_panels(relation):
        classes, draw_spec = answer_classes(relation)
        show = lambda X: class_means(X, classes)[0]                         # one point per answer
        layers = {v: rel_best_layer(relation, v) for v in VARIANTS}
        fig = plot_utils.plot_embedding_panels(
            show(rel_predicted(relation)), {f"{v}, L{l}": show(rel_runs[relation, v]["means"][l]) for v, l in layers.items()},
            spec=draw_spec, cmap=plot_utils.FIG.item_cmap,
            scores={f"{v}, L{l}": rel_scores[relation, v]["graded"][l] for v, l in layers.items()})
        fig.suptitle(f"{relation}: graded prediction", y=1.02)
        return plot_utils.save_figure(fig, plot_utils.figure_path(rel_config(relation, "8B, examples"), "llm_incontext_panels"))
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
