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
    examples with no instruction. Each key's representation is the residual at the separator after it, averaged over
    its appearances in the last 50 steps and over sequences, and is compared to the balanced
    factor $E = U\sqrt{S}$ of a row-centred log-conditional.

    **Walks.** `word, word, ...`: ring and grid under the one-step adjacency kernel, the 3-cube
    under the hypercube setting's heat kernel $\exp(\gamma A)$, the tree's leaves under the
    tree's own diffusion kernel (both $\gamma = 0.8$), and ring and grid again under the Gaussian
    kernel $\log p \propto -\tfrac{\gamma}{2}\lVert p_i - p_x \rVert^2$ in their canonical
    coordinates (von Mises on the ring, rank 2; a quadratic on the lattice, rank 3). Every world is
    scored against two predictions: the **diffusion** kernel on its graph,
    $p_\gamma \propto (I - \gamma P)^{-1}$ with $\gamma$ fitted (the tree's own kernel is one), and
    the kernel the data was sampled from.

    **Relations.** Triples `a b : c ,` read at the colon: `entailment` (15 tree nodes, $c$ uniform
    over the common ancestors of $a, b$) and `set_difference` (8 feature sets; six outcome words
    for "feature $f$ is in $a$ but not $b$: yes / no", asked with salience $1 : \tfrac12 : \tfrac14$).
    Read with the 8B and the 70B model. The prediction is the graded kernel's $(a,b) \times c$
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
    from src.code.data import init_dataset, graph_conditional, relation_tensor, answer_classes, apply_lapse
    from src.code.geometry import predicted_geometry, log_conditional, class_means
    from src.code.analysis_utils import cka, align_representations
    from src.code.llm import node_tokens, node_means, pair_means, triple_stream, row_entropy
    from src.code import plot_utils
    return (
        Path,
        align_representations,
        answer_classes,
        apply_lapse,
        cka,
        class_means,
        graph_conditional,
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
        relation_tensor,
        resolve_config,
        row_entropy,
        torch,
        triple_stream,
    )


@app.cell
def _(Path, graph_overrides, resolve_config):
    WINDOW, L, B = 50, 1024, 256               # steps averaged, walk length, sequences
    WORLDS = {"ring": {"spec": {"structure": "ring", "n": 24, "cooccurrence": "adjacency"}, "backtrack": True},
              "grid": {"spec": {"structure": "grid", "n": 5, "cooccurrence": "adjacency"}, "backtrack": True},
              "hypercube": {"spec": {"structure": "hypercube", "n": 3, "gamma": 0.8, "cooccurrence": "heat"}, "backtrack": True},
              "hypercube, sharper": {"spec": {"structure": "hypercube", "n": 3, "gamma": 0.3, "cooccurrence": "heat"}, "backtrack": True},
              "tree": {"spec": {"structure": "tree", "n": 3, "k": 2}, "backtrack": True},    # leaves under the tree's own kernel
              "ring, gaussian": {"spec": {"structure": "ring", "n": 24, "gamma": 8, "cooccurrence": "gaussian"}, "backtrack": True},
              "grid, gaussian": {"spec": {"structure": "grid", "n": 5, "gamma": 2, "cooccurrence": "gaussian"}, "backtrack": True}}

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
    VARIANTS = {"8B": "llm", "70B": "llm_70b"}   # the model each relation run reads
    BATCH = {"llm": 16, "llm_70b": 4}          # sequences per forward pass

    def rel_config(relation, variant):
        return resolve_config(["runner=rel_transformer", f"relation={relation}", f"model={VARIANTS[variant]}",
                               "name=llm", "++lapse=0.0", f"training_kwargs.batch_size={B * K}"])

    def rel_cache(relation, variant):
        return Path(rel_config(relation, variant).run_dir) / "pair_means.npz"
    return B, BATCH, K, L, RELATIONS, VARIANTS, WINDOW, WORLDS, cache_path, config, rel_cache, rel_config


@app.cell
def _(B, BATCH, K, RELATIONS, VARIANTS, WINDOW, WORLDS, cache_path, config, init_dataset, node_means, node_tokens, np, pair_means, rel_cache, rel_config, torch, triple_stream):
    # the expensive step, cached raw: the streams, the words and the per-layer means per key; one
    # model load per model that still has work
    _jobs = {}
    for _w in WORLDS:
        if not cache_path(_w).exists():
            _jobs.setdefault("llm", []).append((config(_w), "walk", _w, None))
    for _r in RELATIONS:
        for _v in VARIANTS:
            if not rel_cache(_r, _v).exists():
                _jobs.setdefault(VARIANTS[_v], []).append((rel_config(_r, _v), "relation", _r, _v))
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
            _n, _m = _ds.n_items, _ds.n_outcomes
            _ids, _words = node_tokens(_tok, _n + _m * _ds.bipartite, seed=_cfg.random_seed)   # own outcome alphabet if bipartite
            _items, _outs = _ids[:_n], (_ids[_n:] if _ds.bipartite else _ids)
            _out = pair_means(_llm, _triples, _items, _outs, colon=_colon, sep=_sep, window=WINDOW,
                              prefix=(_tok.bos_token_id,), batch_size=BATCH[_model])
            rel_cache(_key, _variant).parent.mkdir(parents=True, exist_ok=True)
            np.savez(rel_cache(_key, _variant), triples=_triples.numpy(), tokens=_ids, words=_words, **_out)
        del _llm; torch.cuda.empty_cache()
    runs = {w: dict(np.load(cache_path(w))) for w in WORLDS}
    rel_runs = {(r, v): dict(np.load(rel_cache(r, v))) for r in RELATIONS for v in VARIANTS}
    return rel_runs, runs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Walks: fit

    CKA between the LLM node means and the predicted code per layer, for the diffusion kernel at
    each $\gamma$ and for the kernel the walk was sampled from.
    """)
    return


@app.cell
def _(WORLDS, cka, graph_conditional, np, predicted_geometry, runs):
    GAMMAS = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99])

    def logcond(world, gamma=None):
        """The world's log-conditional: ``None`` is the kernel the data was sampled from; at
        ``gamma``, the diffusion kernel on the same graph (the tree's own kernel is one)."""
        spec = WORLDS[world]["spec"]
        return graph_conditional(spec if gamma is None else {**spec, "gamma": gamma, "cooccurrence": "diffusion"})

    def predicted(world, gamma=None):
        """Predicted code: the balanced factor of ``logcond(world, gamma)``."""
        return predicted_geometry(logcond(world, gamma))

    scores = {}
    for _w in WORLDS:
        _M = runs[_w]["means"]
        scores[_w] = {"family": np.array([[cka(_m, predicted(_w, g)) for g in GAMMAS] for _m in _M]),
                      "data": np.array([cka(_m, predicted(_w)) for _m in _M])}

    def best_layer(world):
        """The layer (embedding excluded), kernel and gamma index where a prediction fits best;
        the gamma index is None when the data's own kernel wins."""
        f, d = scores[world]["family"].max(1), scores[world]["data"]
        if d[1:].max() > f[1:].max():
            return 1 + int(d[1:].argmax()), "data", None
        layer = 1 + int(f[1:].argmax())
        return layer, "family", int(scores[world]["family"][layer].argmax())

    def best_prediction(world):
        """``(label, E, CKA)`` of the best-fitting prediction at its best layer."""
        layer, kernel, g = best_layer(world)
        if kernel == "data":
            return "the data's kernel", predicted(world), scores[world]["data"][layer]
        return rf"diffusion, $\gamma$={GAMMAS[g]}", predicted(world, GAMMAS[g]), scores[world]["family"][layer, g]
    return GAMMAS, best_layer, best_prediction, logcond, predicted, scores


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Walks: figures

    Curves: next-word NLL by step against the walk's Bayes floor (shaded: the averaged window);
    CKA by layer for the diffusion kernel at its best $\gamma$ and for the data's kernel; CKA
    against $\gamma$ at the best layer. Panels: the predicted code beside Llama at that layer, top two PCs
    with the LLM's rotated (Procrustes) onto the prediction's PC frame, Grams below.
    """)
    return


@app.cell
def _(GAMMAS, L, WINDOW, best_layer, config, logcond, np, plot_utils, plt, row_entropy, runs, scores):
    def curves(world):
        run, sc = runs[world], scores[world]
        layer, kernel, g = best_layer(world)
        best = sc["family"].max(1)
        floor = row_entropy(np.exp(logcond(world)), run["walks"][:, :-1]).mean(0)
        fig, (a, b, c) = plt.subplots(1, 3, figsize=(16, 4.2), width_ratios=[2, 1, 1])
        a.plot(run["nll"], label="Llama-3.1-8B"); a.plot(floor, "--", color="k", label="Bayes floor")
        a.axvspan(L - 1 - WINDOW, L - 1, color="gray", alpha=0.15)
        a.set_xlabel("step"); a.set_ylabel("next-word NLL (nats)"); a.set_ylim(0, None); a.legend()
        a.set_title(f"{world}: learning the walk in context")
        b.plot(best, label=r"diffusion, best $\gamma$ per layer"); b.plot(sc["data"], "--", label="the data's kernel")
        b.set_xlabel("layer"); b.set_ylabel("CKA to prediction"); b.set_ylim(0, 1); b.legend()
        c.semilogx(GAMMAS, sc["family"][layer], marker="o", label=f"diffusion, layer {layer}")
        c.axhline(sc["data"][layer], ls="--", color="C1", label="the data's kernel")
        c.set_xlabel(r"$\gamma$"); c.set_ylabel("CKA to prediction"); c.set_ylim(0, 1); c.legend()
        fig.tight_layout()
        print(f"{world}: best {kernel} at layer {layer}, CKA diffusion {best[layer]:.3f} (γ={GAMMAS[sc['family'][layer].argmax()]}) "
              f"vs data kernel {sc['data'][layer]:.3f}; final NLL {run['nll'][-WINDOW:].mean():.3f} vs floor {floor[-WINDOW:].mean():.3f}")
        return plot_utils.save_figure(fig, plot_utils.figure_path(config(world), "llm_incontext_curves"))
    return (curves,)


@app.cell
def _(WORLDS, align_representations, best_layer, best_prediction, config, plot_utils, runs):
    def panels(world):
        layer, _, _ = best_layer(world)
        label, E, score = best_prediction(world)
        name, M, spec = f"Llama L{layer}", runs[world]["means"][layer], WORLDS[world]["spec"]
        k = 3 if plot_utils._view(spec) == "3d" else 2                          # the structure's own view
        ref = plot_utils._pca(E, k)                                             # the prediction's PC frame
        coords = {"predicted": ref, name: align_representations(M, ref, method="rotation", rank=k)[0]}
        fig = plot_utils.plot_embedding_panels(E, {name: M}, spec, scores={name: score}, coords=coords)
        fig.suptitle(f"{world}: prediction from {label}", y=1.02)
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
    curves("ring, gaussian")
    return


@app.cell
def _(panels):
    panels("ring, gaussian")
    return


@app.cell
def _(curves):
    curves("grid, gaussian")
    return


@app.cell
def _(panels):
    panels("grid, gaussian")
    return


@app.cell
def _(curves):
    curves("hypercube")
    return


@app.cell
def _(panels):
    panels("hypercube")
    return


@app.cell
def _(curves):
    curves("hypercube, sharper")
    return


@app.cell
def _(panels):
    panels("hypercube, sharper")
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
def _(K, RELATIONS, VARIANTS, WINDOW, np, plot_utils, plt, rel_best_layer, rel_config, rel_runs, rel_scores, relation_tensor, row_entropy):
    def rel_curves(relation):
        T = relation_tensor({"relation": relation}); flat = T.reshape(-1, T.shape[-1])
        blind = -(flat * np.log(flat.mean(0))).sum(-1).mean()                   # predict the marginal of c
        fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2), width_ratios=[2, 1])
        for _i, _v in enumerate(VARIANTS):
            run, sc = rel_runs[relation, _v], rel_scores[relation, _v]
            floor = row_entropy(T, run["triples"][..., 0] * T.shape[0] + run["triples"][..., 1])
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
        return plot_utils.save_figure(fig, plot_utils.figure_path(rel_config(relation, "8B"), "llm_incontext_curves"))
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
        return plot_utils.save_figure(fig, plot_utils.figure_path(rel_config(relation, "8B"), "llm_incontext_panels"))
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
