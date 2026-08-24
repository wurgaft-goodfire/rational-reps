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
    # Representational geometry: predicted vs. learned

    The project's claim: the geometry a network learns is **predictable in closed form** from the
    structure of the task.

    For sweeps, use the CLI instead, e.g. `python -m src.code.train -m runner=linear setting=graph structure=ring,chain`.
    """)
    return


@app.cell
def _(mo):
    import os, sys
    import numpy as np

    # Project root, independent of the working directory (a headless run can start outside
    # the repo). chdir so Hydra resolves conf/, and add it to sys.path.
    ROOT = mo.notebook_dir().parent.parent    # src/notebooks -> <repo>
    os.chdir(ROOT); sys.path.insert(0, str(ROOT))

    from src.code.data import graph_conditional
    from src.code.geometry import predicted_geometry
    from src.code.eval import cka, rsa
    from src.code.train import train_inline, graph_overrides
    from src.code import plot_utils           # importing this applies the house plotting style

    return (
        cka,
        graph_conditional,
        graph_overrides,
        plot_utils,
        predicted_geometry,
        rsa,
        train_inline,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. The predicted geometry (closed form)

    Factor the ring's log-conditional — no training. The translation symmetry gives a degenerate
    top Fourier pair, and the top-2 code lands on a circle.
    """)
    return


@app.cell
def _(graph_conditional, plot_utils, predicted_geometry):
    # pick a structure here — everything below follows from it. `cooccurrence` is part of the spec
    # (diffusion | geodesic | adjacency | heat) and flows to both the predicted geometry and the trained
    # models; a product diffuses, but e.g. a hypercube wants "heat" so its bits stay independent (analogies).
    spec = {"structure": "chain", "cooccurrence": "diffusion",
            "n": 15}
    L = graph_conditional(spec)
    E_pred = predicted_geometry(L)
    fig_fact, _ = plot_utils.plot_factorization(L)                   # the decomposition schematic
    fig_pred = plot_utils.plot_predicted_geometry(spec, L, E_pred)   # saved once we have a run config (below)
    return E_pred, L, fig_fact, fig_pred, spec


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Train the model ladder inline

    Train four architectures on the *same* ring task — a transformer (on Markov walks) and k-linear /
    MLP / RNN readouts (on item→neighbor pairs) — each from a prespecified runner, no terminal.
    `train_inline` returns the live model + an in-memory history, so we can watch the geometry form.
    """)
    return


@app.cell
def _(
    cka,
    fig_fact,
    fig_pred,
    graph_overrides,
    plot_utils,
    predicted_geometry,
    rsa,
    spec,
    train_inline,
):
    ARCHS = ["transformer", "linear", "mlp", "rnn"]
    results = {}
    for arch in ARCHS:
        model, dataset, evaluator, history, config = train_inline(graph_overrides(spec) + [f"runner={arch}"])
        E_true = predicted_geometry(dataset.conditional)                       # matches the trained items
        reps = evaluator.item_representations(model)                            # pre-decode activations
        results[arch] = {"reps": reps, "model": model, "history": history,
                         "cka": cka(reps, E_true), "rsa": rsa(reps, E_true)}
        print(f"{arch:12s} CKA={results[arch]['cka']:.3f}  RSA={results[arch]['rsa']:.3f}")

    # every arch for this structure shares one experiment figure dir; `config` keys it (model level dropped)
    plot_utils.save_figure(fig_fact, plot_utils.figure_path(config, "factorization"))
    plot_utils.save_figure(fig_pred, plot_utils.figure_path(config, "predicted_geometry"))
    fig_rsa = plot_utils.plot_rsa_curves({a: results[a]["history"] for a in ARCHS})
    plot_utils.save_figure(fig_rsa, plot_utils.figure_path(config, "rsa_curves"))
    return ARCHS, config, results


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Learned embeddings & representational convergence

    Every architecture's token embedding (PCA to 2-D) beside the closed-form prediction — they land on
    the same circle, up to a rotation/reflection CKA/RSA are invariant to. The bars quantify the match;
    linear-readout heads (k-linear) print the geometry most sharply, since a nonlinear head can offload
    structure into its weights instead of the embedding.
    """)
    return


@app.cell
def _(ARCHS, E_pred, L, config, plot_utils, results, spec):
    fig_emb = plot_utils.plot_embedding_panels(E_pred, {a: results[a]["reps"] for a in ARCHS}, L, spec=spec)
    plot_utils.save_figure(fig_emb, plot_utils.figure_path(config, "embedding_panels"))
    fig_conv = plot_utils.plot_convergence_bars({a: {"cka": results[a]["cka"], "rsa": results[a]["rsa"]} for a in ARCHS})
    plot_utils.save_figure(fig_conv, plot_utils.figure_path(config, "convergence_bars"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. The decoder is predicted too

    `factor_logcond` returns both the code `E` (compared above) **and** the decoder `D` — the readout
    that maps code → output logits. Every architecture here has a *linear* readout (k-linear's tensor,
    the MLP/RNN final layer, the transformer's unembedding), so its `(embedding, readout)` pair is a
    learned factorization of the conditional: `code · readoutᵀ ≈ Lc`, exactly the SVD factorization
    `E · D = Lc` the oracle predicts.

    That factorization has a **scale gauge**: `(code, readout) → (code·M, readout·M⁻ᵀ)` leaves the
    product fixed. Trained from small init, gradient descent picks the *balanced* representative — the
    two factors grow together, so each carries `√S` of the scale (the `codeᵀcode − readoutᵀreadout`
    invariant stays ≈ 0). The oracle (`factor_logcond`) matches this: it splits the scale symmetrically,
    `E = U·√S` and `D = √S·V`, so the predicted decoder already carries structure — no rescaling needed.
    (The raw SVD instead dumps all of `S` into the code and leaves `D = V` a flat, structureless blob.)

    We score the match with **CKA** — the same metric as the code (§3), matched at the task's rank `r`.
    The panels lay the predicted decoder beside each architecture's readout (outcomes on the same
    manifold as the items); the bars show every readout recovers it.
    """)
    return


@app.cell
def _(ARCHS, L, cka, config, plot_utils, results, rsa, spec):
    from src.code.geometry import factor_logcond

    _, _, D_pred, S, _ = factor_logcond(L)          # balanced decoder D = sqrt(S)·V, (k, num_outcomes)
    r = int((S > 1e-6).sum())                        # the conditional's true rank (~ the models' capacity)
    D_geom = D_pred[:r].T                            # decoder on the meaningful subspace (num_outcomes, r)
    n_out = D_geom.shape[0]                          # drop the S=0 null directions that carry no structure

    def readout(model, arch):
        """Per-outcome readout (decoder) weights, restricted to the item outcomes."""
        W = (model.readout if arch == "linear" else
             model.mlp[-1].weight if arch == "mlp" else
             model.head.weight if arch == "rnn" else
             model.get_output_embeddings().weight)   # transformer unembedding
        return W.detach().cpu().numpy()[:n_out]

    # CKA — the code's metric (§3) — matched at rank r; the oracle's balanced decoder carries scale.
    dec_scores = {a: {"cka": cka(readout(results[a]["model"], a), D_geom),
                      "rsa": rsa(readout(results[a]["model"], a), D_geom)} for a in ARCHS}
    for a in ARCHS:
        print(f"  readout vs predicted decoder D (rank {r}):  {a:12s} "
              f"CKA={dec_scores[a]['cka']:.3f}  RSA={dec_scores[a]['rsa']:.3f}")

    # The predicted decoder geometry beside each architecture's readout (outcomes on the same manifold as
    # items), decoder RDMs below — the decoder mirror of the code panels in §3.
    fig_dec = plot_utils.plot_decoder_panels(D_geom, {a: readout(results[a]["model"], a) for a in ARCHS}, spec=spec)
    plot_utils.save_figure(fig_dec, plot_utils.figure_path(config, "decoder_panels"))

    # Every readout recovers the predicted decoder up to a rotation (CKA), the same bar as the code in §3.
    fig_dec_bars = plot_utils.plot_convergence_bars(
        dec_scores, ylabel="similarity to predicted decoder D",
        title="the decoder is predicted across architectures")
    plot_utils.save_figure(fig_dec_bars, plot_utils.figure_path(config, "decoder_convergence_bars"))
    return


if __name__ == "__main__":
    app.run()
