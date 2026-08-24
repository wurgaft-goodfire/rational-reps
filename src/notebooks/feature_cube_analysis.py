import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Word → feature: sparsity vs importance under a rank budget

    $N$ words, each a bundle of $m$ features drawn i.i.d. Bernoulli($p$) in $\{0, 1\}$;
    **sparsity** $= 1 - p$. Outcomes are the $m$ feature tokens:
    $\log p(i \mid x) = \mathrm{log\,softmax}_i(\alpha_i Z_{xi})$, $\alpha_i = s \cdot r^i$.
    Knobs: $p$ = co-activation (the TMS sparsity dial), $r$ = importance ladder, $s$ =
    sharpness. No null token: the softmax gauge absorbs one dimension, so effective rank is
    $m - 1$ and the budget question is $d < m - 1$.

    The sweep stops at $p = 0.5$: complementing the features ($z \mapsto 1 - z$) maps
    $p > 0.5$ onto the sparse task plus a fixed outcome field $\alpha_i$ — a frequency
    ladder, not new sparsity physics.

    Oracles in the trained model's class (free per-item codes, bias-free readout):
    **truncation** (`factor_logcond`, rank-$d$ prefix) and **nuclear**
    (`nuclear_features(logP, I, d)`, the exact-CE optimum). Dropout is read from the fitted
    per-feature swing $\hat\alpha_i$ (`feature_stats`: gauge-free least squares
    $G_{xi} \sim a_i Z_{xi} + b_i + c_x$).
    """)
    return


@app.cell
def _():
    import os, sys
    from functools import lru_cache
    import numpy as np
    import marimo as mo

    ROOT = mo.notebook_dir().parent.parent    # src/notebooks -> <repo>
    os.chdir(ROOT); sys.path.insert(0, str(ROOT))
    from src.code.data import feature_cube
    from src.code.geometry import (truncation_features, nuclear_features, expected_ce,
                                   feature_stats, feature_vectors, superposition_measure)
    from src.code.train import resolve_config, train_inline
    from src.code import plot_utils
    import matplotlib.pyplot as plt

    # The task is conf/setting/feature_cube.yaml; read its parameters off the config rather than
    # restating them. Append an override to explore a variant — it keys its own figure directory.
    TASK = ["setting=feature_cube"]
    CONFIG = resolve_config(TASK)
    N, M, DECAY, SCALE, SEED = (CONFIG.structure_kwargs[k] for k in ("N", "m", "decay", "scale", "seed"))

    # sweep axes: p and d vary within the notebook, so they stay out of the task and its directory
    P_SWEEP = [float(round(p, 4)) for p in np.geomspace(0.5, 0.005, 10)]
    D_GRID = [2, 3, 4]
    D_PANEL = 3


    SPARSITY = [1.0 - p for p in P_SWEEP]
    COLORS = {2: "#2a78d6", 3: "#eb6834", 4: "#1baf7a"}


    def sparsity_ticks(ax):
        """Logit x-scale with plain floating-point tick labels."""
        from matplotlib.ticker import FixedLocator, NullLocator
        ticks = [0.5, 0.7, 0.9, 0.95, 0.98, 0.995]
        ax.set_xscale("logit")
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels([f"{t:g}" for t in ticks])


    @lru_cache(maxsize=None)
    def oracle(p, d):
        """Truncation and nuclear fits at (p, d): task, codes/readouts, excesses over Bayes,
        and fitted per-feature (swing, intercept)."""
        logP, info = feature_cube(N, M, p, DECAY, SCALE, SEED)
        I = np.eye(len(logP))
        zeros = np.zeros(logP.shape[1])
        A_cf, D_cf = truncation_features(logP, d)
        ce_cf, bayes = expected_ce(I, logP, A_cf, D_cf, zeros)
        A, D, b, _ = nuclear_features(logP, I, d, fit_bias=False)
        ce_nuc, _ = expected_ce(I, logP, A, D, b)
        return dict(**info,
                    cf=dict(E=A_cf.T, ex=ce_cf - bayes,
                            stats=feature_stats(A_cf.T @ D_cf, info["Z"])),
                    nuc=dict(E=A.T, ex=ce_nuc - bayes,
                             stats=feature_stats(A.T @ D, info["Z"])))

    return (
        COLORS,
        CONFIG,
        D_GRID,
        D_PANEL,
        M,
        P_SWEEP,
        SPARSITY,
        TASK,
        feature_vectors,
        mo,
        np,
        oracle,
        plot_utils,
        plt,
        sparsity_ticks,
        superposition_measure,
        train_inline,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 1 — excess CE across the sparsity sweep

    For each sparsity $1 - p$ and budget $d$: excess CE over Bayes of the rank-$d$
    truncation (dashed) and the nuclear optimum (solid), uniform over the $N$ words.
    Color = budget, linestyle = oracle. Criterion: if sparsity is a packing lever, the
    nuclear-over-truncation gain should grow with sparsity (rare co-activation makes shared
    dimensions cheap); a constant ratio between the line styles falsifies it.
    """)
    return


@app.cell
def _(
    COLORS,
    CONFIG,
    D_GRID,
    P_SWEEP,
    SPARSITY,
    oracle,
    plot_utils,
    plt,
    sparsity_ticks,
):
    from matplotlib.lines import Line2D

    _fig, _ax = plt.subplots(figsize=(5.8, 3.8))
    for d in D_GRID:
        _ax.plot(SPARSITY, [oracle(p, d)["nuc"]["ex"] for p in P_SWEEP],
                marker="o", ms=4, color=COLORS[d], lw=2)
        _ax.plot(SPARSITY, [oracle(p, d)["cf"]["ex"] for p in P_SWEEP],
                marker="o", ms=4, color=COLORS[d], lw=2, ls="--")
    _ax.set_yscale("log")
    sparsity_ticks(_ax)
    _ax.set_xlabel("sparsity")
    _ax.set_ylabel("excess CE")
    handles = [Line2D([], [], color=COLORS[d], lw=2, label=f"d = {d}") for d in D_GRID] + \
              [Line2D([], [], color="0.25", lw=2, ls=ls, label=lab)
               for ls, lab in [("-", "nuclear"), ("--", "truncation")]]
    _ax.legend(handles=handles, fontsize=8, ncol=2)
    plot_utils.FIG.style_axes(_ax)
    _fig.tight_layout()
    plot_utils.save_figure(_fig, plot_utils.figure_path(CONFIG, "fig1_excess_sweep"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 2 — trained architectures vs the oracles (code RDMs)

    Trained architectures at $d = 3$ (linear and MLP pointwise; GRU and transformer as
    causal LMs over [word, attribute_1 ... attribute_L] episodes: the word is the prefix and
    its features are drawn exchangeably from the same kernel row) at a dense and a sparse
    corner, next to the truncation and nuclear codes. Excess CE is the
    evaluator's KL against the Bayes targets, so every architecture is scored identically.
    Criterion: trained excess and RDM match nuclear, as in the feature cube.
    """)
    return


@app.cell
def _(D_PANEL, P_SWEEP, TASK, np, oracle, train_inline):
    ARCH = {
        "linear": ["runner=linear", f"model_kwargs.model_config.embed_dim={D_PANEL}",
                    "training_kwargs.train_steps=6000"],
        "mlp": ["runner=mlp", f"model_kwargs.model_config.hidden_size={D_PANEL}",
                "model_kwargs.model_config.num_layers=1",
                "training_kwargs.learning_rate=0.003", "training_kwargs.train_steps=8000"],
        "rnn": ["runner=rnn", "sample_mode=pointwise", f"model_kwargs.model_config.hidden_size={D_PANEL}",
                "training_kwargs.learning_rate=0.002", "training_kwargs.train_steps=12000"],
        "transformer": ["runner=transformer", "sample_mode=pointwise",
                        f"model_kwargs.model_config.hidden_size={D_PANEL}",
                        "model_kwargs.model_config.num_attention_heads=1",
                        "model_kwargs.model_config.intermediate_size=16",
                        "training_kwargs.learning_rate=0.001", "training_kwargs.train_steps=12000"],
    }
    P_CORNERS = [P_SWEEP[1], P_SWEEP[6]]              # on the sweep grid: dense / sparse


    def trained_run(_p, ov_arch):
        """Excess CE over Bayes (the evaluator's KL) and the item code, for one architecture."""
        model, ds, evaluator, *_ = train_inline(
            TASK + [f"structure_kwargs.p={_p}"] + ov_arch)
        metrics, _ = evaluator.evaluate(model)
        return dict(ex=metrics["loss"], E=np.asarray(evaluator.item_representations(model), float))


    arch_results = {}
    for _p in P_CORNERS:
        _o = oracle(_p, D_PANEL)
        entry = {"truncation": (_o["cf"]["ex"], _o["cf"]["E"]),
                 "nuclear": (_o["nuc"]["ex"], _o["nuc"]["E"])}
        for name, ov in ARCH.items():
            r = trained_run(_p, ov)
            entry[name] = (r["ex"], r["E"])
            print(f"p={_p:g} {name:8s} excess CE={r['ex']:.4f}  "
                  f"(truncation {_o['cf']['ex']:.4f} / nuclear {_o['nuc']['ex']:.4f})")
        arch_results[_p] = entry
    return ARCH, P_CORNERS, arch_results


@app.cell
def _(ARCH, CONFIG, P_CORNERS, arch_results, plot_utils, plt):
    names = ["truncation", "nuclear"] + list(ARCH)
    _fig = plt.figure(figsize=(2.7 * len(names), 3.1 * len(P_CORNERS)))
    _gs = _fig.add_gridspec(len(P_CORNERS), len(names), hspace=0.35, wspace=0.3)
    for _row, _p in enumerate(P_CORNERS):
        for _col, n in enumerate(names):
            ex, E = arch_results[_p][n]
            _a = _fig.add_subplot(_gs[_row, _col])
            plot_utils.rdm_heatmap(_a, E, title=f"{n}\nexcess CE {ex:.3f}")
            if _col == 0:
                _a.set_ylabel(f"$p$ = {_p:g}  (sparsity {1 - _p:g})", fontsize=10)
    plot_utils.save_figure(_fig, plot_utils.figure_path(CONFIG, "fig2_geometry_rdms"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 3 — feature regimes across sparsity

    How each point is computed: build the task at this sparsity, fit the oracle at
    $d = 3$, form its logits $\hat G = E D$ ($N \times m$), and run the gauge-free least
    squares $\hat G_{xi} \sim a_i Z_{xi} + b_i + c_x$; the plotted value is
    $\hat a_i / \alpha_i$ — the fraction of feature $i$'s true log-odds swing the code
    actually delivers (1 = fully represented, 0 = dropped). Criterion (TMS): sparsifying
    should let the unimportant tail back in (graded, superposed); densifying should force
    dropout. Swings above 1 are saturation overshoot: once predictions are near one-hot, CE
    no longer penalizes overdriving the kept logits. Caveat at the extreme sparse end: with
    $N = 256$, $p = 0.005$ leaves $\approx N p = 1.3$ carrier words per feature, so
    swings there reflect the sampled lexicon, not the ensemble.
    """)
    return


@app.cell
def _(
    CONFIG,
    D_PANEL,
    M,
    P_SWEEP,
    SPARSITY,
    np,
    oracle,
    plot_utils,
    plt,
    sparsity_ticks,
):
    _fig, _ax = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
    for _a, _which in zip(_ax, ["cf", "nuc"]):
        rel = np.array([oracle(p, D_PANEL)[_which]["stats"][0] / oracle(p, D_PANEL)["alpha"]
                        for p in P_SWEEP])
        for j in range(M):
            _a.plot(SPARSITY, rel[:, j], marker="o", ms=3.5, label=f"{j + 1}")
        _a.set_title("truncation" if _which == "cf" else "nuclear", fontsize=11)
        _a.set_xlabel("sparsity")
        plot_utils.FIG.style_axes(_a)
    sparsity_ticks(_ax[0])                         # shared axis
    _ax[0].set_ylabel(r"fitted swing $\hat\alpha_i/\alpha_i$")
    _fig.legend(*_ax[0].get_legend_handles_labels(), title="feature", fontsize=9,
               title_fontsize=9.5, loc="center right")
    _fig.tight_layout(rect=(0, 0, 0.92, 1))
    plot_utils.save_figure(_fig, plot_utils.figure_path(CONFIG, "fig3_feature_regimes"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 4 — phase panel across the sweep ($d = 3$)

    Per sparsity and oracle: signal-direction Gram $\hat V \hat V^\top$ (each row of
    $\hat V$ is feature $i$'s effective direction in code space; red diagonal = own
    dimension, off-diagonal structure = interference) and swing bars colored by
    superposition (yellow = shares dimensions); dotted line = full delivery.
    """)
    return


@app.cell
def _(
    CONFIG,
    D_PANEL,
    M,
    P_SWEEP,
    feature_vectors,
    np,
    oracle,
    plot_utils,
    plt,
    superposition_measure,
):
    P_PANEL = P_SWEEP[::2]                            # every other sweep point
    BAR_MAX = 1.05 * max(float((oracle(_p, D_PANEL)[w]["stats"][0] / oracle(_p, D_PANEL)["alpha"]).max())
                         for _p in P_PANEL for w in ("nuc", "cf"))
    _fig = plt.figure(figsize=(2.35 * len(P_PANEL), 11.6))
    _gs = _fig.add_gridspec(4, len(P_PANEL), height_ratios=(1.1, 1.0, 1.1, 1.0), hspace=0.45,
                          wspace=0.35)
    cmap = plt.get_cmap("summer")                 # green = own dimension, yellow = shared
    for _col, _p in enumerate(P_PANEL):
        _o = oracle(_p, D_PANEL)
        for _row, _which in enumerate(["nuc", "cf"]):
            V, _ = feature_vectors(_o[_which]["E"], _o["Z"])
            G = V @ V.T
            aG = _fig.add_subplot(_gs[2 * _row, _col])
            aG.imshow(G, cmap="RdBu_r", vmin=-np.abs(G).max(), vmax=np.abs(G).max())
            if _row == 0:
                aG.set_title(f"sparsity {1 - _p:.3g}", fontsize=10)
            aG.set_xticks([]); aG.set_yticks([])
            aG.set_xlabel(f"excess {_o[_which]['ex']:.3f}", fontsize=9)
            if _col == 0:
                aG.set_ylabel("nuclear" if _which == "nuc" else "truncation", fontsize=11)
            a_hat = _o[_which]["stats"][0]
            sp = superposition_measure(V.T)
            aB = _fig.add_subplot(_gs[2 * _row + 1, _col])
            aB.barh(np.arange(M), a_hat / _o["alpha"],
                    color=cmap(np.clip(sp / max(sp.max(), 1e-12), 0, 1)))
            aB.invert_yaxis()
            aB.set_xlim(0, BAR_MAX)
            aB.axvline(1.0, color="0.4", lw=0.8, ls=":")
            aB.set_yticks(np.arange(M), [f"{j + 1}" for j in range(M)], fontsize=7)
            aB.set_xlabel(r"$\hat\alpha_i/\alpha_i$", fontsize=9)
            plot_utils.FIG.style_axes(aB)
    plot_utils.save_figure(_fig, plot_utils.figure_path(CONFIG, "fig4_elhage_phase"))
    return


if __name__ == "__main__":
    app.run()
