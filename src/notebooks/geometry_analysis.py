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

    The geometry a network learns is predictable in closed form from the structure of the task.
    """)
    return


@app.cell
def _(mo):
    import os, sys
    import numpy as np
    import torch

    # Project root, independent of the working directory (a headless run can start outside
    # the repo). chdir so Hydra resolves conf/, and add it to sys.path.
    ROOT = mo.notebook_dir().parent.parent    # src/notebooks -> <repo>
    os.chdir(ROOT); sys.path.insert(0, str(ROOT))

    from src.code.data import graph_conditional
    from src.code.geometry import (predicted_geometry, factor_logcond, canonical_gauge,
                                   fourier_basis, circulant_fraction, class_means,
                                   frequency_plane, group_fraction)
    from src.code.eval import cka, rsa
    from src.code.train import train_inline, graph_overrides
    from src.code import plot_utils           # importing this applies the house plotting style

    FIGURES = ROOT / "outputs" / "figures"     # one flat dir: {structure}_{figure}.png
    FIGURES.mkdir(parents=True, exist_ok=True)
    return (
        FIGURES,
        canonical_gauge,
        circulant_fraction,
        cka,
        class_means,
        factor_logcond,
        fourier_basis,
        frequency_plane,
        graph_conditional,
        graph_overrides,
        group_fraction,
        np,
        plot_utils,
        predicted_geometry,
        torch,
        train_inline,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Factoring a task's conditional

    The SVD of the row-centred log-conditional splits into a code $E^{*}$ and a decoder $D^{*}$.
    """)
    return


@app.cell
def _(FIGURES, graph_conditional, plot_utils, predicted_geometry):
    # cooccurrence (diffusion | geodesic | adjacency | heat) flows to both the prediction and the
    # training; a hypercube wants "heat" so its bits stay independent.nalogies).
    spec = {"structure": "ring", "cooccurrence": "diffusion", "n": 30}
    L = graph_conditional(spec)
    E_pred = predicted_geometry(L)
    fig_fact, _ = plot_utils.plot_factorization(L)
    plot_utils.save_figure(fig_fact, FIGURES / f"{spec['structure']}_factorization.png")
    return E_pred, L, spec


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. The predicted geometry

    The structure, the log-conditional it induces, and the closed-form code: for the ring, a circle.
    """)
    return


@app.cell
def _(E_pred, FIGURES, L, plot_utils, spec):
    fig_pred = plot_utils.plot_predicted_geometry(spec, L, E_pred)
    plot_utils.save_figure(fig_pred, FIGURES / f"{spec['structure']}_predicted_geometry.png")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Trained geometry across settings

    Predicted code Gram $E^{*}E^{*\top}$ beside each learned $EE^{\top}$, per architecture.
    """)
    return


@app.cell
def _(
    FIGURES,
    cka,
    graph_conditional,
    graph_overrides,
    plot_utils,
    predicted_geometry,
    train_inline,
):
    ARCHS = ["transformer", "linear", "mlp", "rnn"]

    SPECS = [
        {"structure": "ring", "cooccurrence": "diffusion", "n": 30},
        {"structure": "grid", "cooccurrence": "heat", "n": 5, "m": 5},   # heat: additive over the chain factors, so the lattice sides stay straight
        {"structure": "product", "cooccurrence": "heat", "view": "3d",   # helix: line x circle
         "factors": [{"structure": "chain", "n": 8}, {"structure": "ring", "n": 8}]},
        {"structure": "hypercube", "cooccurrence": "heat", "n": 3},
        {"structure": "complete", "cooccurrence": "diffusion", "n": 4},
        {"structure": "tree", "n": 2, "k": 3},
    ]
    SETTING_CMAP = plot_utils.FIG.item_cmap        # twilight minus its grey ends (house item map)
    figs_settings = {}
    for _sp in SPECS:
        _L = graph_conditional(_sp)
        _reps, _cfg = {}, None
        for _arch in ARCHS:
            # runner defaults: readouts pointwise on the exact rows, LMs on [start, item, outcome]
            _model, _ds, _ev, _hist, _cfg = train_inline(graph_overrides(_sp) + [f"runner={_arch}"])
            _reps[_arch] = _ev.item_representations(_model)
            print(f"{_sp['structure']:10s} {_arch:12s} "
                  f"CKA={cka(_reps[_arch], predicted_geometry(_ds.conditional)):.3f}")
        _fig = plot_utils.plot_embedding_panels(predicted_geometry(_L), _reps, spec=_sp,
                                                cmap=SETTING_CMAP)
        _key = "helix" if _sp["structure"] == "product" else _sp["structure"]
        plot_utils.save_figure(_fig, FIGURES / f"{_key}_embedding_panels.png")
        figs_settings[_key] = _fig
    return (figs_settings,)


@app.cell
def _(figs_settings, mo):
    mo.ui.tabs(figs_settings)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### The six settings merged
    """)
    return


@app.cell
def _(FIGURES, plot_utils):
    import matplotlib.image as mpimg

    # letters follow the columns: (a-c) ring, grid, helix; (d-f) hypercube, simplex, hierarchy
    _grid = [["ring", "hypercube"], ["grid", "complete"], ["helix", "tree"]]
    _imgs = {_n: mpimg.imread(FIGURES / f"{_n}_embedding_panels.png") for _n in sum(_grid, [])}
    _ratios = [max(_imgs[_n].shape[0] / _imgs[_n].shape[1] for _n in _row) for _row in _grid]
    fig_main = plot_utils.plt.figure(figsize=(21, 10.5 * sum(_ratios) + 2.5))
    _gs = fig_main.add_gridspec(3, 2, height_ratios=_ratios, hspace=0.22, wspace=0.08,
                                left=0.02, right=0.98, top=0.96, bottom=0.03)
    for _i, _row in enumerate(_grid):
        for _j, _n in enumerate(_row):
            _ax = fig_main.add_subplot(_gs[_i, _j])
            _ax.imshow(_imgs[_n]); _ax.axis("off")
            _ax.set_title(f"({'abcdef'[_i + 3 * _j]})", loc="left", fontsize=30, pad=14)
    plot_utils.save_figure(fig_main, FIGURES / "geometry_predictions_main.png", dpi=200)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Relational settings

    $c = a \ast b$, so the conditional is a tensor $T[a,b,c]$ and the code is read per input pair.
    These operations have a decaying spectrum, so CKA against $E^{*}$ measures the geometry.
    """)
    return


@app.cell
def _(torch, train_inline):
    from src.code.analysis_utils import readout_basis

    # every knob (width, decay, input format) lives in the runner preset, so a run is named by
    # the relation and the runner alone — nothing here can drift from what the CLI trained
    RUNNERS_REL = ["rel_transformer", "rel_rnn", "rel_mlp", "rel_linear_paired", "rel_mlp_paired"]
    OPS_REL = ["maximum", "set_intersection", "entailment", "standard_add", "standard_mult",
               "averaging"]

    def relational_code(relation, runner, seed=0):
        """Readout-projected code of every input tuple for a cached run, with its dataset."""
        _model, _ds, _, _, _ = train_inline(
            [f"runner={runner}", f"relation={relation}", f"random_seed={seed}"])
        with torch.no_grad():
            _X = _model.input_reps(_ds.input_tuples()).float().cpu().numpy()
        return _X @ readout_basis(_model, _ds).T, _ds


    return OPS_REL, RUNNERS_REL, readout_basis, relational_code


@app.cell
def _(FIGURES, OPS_REL, RUNNERS_REL, cka, np, plot_utils, predicted_geometry,
      relational_code):
    # Grams down the rows, one relation each: the predicted code beside every architecture's.
    # These relations have decaying spectra, so the relative radii are predicted too and CKA
    # against E* is the measure, as in the static settings.
    _rng_null = np.random.default_rng(0)

    fig_rel4, _axes = plot_utils.plt.subplots(
        len(OPS_REL), len(RUNNERS_REL) + 1,
        figsize=(2.15 * (len(RUNNERS_REL) + 1), 2.35 * len(OPS_REL)))
    for _i, _op in enumerate(OPS_REL):
        _codes, _E = {}, None
        for _v in RUNNERS_REL:
            _X, _ds = relational_code(_op, _v)
            _codes[_v] = _X
            if _E is None:                       # per-pair oracle: SVD of the flattened conditional
                _T = np.asarray(_ds.conditional, float)
                _M = _T.reshape(-1, _T.shape[-1])
                _L = np.where(_M > 0, np.log(np.maximum(_M, 1e-12)), -20.0)
                _E = np.asarray(predicted_geometry(_L), float)
                _null = np.mean([cka(_rng_null.normal(size=_X.shape), _E) for _ in range(20)])

        plot_utils.gram_heatmap(_axes[_i, 0], _E, title=r"$E^{*}E^{*\top}$" if _i == 0 else "")
        _axes[_i, 0].set_ylabel(f"{_op}\nnull {_null:.2f}", fontsize=9)
        for _j, _v in enumerate(RUNNERS_REL, start=1):
            plot_utils.gram_heatmap(_axes[_i, _j], _codes[_v],
                                    title=_v.removeprefix("rel_") if _i == 0 else "")
            _axes[_i, _j].set_xlabel(f"CKA {cka(_codes[_v], _E):.2f}", fontsize=9)
    fig_rel4.subplots_adjust(hspace=0.35, wspace=0.15)
    plot_utils.save_figure(fig_rel4, FIGURES / "relational_relations_merged.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Appendix. Modular addition

    A group's spectrum is flat, so the prediction fixes the code only up to a choice of frequency.
    """)
    return


@app.cell
def _(FIGURES, canonical_gauge, factor_logcond, fourier_basis, np, plot_utils):
    from src.code.data import relation_tensor

    P_MOD = 11                                        # prime: every frequency has full period p
    _T_rel = relation_tensor({"relation": "modular_add", "p": P_MOD}).reshape(P_MOD**2, P_MOD)
    # log p = -inf off the diagonal; floor it. Centering makes the floor a pure rescaling.
    M_rel = np.where(_T_rel > 0, 0.0, -20.0)
    S_REL = (np.arange(P_MOD**2) // P_MOD + np.arange(P_MOD**2) % P_MOD) % P_MOD

    # degenerate frequencies leave the SVD basis arbitrary: re-express in the DFT gauge
    _Lc, _E, _D, _S = factor_logcond(M_rel)
    E_gauge, D_gauge = canonical_gauge(_E, _D, fourier_basis(P_MOD))
    # no frequency is privileged: show the f1 cos/sin pair, the two dimensions drawing the circle
    _ROWS = slice(0, 4 * P_MOD)                       # four a-blocks: the pattern, not all 144 rows
    _fig, _ = plot_utils.plot_factorization(M_rel[_ROWS], k=2,
                                            factors=(E_gauge[_ROWS], D_gauge),
                                            item_label=r"inputs  $(a,b)$", outcome_label=r"outputs  $c$",
                                            figsize=(16, 5.6), embedding_width=0.1,
                                            decoder_height=0.15, conditional_aspect=0.5)
    plot_utils.save_figure(_fig, FIGURES / "relational_factorization.png")
    return M_rel, P_MOD, S_REL


@app.cell
def _(
    M_rel,
    P_MOD,
    RUNNERS_REL,
    S_REL,
    circulant_fraction,
    group_fraction,
    predicted_geometry,
    relational_code,
):
    SEEDS_REL = range(5)
    ARCHS_REL = RUNNERS_REL                          # same presets, on the default relation
    KS_REL = list(range(1, P_MOD // 2 + 1))          # k and p-k span the same real plane

    E_rel = predicted_geometry(M_rel)
    rel_codes = {}                                   # (runner, seed) -> readout-projected pair code
    for _r in ARCHS_REL:
        for _s in SEEDS_REL:
            rel_codes[_r, _s] = relational_code("modular_add", _r, seed=_s)[0]
            print(f"{_r:20s} seed {_s}  "
                  f"circulant={circulant_fraction(rel_codes[_r, _s], S_REL):.2f}  "
                  f"s-frac={group_fraction(rel_codes[_r, _s], S_REL):.2f}", flush=True)

    return ARCHS_REL, E_rel, KS_REL, SEEDS_REL, rel_codes


@app.cell
def _(
    ARCHS_REL,
    E_rel,
    FIGURES,
    KS_REL,
    P_MOD,
    SEEDS_REL,
    S_REL,
    circulant_fraction,
    class_means,
    frequency_plane,
    group_fraction,
    np,
    plot_utils,
    rel_codes,
):
    # One representative run per arch: the median seed by circulance, not the best or the worst.
    # A flat spectrum has no argmax, so ties take f1.
    def dominant_frequency(X, tol=0.02):
        """Frequency carrying most of the energy, or 1 when every irrep carries the same."""
        _en = {k: frequency_plane(X, S_REL, k)[1] for k in KS_REL}
        return (1 if max(_en.values()) - min(_en.values()) < tol else max(_en, key=_en.get)), _en

    def median_seed(label):
        """The seed whose circulance is the median over the sweep."""
        _o = sorted(SEEDS_REL, key=lambda s: circulant_fraction(rel_codes[label, s], S_REL))
        return _o[len(_o) // 2]

    _panels = ([("predicted", np.asarray(E_rel, float))]
               + [(_l.removeprefix('rel_'), rel_codes[_l, median_seed(_l)]) for _l in ARCHS_REL])

    fig_rel = plot_utils.plt.figure(figsize=(2.75 * len(_panels), 4.9))
    _gs = fig_rel.add_gridspec(2, len(_panels), hspace=0.05, wspace=0.3, height_ratios=[1.15, 1])
    for _j, (_nm, _X) in enumerate(_panels):
        _k, _en = dominant_frequency(_X)
        _xy, _, _c = frequency_plane(_X, S_REL, _k)
        _m, _ = class_means(_xy, S_REL)

        _ax = fig_rel.add_subplot(_gs[0, _j])
        # join the means by angle, not by s: at frequency k the s-order traces a {p/k} star
        _o = np.argsort(np.arctan2(_m[:, 1], _m[:, 0]))
        _ax.plot(*np.r_[_m[_o], _m[_o][:1]].T, color="0.85", lw=1.0, zorder=1)
        _ax.scatter(_xy[:, 0], _xy[:, 1], c=S_REL, cmap=plot_utils.FIG.cyclic_cmap,
                    s=9, alpha=0.22, zorder=2)
        _ax.scatter(_m[:, 0], _m[:, 1], c=np.arange(P_MOD), cmap=plot_utils.FIG.cyclic_cmap,
                    s=75, zorder=4, edgecolor="white", lw=0.8)
        _ax.set_title(f"{_nm}\n" + rf"$f_{_k}$: circularity {_c:.2f}", fontsize=11)
        _ax.set_aspect("equal"); _ax.margins(0.15)
        plot_utils.FIG.clean(_ax); _ax.set_anchor("N")

        _axb = fig_rel.add_subplot(_gs[1, _j])
        _axb.bar(KS_REL, [_en[k] for k in KS_REL], color="0.35", width=0.62)
        _axb.axhline(2 / (P_MOD - 1), ls="--", lw=1.2, color=plot_utils.FIG.colors["orange"])
        _axb.set_xticks(KS_REL); _axb.set_ylim(0, 1.0)
        _axb.set_xlabel("frequency $k$", fontsize=11)
        if _j == 0:
            _axb.set_ylabel("energy", fontsize=11)
        _axb.set_title(f"$s$-frac {group_fraction(_X, S_REL):.2f}, "
                       f"circulant {circulant_fraction(_X, S_REL):.2f}", fontsize=10)
        plot_utils.FIG.style_axes(_axb)
    plot_utils.save_figure(fig_rel, FIGURES / "relational_merged.png")

    return dominant_frequency, median_seed


@app.cell
def _(mo):
    mo.md(r"""
    ### Paired vs separate inputs
    """)
    return


@app.cell
def _(
    FIGURES,
    P_MOD,
    S_REL,
    circulant_fraction,
    class_means,
    dominant_frequency,
    frequency_plane,
    median_seed,
    np,
    plot_utils,
    rel_codes,
):
    # paired vs separate inputs, each at its dominant frequency. The paired readout gets a free
    # p^2-row table; the rest compose a and b from one shared p-row table.
    _RUNS = [("rel_mlp_paired", 0), ("rel_mlp", 0),
             ("rel_mlp", median_seed("rel_mlp")), ("rel_rnn", 0)]

    fig_sep, _axes = plot_utils.plt.subplots(1, len(_RUNS), figsize=(3.2 * len(_RUNS), 4.0))
    for _ax, (_lbl, _seed) in zip(_axes, _RUNS):
        _X = rel_codes[_lbl, _seed]
        _k, _ = dominant_frequency(_X)
        _xy, _e, _c = frequency_plane(_X, S_REL, _k)
        _m, _ = class_means(_xy, S_REL)
        _o = np.argsort(np.arctan2(_m[:, 1], _m[:, 0]))
        _ax.plot(*np.r_[_m[_o], _m[_o][:1]].T, color="0.85", lw=1.0, zorder=1)
        _ax.scatter(_xy[:, 0], _xy[:, 1], c=S_REL, cmap=plot_utils.FIG.cyclic_cmap,
                    s=10, alpha=0.25, zorder=2)
        _ax.scatter(_m[:, 0], _m[:, 1], c=np.arange(P_MOD), cmap=plot_utils.FIG.cyclic_cmap,
                    s=70, zorder=4, edgecolor="white", lw=0.8)
        _ax.set_title(f"{_lbl.removeprefix('rel_')}, seed {_seed}\n"
                      rf"$f_{_k}$ (energy {_e:.2f}), circularity {_c:.2f}" + "\n"
                      f"circulant {circulant_fraction(_X, S_REL):.2f}", fontsize=10)
        _ax.set_aspect("equal"); _ax.margins(0.18)
        plot_utils.FIG.clean(_ax); _ax.set_anchor("N")
    fig_sep.subplots_adjust(top=0.78, wspace=0.25)
    plot_utils.save_figure(fig_sep, FIGURES / "relational_paired_vs_separate.png")

    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Is the ring hidden by the projection?
    """)
    return


@app.cell
def _(
    ARCHS_REL,
    FIGURES,
    P_MOD,
    S_REL,
    class_means,
    np,
    plot_utils,
    rel_codes,
    torch,
    train_inline,
):
    # the paired mlp's raw penultimate activations: no readout projection, no frequency fitting
    _mp, _dsp, _, _, _ = train_inline(
        ["runner=rel_mlp_paired", "relation=modular_add", "random_seed=0"])
    with torch.no_grad():
        RAW_PAIRED = _mp.input_reps(_dsp.input_tuples()).float().cpu().numpy()
    _Xc = RAW_PAIRED - RAW_PAIRED.mean(0)
    _Vt = np.linalg.svd(_Xc, full_matrices=False)[2]
    _share = np.linalg.svd(_Xc, compute_uv=False) ** 2
    _share = _share / _share.sum()
    _mc, _ = class_means(_Xc, S_REL)

    fig_raw, _axs = plot_utils.plt.subplots(1, 4, figsize=(15, 4.2))
    for _ax, (_i, _j) in zip(_axs[:3], [(0, 1), (1, 2), (2, 3)]):
        _P2, _M2 = _Xc @ _Vt[[_i, _j]].T, _mc @ _Vt[[_i, _j]].T
        _ax.plot(*np.r_[_M2, _M2[:1]].T, color="0.85", lw=1.0, zorder=1)
        _ax.scatter(_P2[:, 0], _P2[:, 1], c=S_REL, cmap=plot_utils.FIG.cyclic_cmap, s=13, alpha=0.35)
        _ax.scatter(_M2[:, 0], _M2[:, 1], c=np.arange(P_MOD), cmap=plot_utils.FIG.cyclic_cmap,
                    s=75, zorder=4, edgecolor="white", lw=0.8)
        _ax.set_title(f"PC{_i+1} vs PC{_j+1}  ({_share[_i]:.0%}, {_share[_j]:.0%})", fontsize=11)
        _ax.set_aspect("equal"); plot_utils.FIG.clean(_ax)

    # class-mean radii: a code on a ring is a flat line at 1
    _ax = _axs[3]
    for _l in ARCHS_REL:
        _r = np.linalg.norm(class_means(rel_codes[_l, 0] - rel_codes[_l, 0].mean(0), S_REL)[0], axis=1)
        _ax.plot(np.sort(_r / _r.mean()), "o-", ms=4, lw=1.2,
                 label=f"{_l.removeprefix('rel_')} (CV {_r.std()/_r.mean():.2f})")
    _ax.axhline(1.0, ls="--", lw=1.0, color="0.6")
    _ax.set_xlabel("class, sorted by radius", fontsize=11)
    _ax.set_ylabel("radius / mean", fontsize=11)
    _ax.set_title("class-mean radii", fontsize=11)
    _ax.legend(fontsize=8, frameon=False)
    plot_utils.FIG.style_axes(_ax)
    fig_raw.subplots_adjust(top=0.86, wspace=0.3)
    plot_utils.save_figure(fig_raw, FIGURES / "relational_paired_mlp_raw.png")

    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Against the null
    """)
    return


@app.cell
def _(
    ARCHS_REL,
    FIGURES,
    KS_REL,
    P_MOD,
    SEEDS_REL,
    S_REL,
    circulant_fraction,
    dominant_frequency,
    group_fraction,
    np,
    plot_utils,
    rel_codes,
):
    # Neither measure starts at zero, so every panel carries its random-code null.
    _NULL = np.array([(group_fraction(_Z, S_REL), circulant_fraction(_Z, S_REL))
                      for _Z in (np.random.default_rng(_i).normal(size=(P_MOD**2, P_MOD - 1))
                                 for _i in range(300))])
    _LABELS = [l.removeprefix('rel_') for l in ARCHS_REL]   # display only; key by runner

    fig_null, _axm = plot_utils.plt.subplots(1, 3, figsize=(14.5, 3.9))
    for _ax, _mi, _lbl in zip(_axm[:2], (0, 1), ("$s$-fraction", "circulant fraction")):
        _lo, _hi = np.percentile(_NULL[:, _mi], [5, 95])
        _ax.axhspan(_lo, _hi, color=plot_utils.FIG.colors["orange"], alpha=0.15, zorder=0)
        _ax.axhline(_NULL[:, _mi].mean(), ls="--", lw=1.0,
                    color=plot_utils.FIG.colors["orange"], zorder=1)
        for _x, _l in enumerate(ARCHS_REL):
            _v = [(group_fraction(rel_codes[_l, _s], S_REL),
                   circulant_fraction(rel_codes[_l, _s], S_REL)) for _s in SEEDS_REL]
            _ax.scatter([_x] * len(_v), [t[_mi] for t in _v], s=42, color="0.25", zorder=3)
        _ax.set_xticks(range(len(_LABELS)))
        _ax.set_xticklabels(_LABELS, rotation=30, ha="right", fontsize=9)
        _ax.set_ylim(0, 1.05); _ax.set_ylabel(_lbl, fontsize=11)
        _ax.set_title(f"{_lbl} (band = random-code null)", fontsize=11)
        plot_utils.FIG.style_axes(_ax)

    # which irrep each run picks: a flat spectrum implies no preference
    _ax = _axm[2]
    for _l, _off in [("rel_rnn", -0.18), ("rel_mlp", 0.18)]:
        _ks = [dominant_frequency(rel_codes[_l, _s])[0] for _s in SEEDS_REL]
        _ax.bar(np.array(KS_REL) + _off, [_ks.count(k) for k in KS_REL], width=0.34,
                label=_l.removeprefix("rel_"))
    _ax.axhline(len(SEEDS_REL) / len(KS_REL), ls="--", lw=1.2, color=plot_utils.FIG.colors["orange"])
    _ax.set_xticks(KS_REL); _ax.set_xlabel("dominant frequency $k$", fontsize=11)
    _ax.set_ylabel(f"runs (of {len(SEEDS_REL)})", fontsize=11)
    _ax.set_title("which irrep is chosen (dashed = uniform)", fontsize=11)
    _ax.legend(fontsize=9, frameon=False)
    plot_utils.FIG.style_axes(_ax)
    fig_null.subplots_adjust(wspace=0.32, bottom=0.24)
    plot_utils.save_figure(fig_null, FIGURES / "relational_metrics_vs_null.png")

    return


if __name__ == "__main__":
    app.run()
