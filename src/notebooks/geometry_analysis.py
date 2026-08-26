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
        train_inline,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. The predicted geometry (closed form)

    Factor the ring's log-conditional. The translation symmetry gives a
    top Fourier pair, and the top-2 code lands on a circle.
    """)
    return


@app.cell
def _(FIGURES, graph_conditional, plot_utils, predicted_geometry):
    # pick a structure here — everything below follows from it. `cooccurrence` is part of the spec
    # (diffusion | geodesic | adjacency | heat) and flows to both the predicted geometry and the trained
    # models; a product diffuses, but e.g. a hypercube wants "heat" so its bits stay independent (analogies).
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

    The structure, the log-conditional it induces, and the closed-form embedding: the ring's
    top-2 code lands on a circle.
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
    ## 3. Train everything: the prediction across settings

    Loop over structures: train all four architectures per structure (hparams per arch from a
    ring sweep: the sequence models want longer training and a lower lr), then save that
    structure's embedding-panel figure into its own run directory. One shared point colormap
    (half of twilight, acyclic); matrix row = the code Gram, $E^{*}E^{*\top}$ predicted beside
    each learned $EE^{\top}$.
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
    ## 4. Main figure: the six settings merged

    The per-setting embedding panels tiled (3, 2): ring, grid, helix down the first column;
    hypercube, simplex, hierarchy down the second.
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Relational: modular addition, a stack of circles

    $c = (a + b) \bmod p$. Each pair $(a, b)$ is **one input token** for the readouts, so the
    conditional is the $(p^2, p)$ matrix $M[(a,b), c]$ and the oracle is a plain SVD, not a tensor
    factorization. Off the diagonal $\log p = -\infty$; we floor it, which after centering is a pure
    rescaling (a softmax temperature would be too), so the geometry does not depend on the floor.
    The sequence models read $[a, b]$ as separate tokens, and their pair code is the hidden state
    after $b$.

    Every Fourier frequency of $\mathbb{Z}_p$ carries the same singular value, so the SVD basis is
    arbitrary. Re-expressed in the DFT gauge, the predicted decoder is the frequency ladder and the
    predicted code is **five perfect circles** (frequencies 1–5, each energy 0.18, circularity 1.00)
    plus the one-dimensional Nyquist mode — not a single circle.

    Below, each model's code averaged per sum $s$, shown in its **dominant** frequency's plane;
    frequency $k$ winds $k$ times, so it visits $p/\gcd(k,p)$ positions. The models solve the task
    exactly (100% accuracy) but distribute themselves differently over the circles: the rnn commits
    to $f_2$ (energy 0.50, circularity 0.98), linear spreads over all of them, and the mlp's
    dominant component is squashed flat (circularity 0.11) because its pre-decode activation is
    post-ReLU and cannot centre into a circle. The $s$-fraction — how much of the code is a function
    of the sum at all — is 1.00 for the readouts but 0.73 / 0.32 for transformer / rnn, whose hidden
    state still carries $a$ and $b$ separately.
    """)
    return


@app.cell
def _(FIGURES, canonical_gauge, factor_logcond, fourier_basis, np, plot_utils):
    from src.code.data import relation_tensor

    P_MOD = 11                                        # prime: every frequency has full period p
    _T_rel = relation_tensor({"operation": "modular_add", "p": P_MOD}).reshape(P_MOD**2, P_MOD)
    # a deterministic relation has log p = -inf off the diagonal; floor it. Centering makes the floor
    # (and equally a softmax temperature) a pure rescaling, so the geometry does not depend on it.
    M_rel = np.where(_T_rel > 0, 0.0, -20.0)
    S_REL = (np.arange(P_MOD**2) // P_MOD + np.arange(P_MOD**2) % P_MOD) % P_MOD

    # the frequencies are degenerate, so the SVD basis is arbitrary: re-express in the DFT gauge,
    # where the decoder rows are the Fourier modes of Z_p and the code carries the matching pair
    _Lc, _E, _D, _S = factor_logcond(M_rel)
    E_gauge, D_gauge = canonical_gauge(_E, _D, fourier_basis(P_MOD))
    # every frequency carries the same singular value, so no pair is privileged: show the
    # frequency-1 cos/sin pair, the two dimensions that draw the circle (rank 2, hence the ~)
    _ROWS = slice(0, 4 * P_MOD)                       # four a-blocks: the pattern, not all 144 rows
    _fig, _ = plot_utils.plot_factorization(M_rel[_ROWS], k=2,
                                            factors=(E_gauge[_ROWS], D_gauge),
                                            item_label=r"inputs  $(a,b)$", outcome_label=r"outputs  $c$",
                                            figsize=(16, 5.6), embedding_width=0.1,
                                            decoder_height=0.15, conditional_aspect=0.5)
    plot_utils.save_figure(_fig, FIGURES / "relational_factorization.png")
    return M_rel, P_MOD, S_REL


@app.cell
def _(M_rel, P_MOD, S_REL, circulant_fraction, cka, predicted_geometry):
    from src.scripts.relational_metrics import pair_codes

    # every arch reads (a, b) as two tokens from one p-row embedding table; the paired variant, which
    # gets a free p^2-row table instead, is kept only as the control that loses the ring
    ARCHS_REL = [("transformer", False), ("linear", True), ("mlp", False), ("rnn", False)]
    KS_REL = list(range(1, P_MOD // 2 + 1))          # k and p-k span the same real plane

    E_rel = predicted_geometry(M_rel)
    rel_codes = {}
    for _arch, _paired in ARCHS_REL:
        _X, _kl, _acc = pair_codes(_arch, wd=0.0, separate=not _paired)
        rel_codes[_arch] = _X
        print(f"{_arch:12s} paired={str(_paired):5s} KL={_kl:.4f} acc={_acc:.2f} "
              f"CKA={cka(_X, E_rel):.3f}  circulant={circulant_fraction(_X, S_REL):.2f}")

    return E_rel, KS_REL, pair_codes, rel_codes


@app.cell
def _(
    E_rel,
    FIGURES,
    KS_REL,
    P_MOD,
    S_REL,
    circulant_fraction,
    class_means,
    frequency_plane,
    group_fraction,
    np,
    pair_codes,
    plot_utils,
):
    # Appendix figure: the degenerate case. Columns are the predicted code and one representative
    # run per arch (median seed by circulance). "paired" gives the mlp a free p^2-row embedding table
    # instead of two tokens from a shared p-row one. A flat spectrum has no argmax, so ties take f1.
    REL_PANELS = [("transformer", True, 0), ("linear", False, 1), ("rnn", True, 3),
                  ("mlp separate", True, 1), ("mlp paired", False, 3)]

    def dominant_frequency(X, tol=0.02):
        """Frequency carrying most energy, or 1 when the spectrum is flat (every irrep equal)."""
        en = {k: frequency_plane(X, S_REL, k)[1] for k in KS_REL}
        return (1 if max(en.values()) - min(en.values()) < tol else max(en, key=en.get)), en

    _panels = [("predicted", np.asarray(E_rel, float))] + [
        (nm, pair_codes("mlp" if nm.startswith("mlp") else nm, wd=0.0, separate=sep, seed=sd)[0])
        for nm, sep, sd in REL_PANELS]

    fig_rel = plot_utils.plt.figure(figsize=(2.75 * len(_panels), 4.9))
    _gs = fig_rel.add_gridspec(2, len(_panels), hspace=0.05, wspace=0.3, height_ratios=[1.15, 1])
    for _j, (_nm, _X) in enumerate(_panels):
        _k, _en = dominant_frequency(_X)
        _xy, _, _c = frequency_plane(_X, S_REL, _k)
        _m, _ = class_means(_xy, S_REL)

        _ax = fig_rel.add_subplot(_gs[0, _j])
        # join the means by angle, not by s: at frequency k the s-order traces a {p/k} star,
        # which reads as noise. Whether the winding matches s is what circularity already reports.
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

    return (REL_PANELS,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Paired vs separate inputs

    The task is a delta over outcomes, so the loss constrains only which sum each pair predicts, never
    how the sums sit relative to one another: for any invertible $A$, the code $hA$ with decoder
    $WA^{-\top}$ computes the same function. Geometry is therefore fixed by the architecture and the
    implicit bias, not by the task.

    Giving the readout one token per $(a,b)$ pair hands it a free $p^2$-row table, so nothing ties the
    code of $(a,b)$ to that of $(a{+}1,b)$ and the ring is lost. Sharing one $p$-row table across both
    slots forces $p$ embeddings to cover $p^2$ combinations, and the ring returns.
    """)
    return


@app.cell
def _(
    FIGURES,
    KS_REL,
    P_MOD,
    S_REL,
    circulant_fraction,
    class_means,
    frequency_plane,
    np,
    pair_codes,
    plot_utils,
):
    # each run at its own dominant frequency. Joining the means in order s = 0..p-1 traces a {p/k}
    # star at frequency k, since that character winds k times: the point positions carry the claim.
    _RUNS = [("mlp", True, 0), ("mlp", False, 0), ("mlp", False, 2), ("rnn", False, 0)]

    fig_sep, _axes = plot_utils.plt.subplots(1, len(_RUNS), figsize=(3.2 * len(_RUNS), 4.0))
    for _ax, (_arch, _paired, _seed) in zip(_axes, _RUNS):
        _X = pair_codes(_arch, wd=0.0, separate=not _paired, seed=_seed)[0]
        _k = max(KS_REL, key=lambda k: frequency_plane(_X, S_REL, k)[1])
        _xy, _e, _c = frequency_plane(_X, S_REL, _k)
        _m, _ = class_means(_xy, S_REL)
        _ax.plot(*np.r_[_m, _m[:1]].T, color="0.85", lw=1.0, zorder=1)
        _ax.scatter(_xy[:, 0], _xy[:, 1], c=S_REL, cmap=plot_utils.FIG.cyclic_cmap,
                    s=12, alpha=0.35, zorder=2)
        _ax.scatter(_m[:, 0], _m[:, 1], c=np.arange(P_MOD), cmap=plot_utils.FIG.cyclic_cmap,
                    s=70, zorder=4, edgecolor="white", lw=0.8)
        _ax.set_title(f"{_arch} {'paired' if _paired else 'separate'}, seed {_seed}\n"
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
    ### Is the paired mlp's circle just hidden by the projection?

    No. Below are its raw penultimate activations — no readout projection, no frequency fitting — next
    to the radii of every arch's class means. A code on a ring has all $p$ means at one radius, so the
    coefficient of variation of the radii tests the ring without assuming any cyclic alignment (and so
    carries over to the non-group operations).
    """)
    return


@app.cell
def _(
    FIGURES,
    P_MOD,
    S_REL,
    class_means,
    np,
    plot_utils,
    rel_codes,
    train_inline,
):
    import torch

    # raw penultimate activations of the paired mlp: plain PCA, nothing fitted
    _mp, _dsp, _, _, _ = train_inline(["setting=relational", f"operation_kwargs.p={P_MOD}",
        "runner=mlp", "++pair_inputs=true", "model_kwargs.model_config.hidden_size=128",
        "training_kwargs.weight_decay=0.0"])
    _idp = torch.arange(P_MOD**2, device=next(_mp.parameters()).device)[:, None]
    with torch.no_grad():
        RAW_PAIRED = _mp.mlp[:-1](_mp.embedding(_idp).flatten(1)).float().cpu().numpy()
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

    # every arch's class-mean radii, normalised: a ring is a flat line at 1
    _ax = _axs[3]
    for _nm, _X in [("transformer", rel_codes["transformer"]), ("rnn", rel_codes["rnn"]),
                    ("mlp separate", rel_codes["mlp"]), ("mlp paired", RAW_PAIRED)]:
        _r = np.linalg.norm(class_means(_X - _X.mean(0), S_REL)[0], axis=1)
        _ax.plot(np.sort(_r / _r.mean()), "o-", ms=4, lw=1.2,
                 label=f"{_nm} (CV {_r.std()/_r.mean():.2f})")
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
    ### What the task pins, and what it does not

    Both measures need a null, because neither starts at zero. For $p^2$ random codes of the same rank
    the s-fraction sits at $1/p$, but circulance already averages $0.53$ — 11 arbitrary points in 10
    dimensions are half-circulant by accident.

    Read against those bands, the task forces the **grouping** (every arch reaches $s$-fraction well
    above chance) but not the **arrangement**: the paired mlp sits *below* the circulance null while
    every shared-embedding arch sits at the ceiling. The right-hand panel asks whether the choice of
    irrep is uniform, as the flat spectrum implies it should be.
    """)
    return


@app.cell
def _(
    FIGURES,
    KS_REL,
    P_MOD,
    REL_PANELS,
    S_REL,
    circulant_fraction,
    frequency_plane,
    group_fraction,
    np,
    pair_codes,
    plot_utils,
):
    _SEEDS = (0, 1, 2, 3)
    _NULL_N = 300
    _rng_null = np.random.default_rng(0)
    _null = [(group_fraction(_Z, S_REL), circulant_fraction(_Z, S_REL))
             for _Z in (_rng_null.normal(size=(P_MOD**2, P_MOD - 1)) for _ in range(_NULL_N))]
    _null = np.array(_null)

    fig_null, _axm = plot_utils.plt.subplots(1, 3, figsize=(14.5, 3.9))
    for _ax, _mi, _lbl in zip(_axm[:2], (0, 1), ("$s$-fraction", "circulant fraction")):
        _lo, _hi = np.percentile(_null[:, _mi], [5, 95])
        _ax.axhspan(_lo, _hi, color=plot_utils.FIG.colors["orange"], alpha=0.15, zorder=0)
        _ax.axhline(_null[:, _mi].mean(), ls="--", lw=1.0,
                    color=plot_utils.FIG.colors["orange"], zorder=1)
        for _x, (_nm, _sep, _) in enumerate(REL_PANELS):
            _v = [(group_fraction(_X, S_REL), circulant_fraction(_X, S_REL)) for _X in
                  (pair_codes("mlp" if _nm.startswith("mlp") else _nm, wd=0.0, separate=_sep, seed=_s)[0]
                   for _s in _SEEDS)]
            _ax.scatter([_x] * len(_v), [t[_mi] for t in _v], s=42, color="0.25", zorder=3)
        _ax.set_xticks(range(len(REL_PANELS)))
        _ax.set_xticklabels([n for n, _, _ in REL_PANELS], rotation=30, ha="right", fontsize=9)
        _ax.set_ylim(0, 1.05); _ax.set_ylabel(_lbl, fontsize=11)
        _ax.set_title(f"{_lbl} (band = random-code null)", fontsize=11)
        plot_utils.FIG.style_axes(_ax)

    # which irrep gets chosen, over 16 seeds: the flat spectrum implies no preference
    _ax = _axm[2]
    for _nm, _sep, _off in [("rnn", True, -0.18), ("mlp separate", True, 0.18)]:
        _ks = [max(KS_REL, key=lambda k: frequency_plane(
                  pair_codes("mlp" if _nm.startswith("mlp") else _nm, wd=0.0, separate=_sep, seed=_s)[0],
                  S_REL, k)[1]) for _s in range(16)]
        _ax.bar(np.array(KS_REL) + _off, [_ks.count(k) for k in KS_REL], width=0.34, label=_nm)
    _ax.axhline(16 / len(KS_REL), ls="--", lw=1.2, color=plot_utils.FIG.colors["orange"])
    _ax.set_xticks(KS_REL); _ax.set_xlabel("dominant frequency $k$", fontsize=11)
    _ax.set_ylabel("runs (of 16)", fontsize=11)
    _ax.set_title("which irrep is chosen (dashed = uniform)", fontsize=11)
    _ax.legend(fontsize=9, frameon=False)
    plot_utils.FIG.style_axes(_ax)
    fig_null.subplots_adjust(wspace=0.32, bottom=0.24)
    plot_utils.save_figure(fig_null, FIGURES / "relational_metrics_vs_null.png")

    return


if __name__ == "__main__":
    app.run()
