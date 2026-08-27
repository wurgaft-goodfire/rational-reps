import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Predicting the geometry a rank-limited softmax learns

    $N$ words, each a bundle of $m$ features drawn i.i.d. Bernoulli($p$) in $\{0,1\}$.
    Outcomes are the $m$ feature tokens, $\log p(i \mid x) = \mathrm{log\,softmax}_i(\alpha_i Z_{xi})$
    with an importance ladder $\alpha_i = s \cdot i^{-\beta}$. The model is a rank-$d$ softmax
    readout: a code $a_x \in \mathbb{R}^d$ per word and a decoder $D_i \in \mathbb{R}^d$ per
    feature, trained by SGD on cross-entropy. With $d \ll m$ the model is in superposition.

    The question is not *how well* it does but *what geometry* it picks. The claim is that
    the geometry is a deterministic function of $(N, m, p, \alpha, d)$, computable in closed
    form with no training. Full derivation in `notes/head_tail_derivation.md`; the pipeline is
    `src/notebooks/_predicted_geometry.py`.

    **The prediction.** Write $a_x = \sum_k w_k (z_{xk} - p)$, so each feature owns a write
    vector $w_k$. Stationarity of CE gives a water-filling law: a level $\mu$ set by the budget,

    $$\mu\, d \;=\; \sqrt{Npq}\,\frac{\bar Q}{Q_1}\sum_i \|w_i\|^2, \qquad
      \|w_i\|^2 \;=\; \frac{A\,(\alpha_i - \mu)^+}{\sqrt{Np(1-p)}},$$

    features below $\mu$ get nothing. Directions minimise interference
    $\sum_{i \neq k}\langle w_i, w_k\rangle^2$ at those fixed norms. The decoder is then the
    convex CE optimum given the code. One constant, $A$, is fitted per budget $d$; everything
    else comes from the task.

    **Gauge.** A softmax ignores per-word shifts and the factorisation $AD^\top$ is invariant
    under $O(d)$. Every comparison here row-centres the logit matrix, splits its SVD balanced
    ($A = U\sqrt{S}$, $D^\top = \sqrt{S}V^\top$ — the split gradient descent converges to), and
    scores with CKA, which is $O(d)$-invariant.
    """)
    return


@app.cell
def _():
    import os, sys
    import numpy as np
    import marimo as mo
    from itertools import combinations
    from scipy.special import logsumexp

    ROOT = mo.notebook_dir().parent.parent    # src/notebooks -> <repo>
    os.chdir(ROOT); sys.path.insert(0, str(ROOT))
    from src.code.data import feature_cube
    from src.code.eval import cka
    from src.code.geometry import truncation_features
    from src.code.train import resolve_config, train_inline
    from src.code import plot_utils
    from src.notebooks._predicted_geometry import (predict, gauge, encoding_weights,
                                                   water_level)
    import matplotlib.pyplot as plt

    # The superposition regime the theory is written for: a long feature ladder (m = 64) with a
    # power-law importance spectrum, many more words than features, and a budget d << m.
    # p is supplied per run (it is a sweep axis), so it stays out of TASK.
    TASK = ["setting=feature_cube", "structure_kwargs.N=2048", "structure_kwargs.m=64",
            "structure_kwargs.scale=8.0", "+structure_kwargs.spectrum=power",
            "+structure_kwargs.beta=0.5"]
    CONFIG = resolve_config(TASK + ["structure_kwargs.p=0.3"])
    N, M, P0, SCALE, BETA = 2048, 64, 0.3, 8.0, 0.5
    D_GRID = [3, 6, 8, 12]
    SEEDS = range(5)
    STEPS = 8000
    COL = plot_utils.FIG.colors


    def task_of(p=P0, **kw):
        """Bayes targets and the feature draw for one variant of the task."""
        _logP, _info = feature_cube(N, M, p, scale=SCALE, seed=0, spectrum="power",
                                    beta=BETA, **kw)
        _P = np.exp(_logP - logsumexp(_logP, 1, keepdims=True))
        return _info["Z"], _info["alpha"], _P


    @mo.cache
    def trained(d, seed, p=P0, extra=()):
        """Train (or load) one rank-d linear model; return its gauge-fixed (code, decoder)."""
        model, _ds, ev, *_ = train_inline(
            TASK + [f"structure_kwargs.p={p}", "runner=linear",
                    f"model_kwargs.model_config.embed_dim={d}",
                    f"training_kwargs.train_steps={STEPS}", f"random_seed={seed}", *extra],
            name=f"fc_d{d}_p{p}_s{seed}")
        E = np.asarray(ev.input_code(model), float)
        return gauge(E, model.readout.detach().cpu().numpy()[:M], d)


    @mo.cache
    def predicted(d, p=P0, **kw):
        """Closed-form code and decoder at budget d. No training."""
        _Z, _alpha, _ = task_of(p, **kw)
        return predict(_Z, _alpha, N, p, d)


    def unit(X):
        """Row-normalise, so CKA compares directions rather than the norm profile."""
        return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


    def seed_ceiling(mats):
        """Mean pairwise CKA between independently trained runs: the reproducible ceiling."""
        return np.mean([cka(unit(mats[i]), unit(mats[j]))
                        for i, j in combinations(range(len(mats)), 2)])

    return (
        COL,
        CONFIG,
        D_GRID,
        M,
        N,
        P0,
        SEEDS,
        cka,
        combinations,
        encoding_weights,
        mo,
        np,
        plot_utils,
        plt,
        predicted,
        seed_ceiling,
        task_of,
        truncation_features,
        trained,
        unit,
        water_level,
    )


@app.cell
def _(D_GRID, SEEDS, np, predicted, task_of, trained):
    Z, ALPHA, PBAYES = task_of()
    RUNS = {d: [trained(d, s) for s in SEEDS] for d in D_GRID}
    PRED = {d: predicted(d) for d in D_GRID}
    for _d in D_GRID:
        print(f"d={_d:2d}  mu={PRED[_d]['mu']:.3f}  "
              f"active features={int((PRED[_d]['n2'] > 1e-12).sum()):2d}  "
              f"excess CE={PRED[_d]['excess_ce']:.4f}")
    return ALPHA, PBAYES, PRED, RUNS, Z


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 1 — the water-filling profile

    Recover each trained model's write vectors by regressing its code on the centred feature
    indicators, $a_x \sim \sum_k w_k (z_{xk} - p)$, and plot $\|w_i\|^2\sqrt{Npq}$ against the
    task importance $\alpha_i$ (blue dots, one per feature, 5 seeds overlaid). Orange is the
    closed form $A(\alpha_i - \mu)^+$ with $\mu$ solved self-consistently from the budget
    equation; the dashed line marks $\mu$.

    Criterion: a straight line through the head that hits zero at $\mu$, with every feature
    below $\mu$ flat at zero — that is water filling. Curvature in the head, or non-zero mass
    below $\mu$, falsifies the linear-in-$(\alpha - \mu)$ form. As $d$ grows the water level
    should drop and admit more of the tail.
    """)
    return


@app.cell
def _(ALPHA, COL, CONFIG, N, P0, PRED, RUNS, Z, encoding_weights, np, plot_utils, plt):
    _panel = [3, 6, 12]
    _rt = np.sqrt(N * P0 * (1 - P0))
    _fig, _axes = plt.subplots(1, len(_panel), figsize=(4.6 * len(_panel), 4.2), sharex=True)
    for _ax, _d in zip(_axes, _panel):
        for _j, (_A, _D) in enumerate(RUNS[_d]):
            _y = encoding_weights(_A, Z) ** 2 * _rt
            _ax.scatter(ALPHA, _y, s=26, color=COL["blue"], alpha=0.5, edgecolors="none",
                        zorder=3, label="trained (5 seeds)" if _j == 0 else None)
        _mu, _n2 = PRED[_d]["mu"], PRED[_d]["n2"]
        _o = np.argsort(ALPHA)
        _ax.plot(ALPHA[_o], (_n2 * _rt)[_o], color=COL["orange"], lw=2.6, zorder=4,
                 label=r"$A(\alpha_i-\mu)^+$")
        _ax.axvline(_mu, color="0.45", lw=1.4, ls="--", zorder=2)
        _ax.text(_mu, _ax.get_ylim()[1] * 0.60, r"  $\mu$", color="0.35", fontsize=13)
        _ax.set_title(f"$d = {_d}$", fontsize=15)
        _ax.set_xlabel(r"importance $\alpha_i$")
        plot_utils.FIG.style_axes(_ax)
    _axes[0].set_ylabel(r"$\|w_i\|^2\sqrt{Npq}$")
    _axes[0].legend(frameon=False, fontsize=12, loc="upper left")
    _fig.tight_layout()
    plot_utils.save_figure(_fig, plot_utils.figure_path(CONFIG, "fig1_water_filling"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 2 — predicted decoder vs trained decoders

    CKA between the predicted decoder (row-normalised, so this scores directions) and each of
    5 trained runs, against the budget $d$. Two references:

    - **Ceiling** (grey): mean pairwise CKA between the trained runs — how much of the geometry
      is reproducible at all. Nothing can beat this, because the part that varies seed to seed
      is not a function of the task.
    - **Baseline** (orange): the rank-$d$ truncation of the centred log-conditional, i.e. plain
      SVD of the task. This is the fair competitor — also closed form, also uses only the task,
      but ignores the CE stationarity argument entirely.

    Criterion: the prediction should sit well above the truncation baseline and inside or just
    under the grey band. Matching the baseline would mean the stationarity argument buys
    nothing over taking the top-$d$ singular directions; exceeding the ceiling would mean a bug.
    """)
    return


@app.cell
def _(COL, CONFIG, D_GRID, PBAYES, PRED, RUNS, cka, combinations, np, plot_utils, plt,
      seed_ceiling, truncation_features, unit):
    _x = np.arange(len(D_GRID))
    _pred = [np.mean([cka(unit(PRED[d]["decoder"]), unit(X)) for _, X in RUNS[d]])
             for d in D_GRID]
    _ceil = [seed_ceiling([X for _, X in RUNS[d]]) for d in D_GRID]
    _lo = [np.min([cka(unit(RUNS[d][i][1]), unit(RUNS[d][j][1]))
                   for i, j in combinations(range(len(RUNS[d])), 2)]) for d in D_GRID]
    _hi = [np.max([cka(unit(RUNS[d][i][1]), unit(RUNS[d][j][1]))
                   for i, j in combinations(range(len(RUNS[d])), 2)]) for d in D_GRID]
    # Competitor: the rank-d truncation of the centred log-conditional (plain SVD/PCA of the
    # task) -- a closed form that uses the task but none of the CE stationarity argument.
    _base = []
    for _d in D_GRID:
        _, _Dt = truncation_features(np.log(np.clip(PBAYES, 1e-300, None)), _d)
        _base.append(np.mean([cka(unit(_Dt.T), unit(X)) for _, X in RUNS[_d]]))

    _fig, _ax = plt.subplots(figsize=(7.4, 4.4))
    _ax.fill_between(_x, _lo, _hi, color=COL["gray"], alpha=0.22, lw=0, zorder=1)
    _ax.plot(_x, _ceil, "s--", color=COL["gray"], lw=2.2, ms=9, zorder=3,
             label="seed-to-seed ceiling")
    _ax.vlines(_x, 0, _pred, color=COL["blue"], lw=2.2, zorder=2)
    _ax.plot(_x, _pred, "o", color=COL["blue"], ms=12, zorder=4, label="closed-form prediction")
    _ax.plot(_x, _base, "^:", color=COL["orange"], lw=2, ms=10, zorder=3,
             label="closed-form baseline (rank-$d$ truncation)")
    for _i, (_p, _c) in enumerate(zip(_pred, _ceil)):
        _ax.annotate(f"{100 * _p / _c:.0f}% of ceiling", (_i, _p), textcoords="offset points",
                     xytext=(0, 14), ha="center", fontsize=11.5, color=COL["blue"])
    _ax.set_xticks(_x); _ax.set_xticklabels([f"$d = {d}$" for d in D_GRID])
    _ax.set_ylim(0, 1.12); _ax.set_ylabel("decoder CKA")
    _ax.legend(frameon=False, fontsize=11.5, loc="upper center",
               bbox_to_anchor=(0.5, -0.09), ncol=2)
    plot_utils.FIG.style_axes(_ax)
    _fig.tight_layout()
    plot_utils.save_figure(_fig, plot_utils.figure_path(CONFIG, "fig2_predicted_decoder"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 3 — what the tail decoders do

    Features below the water level get no write vector, but they still get a decoder. The
    stationarity condition says the tail decoder is forced: with the code fixed, feature $i$'s
    optimal $D_i$ solves $M_Z D_i = c_i v - u$ where $c_i = p e^{\alpha_i} + (1-p)$ and
    $M_Z, u, v$ are code moments shared across the tail. So the whole tail lies on a **line**
    in $\mathbb{R}^d$, parameterised by $c_i$.

    Left: the trained tail decoders at $d = 3$ projected on their top two principal components,
    coloured by $\log c_i$, with the predicted line overlaid; equal aspect, so collinearity is
    visible rather than an artefact of axis scaling. Right: the fraction of tail decoder
    variance on PC1, per budget.

    Criterion: at small $d$ the tail should be nearly one-dimensional and ordered by $c_i$
    along the line. If the tail is isotropic, or if colour does not order along PC1, the
    closed form is wrong. The prediction is a small-$d$ statement — the shared-moment argument
    weakens once $d$ is large enough that the tail can spread.
    """)
    return


@app.cell
def _(ALPHA, COL, CONFIG, D_GRID, M, N, P0, PRED, RUNS, Z, np, plot_utils, plt):
    def _tail_line(A, D, d, alpha):
        """Predicted tail decoders from the code moments: M_Z D_i = c_i v - u."""
        _tail = np.where(PRED[d]["n2"] <= 1e-12)[0]
        _W = np.exp(Z * alpha).sum(1)
        _Zm = np.exp(np.clip(A @ D.T, -60, 60)).sum(1)
        _u = (A / _Zm[:, None]).mean(0)
        _v = (A / _W[:, None]).mean(0)
        _Mz = (A[:, :, None] * A[:, None, :] / _Zm[:, None, None]).mean(0)
        _c = P0 * np.exp(np.clip(alpha, None, 50)) + (1 - P0)
        return _tail, np.array([np.linalg.inv(_Mz) @ (_c[i] * _v - _u) for i in _tail])

    _fig, _axes = plt.subplots(1, 2, figsize=(12.0, 4.6),
                              gridspec_kw=dict(width_ratios=(1.35, 1)))
    _A0, _D0 = RUNS[3][0]
    _tail, _line = _tail_line(_A0, _D0, 3, ALPHA)
    _obs = _D0[_tail]
    _mean = _obs.mean(0)
    _, _, _Vt = np.linalg.svd(_obs - _mean, full_matrices=False)
    _pc = (_obs - _mean) @ _Vt[:2].T
    _pl = (_line - _line.mean(0)) @ _Vt[:2].T
    _sc = _axes[0].scatter(_pc[:, 0], _pc[:, 1], c=np.log(P0 * np.exp(ALPHA[_tail]) + 1 - P0),
                           cmap="viridis", s=60, zorder=3, edgecolors="none")
    _o = np.argsort(_pl[:, 0])
    _axes[0].plot(_pl[_o, 0], _pl[_o, 1], color=COL["orange"], lw=2.4, zorder=2,
                  label="predicted line")
    _axes[0].set_aspect("equal", adjustable="box")   # honest aspect: a thin band is thin
    _pad = 0.12 * np.ptp(_pc[:, 0])
    _axes[0].set_xlim(_pc[:, 0].min() - _pad, _pc[:, 0].max() + _pad)
    _axes[0].set_ylim(_pc[:, 1].min() - _pad, _pc[:, 1].max() + _pad)
    _axes[0].set_xlabel("tail decoder PC1"); _axes[0].set_ylabel("PC2")
    _axes[0].set_title(f"$d = 3$: {M - len(_tail)} head, {len(_tail)} tail", fontsize=15)
    _axes[0].legend(frameon=False, fontsize=12, loc="upper left")
    _fig.colorbar(_sc, ax=_axes[0], label=r"$\log c_i$")
    plot_utils.FIG.style_axes(_axes[0])

    _frac = []
    for _d in D_GRID:
        _t = np.where(PRED[_d]["n2"] <= 1e-12)[0]
        _s = np.linalg.svd(RUNS[_d][0][1][_t] - RUNS[_d][0][1][_t].mean(0), compute_uv=False)
        _frac.append(_s[0] ** 2 / (_s ** 2).sum())
    _axes[1].bar(np.arange(len(D_GRID)), _frac, color=COL["purple"], width=0.6, zorder=3)
    _axes[1].set_xticks(np.arange(len(D_GRID)))
    _axes[1].set_xticklabels([f"$d = {d}$" for d in D_GRID])
    _axes[1].set_ylim(0, 1.05); _axes[1].set_ylabel("tail variance on PC1")
    _axes[1].set_title("collinearity falls as the budget grows", fontsize=15)
    plot_utils.FIG.style_axes(_axes[1])
    _fig.tight_layout()
    plot_utils.save_figure(_fig, plot_utils.figure_path(CONFIG, "fig3_tail_collinearity"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 4 — where the prediction holds and where it breaks

    Two stress axes, both scored against what is reproducible rather than against 1.

    **Sparsity** (left). Vary $p$ at $d = 3$. Blue is the agreement between predicted and
    measured write-vector norms, $\mathrm{corr}(\|w_i\|)$; grey is the seed-to-seed decoder
    ceiling. The prediction holds all the way down, but below $p \approx 0.2$ the ceiling
    collapses — independently trained networks stop agreeing with each other, so there is no
    reproducible geometry left to predict. That is the target vanishing, not the model failing.

    **Correlation** (right). Break the i.i.d. assumption with a Gaussian copula at
    $\rho = 0.6$: one common factor (equicorrelated), or 4 and 16 blocks. Blue is the code
    prediction, purple the tail closed form.

    Criterion: the code should survive both, because it only uses marginals and the budget.
    The tail closed form assumes shared code moments across the tail, so block structure —
    which gives each block its own moment — should break it. A tail bar that stays high under
    blocks would mean the collinearity in Figure 3 is not coming from the mechanism claimed.
    """)
    return


@app.cell
def _(COL, CONFIG, N, P0, cka, combinations, encoding_weights, np, plot_utils, plt,
      predicted, task_of, trained, unit, water_level):
    _ps = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    _cw, _cl = [], []
    for _p in _ps:
        _Zp, _ap, _Pp = task_of(_p)
        _, _n2 = water_level(_Zp, _ap, _Pp, N, _p, 3)
        _runs = [trained(3, _s, p=_p) for _s in range(3)]
        _cw.append(np.mean([np.corrcoef(encoding_weights(_A, _Zp), np.sqrt(_n2))[0, 1]
                            for _A, _ in _runs]))
        _cl.append(np.mean([cka(unit(_runs[i][1]), unit(_runs[j][1]))
                            for i, j in combinations(range(3), 2)]))

    _fig, _axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    _axes[0].axvspan(0.0, 0.15, color=COL["orange"], alpha=0.12, lw=0)
    _axes[0].text(0.077, 0.12, "target not\ndetermined", ha="center", fontsize=12,
                  color=COL["orange"])
    _axes[0].plot(_ps, _cw, "o-", color=COL["blue"], lw=2.5, ms=10, zorder=3,
                  label=r"code: $\mathrm{corr}(\|w_i\|)$")
    _axes[0].plot(_ps, _cl, "s--", color=COL["gray"], lw=2.5, ms=9, zorder=3,
                  label="seed-to-seed ceiling")
    _axes[0].set_xlabel("feature rate $p$"); _axes[0].set_ylabel("agreement")
    _axes[0].set_ylim(0, 1.08); _axes[0].set_xlim(0, 0.55)
    _axes[0].set_title("sparsity", fontsize=15)
    _axes[0].legend(frameon=False, fontsize=12, loc="upper center",
                    bbox_to_anchor=(0.5, -0.22), ncol=2)
    plot_utils.FIG.style_axes(_axes[0])

    _labels, _code, _tail = [], [], []
    for _lab, _bl in [("i.i.d.", None), ("equi-\ncorrelated", 1), ("blocks\n$= 4$", 4),
                      ("blocks\n$= 16$", 16)]:
        _kw = {} if _bl is None else dict(rho=0.6, blocks=_bl)
        _ex = () if _bl is None else (f"+structure_kwargs.rho=0.6",
                                      f"+structure_kwargs.blocks={_bl}")
        _Zb, _ab, _Pb = task_of(**_kw)
        _pr = predicted(3, **_kw)
        _A, _D = trained(3, 0, extra=_ex)
        _t = np.where(_pr["n2"] <= 1e-12)[0]
        _Wt = np.exp(_Zb * _ab).sum(1)
        _Zm = np.exp(np.clip(_A @ _D.T, -60, 60)).sum(1)
        _u = (_A / _Zm[:, None]).mean(0); _v = (_A / _Wt[:, None]).mean(0)
        _Mz = (_A[:, :, None] * _A[:, None, :] / _Zm[:, None, None]).mean(0)
        _c = P0 * np.exp(np.clip(_ab, None, 50)) + (1 - P0)
        _line = np.array([np.linalg.inv(_Mz) @ (_c[i] * _v - _u) for i in _t])
        _labels.append(_lab)
        _code.append(np.corrcoef(encoding_weights(_A, _Zb), np.sqrt(_pr["n2"]))[0, 1])
        _tail.append(cka(unit(_line), unit(_D[_t])))
    _xb = np.arange(len(_labels))
    for _off, _vals, _c, _lab in [(-0.19, _code, COL["blue"], r"code: $\mathrm{corr}(\|w_i\|)$"),
                                  (0.19, _tail, COL["purple"], "tail closed form (CKA)")]:
        _axes[1].bar(_xb + _off, _vals, width=0.36, color=_c, zorder=3, label=_lab)
        for _x, _y in zip(_xb + _off, _vals):
            _axes[1].text(_x, _y + 0.02, f"{_y:.2f}", ha="center", fontsize=11, color=_c)
    _axes[1].set_xticks(_xb); _axes[1].set_xticklabels(_labels, fontsize=12.5)
    _axes[1].set_ylim(0, 1.15); _axes[1].set_ylabel("agreement")
    _axes[1].set_title(r"feature correlation ($\rho = 0.6$)", fontsize=15)
    _axes[1].legend(frameon=False, fontsize=12, loc="upper center",
                    bbox_to_anchor=(0.5, -0.22), ncol=2)
    plot_utils.FIG.style_axes(_axes[1])
    _fig.tight_layout()
    print("corr(w) by p:", dict(zip(_ps, np.round(_cw, 3))))
    print("ceiling by p:", dict(zip(_ps, np.round(_cl, 3))))
    print("code:", np.round(_code, 3), " tail:", np.round(_tail, 3))
    plot_utils.save_figure(_fig, plot_utils.figure_path(CONFIG, "fig4_scope"))
    return


if __name__ == "__main__":
    app.run()
