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
    # Inference: evidence walks the code space

    The static worlds of the concepts notebook give each input $z$ a point in the code plane.
    Here $z$ is hidden and the model sees examples $x_{1:t} \sim p(x \mid z)$. The posterior
    factors through an **additive** state, $\phi(x_{1:t}) = \phi(\varnothing) + \sum_i D(x_i)$,
    so a history is a trajectory over the readout's decision regions.
    """)
    return


@app.cell
def _(mo):
    import os, sys
    import numpy as np

    ROOT = mo.notebook_dir().parent.parent    # src/notebooks -> <repo>
    os.chdir(ROOT); sys.path.insert(0, str(ROOT))
    from src.code.train import train_inline, graph_overrides
    from src.code import plot_utils
    import matplotlib.pyplot as plt

    from pathlib import Path
    FIGURES = Path(os.environ.get("CACHE_DIR", ROOT / "outputs")) / "figures"
    FIGURES.mkdir(parents=True, exist_ok=True)
    return FIGURES, graph_overrides, np, plot_utils, plt, train_inline


@app.cell
def _(np, plot_utils, plt):
    # the three worlds of the concepts notebook (binary property, temperature line, week circle),
    # restated so the trajectories can be drawn over their decision regions
    def vec(ax, v, color="0.15", lw=1.6):
        """Arrow from the origin to ``v``."""
        ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=lw, color=color))

    def frame(ax, title, xlim, ylim, pad=14):
        ax.set_title(title, fontsize=13, pad=pad)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_aspect("equal"); plot_utils.FIG.clean(ax)

    def argmax_regions(Dcols, r_x, r_y, res=500):
        """Argmax-outcome index over the code plane. Dcols (2, n) -> (res, res) image + extent."""
        _gx, _gy = np.linspace(*r_x, res), np.linspace(*r_y, res)
        C = np.stack(np.meshgrid(_gx, _gy), -1)                        # (res, res, 2), [i,j] = (x_j, y_i)
        return (C @ Dcols).argmax(-1), (*r_x, *r_y)

    def region_bg(ax, win, ext, rgb, fade=0.5):
        """Decision regions as a white-blended background image (rgb: (n, 3-or-4) region colors)."""
        ax.imshow(1 - fade * (1 - np.asarray(rgb)[win][..., :3]), origin="lower", extent=ext, zorder=0)

    from matplotlib.colors import to_rgb

    c_red = plt.get_cmap(plot_utils.FIG.diverging_cmap)(0.92)
    yes_no = [to_rgb(plot_utils.FIG.colors["blue"]), to_rgb(plot_utils.FIG.colors["orange"])]
    cols5 = plt.get_cmap(plot_utils.FIG.diverging_cmap)(np.linspace(0.06, 0.94, 5))
    hue7 = plt.cm.twilight(np.arange(7) / 7)

    u = np.array([np.cos(np.deg2rad(15)), np.sin(np.deg2rad(15))])     # the binary contrast
    mAB = 0.55 * np.array([-u[1], u[0]])                 # shared component — a gauge the softmax ignores
    d_yes, d_no = mAB + 0.9 * u, mAB - 0.9 * u           # the two prototypes, astride the contrast

    adjs = ["freezing", "cold", "moderate", "warm", "hot"]
    mus = np.linspace(-1, 1, 5)                          # adjective centers on the latent

    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    day_angs = 90 - 360 * np.arange(7) / 7                             # Monday at the top, clockwise
    Dw = 0.85 * np.stack([np.cos(np.deg2rad(day_angs)), np.sin(np.deg2rad(day_angs))], 1)
    return (
        Dw,
        adjs,
        argmax_regions,
        c_red,
        cols5,
        d_no,
        d_yes,
        day_angs,
        days,
        frame,
        hue7,
        mAB,
        mus,
        region_bg,
        vec,
        yes_no,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Three worlds, three trajectories

    - **Binary.** The belief point (the posterior-weighted concept) travels the chord between
      $e_A$ and $e_B$ at position $\sigma(\Lambda_t)$ of the cumulative log-odds.
    - **Line.** The evidence needs the count: $D(x) = (1, \mu_x)$, $E(\theta) = (-A(\theta),
      \theta)$ — a target-adapted embedding, not the next-token decoder. Cones in the
      (count, evidence) plane; the trajectory's slope is the running mean evidence.
    - **Circle.** Homogeneity makes $A$ constant, so the raw sum $\sum_i d(x_i)$ is the whole
      state: it spirals into the true day's sector — angle converges (content), radius grows
      (confidence).
    """)
    return


@app.cell
def _(
    Dw,
    FIGURES,
    adjs,
    argmax_regions,
    cols5,
    d_no,
    d_yes,
    day_angs,
    days,
    frame,
    hue7,
    mAB,
    mus,
    np,
    plot_utils,
    plt,
    region_bg,
    vec,
    yes_no,
):
    T = 40
    fig_infer, _ax = plt.subplots(1, 3, figsize=(15.5, 4.8))

    # binary z: the belief point travels the chord between the two concepts
    _a = _ax[0]
    _eA, _eB = d_yes, d_no                               # symmetric world: concepts = the prototypes
    _xr, _yr = (-1.65, 1.65), (-0.55, 1.45)
    region_bg(_a, *argmax_regions(np.stack([_eA, _eB], 1), _xr, _yr), yes_no, fade=0.16)
    _a.plot(*zip(2.6 * mAB, -1.0 * mAB), ls="--", color="0.4", lw=1.3, zorder=2)   # boundary ⟂ e_A - e_B
    for _ez, _lab, _off in [(_eA, r"$e_A$", (0.14, 0.06)), (_eB, r"$e_B$", (-0.14, 0.06))]:
        vec(_a, _ez, lw=2.0)
        _a.annotate(_lab, _ez, _ez + np.array(_off), fontsize=13, ha="center", va="bottom", color="0.2")
    _a.plot(*zip(_eA, _eB), color="0.75", lw=1.0, zorder=1)
    _p1 = 0.65                                                         # p(x says "yes" | z = A)
    _llr = np.log(_p1 / (1 - _p1))
    _lam = np.concatenate([[0.0], np.cumsum(np.where(np.random.default_rng(0).random(T) < _p1,
                                                     _llr, -_llr))])
    _b = 1 / (1 + np.exp(-_lam))                                       # posterior p(A | x_{1:t})
    _pts = _eB + np.outer(_b, _eA - _eB)
    _a.plot(*_pts.T, color="0.6", lw=0.8, zorder=3)
    _a.scatter(*_pts.T, c=range(T + 1), cmap="viridis", s=30, zorder=4)
    _a.annotate("$t=0$", mAB, mAB + np.array([-0.1, -0.3]), fontsize=11, color="0.3",
                arrowprops=dict(arrowstyle="->", color="0.5", lw=0.9))
    _a.text(1.25, -0.42, r"$\hat z = A$", fontsize=12.5, color="0.35", ha="center")
    _a.text(-1.25, -0.42, r"$\hat z = B$", fontsize=12.5, color="0.35", ha="center")
    frame(_a, "binary: the belief travels the chord", _xr, _yr, pad=6)

    # day-type from adjective mentions: cones in the (count, evidence) plane
    _a = _ax[1]
    _A_th = np.log(np.exp(mus[:, None] * mus[None, :] - mus[None, :] ** 2 / 2).sum(1))   # A(theta_k)
    _Slim = 0.45 * T
    _tg, _Sg = np.meshgrid(np.linspace(0, T, 400), np.linspace(-_Slim, _Slim, 400))
    _win = (mus * _Sg[..., None] - _A_th * _tg[..., None]).argmax(-1)  # argmax_k theta_k S - t A(theta_k)
    _a.imshow(1 - 0.55 * (1 - cols5[_win][..., :3]), origin="lower",
              extent=(0, T, -_Slim, _Slim), aspect="auto", zorder=0)
    _theta_star = 0.55                                                 # the warm day of the concepts notebook
    _p_star = np.exp(_theta_star * mus - mus ** 2 / 2); _p_star /= _p_star.sum()
    _xs = np.random.default_rng(26).choice(5, T, p=_p_star)
    _S = np.concatenate([[0.0], np.cumsum(mus[_xs])])
    _a.plot([0, T], [0, T * (_p_star @ mus)], ls="--", color="0.15", lw=1.1, zorder=2)
    _a.plot(range(T + 1), _S, color="0.35", lw=1.0, zorder=3)
    _a.scatter(range(T + 1), _S, c=range(T + 1), cmap="viridis", s=22, zorder=4)
    _bounds = np.diff(_A_th) / np.diff(mus)                            # region slopes S/t
    _mid = np.concatenate([[_bounds[0] - 0.14], (_bounds[:-1] + _bounds[1:]) / 2, [_bounds[-1] + 0.14]])
    for _slope, _lab in zip(_mid, adjs):
        _a.text(0.985 * T, np.clip(_slope * T, -0.93 * _Slim, 0.93 * _Slim), _lab,
                fontsize=10.5, ha="right", va="center", color="0.25",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.5))
    _a.set(xlim=(0, T), ylim=(-_Slim, _Slim), xlabel="examples  $t$",
           ylabel=r"evidence  $S_t = \sum_i \mu(x_i)$")
    _a.set_title("continuous: cones in (count, evidence)", fontsize=13)
    plot_utils.FIG.style_axes(_a)

    # day of the week: additive evidence spirals into the true day's sector
    _a = _ax[2]
    _zday = 2                                                          # true day: Wednesday
    _ang = np.deg2rad(day_angs)
    _pw = np.exp(1.5 * np.cos(_ang - _ang[_zday])); _pw /= _pw.sum()   # von Mises day mentions
    _xs = np.random.default_rng(2).choice(7, T, p=_pw)
    _traj = np.vstack([np.zeros(2), np.cumsum(Dw[_xs], axis=0)])
    _rr = np.abs(_traj).max() * 1.35
    region_bg(_a, *argmax_regions(Dw.T, (-_rr, _rr), (-_rr, _rr)), hue7, fade=0.45)
    for _dv, _lab in zip(Dw, days):
        _a.text(*(_dv / 0.85 * 0.88 * _rr), _lab, fontsize=11, ha="center", va="center",
                color="0.1" if _lab == days[_zday] else "0.35")
    _udir = Dw[_zday] / np.linalg.norm(Dw[_zday])
    _a.plot([0, _udir[0] * 0.78 * _rr], [0, _udir[1] * 0.78 * _rr], ls="--", color="0.15", lw=1.2, zorder=2)
    _a.plot(*_traj.T, color="0.35", lw=1.0, zorder=3)
    _a.scatter(*_traj.T, c=range(T + 1), cmap="viridis", s=26, zorder=4)
    _a.scatter(*(_udir * 0.78 * _rr), marker="*", s=260, facecolor="gold", edgecolor="k", lw=0.8, zorder=5)
    _a.annotate("$t=0$", _traj[0], _traj[0] + _rr * np.array([-0.22, -0.06]), fontsize=11, color="0.3")
    _a.set_title("week: evidence spirals into the true day", fontsize=13)
    _a.set_aspect("equal"); plot_utils.FIG.clean(_a)
    fig_infer.tight_layout()
    plot_utils.save_figure(fig_infer, FIGURES / "inference_trajectories.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Inference from a trained model

    The circular case, real: the day $z$ is hidden, a transformer reads mention streams
    $x_{1:t} \sim \mathrm{vonMises}(z)$ (the week world's Gaussian kernel) and predicts $z$ at
    every position (``infer_latent`` with ``distinct_outcomes``). Bayes is additive in the
    evidence, so the hidden state should walk the code plane like §1's spiral. Left: one stream's
    hidden-state trajectory over the argmax regions of the model's $z$-readout (its top-2 plane —
    the kernel is rank 2). Right: the model's posterior on the true day against exact Bayes.
    """)
    return


@app.cell
def _(graph_overrides, train_inline):
    import torch

    # a fine latent circle (28 times of week) emitting the 7 coarse day mentions: bipartite by
    # construction, so the input is genuinely hidden
    INFER = {"structure": "ring", "cooccurrence": "gaussian", "n": 28, "m": 7, "gamma": 1.5}
    model_inf, ds_inf, ev_inf, _, _ = train_inline(graph_overrides(INFER) + [
        "runner=transformer", "sample_mode=sequence", "++infer_latent=true",
        "training_kwargs.context_length=40", "training_kwargs.batch_size=256"])
    return ds_inf, model_inf, torch


@app.cell
def _(
    FIGURES,
    argmax_regions,
    c_red,
    ds_inf,
    frame,
    model_inf,
    np,
    plot_utils,
    plt,
    region_bg,
    torch,
):
    _T, _Z = 40, 9                                                     # true time: Wednesday night
    _g = torch.Generator(device=ds_inf.device).manual_seed(7)
    _xs = torch.multinomial(ds_inf.P[_Z], _T, replacement=True, generator=_g)
    _ids = torch.cat([torch.tensor([ds_inf.start_token], device=ds_inf.device), _xs])[None]
    with torch.no_grad():
        _out = model_inf(_ids, output_hidden_states=True)
    _H = _out["hidden_states"][-1][0].float().cpu().numpy()            # h_t after t mentions
    _W = model_inf.get_output_embeddings().weight[ds_inf.item_tokens.cpu()].detach().float().cpu().numpy()
    _B = np.linalg.svd(_W, full_matrices=False)[2][:2]                 # the z-readout's top-2 plane
    _H2, _W2 = _H @ _B.T, _W @ _B.T

    _days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig_inf, _axs = plt.subplots(1, 2, figsize=(13.5, 5.4))
    _a = _axs[0]
    _rr = 1.25 * max(np.abs(_H2).max(), np.abs(_W2).max())
    _n_z = len(_W2)
    region_bg(_a, *argmax_regions(_W2.T, (-_rr, _rr), (-_rr, _rr)),
              plt.cm.twilight(np.arange(_n_z) / _n_z), fade=0.45)
    for _j, _lab in enumerate(_days):                                  # day labels at their latent angle
        _w = _W2[_j * _n_z // 7]
        _a.text(*(_w / np.linalg.norm(_w) * 0.9 * _rr), _lab, fontsize=11, ha="center", va="center",
                color="0.1" if _lab == "Wed" else "0.35")
    _a.plot(*_H2.T, color="0.35", lw=1.0, zorder=3)
    _a.scatter(*_H2.T, c=np.arange(_T + 1), cmap="viridis", s=26, zorder=4)
    _a.annotate("$t=0$", _H2[0], _H2[0] + _rr * np.array([-0.2, -0.06]), fontsize=11, color="0.3")
    frame(_a, "hidden state walks to the true time of week", (-_rr, _rr), (-_rr, _rr))

    _a = _axs[1]
    _bel = ds_inf.optimal_predictions(_ids)[0].cpu().numpy()           # Bayes over z, row t = t mentions
    _lg = _out["logits"][0][:-1][:, ds_inf.item_tokens.cpu()].float().cpu().numpy()
    _pm = np.exp(_lg - _lg.max(-1, keepdims=True)); _pm /= _pm.sum(-1, keepdims=True)
    _a.plot(range(_T), _bel[:, _Z], color="0.15", lw=2.0, label="Bayes")
    _a.plot(range(_T), _pm[:, _Z], color=c_red, lw=1.6, label="model")
    _a.set(xlabel="mentions  $t$", ylabel=r"p(z = Wed night $\mid x_{1:t}$)", ylim=(0, 1.02))
    _a.legend(frameon=False)
    _a.set_title("the posterior tracks Bayes", fontsize=13)
    plot_utils.FIG.style_axes(_a)
    fig_inf.tight_layout()
    plot_utils.save_figure(fig_inf, FIGURES / "inference_learned.png")
    return


if __name__ == "__main__":
    app.run()
