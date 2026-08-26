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
    # What is a concept? Instructive figures on the ring

    An environment is a conditional over alternatives; every notion of "concept" is a reading of
    its centered log factorization $L_c = E\,D$. One space, one invariant:
    $$ \langle e(i),\, d(x) \rangle = \log \tilde p(x \mid i), $$
    concepts at the rows of $E$, outcomes at the columns of $D$, axes indexed by the rows of $D$.
    In the symmetric case $E = D^\top$ up to scale: concept points and outcome prototypes share
    one circle.

    Environment: a 60-node ring with the **heat** kernel, so each concept's output is a wrapped
    Gaussian bump.
    """)
    return


@app.cell
def _(mo):
    import os, sys
    import numpy as np

    ROOT = mo.notebook_dir().parent.parent    # src/notebooks -> <repo>
    os.chdir(ROOT); sys.path.insert(0, str(ROOT))
    from src.code.data import graph_conditional
    from src.code.geometry import factor_logcond, predicted_geometry
    from src.code.eval import cka
    from src.code.train import train_inline, graph_overrides
    from src.code import plot_utils
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    FIGURES = ROOT / "outputs" / "figures"
    FIGURES.mkdir(parents=True, exist_ok=True)

    N = 60
    SPEC = {"structure": "ring", "n": N, "gamma": 8.0, "cooccurrence": "gaussian"}   # exact von Mises
    L = graph_conditional(SPEC)
    _, E, D, _S = factor_logcond(L)
    hue = plt.cm.twilight(np.arange(N) / N)                             # house cyclic colormap

    KDIMS = 2                                                          # the kernel's exact rank
    KSTAR = 0                                                          # highlighted row (dimension)
    XSTAR = int(np.argmax(np.minimum(D[0], D[1])))                     # highlighted column: a concept
                                                                       # at a generic angle
    return (
        E,
        FIGURES,
        KDIMS,
        L,
        N,
        SPEC,
        cka,
        graph_conditional,
        graph_overrides,
        hue,
        mpatches,
        np,
        plot_utils,
        plt,
        predicted_geometry,
        train_inline,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. One intrinsic dimension, two linear dimensions

    The ring's latent angle $\theta$ is a single intrinsic dimension, but a linear code needs
    two: the von Mises kernel factorizes exactly as $L_c = E\,D$ with the $(\cos\theta,
    \sin\theta)$ pair. Top: the structure, its log-conditional, and the closed-form embedding.
    Middle: the exact rank-2 factorization. Bottom: an input $z$'s angle → its two linear
    coordinates → its point on the circle.
    """)
    return


@app.cell
def _(E, FIGURES, L, SPEC, plot_utils):
    _fig = plot_utils.plot_predicted_geometry(SPEC, L, E)
    plot_utils.save_figure(_fig, FIGURES / "concepts_ring_predicted.png")
    return


@app.cell
def _(E, FIGURES, N, hue, np, plot_utils, plt):
    # fix the rotation gauge so the two columns are literally cos(theta), sin(theta):
    # item 0 on the +x axis, theta increasing counterclockwise
    _E2 = E[:, :2].copy()
    _a0 = np.arctan2(_E2[0, 1], _E2[0, 0])
    _E2 = _E2 @ np.array([[np.cos(-_a0), np.sin(-_a0)], [-np.sin(-_a0), np.cos(-_a0)]]).T
    if _E2[1, 1] < 0:
        _E2[:, 1] *= -1
    XSTAR2 = 20                                        # theta ~ 120°: cos and sin differ in sign
    _lim = np.abs(_E2).max()

    def _strip(ax, M, edge):
        ax.imshow(M, cmap=plot_utils.FIG.diverging_cmap, vmin=-_lim, vmax=_lim, aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
        for _sp in ax.spines.values():
            _sp.set_visible(True); _sp.set_color(edge); _sp.set_linewidth(2.2)

    fig_dual = plt.figure(figsize=(11.5, 3.8))
    _gs = fig_dual.add_gridspec(1, 2, width_ratios=[1.45, 1], wspace=0.3)

    # the intrinsic angle -> the two linear dimensions (the columns of E over inputs z)
    _axc = fig_dual.add_subplot(_gs[0, 0])
    _axc.plot(_E2[:, 0], color="0.15", lw=1.5, label=r"$\propto \cos\theta$")
    _axc.plot(_E2[:, 1], color="0.15", lw=1.5, ls="--", label=r"$\propto \sin\theta$")
    _axc.axvline(XSTAR2, ls=":", color="0.6", lw=1.0)
    _axc.scatter([XSTAR2, XSTAR2], _E2[XSTAR2], c="0.15", s=42, zorder=4)
    _axc.set_xlim(-0.5, N - 0.5)
    _axc.set_xlabel(r"input $z$  (angle $\theta$)")
    _axc.legend(frameon=False)
    _axc.set_title(r"$\theta \to$ two linear dimensions", fontsize=12)
    plot_utils.FIG.style_axes(_axc)

    # e(z): the 2-vector, and its point on the circle in activation space
    _gsB = _gs[0, 1].subgridspec(1, 2, width_ratios=[0.09, 1], wspace=0.12)
    _axv = fig_dual.add_subplot(_gsB[0]); _strip(_axv, _E2[XSTAR2][:, None], "black")
    _axv.set_xlabel(r"$e(z)$", fontsize=13)
    _axp = fig_dual.add_subplot(_gsB[1])
    _r = _lim * 1.35
    _axp.annotate("", (_r, 0), (-_r, 0), arrowprops=dict(arrowstyle="->", color="0.45", lw=1.6))
    _axp.annotate("", (0, _r), (0, -_r), arrowprops=dict(arrowstyle="->", color="0.7", lw=1.6))
    _axp.scatter(_E2[:, 0], _E2[:, 1], c=hue, s=16, zorder=3)
    _px, _py = _E2[XSTAR2]
    _axp.scatter(_px, _py, s=190, facecolor="none", edgecolor="black", lw=2.4, zorder=4)
    _axp.plot([_px, _px], [0, _py], ls="--", color="0.15", lw=1.1, zorder=2)
    _axp.plot([0, _px], [_py, _py], ls="--", color="0.15", lw=1.1, zorder=2)
    _axp.set_xlim(-_r * 1.15, _r * 1.15); _axp.set_ylim(-_r * 1.15, _r * 1.15)
    _axp.set_aspect("equal")
    _axp.set_xlabel("activation space", fontsize=11, color="0.5")
    plot_utils.FIG.clean(_axp)
    plot_utils.save_figure(fig_dual, FIGURES / "concepts_dual_reading.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Three worlds: the one law as geometry

    $\log p(x \mid i) = \langle e_i, d_x\rangle$, drawn (the SVD checks live in the geometry
    notebook).

    - **Binary property.** The contrast $u = d_{\mathrm{yes}} - d_{\mathrm{no}}$ is one
      direction; a concept's log-odds is its projection on $u$. The prototypes' shared component
      is a gauge the softmax ignores, so the chord between them is the log-odds axis.
    - **Line (temperature).** Gaussian tuning $\log p(x \mid \theta) = -(\theta - \mu_x)^2/2 +
      \mathrm{const}(\theta)$ factorizes exactly in 2-D: embeddings stay the straight line
      $e_\theta = (\theta, 1)$ and the curvature moves into the decoder parabola
      $d_x = (\mu_x, -\mu_x^2/2)$.
    - **Circle (time of week).** Von Mises tuning; symmetry makes the normalizer constant, so
      $e = d$ up to scale — concentric circles, the self-legible symmetric world.
    """)
    return


@app.cell
def _(mpatches, np, plot_utils, plt):
    def vec(ax, v, color="0.15", lw=1.6):
        """Arrow from the origin to ``v``."""
        ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=lw, color=color))

    def frame(ax, title, xlim, ylim, pad=14):
        ax.set_title(title, fontsize=13, pad=pad)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_aspect("equal"); plot_utils.FIG.clean(ax)

    def code_axes(ax, rx, y0, y1):
        """Light coordinate axes of the code space."""
        ax.annotate("", xy=(rx, 0), xytext=(-rx, 0), arrowprops=dict(arrowstyle="->", lw=1.0, color="0.8"))
        ax.annotate("", xy=(0, y1), xytext=(0, y0), arrowprops=dict(arrowstyle="->", lw=1.0, color="0.8"))
        ax.text(rx, -0.1, "dim 1", fontsize=10, color="0.55", ha="right", va="top")
        ax.text(0.05, y1, "dim 2", fontsize=10, color="0.55", ha="left", va="top")

    def angle_arcs(ax, e_ang, d_angs, r0, dr):
        """Arcs from the embedding's angle to each decoder's, nested by angular distance."""
        d_angs = d_angs + 360 * ((d_angs - e_ang) < -180) - 360 * ((d_angs - e_ang) > 180)
        for _ang, _rk in zip(d_angs, np.argsort(np.argsort(np.abs(d_angs - e_ang)))):
            _r = r0 + dr * _rk
            ax.add_patch(mpatches.Arc((0, 0), 2 * _r, 2 * _r, theta1=min(e_ang, _ang),
                                      theta2=max(e_ang, _ang), color="0.78", lw=1.0))

    c_blue = plt.get_cmap(plot_utils.FIG.diverging_cmap)(0.08)
    c_red = plt.get_cmap(plot_utils.FIG.diverging_cmap)(0.92)

    # the three worlds' decoders (and, for binary, the one drawn embedding)
    u = np.array([np.cos(np.deg2rad(15)), np.sin(np.deg2rad(15))])     # the binary contrast
    mAB = 0.55 * np.array([-u[1], u[0]])                 # shared component — a gauge the softmax ignores
    d_yes, d_no = mAB + 0.9 * u, mAB - 0.9 * u           # the two prototypes, astride the contrast
    ei = 1.1 * np.array([np.cos(np.deg2rad(62)), np.sin(np.deg2rad(62))])

    adjs = ["freezing", "cold", "moderate", "warm", "hot"]
    mus = np.linspace(-1, 1, 5)                          # adjective centers on the latent
    Dv = np.stack([mus, (np.mean(mus**2) - mus**2) / 2], 1) * 1.2      # centered decoder parabola
    tstar = 0.55                                         # one embedding: a warm day

    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    day_angs = 90 - 360 * np.arange(7) / 7                             # Monday at the top, clockwise
    Dw = 0.85 * np.stack([np.cos(np.deg2rad(day_angs)), np.sin(np.deg2rad(day_angs))], 1)
    return (
        Dv,
        Dw,
        adjs,
        angle_arcs,
        c_blue,
        c_red,
        code_axes,
        d_no,
        d_yes,
        day_angs,
        days,
        ei,
        frame,
        mAB,
        mus,
        tstar,
        u,
        vec,
    )


@app.cell
def _(
    Dv,
    Dw,
    FIGURES,
    adjs,
    angle_arcs,
    c_blue,
    c_red,
    code_axes,
    d_no,
    d_yes,
    day_angs,
    days,
    ei,
    frame,
    mAB,
    np,
    plot_utils,
    plt,
    tstar,
    u,
    vec,
):
    fig_worlds, _ax = plt.subplots(1, 3, figsize=(19.5, 6))

    # binary property: the yes/no prototypes astride their contrast, one embedding
    _a = _ax[0]
    code_axes(_a, 1.55, -1.0, 1.15)
    _a.plot(*zip(mAB + 1.35 * u, mAB - 1.35 * u), ls=":", color="0.8", lw=1.2, zorder=0)   # the chord
    _a.text(*(mAB + 1.42 * u), r"$u = d_{\mathrm{yes}} - d_{\mathrm{no}}$", fontsize=11.5,
            color="0.55", ha="left", va="center")
    vec(_a, ei, color=c_blue, lw=2.6)
    _a.annotate(r"$e_i$", ei + np.array([0.02, 0.14]), ha="center", va="bottom",
                fontsize=13, color=c_blue)
    for _d, _lab, _off in [(d_yes, r"$d_{\mathrm{yes}}$", (0.2, -0.18)),
                           (d_no, r"$d_{\mathrm{no}}$", (-0.2, -0.18))]:
        vec(_a, _d)
        _a.annotate(f"{_lab}\n{ei @ _d:+.2f}", _d, _d + np.array(_off), fontsize=11,
                    ha="center", va="top", color="0.3")
    angle_arcs(_a, 62, np.degrees(np.arctan2([d_yes[1], d_no[1]], [d_yes[0], d_no[0]])), 0.3, 0.15)
    frame(_a, "binary property: log-odds = projection", (-1.75, 1.75), (-1.15, 1.7))

    # temperature on a line: e straight (offset), curvature lives in the decoder parabola
    _a = _ax[1]
    code_axes(_a, 1.7, -0.75, 1.45)
    _a.plot([-1.45, 1.45], [1, 1], ls=":", color="0.8", lw=1.2, zorder=0)   # the embedding line
    _a.text(1.52, 1.0, r"$e_\theta = (\theta,\ 1)$", fontsize=12, color="0.55", ha="left", va="center")
    _e = np.array([tstar, 1.0])
    vec(_a, _e, color=c_red, lw=2.6)
    _a.annotate(r"$e_\theta$  (warm day)", _e + np.array([0.02, 0.14]), ha="center", va="bottom",
                fontsize=12.5, color=c_red)
    _offs = {0: (0, -0.14, "center", "top"), 1: (-0.12, 0.06, "right", "center"),
             2: (-0.04, 0.24, "right", "bottom"), 3: (0.12, 0.06, "left", "center"),
             4: (0, -0.14, "center", "top")}
    for _j, (_dv, _lab) in enumerate(zip(Dv, adjs)):
        vec(_a, _dv)
        _dx, _dy, _ha, _va = _offs[_j]
        _a.annotate(f"{_lab}\n{_e @ _dv:+.2f}", _dv, _dv + np.array([_dx, _dy]), fontsize=11,
                    ha=_ha, va=_va, color="0.3")
    angle_arcs(_a, np.degrees(np.arctan2(_e[1], _e[0])),
               np.degrees(np.arctan2(Dv[:, 1], Dv[:, 0])), 0.2, 0.08)
    frame(_a, "continuous latent: embeddings stay a line, the decoder curves",
          (-1.9, 1.9), (-0.95, 1.85), pad=18)

    # circular latent (weektime): the symmetric case, e = d up to scale
    _a = _ax[2]
    code_axes(_a, 1.95, -1.5, 1.68)
    _tt = np.linspace(0, 2 * np.pi, 200)
    _a.plot(np.cos(_tt), np.sin(_tt), ls=":", color="0.8", lw=1.2, zorder=0)   # the embedding circle
    _phi = 90 - 360 * 2.6 / 7                                          # one embedding: Wednesday night
    _e = np.array([np.cos(np.deg2rad(_phi)), np.sin(np.deg2rad(_phi))])
    vec(_a, _e, color=c_red, lw=2.6)
    _a.annotate(r"$e_\varphi$  (Wed night)", _e * 1.85, ha="center", va="center",
                fontsize=12.5, color=c_red)
    for _dv, _lab in zip(Dw, days):
        vec(_a, _dv)
        _a.annotate(f"{_lab}\n{_e @ _dv:+.2f}", _dv, _dv / 0.85 * 1.3, fontsize=11,
                    ha="center", va="center", color="0.3")
    angle_arcs(_a, _phi, day_angs, 0.16, 0.06)
    frame(_a, "circular latent: the symmetric case, $e = d$ up to scale",
          (-2.25, 2.25), (-1.85, 2.1), pad=18)
    fig_worlds.tight_layout()
    plot_utils.save_figure(fig_worlds, FIGURES / "concepts_worlds.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2b. The same three worlds, with the readout's decision regions

    Logits are linear in the code with no bias, so every token's argmax region is a **cone from
    the origin**: angle decides, radius only sharpens the softmax. Binary: two half-planes split
    $\perp u$, the chord crossing the boundary at its midpoint. Line: five cones; the embedding
    line crosses their boundaries at $\theta = (\mu_i + \mu_j)/2$. Circle: seven equal sectors,
    the Voronoi of the day decoders.
    """)
    return


@app.cell
def _(
    Dv,
    Dw,
    FIGURES,
    adjs,
    c_blue,
    c_red,
    code_axes,
    d_no,
    d_yes,
    days,
    ei,
    frame,
    mAB,
    mus,
    np,
    plot_utils,
    plt,
    tstar,
    u,
    vec,
):
    from matplotlib.colors import to_rgb

    def argmax_regions(Dcols, r_x, r_y, res=500):
        """Argmax-outcome index over the code plane. Dcols (2, n) -> (res, res) image + extent."""
        _gx, _gy = np.linspace(*r_x, res), np.linspace(*r_y, res)
        C = np.stack(np.meshgrid(_gx, _gy), -1)                        # (res, res, 2), [i,j] = (x_j, y_i)
        return (C @ Dcols).argmax(-1), (*r_x, *r_y)

    def region_bg(ax, win, ext, rgb, fade=0.5):
        """Decision regions as a white-blended background image (rgb: (n, 3-or-4) region colors)."""
        ax.imshow(1 - fade * (1 - np.asarray(rgb)[win][..., :3]), origin="lower", extent=ext, zorder=0)

    fig_regions, _ax = plt.subplots(1, 3, figsize=(19.5, 6))

    # binary property: two half-planes
    _a = _ax[0]
    yes_no = [to_rgb(plot_utils.FIG.colors["blue"]), to_rgb(plot_utils.FIG.colors["orange"])]
    region_bg(_a, *argmax_regions(np.stack([d_yes, d_no], 1), (-1.75, 1.75), (-1.15, 1.7)),
              yes_no, fade=0.16)
    code_axes(_a, 1.55, -1.0, 1.15)
    _perp = np.array([-u[1], u[0]])                                    # the decision boundary, ⟂ u
    _a.plot(*zip(1.55 * _perp, -1.1 * _perp), ls="--", color="0.4", lw=1.3, zorder=2)
    _a.plot(*zip(mAB + 1.35 * u, mAB - 1.35 * u), ls=":", color="0.7", lw=1.2, zorder=1)   # the chord
    _a.text(*(mAB + 1.42 * u), r"$u$", fontsize=11.5, color="0.55", ha="left", va="center")
    _a.scatter(*mAB, s=30, facecolor="white", edgecolor="0.35", lw=1.2, zorder=4)   # midpoint, on the boundary
    vec(_a, ei, color=c_blue, lw=2.6)
    _a.annotate(r"$e_i$", ei + np.array([0.02, 0.14]), ha="center", va="bottom",
                fontsize=13, color=c_blue)
    for _d, _lab, _off in [(d_yes, r"$d_{\mathrm{yes}}$", (0.2, -0.16)),
                           (d_no, r"$d_{\mathrm{no}}$", (-0.2, -0.16))]:
        vec(_a, _d)
        _a.annotate(_lab, _d + np.array(_off), fontsize=11.5, ha="center", va="top", color="0.3")
    _a.text(1.28, -0.95, "argmax: yes", fontsize=11.5, color="0.35", ha="center")
    _a.text(-1.28, -0.95, "argmax: no", fontsize=11.5, color="0.35", ha="center")
    frame(_a, "binary: two half-planes", (-1.75, 1.75), (-1.15, 1.7))

    # temperature line: five cones, thresholds where the embedding line crosses them
    _a = _ax[1]
    cols5 = plt.get_cmap(plot_utils.FIG.diverging_cmap)(np.linspace(0.06, 0.94, 5))
    region_bg(_a, *argmax_regions(Dv.T, (-1.9, 1.9), (-0.95, 1.85), res=600), cols5, fade=0.55)
    code_axes(_a, 1.7, -0.75, 1.45)
    for _dv in Dv:
        vec(_a, _dv)
    for _mu, _lab in zip(mus, adjs):
        _a.text(1.55 * _mu, 1.58, _lab, fontsize=11, ha="center", color="0.25")
    _a.plot([-1.45, 1.45], [1, 1], ls=":", color="0.3", lw=1.4, zorder=3)
    _a.text(1.52, 1.0, r"$e_\theta = (\theta,\,1)$", fontsize=12, color="0.4", ha="left", va="center")
    vec(_a, np.array([tstar, 1.0]), color=c_red, lw=2.6)               # the warm day of fig 2
    _thr = (mus[:-1] + mus[1:]) / 2
    _a.scatter(_thr, np.ones(4), marker="|", s=110, color="0.1", zorder=5)
    for _t in _thr:
        _a.text(_t, 0.86, f"{_t:+.2f}", fontsize=9.5, ha="center", va="top", color="0.35")
    frame(_a, "line: five cones — thresholds on the latent", (-1.9, 1.9), (-0.95, 1.85))

    # weektime circle: seven equal sectors
    _a = _ax[2]
    hue7 = plt.cm.twilight(np.arange(7) / 7)
    region_bg(_a, *argmax_regions(Dw.T, (-2.25, 2.25), (-1.85, 2.1)), hue7, fade=0.45)
    code_axes(_a, 1.95, -1.5, 1.68)
    _tt = np.linspace(0, 2 * np.pi, 200)
    _a.plot(np.cos(_tt), np.sin(_tt), ls=":", color="0.55", lw=1.2, zorder=1)
    for _dv, _lab in zip(Dw, days):
        vec(_a, _dv)
        _a.annotate(_lab, _dv, _dv / 0.85 * 1.3, fontsize=11, ha="center", va="center", color="0.25")
    _phi = 90 - 360 * 2.3 / 7                                          # Wed night, inside Wed's sector
    _e = np.array([np.cos(np.deg2rad(_phi)), np.sin(np.deg2rad(_phi))])
    vec(_a, _e, color=c_red, lw=2.6)
    _a.annotate(r"$e_\varphi$  (Wed night)", _e * 1.78, ha="center", va="center",
                fontsize=12, color=c_red)
    frame(_a, "circle: seven equal sectors", (-2.25, 2.25), (-1.85, 2.1))
    fig_regions.tight_layout()
    plot_utils.save_figure(fig_regions, FIGURES / "concepts_decision_regions.png")
    return argmax_regions, cols5, hue7, region_bg, yes_no


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Inference: examples of $x$ walk the code space

    $z$ is hidden and the model sees examples $x_{1:t} \sim p(x \mid z)$. The posterior factors
    through an **additive** state, $\phi(x_{1:t}) = \phi(\varnothing) + \sum_i D(x_i)$, so a
    history is a trajectory over the regions of §2b.

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
    _theta_star = 0.55                                                 # the warm day of fig 2
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
    plot_utils.save_figure(fig_infer, FIGURES / "concepts_inference.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. The same three worlds, from trained models

    Each world as a **Gaussian kernel in its canonical coordinates**
    ($\log p(x \mid i) = -\tfrac{\gamma}{2}\lVert p_i - p_x \rVert^2 + \mathrm{const}_i$):
    the chain gives the drawn quadratic tuning, the ring exact von Mises, the 2-item world the
    binary property. All are rank 2 after centering, so a model with ``embed_dim=2`` is exact —
    its embedding rows and readout rows are $e(i)$ and $d(x)$ with no projection, and the
    decision regions are the model's true regions.
    """)
    return


@app.cell
def _(
    cka,
    graph_conditional,
    graph_overrides,
    predicted_geometry,
    train_inline,
):
    WORLDS = {
        # words -> binary features (the superposition setting, dense p and full rank)
        "binary": {"structure": "feature_cube", "N": 12, "m": 2, "p": 0.5, "scale": 3.0, "seed": 0},
        # continuous latents, coarse outcomes: 21 temperatures -> 5 adjectives, 28 times -> 7 days
        "line": {"structure": "chain", "cooccurrence": "gaussian", "n": 21, "m": 5, "gamma": 0.05},
        "week": {"structure": "ring", "cooccurrence": "gaussian", "n": 28, "m": 7, "gamma": 1.5},
    }
    learned = {}
    for _name, _sp in WORLDS.items():
        # the kernels are exactly rank 2 (centered), so a 2-D code loses nothing
        _model, _ds, _ev, _, _ = train_inline(graph_overrides(_sp) +
                                              ["runner=linear", "model_kwargs.model_config.embed_dim=2"])
        _E = _ev.item_representations(_model)
        _D = _model.readout.detach().cpu().numpy()
        print(f"{_name:8s} CKA={cka(_E, predicted_geometry(graph_conditional(_sp))):.3f}")
        learned[_name] = {"E": _E, "D": _D}
    return (learned,)


@app.cell
def _(
    FIGURES,
    argmax_regions,
    code_axes,
    frame,
    learned,
    np,
    plot_utils,
    plt,
    region_bg,
    vec,
    yes_no,
):
    fig_learned, _ax = plt.subplots(1, 3, figsize=(19.5, 6))
    _days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    _adjs = ["freezing", "cold", "moderate", "warm", "hot"]
    _panels = [("binary", "learned words: feature log-odds axis", yes_no, [r"$f_1$", r"$f_2$"]),
               ("line", "learned line: latent temperatures, adjective decoders",
                plt.get_cmap(plot_utils.FIG.diverging_cmap)(np.linspace(0.06, 0.94, 5)), _adjs),
               ("week", "learned week: latent times, day decoders",
                plt.cm.twilight(np.arange(7) / 7), _days)]
    for _a, (_name, _title, _cols, _labs) in zip(_ax, _panels):
        _E2, _D2 = learned[_name]["E"], learned[_name]["D"]        # the code IS the plane: exact
        _r = 1.3 * max(np.abs(_E2).max(), np.abs(_D2).max())
        region_bg(_a, *argmax_regions(_D2.T, (-_r, _r), (-_r, _r)), _cols,
                  fade=0.2 if _name == "binary" else 0.45)
        code_axes(_a, 0.95 * _r, -0.95 * _r, 0.95 * _r)
        for _j, _dv in enumerate(_D2):
            vec(_a, _dv)
            if _labs is not None:
                _a.annotate(_labs[_j], _dv * 1.25, fontsize=11, ha="center", va="center", color="0.25")
        _a.scatter(*_E2.T, c=np.arange(len(_E2)), cmap=plot_utils.FIG.item_cmap,
                   s=45, zorder=4, edgecolor="white", lw=0.7)
        frame(_a, _title, (-_r, _r), (-_r, _r))
    fig_learned.tight_layout()
    plot_utils.save_figure(fig_learned, FIGURES / "concepts_worlds_learned.png")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Inference from a trained model

    The latent-inference case of §3, real: the day $z$ is hidden, a transformer reads mention
    streams $x_{1:t} \sim \mathrm{vonMises}(z)$ (the week world's Gaussian kernel) and predicts
    $z$ at every position (``infer_latent`` with ``distinct_outcomes``). Bayes is additive in the
    evidence, so the hidden state should walk the code plane like §3's spiral. Left: one stream's
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
    plot_utils.save_figure(fig_inf, FIGURES / "concepts_inference_learned.png")
    return


@app.cell
def _(FIGURES, KDIMS, L, plot_utils):
    _fig2, _ = plot_utils.plot_factorization(L, k=KDIMS)
    plot_utils.save_figure(_fig2, FIGURES / "concepts_factorization.png")
    return


@app.cell
def _(FIGURES):
    from PIL import Image

    _names = ["concepts_ring_predicted", "concepts_factorization", "concepts_dual_reading"]
    _imgs = [Image.open(FIGURES / f"{_n}.png") for _n in _names]
    _w = min(_i.width for _i in _imgs)
    _scaled = [_i.resize((_w, round(_i.height * _w / _i.width))) for _i in _imgs]
    _gap = 60
    story = Image.new("RGB", (_w, sum(_i.height for _i in _scaled) + _gap * 2), "white")
    _y = 0
    for _i in _scaled:
        story.paste(_i, (0, _y)); _y += _i.height + _gap
    story.save(FIGURES / "concepts_embedding_story.png")
    print(f"Saved figure to {FIGURES / 'concepts_embedding_story.png'}")
    story
    return


if __name__ == "__main__":
    app.run()
