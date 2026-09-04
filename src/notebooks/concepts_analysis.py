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
    from src.code.data import graph_conditional, cube_codes
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
        D,
        E,
        FIGURES,
        KDIMS,
        L,
        N,
        SPEC,
        cka,
        cube_codes,
        factor_logcond,
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
def _(E, FIGURES, KDIMS, L, N, SPEC, hue, np, plot_utils, plt):
    # the whole 1 story in one image: predicted geometry, the factorization, the dual reading
    _figA = plot_utils.plot_predicted_geometry(SPEC, L, E)
    plot_utils.save_figure(_figA, FIGURES / "concepts_ring_predicted.png")
    _figB, _ = plot_utils.plot_factorization(L, k=KDIMS)
    plot_utils.save_figure(_figB, FIGURES / "concepts_factorization.png")

    # fix the rotation gauge so the two columns are literally cos(theta), sin(theta):
    # item 0 on the +x axis, theta increasing counterclockwise
    _E2 = E[:, :2].copy()
    _a0 = np.arctan2(_E2[0, 1], _E2[0, 0])
    _E2 = _E2 @ np.array([[np.cos(-_a0), np.sin(-_a0)], [-np.sin(-_a0), np.cos(-_a0)]]).T
    if _E2[1, 1] < 0:
        _E2[:, 1] *= -1
    _XSTAR2 = 20                                       # theta ~ 120°: cos and sin differ in sign
    _lim = np.abs(_E2).max()

    def _strip(ax, M, edge):
        ax.imshow(M, cmap=plot_utils.FIG.diverging_cmap, vmin=-_lim, vmax=_lim, aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
        for _sp in ax.spines.values():
            _sp.set_visible(True); _sp.set_color(edge); _sp.set_linewidth(2.2)

    _figC = plt.figure(figsize=(11.5, 3.8))
    _gs = _figC.add_gridspec(1, 2, width_ratios=[1.45, 1], wspace=0.3)

    # the intrinsic angle -> the two linear dimensions (the columns of E over inputs z)
    _axc = _figC.add_subplot(_gs[0, 0])
    _axc.plot(_E2[:, 0], color="0.15", lw=1.5, label=r"$\propto \cos\theta$")
    _axc.plot(_E2[:, 1], color="0.15", lw=1.5, ls="--", label=r"$\propto \sin\theta$")
    _axc.axvline(_XSTAR2, ls=":", color="0.6", lw=1.0)
    _axc.scatter([_XSTAR2, _XSTAR2], _E2[_XSTAR2], c="0.15", s=42, zorder=4)
    _axc.set_xlim(-0.5, N - 0.5)
    _axc.set_xlabel(r"input $z$  (angle $\theta$)")
    _axc.legend(frameon=False)
    _axc.set_title(r"$\theta \to$ two linear dimensions", fontsize=12)
    plot_utils.FIG.style_axes(_axc)

    # e(z): the 2-vector, and its point on the circle in activation space
    _gsB = _gs[0, 1].subgridspec(1, 2, width_ratios=[0.09, 1], wspace=0.12)
    _axv = _figC.add_subplot(_gsB[0]); _strip(_axv, _E2[_XSTAR2][:, None], "black")
    _axv.set_xlabel(r"$e(z)$", fontsize=13)
    _axp = _figC.add_subplot(_gsB[1])
    _r = _lim * 1.35
    _axp.annotate("", (_r, 0), (-_r, 0), arrowprops=dict(arrowstyle="->", color="0.45", lw=1.6))
    _axp.annotate("", (0, _r), (0, -_r), arrowprops=dict(arrowstyle="->", color="0.7", lw=1.6))
    _axp.scatter(_E2[:, 0], _E2[:, 1], c=hue, s=16, zorder=3)
    _px, _py = _E2[_XSTAR2]
    _axp.scatter(_px, _py, s=190, facecolor="none", edgecolor="black", lw=2.4, zorder=4)
    _axp.plot([_px, _px], [0, _py], ls="--", color="0.15", lw=1.1, zorder=2)
    _axp.plot([0, _px], [_py, _py], ls="--", color="0.15", lw=1.1, zorder=2)
    _axp.set_xlim(-_r * 1.15, _r * 1.15); _axp.set_ylim(-_r * 1.15, _r * 1.15)
    _axp.set_aspect("equal")
    _axp.set_xlabel("activation space", fontsize=11, color="0.5")
    plot_utils.FIG.clean(_axp)
    plot_utils.save_figure(_figC, FIGURES / "concepts_dual_reading.png")
    for _f in (_figA, _figB, _figC):
        plt.close(_f)

    from PIL import Image

    _names = ["concepts_ring_predicted", "concepts_factorization", "concepts_dual_reading"]
    _imgs = [Image.open(FIGURES / "png" / f"{_n}.png") for _n in _names]
    _w = min(_i.width for _i in _imgs)
    _scaled = [_i.resize((_w, round(_i.height * _w / _i.width))) for _i in _imgs]
    _gap = 60
    story = Image.new("RGB", (_w, sum(_i.height for _i in _scaled) + _gap * 2), "white")
    _y = 0
    for _i in _scaled:
        story.paste(_i, (0, _y)); _y += _i.height + _gap
    story.save(FIGURES / "concepts_embedding_story.png")
    story
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

    def frame(ax, title, xlim, ylim, pad=14, fontsize=13):
        ax.set_title(title, fontsize=fontsize, pad=pad)
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
    fig_worlds, _ax = plt.subplots(1, 3, figsize=(24, 7.4))

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
    frame(_a, "", (-1.75, 1.75), (-1.15, 1.7))

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
    frame(_a, "", (-1.9, 1.9), (-0.95, 1.85))

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
    frame(_a, "", (-2.25, 2.25), (-1.85, 2.1))
    fig_worlds.tight_layout()
    for _aa, _eq in zip(_ax, [r"$\log\frac{p(\mathrm{yes}\,\mid\, i)}{p(\mathrm{no}\,\mid\, i)} = \langle e_i,\, u\rangle$", r"$\log p(x \mid \theta) = -\frac{1}{2}(\theta - \mu_x)^2 + \mathrm{const}(\theta)$", r"$\log p(x \mid \varphi) = r\cos(\varphi - \theta_x)$"]):
        _p = _aa.get_position()   # one shared height for the three kernel equations
        fig_worlds.text((_p.x0 + _p.x1) / 2, 0.97, _eq, ha="center", va="top", fontsize=17)
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
    ## 2c. The same three worlds, from trained networks

    The centered kernels are rank $\le 2$ (binary is rank 1: two outcomes leave only the
    contrast), so a linear net with a 2-D code is exact. Each panel draws the trained network's
    own embedding rows and readout rows over its argmax regions, rotated/reflected into the
    canonical position of §2 (the isometries the loss leaves free). The binary decoder pair
    trains to $\pm u$ — the shared offset drawn in §2 is gauge, and weight decay removes it —
    and the plane's second coordinate is loss-flat.
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
        # the centered kernels are rank <= 2, so a 2-D code loses nothing
        _model, _ds, _ev, _, _ = train_inline(graph_overrides(_sp) +
                                              ["runner=linear", "model_kwargs.model_config.embed_dim=2"])
        _E = _ev.input_code(_model, project=False)     # the raw 2-D plane: D is drawn in it too
        _D = _model.readout.detach().cpu().numpy()
        print(f"{_name:8s} CKA={cka(_E, predicted_geometry(graph_conditional(_sp))):.3f}")
        learned[_name] = {"E": _E, "D": _D}
    return (learned,)


@app.cell
def _(
    FIGURES,
    adjs,
    argmax_regions,
    c_red,
    code_axes,
    cols5,
    days,
    frame,
    hue7,
    learned,
    np,
    plot_utils,
    plt,
    region_bg,
    vec,
    yes_no,
):
    def _rot(a):
        _c, _s = np.cos(a), np.sin(a)
        return np.array([[_c, -_s], [_s, _c]])

    def _gauge(E, D, world):
        """Rotate/reflect the learned plane into canonical position (the isometries the loss leaves)."""
        if world == "binary":                          # the contrast d_f1 - d_f2 on +x
            _u = D[0] - D[1]
            R = _rot(-np.arctan2(_u[1], _u[0]))
        elif world == "line":                          # the embedding line horizontal
            _v = np.linalg.svd(E - E.mean(0), full_matrices=False)[2][0]
            R = _rot(-np.arctan2(_v[1], _v[0]))
        else:                                          # week: Monday's decoder at the top
            R = _rot(np.pi / 2 - np.arctan2(D[0, 1], D[0, 0]))
        E, D = E @ R.T, D @ R.T
        if world == "line":
            if E[-1, 0] < E[0, 0]:                     # cold -> hot rightward
                E, D = E * [-1, 1], D * [-1, 1]
            if E[:, 1].mean() < 0:                     # the line above the origin
                E, D = E * [1, -1], D * [1, -1]
        if world == "week" and D[0, 0] * D[1, 1] - D[0, 1] * D[1, 0] > 0:
            E, D = E * [-1, 1], D * [-1, 1]            # days clockwise, as on a clock face
        return E, D

    fig_learned, _ax = plt.subplots(1, 3, figsize=(19.5, 6))

    # binary: words on the log-odds axis; the second coordinate is loss-flat
    _a = _ax[0]
    _E, _D = _gauge(learned["binary"]["E"], learned["binary"]["D"], "binary")
    _r = 1.3 * max(np.abs(_E).max(), np.abs(_D).max())
    region_bg(_a, *argmax_regions(_D.T, (-_r, _r), (-_r, _r)), yes_no, fade=0.16)
    code_axes(_a, 0.95 * _r, -0.95 * _r, 0.95 * _r)
    for _d, _lab, _dy in [(_D[0], r"$d_{f_1}$", 0.1), (_D[1], r"$d_{f_2}$", -0.1)]:
        vec(_a, _d)
        _a.annotate(_lab, _d * 1.2 + [0, _dy * _r], fontsize=12.5, ha="center", va="center", color="0.3")
    _a.scatter(*_E.T, c=np.arange(len(_E)), cmap=plot_utils.FIG.item_cmap,
               s=48, zorder=4, edgecolor="white", lw=0.7)
    _a.text(0, -0.85 * _r, r"log-odds $f_1\!:\!f_2$ = x coordinate", fontsize=11.5,
            ha="center", color="0.35")
    frame(_a, "binary: trained words on the log-odds axis", (-_r, _r), (-_r, _r))

    # line: embeddings on an affine line crossing the five cones in order
    _a = _ax[1]
    _E, _D = _gauge(learned["line"]["E"], learned["line"]["D"], "line")
    _r = 1.3 * max(np.abs(_E).max(), np.abs(_D).max())
    region_bg(_a, *argmax_regions(_D.T, (-_r, _r), (-_r, _r), res=600), cols5, fade=0.55)
    code_axes(_a, 0.55 * _r, -0.55 * _r, 0.55 * _r)
    for _d, _lab in zip(_D, adjs):
        vec(_a, _d)
        _a.annotate(_lab, _d / np.linalg.norm(_d) * 0.85 * _r, fontsize=11,
                    ha="center", va="center", color="0.25")
    _a.plot(*_E.T, color="0.3", lw=1.2, ls=":", zorder=3)
    _a.scatter(*_E.T, c=np.linspace(0, 1, len(_E)), cmap="coolwarm", s=34, zorder=4,
               edgecolor="white", lw=0.6)
    _t = np.linspace(0, 1, 4000)                       # boundary crossings along the embedding line
    _path = (1 - _t)[:, None] * _E[0] + _t[:, None] * _E[-1]
    _wins = (_path @ _D.T).argmax(1)
    _a.scatter(*_path[np.flatnonzero(np.diff(_wins)) + 1].T, marker="|", s=130, color="0.1", zorder=5)
    vec(_a, _E[14], color=c_red, lw=2.4)               # one warm day
    frame(_a, "line: trained latents cross the cones in order", (-_r, _r), (-_r, _r))

    # week: one circle of times inside the day decoders' sectors
    _a = _ax[2]
    _E, _D = _gauge(learned["week"]["E"], learned["week"]["D"], "week")
    _r = 1.35 * max(np.abs(_E).max(), np.abs(_D).max())
    region_bg(_a, *argmax_regions(_D.T, (-_r, _r), (-_r, _r)), hue7, fade=0.45)
    code_axes(_a, 0.55 * _r, -0.55 * _r, 0.55 * _r)
    for _d, _lab in zip(_D, days):
        vec(_a, _d)
        _a.annotate(_lab, _d / np.linalg.norm(_d) * 0.88 * _r, fontsize=11,
                    ha="center", va="center", color="0.25")
    _a.scatter(*_E.T, c=np.arange(len(_E)), cmap="twilight", s=34, zorder=4,
               edgecolor="white", lw=0.6)
    vec(_a, _E[10], color=c_red, lw=2.4)               # Wednesday night
    _a.annotate(r"$e$ (Wed night)", _E[10] * 1.45, fontsize=11.5, ha="center",
                va="center", color=c_red)
    frame(_a, "circle: trained times inside the day sectors", (-_r, _r), (-_r, _r))
    fig_learned.tight_layout()
    plot_utils.save_figure(fig_learned, FIGURES / "concepts_worlds_learned.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Many binary features: LRH is the additive special case

    The hypercube with the **heat** kernel makes the log-conditional additive over bits,
    $L(z, x) = \sum_k a_k z_k x_k$, so the optimal code is one direction per feature:
    flipping bit $k$ adds the same vector $2\sqrt{a_k}\,e_k$ *whatever the other bits are*.
    That is the linear representation hypothesis — parallel analogy edges — recovered as the
    special case of an additive kernel. The oracle's twelve edges come in three exactly
    parallel quadruples; a trained model's should be near-parallel within a bit and
    near-orthogonal across bits.
    """)
    return


@app.cell
def _(
    cka,
    cube_codes,
    factor_logcond,
    graph_conditional,
    graph_overrides,
    np,
    train_inline,
):
    CUBE = {"structure": "hypercube", "cooccurrence": "heat", "n": 3, "gamma": 0.8}
    NBITS = CUBE["n"]
    L_cube = graph_conditional(CUBE)
    _, _Ec, _Dc, _ = factor_logcond(L_cube)              # additive kernel: centered rank = n bits
    E_cube, D_cube = _Ec[:, :NBITS], _Dc[:NBITS]
    Z_cube = cube_codes(NBITS)                           # (2^n, n) bit codes, node order

    _model, _ds, _ev, _, _ = train_inline(graph_overrides(CUBE) +
                                          ["runner=linear", f"model_kwargs.model_config.embed_dim={NBITS}"])
    E_cube_hat = _ev.input_code(_model)
    print(f"hypercube CKA(learned, oracle) = {cka(E_cube_hat, E_cube):.3f}")

    def analogy_vectors(E, k):
        """Edge vectors of bit k: E[z with bit k flipped to 1] - E[z with bit k = 0], (2^(n-1), d)."""
        lo = np.flatnonzero(Z_cube[:, k] == 0)
        return E[lo + 2 ** k] - E[lo]

    def cosine_stats(E):
        """(mean within-bit cosine, mean |across-bit| cosine) over the analogy edges."""
        V = [analogy_vectors(E, k) for k in range(NBITS)]
        U = [v / np.linalg.norm(v, axis=1, keepdims=True) for v in V]
        within = np.mean([(u @ u.T)[np.triu_indices(len(u), 1)].mean() for u in U])
        across = np.mean([np.abs(U[i] @ U[j].T).mean()
                          for i in range(NBITS) for j in range(i + 1, NBITS)])
        return within, across

    for _name, _e in [("oracle", E_cube), ("learned", E_cube_hat)]:
        _w, _x = cosine_stats(_e)
        print(f"{_name:8s} analogy edges: within-bit cos = {_w:.3f}   across-bit |cos| = {_x:.3f}")
    return D_cube, E_cube, E_cube_hat, NBITS, Z_cube, analogy_vectors


@app.cell
def _(
    E_cube,
    E_cube_hat,
    FIGURES,
    NBITS,
    Z_cube,
    analogy_vectors,
    frame,
    np,
    plot_utils,
    plt,
):
    # projection built from the oracle's own bit directions: u0, u1 in the plane, u2 as the
    # axonometric offset. Linear, so parallel edges stay parallel in the drawing.
    _U = np.stack([analogy_vectors(E_cube, _k).mean(0) for _k in range(NBITS)])
    _U /= np.linalg.norm(_U, axis=1, keepdims=True)
    P_cube = np.stack([_U[0] + 0.42 * _U[2], _U[1] + 0.28 * _U[2]]).T   # (d, 2)

    from src.code.analysis_utils import align_representations
    E_hat_al, _r2 = align_representations(E_cube_hat, E_cube, method="linear")

    _bitcols = [plot_utils.FIG.colors[_c] for _c in ("blue", "orange", "green")]
    fig_cube, _ax = plt.subplots(1, 2, figsize=(12.5, 5.6))
    for _a, _E, _title in [(_ax[0], E_cube, "oracle: one direction per feature"),
                           (_ax[1], E_hat_al, "learned (aligned): the same parallel edges")]:
        _XY = _E @ P_cube
        for _k in range(NBITS):
            _lo = np.flatnonzero(Z_cube[:, _k] == 0)
            for _i in _lo:
                _a.plot(*np.stack([_XY[_i], _XY[_i + 2 ** _k]]).T, color=_bitcols[_k],
                        lw=2.0, alpha=0.85, zorder=2,
                        label=f"flip bit {_k}" if _i == _lo[0] else None)
        _a.scatter(*_XY.T, c="0.15", s=40, zorder=3)
        for _i, _z in enumerate(Z_cube):
            _a.annotate("".join(str(int(_b)) for _b in _z), _XY[_i], _XY[_i] + 0.055 * np.array([1, 1]),
                        fontsize=10, color="0.4")
        _m = 1.25 * np.abs(_XY).max()
        _a.legend(frameon=False, fontsize=10, loc="upper left")
        frame(_a, _title, (-_m, _m), (-_m, _m))
    fig_cube.tight_layout()
    plot_utils.save_figure(fig_cube, FIGURES / "concepts_hypercube_analogies.png")
    return (P_cube,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Uncertainty is location in the space

    Mixing codes is exact log-linear pooling: for $e_w = \sum_z w_z\, e(z)$,
    $$ e_w \cdot d_x = \sum_z w_z \log \tilde p(x \mid z)
       \;\;\Longrightarrow\;\; \hat p_w(x) \propto \prod_z p(x \mid z)^{w_z}. $$
    The code plane is the natural-parameter space of the outcome family; the item embeddings sit
    on a fixed-confidence shell inside it, and a belief lives at its pooled code.

    - **Cube.** Belief over one feature slides the code along that feature's analogy edge — a
      chord far from the origin, offset by the other features' coordinates; the midpoint is
      maximal feature uncertainty. Because the code is linear in $z$, it carries only the
      per-feature marginals: feature correlations are invisible.
    - **Line.** The embedding is affine in $\theta$, so pooling collapses to the mean: every
      belief with the same mean lands on the same on-shell point. The confidence axis still
      exists — the second coordinate scales the quadratic term, the predictive's precision —
      and the Bayes predictive sits below the shell at $s = 1 - \gamma\,\mathrm{Var}(\theta)$
      (variance of the belief, tilted by the outcome normalizer; the tilt vanishes on
      homogeneous worlds). Mixing embeddings never reaches it; only evidence does — the
      inference notebook's (count, evidence) plane is exactly this axis.
    - **Circle.** The shell is curved, so pooling leaves it automatically: spread shortens the
      resultant, radius = von Mises concentration, the center = uniform.
    """)
    return


@app.cell
def _(
    D,
    D_cube,
    E,
    E_cube,
    FIGURES,
    L,
    NBITS,
    P_cube,
    Z_cube,
    cols5,
    frame,
    np,
    plot_utils,
    plt,
):
    # the pooling law, verified on the ring: e_w . d_x = sum_z w_z log p~(x|z)
    _w = np.random.default_rng(0).dirichlet(np.ones(len(E)))
    _err = np.abs((_w @ E) @ D - _w @ (L - L.mean(1, keepdims=True))).max()
    print(f"pooling identity  max |e_w · d_x - Σ w log p̃|  =  {_err:.2e}")

    fig_unc, _ax = plt.subplots(1, 3, figsize=(17, 5.4))

    # cube: belief over bit 2 walks the analogy edge 000 -> 001, far from the origin
    _a = _ax[0]
    _XY = E_cube @ P_cube
    for _k in range(NBITS):
        for _i in np.flatnonzero(Z_cube[:, _k] == 0):
            _a.plot(*np.stack([_XY[_i], _XY[_i + 2 ** _k]]).T, color="0.85", lw=1.4, zorder=1)
    _a.scatter(*_XY.T, c="0.4", s=28, zorder=3)
    for _i, _z in enumerate(Z_cube):                   # vertex labels pushed outward, off the edges
        _o = _XY[_i] - _XY.mean(0)
        _a.annotate("".join(str(int(_b)) for _b in _z), _XY[_i] + 0.16 * _o / np.linalg.norm(_o),
                    fontsize=9.5, color="0.55", ha="center", va="center")
    _bs = np.linspace(0, 1, 11)
    _chord = np.outer(1 - _bs, E_cube[0]) + np.outer(_bs, E_cube[4])
    _a.scatter(*(_chord @ P_cube).T, c=_bs, cmap="viridis", s=46, zorder=4)
    _a.scatter([0], [0], c="0.3", marker="+", s=90, zorder=4)
    _a.annotate("origin", (0, 0), (0.07, -0.16), fontsize=10, color="0.45")
    _a.annotate(r"$p(f_3{=}1)$: 0 $\to$ 1", (_chord[5] @ P_cube), (_chord[5] @ P_cube) + [0.2, -0.14],
                fontsize=10.5, color="0.3", va="top")
    # the chord's readout is the exact geometric pool of the two vertex conditionals
    _p0, _p1 = (np.exp(_l) / np.exp(_l).sum() for _l in (E_cube[0] @ D_cube, E_cube[4] @ D_cube))
    _pool = _p0[None] ** (1 - _bs[:, None]) * _p1[None] ** _bs[:, None]
    _pool /= _pool.sum(1, keepdims=True)
    _lc = _chord @ D_cube
    _pc = np.exp(_lc - _lc.max(1, keepdims=True)); _pc /= _pc.sum(1, keepdims=True)
    print(f"cube: max |softmax(chord) - geometric pool| = {np.abs(_pc - _pool).max():.2e}")
    _m = 1.3 * np.abs(_XY).max()
    frame(_a, "cube: belief over one feature = its analogy edge", (-_m, _m), (-_m, _m))

    # line: the (mean, precision) plane; items on the unit-precision shell s = 1
    _a = _ax[1]
    _gam, _n, _m5 = 0.05, 21, 5
    _th = np.arange(_n, dtype=float)
    _mu = np.linspace(0, _n - 1, _m5)
    _tt = np.linspace(-1.5, _n + 0.5, 420)
    _ss = np.linspace(-0.03, 1.3, 320)
    _lg = _gam * (_tt[None, :, None] * _mu - _ss[:, None, None] * _mu ** 2 / 2)
    _a.imshow(1 - 0.55 * (1 - cols5[_lg.argmax(-1)][..., :3]), origin="lower",
              extent=(_tt[0], _tt[-1], _ss[0], _ss[-1]), aspect="auto", zorder=0)
    _a.axhline(1.0, color="0.5", lw=0.8, ls=":", zorder=3)
    _a.scatter(_th, np.ones(_n), c=np.linspace(0, 1, _n), cmap="coolwarm", s=26,
               zorder=4, edgecolor="white", lw=0.5)
    # same-mean beliefs: the pool stays at (mean, 1); the Bayes predictive drops below the shell
    # by the belief's variance: s = 1 - gamma Var(theta), with the belief tilted by the outcome
    # normalizer Z(theta) (the tilt vanishes on homogeneous worlds like the ring)
    _tstar, _taus = 10.0, np.array([0.0, 1.5, 2.0, 3.0])
    _K = np.exp(_gam * np.outer(_th, _mu) - _gam * _mu ** 2 / 2)       # exp-family form of the kernel
    _Z = _K.sum(1); _P = _K / _Z[:, None]                              # (theta's own factor cancels in P)
    _A = np.c_[np.ones(_m5), _gam * _mu, -_gam * _mu ** 2 / 2]         # logit basis: const, mean, precision
    _fit, _s_pred = [], []
    for _tau in _taus:
        _wb = (_th == _tstar).astype(float) if _tau == 0 else np.exp(-(_th - _tstar) ** 2 / (2 * _tau ** 2))
        _wb = _wb / _wb.sum()
        _fit.append(np.linalg.lstsq(_A, np.log(_wb @ _P), rcond=None)[0][1:])
        _wt = _wb / _Z; _wt = _wt / _wt.sum()
        _s_pred.append(1 - _gam * (_wt @ _th ** 2 - (_wt @ _th) ** 2))
    _fit = np.array(_fit)                                              # columns: mean, s
    _a.plot(_fit[:, 0], _fit[:, 1], color="0.4", lw=0.8, zorder=4)
    _a.scatter(_fit[:, 0], _fit[:, 1], c=_taus, cmap="viridis", marker="s", s=56, zorder=5)
    _a.scatter([_tstar], [1.0], marker="*", s=260, facecolor="gold", edgecolor="k", lw=0.8, zorder=6)
    _a.annotate("pool of every\nsame-mean belief", (_tstar, 1.0), (_tstar + 2.0, 1.12),
                fontsize=10.5, color="0.3")
    _a.annotate(r"Bayes predictive: $s = 1 - \gamma\,\mathrm{Var}(\theta)$", _fit[-1],
                _fit[-1] + [1.0, -0.04], fontsize=10.5, color="0.3", va="top")
    print("line: fitted s =", np.round(_fit[:, 1], 4),
          " vs 1 - gam Var_tilted =", np.round(_s_pred, 4))
    _a.set(xlabel="mean coordinate", ylabel=r"precision coordinate  $s$")
    _a.set_title("line: pooling collapses to the mean; confidence is the second axis", fontsize=12.5)
    plot_utils.FIG.style_axes(_a)

    # circle: the same construction leaves the shell by itself — radius = concentration
    _a = _ax[2]
    _nc = len(E)
    _thc = 2 * np.pi * np.arange(_nc) / _nc
    _circ = E[:, :2]
    _ks = [12, 6, 3, 1.5, 0.7, 0.3, 0.1]
    _codes = np.array([(lambda _b: _b / _b.sum())(np.exp(_k * np.cos(_thc - _thc[21]))) @ _circ
                       for _k in _ks])
    _a.scatter(*_circ.T, c="0.85", s=16, zorder=2)
    _a.scatter([0], [0], c="0.5", marker="+", s=80, zorder=3)
    _a.plot(*_codes.T, color="0.6", lw=0.8, zorder=3)
    _a.scatter(*_codes.T, c=range(len(_ks)), cmap="viridis", marker="s", s=52, zorder=4)
    _rad = np.linalg.norm(_codes, axis=1)
    print("circle: pooled-code radius by spread =", np.round(_rad, 2),
          " monotone:", bool(np.all(np.diff(_rad) < 0)))
    _mc = 1.15 * np.abs(_circ).max()
    frame(_a, r"circle: pooling walks rim $\to$ center (radius = confidence)", (-_mc, _mc), (-_mc, _mc))
    fig_unc.tight_layout()
    plot_utils.save_figure(fig_unc, FIGURES / "concepts_uncertainty.png")
    return


if __name__ == "__main__":
    app.run()
