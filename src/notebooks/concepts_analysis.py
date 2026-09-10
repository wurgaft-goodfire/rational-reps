import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # What is a concept? Instructive figures
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
    from src.code.analysis_utils import cka, load_run
    from src.code.train import graph_overrides
    from src.code import plot_utils
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    from pathlib import Path
    FIGURES = Path(os.environ.get("CACHE_DIR", ROOT / "outputs")) / "figures"
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
        factor_logcond,
        graph_conditional,
        graph_overrides,
        hue,
        load_run,
        mpatches,
        np,
        plot_utils,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. One intrinsic dimension, two linear dimensions

    The ring's angle is one intrinsic dimension, but the linear code needs the
    $(\cos\theta, \sin\theta)$ pair: structure, log-conditional, factorization, and the dual
    reading of one input.
    """)
    return


@app.cell
def _(E, FIGURES, KDIMS, L, N, SPEC, hue, np, plot_utils, plt):
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

    def _dual_reading(fig):
        _gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1], wspace=0.3)

        # the intrinsic angle -> the two linear dimensions (the columns of E over inputs z)
        _axc = fig.add_subplot(_gs[0, 0])
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
        _axv = fig.add_subplot(_gsB[0]); _strip(_axv, _E2[_XSTAR2][:, None], "black")
        _axv.set_xlabel(r"$e(z)$", fontsize=13)
        _axp = fig.add_subplot(_gsB[1])
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

    # the whole story: predicted geometry, the factorization, the dual reading — each panel saved
    # on its own and then drawn again into one stacked vector figure
    _panels = [
        (3.9, "concepts_ring_predicted", lambda fig: plot_utils.plot_predicted_geometry(SPEC, L, E, fig=fig)),
        (4.2, "concepts_factorization", lambda fig: plot_utils.plot_factorization(L, k=KDIMS, fig=fig)),
        (3.8, "concepts_dual_reading", _dual_reading),
    ]
    for _h, _name, _draw in _panels:
        _f = plt.figure(figsize=(13, _h)); _draw(_f)
        plot_utils.save_figure(_f, FIGURES / f"{_name}.png"); plt.close(_f)
    story = plot_utils.stack_figures([(_h, _draw) for _h, _, _draw in _panels])
    plot_utils.save_figure(story, FIGURES / "concepts_embedding_story.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Same ring, diffusion kernel instead of Gaussian: the top two dimensions are still the circle,
    but every harmonic is present, so the factorization is a tower of harmonic pairs with
    geometrically decaying scale — the kernel, not the structure, sets how many dimensions the
    code needs.
    """)
    return


@app.cell
def _(FIGURES, N, factor_logcond, graph_conditional, np, plot_utils, plt):
    # the same ring under the diffusion kernel: every harmonic present, geometrically decaying.
    # Each harmonic pair shares a singular value, so the SVD fixes it only up to a phase: the
    # columns are cos(k theta + phi_k), sin(k theta + phi_k) with an arbitrary phi_k per pair
    N2 = N
    SPEC2 = {"structure": "ring", "n": N2, "gamma": 0.8, "cooccurrence": "diffusion"}
    L2 = graph_conditional(SPEC2)
    _, E2, D2, S2 = factor_logcond(L2)
    _rank = int((S2 > 1e-8 * S2[0]).sum())
    _ratio = S2[2] / S2[0]
    print(f"diffusion ring: centered rank {_rank} of {N2 - 1}; harmonic pairs decay by x{_ratio:.2f} "
          f"({(np.cumsum(S2 ** 2) / (S2 ** 2).sum() < 0.99).sum() + 1} dims for 99% of the energy)")
    print(f"symmetric kernel: max |E - D^T| = {np.abs(E2 - D2.T).max():.1e}")
    _KD = 6                                                        # harmonic pairs drawn as curves

    def _harmonics(fig):
        _gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1], wspace=0.3)
        _axc = fig.add_subplot(_gs[0, 0])                          # the harmonics over the input
        for _k in range(_KD):
            _axc.plot(E2[:, _k], color=str(0.15 + 0.2 * (_k // 2)), lw=1.5, ls="-" if _k % 2 == 0 else "--")
        _axc.set_xlim(-0.5, N2 - 0.5)
        _axc.set_xlabel(r"input $z$  (angle $\theta$)")
        _axc.set_title(r"$\theta \to$ harmonic pairs at $k = 1$ (dark) to $3$ (light), each up to a phase", fontsize=12)
        plot_utils.FIG.style_axes(_axc)
        _axs = fig.add_subplot(_gs[0, 1])                          # the spectrum: one bar per dimension
        _axs.bar(np.arange(1, N2), S2[:N2 - 1], color=["0.25" if _k % 4 < 2 else "0.6" for _k in range(N2 - 1)], width=0.8)
        _axs.set(xticks=[1, 10, 20, 30, 40, 50, N2 - 1], xlabel="dimension", yscale="log")
        _axs.set_title("singular values decay by harmonic pair", fontsize=12)
        plot_utils.FIG.style_axes(_axs)

    _panels = [
        (3.9, "concepts_ring_predicted_diffusion",
         lambda fig: plot_utils.plot_predicted_geometry(SPEC2, L2, E2, fig=fig)),
        (5.0, "concepts_factorization_diffusion",                  # the full decomposition, equal cells
         lambda fig: plot_utils.plot_factorization(L2, k=N2 - 1, embedding_width=1 / N2, fig=fig)),
        (3.8, "concepts_harmonics_diffusion", _harmonics),
    ]
    for _h, _name, _draw in _panels:
        _f = plt.figure(figsize=(13, _h)); _draw(_f)
        plot_utils.save_figure(_f, FIGURES / f"{_name}.png"); plt.close(_f)
    story2 = plot_utils.stack_figures([(_h, _draw) for _h, _, _draw in _panels])
    plot_utils.save_figure(story2, FIGURES / "concepts_embedding_story_diffusion.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Three worlds: the one law as geometry, and uncertainty as movement

    Binary features, temperature line, time-of-week circle, each from a trained 2-D linear net in
    canonical gauge. Top: one embedding, the decoders, and their dot products, with the kernel above
    and each token's argmax region, a cone from the origin, behind. Below: two movements of the
    representation in each world — along the manifold (which output, dashed) and toward the origin
    (uncertainty, dotted) — with the outcome probabilities they produce underneath.
    """)
    return


@app.cell
def _(mpatches, np, plot_utils, plt):
    from matplotlib.colors import to_rgb

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

    def angle_arcs(ax, e, D, r0, dr):
        """Arcs from the embedding's angle to each decoder's, nested by angular distance."""
        e_ang = np.degrees(np.arctan2(e[1], e[0]))
        d_angs = np.degrees(np.arctan2(D[:, 1], D[:, 0]))
        d_angs = d_angs + 360 * ((d_angs - e_ang) < -180) - 360 * ((d_angs - e_ang) > 180)
        for _ang, _rk in zip(d_angs, np.argsort(np.argsort(np.abs(d_angs - e_ang)))):
            _r = r0 + dr * _rk
            ax.add_patch(mpatches.Arc((0, 0), 2 * _r, 2 * _r, theta1=min(e_ang, _ang),
                                      theta2=max(e_ang, _ang), color="0.45", lw=1.2))

    def region_fill(ax, D, rgb, R, fade=0.5, res=7200):
        """Each decoder's argmax cone as a vector wedge from the origin (crisp at any zoom, editable
        in the PDF), filled with its colour blended toward white by ``fade``."""
        _phi = np.linspace(-np.pi, np.pi, res, endpoint=False)
        _U = np.c_[np.cos(_phi), np.sin(_phi)]
        _win = (_U @ D.T).argmax(1)
        for _k in range(len(D)):
            _idx = np.flatnonzero(_win == _k)
            if len(_idx) == 0:
                continue
            _brk = np.flatnonzero(np.diff(_idx) > 1)               # a cone wrapping past -pi
            _idx = np.roll(_idx, -(_brk[0] + 1)) if len(_brk) else _idx
            _arc = _U[_idx] * 3 * R
            _col = 1 - fade * (1 - np.asarray(rgb[_k])[:3])
            ax.fill(np.r_[0, _arc[:, 0]], np.r_[0, _arc[:, 1]], facecolor=_col, edgecolor=_col,
                    lw=0.8, zorder=0)                                  # own-colour edge hides the seams

    def rot(a):
        _c, _s = np.cos(a), np.sin(a)
        return np.array([[_c, -_s], [_s, _c]])

    def gauge(world, E, D):
        """Rotate/reflect a rank-2 code into canonical position, the isometries the loss leaves free.
        binary: feature axes (u_1 = d_1+ - d_1- on +x, u_2 up); line: the central adjective up (the
        kernel's symmetry axis), cold -> hot; week: Monday up, days clockwise. E (n, 2), D (m, 2)."""
        D = D - D.mean(0)                                # the decoders' common offset is gauge
        if world == "binary":
            v, up = D[0] - D[2], False
        elif world == "line":
            v, up = D[len(D) // 2], True
        else:
            v, up = D[0], True
        R = rot(-np.arctan2(v[1], v[0]) + (np.pi / 2 if up else 0.0))
        E, D = E @ R.T, D @ R.T
        flip_x = (world == "line" and E[-1, 0] < E[0, 0]) or \
                 (world == "week" and D[0, 0] * D[1, 1] - D[0, 1] * D[1, 0] > 0)
        flip_y = world == "binary" and (D[1] - D[3])[1] < 0
        S = np.array([-1.0 if flip_x else 1.0, -1.0 if flip_y else 1.0])
        return E * S, D * S

    def _highlight(ax, e, label, offset):
        """The highlighted embedding: a dashed black arrow plus its label at ``e + offset``."""
        ax.annotate("", xy=e, xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=2.2, color="0.1", ls=(0, (3, 2.5))))
        ax.annotate(label, e + offset, ha="center", va="center", fontsize=24, color="0.1")

    def _cone_dir(D, k, res=3600):
        """Unit vector through the middle of decoder k's argmax cone among ``D``, and the unit normal
        to the arrow pointing into the wider side of that cone."""
        _phi = np.linspace(-np.pi, np.pi, res, endpoint=False)
        _U = np.c_[np.cos(_phi), np.sin(_phi)]
        _in = _U[(_U @ D.T).argmax(1) == k]
        _n = np.array([-D[k, 1], D[k, 0]]) / np.linalg.norm(D[k])
        _mid = _in.mean(0)
        return _mid / np.linalg.norm(_mid), _n * (1 if (_in @ _n > 0).sum() >= (_in @ _n < 0).sum() else -1)

    def _decoders(ax, e, D, labels, pad, values=True, color="0.15", lw=1.6, rmax=np.inf, numbers=True):
        """Decoder arrows; each name (and, smaller, its logit e.d) sits inside its own cone: past the
        tip along the cone's middle, or — when that would leave the frame (``rmax``) — beside the
        arrow near its tip, on the wider side of the cone."""
        for _k, (_d, _lab) in enumerate(zip(D, labels)):
            vec(ax, _d, color=color, lw=lw)
            _len = np.linalg.norm(_d)
            if not values:
                _p = _d / _len * (_len + pad)
            else:
                _mid, _side = _cone_dir(D, _k)
                _p = _mid * (_len + pad) if _len + pad <= rmax else 0.8 * _d + max(0.6 * pad, 0.13 * _len) * _side
            if values and numbers:
                ax.annotate(_lab, _p, ha="center", va="bottom", fontsize=19, color="0.3")
                ax.annotate(f"{e @ _d:+.2f}", _p, ha="center", va="top", fontsize=14, color="0.45")
            elif values:
                _u = _p / np.linalg.norm(_p)
                _ha = "left" if _u[0] > 0.3 else "right" if _u[0] < -0.3 else "center"
                _va = "bottom" if _u[1] > 0.3 else "top" if _u[1] < -0.3 else "center"
                ax.annotate(_lab, _p, ha=_ha, va=_va, fontsize=19, color="0.3")
            else:
                ax.annotate(_lab, _p, ha="center", va="center", fontsize=19, color="0.55")

    def draw_binary(ax, E, D, i, regions=False, arrow=True, logits=True):
        """Paired yes/no world in the feature-axes gauge (tokens [1+, 2+, 1-, 2-]): the four words at
        the corners, word i read out by the feature-1 pair; regions = that pair's yes/no half-planes."""
        e, _g = E[i], max(np.abs(E).max(), np.abs(D).max()); R = 1.5 * _g
        if regions:
            region_fill(ax, D[[0, 2]], yes_no, R, fade=0.1)
        ax.scatter(*E.T, s=70, color="0.6", zorder=3)                                   # the four words
        for _j, _w in enumerate(WORDS):
            ax.annotate(_w, E[_j] + 0.1 * R * E[_j] / np.linalg.norm(E[_j]), fontsize=19,
                        ha="center", va="center", color="0.45")
        _decoders(ax, e, D[[1, 3]], ["Female +", "Female −"], 0.1 * _g,
                  values=False, color="0.6", lw=1.2)
        _decoders(ax, e, D[[0, 2]], ["Royal +", "Royal −"], (0.15 if logits else 0.05) * _g, numbers=logits)
        _perp = np.array([e[1], -e[0]]) / np.linalg.norm(e)             # label beside the arrow
        if arrow:
            _highlight(ax, e, r"$e_z$", 0.1 * e + 0.09 * R * _perp)
        angle_arcs(ax, e, D[[0, 2]], 0.12 * R, 0.05 * R)
        frame(ax, "", (-R, R), (-R, R))

    def draw_line(ax, E, D, i, regions=False, arrow=True, logits=True):
        """Temperature line: latents E on a line above the origin, adjective decoders D, latent i
        highlighted; regions = the five cones and where the line crosses them."""
        e, _g = E[i], max(np.abs(E).max(), np.abs(D).max()); R = 1.2 * _g
        _d = E[-1] - E[0]
        if regions:
            region_fill(ax, D, cols5, R, fade=0.35)
        code_axes(ax, 0.92 * R, -0.92 * R, 0.92 * R)
        ax.plot(*np.stack([E[0] - 0.15 * _d, E[-1] + 0.15 * _d]).T, ls=":", lw=1.2, zorder=0,
                color="0.3" if regions else "0.8")                                       # the embedding line
        ax.scatter(*E.T, c=np.linspace(0, 1, len(E)), cmap="coolwarm", s=46, zorder=3)
        _decoders(ax, e, D, adjs, (0.22 if logits else 0.08) * _g, rmax=0.82 * R, numbers=logits)
        if arrow:
            _highlight(ax, e, r"$e_z$", 0.09 * R * e / np.linalg.norm(e))
        angle_arcs(ax, e, D, 0.06 * R, 0.02 * R)
        frame(ax, "", (-R, R), (-R, R))                    # square like the other worlds, so the three boxes match

    def draw_circle(ax, E, D, i, regions=False, arrow=True, logits=True):
        """Week circle: times E on a circle, day decoders D, time i highlighted; regions = the seven
        sectors."""
        e, _g = E[i], max(np.abs(E).max(), np.abs(D).max()); R = 1.3 * _g
        if regions:
            region_fill(ax, D, hue7, R, fade=0.3)
        code_axes(ax, 0.92 * R, -0.92 * R, 0.92 * R)
        _rr, _tt = np.linalg.norm(E, axis=1).mean(), np.linspace(0, 2 * np.pi, 200)
        ax.plot(_rr * np.cos(_tt), _rr * np.sin(_tt), ls=":", lw=1.2, zorder=0,
                color="0.55" if regions else "0.8")                                      # the embedding circle
        ax.scatter(*E.T, c=np.arange(len(E)), cmap="twilight", s=46, zorder=3)
        _decoders(ax, e, D, days, (0.16 if logits else 0.06) * _g, numbers=logits)
        if arrow:
            _highlight(ax, e, r"$e_z$", 0.1 * R * e / np.linalg.norm(e))
        angle_arcs(ax, e, D, 0.08 * R, 0.03 * R)
        frame(ax, "", (-R, R), (-R, R))

    c_blue = plt.get_cmap(plot_utils.FIG.diverging_cmap)(0.08)
    c_red = plt.get_cmap(plot_utils.FIG.diverging_cmap)(0.92)
    yes_no = [to_rgb(plot_utils.FIG.colors["blue"]), to_rgb(plot_utils.FIG.colors["orange"])]
    cols5 = plt.get_cmap(plot_utils.FIG.diverging_cmap)(np.linspace(0.06, 0.94, 5))
    hue7 = plt.cm.twilight(np.arange(7) / 7)
    adjs = ["freezing", "cold", "moderate", "warm", "hot"]
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    WORDS = ["man", "king", "woman", "queen"]                 # bundles (royal, female) in cube_codes order
    HIGHLIGHT = {"binary": 3, "line": 8, "week": 10}           # queen, a warm-to-hot day, Wed night
    EQS = [  # each kernel as the explicit dot product it factorizes into, then its closed form
        "$\\log\\frac{p(\\mathrm{royal}{+} \\mid z)}{p(\\mathrm{royal}{-} \\mid z)} = (z_{\\mathrm{royal}},\\, z_{\\mathrm{female}})^{\\top}(\\alpha,\\, 0)$\n"
        "$= \\alpha\\, z_{\\mathrm{royal}}$",
        "$\\log\\tilde p(\\mathrm{warm} \\mid z) = (\\mu_z,\\, 1)^{\\top}(\\mu_{\\mathrm{warm}},\\, -\\frac{1}{2}\\mu_{\\mathrm{warm}}^2)$\n"
        "$= -\\frac{1}{2}(\\mu_z - \\mu_{\\mathrm{warm}})^2$",
        "$\\log\\tilde p(\\mathrm{Wed} \\mid z) = \\sqrt{r}(\\cos\\theta_z,\\, \\sin\\theta_z)^{\\top}\\, \\sqrt{r}(\\cos\\theta_{\\mathrm{Wed}},\\, \\sin\\theta_{\\mathrm{Wed}})$\n"
        "$= r\\cos(\\theta_z - \\theta_{\\mathrm{Wed}})$",
    ]
    DRAW = {"binary": draw_binary, "line": draw_line, "week": draw_circle}

    def worlds_row(fig, cells, vectors, regions=True):
        """The three worlds side by side in the gridspec row ``cells`` of ``fig``, from ``vectors[name] =
        (E, D)``: kernel equations above, each token's argmax cone behind."""
        for _k, (_name, _eq) in enumerate(zip(DRAW, EQS)):
            _a = fig.add_subplot(cells[_k])
            DRAW[_name](_a, *vectors[_name], HIGHLIGHT[_name], regions=regions)
            _p = _a.get_position()                     # one shared height for the three equations
            fig.text((_p.x0 + _p.x1) / 2, _p.y1 + 0.01, _eq, ha="center", va="bottom", fontsize=27, linespacing=1.5)


    return (
        DRAW,
        HIGHLIGHT,
        WORDS,
        adjs,
        cols5,
        days,
        gauge,
        hue7,
        worlds_row,
        yes_no,
    )


@app.cell
def _(factor_logcond, gauge, graph_conditional):
    WORLDS = {
        # words with two binary features, a yes/no token per feature (rank 2 at equal alpha)
        "binary": {"structure": "feature_cube", "m": 2, "decay": 1.0, "scale": 2.0,
                   "paired": True, "all_bundles": True},
        # continuous latents, coarse outcomes: 21 temperatures -> 5 adjectives, 28 times -> 7 days
        "line": {"structure": "chain", "cooccurrence": "gaussian", "n": 13, "m": 5, "gamma": 0.14},
        "week": {"structure": "ring", "cooccurrence": "gaussian", "n": 28, "m": 7, "gamma": 1.5},
    }

    def oracle_vectors(name):
        """The exact rank-2 code of a world's kernel, in canonical gauge: words E (n, 2), tokens D (m, 2)."""
        _, _E, _D, _ = factor_logcond(graph_conditional(WORLDS[name]))
        return gauge(name, _E[:, :2], _D[:2].T)

    oracle = {_name: oracle_vectors(_name) for _name in WORLDS}
    return WORLDS, oracle


@app.cell
def _(WORLDS, cka, gauge, graph_overrides, load_run, np, oracle):
    learned = {}
    for _name, _sp in WORLDS.items():
        # every centered kernel is rank 2, so a 2-D code loses nothing; SGD from small init
        # conserves E'E - D'D ~ 0, so the trained split is the balanced one
        _run = load_run(graph_overrides(_sp) +
                        ["runner=linear", "training_kwargs.optimizer=sgd", "training_kwargs.learning_rate=0.5", "model_kwargs.model_config.embed_dim=2"])
        _E, _D = _run["representation"], _run["readout"]   # the raw 2-D plane: D is drawn in it too
        _imb = np.abs(_E.T @ _E - _D.T @ _D).max() / np.abs(_D.T @ _D).max()
        print(f"{_name:8s} CKA={cka(_E, oracle[_name][0]):.3f}   |E'E - D'D| / |D'D| = {_imb:.4f}")
        learned[_name] = gauge(_name, _E, _D)
    return (learned,)


@app.cell
def _(
    DRAW,
    FIGURES,
    HIGHLIGHT,
    adjs,
    cols5,
    days,
    hue7,
    learned,
    np,
    plot_utils,
    plt,
    worlds_row,
    yes_no,
):
    from matplotlib.lines import Line2D
    C_HYP = C_CONF = "0.25"                                          # movements differ by dash, not colour

    def _movements(ax, P, ray):
        """Both movements as paths: ``P`` between hypotheses (dashed) and ``ray`` the confidence
        scaling of e_z (dotted)."""
        ax.plot(*P.T, ls=(0, (5, 3)), lw=2.4, color=C_HYP, zorder=7)
        ax.plot(*ray.T, ls=":", lw=2.6, color=C_CONF, zorder=7)

    def _chord(a, b, n=9):
        """Mixtures (1 - t) a + t b of two representations: the belief moving between hypotheses."""
        _t = np.linspace(0, 1, n)[:, None]
        return (1 - _t) * a + _t * b

    def _ray(e, n=9):
        """Scalings s e of a representation, from the item to the origin: uncertainty, low to high."""
        return np.linspace(1.0, 0.0, n)[:, None] * e

    def _probs(ax, P, D, colors, labels, ends, title, ylabel=True, order=None, column=True):
        """The readout along a movement: p(x) = softmax(P D^T) per outcome; a fixed-order label
        column at the right (``order``: indices top to bottom; default by final value)."""
        _lg = P @ D.T
        _Q = np.exp(_lg - _lg.max(1, keepdims=True)); _Q /= _Q.sum(1, keepdims=True)
        _t = np.linspace(0, 1, len(P))
        for _k, _c in enumerate(colors):
            ax.plot(_t, _Q[:, _k], color=_c, lw=2.2)
        _order = list(order) if order is not None else list(np.argsort(_Q[-1])[::-1])
        for _rank, _k in enumerate(_order if column else []):
            ax.text(1.04, 0.95 - 0.12 * _rank, labels[_k], color=colors[_k], fontsize=15, va="center")
        ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[0, 1], xticklabels=ends, yticks=[0, 0.5, 1])
        for _lab, _ha in zip(ax.get_xticklabels(), ["left", "right"]):   # end ticks read inward
            _lab.set_ha(_ha)
        if column:
            ax.set_yticklabels([])                              # the pair shares the left panel's axis
        if ylabel:
            ax.set_ylabel(r"$p(x \mid z)$", fontsize=18)
        ax.set_title(title, fontsize=21, pad=8)
        ax.tick_params(labelsize=15)
        plot_utils.FIG.style_axes(ax)

    _cols5_grey = list(cols5); _cols5_grey[2] = "0.5"                # moderate readable on white
    _Eb, _Db = learned["binary"]; _El, _Dl = learned["line"]; _Ew, _Dw = learned["week"]
    ORDER = {"binary": [0, 1], "line": [4, 3, 2, 1, 0], "week": list(range(7))}   # label columns, top to bottom
    MOVES = {  # world: (path between hypotheses, its end labels, ray item, readout decoders, colours, labels)
        "binary": (_chord(_Eb[2], _Eb[3]), ["woman", "queen"], HIGHLIGHT["binary"],
                   _Db[[0, 2]], yes_no, ["Royal +", "Royal −"]),                 # within the royal pair
        "line": (_El[1:12], ["very cold", "very hot"], HIGHLIGHT["line"], _Dl, _cols5_grey, adjs),   # along the line
        "week": (_Ew[0:18], ["Mon", "Fri"], HIGHLIGHT["week"], _Dw, hue7, days),                    # around the ring
    }

    def uncertainty_rows(fig, moves, probs):
        """Each trained world with its two movements drawn in (gridspec row ``moves``) and the readout
        along each movement below it (row ``probs``, a pair per world with a label column)."""
        for _k, _name in enumerate(DRAW):
            _a = fig.add_subplot(moves[_k])
            _inner = probs[_k].subgridspec(1, 3, width_ratios=[1, 1, 0.4], wspace=0.2)   # third cell: label column
            _pa, _pb = fig.add_subplot(_inner[0]), fig.add_subplot(_inner[1])
            _E, _D = learned[_name]
            DRAW[_name](_a, _E, _D, HIGHLIGHT[_name], regions=False, arrow=False, logits=False)
            _P, _ends, _i, _Dr, _colors, _labels = MOVES[_name]
            _Pr = _ray(_E[_i])
            _movements(_a, _P, _Pr)
            _probs(_pa, _P, _Dr, _colors, _labels, _ends, "which output", ylabel=True, order=ORDER[_name], column=False)
            _pa.title.set_color(C_HYP)
            _probs(_pb, _Pr, _Dr, _colors, _labels, ["low", "high"], "uncertainty", ylabel=False, order=ORDER[_name])
            _pb.title.set_color(C_CONF)

    # rows in panel units: worlds, a spacer for the movement legend, movements, readouts
    fig_all = plt.figure(figsize=(28, 25))
    _gs = fig_all.add_gridspec(4, 3, height_ratios=[1, 0.1, 1, 0.48], hspace=0.03, wspace=0.1,
                               left=0.01, right=0.99, top=0.92, bottom=0.02)
    worlds_row(fig_all, _gs[0, :].subgridspec(1, 3, wspace=0.1), learned)
    uncertainty_rows(fig_all, _gs[2, :].subgridspec(1, 3, wspace=0.1), _gs[3, :].subgridspec(1, 3, wspace=0.1))
    fig_all.legend([Line2D([], [], ls=(0, (5, 3)), lw=2.4, color=C_HYP), Line2D([], [], ls=":", lw=2.6, color=C_CONF)],
                   ["which output", "uncertainty"], loc="center", bbox_to_anchor=_gs[1, :].get_position(fig_all),
                   ncol=2, fontsize=26, frameon=False, handlelength=3.0, columnspacing=3.0)
    plot_utils.save_figure(fig_all, FIGURES / "concepts_worlds_uncertainty.png")

    return MOVES, ORDER


@app.cell
def _(mo):
    mo.md(r"""
    ### Drag the representation

    The same trained worlds, live: drag $e_z$ anywhere in the plane and the bars show
    $p(x \mid z) = \mathrm{softmax}(e_z D^\top)$ for the readout tokens. Along the manifold the winner
    changes; toward the origin every logit shrinks and the bars flatten.
    """)
    return


@app.cell
def _(DRAW, FIGURES, MOVES, ORDER, WORDS, adjs, days, learned, mo, np, plt):
    import json, html
    from matplotlib.colors import to_hex

    DRAG_WORLD = """<!doctype html><html><body style="margin:0;font-family:Avenir Next,Avenir,Helvetica Neue,sans-serif;color:#555;user-select:none">
    <div style="display:flex;align-items:center;gap:10px">
    <svg id="plane" width="__S__" height="__S__" style="touch-action:none;cursor:crosshair;overflow:visible">
    <defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#262626"/></marker>
    <marker id="ahg" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#aaa"/></marker></defs></svg>
    <svg id="bars" width="__B__" height="__S__"></svg></div>
    <script>
    const d = __DATA__, S = __S__, B = __B__, c = S / 2, k = (S / 2 - 40) / d.R;
    const X = v => c + k * v[0], Y = v => c - k * v[1];
    const ns = "http://www.w3.org/2000/svg", plane = document.getElementById("plane"), bars = document.getElementById("bars");
    function el(tag, attrs, parent, text) { const e = document.createElementNS(ns, tag); for (const a in attrs) e.setAttribute(a, attrs[a]);
      if (text !== undefined) e.textContent = text; (parent || plane).appendChild(e); return e; }
    el("line", {x1: 6, y1: c, x2: S - 6, y2: c, stroke: "#ddd"}); el("line", {x1: c, y1: 6, x2: c, y2: S - 6, stroke: "#ddd"});
    if (d.guide) el(d.guide === "closed" ? "polygon" : "polyline",
      {points: d.points.map(p => X(p) + "," + Y(p)).join(" "), fill: "none", stroke: "#bbb", "stroke-dasharray": "2 4"});
    d.decoders.forEach((v, i) => { const ro = d.readout.some(r => r[0] === v[0] && r[1] === v[1]), col = ro ? "#262626" : "#aaa";
      el("line", {x1: c, y1: c, x2: X(v), y2: Y(v), stroke: col, "stroke-width": 1.6, "marker-end": ro ? "url(#ah)" : "url(#ahg)"});
      const n = Math.hypot(v[0], v[1]), u = [v[0] / n, v[1] / n];
      const t = el("text", {x: X(v) + d.pad * u[0], y: Y(v) - d.pad * u[1], fill: col, "font-size": 13,
          "text-anchor": u[0] > 0.3 ? "start" : u[0] < -0.3 ? "end" : "middle",
          "dominant-baseline": u[1] > 0.3 ? "auto" : u[1] < -0.3 ? "hanging" : "middle"}, plane, d.decoder_labels[i]);
      const bb = t.getBBox(), over = Math.min(0, bb.x - 2) + Math.max(0, bb.x + bb.width - S + 2);   // keep the label inside the plane
      if (over) t.setAttribute("x", +t.getAttribute("x") - over); });
    d.points.forEach((p, i) => { el("circle", {cx: X(p), cy: Y(p), r: 4, fill: d.point_colors[i]});
      if (d.point_labels) el("text", {x: X(p), y: Y(p) - 10, "text-anchor": "middle", fill: "#888", "font-size": 13}, plane, d.point_labels[i]); });
    const ray = el("line", {x1: c, y1: c, stroke: "#262626", "stroke-width": 2.2, "stroke-dasharray": "6 5"});
    const knob = el("circle", {r: 7, fill: "#262626", stroke: "#fff", "stroke-width": 2, cursor: "grab"});
    const ez = el("text", {"font-size": 15, fill: "#262626", "font-style": "italic"}, plane, "e");
    el("text", {"font-size": 12, fill: "#262626", "font-style": "italic", x: 0, y: 0}, plane, "");
    el("text", {x: 8, y: 16, "font-size": 14, fill: "#555"}, bars, "p(x | z)");
    const rowH = Math.min(30, (S - 30) / d.order.length), x0 = 70, w = B - x0 - 40;
    const rows = d.order.map((kk, r) => { const y = 32 + r * rowH;
      el("text", {x: x0 - 8, y: y + 12, "text-anchor": "end", "font-size": 13, fill: d.colors[kk]}, bars, d.labels[kk]);
      const bar = el("rect", {x: x0, y: y, height: rowH - 8, width: 0, fill: d.colors[kk]}, bars);
      const val = el("text", {x: x0, y: y + 12, "font-size": 11, fill: "#777"}, bars, "");
      return {kk, bar, val}; });
    function update(e) { ray.setAttribute("x2", X(e)); ray.setAttribute("y2", Y(e));
      knob.setAttribute("cx", X(e)); knob.setAttribute("cy", Y(e)); ez.setAttribute("x", X(e) + 12); ez.setAttribute("y", Y(e) + 21);
      const lg = d.readout.map(r => r[0] * e[0] + r[1] * e[1]), m = Math.max(...lg), q = lg.map(l => Math.exp(l - m)), z = q.reduce((a, b) => a + b);
      rows.forEach(({kk, bar, val}) => { const p = q[kk] / z; bar.setAttribute("width", w * p);
        val.setAttribute("x", x0 + w * p + 5); val.textContent = p.toFixed(2); }); }
    let drag = false;
    const at = ev => { const r = plane.getBoundingClientRect(); return [(ev.clientX - r.left - c) / k, (c - (ev.clientY - r.top)) / k]; };
    plane.addEventListener("pointerdown", ev => { drag = true; update(at(ev)); });
    plane.addEventListener("pointermove", ev => { if (drag) update(at(ev)); });
    window.addEventListener("pointerup", () => { drag = false; });
    update(d.e);
    </script></body></html>"""

    def world_widget(name, size=300, bars=180):
        """Standalone HTML page: drag e_z around trained world ``name``; the bars are softmax(e_z D^T) over
        the readout tokens, recomputed in the browser."""
        _E, _D = learned[name]
        _P, _ends, _i, _Dr, _colors, _labels = MOVES[name]
        _R = {"binary": 1.5, "line": 1.2, "week": 1.3}[name] * max(np.abs(_E).max(), np.abs(_D).max())
        _pcol = {"binary": ["#999999"] * len(_E), "line": plt.cm.coolwarm(np.linspace(0, 1, len(_E))),
                 "week": plt.cm.twilight(np.arange(len(_E)) / len(_E))}[name]
        _data = dict(R=float(_R), e=_E[_i].tolist(), points=_E.tolist(), point_colors=[to_hex(_c) for _c in _pcol],
                     point_labels=WORDS if name == "binary" else None, decoders=_D.tolist(),
                     decoder_labels={"binary": ["Royal +", "Female +", "Royal −", "Female −"], "line": adjs, "week": days}[name],
                     readout=_Dr.tolist(), colors=[to_hex(_c) for _c in _colors], labels=list(_labels), order=ORDER[name],
                     guide={"binary": None, "line": "open", "week": "closed"}[name],
                     pad={"binary": 9, "line": 24, "week": 9}[name])          # label step past the tip, px
        return DRAG_WORLD.replace("__DATA__", json.dumps(_data)).replace("__S__", str(size)).replace("__B__", str(bars))

    def save_html(doc, name):
        """Write a standalone page beside the figures (``figures/html/name.html``); returns its path."""
        _path = FIGURES / "html" / f"{name}.html"
        _path.parent.mkdir(exist_ok=True); _path.write_text(doc)
        return _path

    _docs = {_name: world_widget(_name) for _name in DRAW}
    for _name, _doc in _docs.items():
        save_html(_doc, f"concepts_drag_{_name}")
    save_html("<!doctype html><html><body style='margin:0;display:flex;gap:16px'>" +
              "".join(f'<iframe srcdoc="{html.escape(_doc)}" width="504" height="312" frameborder="0"></iframe>' for _doc in _docs.values()) +
              "</body></html>", "concepts_drag")
    mo.hstack([mo.Html(f'<iframe srcdoc="{html.escape(_doc)}" width="504" height="312" frameborder="0"></iframe>') for _doc in _docs.values()],
              justify="start", gap=1)

    return


if __name__ == "__main__":
    app.run()
