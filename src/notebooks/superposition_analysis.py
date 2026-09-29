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
    # Feature geometry of a rank-limited softmax

    Two parts. **Part I** checks the closed forms of `notes/superposition_new.tex` on the signed
    feature cube: features $z_i \in \{-1, 0, 1\}$, one yes/no token pair per feature, logits
    $\pm\alpha_i z_i$, one softmax over the $2m$ tokens. Each case is a trained linear network
    (three seeds) beside the note's closed form (the rank-$d$ optimum of the same objective, exponential-family PCA,
    `runner=optimum`, is left to an appendix). These small networks are trained by gradient descent from small
    init, which keeps $E^\top E - D^\top D$ conserved, so their saved representation and readout are balanced and
    are drawn as saved, in one frame. **Part II** is the Bernoulli feature cube of the sweep: the trained linear network beside the optimum,
    read from the first-order vectors $E_i$ of the least-squares fit
    $h(z) \approx \bar h + \sum_i E_i \tilde z_i$ on the standardized features
    $\tilde z_i = (z_i - \bar z_i)/\mathrm{std}(z_i)$; correlated cells standardize the predictive
    modes $z\,C^{-1/2}$. Five runs per cell, task and training seeds coupled.
    """)
    return


@app.cell
def _(mo):
    import os, sys
    import numpy as np

    ROOT = mo.notebook_dir().parent.parent    # src/notebooks -> <repo>
    os.chdir(ROOT); sys.path.insert(0, str(ROOT))
    from hydra.core.override_parser.overrides_parser import OverridesParser
    from omegaconf import OmegaConf
    from src.code.analysis_utils import feature_vectors, has_run, leverage, load_run, rigid_map
    from src.code.cases import CASES, SEEDS as CASE_SEEDS, case_overrides
    from src.code.geometry import excess_ce, factor_logcond, frequency_threshold, modes
    from src.code.train import resolve_config
    from src.code import plot_utils
    import matplotlib.pyplot as plt

    # the sweep is the setting's: conf/setting/superposition.yaml declares it under
    # hydra.sweeper.params, and `python -m src.code.train -m runner=<class> setting=superposition`
    # trains it; here the same params name the cells to read
    SETTING = "superposition"
    _params = OmegaConf.load(ROOT / "conf" / "setting" / f"{SETTING}.yaml").hydra.sweeper.params
    GRID = {_o.key_or_group: list(_o.sweep_string_iterator()) if _o.is_sweep_override() else [str(_o.value())]
            for _o in OverridesParser.create().parse_overrides([f"{_k}={_v}" for _k, _v in _params.items()])}
    P_KEY, B_KEY, R_KEY = "structure_kwargs.p", "structure_kwargs.beta", "structure_kwargs.rho"
    D_KEY, S_KEY = "model_kwargs.model_config.embed_dim", "random_seed"
    D_GRID, SEEDS = GRID[D_KEY], GRID[S_KEY]

    def cell_overrides(runner, p, b, r, d, s):
        """One cell's run: the setting, the model class and one value per swept key."""
        _picks = {P_KEY: p, B_KEY: b, R_KEY: r, D_KEY: d, S_KEY: s}
        return [f"setting={SETTING}", f"runner={runner}"] + [f"{_k}={_picks.get(_k, _v[0])}" for _k, _v in GRID.items()]

    CONFIG = resolve_config(cell_overrides("linear", "0.3", "0.5", "0.0", D_GRID[0], SEEDS[0]))   # the figure cache lives under the reference cell
    COL = plot_utils.FIG.colors
    CLASSES = ["linear", "paired_features", "additive_features", "optimum"]   # every model class a cell can hold
    return (
        B_KEY,
        CASES,
        CASE_SEEDS,
        CLASSES,
        COL,
        CONFIG,
        D_GRID,
        GRID,
        R_KEY,
        SEEDS,
        case_overrides,
        cell_overrides,
        excess_ce,
        factor_logcond,
        feature_vectors,
        frequency_threshold,
        has_run,
        leverage,
        load_run,
        modes,
        np,
        plot_utils,
        plt,
        rigid_map,
    )


@app.cell
def _(
    CLASSES,
    SEEDS,
    cell_overrides,
    feature_vectors,
    has_run,
    load_run,
    mo,
    modes,
    np,
    rigid_map,
):
    # loaded cells, filled on demand the first time a figure asks for one
    SW = {}

    def sweep_rows(p, b, r, d, classes):
        """One cell, per seed: the cell's coordinates X (features, or predictive modes), the
        raw features Z (the target's coordinates), ladder alpha, and for every model class present (the optimum included) its saved
        representation and readout ('model') and the first-order vectors E_i of the least-squares
        fit h(z) ~ h_bar + sum_i E_i x_i ('E')."""
        _rows = []
        for _s in SEEDS:
            _runs = {_c: load_run(cell_overrides(_c, p, b, r, d, _s)) for _c in classes}
            _ds = _runs["linear"]["dataset"]
            _, _X = modes(_ds.info["alpha"], _ds.features, float(r))
            _model = {_c: (_runs[_c]["representation"], _runs[_c]["readout"]) for _c in classes}
            _rows.append(dict(X=_X, Z=np.asarray(_ds.features, float), alpha=_ds.info["alpha"], model=_model,
                              E={_c: feature_vectors(_H, _X)[0] for _c, (_H, _D) in _model.items()}))
        return _rows

    HEAD = 3                                          # the first three features by alpha are the head ...
    CUT = 0.2                                         # ... the rest with leverage >= 1/5 are the superposed band; below is not drawn

    def whiten(V):
        """V (V'V)^{-1/2} up to a rotation: rows are the whitened vectors, squared lengths the leverage."""
        _U, _, _Vt = np.linalg.svd(np.asarray(V, float), full_matrices=False)
        return _U @ _Vt

    def aligned_mean(mats, ref):
        """The shared geometry of a set of frames: whiten each, rotate it rigidly onto ``ref`` (orthogonal
        Procrustes over all rows), average. Seed-specific wobble cancels; what every seed agrees on stays."""
        return np.mean([whiten(_M) @ rigid_map(whiten(_M), ref) for _M in mats], 0)


    def load_cell(p, b, r, d):
        """One loaded cell, cached in SW: every model class whose seeds are all saved; None
        until the trained linear network and the optimum are."""
        if (p, b, r, d) not in SW:
            _have = [_c for _c in CLASSES if all(has_run(cell_overrides(_c, p, b, r, d, _s)) for _s in SEEDS)]
            SW[(p, b, r, d)] = sweep_rows(p, b, r, d, _have) if {"linear", "optimum"} <= set(_have) else None
        return SW[(p, b, r, d)]

    NOT_YET = mo.md("this setting is not trained yet")
    return CUT, HEAD, NOT_YET, aligned_mean, load_cell, whiten


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part I — the canonical cases

    Signed features, one yes/no token pair each, the trained network beside the note's closed form. Cases from
    `src/code/cases.py`; train them with `python -m src.code.train <overrides>` for every line of
    `case_overrides` (linear and optimum runners, seeds 0–2).
    """)
    return


@app.cell
def _(
    CASE_SEEDS,
    case_overrides,
    excess_ce,
    factor_logcond,
    load_run,
    np,
    rigid_map,
):
    def case_runs(name, d):
        """One canonical case at rank d: the task (raw features Z, the ladder alpha, the log target L,
        and the patterns and mixing A when mixed) and, per model class, the saved representation E (n, d)
        and readout D (2m, d) of every seed in the run's own gauge, each rotated onto seed 0's E by orthogonal
        Procrustes; 'svd' is the rank-d SVD truncation of the target table, the weak-logit closed form."""
        _runs = {_c: [load_run(case_overrides(name, _c, d, _s)) for _s in CASE_SEEDS] for _c in ("linear", "optimum")}
        _ds = _runs["linear"][0]["dataset"]
        _task = dict(Z=np.asarray(_ds.features, float), alpha=np.asarray(_ds.info["alpha"], float),
                     L=np.log(_ds.P.cpu().numpy().astype(float)), patterns=_ds.info.get("patterns"), A=_ds.info.get("A"))
        _P = np.exp(_task["L"])
        _task["excess_ce"] = {_c: [excess_ce(_r["representation"] @ _r["readout"].T, _task["L"], _P) for _r in _rs] for _c, _rs in _runs.items()}
        _ref = _runs["linear"][0]["representation"]                                        # every frame rotated onto seed 0's
        _vectors = {}
        for _c, _rs in _runs.items():
            _vectors[_c] = []
            for _r in _rs:
                _E, _D = _r["representation"], _r["readout"]
                _R = rigid_map(_E, _ref)
                _vectors[_c].append((_E @ _R, _D @ _R))
        # the weak-logit closed form: the rank-d SVD truncation of the target table, scale split evenly
        _, _Et, _Dt, _ = factor_logcond(_task["L"])
        _Et, _Dt = _Et[:, :d], _Dt[:d].T
        _R = rigid_map(_Et, _ref)
        _vectors["svd"] = [(_Et @ _R, _Dt @ _R)]
        return _task, _vectors


    def nearest_token_cosine(D):
        """The largest cosine between two distinct token directions of a decoder D (2m, d): the
        angular gap of the closest pair. A regular 2m-gon gives cos(pi / m)."""
        _n = np.linalg.norm(D, axis=1)
        _C = D @ D.T / np.outer(_n, _n)
        np.fill_diagonal(_C, -1.0)
        return _C.max()

    return case_runs, nearest_token_cosine


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 1 — weak logits: from cube to rectangle
    Three signed attributes at weak strength, strongest first: royal (+royal, −commoner), female (+female,
    −male), adult (+adult, −child), 0 unstated; all 27 states, the eight corners the words king, queen,
    prince, princess, man, woman, boy, girl. Columns: the trained network and the weak-logit closed form, the
    rank-$d$ SVD truncation of the target table. At $d = 3$ the representation is the box of the three
    attributes; at $d = 2$ the weak-logit solution is the SVD truncation, which keeps the two strongest and
    drops adult: king lands on prince, queen on princess, man on boy, woman on girl. The outline joins the eight corners; the three edges at the front
    corner are coloured and named by the attribute they change; at $d = 2$ the two adult tokens' readouts are
    drawn from the origin, where they have collapsed.
    """)
    return


@app.cell
def _(COL, CONFIG, case_runs, np, plot_utils, plt):
    _task3, _v3 = case_runs("weak", 3)
    _task2, _v2 = case_runs("weak", 2)
    _Z = _task3["Z"]
    _ATTR = [("Royal", COL["blue"]), ("Female", COL["orange"]), ("Adult", COL["green"])]     # strongest first
    _WORDS = {(1, -1, 1): "king", (1, 1, 1): "queen", (1, -1, -1): "prince", (1, 1, -1): "princess",
              (-1, -1, 1): "man", (-1, 1, 1): "woman", (-1, -1, -1): "boy", (-1, 1, -1): "girl"}
    _corners = np.flatnonzero((np.abs(_Z) == 1).all(1))
    _outline = [(_a, _b, int(np.flatnonzero(_Z[_a] != _Z[_b])[0])) for _a in _corners for _b in _corners
                if _a < _b and np.abs(_Z[_a] - _Z[_b]).sum() == 2]                        # the cube's 12 edges, by attribute
    _face = np.flatnonzero((np.abs(_Z[:, :2]) == 1).all(1) & (_Z[:, 2] == 0))
    _rect = [(_a, _b, int(np.flatnonzero(_Z[_a] != _Z[_b])[0])) for _a in _face for _b in _face
             if _a < _b and np.abs(_Z[_a] - _Z[_b]).sum() == 2]                           # the rectangle's 4 edges

    from mpl_toolkits.mplot3d.proj3d import proj_transform as _proj_transform
    _fig = plt.figure(figsize=(13, 12), layout="constrained")
    for _col, (_lbl, _c) in enumerate([("trained", "linear"), ("theory: SVD truncation", "svd")]):
        _E3, _D3 = _v3[_c][0]
        _E2, _D2 = _v2[_c][0]
        # d = 3: the box, white frame
        _ax = _fig.add_subplot(2, 2, _col + 1, projection="3d")
        _elev, _azim = 16, -49                                                                  # nearly face-on: front square, back square behind
        _view = np.array([np.cos(np.radians(_elev)) * np.cos(np.radians(_azim)), np.cos(np.radians(_elev)) * np.sin(np.radians(_azim)), np.sin(np.radians(_elev))])
        _depth = _E3 @ _view                                                                  # toward the viewer
        _span = _depth.max() - _depth.min()
        _labels = []                                                                          # (position, text, colour, size), placed once the view is set
        _near = lambda _p: (_depth.max() - _p @ _view) / _span                                 # 1 at the viewer's side, 0 at the back
        _front = _corners[np.argmax([_near(_E3[_c]) for _c in _corners])]                       # the corner nearest the viewer
        _centre = _E3.mean(0)
        for _a, _b, _k in _outline:
            _f = _near((_E3[_a] + _E3[_b]) / 2)
            if _front in (_a, _b):
                _ax.plot(*zip(_E3[_a], _E3[_b]), color=_ATTR[_k][1], lw=2.6)
                _far = _b if _a == _front else _a
                _labels.append(("edge", (_E3[_front], _E3[_far]), _ATTR[_k][0], _ATTR[_k][1], 11))
            else:
                _ax.plot(*zip(_E3[_a], _E3[_b]), color=str(0.75 - 0.6 * _f), lw=0.6 + 0.7 * _f)     # back edges lighter and thinner
        for _n, _z in enumerate(_Z):
            _is_corner = (np.abs(_z) == 1).all()
            _on_edge = (np.abs(_z) == 1).sum() == 2
            _f = _near(_E3[_n])
            _ax.scatter(*_E3[_n], color="0.25" if _is_corner else "0.72" if _on_edge else "0.86", s=48 if _is_corner else 14 if _on_edge else 9, depthshade=False, zorder=3)
            if _is_corner:
                _labels.append(("vertex", _E3[_n], _WORDS[tuple(int(_x) for _x in _z)], "0.15", 10))
        _lim = 1.2 * np.abs(_E3).max()
        _ax.set_xlim(-_lim, _lim); _ax.set_ylim(-_lim, _lim); _ax.set_zlim(-_lim, _lim)
        _ax.set_proj_type("ortho"); _ax.view_init(elev=_elev, azim=_azim); _ax.set_box_aspect((1, 1, 1), zoom=1.25)
        _M = _ax.get_proj()
        _P2 = lambda _p: np.array(_proj_transform(*_p, _M)[:2])                                  # a 3-D point in the axes' projected plane
        _c2 = _P2(_centre)
        for _kind, _where, _txt, _colr, _fs in _labels:                                        # anchors in projected coordinates, offsets in points
            if _kind == "vertex":
                _xy = _P2(_where)
                _dir = _xy - _c2
            else:
                _p0, _p1 = _P2(_where[0]), _P2(_where[1])
                _xy = _p0 + 0.45 * (_p1 - _p0)
                _dir = np.array([-(_p1 - _p0)[1], (_p1 - _p0)[0]])                              # perpendicular to the edge ...
                _dir *= np.sign(_dir @ (_xy - _c2)) or 1.0                                        # ... on the side away from the box
            _dir = _dir / (np.linalg.norm(_dir) + 1e-12)
            _ax.annotate(_txt, xy=_xy, xytext=12 * _dir, textcoords="offset points", color=_colr, fontsize=_fs,
                         fontweight="medium" if _fs > 10 else None,
                         ha="left" if _dir[0] > 0.3 else "right" if _dir[0] < -0.3 else "center",
                         va="bottom" if _dir[1] > 0.3 else "top" if _dir[1] < -0.3 else "center",
                         bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.8))
        _ax.set_axis_off()
        _ax.set_title(f"d = 3 — {_lbl}", fontsize=13)
        # d = 2: the rectangle, the three adult layers stacked
        _ax = _fig.add_subplot(2, 2, _col + 3)
        _front2 = _face[np.argmin(_E2[_face].sum(1))]                                         # the lower-left corner
        _centre2 = _E2.mean(0)
        for _a, _b, _k in _rect:
            if _front2 in (_a, _b):
                _ax.plot(*zip(_E2[_a], _E2[_b]), color=_ATTR[_k][1], lw=2.4, zorder=1)
                _far = _b if _a == _front2 else _a
                _at = _E2[_front2] + 0.3 * (_E2[_far] - _E2[_front2])
                _side = _at - _centre2
                _ax.annotate(_ATTR[_k][0], _at + 0.04 * np.abs(_E2).max() * np.sign(_side) * (np.abs(_side) > 1e-9), fontsize=12, color=_ATTR[_k][1], fontweight="medium",
                             ha="left" if _side[0] > 1e-9 else "right" if _side[0] < -1e-9 else "center",
                             va="bottom" if _side[1] > 1e-9 else "top" if _side[1] < -1e-9 else "center")
            else:
                _ax.plot(*zip(_E2[_a], _E2[_b]), color="0.2", lw=0.8, zorder=1)
        _is_corner2 = (np.abs(_Z[:, :2]) == 1).all(1)
        _ax.scatter(*_E2[~_is_corner2].T, color="0.72", s=18, zorder=3)
        _ax.scatter(*_E2[_is_corner2].T, color="0.25", s=70, zorder=4)
        for _k in (2, 5):                                                                     # the dropped attribute's two tokens, from the origin
            _ax.annotate("", xy=_D2[_k], xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=1.8, color=COL["green"], shrinkA=0, shrinkB=0), zorder=5)
        _ax.annotate("Adult", _D2[2] + 0.06 * np.abs(_E2).max() * np.array([1.0, 0.6]), ha="left", va="bottom", fontsize=12, color=COL["green"], fontweight="medium")
        for _a, _b in [((1, -1), "king · prince"), ((1, 1), "queen · princess"), ((-1, -1), "man · boy"), ((-1, 1), "woman · girl")]:
            _n = int(np.flatnonzero((_Z[:, 0] == _a[0]) & (_Z[:, 1] == _a[1]) & (_Z[:, 2] == 0))[0])
            _ax.annotate(_b, _E2[_n] + 0.05 * np.abs(_E2).max() * np.sign(_E2[_n]), ha="left" if _E2[_n][0] > 0 else "right",
                         va="bottom" if _E2[_n][1] > 0 else "top", fontsize=11, color="0.15")
        _lim = 1.3 * np.abs(_E2).max()
        _ax.set_xlim(-_lim, _lim); _ax.set_ylim(-_lim, _lim); _ax.set_aspect("equal")
        plot_utils.FIG.clean(_ax)
        _ax.set_title(f"d = 2 — {_lbl}", fontsize=13)
    _out = plot_utils.save_figure(_fig, plot_utils.figure_path(CONFIG, "fig1_cube_to_rectangle"))
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 2 — sharp singletons: the regular $2m$-gon
    Eight features of equal strength $\alpha = 12$, one active at a time, $d = 2$. With one token to
    win per state, the sharp limit is a point packing: the sixteen token directions spread evenly on the
    circle, the closest pair at $\cos(\pi/8) = 0.924$, and which token sits at which vertex is free, since
    the only thing that distinguishes a feature's other token from an unrelated one is a logit of $-\alpha$
    among the losers. Left: one trained seed, each readout $D_{i\pm}$ an arrow from the origin, small dots the
    state representations. Right: the packing.
    """)
    return


@app.cell
def _(COL, CONFIG, case_runs, nearest_token_cosine, np, plot_utils, plt):
    _task, _v = case_runs("sharp", 2)
    _m = len(_task["alpha"])
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 6), layout="constrained")
    _mu_all = np.mean([nearest_token_cosine(_D) for _, _D in _v["linear"]])                  # the 3-seed mean, for the title
    _E0, _D0 = _v["linear"][0]                                                                 # one seed drawn: each seed is its own 16-gon with its own pairing
    _r = np.linalg.norm(_D0, axis=1).mean()
    _phi = np.arctan2(_D0[0, 1], _D0[0, 0])
    _th = _phi + np.pi * np.arange(2 * _m) / _m
    _poly = _r * np.c_[np.cos(_th), np.sin(_th)]
    _v = dict(_v, theory=[(_poly, _poly)])                                                     # the point packing: 2m tokens on the regular 2m-gon, any pairing
    for _ax, (_lbl, _c, _colr) in zip(_axes, [("trained (seed 0)", "linear", COL["blue"]), ("theory: 16 tokens packed on the circle", "theory", "0.45")]):
        _ax.plot(*np.vstack([_poly, _poly[:1]]).T, color="0.85", lw=1.0, zorder=0)
        _E, _D = _v[_c][0]
        for _k in range(2 * _m):                                                               # each token's readout, from the origin
            _ax.annotate("", xy=_D[_k], xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", lw=1.4, color=_colr, shrinkA=0, shrinkB=0), zorder=2)
        if _c == "linear":
            _ax.scatter(*_E.T, color="0.3", s=9, zorder=4)
            for _i in range(_m):
                _ax.annotate(f"{_i + 1}+", _D[_i] * 1.16, ha="center", va="center", fontsize=10, color="0.35")
                _ax.annotate(f"{_i + 1}−", _D[_m + _i] * 1.16, ha="center", va="center", fontsize=10, color="0.35")
        _lim = 1.3 * _r
        _ax.set_xlim(-_lim, _lim); _ax.set_ylim(-_lim, _lim); _ax.set_aspect("equal")
        plot_utils.FIG.clean(_ax)
        _ax.set_title(f"{_lbl}: closest tokens at cos {nearest_token_cosine(_D):.3f}" + (f" (3 seeds {_mu_all:.3f})" if _c == "linear" else ""), fontsize=12)
    _out = plot_utils.save_figure(_fig, plot_utils.figure_path(CONFIG, "fig2_polygon"))
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 3 — unequal frequencies: a dedicated head and a shared plane
    Sixteen features of equal strength $\alpha = 1.35$, one active at a time, at Zipf input frequencies
    $\pi_i \propto 1/i$, $d = 5$. Left: the three most frequent features keep one direction each (seed 0).
    Middle: the next three share the remaining plane, whitened vectors $E(E^\top E)^{-1/2}$ of the $+e_i$
    states, squared length the leverage; the $-e_i$ states are antipodal. Right: leverage over three seeds,
    with the note's linear-instability band ($\pi_j / \pi_i > \rho_c$ for a kept $i$ and a dropped $j$) shaded.
    """)
    return


@app.cell
def _(
    Arrow3D,
    CASES,
    CASE_SEEDS,
    COL,
    CONFIG,
    basis3,
    case_overrides,
    frequency_threshold,
    leverage,
    load_run,
    np,
    plot_utils,
    plt,
    whiten,
):
    _case = CASES["frequency"]
    _d, _alpha0 = _case["d"][0], _case["spec"]["scale"]
    _runs = [load_run(case_overrides("frequency", "linear", _d, _s)) for _s in CASE_SEEDS]
    _Z = np.asarray(_runs[0]["dataset"].features, float)
    _m = _Z.shape[1]
    _plus = [int(np.flatnonzero(_Z[:, _i] == 1)[0]) for _i in range(_m)]                     # the +e_i states
    _levs = np.array([leverage(_r["representation"][_plus])[0] for _r in _runs])             # (seeds, m)
    # the note's linear-instability band at this alpha: a kept feature i <= d or a dropped j > d whose pair
    # (i, j) has pi_j / pi_i > rho_c
    _pi = 1.0 / (1.0 + np.arange(_m))
    _rho = frequency_threshold(_alpha0, _m)
    _band = [_i for _i in range(_d) if _pi[_d] / _pi[_i] > _rho] + [_j for _j in range(_d, _m) if _pi[_j] / _pi[_d - 1] > _rho]
    _head = [_i for _i in range(_d) if _i < min(_band)]
    # seed 0 drawn: whitened +e_i vectors (squared length = leverage), the head's span and the plane beside it
    _W = whiten(_runs[0]["representation"][_plus])
    _Q = np.linalg.qr(_W[_head].T)[0].T                                                      # head span (3, d)
    _rest = _W - _W @ _Q.T @ _Q
    _P = np.linalg.svd(_rest, full_matrices=False)[2][:_d - len(_head)]                       # the plane (2, d)
    _on = [_i for _i in range(len(_head), _m) if _levs[:, _i].mean() > 0.02]
    _FC = {_i: plt.cm.viridis(_k / max(1, len(_band) - 1) * 0.85) for _k, _i in enumerate(_band)}

    _fig = plt.figure(figsize=(15, 5), layout="constrained")
    _axh = _fig.add_subplot(1, 3, 1, projection="3d")
    _B = basis3(_W[_head], _W, True)                                                         # the head along the axes
    for _i in _head:
        _v = _W[_i] @ _B.T
        _axh.add_artist(Arrow3D(_v, arrowstyle="-|>", mutation_scale=16, lw=2.4, color="0.25", shrinkA=0, shrinkB=0))
        _axh.text(*(1.08 * _v), str(_i + 1), fontsize=12, ha="center", va="center")
    for _a in range(3):                                                                       # the unit cube the three span
        for _b in (0, 1):
            for _c in (0, 1):
                _p0 = np.zeros(3); _p0[(_a + 1) % 3] = _b; _p0[(_a + 2) % 3] = _c
                _p1 = _p0.copy(); _p1[_a] = 1
                _axh.plot(*zip(_p0, _p1), color="0.85", lw=0.8)
    _axh.set_xlim(0, 1.1); _axh.set_ylim(0, 1.1); _axh.set_zlim(0, 1.1); _axh.set_box_aspect((1, 1, 1), zoom=0.95)
    _axh.set_proj_type("ortho"); _axh.view_init(elev=24, azim=35); _axh.set_axis_off()
    _axh.set_title(f"head: features {_head[0] + 1}–{_head[-1] + 1}, one direction each", fontsize=12)

    _axp = _fig.add_subplot(1, 3, 2)
    _th = np.linspace(0, 2 * np.pi, 200)
    _axp.plot(np.cos(_th), np.sin(_th), color="0.85", lw=1.0, zorder=0)                        # leverage 1
    for _i in _on:
        _v = _W[_i] @ _P.T
        _axp.annotate("", xy=_v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", lw=2.2, color=_FC.get(_i, "0.5"), shrinkA=0, shrinkB=0))
        _axp.annotate(str(_i + 1), _v / np.linalg.norm(_v) * (np.linalg.norm(_v) + 0.12), ha="center", va="center", fontsize=12)
    _axp.set_xlim(-1.2, 1.2); _axp.set_ylim(-1.2, 1.2); _axp.set_aspect("equal")
    plot_utils.FIG.clean(_axp)
    _axp.set_title(f"the plane beside the head: features {_on[0] + 1}–{_on[-1] + 1} share it", fontsize=12)

    _axb = _fig.add_subplot(1, 3, 3)
    _n = 10
    _axb.axvspan(min(_band) + 0.5, max(_band) + 1.5, color=COL["orange"], alpha=0.12, lw=0)
    _axb.text((min(_band) + max(_band)) / 2 + 1, 1.08, rf"theory: unstable band ($\rho_c$ = {_rho:.2f})", ha="center", fontsize=10, color=COL["orange"])
    _axb.bar(np.arange(1, _n + 1), _levs[:, :_n].mean(0), color=[_FC.get(_i, "0.25") for _i in range(_n)], width=0.7)
    for _l in _levs:
        _axb.scatter(np.arange(1, _n + 1), _l[:_n], color="black", s=10, zorder=3)
    _axb.set_xticks(np.arange(1, _n + 1)); _axb.set_ylim(0, 1.15)
    _axb.set_xlabel(r"feature $i$ (by frequency, $\pi_i \propto 1/i$)"); _axb.set_ylabel(r"leverage $d_i$")
    plot_utils.FIG.style_axes(_axb)
    _axb.set_title("trained leverage, 3 seeds", fontsize=12)
    _fig.suptitle(rf"$m$ = {_m}, $d$ = {_d}, $\alpha$ = {_alpha0}")
    _out = plot_utils.save_figure(_fig, plot_utils.figure_path(CONFIG, "fig3_frequency"))
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part II — the general case

    The Bernoulli feature cube of the sweep: $m = 64$ features active with probability $p$ on the
    ladder $\alpha_i = 8\,(i+1)^{-\beta}$, one token per feature, $d \in \{3, 6, 12\}$.
    """)
    return


@app.cell
def _(CUT, HEAD, np, plot_utils, plt):
    # the frame drawing of figures 3 and 4
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d.proj3d import proj_transform

    class Arrow3D(FancyArrowPatch):
        """A FancyArrowPatch from the origin to a 3-D point: a filled head, unlike quiver's three lines."""
        def __init__(self, xyz, **kw):
            super().__init__((0, 0), (0, 0), **kw); self._xyz = xyz
        def do_3d_projection(self, renderer=None):
            _x0, _y0, _z0 = proj_transform(0, 0, 0, self.axes.M)
            _x1, _y1, _z1 = proj_transform(*self._xyz, self.axes.M)
            self.set_positions((_x0, _y0), (_x1, _y1))
            return min(_z0, _z1)

    def band(lev):
        """The superposed band: features with leverage above CUT, the head excluded."""
        return np.flatnonzero(lev > CUT)[HEAD:]

    def basis3(group, frame, tripod):
        """A view of the group. The head: Gram-Schmidt on its three vectors, the first along z, the next along
        x, the third along y. A superposed set: its top-3 principal directions, which spread it. Completed
        from the whole frame when the group has fewer than three vectors."""
        if tripod:
            _Qm, _R = np.linalg.qr(group[:3].T)
            _Q = (_Qm * np.sign(np.diag(_R))).T                                         # each axis points along its own vector
        else:
            _Q = np.linalg.svd(group, full_matrices=False)[2][:3]
        if len(_Q) < 3:
            _rest = frame - frame @ _Q.T @ _Q                                           # the frame outside the group's span
            _Q = np.vstack([_Q, np.linalg.svd(_rest, full_matrices=False)[2][:3 - len(_Q)]])
        return _Q[[1, 2, 0]]                                                            # x <- second, y <- third, z <- head

    # the house diverging map with its pale middle cut out: blue -> light blue | light red -> red over log alpha,
    # so no arrow washes out on the white page
    _vlag = plt.get_cmap(plot_utils.FIG.diverging_cmap)
    ALPHA_CMAP = plt.matplotlib.colors.ListedColormap(np.vstack([_vlag(np.linspace(0.0, 0.36, 128)), _vlag(np.linspace(0.64, 1.0, 128))]), name="vlag_cut")

    def frame_panel(ax, W, idx, B, title, alpha, norm, zoom=1.6, box=True):
        """The vectors W[idx] in the 3-D view B (3, d), each an arrow coloured by alpha; ``box`` draws the axes' box."""
        _xyz = W[idx] @ B.T
        for _k, _v in zip(idx, _xyz):
            ax.add_artist(Arrow3D(_v, arrowstyle="-|>", mutation_scale=16, lw=2.4, color=ALPHA_CMAP(norm(alpha[_k])), shrinkA=0, shrinkB=0))
        ax.set_xlim(-0.85, 0.85); ax.set_ylim(-0.85, 0.85); ax.set_zlim(-0.85, 0.85); ax.set_box_aspect((1, 1, 1), zoom=zoom)
        ax.set_proj_type("ortho"); ax.view_init(elev=24, azim=-45)                     # no perspective, x and y foreshortened alike
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        for _pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            _pane.pane.set_facecolor((1, 1, 1, 0))
        if not box:
            ax.set_axis_off()
        ax.set_title(title, fontsize=11)

    return ALPHA_CMAP, Arrow3D, band, basis3, frame_panel


@app.cell
def _(mo):
    mo.md(r"""
    ## Figure 4 — the sparsity sweep
    Sparsity $p$ falls left to right; each mega-row is one $d$, largest first. Per cell, a predicted block and a trained
    block: the whitened first-order vectors $E(E^\top E)^{-1/2}$ (squared length the leverage $d_i$) of the superposed band
    (the features past the top three with $d_i > 0.2$) in its PCA frame, coloured by $\alpha_i$ (the top three join when the band has
    fewer than three vectors); the Gram of the shared frame (top 16 features, one colour scale over every panel, set by the
    largest off-diagonal entry); and the leverage $d_i$ of the first $3d$ features (seed mean).
    """)
    return


@app.cell
def _(B_KEY, GRID, R_KEY, mo):
    P_SWEEP = ["0.3", "0.15", "0.05"]
    SEL_F7 = mo.ui.dictionary({
        "beta": mo.ui.dropdown(options=GRID[B_KEY], value="0.5", label="ladder slope β"),
        "rho": mo.ui.dropdown(options=GRID[R_KEY], value="0.0", label="correlation ρ"),
    })
    return P_SWEEP, SEL_F7


@app.cell
def _(
    ALPHA_CMAP,
    CONFIG,
    CUT,
    D_GRID,
    NOT_YET,
    P_SWEEP,
    SEL_F7,
    aligned_mean,
    band,
    basis3,
    frame_panel,
    leverage,
    load_cell,
    mo,
    np,
    plot_utils,
    plt,
    whiten,
):
    _b, _r = SEL_F7.value["beta"], SEL_F7.value["rho"]
    _cells = {(_d, _p): load_cell(_p, _b, _r, _d) for _d in D_GRID for _p in P_SWEEP}
    mo.stop(any(_rows is None for _rows in _cells.values()), NOT_YET)
    _K = 16
    _SIDES = [("optimum", "predicted"), ("linear", "trained")]

    _frames = {_key: {_c: aligned_mean([_row["E"][_c] for _row in _rows], whiten(_rows[0]["E"]["optimum"])) for _c, _ in _SIDES}
               for _key, _rows in _cells.items()}
    _grams = {(*_key, _c): _W[:_K] @ _W[:_K].T for _key, _fr in _frames.items() for _c, _W in _fr.items()}
    _v = max(np.abs(_Gm - np.diag(np.diag(_Gm))).max() for _Gm in _grams.values())      # one scale: the largest off-diagonal entry
    _levs = {(*_key, _c): np.mean([leverage(_row["E"][_c])[0] for _row in _rows], 0) for _key, _rows in _cells.items() for _c, _ in _SIDES}
    _alpha = np.asarray(next(iter(_cells.values()))[0]["alpha"], float)
    _norm = plt.matplotlib.colors.LogNorm(_alpha.min(), _alpha.max())

    _fig = plt.figure(figsize=(15.5, 6.6 * len(D_GRID)), layout="constrained")
    _sf_main, _sf_bar = _fig.subfigures(1, 2, wspace=0.0, width_ratios=[1, 0.05])           # the colourbars hug the last block
    for _row, (_d, _sf_row) in enumerate(zip(D_GRID[::-1], _sf_main.subfigures(len(D_GRID), 1, hspace=0.06))):   # a mega-row per d, largest first
        _nbar = 3 * int(_d)                                                                  # bars: the first 3d features, past every tail
        _sf_d, _sf_blocks = _sf_row.subfigures(1, 2, wspace=0.0, width_ratios=[0.045, 1])
        _sf_d.text(0.35, 0.47, f"$d$ = {_d}", rotation=90, va="center", ha="center", fontsize=26)
        for _col, (_p, _sf) in enumerate(zip(P_SWEEP, _sf_blocks.subfigures(1, 3, wspace=0.22))):   # gaps between p, none within
            if _row == 0:
                _sf.suptitle(f"$p$ = {_p}", fontsize=26)
            _gs = _sf.add_gridspec(3, 2, wspace=0.0, hspace=0.0, height_ratios=[0.8, 0.95, 0.85], left=0, right=1, top=0.94, bottom=0)
            _Wp = _frames[_d, _p]["optimum"]
            _lev = (_Wp ** 2).sum(1)
            _whole = len(band(_lev)) < 3                                                     # a band too small for its own frame: draw the head with it
            _idx = np.flatnonzero(_lev > CUT) if _whole else band(_lev)
            _B = basis3(_Wp[_idx], _Wp, _whole)
            for _j, (_c, _lbl) in enumerate(_SIDES):
                _ax3 = _sf.add_subplot(_gs[0, _j], projection="3d")
                frame_panel(_ax3, _frames[_d, _p][_c], _idx, _B, _lbl, _alpha, _norm, zoom=1.55, box=False)
                _axg = _sf.add_subplot(_gs[1, _j])
                _im = _axg.imshow(_grams[_d, _p, _c], cmap="vlag", vmin=-_v, vmax=_v)
                _axg.set_xticks([]); _axg.set_yticks([])
                _axb = _sf.add_subplot(_gs[2, _j])
                _axb.barh(np.arange(_nbar), _levs[_d, _p, _c][:_nbar], color="black", height=0.75)
                _axb.set_xlim(0, 1); _axb.set_ylim(_nbar - 0.5, -0.5); _axb.set_yticks([]); _axb.set_xticks([0, 0.5, 1])
                _axb.set_xlabel("leverage $d_i$")
                plot_utils.FIG.style_axes(_axb)
                if _col == 0 and _j == 0:
                    _ax3.text2D(-0.12, 0.5, f"{'all' if _whole else 'band'}, $d_i > {CUT}$", transform=_ax3.transAxes, rotation=90, va="center", ha="center", fontsize=12)
                    _axg.set_ylabel("Gram", fontsize=12)
                    _axb.set_ylabel("feature $i$", fontsize=12)
    _fig.suptitle(rf"$\beta$ = {_b}, $\rho$ = {_r}")
    _cb = _fig.colorbar(plt.cm.ScalarMappable(norm=_norm, cmap=ALPHA_CMAP), cax=_sf_bar.add_axes([0.15, 0.52, 0.25, 0.09]), label=r"$\alpha_i$")
    _cb.set_ticks([1, 2, 4, 8]); _cb.set_ticklabels(["1", "2", "4", "8"]); _cb.ax.minorticks_off()
    _fig.colorbar(_im, cax=_sf_bar.add_axes([0.15, 0.41, 0.25, 0.09]))
    _out = plot_utils.save_figure(_fig, plot_utils.figure_path(CONFIG, "fig4_sparsity"))
    mo.vstack([mo.hstack([SEL_F7["beta"], SEL_F7["rho"]], justify="start", gap=2), _out])
    return


if __name__ == "__main__":
    app.run()
