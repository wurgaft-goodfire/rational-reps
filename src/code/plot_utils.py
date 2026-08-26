#!/usr/bin/env python3
"""Publication plotting: `FigureGenerator` carries the house style (large Avenir fonts, twilight/vlag
colormaps, borderless manifold panels) and the geometry-analysis helpers render through it."""
import itertools
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap


class FigureGenerator:
    """Consistent styling + high-quality saving for figures."""

    def __init__(self):
        # palette + colormaps from demo_graphs
        self.colors = {'blue': '#4C72B0', 'orange': '#DD8452', 'green': '#55A868',
                       'purple': '#8172B3', 'gray': '#8C8C8C'}
        self.categorical = ['#4C72B0', '#DD8452', '#55A868', '#8172B3']   # per-series (curves/bars)
        self.cyclic_cmap = 'twilight'      # item index -> color (a ring reads as a ring)
        self.diverging_cmap = 'vlag'       # centered conditional heatmaps
        # twilight minus its pale grey ends: blue -> dark -> red, acyclic; the item colormap
        self.item_cmap = ListedColormap(
            plt.get_cmap('twilight')(np.linspace(0.1, 0.9, 256)), name='twilight_nogrey')
        # large fonts (Avenir), kept big for publication
        self.font_sizes = {'title': 18, 'axis_label': 15, 'tick': 13, 'legend': 13}
        self._setup_matplotlib_defaults()

    def _setup_matplotlib_defaults(self):
        """Global style: demo_graphs colormaps/interpolation + spines, with large Avenir fonts."""
        import seaborn  # noqa: F401  (registers the 'vlag' colormap)
        plt.rcParams.update({
            'figure.dpi': 120,
            'axes.spines.top': False, 'axes.spines.right': False,
            'image.interpolation': 'nearest',
            'mathtext.fontset': 'cm', 'pdf.fonttype': 42, 'ps.fonttype': 42,
            'axes.titlesize': self.font_sizes['title'], 'axes.labelsize': self.font_sizes['axis_label'],
            'xtick.labelsize': self.font_sizes['tick'], 'ytick.labelsize': self.font_sizes['tick'],
            'legend.fontsize': self.font_sizes['legend'],
        })
        # findSystemFonts is unsorted and returns several Avenir faces (Avenir / Next / Condensed /
        # Light) with distinct family names — pick Avenir Light deterministically, not [0].
        avenir = sorted(f for f in fm.findSystemFonts(fontext='ttc') if 'avenir' in f.lower())
        for f in avenir:
            fm.fontManager.addfont(f)
        pick = next((f for f in avenir if f.lower().endswith('avenirlight.ttf')), avenir[0] if avenir else None)
        if pick:
            plt.rcParams['font.family'] = fm.FontProperties(fname=pick).get_name()
            plt.rcParams['axes.labelweight'] = 'regular'
        else:
            plt.rcParams['font.family'] = 'sans-serif'
            print("Avenir font not found, using sans-serif instead.")

    def clean(self, ax):
        """Borderless manifold panel (demo _clean2d): no ticks, no spines."""
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    def style_axes(self, ax, spine_width=1.5, tick_length=5, tick_width=1.5):
        """Framed line/bar axes: drop top/right, thicken left/bottom (tick sizes via rcParams)."""
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_linewidth(spine_width)
        ax.tick_params(length=tick_length, width=tick_width)

    def save_figure(self, fig, filepath, format=None, dpi=300):
        """Save figure with publication-quality settings and selective rasterization. ``format=None``
        infers the format from the filepath extension (``.png`` for viewing, ``.pdf`` for papers).
        Returns the figure, so a save call at the end of a marimo cell renders it."""
        for ax in fig.get_axes():
            for collection in ax.collections:
                if collection.__class__.__name__ in ['QuadMesh', 'AxesImage']:
                    collection.set_rasterized(True)
                if hasattr(collection, '_offsets') and hasattr(collection._offsets, 'shape'):
                    if collection._offsets.shape[0] > 500:
                        collection.set_rasterized(True)
        fig.savefig(filepath, bbox_inches='tight', dpi=dpi, format=format)
        print(f"Saved figure to {filepath}")
        return fig


# Module-level singleton: importing plot_utils applies the house style.
FIG = FigureGenerator()


def save_figure(fig, filepath, **kwargs):
    """Save a figure with the house rasterization/quality settings; returns the figure."""
    return FIG.save_figure(fig, filepath, **kwargs)


def figure_path(config, name, ext="png"):
    """Path for a figure under the gitignored cache — ``{cache_dir}/{setting}/{override_dirname}/
    figures/{name}.{ext}``, shared by every model on that structure. Creates the parent dir;
    pass ``ext="pdf"`` for a paper-final vector figure."""
    path = Path(config.run_dir).parent / "figures" / f"{name}.{ext}"   # run_dir = .../{override}/{model}
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


######### Geometry-analysis plots (used by geometry_analysis.py) #########

def _pca(X, k=2):
    """Top-k PCA coordinates of a representation (rows = items)."""
    Xc = np.asarray(X) - np.asarray(X).mean(0)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    return U[:, :k] * S[:k]


# Structures whose geometry needs three dimensions to read (everything else is 2-D by default).
# Per-structure default presentation view; anything unlisted defaults to "2d".
_DEFAULT_VIEW = {"sphere": "3d", "hypercube": "3d", "complete": "3d", "product": "3d",
                 "helix": "3d", "tree": "ultrametric"}


def _view(spec):
    """The presentation view for a structure's geometry: ``spec['view']`` if set, else the
    per-structure default. Values: ``"2d"`` | ``"3d"`` (PCA scatters) | ``"torus"`` (3-D donut) |
    ``"ultrametric"`` | ``"simplex"`` | ``"pca"`` (tree; a 2-D scatter)."""
    if spec is None:
        return "2d"
    if spec.get("view"):
        return spec["view"]
    if spec.get("structure") == "product":
        facs = spec.get("factors", [])
        if len(facs) == 2 and all(f["structure"] == "ring" for f in facs):
            return "torus"
        return "3d"
    return _DEFAULT_VIEW.get(spec.get("structure"), "2d")


def _torus_donut(r):
    """Map a ring x ring torus onto the classic 3-D donut by reading the two ring angles straight from
    the geometry (as in demo_graphs): the two circles occupy the top-2 and next-2 principal axes (they
    separate when the rings differ in size), so theta/phi are those coordinate pairs' angles."""
    E4 = _pca(r, 4)
    th, ph = np.arctan2(E4[:, 1], E4[:, 0]), np.arctan2(E4[:, 3], E4[:, 2])
    Rmaj, rmin = 2.0, 0.9
    return np.c_[(Rmaj + rmin * np.cos(ph)) * np.cos(th),
                 (Rmaj + rmin * np.cos(ph)) * np.sin(th), rmin * np.sin(ph)]


def _shape_positions(name, p):
    """Canonical node positions (2-D or 3-D) for a single graph shape."""
    if name in ("ring", "complete"):
        th = 2 * np.pi * np.arange(p["n"]) / p["n"]
        return np.c_[np.cos(th), np.sin(th)]
    if name == "chain":
        return np.c_[np.arange(p["n"], dtype=float), np.zeros(p["n"])]
    if name == "grid":
        nr, nc = p["n"], p.get("m", p["n"])
        return np.array([[j, -i] for i in range(nr) for j in range(nc)], float)
    if name == "hypercube":
        n = p["n"]; B = np.array([[(v >> b) & 1 for b in range(n)] for v in range(2 ** n)], float)
        P = B[:, :3]
        return P if P.shape[1] == 3 else np.c_[P, np.zeros((len(P), 3 - P.shape[1]))]
    if name == "sphere":
        idx = np.arange(p["n"]); gold = np.pi * (1 + 5 ** 0.5)
        zc = 1 - 2 * (idx + 0.5) / p["n"]; rc = np.sqrt(1 - zc ** 2); ph = gold * idx
        return np.stack([rc * np.cos(ph), rc * np.sin(ph), zc], 1)
    if name == "communities":
        n_comm, sz = p.get("k", 3), p.get("n", 25)
        out = []
        for c in range(n_comm):
            hub = 2.2 * np.array([np.cos(2 * np.pi * c / n_comm), np.sin(2 * np.pi * c / n_comm)])
            a = 2 * np.pi * np.arange(sz) / sz
            out.append(hub + 0.7 * np.c_[np.cos(a), np.sin(a)])
        return np.vstack(out)
    raise ValueError(f"No layout for shape {name!r}")


# tree / ultrametric vis are drawn tall: vertical extent ~ this multiple of the horizontal
TREE_ASPECT = 1.6


def _elongate(pos):
    """Stretch a 2-D layout vertically so it reads as a tall tree (height ~ TREE_ASPECT * width)."""
    pos = np.asarray(pos, float); xr, yr = np.ptp(pos[:, 0]), np.ptp(pos[:, 1])
    if yr > 0 and xr > 0:
        pos = pos.copy(); pos[:, 1] *= TREE_ASPECT * xr / yr
    return pos


def _tree_graph(parents):
    """(adjacency, hierarchical positions) for a tree given a parent-index list (root = -1):
    leaves along the bottom, each internal node above the mean of its children, stretched tall."""
    from .data import tree_adjacency
    N = len(parents)
    children = {i: [] for i in range(N)}; root = 0
    for c, p in enumerate(parents):
        if p is None or p < 0:
            root = c
        else:
            children[p].append(c)
    depth = [0] * N
    def set_depth(u, d):
        depth[u] = d
        for ch in children[u]:
            set_depth(ch, d + 1)
    set_depth(root, 0)
    xpos, nxt = {}, [0]
    def set_x(u):
        if not children[u]:
            xpos[u] = float(nxt[0]); nxt[0] += 1
        else:
            for ch in children[u]:
                set_x(ch)
            xpos[u] = float(np.mean([xpos[ch] for ch in children[u]]))
    set_x(root)
    pos = np.array([[xpos[i], -depth[i]] for i in range(N)], float)
    return tree_adjacency(parents), _elongate(pos)


def _structure_graph(spec):
    """(adjacency, node positions) for drawing any structure's latent graph (2-D or 3-D)."""
    from .data import structure_adjacency, tree_parents
    name = spec["structure"]; p = {k: v for k, v in spec.items() if k != "structure"}
    if name == "tree":
        return _tree_graph(tree_parents(p.get("n", 3), p.get("k", 2), p.get("parents")))
    if name in ("product", "stack"):
        factors = p["factors"]
        if name == "product" and len(factors) == 2 and all(f["structure"] == "ring" for f in factors):
            # a torus is 4-D; draw it as the classic 3-D donut (the two ring angles on the surface),
            # else a 3-D truncation collapses the minor circle and it reads as a cylinder.
            nA, nB = factors[0]["n"], factors[1]["n"]
            i = np.repeat(np.arange(nA), nB); j = np.tile(np.arange(nB), nA)   # node = i*nB + j
            th, ph = 2 * np.pi * i / nA, 2 * np.pi * j / nB
            Rmaj, rmin = 2.0, 0.9
            pos = np.c_[(Rmaj + rmin * np.cos(ph)) * np.cos(th),
                        (Rmaj + rmin * np.cos(ph)) * np.sin(th), rmin * np.sin(ph)]
            return structure_adjacency(spec), pos
        subs = [_structure_graph(f) for f in factors]
        A = structure_adjacency(spec)
        if name == "stack":                                   # lay factor blocks side by side
            blocks, xoff = [], 0.0
            for _, pi in subs:
                q = np.asarray(pi, float)[:, :2].copy()
                q[:, 0] += xoff - q[:, 0].min(); xoff = q[:, 0].max() + 2.0
                blocks.append(q)
            return A, np.vstack(blocks)
        # product: concat each factor's *intrinsic* coords (drop padding/constant columns, so a
        # chain contributes 1-D and a ring 2-D) — otherwise a leading chain's padding column would
        # push a real axis of a later factor past the 3-D truncation (a cylinder would look flat).
        def _intrinsic(P):
            P = np.asarray(P, float); keep = P.std(0) > 1e-9
            return P[:, keep] if keep.any() else P[:, :1]
        grids = [_intrinsic(pi) for _, pi in subs]
        pos = np.array([np.concatenate(combo) for combo in itertools.product(*grids)])
        pos = pos[:, :3] if pos.shape[1] >= 3 else np.c_[pos, np.zeros((len(pos), 3 - pos.shape[1]))]
        return A, pos
    return structure_adjacency(spec), _shape_positions(name, p)


def _hypercube_labels(spec):
    """Binary vertex labels for a hypercube (``'101'`` etc.), one per node in adjacency order."""
    n = spec.get("n", 3)
    return [format(v, f"0{n}b") for v in range(2 ** n)]


def draw_structure(ax, A, pos, title="structure", item_ids=None, cmap=None, labels=None):
    """Draw a latent graph: light edges + nodes at `pos` (2-D or 3-D). Nodes in `item_ids` are
    colored by item index (via `cmap`, default the house cyclic map); others (e.g. inner tree nodes
    in leaves-only mode) are gray. `labels`, if given, annotates each node (e.g. hypercube bit strings)."""
    cmap = cmap or FIG.cyclic_cmap
    A, pos = np.asarray(A), np.asarray(pos); N = len(pos)
    item_ids = list(range(N)) if item_ids is None else list(item_ids)
    cidx = np.full(N, -1)
    for k, i in enumerate(item_ids):
        cidx[i] = k
    it = cidx >= 0; vmax = max(len(item_ids) - 1, 1)
    ii, jj = np.where(np.triu(A, 1) > 0)
    if pos.shape[1] == 2:
        ax.add_collection(LineCollection([[pos[a], pos[b]] for a, b in zip(ii, jj)],
                                         colors="0.85", linewidths=0.5, zorder=0))
        if (~it).any():
            ax.scatter(pos[~it, 0], pos[~it, 1], c="0.8", s=28, zorder=1)
        ax.scatter(pos[it, 0], pos[it, 1], c=cidx[it], cmap=cmap,
                   vmin=0, vmax=vmax, s=40, zorder=2)
        if labels is not None:
            for (x, y), lab in zip(pos[:, :2], labels):
                ax.text(x, y, lab, fontsize=6, ha="center", va="bottom", zorder=3)
        d0, d1 = np.ptp(pos[:, 0]), np.ptp(pos[:, 1])
        if min(d0, d1) < 0.1 * max(d0, d1):               # ~1-D (chain): fill, don't squash
            pad = 0.25 * max(d0, d1)
            ax.set_ylim(pos[:, 1].min() - pad, pos[:, 1].max() + pad); ax.margins(x=0.05)
        else:
            ax.set_aspect("equal"); ax.margins(0.12)      # pad so edge nodes aren't clipped
        FIG.clean(ax); ax.set_anchor("N")
    else:
        for a, b in zip(ii, jj):
            ax.plot(*zip(pos[a], pos[b]), color="0.85", lw=0.25, zorder=0)
        if (~it).any():
            ax.scatter(pos[~it, 0], pos[~it, 1], pos[~it, 2], c="0.8", s=16)
        ax.scatter(pos[it, 0], pos[it, 1], pos[it, 2], c=cidx[it], cmap=cmap,
                   vmin=0, vmax=vmax, s=25, depthshade=False, alpha=0.7)
        if labels is not None:
            for (x, y, z), lab in zip(pos, labels):
                ax.text(x, y, z, lab, fontsize=6, ha="center", va="bottom")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):    # white background, points never clipped
            axis.pane.set_facecolor((1, 1, 1, 0))
        ax.margins(0.1)
    ax.set_title(title)


def scatter_geometry(ax, XY, title, cmap=None):
    """Embedding scatter colored by item index (borderless manifold panel); 2-D or 3-D by XY's width."""
    XY = np.asarray(XY); c = np.arange(len(XY)); cmap = cmap or FIG.cyclic_cmap
    if XY.shape[1] >= 3:
        # depthshade off: it modulates alpha by depth, so identical geometries in different
        # orientations would render with different colors across panels. The fixed soft alpha
        # keeps the light look depthshade used to give the predicted panel.
        ax.scatter(XY[:, 0], XY[:, 1], XY[:, 2], c=c, cmap=cmap, s=28, depthshade=False, alpha=0.7)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.set_title(title)
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis): pane.pane.set_facecolor((1, 1, 1, 0))
    else:
        ax.scatter(XY[:, 0], XY[:, 1], c=c, cmap=cmap, s=45)
        ax.set_title(title); ax.set_aspect("equal"); ax.margins(0.15); FIG.clean(ax); ax.set_anchor("N")


def draw_embedding(fig, cell, r, spec, title, cmap=None):
    """One embedding panel per the structure's view (see `_view`): a 3-D donut for a torus, else a
    2-D or 3-D PCA scatter. Tree ultrametric/simplex views route separately in the callers."""
    view = _view(spec)
    if view == "torus":
        ax = fig.add_subplot(cell, projection="3d")
        scatter_geometry(ax, _torus_donut(r), title, cmap)
        return
    ndim = 3 if view == "3d" else 2
    ax = fig.add_subplot(cell, projection="3d" if ndim == 3 else None)
    name = spec.get("structure") if spec is not None else None
    if name in ("hypercube", "complete", "grid"):
        # draw the graph edges over the learned positions so the cube / simplex / lattice is
        # legible as a connected polytope, not just a point cloud (as demo_graphs did for the cube)
        from .data import structure_adjacency
        draw_structure(ax, structure_adjacency(spec), _pca(r, ndim), title, cmap=cmap)
        return
    scatter_geometry(ax, _pca(r, ndim), title, cmap)


def plot_factorization(L, k=None, item_label=r"inputs  $z$", outcome_label=r"outputs  $x$",
                       caption=None, highlight_row=None, highlight_col=None, factors=None,
                       figsize=(13, 4.2), embedding_width=0.05, decoder_height=0.09,
                       conditional_aspect="equal"):
    """Decomposition schematic: centered log-conditional = embedding × decoder, three aligned
    heatmaps with per-panel symmetric color scales. ``k=None`` shows the full-rank factorization
    (exact, drawn with ``=``); an int truncates (drawn with ``≈``). ``factors`` supplies an
    alternative (E, D), e.g. re-expressed in a canonical gauge. Layout: ``embedding_width`` is the
    width of one code column (as a fraction of the conditional's width), ``decoder_height`` the
    height of one decoder row (as a fraction of the panel row); ``conditional_aspect="auto"``
    stretches the conditional to fill its cell like the embedding, instead of square cells. The
    decoder is centered on the operator row. ``highlight_row``/``highlight_col`` box one decoder
    row/column. Returns ``(fig, {"L","E","D"})``."""
    from matplotlib.patches import Rectangle
    from .geometry import factor_logcond
    Lc, E, D, S = factor_logcond(L)
    if factors is not None:                   # e.g. re-expressed in a canonical gauge
        E, D = factors
    k = min(Lc.shape) if k is None else k
    exact = k >= min(Lc.shape) or S[k] < 1e-9 * S[0]      # truncation past the rank is still exact
    Ek, Dk = E[:, :k], D[:k]
    n_i, n_o = Lc.shape

    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(bottom=0.16)
    gap = 0.14 * n_o
    gs = fig.add_gridspec(1, 5, width_ratios=[n_o, gap, embedding_width * n_o * k, gap, n_o],
                          wspace=0.05)

    def _panel(ax, M, aspect="auto", pct=100):
        # the factors clip at their max (exact SVD, no outliers): percentile clipping would
        # amplify their near-zero tail dims into noise
        lim = np.percentile(np.abs(M), pct)
        ax.imshow(M, cmap=FIG.diverging_cmap, vmin=-lim, vmax=lim, aspect=aspect)
        FIG.clean(ax)

    axL = fig.add_subplot(gs[0, 0])
    _panel(axL, Lc, aspect=conditional_aspect, pct=98)
    axL.set_title("log-conditional\n" + r"$L[z, x] = \log \tilde{p}(x \mid z)$", fontsize=15)
    axL.set_xlabel(outcome_label, fontsize=13); axL.set_ylabel(item_label, fontsize=13)

    axE = fig.add_subplot(gs[0, 2])
    _panel(axE, Ek)
    axE.set_title("embedding", fontsize=15)
    axE.set_xlabel("k dims", fontsize=13)

    # decoder: k rows drawn at the same cell height as the other panels, vertically centered
    axD = fig.add_subplot(gs[0, 4])
    _panel(axD, Dk)
    axD.set_title("decoder", fontsize=15)
    axD.set_xlabel(outcome_label, fontsize=13)
    pos = axD.get_position()
    shrink = min(1.0, decoder_height * k)                         # chunky rows, like the manuscript
    axD.set_position([pos.x0, pos.y0 + pos.height * (1 - shrink) / 2,
                      pos.width, pos.height * shrink])
    # E is (n, k) and D is (k, n): mirror their cell geometry — E's column width (inches)
    # equals D's row thickness
    figw, figh = fig.get_size_inches()
    cell_in = axD.get_position().height * figh / k
    posE = axE.get_position()
    new_w = k * cell_in / figw
    axE.set_position([posE.x0 + (posE.width - new_w) / 2, posE.y0, new_w, posE.height])
    # the operator symbols, horizontally between the *drawn* panel boxes (L letterboxes to a
    # square, so gridspec-cell centering would drift) and vertically on the panel row's center
    fig.canvas.draw()
    pL, pE, pD = axL.get_position(), axE.get_position(), axD.get_position()
    yc = (pE.y0 + pE.y1) / 2                              # the full-height panel: the row's middle
    fig.text((pL.x1 + pE.x0) / 2, yc, r"$=$" if exact else r"$\approx$",
             fontsize=26, ha="center", va="center", color="0.15")
    fig.text((pE.x1 + pD.x0) / 2, yc, r"$\times$", fontsize=26, ha="center", va="center", color="0.15")
    if highlight_row is not None:
        axD.add_patch(Rectangle((-0.5, highlight_row - 0.5), n_o, 1, fill=False,
                                ec="0.45", lw=2.2, zorder=5, clip_on=False))
    if highlight_col is not None:
        axD.add_patch(Rectangle((highlight_col - 0.5, -0.5), 1, k, fill=False,
                                ec="black", lw=2.2, zorder=5, clip_on=False))
    if caption:
        fig.text(0.5, 0.015, caption, ha="center", fontsize=12.5, color="0.45")
    return fig, {"L": axL, "E": axE, "D": axD}


def conditional_heatmap(ax, L, title=r"$\log \tilde{p}(x \mid z)$", xlabel=r"output  $x$",
                        ylabel=r"input  $z$", aspect="equal", cmap=None):
    """Diverging heatmap of the globally mean-centered log-conditional, clipped at the ±98th
    percentile so a few extreme entries don't wash out the range. Pass ``aspect="auto"`` for
    far-from-square conditionals (e.g. many items, few outcomes), which an equal aspect would
    collapse to a sliver."""
    Ld = np.asarray(L) - np.asarray(L).mean()
    lim = np.percentile(np.abs(Ld), 98)
    ax.imshow(Ld, cmap=cmap or FIG.diverging_cmap, vmin=-lim, vmax=lim, aspect=aspect)
    FIG.clean(ax); ax.set_anchor("N")                     # drop ticks + spines, pin to top
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)     # keep just the axis labels


def plot_predicted_geometry(spec, L, E, cmap=None):
    """Triptych: latent structure -> conditional -> the geometry it encodes. ``spec['view']`` picks
    the third panel (tree: ``"ultrametric"`` | ``"pca"`` | ``"simplex"``; else ``"2d"``/``"3d"`` PCA);
    ``spec['leaves_only']`` makes only leaves items. ``cmap`` overrides the house cyclic colormap."""
    A, pos = _structure_graph(spec)
    is3d = pos.shape[1] == 3
    is_tree = spec["structure"] == "tree"
    view = _view(spec)
    parents = items = None
    H = 3.9
    if is_tree:
        parents, items = _tree_items(spec, len(E))
        if view == "simplex":
            n = sum(len(v) >= 2 for v in _tree_children(parents).values())
            H = max(3.9, 2.2 * int(np.ceil(n / np.ceil(np.sqrt(n)))))
    fig = plt.figure(figsize=(13, H))
    gs = fig.add_gridspec(1, 3, wspace=0.45)
    item_ids = items if is_tree and len(items) != len(A) else None
    labels = _hypercube_labels(spec) if spec["structure"] == "hypercube" else None
    draw_structure(fig.add_subplot(gs[0, 0], projection="3d" if is3d else None), A, pos,
                   item_ids=item_ids, cmap=cmap, labels=labels)
    axB = fig.add_subplot(gs[0, 1]); conditional_heatmap(axB, L)
    if is_tree and view == "ultrametric":
        rows, cmax = _tree_leaf_rows(parents, items)
        _draw_ultrametric(fig.add_subplot(gs[0, 2]), E[rows], np.array(rows), cmax, cmap)
    elif is_tree and view == "simplex":
        _draw_simplex_of_simplices(fig, gs[0, 2], parents, items, E, cmap)
    else:
        draw_embedding(fig, gs[0, 2], E, spec, "embedding", cmap)
    # arrows in the column gaps, aligned to the conditional panel's vertical center (never overlap)
    fig.canvas.draw()
    yc = np.mean([axB.get_position().y0, axB.get_position().y1])
    for lc, rc in [(gs[0, 0], gs[0, 1]), (gs[0, 1], gs[0, 2])]:
        bl, br = lc.get_position(fig), rc.get_position(fig)
        fig.text((bl.x1 + br.x0) / 2, yc, r"$\rightarrow$",
                 fontsize=22, ha="center", va="center", color="0.4")
    return fig


def plot_uncertainty(E):
    """Uncertainty over a ring: belief codes ``belief @ E`` on the circle, with two trajectories —
    mixing two beliefs (a chord) and broadening one (rim -> center) — plus the belief distributions.
    Assumes ``E`` is a ring code (items ordered)."""
    n = len(E); th = 2 * np.pi * np.arange(n) / n; circ = np.asarray(E)[:, :2]
    vm = lambda mu, k: (lambda b: b / b.sum())(np.exp(k * np.cos(th - mu)))
    bA, bB = vm(th[int(0.15 * n)], 12), vm(th[int(0.6 * n)], 12)
    ts = np.linspace(0, 1, 11)
    interp = np.array([(t * bA + (1 - t) * bB) @ circ for t in ts])
    ks = [12, 6, 3, 1.5, 0.7, 0.3, 0.1]
    broaden = np.array([vm(th[int(0.35 * n)], k) @ circ for k in ks])

    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    ax[0].scatter(circ[:, 0], circ[:, 1], c="0.85", s=15)
    ax[0].scatter([0], [0], c="0.6", marker="+", s=60)
    ax[0].plot(interp[:, 0], interp[:, 1], "-", color="0.6", lw=0.8, zorder=1)
    ax[0].scatter(interp[:, 0], interp[:, 1], c=ts, cmap="coolwarm", s=26, zorder=3)
    ax[0].plot(broaden[:, 0], broaden[:, 1], "-", color="0.6", lw=0.8, zorder=1)
    ax[0].scatter(broaden[:, 0], broaden[:, 1], c=range(len(ks)), cmap="viridis_r",
                  s=26, marker="s", zorder=3)
    ax[0].set_title("codes on the circle", pad=24)
    ax[0].text(0.5, 1.03, r"$\bullet$ mix A$\leftrightarrow$B     $\blacksquare$ broaden $\to$ center",
               transform=ax[0].transAxes, ha="center", fontsize=12, color="0.35")
    ax[0].set_anchor("C")
    ax[0].set_aspect("equal"); FIG.clean(ax[0])

    for t in [0, 0.25, 0.5, 0.75, 1.0]:
        ax[1].plot(np.arange(n), t * bA + (1 - t) * bB, lw=1.4, color=plt.cm.coolwarm(t))
    ax[1].set(title=r"mix A$\leftrightarrow$B", xlabel="item", yticks=[])
    FIG.style_axes(ax[1])

    for i, k in enumerate(ks):
        ax[2].plot(np.arange(n), vm(th[int(0.35 * n)], k), lw=1.4, color=plt.cm.viridis_r(i / len(ks)))
    ax[2].set(title="broaden one belief", xlabel="item", yticks=[]); FIG.style_axes(ax[2])
    fig.tight_layout()
    return fig


def _circle2d(X):
    """Top-2 PCA of an embedding block, normalized to unit mean radius (for drawing a circle)."""
    Xc = np.asarray(X) - np.asarray(X).mean(0)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    P = U[:, :2] * S[:2]
    return P / (np.linalg.norm(P, axis=1).mean() + 1e-9)


def plot_stacking(n=12, q=10):
    """Combining structures shapes the code (demo_graphs §2): disjoint supports **union** into
    separate circles; shared independent factors form a **product** (torus = two factor circles);
    aligned relations **share** one circle."""
    from .data import stacked_conditional
    from .geometry import factor_logcond
    ring = lambda k: {"structure": "ring", "n": k}
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 4.2))

    # disjoint supports -> union (two separate circles)
    Eu = factor_logcond(stacked_conditional([ring(n), ring(n)], "union"))[1]
    for sl, off in [(slice(0, n), -1.6), (slice(n, 2 * n), 1.6)]:
        P = _circle2d(Eu[sl])
        ax[0].scatter(P[:, 0] + off, P[:, 1], c=np.arange(n), cmap=FIG.cyclic_cmap, s=35)
    ax[0].set_title("disjoint supports: union (two circles)")
    ax[0].set_aspect("equal"); FIG.clean(ax[0])

    # shared support, independent factors -> product (a torus, shown as its two factor circles)
    Ep = factor_logcond(stacked_conditional([ring(n), ring(q)], "product"))[1]
    idx = np.arange(n * q)
    ci = _circle2d(np.stack([Ep[idx // q == i].mean(0) for i in range(n)]))
    cj = _circle2d(np.stack([Ep[idx % q == j].mean(0) for j in range(q)]))
    ax[1].scatter(ci[:, 0] - 1.6, ci[:, 1], c=np.arange(n), cmap=FIG.cyclic_cmap, s=35)
    ax[1].scatter(cj[:, 0] + 1.6, cj[:, 1], c=np.arange(q), cmap=FIG.cyclic_cmap, s=35)
    ax[1].set_title("shared, independent: product (torus)")
    ax[1].set_aspect("equal"); FIG.clean(ax[1])

    # shared support, aligned relations -> share (one circle)
    Es = factor_logcond(stacked_conditional([ring(n), ring(n)], "share"))[1]
    P = _circle2d(Es)
    ax[2].scatter(P[:, 0], P[:, 1], c=np.arange(n), cmap=FIG.cyclic_cmap, s=35)
    ax[2].set_title("shared, aligned: share (one circle)")
    ax[2].set_aspect("equal"); FIG.clean(ax[2])

    fig.tight_layout()
    return fig


def rdm_heatmap(ax, reps, title="RDM", cmap=None):
    """Representational dissimilarity matrix (pairwise Euclidean distances between item reps),
    normalized by its max so panels share one [0, 1] scale regardless of representation norm.
    Default is the reversed house diverging map: similar pairs read red, like the conditional."""
    r = np.asarray(reps)
    D = np.sqrt(((r[:, None] - r[None]) ** 2).sum(-1))
    ax.imshow(D / D.max(), cmap=cmap or f"{FIG.diverging_cmap}_r", vmin=0, vmax=1, aspect="equal")
    FIG.clean(ax); ax.set_anchor("N"); ax.set_title(title)


def gram_heatmap(ax, reps, title="Gram", cmap=None):
    """Gram matrix of the item-centered code (items x items), with symmetric color limits at the
    max |entry| so the sign structure reads on the diverging map. The same quantity for predicted
    (``E E^T`` with ``E = U sqrt(S)``) and learned codes, so the panels compare directly."""
    r = np.asarray(reps, float)
    r = r - r.mean(0)
    G = r @ r.T
    lim = np.abs(G).max()
    ax.imshow(G, cmap=cmap or FIG.diverging_cmap, vmin=-lim, vmax=lim, aspect="equal")
    FIG.clean(ax); ax.set_anchor("N"); ax.set_title(title)


def plot_embedding_panels(E_pred, embeddings, spec=None, cmap=None):
    """Two rows: predicted code beside each model's pre-decode representation (top, laid out per the
    structure's ``view``; see `_view`), and each code's Gram matrix (bottom) — ``E E^T`` for the
    predicted column, so all matrix panels show the same quantity. ``embeddings``: {name: reps}."""
    names = ["predicted"] + list(embeddings)
    reps = [E_pred] + list(embeddings.values())
    is_tree = spec is not None and spec["structure"] == "tree"
    view = _view(spec)
    if is_tree:
        parents, items = _tree_items(spec, len(E_pred))
        leaf_rows, cmax = _tree_leaf_rows(parents, items)
    fig = plt.figure(figsize=(2.7 * len(names), 5.9))
    gs = fig.add_gridspec(2, len(names), hspace=0.25, wspace=0.3)
    headers = []
    for j, (name, r) in enumerate(zip(names, reps)):
        if is_tree and view == "ultrametric":
            ax = fig.add_subplot(gs[0, j])
            _draw_ultrametric(ax, r[leaf_rows], np.array(leaf_rows), cmax, cmap); ax.set_title(name)
        elif is_tree and view == "simplex":
            _draw_simplex_of_simplices(fig, gs[0, j], parents, items, r, cmap, titles=False)
            headers.append((j, name))
        else:
            draw_embedding(fig, gs[0, j], r, spec, name, cmap)
        axb = fig.add_subplot(gs[1, j])
        gram_heatmap(axb, r, title=r"$E^{*}E^{*\top}$" if j == 0 else r"$EE^{\top}$")
    if headers:                                          # column names for the simplex sub-grids
        fig.canvas.draw()
        for j, name in headers:
            bb = gs[0, j].get_position(fig)
            fig.text((bb.x0 + bb.x1) / 2, bb.y1 + 0.01, name, ha="center", va="bottom",
                     fontsize=FIG.font_sizes["axis_label"])
    return fig


def plot_decoder_panels(D_pred, readouts, spec=None, cmap=None):
    """The decoder mirror of `plot_embedding_panels`: predicted decoder ``D`` beside each model's
    readout weights on the structure's ``view`` (top), each decoder's Gram (bottom).
    ``readouts``: {name: readout_weights (outcomes x features)}."""
    names = ["predicted"] + list(readouts)
    mats = [D_pred] + list(readouts.values())
    fig = plt.figure(figsize=(2.7 * len(names), 5.9))
    gs = fig.add_gridspec(2, len(names), hspace=0.25, wspace=0.3)
    for j, (name, r) in enumerate(zip(names, mats)):
        draw_embedding(fig, gs[0, j], r, spec, name, cmap)
        gram_heatmap(fig.add_subplot(gs[1, j]), r, title=r"$D^{*}D^{*\top}$" if j == 0 else r"$DD^{\top}$")
    return fig


def _tree_children(parents):
    """{node -> list of children} from a parent-index list (root parent = -1)."""
    ch = {}
    for c, par in enumerate(parents):
        if par is not None and par >= 0:
            ch.setdefault(par, []).append(c)
    return ch


def _tree_items(spec, n_items):
    """(parents, item node-ids), inferred from the code's row count ``n_items``: the leaves if it
    matches the leaf count, else every node (Park et al.). Reads what the conditional actually
    holds rather than assuming ``tree_conditional``'s default."""
    from .data import tree_parents
    p = {k: v for k, v in spec.items() if k != "structure"}
    parents = tree_parents(p.get("n", 3), p.get("k", 2), p.get("parents"))
    ch = _tree_children(parents)
    leaves = [u for u in range(len(parents)) if u not in ch]
    return parents, (leaves if n_items == len(leaves) else list(range(len(parents))))


def _tree_leaf_rows(parents, items):
    """Row indices of ``items`` that are leaves, and the color range. The dendrogram clusters leaves
    only — inner-node items are subtree centroids and would tangle it."""
    ch = _tree_children(parents)
    leafset = {u for u in range(len(parents)) if u not in ch}
    rows = [i for i, node in enumerate(items) if node in leafset]
    return rows, len(items)


def _node_reps(parents, items, E):
    """Per-node (rep, color): its own code + item index if a node is an item, else the centroid and
    mean color of its subtree's leaves (so leaves-only trees still resolve each split)."""
    E = np.asarray(E, float); pos = {nd: i for i, nd in enumerate(items)}
    ch = _tree_children(parents); reps, col = {}, {}
    def visit(u):
        if u in ch:
            for c in ch[u]:
                if c not in reps:
                    visit(c)
            reps[u] = E[pos[u]] if u in pos else np.mean([reps[c] for c in ch[u]], 0)
            col[u] = pos[u] if u in pos else float(np.mean([col[c] for c in ch[u]]))
        else:
            reps[u], col[u] = E[pos[u]], pos[u]
    for u in range(len(parents)):
        if u not in reps:
            visit(u)
    return (np.array([reps[u] for u in range(len(parents))]),
            np.array([col[u] for u in range(len(parents))], float))


def _draw_ultrametric(ax, leaf_E, colors, cmax, cmap=None):
    """Ultrametric recovered from the leaf codes by average-linkage clustering, drawn in the tree
    style (straight light edges, index-colored leaf circles, gray merge nodes). scipy's linkage is
    strictly binary, so a k-way split comes back as stacked binary merges at nearly equal heights;
    merges within 10% of the height range of their parent are collapsed into one multiway node.
    ``colors`` is each leaf's item index, ``cmax`` the color range. Default tree view."""
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import pdist
    cmap = cmap or FIG.cyclic_cmap
    n = len(leaf_E)
    Z = linkage(pdist(np.asarray(leaf_E, float)), method="average")   # scipy clusters
    heights = np.concatenate([np.zeros(n), Z[:, 2]])
    children = {n + i: [int(a), int(b)] for i, (a, b, *_) in enumerate(Z)}
    tol = 0.2 * heights.max()

    def _kept_children(u):                       # absorb near-equal-height binary sub-merges
        out = []
        for c in children[u]:
            out += _kept_children(c) if c >= n and heights[u] - heights[c] < tol else [c]
        return out

    # layout (straight edges, leaves pinned to item order for cross-panel alignment)
    pos, edges = {}, []
    def _place(u):
        if u < n:
            pos[u] = np.array([float(u), 0.0])
            return
        kept = _kept_children(u)
        for c in kept:
            _place(c)
        edges.extend((u, c) for c in kept)
        pos[u] = np.array([np.mean([pos[c][0] for c in kept]), heights[u]])
    _place(2 * n - 2)
    nodes = sorted(pos)
    P = dict(zip(nodes, _elongate(np.array([pos[u] for u in nodes]))))
    ax.add_collection(LineCollection([[P[a], P[b]] for a, b in edges],
                                     colors="0.85", linewidths=0.5, zorder=0))
    merge = [u for u in nodes if u >= n]
    ax.scatter([P[u][0] for u in merge], [P[u][1] for u in merge], c="0.8", s=24, zorder=1)
    ax.scatter([P[u][0] for u in range(n)], [P[u][1] for u in range(n)], c=colors, cmap=cmap,
               vmin=0, vmax=cmax, s=70, zorder=2, clip_on=False)                 # leaves
    ax.set_title("ultrametric"); ax.set_aspect("equal"); ax.margins(0.12)
    FIG.clean(ax); ax.set_anchor("N")


def _draw_simplex_of_simplices(fig, subspec, parents, items, E, cmap=None, titles=True):
    """Simplex-of-simplices grid: each split's child points, centered and projected to the split's
    intrinsic dimension (segment / triangle / tetrahedron), laid out in a compact grid with ⊕.
    ``titles=False`` drops per-node labels (for dense multi-column panels)."""
    cmap = cmap or FIG.cyclic_cmap
    R, C = _node_reps(parents, items, E); R = R - R.mean(0)
    ch = _tree_children(parents)
    splits = [u for u in sorted(ch) if len(ch[u]) >= 2]
    names = {2: "segment", 3: "triangle", 4: "tetrahedron"}
    ncols = int(np.ceil(np.sqrt(len(splits)))); nrows = int(np.ceil(len(splits) / ncols))
    inner = subspec.subgridspec(nrows, ncols, hspace=0.55, wspace=0.35)
    axes = {}
    for idx, u in enumerate(splits):
        r0, c0 = divmod(idx, ncols)
        kids = np.array(ch[u]); m = len(kids); d = min(m - 1, 3)
        Pc = R[kids] - R[kids].mean(0)
        U, S, _ = np.linalg.svd(Pc, full_matrices=False); P = U[:, :d] * S[:d]
        if d == 1:
            P = np.c_[P[:, 0], np.zeros(m)]
        ax = fig.add_subplot(inner[r0, c0], projection="3d" if d == 3 else None)
        cargs = dict(c=C[kids], cmap=cmap, vmin=0, vmax=len(items) - 1, s=55, zorder=2)
        if d == 3:
            for a, b in zip(*np.triu_indices(m, 1)):
                ax.plot(*zip(P[a], P[b]), color="0.7", lw=1.0, zorder=1)
            ax.scatter(P[:, 0], P[:, 1], P[:, 2], **cargs)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        else:
            ax.add_collection(LineCollection([[P[a], P[b]] for a in range(m) for b in range(a + 1, m)],
                                             colors="0.7", linewidths=1.0, zorder=1))
            ax.scatter(P[:, 0], P[:, 1], **cargs); ax.set_aspect("equal"); ax.margins(0.2); FIG.clean(ax)
        if titles:
            ax.set_title(f"node {u}: {names.get(m, f'{m}-simplex')}", fontsize=FIG.font_sizes["tick"])
        axes[(r0, c0)] = ax
    fig.canvas.draw()
    for (r0, c0), ax in axes.items():
        if (r0, c0 + 1) in axes:
            p0, p1 = ax.get_position(), axes[(r0, c0 + 1)].get_position()
            fig.text((p0.x1 + p1.x0) / 2, (p0.y0 + p0.y1) / 2, r"$\oplus$",
                     fontsize=18, ha="center", va="center", color="0.4")


# vivid, distinguishable series palette (Dark2) for curves/bars
SERIES = ["#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E", "#E6AB02"]


def plot_rsa_curves(histories):
    """RSA-to-predicted vs training step, one line per model. `histories`: {name: history_df}."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (name, h) in enumerate(histories.items()):
        ax.plot(h["step"], h["rsa_vs_predicted"], marker="o", ms=5,
                color=FIG.categorical[i % len(FIG.categorical)], label=name)
    ax.set(xlabel="training step", ylabel="RSA to predicted geometry", ylim=(0, 1))
    ax.legend(frameon=False); FIG.style_axes(ax)
    return fig


def plot_convergence_bars(scores, ylabel="similarity to predicted geometry",
                          title="representational convergence across architectures"):
    """CKA/RSA to a predicted target per model. `scores`: {name: {"cka": .., "rsa": ..}}. Used for
    both the code (vs predicted geometry E) and the decoder (vs predicted decoder D)."""
    names = list(scores); x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.3))
    for off, key, color in [(-w / 2, "cka", SERIES[2]), (w / 2, "rsa", SERIES[1])]:
        ax.bar(x + off, [scores[n][key] for n in names], w, color=color,
               edgecolor="white", linewidth=1.2, label=key.upper())
    ax.set(xticks=x, ylim=(0, 1.16), ylabel=ylabel)
    ax.set_xticklabels(names)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2)
    ax.set_title(title, pad=30)
    FIG.style_axes(ax)
    return fig


# ── unified belief-representation figures (shared by the HMM and PCFG notebooks) ──

def plot_prediction_pair(reference, prediction, name, colors, method="linear", cmap=None,
                         panel=3.8, ref_label="transformer"):
    """The prediction's own 2-D cloud (top-2 PCA) beside ``reference`` aligned into that plane
    (``method``: 'linear' = CCA-level, 'rotation' = CKA-level), R2 annotated; shared axes and
    coloring."""
    from .analysis_utils import align_representations
    Y = np.asarray(prediction, float); Yc = Y - Y.mean(0)
    Y2 = Yc @ np.linalg.svd(Yc, full_matrices=False)[2][:2].T
    mapped, r2 = align_representations(np.asarray(reference, float), Y2, method=method)
    fig, ax = plt.subplots(1, 2, figsize=(panel * 2, panel + 0.6))
    for a, (pts, title) in zip(ax, [(Y2, name), (mapped, f"{ref_label} in {name} space")]):
        a.scatter(pts[:, 0], pts[:, 1], c=colors, cmap=cmap or FIG.cyclic_cmap, s=10,
                  alpha=0.6, edgecolors="none")
        a.set_aspect("equal"); FIG.clean(a); a.set_title(title, fontsize=13, pad=8)
    ax[1].set_xlim(*ax[0].get_xlim()); ax[1].set_ylim(*ax[0].get_ylim())
    ax[1].text(0.5, -0.04, f"R2 = {r2:.2f}", transform=ax[1].transAxes,
               ha="center", va="top", fontsize=11)
    fig.tight_layout()
    return fig


def plot_trajectory_overlay(ref_traj, other_traj, seqs=(0, 1), labels=("reference", "model"),
                            parity=None, suptitle=None, panel=4.3):
    """Per-sequence 2-D belief-path overlays: ``ref_traj`` / ``other_traj`` are (B, L, 2), drawn as
    dots vs crosses colored by position, one panel per index in ``seqs``. ``parity=(x, y, label)``
    appends a coordinate-wise y=x panel."""
    ref_traj = np.asarray(ref_traj, float); other_traj = np.asarray(other_traj, float)
    L = ref_traj.shape[1]
    n_panels = len(seqs) + (parity is not None)
    fig, ax = plt.subplots(1, n_panels, figsize=(panel * n_panels, panel), squeeze=False)
    ax = ax[0]
    for a, seq in zip(ax, seqs):
        a.plot(ref_traj[seq, :, 0], ref_traj[seq, :, 1], "-", color="0.8", lw=1, zorder=0)
        a.scatter(ref_traj[seq, :, 0], ref_traj[seq, :, 1], c=range(L), cmap="viridis", s=38,
                  zorder=1, label=labels[0])
        a.scatter(other_traj[seq, :, 0], other_traj[seq, :, 1], c=range(L), cmap="viridis",
                  marker="x", s=32, lw=1.5, zorder=2, label=labels[1])
        a.set_aspect("equal"); FIG.clean(a); a.set_title(f"sequence {seq}")
    ax[0].legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.02))
    if parity is not None:
        x, y, label = parity
        a = ax[-1]
        a.scatter(x, y, s=4, alpha=0.15, color=FIG.colors["blue"], edgecolors="none")
        lim = [min(np.min(x), np.min(y)), max(np.max(x), np.max(y))]
        a.plot(lim, lim, "--", color="0.4", lw=1)
        a.set_xlabel(labels[0]); a.set_ylabel(labels[1])
        a.set_title(f"coordinate-wise match\n({label})")
        FIG.style_axes(a); a.set_aspect("equal")
    if suptitle:
        fig.suptitle(suptitle, y=1.02)
    fig.tight_layout()
    return fig
