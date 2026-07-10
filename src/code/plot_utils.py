#!/usr/bin/env python3
"""Publication plotting. `FigureGenerator` carries the house style — large Avenir fonts plus the
demo_graphs visual language (twilight/vlag colormaps, nearest-interpolation heatmaps, borderless
manifold panels) — and the geometry-analysis helpers render through it (structure -> conditional
-> embedding), matching src/notebooks/demo_graphs.ipynb.
"""
import itertools
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.collections import LineCollection


class FigureGenerator:
    """Consistent styling + high-quality saving for figures."""

    def __init__(self):
        # palette + colormaps from demo_graphs
        self.colors = {'blue': '#4C72B0', 'orange': '#DD8452', 'green': '#55A868',
                       'purple': '#8172B3', 'gray': '#8C8C8C'}
        self.categorical = ['#4C72B0', '#DD8452', '#55A868', '#8172B3']   # per-series (curves/bars)
        self.cyclic_cmap = 'twilight'      # item index -> color (a ring reads as a ring)
        self.diverging_cmap = 'vlag'       # centered conditional heatmaps
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
        infers the format from the filepath extension (``.png`` for viewing, ``.pdf`` for papers)."""
        for ax in fig.get_axes():
            for collection in ax.collections:
                if collection.__class__.__name__ in ['QuadMesh', 'AxesImage']:
                    collection.set_rasterized(True)
                if hasattr(collection, '_offsets') and hasattr(collection._offsets, 'shape'):
                    if collection._offsets.shape[0] > 500:
                        collection.set_rasterized(True)
        fig.savefig(filepath, bbox_inches='tight', dpi=dpi, format=format)
        print(f"Saved figure to {filepath}")


# Module-level singleton: importing plot_utils applies the house style.
FIG = FigureGenerator()


def save_figure(fig, filepath, **kwargs):
    """Save a figure with the house rasterization/quality settings."""
    FIG.save_figure(fig, filepath, **kwargs)


def figure_path(config, name, ext="png"):
    """Path for a figure under the gitignored cache, keyed by *experiment* — ``{cache_dir}/{setting}/
    figures/{override_dirname}/`` — a per-setting figures tree (separate from the run artifacts),
    dropping the per-model level of the run dir so cross-model comparison figures for one structure
    share a home and switching structures needs no renaming. Defaults to PNG (viewable everywhere);
    pass ``ext="pdf"`` for a paper-final vector figure. Creates the parent dir so ``save_figure`` can
    write immediately."""
    override_dirname = Path(config.run_dir).name                 # run_dir = .../{setting}/{model}/{dirname}
    path = Path(config.cache_dir) / config.setting / "figures" / override_dirname / f"{name}.{ext}"
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


######### Geometry-analysis plots (used by geometry_analysis.ipynb) #########

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
    """The presentation view for a structure's geometry: ``spec['view']`` if set (a yaml-able
    structure kwarg) else the per-structure default — ``"torus"`` for a product of two rings, ``"3d"``
    for intrinsically-3-D structures (sphere, hypercube, complete, ...), ``"ultrametric"`` for trees,
    ``"2d"`` otherwise. Values: ``"2d"`` | ``"3d"`` (PCA scatters) | ``"torus"`` (3-D donut) |
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
    from .data import structure_adjacency, chain, tree_parents
    name = spec["structure"]; p = {k: v for k, v in spec.items() if k != "structure"}
    if name == "log_line":
        n = p.get("n", 30)
        return chain(n), np.c_[np.log1p(np.arange(n)), np.zeros(n)]
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
        if min(d0, d1) < 0.1 * max(d0, d1):               # ~1-D (chain / log line): fill, don't squash
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
                   vmin=0, vmax=vmax, s=25)
        if labels is not None:
            for (x, y, z), lab in zip(pos, labels):
                ax.text(x, y, z, lab, fontsize=6, ha="center", va="bottom")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title(title)


def scatter_geometry(ax, XY, title, cmap=None):
    """Embedding scatter colored by item index (borderless manifold panel); 2-D or 3-D by XY's width."""
    XY = np.asarray(XY); c = np.arange(len(XY)); cmap = cmap or FIG.cyclic_cmap
    if XY.shape[1] >= 3:
        ax.scatter(XY[:, 0], XY[:, 1], XY[:, 2], c=c, cmap=cmap, s=28)
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
    if name in ("hypercube", "complete"):
        # draw the graph edges over the learned positions so the cube / simplex is legible as a
        # connected polytope, not just a point cloud (as demo_graphs did for the cube)
        from .data import structure_adjacency
        draw_structure(ax, structure_adjacency(spec), _pca(r, ndim), title, cmap=cmap)
        return
    scatter_geometry(ax, _pca(r, ndim), title, cmap)


def conditional_heatmap(ax, L, title="log-conditional", xlabel=r"outcome  $x$", ylabel=r"item  $\theta$"):
    """Diverging heatmap of the (centered) conditional the geometry is read off (square, borderless,
    with axis labels)."""
    Ld = np.asarray(L) - np.asarray(L).mean()
    lim = np.percentile(np.abs(Ld), 98)
    ax.imshow(Ld, cmap=FIG.diverging_cmap, vmin=-lim, vmax=lim, aspect="equal")
    FIG.clean(ax); ax.set_anchor("N")                     # drop ticks + spines, pin to top
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)     # keep just the axis labels


def plot_predicted_geometry(spec, L, E, cmap=None):
    """The demo triptych: latent structure -> conditional -> the geometry it encodes. ``spec['view']``
    (a yaml-able structure kwarg) picks the third panel; for a tree the choices are ``"ultrametric"``
    (a dendrogram of the code, the default), ``"pca"`` (the raw scatter), or ``"simplex"`` (the
    simplex-of-simplices grid, Park et al.); other structures use ``"2d"``/``"3d"`` PCA. ``spec`` may
    set ``leaves_only`` to make only leaves items — inner nodes then show as gray junctions and drop
    out of the conditional/code. ``cmap`` overrides the house cyclic colormap for item coloring."""
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
    """Uncertainty over a ring (demo_graphs §3): a belief (distribution over items) is coded as the
    belief-weighted average of item codes, ``belief @ E``. Shows codes on the circle with two
    trajectories — mixing two beliefs (a chord through the interior) and broadening one belief
    (rim -> center) — plus the belief distributions. Assumes ``E`` is a ring code (items ordered)."""
    n = len(E); th = 2 * np.pi * np.arange(n) / n; circ = np.asarray(E)[:, :2]
    vm = lambda mu, k: (lambda b: b / b.sum())(np.exp(k * np.cos(th - mu)))
    bA, bB = vm(th[int(0.15 * n)], 12), vm(th[int(0.6 * n)], 12)
    ts = np.linspace(0, 1, 11)
    interp = np.array([(t * bA + (1 - t) * bB) @ circ for t in ts])
    ks = [12, 6, 3, 1.5, 0.7, 0.3, 0.1]
    broaden = np.array([vm(th[int(0.35 * n)], k) @ circ for k in ks])

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
    ax[0].scatter(circ[:, 0], circ[:, 1], c="0.85", s=15)
    ax[0].scatter([0], [0], c="0.6", marker="+", s=60)
    ax[0].plot(interp[:, 0], interp[:, 1], "-", color="0.6", lw=0.8, zorder=1)
    ax[0].scatter(interp[:, 0], interp[:, 1], c=ts, cmap="coolwarm", s=45, zorder=3)
    ax[0].plot(broaden[:, 0], broaden[:, 1], "-", color="0.6", lw=0.8, zorder=1)
    ax[0].scatter(broaden[:, 0], broaden[:, 1], c=range(len(ks)), cmap="viridis_r",
                  s=45, marker="s", zorder=3)
    ax[0].set_title("codes on the circle\n(o) mix A-B    (s) broaden to center")
    ax[0].set_aspect("equal"); FIG.clean(ax[0])

    for t in [0, 0.25, 0.5, 0.75, 1.0]:
        ax[1].plot(np.arange(n), t * bA + (1 - t) * bB, lw=1.4, color=plt.cm.coolwarm(t))
    ax[1].set(title="mix A-B (midpoint two-peaked)", xlabel="item", yticks=[]); FIG.style_axes(ax[1])

    for i, k in enumerate(ks):
        ax[2].plot(np.arange(n), vm(th[int(0.35 * n)], k), lw=1.4, color=plt.cm.viridis_r(i / len(ks)))
    ax[2].set(title="broaden one belief (sharp to flat)", xlabel="item", yticks=[]); FIG.style_axes(ax[2])
    fig.tight_layout()
    return fig


def _circle2d(X):
    """Top-2 PCA of an embedding block, normalized to unit mean radius (for drawing a circle)."""
    Xc = np.asarray(X) - np.asarray(X).mean(0)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    P = U[:, :2] * S[:2]
    return P / (np.linalg.norm(P, axis=1).mean() + 1e-9)


def plot_stacking(n=12, q=10, n_features=5, sae_steps=3000):
    """Combining structures shapes the code (demo_graphs §2): disjoint supports **union** into
    separate circles; shared independent factors form a **product** (torus = two factor circles);
    aligned relations **share** one circle; over-capacity features **superpose** into a polygon."""
    from .data import stacked_conditional
    from .geometry import factor_logcond
    from .models import train_sparse_autoencoder
    ring = lambda k: {"structure": "ring", "n": k}
    fig, ax = plt.subplots(2, 2, figsize=(8.5, 8))

    # disjoint supports -> union (two separate circles)
    Eu = factor_logcond(stacked_conditional([ring(n), ring(n)], "union"))[1]
    for sl, off in [(slice(0, n), -1.6), (slice(n, 2 * n), 1.6)]:
        P = _circle2d(Eu[sl])
        ax[0, 0].scatter(P[:, 0] + off, P[:, 1], c=np.arange(n), cmap=FIG.cyclic_cmap, s=35)
    ax[0, 0].set_title("disjoint supports: union (two circles)")
    ax[0, 0].set_aspect("equal"); FIG.clean(ax[0, 0])

    # shared support, independent factors -> product (a torus, shown as its two factor circles)
    Ep = factor_logcond(stacked_conditional([ring(n), ring(q)], "product"))[1]
    idx = np.arange(n * q)
    ci = _circle2d(np.stack([Ep[idx // q == i].mean(0) for i in range(n)]))
    cj = _circle2d(np.stack([Ep[idx % q == j].mean(0) for j in range(q)]))
    ax[0, 1].scatter(ci[:, 0] - 1.6, ci[:, 1], c=np.arange(n), cmap=FIG.cyclic_cmap, s=35)
    ax[0, 1].scatter(cj[:, 0] + 1.6, cj[:, 1], c=np.arange(q), cmap=FIG.cyclic_cmap, s=35)
    ax[0, 1].set_title("shared, independent: product (torus)")
    ax[0, 1].set_aspect("equal"); FIG.clean(ax[0, 1])

    # shared support, aligned relations -> share (one circle)
    Es = factor_logcond(stacked_conditional([ring(n), ring(n)], "share"))[1]
    P = _circle2d(Es)
    ax[1, 0].scatter(P[:, 0], P[:, 1], c=np.arange(n), cmap=FIG.cyclic_cmap, s=35)
    ax[1, 0].set_title("shared, aligned: share (one circle)")
    ax[1, 0].set_aspect("equal"); FIG.clean(ax[1, 0])

    # over capacity -> superpose (feature directions spread into a regular polygon)
    W = train_sparse_autoencoder(n_features, d=2, steps=sae_steps)
    for i in range(n_features):
        ax[1, 1].annotate("", xy=(W[0, i], W[1, i]), xytext=(0, 0),
                          arrowprops=dict(arrowstyle="-|>", color=plt.cm.twilight(i / n_features), lw=1.8))
    lim = np.abs(W).max() * 1.3
    ax[1, 1].set(xlim=(-lim, lim), ylim=(-lim, lim), title=f"over capacity: superpose ({n_features}-gon)")
    ax[1, 1].set_aspect("equal"); FIG.clean(ax[1, 1])

    fig.tight_layout()
    return fig


def rdm_heatmap(ax, reps, title="RDM"):
    """Representational dissimilarity matrix (pairwise Euclidean distances between item reps)."""
    r = np.asarray(reps)
    D = np.sqrt(((r[:, None] - r[None]) ** 2).sum(-1))
    ax.imshow(D, cmap="magma", aspect="equal")
    FIG.clean(ax); ax.set_anchor("N"); ax.set_title(title)


def plot_embedding_panels(E_pred, embeddings, L, spec=None, cmap=None):
    """Two rows. Top: predicted code beside each model's representation (pre-decode activations).
    Bottom: the predicted log-conditional beside each model's RDM. The top row follows the structure's
    ``view`` (see `_view`): for a tree ``"ultrametric"`` | ``"pca"`` | ``"simplex"``, else ``"2d"``/
    ``"3d"`` PCA. ``embeddings``: {name: reps}."""
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
        conditional_heatmap(axb, L, title="log-conditional") if j == 0 else rdm_heatmap(axb, r)
    if headers:                                          # column names for the simplex sub-grids
        fig.canvas.draw()
        for j, name in headers:
            bb = gs[0, j].get_position(fig)
            fig.text((bb.x0 + bb.x1) / 2, bb.y1 + 0.01, name, ha="center", va="bottom",
                     fontsize=FIG.font_sizes["axis_label"])
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
    style (straight light edges, index-colored leaf circles, gray merge nodes). ``colors`` is each
    leaf's item index, ``cmax`` the color range. Default tree view."""
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import pdist
    cmap = cmap or FIG.cyclic_cmap
    n = len(leaf_E)
    Z = linkage(pdist(np.asarray(leaf_E, float)), method="average")   # scipy clusters
    # custom layout off Z (straight edges, leaves pinned to item order for cross-panel alignment)
    pos = np.zeros((n + len(Z), 2)); pos[:n] = np.c_[np.arange(n, dtype=float), np.zeros(n)]
    edges = []
    for i, (a, b, h, _) in enumerate(Z):
        a, b, c = int(a), int(b), n + i
        pos[c] = [0.5 * (pos[a, 0] + pos[b, 0]), h]
        edges += [(c, a), (c, b)]
    pos = _elongate(pos)
    ax.add_collection(LineCollection([[pos[a], pos[b]] for a, b in edges],
                                     colors="0.85", linewidths=0.5, zorder=0))
    ax.scatter(pos[n:, 0], pos[n:, 1], c="0.8", s=24, zorder=1)                  # merge nodes
    ax.scatter(pos[:n, 0], pos[:n, 1], c=colors, cmap=cmap,
               vmin=0, vmax=cmax, s=70, zorder=2, clip_on=False)                 # leaves
    ax.set_title("ultrametric"); ax.set_aspect("equal"); ax.margins(0.12)
    FIG.clean(ax); ax.set_anchor("N")


def _draw_simplex_of_simplices(fig, subspec, parents, items, E, cmap=None, titles=True):
    """Park et al.'s "simplex of simplices": each split's child points (a child's rep, or its
    subtree-leaf centroid in leaves-only mode), minus their centroid and projected to the split's
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


def plot_convergence_bars(scores):
    """CKA/RSA to predicted per model. `scores`: {name: {"cka": .., "rsa": ..}}."""
    names = list(scores); x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.3))
    for off, key, color in [(-w / 2, "cka", SERIES[2]), (w / 2, "rsa", SERIES[1])]:
        ax.bar(x + off, [scores[n][key] for n in names], w, color=color,
               edgecolor="white", linewidth=1.2, label=key.upper())
    ax.set(xticks=x, ylim=(0, 1.16), ylabel="similarity to predicted geometry")
    ax.set_xticklabels(names)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2)
    ax.set_title("representational convergence across architectures", pad=30)
    FIG.style_axes(ax)
    return fig
