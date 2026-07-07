#!/usr/bin/env python3
"""Publication plotting. `FigureGenerator` carries the house style — large Avenir fonts plus the
demo_graphs visual language (twilight/vlag colormaps, nearest-interpolation heatmaps, borderless
manifold panels) — and the geometry-analysis helpers render through it (structure -> conditional
-> embedding), matching src/notebooks/demo_graphs.ipynb.
"""
import itertools
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
        avenir = [f for f in fm.findSystemFonts(fontext='ttc') if 'Avenir' in f]
        if avenir:
            fm.fontManager.addfont(avenir[0])
            plt.rcParams['font.family'] = fm.FontProperties(fname=avenir[0]).get_name()
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

    def save_figure(self, fig, filepath, format='pdf', dpi=1000):
        """Save figure with publication-quality settings and selective rasterization."""
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


######### Geometry-analysis plots (used by geometry_analysis.ipynb) #########

def _pca2(X):
    """Top-2 PCA coordinates of a representation (rows = items)."""
    Xc = np.asarray(X) - np.asarray(X).mean(0)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    return U[:, :2] * S[:2]


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


def _tree_graph(n, k):
    """Full depth-n, branching-k tree (internal nodes + leaves) with a hierarchical layout:
    leaves along the bottom, each internal node above the mean of its children."""
    nodes = [t for L in range(n + 1) for t in itertools.product(range(k), repeat=L)]
    idx = {nd: i for i, nd in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)))
    for nd in nodes:
        if nd:
            A[idx[nd], idx[nd[:-1]]] = A[idx[nd[:-1]], idx[nd]] = 1.0
    leaf_x = {nd: i for i, nd in enumerate(nd for nd in nodes if len(nd) == n)}
    def x(nd):
        return leaf_x[nd] if len(nd) == n else float(np.mean([x(nd + (c,)) for c in range(k)]))
    pos = np.array([[x(nd), -len(nd)] for nd in nodes], float)
    return A, pos


def _structure_graph(spec):
    """(adjacency, node positions) for drawing any structure's latent graph (2-D or 3-D)."""
    from .data import structure_adjacency, chain
    name = spec["structure"]; p = {k: v for k, v in spec.items() if k != "structure"}
    if name == "log_line":
        n = p.get("n", 30)
        return chain(n), np.c_[np.log1p(np.arange(n)), np.zeros(n)]
    if name == "tree":
        return _tree_graph(p.get("n", 3), p.get("k", 2))
    if name in ("product", "stack"):
        subs = [_structure_graph(f) for f in p["factors"]]
        A = structure_adjacency(spec)
        if name == "stack":                                   # lay factor blocks side by side
            blocks, xoff = [], 0.0
            for _, pi in subs:
                q = np.asarray(pi, float)[:, :2].copy()
                q[:, 0] += xoff - q[:, 0].min(); xoff = q[:, 0].max() + 2.0
                blocks.append(q)
            return A, np.vstack(blocks)
        grids = [np.asarray(pi, float) for _, pi in subs]     # product: concat factor coords
        pos = np.array([np.concatenate(combo) for combo in itertools.product(*grids)])
        pos = pos[:, :3] if pos.shape[1] >= 3 else np.c_[pos, np.zeros((len(pos), 3 - pos.shape[1]))]
        return A, pos
    return structure_adjacency(spec), _shape_positions(name, p)


def draw_structure(ax, A, pos, title="structure"):
    """Draw a latent graph: light edges + index-colored nodes at `pos` (2-D or 3-D)."""
    A, pos = np.asarray(A), np.asarray(pos); c = np.arange(len(pos))
    ii, jj = np.where(np.triu(A, 1) > 0)
    if pos.shape[1] == 2:
        ax.add_collection(LineCollection([[pos[a], pos[b]] for a, b in zip(ii, jj)],
                                         colors="0.85", linewidths=0.5, zorder=0))
        ax.scatter(pos[:, 0], pos[:, 1], c=c, cmap=FIG.cyclic_cmap, s=40, zorder=2)
        d0, d1 = np.ptp(pos[:, 0]), np.ptp(pos[:, 1])
        if min(d0, d1) < 0.1 * max(d0, d1):               # ~1-D (chain / log line): fill, don't squash
            pad = 0.25 * max(d0, d1)
            ax.set_ylim(pos[:, 1].min() - pad, pos[:, 1].max() + pad)
        else:
            ax.set_aspect("equal")
        FIG.clean(ax); ax.set_anchor("N")
    else:
        for a, b in zip(ii, jj):
            ax.plot(*zip(pos[a], pos[b]), color="0.85", lw=0.25, zorder=0)
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=c, cmap=FIG.cyclic_cmap, s=25)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title(title)


def scatter_geometry(ax, XY, title):
    """Embedding scatter colored by item index (borderless manifold panel)."""
    XY = np.asarray(XY)
    ax.scatter(XY[:, 0], XY[:, 1], c=np.arange(len(XY)), cmap=FIG.cyclic_cmap, s=45)
    ax.set_title(title); ax.set_aspect("equal"); FIG.clean(ax); ax.set_anchor("N")


def conditional_heatmap(ax, L, title="log-conditional", xlabel=r"outcome  $x$", ylabel=r"item  $\theta$"):
    """Diverging heatmap of the (centered) conditional the geometry is read off (square, borderless,
    with axis labels)."""
    Ld = np.asarray(L) - np.asarray(L).mean()
    lim = np.percentile(np.abs(Ld), 98)
    ax.imshow(Ld, cmap=FIG.diverging_cmap, vmin=-lim, vmax=lim, aspect="equal")
    FIG.clean(ax); ax.set_anchor("N")                     # drop ticks + spines, pin to top
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)     # keep just the axis labels


def plot_predicted_geometry(spec, L, E):
    """The demo triptych: latent structure -> conditional -> the closed-form embedding it encodes."""
    A, pos = _structure_graph(spec)
    is3d = pos.shape[1] == 3
    fig = plt.figure(figsize=(13, 3.9))
    axA = fig.add_subplot(1, 3, 1, projection="3d" if is3d else None)
    axB, axC = fig.add_subplot(1, 3, 2), fig.add_subplot(1, 3, 3)
    draw_structure(axA, A, pos)
    conditional_heatmap(axB, L)
    scatter_geometry(axC, E[:, :2], "embedding")
    fig.tight_layout(w_pad=5)
    # place the arrows in the gaps between panels (from the laid-out axes), so they never overlap
    for left, right in [(axA, axB), (axB, axC)]:
        pl, pr = left.get_position(), right.get_position()
        fig.text((pl.x1 + pr.x0) / 2, (pl.y0 + pl.y1) / 2, r"$\rightarrow$",
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


def plot_embedding_panels(E_pred, embeddings, L):
    """Two rows. Top: predicted code beside each model's representation (pre-decode activations,
    PCA'd to 2-D). Bottom: the predicted log-conditional beside each model's RDM. `embeddings`:
    {name: reps}."""
    names = ["predicted"] + list(embeddings)
    reps = [E_pred] + list(embeddings.values())
    fig, axes = plt.subplots(2, len(names), figsize=(2.7 * len(names), 5.6))
    for j, (name, r) in enumerate(zip(names, reps)):
        scatter_geometry(axes[0, j], _pca2(r), name)
        if j == 0:
            conditional_heatmap(axes[1, j], L, title="log-conditional")
        else:
            rdm_heatmap(axes[1, j], r)
    fig.tight_layout()
    return fig


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
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, [scores[n]["cka"] for n in names], w, color=FIG.colors["blue"], label="CKA")
    ax.bar(x + w / 2, [scores[n]["rsa"] for n in names], w, color=FIG.colors["orange"], label="RSA")
    ax.set(xticks=x, ylim=(0, 1), ylabel="similarity to predicted geometry",
           title="representational convergence across architectures")
    ax.set_xticklabels(names); ax.legend(frameon=False); FIG.style_axes(ax)
    return fig
