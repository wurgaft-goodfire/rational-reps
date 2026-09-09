"""Figure artifacts land under the (gitignored) cache, keyed by experiment (setting + config
overrides), not by model — so cross-model comparison figures for one structure share a dir."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from src.code.config import Config
from src.code.plot_utils import figure_path, save_figure


def _config(cache_dir, setting, model_name, override_dirname):
    """A minimal Config whose run_dir mirrors the real {cache}/{setting}/{dirname}/{model} layout."""
    run_dir = str(Path(cache_dir) / setting / override_dirname / model_name)
    cfg = OmegaConf.create({"random_seed": 0, "cache_dir": str(cache_dir),
                            "setting": setting, "run_dir": run_dir})
    return Config(cfg)


def test_figure_path_is_experiment_keyed_dropping_model_level(tmp_path):
    # two models, same structure -> same figures dir (comparison figures share a home)
    p_linear = figure_path(_config(tmp_path, "graph", "linear", "structure=ring"), "rsa_by_arch")
    p_transf = figure_path(_config(tmp_path, "graph", "transformer", "structure=ring"), "rsa_by_arch")

    # {setting}/{override_dirname}/figures/ — beside the per-model run dirs for that structure
    assert Path(p_linear) == tmp_path / "graph" / "structure=ring" / "figures" / "rsa_by_arch.png"
    assert p_linear == p_transf                       # model level dropped -> shared
    assert Path(p_linear).parent.is_dir()             # parent created for immediate savefig


def test_figure_path_separates_structures(tmp_path):
    ring = figure_path(_config(tmp_path, "graph", "linear", "structure=ring"), "geom")
    sphere = figure_path(_config(tmp_path, "graph", "linear", "structure=sphere"), "geom")

    assert ring != sphere                               # different structure -> no renaming needed


def test_figure_path_pdf_for_paper_final(tmp_path):
    p = figure_path(_config(tmp_path, "graph", "linear", "structure=ring"), "geom", ext="pdf")
    assert Path(p).suffix == ".pdf"


def test_save_figure_writes_one_format_when_asked(tmp_path):
    """``format=`` forces a single file rather than the PNG+PDF pair, still filed under its format
    subfolder — and the bytes match the format, not just the extension."""
    config = _config(tmp_path, "graph", "linear", "structure=ring")
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    base = Path(figure_path(config, "f"))
    save_figure(fig, base, format="png")
    save_figure(fig, base.with_suffix(".pdf"), format="pdf")
    plt.close(fig)

    assert (base.parent / "png" / "f.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert (base.parent / "pdf" / "f.pdf").read_bytes()[:5] == b"%PDF-"


def test_plot_trajectory_overlay_panels_and_parity():
    import numpy as np
    from src.code.plot_utils import plot_trajectory_overlay
    rng = np.random.default_rng(0)
    A = rng.normal(size=(4, 10, 2)); B = A + 0.05 * rng.normal(size=A.shape)

    fig = plot_trajectory_overlay(A, B, seqs=(0, 2), labels=("belief", "network"))
    assert len(fig.axes) == 2                             # one panel per requested sequence
    plt.close(fig)

    fig = plot_trajectory_overlay(A, B, seqs=(0,), parity=(A.ravel(), B.ravel(), "rotation only"))
    assert len(fig.axes) == 2                             # sequence panel + parity panel
    plt.close(fig)


def test_plot_prediction_pair_is_a_two_panel_row():
    import numpy as np
    from src.code.plot_utils import plot_prediction_pair
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(200, 5))
    H = np.c_[Y, rng.normal(size=(200, 7))]               # reference contains Y linearly
    for method in ("linear", "rotation"):
        fig = plot_prediction_pair(H, Y, "belief", colors=np.arange(200) % 4, method=method)
        assert len(fig.axes) == 2                         # [prediction, aligned reference]
        r2s = [t.get_text() for a in fig.axes for t in a.texts if "R2" in t.get_text()]
        assert len(r2s) == 1
        plt.close(fig)


def test_plot_factorization_renders_the_three_panel_decomposition():
    from src.code.data import graph_conditional
    from src.code.plot_utils import plot_factorization
    L = graph_conditional({"structure": "ring", "n": 24, "gamma": 0.8, "cooccurrence": "diffusion"})
    fig, axes = plot_factorization(L, k=3, highlight_row=0, highlight_col=6)
    assert set(axes) == {"L", "E", "D"}
    assert axes["L"].images[0].get_array().shape == (24, 24)
    assert axes["E"].images[0].get_array().shape == (24, 3)
    assert axes["D"].images[0].get_array().shape == (3, 24)
    plt.close(fig)


def test_save_figure_writes_both_formats_into_per_format_subfolders(tmp_path):
    """One call, both formats, filed by format: the caller names ``.../figures/name.png`` and the
    files land in ``figures/png/name.png`` and ``figures/pdf/name.pdf``, so a directory holds one
    kind of file and the PDFs never clutter the browsable PNGs."""
    config = _config(tmp_path, "graph", "linear", "structure=ring")
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    asked = Path(figure_path(config, "both"))
    save_figure(fig, asked)
    plt.close(fig)

    png, pdf = asked.parent / "png" / "both.png", asked.parent / "pdf" / "both.pdf"
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert pdf.read_bytes()[:5] == b"%PDF-"
    assert not asked.exists()                      # nothing left loose in figures/


def test_embedding_panels_color_by_a_supplied_scalar_and_annotate_the_gram():
    # relational panels color pairs by their mean outcome, not by row index, and carry CKA
    import numpy as np
    from src.code.plot_utils import plot_embedding_panels

    E = np.random.default_rng(0).normal(size=(12, 4))
    colors = np.arange(12) % 3
    fig = plot_embedding_panels(E, {"mlp": E}, colors=colors, scores={"mlp": 0.87})

    scatter = fig.axes[2].collections[0]                       # the mlp scatter panel
    assert np.allclose(scatter.get_array(), colors)
    assert "CKA 0.87" in fig.axes[3].get_title()               # its Gram carries the score
    plt.close(fig)


def test_scatter_geometry_does_not_collapse_a_one_dimensional_code():
    """approx_add's predicted code is a straight line in log(a+b): equal aspect on a zero-height
    cloud shrinks the axes box to nothing and the panel renders empty. A near-1-D cloud must get
    padded height instead."""
    import numpy as np
    from src.code.plot_utils import scatter_geometry

    XY = np.c_[np.linspace(0, 4, 20), np.zeros(20)]
    fig, ax = plt.subplots()
    scatter_geometry(ax, XY, "a line")

    lo, hi = ax.get_ylim()
    assert hi - lo > 1.0, f"1-D code squashed to a {hi - lo:.2g}-tall panel"
    plt.close(fig)


def test_embedding_panels_accept_precomputed_scatter_coordinates():
    """A group's spectrum is flat, so top-2 PCA picks an arbitrary mixture of irreps and the circle
    vanishes. The panel must be able to draw a chosen plane while the Gram below still shows the
    whole code."""
    import numpy as np
    from src.code.plot_utils import plot_embedding_panels

    rng = np.random.default_rng(0)
    E, X = rng.normal(size=(12, 6)), rng.normal(size=(12, 6))
    ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    circle = np.c_[np.cos(ang), np.sin(ang)]

    fig = plot_embedding_panels(E, {"mlp": X}, coords={"predicted": circle, "mlp": circle})

    assert np.allclose(fig.axes[2].collections[0].get_offsets(), circle)   # the chosen plane
    assert fig.axes[3].images[0].get_array().shape == (12, 12)             # Gram is still 12x12
    plt.close(fig)


def test_ultrametric_survives_leaves_that_are_all_far_apart():
    """The merge-absorbing tolerance must scale with the spread of merge heights, not their
    absolute size. A relational pair code puts every leaf far from every other -- entailment's
    merge heights are 2.93 / 3.22 / 3.33, so the level gaps are 10% and 3% of the maximum and a
    tolerance of 0.2 * max swallows every merge, collapsing an exact ultrametric to a star."""
    import numpy as np
    from src.code.data import apply_lapse, entailment
    from src.code.geometry import predicted_geometry
    from src.code.plot_utils import _draw_ultrametric

    n_nodes, leaves = 13, list(range(4, 13))            # depth 2, branch 3: 9 leaves, 4 inner nodes
    T = np.asarray(entailment(2, 3), float).reshape(-1, n_nodes)
    E = np.asarray(predicted_geometry(np.log(apply_lapse(T, 0.05))), float)
    parent = [-1, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    lca = [max(set(_chain(a, parent)) & set(_chain(b, parent)),
               key=lambda u: len(_chain(u, parent)))
           for a in range(n_nodes) for b in range(n_nodes)]
    lca = np.array(lca)
    leaf_code = np.stack([E[lca == c].mean(0) for c in leaves])

    fig, ax = plt.subplots()
    _draw_ultrametric(ax, leaf_code, np.arange(len(leaves)), len(leaves))
    # the merge nodes are the one scatter that is not the 9 coloured leaves
    merges = [c for c in ax.collections if len(c.get_offsets()) not in (0, len(leaves))]
    drawn = sum(len(c.get_offsets()) for c in merges)
    assert drawn >= 4, f"dendrogram collapsed to {drawn} internal node(s); the 3 branches are lost"
    plt.close(fig)


def _chain(v, parent):
    """A node's ancestors, root last."""
    out = []
    while v != -1:
        out.append(v); v = parent[v]
    return out


def test_embedding_panels_accept_a_spec_that_only_sets_the_view():
    """A relational panel may want a 3-D scatter without naming a graph structure, so the spec can
    carry ``view`` alone."""
    import numpy as np
    from src.code.plot_utils import plot_embedding_panels

    E = np.random.default_rng(0).normal(size=(10, 5))
    fig = plot_embedding_panels(E, {"mlp": E}, spec={"view": "3d"})
    assert len(fig.axes) == 4
    plt.close(fig)


def test_a_wide_flat_scatter_is_not_clipped_at_the_edges():
    """approx_add's code is a long shallow arc. The near-1-D branch used a fixed 5% x-margin, too
    tight for the marker radius, so the extreme points were cut in half at both ends."""
    import numpy as np
    from src.code.plot_utils import scatter_geometry

    x = np.linspace(0, 10, 24)
    XY = np.c_[x, 0.2 * np.sin(x / 3)]                    # wide and shallow
    fig, ax = plt.subplots()
    scatter_geometry(ax, XY, "arc")

    lo, hi = ax.get_xlim()
    pad = min(XY[:, 0].min() - lo, hi - XY[:, 0].max()) / np.ptp(XY[:, 0])
    assert pad > 0.1, f"only {pad:.1%} of x-padding: the end markers will be clipped"
    plt.close(fig)


def test_an_elongated_equal_aspect_scatter_does_not_clip_its_markers():
    """A 7:1 cloud under equal aspect gets an axes box a fraction of an inch tall, so the 15% data
    margin is thinner than the marker radius and the extreme marker is sliced in half (approx_add's
    mlp_paired panel). The panel is borderless, so markers must not be clipped at all."""
    import numpy as np
    from src.code.plot_utils import scatter_geometry

    x = np.linspace(0, 10, 24)
    XY = np.c_[x, 0.13 * x]                               # elongated, but past the near-1-D branch
    fig, ax = plt.subplots(figsize=(2.7, 2.95))
    scatter_geometry(ax, XY, "elongated")

    assert not ax.collections[0].get_clip_on(), "markers are clipped to the axes patch"
    plt.close(fig)


def test_merge_panels_lays_saved_panels_out_column_major_with_letters(tmp_path):
    """The merged figures are just saved panels tiled and lettered; the notebook should say that
    in one call rather than rebuild a gridspec each time."""
    from src.code.plot_utils import merge_panels, save_figure

    paths = []
    for i in range(4):
        f, ax = plt.subplots(figsize=(2, 1)); ax.plot([0, i])
        save_figure(f, tmp_path / f"panel{i}.png", format="png"); plt.close(f)
        paths.append(tmp_path / "png" / f"panel{i}.png")

    fig = merge_panels(paths, labels=["w", "x", "y", "z"], ncols=2)
    assert [a.get_title(loc="left") for a in fig.axes] == ["(a) w", "(c) y", "(b) x", "(d) z"]
    assert all(not a.axison for a in fig.axes)
    plt.close(fig)


def test_stack_figures_draws_each_panel_into_its_own_subfigure_row():
    """The story figures are three saved panels stacked vertically; drawing them into one
    figure's subfigures keeps the composite vector (pdf) instead of pasting PNGs."""
    import numpy as np
    from src.code.plot_utils import stack_figures

    def _draw(tag):
        def draw(fig):
            fig.add_subplot().set_title(tag)
        return draw

    fig = stack_figures([(3.0, _draw("a")), (5.0, _draw("b")), (2.0, _draw("c"))], width=10)
    assert [a.get_title() for a in fig.axes] == ["a", "b", "c"]
    # each panel lives in its own subfigure, at the requested height ratio, top to bottom
    heights = [a.figure.bbox.height for a in fig.axes]
    assert np.allclose(np.array(heights) / heights[2], [1.5, 2.5, 1.0])
    tops = [a.figure.bbox.y1 for a in fig.axes]
    assert tops[0] > tops[1] > tops[2]
    plt.close(fig)
