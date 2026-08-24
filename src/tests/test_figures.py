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


def test_save_figure_infers_format_from_extension(tmp_path):
    config = _config(tmp_path, "graph", "linear", "structure=ring")
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    png, pdf = figure_path(config, "f"), figure_path(config, "f", ext="pdf")
    save_figure(fig, png)
    save_figure(fig, pdf)
    plt.close(fig)

    assert Path(png).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"   # PNG magic, not PDF bytes in a .png
    assert Path(pdf).read_bytes()[:5] == b"%PDF-"


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
