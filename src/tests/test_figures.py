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
    p_klinear = figure_path(_config(tmp_path, "graph", "klinear", "structure=ring"), "rsa_by_arch")
    p_transf = figure_path(_config(tmp_path, "graph", "transformer", "structure=ring"), "rsa_by_arch")

    # {setting}/{override_dirname}/figures/ — beside the per-model run dirs for that structure
    assert Path(p_klinear) == tmp_path / "graph" / "structure=ring" / "figures" / "rsa_by_arch.png"
    assert p_klinear == p_transf                       # model level dropped -> shared
    assert Path(p_klinear).parent.is_dir()             # parent created for immediate savefig


def test_figure_path_separates_structures(tmp_path):
    ring = figure_path(_config(tmp_path, "graph", "klinear", "structure=ring"), "geom")
    sphere = figure_path(_config(tmp_path, "graph", "klinear", "structure=sphere"), "geom")

    assert ring != sphere                               # different structure -> no renaming needed


def test_figure_path_pdf_for_paper_final(tmp_path):
    p = figure_path(_config(tmp_path, "graph", "klinear", "structure=ring"), "geom", ext="pdf")
    assert Path(p).suffix == ".pdf"


def test_save_figure_infers_format_from_extension(tmp_path):
    config = _config(tmp_path, "graph", "klinear", "structure=ring")
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    png, pdf = figure_path(config, "f"), figure_path(config, "f", ext="pdf")
    save_figure(fig, png)
    save_figure(fig, pdf)
    plt.close(fig)

    assert Path(png).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"   # PNG magic, not PDF bytes in a .png
    assert Path(pdf).read_bytes()[:5] == b"%PDF-"


def test_plot_alignment_grid_annotates_discriminating_r2():
    import numpy as np
    from src.code.plot_utils import plot_alignment_grid
    rng = np.random.default_rng(0)
    th = rng.uniform(0, 2 * np.pi, 300)
    ref = np.c_[np.cos(th), np.sin(th)]
    Q = np.linalg.qr(rng.normal(size=(2, 2)))[0]
    good = ref @ Q                                        # matches under rotation
    bad = rng.normal(size=(300, 2))                       # matches under neither

    fig = plot_alignment_grid(ref, {"good": good, "bad": bad}, methods=("rotation",), colors=th)
    axes = [a for a in fig.axes if a.get_label() != "<colorbar>"]
    assert len(axes) == 3                                 # reference + two candidates, one method row
    r2s = [float(t.get_text().split("=")[1]) for a in axes for t in a.texts if "R2" in t.get_text()]
    assert r2s[0] == 1.0 and r2s[1] > 0.99 and r2s[2] < 0.5
    plt.close(fig)

    fig = plot_alignment_grid(ref, {"good": good}, methods=("linear", "rotation"), colors=th)
    axes = [a for a in fig.axes if a.get_label() != "<colorbar>"]
    assert len(axes) == 4                                 # 2 methods x (reference + 1 candidate)
    plt.close(fig)


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
