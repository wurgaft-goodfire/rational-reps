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
    """A minimal Config whose run_dir mirrors the real {cache}/{setting}/{model}/{dirname} layout."""
    run_dir = str(Path(cache_dir) / setting / model_name / override_dirname)
    cfg = OmegaConf.create({"random_seed": 0, "cache_dir": str(cache_dir),
                            "setting": setting, "run_dir": run_dir})
    return Config(cfg)


def test_figure_path_is_experiment_keyed_dropping_model_level(tmp_path):
    # two models, same structure -> same figures dir (comparison figures share a home)
    p_klinear = figure_path(_config(tmp_path, "graph", "klinear", "structure=ring"), "rsa_by_arch")
    p_transf = figure_path(_config(tmp_path, "graph", "transformer", "structure=ring"), "rsa_by_arch")

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
