"""Unit test for the offline (local) eval-results loader."""
import torch
from omegaconf import OmegaConf

from src.code.analysis_utils import load_eval_results


def _write_run(root, override, steps):
    run_dir = root / "graph-cooccurrence" / override / "my_run"
    run_dir.mkdir(parents=True)
    OmegaConf.save(OmegaConf.create({"random_seed": 0, "name": "my_run"}),
                   run_dir / "config_resolved.yaml")
    for step in steps:
        ckpt = run_dir / "checkpoints" / str(step)
        ckpt.mkdir(parents=True)
        torch.save({"metrics": {"loss": 0.1 * step}, "extras": {}, "step": step},
                   ckpt / "metrics.pt")


def test_load_local_spreads_metrics_and_sweep_params(tmp_path, monkeypatch):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path))
    _write_run(tmp_path, "structure=ring", steps=[10, 20])

    df = load_eval_results("graph-cooccurrence", "my_run", source="local")

    assert len(df) == 2
    assert set(["structure", "step", "config", "loss"]).issubset(df.columns)
    assert df["structure"].tolist() == ["ring", "ring"]
    assert sorted(df["step"].tolist()) == [10, 20]
    assert df.sort_values("step")["loss"].tolist() == [1.0, 2.0]


def test_align_representations_rotation_separates_what_linear_cannot():
    import numpy as np
    from src.code.analysis_utils import align_representations
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(400, 4))
    Q = np.linalg.qr(rng.normal(size=(4, 4)))[0]

    rotated = ref @ Q                                       # pure rotation: both methods perfect
    for method in ("linear", "rotation"):
        aligned, r2 = align_representations(rotated, ref, method=method)
        assert r2 > 0.999
        assert np.allclose(aligned, ref, atol=1e-6)         # aligned output lives in ref's space

    stretched = (ref * np.array([5.0, 1.0, 1.0, 1.0])) @ Q  # per-axis stretch: rotation can't fix it
    _, r2_lin = align_representations(stretched, ref, method="linear")
    _, r2_rot = align_representations(stretched, ref, method="rotation")
    assert r2_lin > 0.999 and r2_rot < 0.9

    wide = np.c_[ref @ Q, 0.01 * rng.normal(size=(400, 60))]   # wide hidden: PCA-truncate then rotate
    _, r2 = align_representations(wide, ref, method="rotation")
    assert r2 > 0.99


def test_heldout_decode_r2_recovers_linear_map_and_respects_groups():
    import numpy as np
    from src.code.analysis_utils import heldout_decode_r2
    rng = np.random.default_rng(0)
    F = rng.normal(size=(80, 3))
    groups = np.repeat(np.arange(40), 2)                    # duplicated prefixes share a group
    H = np.repeat(F[::2] @ rng.normal(size=(3, 32)), 2, axis=0)  # hidden is a linear image of F
    assert heldout_decode_r2(H, np.repeat(F[::2], 2, 0), groups, seed=0) > 0.95
    assert heldout_decode_r2(H, rng.normal(size=(80, 3)), groups, seed=0) < 0.3


def test_heldout_decode_r2_ignores_units_flat_on_the_train_split():
    """A saturated unit (tanh-flat over the training prefixes, moving only on a held-out one) carries
    no fitted information: standardizing it by its ~0 train std turns that flat noise into an O(1)
    regressor, then multiplies the held-out excursion by the same huge factor and wrecks R². Such
    units must be dropped, not rescaled."""
    import numpy as np
    from src.code.analysis_utils import heldout_decode_r2
    rng = np.random.default_rng(0)
    F = rng.normal(size=(80, 3))
    groups = np.arange(80)
    train_mask = np.arange(80) < 60
    flat = 1e-9 * rng.normal(size=80); flat[-1] = 50.0       # flat on train, excursion on one test row
    H = F @ rng.normal(size=(3, 16))
    clean = heldout_decode_r2(H, F, groups, train_mask=train_mask)
    assert clean > 0.95
    assert heldout_decode_r2(np.c_[H, flat], F, groups, train_mask=train_mask) > clean - 0.01
