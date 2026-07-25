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


# ── untrained-model control (shape only; the CCA gap is a notebook/slow check) ──
def test_untrained_twin_matches_architecture():
    import torch
    from omegaconf import OmegaConf
    from src.code.config import Config
    from src.code.data import PCFGDataset
    from src.code.analysis_utils import untrained_twin
    cfg = Config(OmegaConf.create({
        "setting": "pcfg", "grammar": "dyck", "grammar_kwargs": {"k": 2, "depth": 3},
        "max_len": 6, "sample_mode": "sequence", "random_seed": 0,
        "model_kwargs": {"model_type": "transformer", "model_config": {
            "hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 2,
            "intermediate_size": 64}},
        "training_kwargs": {"batch_size": 8, "context_length": 40},
        "eval_kwargs": {"num_eval_batches": 1, "output_hidden_states": True},
    }))
    ds = PCFGDataset(cfg)
    model = untrained_twin(cfg, ds)
    ids = torch.zeros(2, ds.seq_len, dtype=torch.long, device=ds.device)
    out = model(ids, output_hidden_states=True)
    assert out["hidden_states"][-1].shape == (2, ds.seq_len, 32)
