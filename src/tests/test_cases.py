"""The canonical cases of notes/superposition_new.tex (src/code/cases.py): every case's overrides
compose to a signed feature-cube task of the expected size."""
import numpy as np

from src.code.cases import CASES, all_runs, case_overrides
from src.code.data import init_dataset
from src.code.train import resolve_config

EXPECTED_STATES = {"weak": 27, "sharp": 16, "power_half": 32, "power_one": 32, "mixed": 16, "frequency": 32}


def test_every_case_composes_to_a_signed_task_of_the_expected_size():
    for name, case in CASES.items():
        config = resolve_config(case_overrides(name, "linear", case["d"][0], 0))
        ds = init_dataset(config)
        Z, m = np.asarray(ds.features, float), case["spec"]["m"]
        assert Z.shape == (EXPECTED_STATES[name], m) and ds.P.shape == (EXPECTED_STATES[name], 2 * m)
        if "mix" in case["spec"]:
            assert set(np.unique(ds.info["patterns"])) == {-1.0, 0.0, 1.0}
        else:
            assert set(np.unique(Z)) <= {-1.0, 0.0, 1.0}
        assert config.cfg.model_kwargs.model_config.embed_dim == case["d"][0]
    runs = all_runs()
    assert len(runs) == len(set(runs)) == 3 * sum(len(c.get("runners", (0, 0))) * len(c["d"]) for c in CASES.values())
    assert {r[1] for r in runs if r[0] == "frequency"} == {"linear"}
    assert resolve_config(case_overrides("sharp", "linear", 2, 0)).training_kwargs["optimizer"] == "sgd"
    assert resolve_config(case_overrides("frequency", "linear", 5, 0)).training_kwargs["optimizer"] == "adamw_torch"
