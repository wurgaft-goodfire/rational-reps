"""The canonical cases of notes/superposition_new.tex on the signed feature cube: one structure
spec per case, the ranks it is trained at, and the overrides that name a run."""
from .train import graph_overrides

SEEDS = (0, 1, 2)
CASES = {
    # weak logits, dense: all 27 states of three signed features; the cube collapses onto the
    # rectangle of the two strongest features at d = 2
    "weak": dict(spec={"structure": "feature_cube", "m": 3, "signed": True, "all_bundles": True,
                       "scale": 0.5, "spectrum": "power", "beta": 0.5}, d=(3, 2)),
    # sharp singletons of equal strength: the 2m tokens pack the plane as a regular 2m-gon
    "sharp": dict(spec={"structure": "feature_cube", "m": 8, "k": 1, "signed": True, "all_bundles": True,
                        "scale": 12.0, "decay": 1.0}, d=(2,)),
    # singletons on a power-law ladder: the overlap envelope |cos| < (i/j)^(gamma/2)
    "power_half": dict(spec={"structure": "feature_cube", "m": 16, "k": 1, "signed": True, "all_bundles": True,
                             "scale": 10.0, "spectrum": "power", "beta": 0.5}, d=(4,)),
    "power_one": dict(spec={"structure": "feature_cube", "m": 16, "k": 1, "signed": True, "all_bundles": True,
                            "scale": 10.0, "spectrum": "power", "beta": 1.0}, d=(4,)),
    # singleton patterns seen in mixed coordinates: encoder angles are minus the partial correlations;
    # moderate strength so the losing tokens carry weight and training pins the whole geometry
    "mixed": dict(spec={"structure": "feature_cube", "m": 8, "k": 1, "signed": True, "all_bundles": True,
                        "scale": 3.0, "decay": 1.0, "mix": 0.5}, d=(8, 4)),
}


def case_overrides(name, runner, d, seed):
    """Hydra overrides of one run of a case: its structure, the model class, the rank and the seed. Trained
    by plain gradient descent from small init, which keeps E'E - D'D conserved, so the saved factors are
    balanced and representation and readout can be drawn in one frame (as the concepts worlds are)."""
    return graph_overrides(CASES[name]["spec"]) + [f"runner={runner}", f"model_kwargs.model_config.embed_dim={d}",
                                                   f"random_seed={seed}", "training_kwargs.train_steps=8000",
                                                   "training_kwargs.optimizer=sgd", "training_kwargs.learning_rate=0.5"]


def all_runs(runners=("linear", "optimum")):
    """Every (name, runner, d, seed) run of every case."""
    return [(name, runner, d, seed) for name, case in CASES.items() for runner in runners for d in case["d"] for seed in SEEDS]
