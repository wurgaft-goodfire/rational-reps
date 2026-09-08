"""Unit tests for the Evaluator: every input once against its exact target, the saved
representation, the eval.npz artifact, and the collator dropping eval-only keys. Uses the graph
dataset."""
import numpy as np
import torch
from omegaconf import OmegaConf

from src.code.config import Config
from src.code.data import KernelDataset
from src.code.eval import Evaluator
from src.code.train_utils import PreBatchedDataCollator


def _config():
    cfg = OmegaConf.create({
        "setting": "graph-cooccurrence", "structure": "ring", "structure_kwargs": {"n": 8},
        "gamma": 0.8, "sample_mode": "pointwise", "random_seed": 0,
        "model_kwargs": {"model_type": "linear"},
        "training_kwargs": {"batch_size": 4, "context_length": 8}, "eval_file": "eval.npz",
    })
    return Config(cfg)


def test_evaluate_scores_every_input_once_against_its_exact_row():
    # the loss is KL(model || true) averaged over the n distinct inputs, each paired with its own
    # conditional row: the Bayes-optimal logits score zero, a uniform model scores the mean KL
    # from uniform to the rows
    from src.code.models import LinearModel, ModelConfig
    config = _config()
    dataset = KernelDataset(config)
    ev = Evaluator(config, dataset)
    n = len(dataset.input_tuples())
    assert ev.inputs.shape[0] == n and ev.targets.shape == (n, dataset.num_dims)
    assert torch.allclose(ev.targets, dataset.P[dataset._row_index(dataset.input_tuples())].cpu())

    class _Bayes:
        output_key = "logits"
        def __call__(self, input_ids, output_hidden_states=False):
            return {"logits": ev.targets.log()}
    assert abs(Evaluator(config, dataset).evaluate(_Bayes())["loss"]) < 1e-6

    class _Uniform(_Bayes):
        def __call__(self, input_ids, output_hidden_states=False):
            return {"logits": torch.zeros(n, dataset.num_dims)}
    kl_uniform = (ev.targets * (ev.targets.clamp_min(1e-30).log() + np.log(dataset.num_dims))).sum(1).mean()
    assert abs(Evaluator(config, dataset).evaluate(_Uniform())["loss"] - kl_uniform.item()) < 1e-5


def test_evaluate_returns_the_loss_logits_representation_and_readout_as_arrays():
    # one flat dict: the KL, the logits over the inputs, the representation (last hidden state)
    # and the readout as numpy arrays, such that representation @ readout.T reproduces the logits
    from src.code.models import LinearModel, ModelConfig
    config = _config()
    dataset = KernelDataset(config)
    model = LinearModel(ModelConfig(vocab_size=dataset.vocab_size, num_classes=dataset.num_dims,
                                    embed_dim=3, arity=1))
    ev = Evaluator(config, dataset).evaluate(model.to("cuda" if torch.cuda.is_available() else "cpu"))
    n = len(dataset.input_tuples())
    assert set(ev) == {"loss", "logits", "representation", "readout"}
    assert all(isinstance(ev[k], np.ndarray) for k in ("logits", "representation", "readout"))
    assert ev["logits"].shape == (n, dataset.num_dims) and ev["representation"].shape == (n, 3)
    assert ev["readout"].shape == (dataset.num_dims, 3)
    assert np.allclose(ev["representation"] @ ev["readout"].T, ev["logits"], atol=1e-5)


def test_save_writes_one_npz_that_load_evaluation_reads_back(tmp_path):
    # the config's eval_file holds the arrays plus the step; an absent array (a readout that reads
    # no vector) is omitted on disk and comes back as None
    from src.code.analysis_utils import load_evaluation
    config = _config()
    dataset = KernelDataset(config)
    result = {"loss": 0.25, "logits": np.arange(20.0).reshape(5, 4), "representation": None, "readout": None}
    path = Evaluator(config, dataset).save(result, str(tmp_path), step=300)
    assert path == str(tmp_path / "eval.npz")
    back = load_evaluation(path)
    assert back["loss"] == 0.25 and back["step"] == 300
    assert np.array_equal(back["logits"], result["logits"])
    assert back["representation"] is None and back["readout"] is None


def test_collator_unwraps_the_prebatched_dict_and_drops_what_the_model_rejects():
    batch = {"input_ids": torch.zeros(2, 3), "labels": torch.zeros(2, 3),
             "distribution_labels": torch.zeros(2, 4)}
    assert PreBatchedDataCollator()([batch]) == batch
    lm = PreBatchedDataCollator(drop=("distribution_labels",))([batch])
    assert set(lm) == {"input_ids", "labels"}                  # a causal LM never sees the row


def test_hidden_input_evaluator_reads_observations_and_scores_the_posterior():
    # a stream task's inputs are its observation tokens (length-1 histories), whether the target is
    # the next observation or the hidden input; the latent tokens are outputs and never read
    cfg = OmegaConf.create({
        "setting": "graph-cooccurrence", "structure": "feature_cube",
        "structure_kwargs": {"m": 2, "paired": True, "all_bundles": True},
        "sample_mode": "sequence", "infer_latent": True, "random_seed": 0,
        "training_kwargs": {"batch_size": 4, "context_length": 8},
    })
    config = Config(cfg)
    dataset = KernelDataset(config)
    ev = Evaluator(config, dataset)
    assert torch.equal(ev.inputs[:, 1], torch.arange(dataset.n_outcomes))       # start, then one observation
    assert ev.targets.shape == (dataset.n_outcomes, dataset.n_items)
    post = torch.softmax(dataset.prior.log()[:, None] + dataset.P.double().log(), dim=0).T   # p(z | x)
    assert torch.allclose(ev.targets, post.float().cpu(), atol=1e-6)
