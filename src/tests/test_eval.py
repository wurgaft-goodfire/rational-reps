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


def test_evaluate_returns_the_forward_pass_on_every_input_as_arrays():
    # a static task: one flat dict with the KL, the inputs and their exact rows, the logits, every
    # layer's hidden state at the one position, and the readout; the last layer times the readout
    # reproduces the logits
    from src.code.models import LinearModel, ModelConfig
    config = _config()
    dataset = KernelDataset(config)
    model = LinearModel(ModelConfig(vocab_size=dataset.vocab_size, num_classes=dataset.num_dims,
                                    embed_dim=3, arity=1))
    ev = Evaluator(config, dataset).evaluate(model.to("cuda" if torch.cuda.is_available() else "cpu"))
    n, C = len(dataset.input_tuples()), dataset.num_dims
    assert set(ev) == {"loss", "inputs", "targets", "logits", "hidden_states", "readout"}
    assert all(isinstance(ev[k], np.ndarray) for k in set(ev) - {"loss"})
    assert ev["inputs"].shape == (n, 1) and ev["targets"].shape == (n, C) and ev["logits"].shape == (n, C)
    assert ev["hidden_states"].shape == (1, n, 1, 3) and ev["readout"].shape == (C, 3)
    assert np.allclose(ev["hidden_states"][-1][:, 0] @ ev["readout"].T, ev["logits"], atol=1e-5)


def test_sequence_tasks_are_scored_on_one_fixed_batch_of_streams_without_moving_the_training_rng():
    # a sequence task: batch_size streams drawn from a private generator, so two evaluators see the
    # same streams and the dataset's training generator is where it was; targets are exact and
    # aligned so target t pairs with output t; the forward pass is saved whole
    from src.code.models import init_model
    cfg = OmegaConf.create({
        "setting": "graph-cooccurrence", "structure": "ring", "structure_kwargs": {"n": 8},
        "gamma": 0.8, "sample_mode": "sequence", "random_seed": 3, "eval_file": "eval.npz",
        "model_kwargs": {"model_type": "rnn", "model_config": {"embed_dim": 4, "hidden_size": 6,
                                                                "num_layers": 1, "init_scale": 0.3}},
        "training_kwargs": {"batch_size": 4, "context_length": 6},
    })
    config = Config(cfg)
    dataset = KernelDataset(config)
    state = dataset.rng.get_state().clone()
    ev1, ev2 = Evaluator(config, dataset), Evaluator(config, dataset)
    assert torch.equal(dataset.rng.get_state(), state)                 # training stream untouched
    assert torch.equal(ev1.inputs, ev2.inputs) and ev1.inputs.shape == (4, 7)   # start + 6 observations
    assert ev1.targets.shape == (4, 6, dataset.num_dims)
    assert torch.allclose(ev1.targets, dataset.optimal_predictions(ev1.inputs.to(dataset.device)).float().cpu())
    mc = config.cfg.model_kwargs.model_config
    mc.vocab_size, mc.num_classes, mc.arity = dataset.vocab_size, dataset.num_dims, dataset.arity
    out = ev1.evaluate(init_model(config))
    assert out["inputs"].shape == (4, 7) and out["targets"].shape == (4, 6, dataset.num_dims)
    assert out["logits"].shape == (4, 7, dataset.num_dims)             # every position, outcome tokens only
    assert out["hidden_states"].shape[1:] == (4, 7, 6) and out["readout"].shape == (dataset.num_dims, 6)
    assert np.isfinite(out["loss"])


def test_save_writes_one_npz_that_load_evaluation_reads_back(tmp_path):
    # the config's eval_file holds the arrays plus the step; absent arrays (a readout that reads no
    # vector) are omitted on disk and come back as None, and the representation is derived
    from src.code.analysis_utils import load_evaluation
    config = _config()
    dataset = KernelDataset(config)
    result = {"loss": 0.25, "inputs": np.arange(5)[:, None], "targets": np.ones((5, 4)) / 4,
              "logits": np.arange(20.0).reshape(5, 4), "hidden_states": None, "readout": None}
    path = Evaluator(config, dataset).save(result, str(tmp_path), step=300)
    assert path == str(tmp_path / "eval.npz")
    back = load_evaluation(path)
    assert back["loss"] == 0.25 and back["step"] == 300
    assert np.array_equal(back["logits"], result["logits"]) and np.array_equal(back["inputs"], result["inputs"])
    assert back["hidden_states"] is None and back["readout"] is None and back["representation"] is None


def test_collator_unwraps_the_prebatched_dict_and_drops_what_the_model_rejects():
    batch = {"input_ids": torch.zeros(2, 3), "labels": torch.zeros(2, 3),
             "distribution_labels": torch.zeros(2, 4)}
    assert PreBatchedDataCollator()([batch]) == batch
    lm = PreBatchedDataCollator(drop=("distribution_labels",))([batch])
    assert set(lm) == {"input_ids", "labels"}                  # a causal LM never sees the row


def test_hidden_input_evaluator_scores_sampled_streams_against_the_exact_posterior():
    # a stream task with a hidden input: the evaluator draws batch_size streams of context_length
    # observations (start token first) and pairs output t with the exact posterior over z after
    # the first t observations, prior times accumulated likelihood, normalized; the latent tokens
    # are outputs and never read
    cfg = OmegaConf.create({
        "setting": "graph-cooccurrence", "structure": "feature_cube",
        "structure_kwargs": {"m": 2, "paired": True, "all_bundles": True},
        "sample_mode": "sequence", "infer_latent": True, "random_seed": 0,
        "training_kwargs": {"batch_size": 4, "context_length": 8},
    })
    config = Config(cfg)
    dataset = KernelDataset(config)
    ev = Evaluator(config, dataset)
    B, L = 4, 8
    assert ev.inputs.shape == (B, L + 1) and (ev.inputs[:, 0] == dataset.start_token).all()
    assert (ev.inputs[:, 1:] < dataset.n_outcomes).all()                          # observations only
    assert ev.targets.shape == (B, L, dataset.n_items)
    logp, lp0 = dataset.P.double().log().cpu(), dataset.prior.double().log().cpu()
    for b in range(B):
        for t in range(L):
            post = torch.softmax(lp0 + logp[:, ev.inputs[b, 1:1 + t]].sum(1), dim=0)   # after t observations
            assert torch.allclose(ev.targets[b, t], post.float(), atol=1e-6), (b, t)
