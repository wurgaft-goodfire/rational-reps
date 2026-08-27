import hashlib
import os
import shutil
import numpy as np
import pandas as pd
from transformers import TrainingArguments, Trainer
import torch
import wandb
import hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from dotenv import load_dotenv
from pyprojroot import here
from omegaconf import DictConfig, OmegaConf, open_dict
from .models import init_model, load_model, MODEL_CLASSES
from .config import Config, clean_dirname
from .data import init_dataset
from .eval import Evaluator
from .train_utils import (
    CustomSaveCallback,
    StepMetricsCallback,
    PreBatchedDataCollator,
)

# load env variables
load_dotenv()


def print_current_gpus():
    if torch.cuda.is_available():
        print("Current GPUs:")
        for i in range(torch.cuda.device_count()):
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No GPUs available.")


def wandb_init(config):
    """Initialize wandb if enabled; otherwise a no-op (see wandb.enabled)."""
    if not config.cfg.wandb.enabled:
        print("wandb disabled (wandb.enabled=false); skipping wandb.init.")
        return
    run_name = f"{config.name}-{config.run_dir.split('/')[-1]}"
    wandb.init(
        entity=config.cfg.wandb.entity,
        project=config.cfg.wandb.project,
        name=run_name,
        config=OmegaConf.to_container(config.cfg, resolve=True),
        save_code=False,  # Don't save code to wandb
    )
    # Ensure wandb doesn't watch the model (prevents weight logging)
    wandb.watch(models=[], log=None)


def train(config, dataset):
    print_current_gpus()
    wandb_init(config)

    # Derive all data-determined model dimensions from the dataset and inject them, so configs
    # never hardcode vocab_size / num_classes / arity / max_position_embeddings.
    mc = config.cfg.model_kwargs.model_config
    with open_dict(mc):
        mc.vocab_size = dataset.vocab_size
        mc.num_classes = dataset.num_dims               # outcome classes (unused by the LM configs)
        mc.arity = dataset.arity
        mc.max_position_embeddings = dataset.seq_len    # unused by the readout configs
        if config.cfg.model_kwargs.model_type == "sae":
            mc.embed_dim = dataset.features.shape[1]    # the SAE reads feature codes, not ids
    print(f"Model dims from dataset: vocab_size={dataset.vocab_size}, "
          f"num_classes={dataset.num_dims}, arity={dataset.arity}, seq_len={dataset.seq_len}")

    model = init_model(config)
    if hasattr(model, "load_features"):
        model.load_features(dataset.features)

    # Pre-batched data collator: the dataset returns a full batch per __getitem__. The readouts
    # train on distribution_labels; the Trainer filters batch keys per model signature.
    data_collator = PreBatchedDataCollator(drop=())
    train_batch_size = 1  # batch is built in the dataset, so the loader batch size is 1

    config.process_save_steps()

    training_args = TrainingArguments(
        optim=config.training_kwargs["optimizer"],
        output_dir=config.output_dir,
        per_device_train_batch_size=train_batch_size,
        logging_steps=config.training_kwargs["logging_steps"],
        max_steps=config.training_kwargs["train_steps"],
        gradient_accumulation_steps=config.training_kwargs["gradient_accumulation_steps"],
        seed=config.random_seed,
        weight_decay=config.training_kwargs["weight_decay"],
        max_grad_norm=config.training_kwargs["max_grad_norm"],
        learning_rate=config.training_kwargs["learning_rate"],
        warmup_steps=config.training_kwargs["warmup_steps"],
        lr_scheduler_type=config.training_kwargs["lr_scheduler_type"],
        lr_scheduler_kwargs=config.training_kwargs["lr_scheduler_kwargs"],
        save_strategy="no",  # checkpointing handled by CustomSaveCallback
        report_to="wandb" if config.cfg.wandb.enabled else "none",
        save_total_limit=0,
        logging_dir=None,  # disable tensorboard logging
        dataloader_pin_memory=False,  # avoid pinning the pre-batched GPU tensors
        torch_compile=config.training_kwargs["compile"],
        torch_compile_mode=config.training_kwargs["compile_mode"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.add_callback(CustomSaveCallback(config.training_kwargs["save_steps"], trainer=trainer))
    metrics_callback = StepMetricsCallback(
        trainer=trainer, step_interval=config.training_kwargs["eval_steps"], config=config,
    )
    trainer.add_callback(metrics_callback)

    trainer.train()

    if config.cfg.wandb.enabled:
        wandb.finish()

    # Persist the training curve so a later reuse (train_inline load path) returns the same history
    # even when only the final checkpoint was saved.
    history = pd.DataFrame(metrics_callback.history)
    history.to_csv(os.path.join(config.run_dir, "history.csv"), index=False)

    # Return the trained artifacts so callers (notebooks) can use them in-process; the CLI
    # entry point ignores the return value.
    return {
        "model": trainer.model,
        "dataset": dataset,
        "evaluator": metrics_callback.evaluator,
        "history": history,
    }


def _run_key(cfg):
    """The config content that identifies a run: everything except machine-local bookkeeping."""
    c = OmegaConf.to_container(cfg, resolve=True)
    for k in ("cache_dir", "run_dir", "output_dir", "data_path", "eval_cache_dir", "overwrite", "wandb"):
        c.pop(k, None)
    # Online eval scores fixed pre-sampled batches and draws no RNG, so its cadence changes the
    # density of the history curve but never the trained weights.
    c.get("training_kwargs", {}).pop("eval_steps", None)
    return c


def _target_digest(dataset):
    """Fingerprint of the task tensor a run trains on. The run dir is keyed by the override string,
    so a code change to a builder moves the target while leaving the config identical; this catches
    that. Shapes and values both enter the hash."""
    T = np.ascontiguousarray(np.asarray(dataset.conditional, dtype=np.float64))
    return hashlib.sha1(repr(T.shape).encode() + T.tobytes()).hexdigest()[:16]


def _run_exists(config, dataset):
    """Write config_resolved and report whether a completed run of the *same resolved config and
    task tensor* is already on disk; a stale or ``overwrite=true`` run is wiped so training
    re-runs."""
    resolved_path = os.path.join(config.run_dir, "config_resolved.yaml")
    with open_dict(config.cfg):
        config.cfg.target_digest = _target_digest(dataset)
    stale = os.path.exists(resolved_path) and _run_key(OmegaConf.load(resolved_path)) != _run_key(config.cfg)
    config.to_yaml(resolved_path)
    if os.path.exists(config.output_dir) and len(os.listdir(config.output_dir)) > 0:
        if not config.cfg.overwrite and not stale:
            return True
        reason = "overwrite=true" if config.cfg.overwrite else "resolved config or task changed"
        print(f"{reason}: cleaning output directory {config.output_dir}")
        shutil.rmtree(config.output_dir)
        os.makedirs(config.output_dir, exist_ok=True)
    return False


def _load_run(config, dataset, device=None):
    """Load a finished run's final checkpoint + rebuild its evaluator (on the run's cached eval
    batches) and history — the reuse counterpart to `train()`, returning the same artifacts."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = Evaluator.from_cache(config, dataset, config.run_dir)
    last = max(int(d) for d in os.listdir(config.output_dir) if d.isdigit())
    model, _ = load_model(config, os.path.join(config.output_dir, str(last), "model"), device=device)
    history = pd.read_csv(os.path.join(config.run_dir, "history.csv"))
    return model, dataset, evaluator, history


def resolve_config(overrides, name=None, cache_dir=None, make_run_dirs=False):
    """Compose the config for a list of Hydra overrides and key its run dir exactly as the CLI does:
    ``{cache_dir}/{setting}/{clean_dirname}/{name}``. The path half of ``train_inline``, so a notebook
    that trains nothing still gets a config-keyed ``figure_path``."""
    overrides = list(overrides) + ([f"name={name}"] if name is not None else [])
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(here() / "conf")):
        cfg = compose(config_name="config", overrides=overrides, return_hydra_config=True)
    # override_dirname is only computed under Hydra; grab it, then drop the hydra node so the config
    # resolves/saves cleanly.
    override_dirname = clean_dirname(cfg.hydra.job.override_dirname, cfg.get("structure"), cfg.get("relation"))
    with open_dict(cfg):
        del cfg["hydra"]
    cache_dir = cache_dir or os.getenv("CACHE_DIR") or str(here() / "outputs")
    cfg.cache_dir = cache_dir
    cfg.run_dir = os.path.join(cache_dir, cfg.setting, override_dirname, cfg.name)
    return Config(cfg, make_run_dirs=make_run_dirs)


def train_inline(overrides, name=None, cache_dir=None):
    """Train in-process from a list of Hydra overrides (for notebooks) — the same job, run dir, and
    save schedule as the CLI. Returns ``(model, dataset, evaluator, history_df, config)``. If that
    exact run already exists on disk (and ``overwrite`` is not set), loads it instead of retraining."""
    config = resolve_config(overrides, name=name, cache_dir=cache_dir, make_run_dirs=True)
    dataset = init_dataset(config)
    if _run_exists(config, dataset):
        return (*_load_run(config, dataset), config)
    result = train(config, dataset)
    return result["model"], result["dataset"], result["evaluator"], result["history"], config


def graph_overrides(spec):
    """Hydra overrides selecting a graph structure from a spec dict, e.g.
    ``{"structure": "grid", "n": 6, "m": 6}``. Everything past the structure name goes into
    ``structure_kwargs``, replacing the setting's default (deletes it, then re-adds)."""
    def literal(v):                                           # nested list/dict -> Hydra override literal
        if isinstance(v, dict):
            return "{" + ",".join(f"{k}:{literal(x)}" for k, x in v.items()) + "}"
        if isinstance(v, (list, tuple)):
            return "[" + ",".join(literal(x) for x in v) + "]"
        return str(v)
    ov = ["setting=graph", f"structure={spec['structure']}", "~structure_kwargs"]
    return ov + [f"+structure_kwargs.{k}={literal(v)}" for k, v in spec.items() if k != "structure"]


@hydra.main(version_base=None, config_path=str(here("conf")), config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training entry point: ``python -m src.code.train runner=transformer setting=graph``
    (``-m`` for sweeps)."""
    config = Config(cfg, make_run_dirs=True)
    dataset = init_dataset(config)
    if _run_exists(config, dataset):
        print(f"Output directory {config.output_dir} is not empty. Set overwrite=true to overwrite.")
        return
    train(config, dataset)
    print("DONE")


if __name__ == "__main__":
    main()
