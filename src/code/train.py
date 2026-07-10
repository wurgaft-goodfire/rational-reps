import os
import shutil
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
from .models import init_model
from .config import Config
from .data import init_dataset
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


def train(config):
    print_current_gpus()
    wandb_init(config)

    # Initialize the dataset first so we can size the model from it.
    dataset = init_dataset(config)

    # Derive all data-determined model dimensions from the dataset and inject them, so configs
    # never hardcode vocab_size / num_classes / arity / max_position_embeddings.
    mc = config.cfg.model_kwargs.model_config
    with open_dict(mc):
        mc.vocab_size = dataset.vocab_size
        if config.cfg.model_kwargs.model_type in ("klinear", "mlp", "rnn"):
            mc.num_classes = dataset.num_dims          # readout heads: outcome classes + arity
            mc.arity = dataset.arity
            print(f"Model dims from dataset: vocab_size={dataset.vocab_size}, "
                  f"num_classes={dataset.num_dims}, arity={dataset.arity}")
        else:
            mc.max_position_embeddings = dataset.seq_len
            print(f"Model dims from dataset: vocab_size={dataset.vocab_size}, "
                  f"max_position_embeddings={dataset.seq_len}")

    model = init_model(config)

    # Pre-batched data collator: the dataset returns a full batch per __getitem__.
    data_collator = PreBatchedDataCollator()
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

    # Return the trained artifacts so callers (notebooks) can use them in-process; the CLI
    # entry point ignores the return value.
    return {
        "model": trainer.model,
        "dataset": dataset,
        "evaluator": metrics_callback.evaluator,
        "history": metrics_callback.history,
    }


def train_inline(overrides, name=None, cache_dir=None):
    """Train in-process from a list of Hydra overrides (for notebooks). Composes the same config
    the CLI builds — all conf/ groups reused — then runs `train()`. Returns
    ``(model, dataset, evaluator, history_df)`` with one history row per eval step. Real sweeps
    still go through `python -m src.code.train -m ...`.

        model, dataset, evaluator, history = train_inline(["runner=transformer", "setting=graph"])
    """
    overrides = list(overrides) + ([f"name={name}"] if name is not None else [])
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(here() / "conf")):
        cfg = compose(config_name="config", overrides=overrides)
    cache_dir = cache_dir or os.getenv("CACHE_DIR") or str(here() / "outputs")
    cfg.cache_dir = cache_dir
    # run_dir's ${hydra:runtime.output_dir} interpolation only resolves under @hydra.main, so set it.
    cfg.run_dir = os.path.join(cache_dir, cfg.setting, cfg.name, "notebook")
    result = train(Config(cfg, make_run_dirs=True))
    return result["model"], result["dataset"], result["evaluator"], pd.DataFrame(result["history"])


def graph_overrides(spec):
    """Hydra overrides selecting a graph structure from a spec dict, e.g.
    ``{"structure": "grid", "n": 6, "m": 6}`` or a composite
    ``{"structure": "product", "factors": [{"structure": "ring", "n": 9}, ...]}``. Replaces the
    setting's default ``structure_kwargs`` (deletes it, then re-adds this structure's params) so
    switching structures needs no per-key dance: ``train_inline(graph_overrides(spec) + ["runner=klinear"])``."""
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
    """Main training entry point.

    Usage:
      python -m src.code.train runner=transformer setting=graph
      python -m src.code.train -m runner=klinear setting=graph structure=ring,chain
      python -m src.code.train runner=rnn setting=distribution
    """
    config = Config(cfg, make_run_dirs=True)
    config.to_yaml(os.path.join(config.run_dir, "config_resolved.yaml"))

    if os.path.exists(config.output_dir) and len(os.listdir(config.output_dir)) > 0:
        if not cfg.overwrite:
            print(f"Output directory {config.output_dir} is not empty. Set overwrite=true to overwrite.")
            return
        print(f"overwrite=true: cleaning output directory {config.output_dir}")
        shutil.rmtree(config.output_dir)
        os.makedirs(config.output_dir, exist_ok=True)

    train(config)
    print("DONE")


if __name__ == "__main__":
    main()
