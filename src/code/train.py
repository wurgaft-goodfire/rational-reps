import os
import shutil
from transformers import TrainingArguments, Trainer
import torch
import wandb
import hydra
from dotenv import load_dotenv
from pyprojroot import here
from omegaconf import DictConfig, OmegaConf
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

    # Derive model I/O dimensions from the dataset and patch the config so the
    # architecture can't drift out of sync with the data.
    config.cfg.model_kwargs.model_config.vocab_size = dataset.vocab_size
    config.cfg.model_kwargs.model_config.max_position_embeddings = dataset.seq_len
    print(f"Model config patched from dataset: vocab_size={dataset.vocab_size}, "
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
    trainer.add_callback(StepMetricsCallback(
        trainer=trainer, step_interval=config.training_kwargs["eval_steps"], config=config,
    ))

    trainer.train()

    if config.cfg.wandb.enabled:
        wandb.finish()


@hydra.main(version_base=None, config_path=str(here("conf")), config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training entry point.

    Usage:
      python -m src.code.train runner=train_small_lm
      python -m src.code.train -m runner=train_small_lm model_kwargs.model_config.hidden_size=64,128
      python -m src.code.train runner=train_small_lm wandb.enabled=false   # no wandb
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
