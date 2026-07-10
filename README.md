# Template LM Training Codebase

A Hydra-based template for systematic language-model training and evaluation experiments. Designed for reproducible research: composable configs, deterministic evaluation, automatic run-directory management, and optional Weights & Biases tracking.

## Setup

The project is managed by [`uv`](https://docs.astral.sh/uv/). All dependencies are declared in `pyproject.toml` (locked in `uv.lock`).

```bash
uv sync                              # create / update .venv from uv.lock
export CACHE_DIR=/path/to/cache      # optional: where outputs land (defaults to ./outputs)
export WANDB_API_KEY=...             # optional: only if logging to wandb
```

Run everything as `uv run python -m ...` so it uses the project venv.

## Usage

The base config composes a **runner** (a top-level job). Pick one with `runner=...`.

### Train a small LM

```bash
# Single run
uv run python -m src.code.train runner=train_small_lm name=my_run

# Disable wandb (no tracking) — handy for quick / debug runs
uv run python -m src.code.train runner=train_small_lm wandb.enabled=false

# Parameter sweep (multirun)
uv run python -m src.code.train -m runner=train_small_lm model_kwargs.model_config.hidden_size=64,128,256
```

### Evaluate

```bash
# Evaluate every checkpoint of a trained run (same name/overrides as training)
uv run python -m src.code.eval runner=train_small_lm name=my_run

# Evaluate a pretrained LLM (single run)
uv run python -m src.code.eval runner=eval_llm model_kwargs.model_path=google/gemma-2-2b

# Evaluate the Bayesian baseline
uv run python -m src.code.eval runner=eval_bayes

# Re-evaluate, overwriting existing metrics
uv run python -m src.code.eval runner=eval_llm overwrite=true
```

Evaluation is **deterministic**: a fixed set of eval batches is sampled once (seeded), so every checkpoint and every training step is scored on the same batches. Pass `eval_cache_dir=<a prior run dir>` to reuse another run's exact batches.

## Weights & Biases

wandb is fully optional and controlled from config:

```yaml
wandb:
  enabled: true          # set false to disable all wandb logging
  entity: null           # null -> WANDB_ENTITY env / default entity
  project: template-lm-training
```

`wandb.enabled=false` skips `wandb.init` and all online metric logging, and sets the HF Trainer `report_to="none"`.

## Repository structure

```
conf/
  config.yaml              # base defaults, wandb toggle, Hydra run-dir layout
  setting/balls_and_urns.yaml   # task / dataset config
  model/{small_lm,llm,bayesian_model}.yaml
  train/default.yaml       # training_kwargs + eval_kwargs
  runner/                  # top-level jobs (fully specify a run)
    train_small_lm.yaml    # train a small LM from scratch
    eval_llm.yaml          # evaluate a pretrained LLM
    eval_bayes.yaml        # evaluate the Bayesian baseline
src/
  code/
    config.py              # Config wrapper: RNG seeding, checkpoint scheduling
    data.py                # init_dataset + dataset classes (return batched dicts)
    models.py              # init_model / load_model (small LM, LLM, Bayesian)
    train.py               # training entry point (HF Trainer + callbacks)
    train_utils.py         # data collator + online-eval / checkpoint callbacks
    eval.py                # EvalComponents + Evaluator + standalone eval entry
    analysis_utils.py      # load_eval_results -> DataFrame
    plot_utils.py          # publication-quality figure styling helpers
  notebooks/               # analysis notebooks
  tests/                   # pytest suite (-m slow for the end-to-end test)
```

## Output structure

```
$CACHE_DIR/{setting}/{name}/{override_dirname}/
  config_resolved.yaml     # full resolved config (reproducibility + eval cache match)
  eval_batches.pt          # fixed eval batches (for the eval cache loop)
  checkpoints/             # training runs only
    {step}/
      model/               # saved weights
      metrics.pt           # {metrics, extras, step}
```

## Adding components

- **New dataset**: add a class in `src/code/data.py` (return `input_ids` / `labels` / `eval_labels`, expose `compute_metrics`, `vocab_size`, `seq_len`); register it in `init_dataset`; add `conf/setting/<name>.yaml`.
- **New model**: add to `src/code/models.py` (init + load), register in `init_model`/`load_model`; add `conf/model/<name>.yaml`.
- **New runner**: add `conf/runner/<name>.yaml` selecting setting/model/train + any sweep params.

## Testing

```bash
uv run python -m pytest -m "not slow"   # fast unit tests
uv run python -m pytest                 # full suite (includes the end-to-end smoke test)
```

## Notebooks

Analysis notebooks in `src/notebooks/` are plain **percent-format `.py` files** ([jupytext](https://jupytext.readthedocs.io/)
/ VS Code `# %%` cells) — there is no `.ipynb` in the repo, so git stays free of notebook JSON and
output bloat. Run them interactively (VS Code Interactive Window, or open the `.py` as a notebook in
Jupyter Lab), or execute one headless:

```bash
uv run jupytext --to notebook --execute src/notebooks/<name>.py -o /tmp/out.ipynb   # run end-to-end; temp output
```

Keep notebooks thin: real logic belongs in `src/code/` (unit-tested); the notebook just calls those
functions and renders/saves figures.

**Figures** are written to the gitignored cache via `plot_utils.figure_path(config, name)`, keyed by
experiment (`config` is the run config returned by `train_inline`), so the repo stays binary-free —
promote a figure to git by hand only when it is paper-final:

```python
from src.code.plot_utils import save_figure, figure_path

save_figure(fig, figure_path(config, "cca_by_arch"))            # PNG (default, viewable)
# -> $CACHE_DIR/{setting}/figures/{override_dirname}/cca_by_arch.png
save_figure(fig, figure_path(config, "cca_by_arch", ext="pdf")) # vector, for paper-final
```

Figures are keyed by `{setting}/figures/{override_dirname}` (the per-model level of the run dir is
dropped), so all models trained on one structure share a figures dir and switching structures needs
no renaming.

## Analysis

wandb is optional, so analysis works from disk or from wandb — neither is the
hardwired path. `load_eval_results` returns the same wide schema either way: one
row per (run, step), a `config` column, and one column per flat metric.

```python
from src.code.analysis_utils import load_eval_results

# Offline (default): reads metrics.pt from $CACHE_DIR — no wandb needed.
df = load_eval_results(setting="balls-and-urns", experiment_name="my_run")

# Or pull the same shape from wandb.
df = load_eval_results("balls-and-urns", "my_run", source="wandb",
                       entity="my_entity", project="template-lm-training")

# columns: sweep params + 'step' + 'config' + flat metric columns (e.g. 'loss')
```
