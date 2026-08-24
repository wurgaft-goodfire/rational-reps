# rational-reps

Toy settings where the Bayes-optimal ("rational") representation is computable in closed form.
Each setting is a data-generating process with a known geometry oracle (`src/code/geometry.py`);
we train small networks on it from scratch and compare their learned representations to the
rational code. Hydra composes the configs; evaluation is deterministic; analysis lives in thin
notebooks.

## Setup

The project is managed by [`uv`](https://docs.astral.sh/uv/). Dependencies are declared in
`pyproject.toml` (locked in `uv.lock`).

```bash
uv sync                              # create / update .venv from uv.lock
export CACHE_DIR=/path/to/cache      # optional: where outputs land (defaults to ./outputs)
export WANDB_API_KEY=...             # optional: only if logging to wandb
```

Run everything as `uv run python -m ...` so it uses the project venv.

## Tests

```bash
uv run python -m pytest -m "not slow"   # fast suite
uv run python -m pytest                 # full suite (trains small models end-to-end)
```

## Usage

A **runner** (`conf/runner/`) fully specifies a job: setting + model + train defaults. Pick one
with `runner=...`; override any config key on the CLI.

```bash
# Train
uv run python -m src.code.train runner=linear setting=word_attribute
uv run python -m src.code.train runner=hmm_transformer structure=grid
uv run python -m src.code.train runner=hmm_transformer

# Sweep (Hydra runs in multirun mode by default)
uv run python -m src.code.train runner=transformer model_kwargs.model_config.hidden_size=16,64,256

# Evaluate every checkpoint of a trained run (same runner + overrides as training)
uv run python -m src.code.eval runner=hmm_transformer

# Re-evaluate, overwriting existing metrics
uv run python -m src.code.eval runner=hmm_transformer overwrite=true
```

Evaluation is **deterministic**: a fixed set of eval batches is sampled once (seeded), so every
checkpoint and every training step is scored on the same batches. Pass
`eval_cache_dir=<a prior run dir>` to reuse another run's exact batches.

## Settings

Each `conf/setting/*.yaml` defines a data-generating process; its header comment states the task
and the predicted geometry.

| Setting | Task | Rational code |
|---|---|---|
| `graph` | graph → diffusion co-occurrence → next token | structure harmonics (ring → circle) |
| `hypercube` | binary attribute vectors, heat-kernel co-occurrence | additive per-bit code → linear analogies |
| `relational` | order-2 operation `c = a*b` | bilinear tensor factors |
| `word_attribute` | N words with Bernoulli(p) attribute bundles → m attribute tokens | packed (superposed) attribute directions |
| `hmm` | latent walk on a graph, emitted tokens | belief state of the forward algorithm |

Models (`conf/model/`): `transformer`, `rnn`, `mlp`, `linear`, `sae`, `llm` (pretrained).

## Output structure

```
$CACHE_DIR/{setting}/{override_dirname}/{name}/
  config_resolved.yaml     # full resolved config (reproducibility + eval cache match)
  eval_batches.pt          # fixed eval batches (for the eval cache loop)
  checkpoints/
    {step}/
      model/               # saved weights
      metrics.pt           # {metrics, extras, step}
```

## Notebooks

Analysis notebooks in `src/notebooks/` are percent-format `.py` files
([jupytext](https://jupytext.readthedocs.io/) / VS Code `# %%` cells) — no `.ipynb` in the repo.
One notebook per setting (`word_attribute_analysis.py`, `hmm_analysis.py`, …) plus cross-setting
geometry studies. Run them interactively, or headless:

```bash
uv run jupytext --to notebook --execute src/notebooks/<name>.py -o /tmp/out.ipynb
```

The setup cell resolves the project root and `chdir`s to it, so notebooks find `conf/` and `src/`
regardless of the kernel's working directory.

Notebooks train through `train.train_inline(overrides)`, which returns
`(model, dataset, evaluator, history_df, config)` and transparently reuses a cached run when one
exists with the same config.

**Figures** are written to the gitignored cache via `plot_utils.figure_path(config, name)`; promote
a figure to git by hand only when it is paper-final:

```python
from src.code.plot_utils import save_figure, figure_path

save_figure(fig, figure_path(config, "cca_by_arch"))            # PNG (default)
# -> $CACHE_DIR/{setting}/figures/{override_dirname}/cca_by_arch.png
save_figure(fig, figure_path(config, "cca_by_arch", ext="pdf")) # vector, paper-final
```

Figures are keyed by `{setting}/figures/{override_dirname}` (the per-model level is dropped), so
all models trained on one structure share a figures dir.

## Analysis

`load_eval_results` returns the same wide schema from disk (default) or wandb: one row per
(run, step), a `config` column, and one column per flat metric.

```python
from src.code.analysis_utils import load_eval_results

df = load_eval_results(setting="graph-cooccurrence", experiment_name="word_attribute")
```

wandb is optional throughout: `wandb.enabled=false` (the default) skips `wandb.init` and all
online logging.

## Architecture

Hydra composes config from `conf/` and runs one runner (train or eval).

- `conf/config.yaml` — base defaults (setting + model + train + runner), wandb toggle, run-dir layout.
- `src/code/config.py` — `Config` wraps the resolved `DictConfig`; seeds all RNGs; turns `save_steps_method` into a concrete checkpoint schedule.
- `src/code/data.py` — `init_dataset` + three dataset classes (co-occurrence, relational, HMM), plus the structure gallery (graphs and co-occurrence kernels). Datasets return a full batch from `__getitem__` (`input_ids`, `labels`, eval-only `eval_labels`) and expose `compute_metrics`, `vocab_size`, `seq_len`.
- `src/code/geometry.py` — the rational-code oracles: spectral factorizations (`factor_logcond`, `truncation_features`), the nuclear-norm feature oracle (`nuclear_features`), and representation-comparison utilities (principal angles, superposition measures).
- `src/code/models.py` — `init_model` (train from scratch) and `load_model` (checkpoint / pretrained LLM); all return logits via `.output_key`.
- `src/code/train.py` — Hydra entry point + `train_inline` for notebooks. Sizes the model from the dataset, runs the HF `Trainer` with custom callbacks.
- `src/code/train_utils.py` — `PreBatchedDataCollator`, `StepMetricsCallback` (online eval), `CustomSaveCallback`.
- `src/code/eval.py` — `EvalComponents` (fixed eval batches, built once) + `Evaluator` (`evaluate(model) -> (metrics, extras)`). `Evaluator.from_cache` reuses a prior run's batches so online and standalone eval score identically.
- `src/code/analysis_utils.py` — `load_eval_results(..., source="local"|"wandb")`.
- `src/code/plot_utils.py` — figure styling, `figure_path`, `save_figure`.
- `notes/` — derivations and setting write-ups.

## Adding components

- **New setting**: add a dataset class in `src/code/data.py` (return `input_ids` / `labels` / `eval_labels`; expose `compute_metrics`, `vocab_size`, `seq_len`); register it in `init_dataset`; add `conf/setting/<name>.yaml`; add its geometry oracle to `src/code/geometry.py`.
- **New model**: add to `src/code/models.py`, register in `init_model`/`load_model`; add `conf/model/<name>.yaml`.
- **New runner**: add `conf/runner/<name>.yaml` selecting setting/model/train + any fixed overrides.

## Repo invariants

- **Build once, evaluate as a pure function.** Eval setup (fixed batches) happens once; `evaluate(model)` is then a pure function of the model, reproducible across checkpoints.
- **Config-driven, no hardcoding.** All knobs live in `conf/`. Derive model I/O dims from the dataset; never hardcode them.
- **Reproducible artifacts.** Each run writes `config_resolved.yaml`; eval writes `metrics.pt` (+ `eval_batches.pt` for the cache loop).
