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
uv run python -m pytest -m "not slow"          # fast suite
uv run python -m pytest                        # full suite (trains small models end-to-end)
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

# A setting can declare its sweep under hydra.sweeper.params; one multirun per model class.
# The theory is a model class too: runner=theory solves each cell instead of training it.
uv run python -m src.code.train -m runner=linear setting=superposition
uv run python -m src.code.train -m runner=theory setting=superposition

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

Models (`conf/model/`): `transformer`, `rnn`, `mlp`, `linear`, `sae`, `llm` and `llm_70b` (pretrained Llama-3.1-8B / 70B).

## Output structure

A run lives at `{setting}/{task}/{overrides}/{name}`. The task is the structure or relation, a
directory of its own even when it came from the setting yaml rather than the command line (`graph`,
`feature_cube` and `hypercube` all set `setting: graph-cooccurrence`). The leaf is the override
string tidied by `clean_dirname`: bookkeeping and the `*_kwargs.` prefixes dropped, sorted, e.g.
`graph-cooccurrence/feature_cube/beta=0.5,embed_dim=3,p=0.3,random_seed=0,rho=0.0,train_steps=8000/linear`;
no overrides gives `default`. Figures shared by the models of one task sit beside them in
`{overrides}/figures/`.

```
$CACHE_DIR/{setting}/{task}/{overrides}/{name}/
  config_resolved.yaml     # full resolved config (reproducibility)
  history.csv              # the training curve
  checkpoints/
    {step}/
      model/               # saved weights (torch)
      eval.npz             # the saved evaluation (conf eval_file): step, loss, inputs, targets, logits, hidden_states, readout
```

## Notebooks

Analysis notebooks in `src/notebooks/` are [marimo](https://marimo.io/) notebooks: plain `.py`
files, no `.ipynb` in the repo. One notebook per setting (`hmm_analysis.py`,
`concepts_analysis.py`, …) plus cross-setting geometry studies. `llm_analysis.py` feeds graph walks
and relation triples to pretrained LLMs (`conf/model/llm*.yaml`) and compares their in-context
representations to the predicted codes; each LLM pass is cached as `node_means.npz` /
`pair_means.npz` in the run dir, so it needs a GPU only once (the 70B needs two).

```bash
uv run marimo edit src/notebooks/<name>.py     # interactive, in the browser
uv run python src/notebooks/<name>.py          # headless end-to-end run
uv run marimo check src/notebooks/*.py         # lint the notebook format
```

In VS Code, install the `marimo-team.vscode-marimo` extension and use **marimo: Open as marimo
notebook** (or the editor-title icon) to edit a file as a notebook; pick `.venv/bin/python` as the
kernel. Without the extension, run `marimo edit --headless` in a terminal and open the forwarded
port.

A cell reruns when a variable it reads changes, so position in the file decides nothing. Two cells
cannot assign the same variable; marimo rejects the notebook if they do. Give throwaways like
`fig`, `ax` and loop variables a leading underscore, which makes them private to their cell.

The setup cell resolves the project root and `chdir`s to it, so notebooks find `conf/` and `src/`
regardless of the working directory.

Notebooks read runs through `analysis_utils.load_run(overrides)`, which returns the run's config,
its dataset and the final checkpoint's saved evaluation as numpy arrays (`loss`, `logits`,
`hidden_states`, `readout`, plus the derived `representation`). It never trains or loads a model: a missing run raises with the
training command, and a run whose recorded resolved config or task kernel differs from what the
overrides now build raises too. Train from the CLI (above); `train.train_inline(overrides)` exists
for tests and for the one notebook that needs the live model.

**Figures** are written to the gitignored cache via `plot_utils.figure_path(config, name)`; promote
a figure to git by hand only when it is paper-final:

```python
from src.code.plot_utils import save_figure, figure_path

save_figure(fig, figure_path(config, "cca_by_arch"))            # PNG (default)
# -> $CACHE_DIR/{setting}/{task}/{overrides}/figures/cca_by_arch.png
save_figure(fig, figure_path(config, "cca_by_arch", ext="pdf")) # vector, paper-final
```

`save_figure` returns the figure, so ending a cell with the save call renders it in marimo.

Figures are keyed by `{setting}/{task}/{overrides}/figures` (the per-model level is dropped), so
all models trained on one structure share a figures dir. A notebook that trains nothing gets the
same path from `train.resolve_config(overrides)`, which builds the config and its `run_dir` without
running a job.

## Analysis

Three layers, each doing one thing:

- `eval.Evaluator.evaluate(model)` is the only forward pass analysis depends on: one batch, every
  input once for a static task or a fixed seeded batch of streams for a sequence task, scored
  against the exact Bayes targets. It returns the pass whole and untransformed: `loss`, `inputs`,
  `targets` (target t pairs with output t), `logits` over the outcome tokens, `hidden_states` for
  every layer at every position, and the `readout` rows. `Evaluator.save` writes them to `eval.npz`.
- `analysis_utils.load_run(overrides)` reads that file back and adds `representation`, the last
  layer at the positions the readout reads. By default it expresses the width-dimensional arrays in
  the readout's row space (`readout_basis`), the directions the loss constrains; `project=False`
  gives the raw arrays. No model is constructed.
- Everything else is an explicit function a notebook applies and names in the cell above:
  `feature_vectors` (least-squares feature directions), `leverage`, gauge fixes in `geometry.py`,
  and the comparison metrics `cka`, `cca`, `rsa` with their nulls.

wandb is optional throughout: `wandb.enabled=false` (the default) skips `wandb.init` and all
online logging.

## Architecture

Hydra composes config from `conf/` and runs one runner (train or eval).

- `conf/config.yaml` — base defaults (setting + model + train + runner), wandb toggle, run-dir layout.
- `src/code/config.py` — `Config` wraps the resolved `DictConfig`; seeds all RNGs; turns `save_steps_method` into a concrete checkpoint schedule.
- `src/code/data.py` — `init_dataset` + three dataset classes (co-occurrence, relational, HMM), plus the structure gallery (graphs and co-occurrence kernels). Datasets return a full batch from `__getitem__` (`input_ids`, `labels`, eval-only `eval_labels`) and expose `compute_metrics`, `vocab_size`, `seq_len`.
- `src/code/geometry.py` — the rational-code oracles: the spectral factorization `factor_logcond` (row-centred, balanced split, exposed as `predicted_geometry`), floored logs, `class_means`, and the superposition theory (`allocate_d`, `fantope_geometry`, `tight_frame`, `repair`, composed in `predicted_rank_constrained_geometry`, from `notes/superposition.tex`).
- `src/code/models.py` — `init_model` (train from scratch) and `load_model` (checkpoint / pretrained LLM); all return logits via `.output_key`.
- `src/code/train.py` — Hydra entry point + `train_inline` for tests. `train` sizes the model from the dataset and runs the HF `Trainer` with custom callbacks; `solve` fills a run dir from the theory instead (`runner=theory`). Both write the same checkpoint, history and evaluation files.
- `src/code/train_utils.py` — `PreBatchedDataCollator`, `StepMetricsCallback` (online eval), `CustomSaveCallback`.
- `src/code/llm.py` — a pretrained LM on in-context streams: items as single-token nouns, `node_means` (walks `word, word, ...`, read at the separator) and `pair_means` (relation triples `a b : c ,`, read at the colon), each returning per-layer means per key plus the NLL of the next token; `triple_stream` builds relation streams from the dataset, `bayes_floor` / `relation_floor` the exact entropies.
- `src/code/eval.py` — `Evaluator`: `evaluate(model)` scores every input against its exact target and returns the loss, logits, representation and readout as numpy; `save` writes them to a checkpoint's `eval.npz`. Also the standalone eval entry point.
- `src/code/analysis_utils.py` — the array side: `load_run` / `load_evaluation` (with `readout_basis` and `representation`), explicit transforms (`feature_vectors`, `leverage`, `logit_map`), and the similarity metrics (`cka`, `cca`, `rsa`, nulls).
- `src/code/plot_utils.py` — figure styling, `figure_path`, `save_figure`.
- `notes/` — derivations and setting write-ups.

## Adding components

- **New setting**: add a dataset class in `src/code/data.py` (return `input_ids` / `labels` / `eval_labels`; expose `compute_metrics`, `vocab_size`, `seq_len`); register it in `init_dataset`; add `conf/setting/<name>.yaml`; add its geometry oracle to `src/code/geometry.py`.
- **New model**: add to `src/code/models.py`, register in `init_model`/`load_model`; add `conf/model/<name>.yaml`.
- **New runner**: add `conf/runner/<name>.yaml` selecting setting/model/train + any fixed overrides.

## Repo invariants

- **Evaluate as a pure function.** Every model is scored on every input the task defines against its exact outcome distribution, so `evaluate(model)` is a pure function of the model, reproducible across checkpoints and architectures.
- **Config-driven, no hardcoding.** All knobs live in `conf/`. Derive model I/O dims from the dataset; never hardcode them.
- **Reproducible artifacts.** Each run writes `config_resolved.yaml` (the resolved config plus the task's kernel hash) and, at the end of training, `eval.npz` with the model's loss, logits, representation and readout over every input — so analysis reads a run without constructing the model (`analysis_utils.load_run`) and refuses one whose config or task changed under it.
