# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

# Research code philosophy

This is research code. Correctness, succinctness, and clarity take priority over robustness.

- **Fail loudly.** No try/except to paper over errors, no fallbacks, no default values that hide missing inputs. Let exceptions propagate — a crash with a clear traceback is more useful than silent misbehavior.
- **Minimize code.** Prefer principled abstractions over bloated ones. A clean abstraction that reduces duplication is good; a speculative one added for hypothetical future use is not.
- **No defensive validation** at internal boundaries. Trust that callers pass correct inputs; only validate at the outermost entry point (CLI args, config files).
- **Integration tests may load real models.** A test that runs an actual forward pass is more trustworthy than one that mocks the model.
- **Comments explain the current state of the code, not its chronology and development** (that's what git is for). No "(removed) …", "previously …", or "now uses …" notes.

# Development approach

Follow BDD dual-loop TDD. Every feature increment starts from a failing integration test and is driven inward through unit-level red-green-refactor cycles.

## Outer loop (integration)

1. **Red (integration)** — Write one integration/acceptance test describing the next observable behavior from the outside in. Run it. Confirm it fails for the reason you expect.
2. **Inner loop (unit)** — Repeat until the integration test can pass:
   - **Red** — Write the smallest unit test that expresses the next missing piece.
   - **Green** — Write the minimum production code to make that unit test pass. No speculative code.
   - **Refactor** — Clean up while all unit tests stay green.
3. **Green (integration)** — Re-run the integration test; drop back into the inner loop if it still fails.
4. **Refactor (integration)** — With everything green, refactor across module boundaries if needed.

## Discipline rules

- Never skip the red step. If you cannot articulate why a test fails, you do not yet understand the requirement.
- One logical change per cycle.
- Run only the relevant test after each green step, then the full suite before each checkpoint.
- If a refactor breaks a test, revert the refactor — do not fix forward.

# Environment

`uv`-managed. Use `uv run python -m ...` so commands resolve the project venv. Fast tests:
`uv run python -m pytest -m "not slow"`; full suite: `uv run python -m pytest`.

# Architecture

Hydra composes config from `conf/` and runs one **runner** (a top-level job: train or eval).

- `conf/config.yaml` — base defaults (setting + model + train + runner), wandb toggle, Hydra run-dir layout.
- `conf/{setting,model,train,runner}/` — composable groups. `runner/` files fully specify a job (which setting/model/train to use + any sweep params).
- `src/code/config.py` — `Config` wraps the resolved `DictConfig`; seeds all RNGs; turns `save_steps_method` into a concrete checkpoint schedule.
- `src/code/data.py` — `init_dataset` + dataset classes. Datasets return a full batch from `__getitem__` (`input_ids`, `labels`, eval-only `eval_labels`) and expose `compute_metrics`, `vocab_size`, `seq_len`.
- `src/code/models.py` — `init_model` (train from scratch) and `load_model` (checkpoint / pretrained LLM / Bayesian baseline); all return logits via `.output_key`.
- `src/code/train.py` — Hydra entry point. Sizes the model from the dataset, runs the HF `Trainer` with custom callbacks.
- `src/code/train_utils.py` — `PreBatchedDataCollator`, `StepMetricsCallback` (online eval → wandb), `CustomSaveCallback`.
- `src/code/eval.py` — `EvalComponents` (fixed eval batches, built once, deterministic) + `Evaluator` (build once; `evaluate(model) -> (metrics, extras)` per checkpoint/step). `Evaluator.from_cache` reuses a prior run's batches so online and standalone eval score identically.
- `src/code/analysis_utils.py` — `load_eval_results(..., source="local"|"wandb")` loads metrics into a tidy DataFrame from disk (offline, default) or from wandb; same schema either way. wandb is never assumed.

## Key design patterns

- **Build-once, evaluate-as-a-pure-function.** Eval setup (fixed batches) happens once; `evaluate(model)` is then a pure function of the model — reproducible across checkpoints.
- **Config-driven, no hardcoding.** All knobs live in `conf/`. Model I/O dims are derived from the dataset, not hardcoded.
- **wandb is optional.** `wandb.enabled=false` disables all logging (init + online metrics) for quick/debug runs.
- **Reproducible artifacts.** Each run writes `config_resolved.yaml`; eval writes `metrics.pt` (+ `eval_batches.pt` for the cache loop).
