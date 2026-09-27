# Project specifics (general guidance lives in ~/.claude/CLAUDE.md)

- Run tests via `uv run python -m pytest` (`-m "not slow"` for the fast suite); all other commands: see README.
- Save notebook figures to the gitignored cache with `save_figure`, which returns the figure; end a plotting cell with that call so marimo renders it.
