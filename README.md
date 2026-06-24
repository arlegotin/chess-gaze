# chess-gaze

Private Python project scaffold for future stream-analysis work.

The repository is intentionally initialized without product, chess, gaze, or
machine-learning runtime behavior. The current goal is to provide a clean local
Python workspace that future agents can extend with test-first changes.

## Setup

Use `uv` from the repository root:

```sh
uv sync
```

Run local gates:

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

Format code when needed:

```sh
uv run ruff format .
```

## Repository Shape

- `src/chess_gaze/` contains the importable Python package. It is currently
  metadata-only.
- `tests/` contains behavior tests.
- `docs/development/` contains canonical development guidance.
- `docs/superpowers/` contains active specs, implementation plans, and
  closeouts produced by Superpowers workflows.

Add runtime dependencies only when implementation code needs them.
