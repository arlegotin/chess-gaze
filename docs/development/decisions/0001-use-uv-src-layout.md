# ADR-0001: Use uv and a Minimal src Layout

Date: 2026-06-24

## Status

Accepted

## Context

The repository is a private Python project that will later contain runtime
code, but it should not contain chess, gaze, or machine-learning behavior yet.
The user requested an isolated Python workflow, explicitly preferred `uv` or a
similar tool, and ruled out CI/CD or other complex infrastructure at this
stage.

The project still needs to be importable and testable as a package so future
agents can work against stable local gates.

## Decision

Use `uv` for dependency resolution, locking, environment management, and command
execution. Use a packaged `src/chess_gaze/` layout with a metadata-only package
stub, pytest tests, Ruff linting/formatting, and mypy type checking.

Do not add runtime dependencies, ML dependencies, notebooks, task runners,
pre-commit hooks, Docker, or CI/CD until a concrete feature requires them.

## Consequences

Future development uses `uv sync` and `uv run ...` commands rather than global
Python installs. Tests import the installed package instead of relying on the
repository root being on `sys.path`.

The repository has enough structure for reliable agent work while avoiding
speculative source layers. The first real feature can add modules named after
actual concepts instead of filling pre-created buckets.

## Verification

Run these gates from the repository root:

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

Check that runtime dependencies in `pyproject.toml` remain empty until runtime
code needs them.
