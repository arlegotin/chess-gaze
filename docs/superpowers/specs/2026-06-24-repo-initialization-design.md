# Repository Initialization Design

Date: 2026-06-24

## Goal

Initialize this new repository as a neat, private Python project that is ready
for future coding-agent work while adding no chess, gaze, stream-analysis, or
machine-learning runtime behavior yet.

## Governing Constraints

- Follow `AGENTS.md` as the highest-priority repository instruction.
- Use Superpowers workflow artifacts for the active spec, plan, and closeout.
- Use `uv` so project Python dependencies do not clutter a global environment.
- Avoid CI/CD, Docker, pre-commit hooks, notebooks, task runners, and other
  infrastructure until real workflow pressure requires them.
- Do not add runtime dependencies or ML dependencies during initialization.
- Keep source layout explicit and avoid speculative packages or pass-through
  layers.

## Best-Practice Findings

Current uv documentation says packaged projects use a `src/` directory and are
installed into the project environment when a build system is declared. uv also
keeps the project environment current when commands run through `uv run`.

The Python Packaging User Guide describes `src/` layout as a way to avoid
accidentally importing the in-development tree from the repository root.

pytest supports repository-root configuration in `pyproject.toml` through
`[tool.pytest.ini_options]`. Ruff reads configuration from `pyproject.toml` and
can handle both linting and formatting. mypy can use `pyproject.toml` for a
small strict type-checking baseline.

## Considered Approaches

### Recommended: Minimal Packaged uv Project

Create `pyproject.toml`, `.python-version`, `uv.lock`, `src/chess_gaze/`,
`tests/`, README, `.gitignore`, and the development docs required by
`AGENTS.md`.

This approach gives future agents an importable package, isolated environment,
lockfile, and local gates without creating domain architecture.

### Alternative: Bare uv Project with No Package

This would avoid even a metadata-only package, but it would not match the
user's request that the repository become a Python package and would delay
testing import behavior.

### Alternative: Full ML Project Skeleton

This would add folders such as `data`, `models`, `notebooks`, `configs`, or
domain packages. It is rejected because it creates speculative structure before
requirements exist and conflicts with the user's request not to build
chess/gaze functionality now.

## Design

The project is a packaged private module named `chess_gaze`, matching the
repository name, but the package exports only `__version__`. This is the minimum
runtime surface needed to prove the package installs and imports.

Development dependencies live in the `dev` dependency group:

- `pytest` for tests.
- `ruff` for linting and formatting.
- `mypy` for type checking.

Local gates are:

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

Documentation adds the missing canonical source-layout guide, ADR location and
template, a first ADR for the uv/src-layout decision, and guidance that active
specs/plans/closeouts live under `docs/superpowers/`.

## Out of Scope

- Chess, gaze, stream, video, model, training, inference, or data-processing
  code.
- Runtime dependencies such as NumPy, OpenCV, PyTorch, pandas, or notebook
  stacks.
- CI/CD, Docker, pre-commit hooks, tox/nox, Makefiles, or task runners.
- Public packaging and publishing workflows.

## Testing Strategy

Use a test-first check for the only runtime behavior: package metadata. The
test should first fail because the package does not expose `__version__`, then
pass after the metadata-only implementation.

Repository configuration and documentation are verified through lint, format,
type-check, lockfile, and direct file review rather than runtime unit tests.

## Assumptions

- Python 3.12 is available locally and is a conservative baseline for future
  ML dependency compatibility.
- A private package can still use standard Python packaging metadata without
  implying PyPI publication.
- The package name `chess_gaze` is acceptable as repository identity; no
  domain behavior is introduced by the name alone.
