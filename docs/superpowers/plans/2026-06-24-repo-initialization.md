# Repository Initialization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Initialize the repository as a private, uv-managed Python project with no chess/gaze runtime behavior.

**Architecture:** Use a minimal packaged `src/chess_gaze/` layout so local tests exercise the installed package. Keep the package metadata-only until future specs introduce real behavior. Put development workflow rules in canonical docs rather than speculative source layers.

**Tech Stack:** Python 3.12, uv, pytest, Ruff, mypy, hatchling.

**Execution Note:** This plan was executed in-session because the repository
initialization files were tightly coupled and the user requested an end-to-end
deliverable in the current turn. Subagents were used for independent
best-practice research and final review; implementation was performed by the
controller with TDD evidence and fresh verification gates.

## Global Constraints

- Follow `AGENTS.md` as the highest-priority repository instruction.
- Use `uv` for dependency resolution, locking, environment management, and command execution.
- Do not add chess, gaze, stream-analysis, or machine-learning runtime behavior.
- Do not add runtime dependencies during initialization.
- Do not add CI/CD, Docker, pre-commit hooks, notebooks, tox/nox, Makefiles, or task runners.
- Keep active specs, plans, and closeouts under `docs/superpowers/`.
- Keep source modules named after meaningful concepts and avoid speculative pass-through packages.

---

## File Structure

- Create `.gitignore` for Python caches, uv local state, build outputs, secrets, editor files, and local ML artifacts.
- Create `.python-version` with `3.12`.
- Create `README.md` with setup, gates, and current no-runtime-code status.
- Create `pyproject.toml` with project metadata, hatchling build backend, dev dependency group, pytest, Ruff, and mypy config.
- Create `uv.lock` using `uv lock`.
- Create `src/chess_gaze/__init__.py` as the metadata-only package surface.
- Create `tests/test_package_metadata.py` for package import/version behavior.
- Create `docs/development/architecture/source-layout.md`.
- Create `docs/development/decisions/README.md`, `template.md`, and `0001-use-uv-src-layout.md`.
- Create `docs/development/specs/README.md`.
- Create `docs/superpowers/specs/2026-06-24-repo-initialization-design.md`.
- Create `docs/superpowers/plans/2026-06-24-repo-initialization.md`.
- Create `docs/superpowers/closeouts/2026-06-24-repo-initialization.md` after verification.

### Task 1: Project Metadata, Tooling, and Docs

**Files:**
- Create: `.gitignore`
- Create: `.python-version`
- Create: `README.md`
- Create: `pyproject.toml`
- Create: `docs/development/architecture/source-layout.md`
- Create: `docs/development/decisions/README.md`
- Create: `docs/development/decisions/template.md`
- Create: `docs/development/decisions/0001-use-uv-src-layout.md`
- Create: `docs/development/specs/README.md`
- Create: `docs/superpowers/specs/2026-06-24-repo-initialization-design.md`
- Create: `docs/superpowers/plans/2026-06-24-repo-initialization.md`

**Interfaces:**
- Produces: uv project metadata; local gates `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .`, and `uv run mypy`.
- Consumes: `AGENTS.md` repository instructions.

- [x] **Step 1: Write the project and documentation files**

Create the files listed above with the exact repository policy captured in the spec.

- [x] **Step 2: Review for scope violations**

Run:

```sh
rg -n "torch|opencv|numpy|pandas|notebook|jupyter|pre-commit|github actions|docker|tox|nox|makefile|core|services|adapters|engine|domain" .
```

Expected: Any matches are explanatory documentation only, not created runtime dependencies, source layers, or infrastructure files.

### Task 2: Test-First Package Metadata Stub

**Files:**
- Create: `src/chess_gaze/__init__.py`
- Create: `tests/test_package_metadata.py`
- Modify: `src/chess_gaze/__init__.py`

**Interfaces:**
- Produces: `chess_gaze.__version__: str` and `chess_gaze.__all__: tuple[str, ...]`.
- Consumes: installed distribution metadata for `chess-gaze`.

- [x] **Step 1: Create an empty package stub and failing metadata tests**

`src/chess_gaze/__init__.py`:

```python
"""Package metadata for the private chess-gaze project."""
```

`tests/test_package_metadata.py`:

```python
from importlib.metadata import version

import chess_gaze


def test_package_version_matches_installed_distribution() -> None:
    assert chess_gaze.__version__ == version("chess-gaze")


def test_public_api_is_metadata_only() -> None:
    assert chess_gaze.__all__ == ("__version__",)
```

- [x] **Step 2: Run the focused test and verify RED**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_package_metadata.py -q
```

Expected: FAIL because `chess_gaze` does not define `__version__`.

- [x] **Step 3: Implement minimal metadata**

`src/chess_gaze/__init__.py`:

```python
from importlib.metadata import version

__version__ = version("chess-gaze")
__all__ = ("__version__",)
```

- [x] **Step 4: Run the focused test and verify GREEN**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_package_metadata.py -q
```

Expected: PASS.

### Task 3: Lock and Verify Local Gates

**Files:**
- Create: `uv.lock`
- Create: `docs/superpowers/closeouts/2026-06-24-repo-initialization.md`

**Interfaces:**
- Produces: reproducible uv lockfile and closeout evidence.
- Consumes: project metadata and tests from Tasks 1 and 2.

- [x] **Step 1: Generate or refresh the lockfile**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv lock
```

Expected: exit code 0 and `uv.lock` exists.

- [x] **Step 2: Run full tests**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest
```

Expected: all tests pass.

- [x] **Step 3: Run lint**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run ruff check .
```

Expected: all checks pass.

- [x] **Step 4: Run format check**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run ruff format --check .
```

Expected: all files are already formatted.

- [x] **Step 5: Run type check**

Run:

```sh
UV_CACHE_DIR=.uv-cache uv run mypy
```

Expected: no type errors.

- [x] **Step 6: Write closeout**

Create `docs/superpowers/closeouts/2026-06-24-repo-initialization.md` with:

- request summary
- implementation summary
- test-first evidence
- verification command output summary
- remaining limitations

## Self-Review

Spec coverage: the plan covers uv isolation, source layout, no domain runtime behavior, no runtime dependencies, no CI/CD, canonical docs, tests, lint, format, typing, lockfile, and closeout.

Placeholder scan: no placeholders remain in executable task steps.

Type consistency: the only produced runtime API is `chess_gaze.__version__` and `chess_gaze.__all__`, used consistently by tests and docs.
