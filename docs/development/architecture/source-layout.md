# Source Layout

This repository uses a small `src/` layout so tests exercise the installed
package instead of accidentally importing Python files from the repository root.

## Current Map

- `pyproject.toml` is the canonical project and tool configuration.
- `uv.lock` records the resolved local development environment.
- `src/chess_gaze/` is the importable package. It is metadata-only until the
  first real runtime behavior is specified.
- `tests/` contains behavior tests for code in `src/chess_gaze/`.
- `docs/development/architecture/` contains current architecture guidance.
- `docs/development/decisions/` contains ADRs for architecture-significant
  decisions.
- `docs/superpowers/specs/` contains active design specs.
- `docs/superpowers/plans/` contains executable implementation plans.
- `docs/superpowers/closeouts/` contains completed-work closeouts.

## Ownership Rules

Add implementation modules only when they have meaningful behavior or protect a
real invariant. Avoid empty `core`, `services`, `adapters`, `engine`, `domain`,
or similarly generic packages until a concrete runtime seam exists.

Name modules after the concept they own. Prefer one deeper module with a stable
interface over several pass-through files that merely forward calls.

Keep tests near the behavior they verify by mirroring package paths under
`tests/` once modules exist. Configuration and documentation checks may use
top-level tests when they verify repository behavior rather than package
behavior.

If a source file grows past about 800 lines or starts owning three distinct
runtime responsibilities, perform a source-layout review before adding more.
If it grows past about 1,500 lines, write a split plan or document why the file
is intentionally deep.

## Local Development

Use `uv` for all project commands:

```sh
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

Do not install development dependencies into a global Python environment.
