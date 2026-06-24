# ADR-0002: Require Evidence-Based Model and Library Selection

Date: 2026-06-25

## Status

Accepted

## Context

The frame-level gaze analysis spec initially selected an older gaze estimator
without a candidate-complete comparison against a stronger user-provided
alternative. The failure was not only the specific model choice. The durable
process gap was that a spec could cite sources while still omitting the best
current candidates, license constraints, checkpoint availability, runtime
requirements, and integration risks.

Accuracy-critical computer-vision work in this repo depends on external models,
weights, and inference libraries. Those decisions can silently determine the
ceiling of the product. They must be held to a higher evidence standard than
ordinary package convenience.

## Alternatives and Evidence

| Alternative | Evidence | Decision |
| --- | --- | --- |
| Keep informal dependency selection | Fast, but allowed a familiar model to pass without a current comparison. | Rejected. |
| Require citations only | Better than memory, but citations can still be incomplete or cherry-picked. | Rejected. |
| Require primary-source candidate matrices for model and core-library choices | Forces user-provided candidates, license, assets, runtime, maintenance, task fit, and uncertainty into the decision record before implementation. | Accepted. |

## Decision

`AGENTS.md` now requires current primary-source research and candidate matrices
for external models, ML checkpoints, inference libraries, and core technical
dependencies.

`docs/development/specs/README.md` defines the required matrix columns for
specs. The ADR template now includes an `Alternatives and Evidence` section so
architecture-significant dependency decisions preserve the same evidence.

User-provided candidates must be evaluated explicitly. Convenience, familiarity,
or wrapper availability cannot outweigh accuracy, fidelity, or quality when
those are the active task's primary objectives.

## Consequences

- Specs and ADRs take longer to write when they choose important dependencies.
- Future agents have a durable guardrail against stale or weak model choices.
- Model/license/runtime uncertainty must be recorded before implementation
  rather than discovered after the project has coupled to the wrong dependency.
- Decisions can still choose a lower-quality dependency, but only if the spec
  explicitly justifies that tradeoff against the current task constraints.

## Verification

Before implementing a feature that depends on an external model, checkpoint,
inference library, or core dependency, inspect the active spec or ADR and verify
that it includes:

- a candidate matrix with user-provided candidates;
- primary-source URLs and verification date;
- license and intended-use constraints;
- checkpoint or package availability and checksum policy;
- model input/output contract when applicable;
- runtime/platform assumptions and known incompatibilities;
- residual uncertainty and implementation gates.
