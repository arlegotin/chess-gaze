# Specs, Plans, and Closeouts

Active Superpowers workflow artifacts live under `docs/superpowers/`:

- `docs/superpowers/specs/` for design specs.
- `docs/superpowers/plans/` for executable implementation plans.
- `docs/superpowers/closeouts/` for completed-work closeouts.

Use this directory for durable guidance about the specification process, not as
a competing location for active task artifacts.

Every completed change should leave enough evidence for a later agent to answer:

- What was requested?
- What was intentionally out of scope?
- What plan was followed or updated?
- Which tests and gates were run?
- Which failures, exceptions, or limitations remain?

Specs that choose external models, checkpoints, inference libraries, or core
technical dependencies must include a selection matrix before the chosen stack.
Required columns are:

- candidate
- task fit
- primary sources and verification date
- published metrics or direct evidence
- checkpoint or package availability
- license and intended-use constraints
- maintenance status
- runtime and platform fit
- integration cost and reproducibility risk
- known caveats
- whether the user provided the candidate
- decision
- confidence

If the matrix is incomplete, the stack decision is provisional and must not be
implemented as final.
