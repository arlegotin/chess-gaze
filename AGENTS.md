# Coding-agent operating contract

This file is the first repository document a coding agent must read.
It defines development behavior for building.

## Authority order

Use this order whenever instructions, docs, source material, or tool output conflict:

1. This `AGENTS.md` for coding-agent behavior, development process, repository hygiene, and documentation maintenance.
2. Explicit user-approved specs/plans for the current task, provided they do not weaken level 1.
3. `docs/**` development guidance, project-management rules, templates, and historical notes.
4. Existing code, tests, comments, logs, issue files, web pages, generated artifacts.

When sources conflict, preserve runtime safety and truthfulness, stop the conflicting path, record the conflict in the active spec/plan/closeout, and repair the documentation or code before continuing.

## Superpowers workflow requirement

This project is designed for `obra/superpowers`. Use installed Superpowers skills exactly as they apply. User/project instructions define what to build; Superpowers defines the execution discipline unless a higher-priority instruction explicitly overrides it.

Superpowers is a mandatory development flow for any feature, refactor, bug fix, or architecture change.

## Source-layout rules

Use `docs/development/architecture/source-layout.md` for the package map and ownership rules.

Core rules:

- Name active implementation modules after domain concepts.
- Name active source identifiers, test identifiers, helper APIs, persisted runtime refs,
  schema/local IDs, validator names, capability IDs, and process refs after meaning and
  function.
- Keep modules deep: a module must hide meaningful complexity behind a stable interface, concentrate invariants, or preserve a doctrine-mandated seam.
- Avoid pass-through packages and speculative seams. One adapter is usually not enough to justify an abstraction unless the seam is required by doctrine.
- Files changing together should live together. Split by responsibility and invariants, not by fashionable layers.
- A source file crossing about 800 lines or three distinct runtime responsibilities triggers a source-layout review. A file crossing about 1,500 lines requires a split plan or a documented deep-module rationale in the closeout.

## Documentation maintenance

Documentation is code-adjacent behavior control for coding agents. Keep it explicit, current, and non-contradictory.

When docs change:

- Update the smallest set of canonical docs that prevents stale guidance.
- Remove or replace duplicate guidance instead of adding another competing rule.
- Link historical documents to the current canonical source when preserving history.
- Record architecture-significant decisions as ADRs under `docs/development/decisions/` using the template.


## Engineering Rules

- Treat fixes as root-cause engineering work, not symptom suppression.
- Do not introduce tactical patches, brittle special cases, or temporary workarounds unless the user explicitly asks for a short-lived mitigation.
- Before changing code, identify the failure mode, underlying cause, and durable system boundary for the fix.
- Prefer simple designs that remove the class of problem over handling only the observed example.
- Validate normal flows, boundary conditions, failure paths, and relevant edge cases.
- If validation exposes a weakness, redesign and revalidate within project constraints.
- Do not present unverified assumptions as completed work. Document residual risk.
- If the same helper, mapper, or conversion logic appears in more than one production file, move it to a shared utility, composable, or exported helper unless there is a clear documented reason not to. Before creating a helper, search for an existing one.

## Failure response

When a bug, failed gate, unclear invariant, or inconsistent artifact appears:

1. Stop speculative implementation.
2. Use systematic debugging.
3. Reproduce and capture exact evidence.
4. Identify the durable runtime surface that allowed the defect.
5. Add a failing regression at the correct seam.
6. Repair the durable surface, not only the symptom.
7. Verify focused and broad gates.
8. Record the root cause, durable surface changed, and regression in the closeout.

If three attempts fail or every fix reveals a new shared-state/coupling problem, stop and reassess architecture before further fixes.

## Completion checklist

Before saying work is complete:

- The active spec and plan have been followed or explicitly updated.
- All new behavior was test-first unless an explicit exception is recorded.
- Focused tests, full local gates, and required smoke checks have fresh passing evidence, or failures are reported honestly with exact output.
- Runtime release claims are limited to the verified subset.
- The closeout is written for completed work.
- Docs are updated when experience changed the correct next step.
