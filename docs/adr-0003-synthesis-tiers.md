# ADR 0003: Synthesis Tier Contract

## Status

Accepted.

## Context

The project has both a lightweight function-tool pipeline and a scenario-bound headless coding
agent. Treating every successful run as equivalent hides important differences in data quality.
A calculator smoke test proves provider connectivity and export plumbing, but it does not prove
that the system can create realistic multi-turn agent trajectories grounded in a workspace.

## Decision

Easy Agentic Data exposes three synthesis tiers:

1. `smoke`: inexpensive provider and exporter validation through the lightweight pipeline.
2. `complex_synthetic`: deterministic repository-like fixtures using `HeadlessAgent`,
   `MemorySandbox`, simulated users, coding tools, hidden evaluation, replay, and derived exports.
3. `registry_backed`: production-style query and workspace seeds materialized through the
   scenario registry and executed in the Docker sandbox.

Each tier must state the runtime, expected data shape, verifier signal, and artifacts it produces.
The `complex_synthetic` tier exists to prove complex trajectory support without relying on paid
model calls or external repositories.

## Consequences

- Smoke-test success must not be interpreted as complex data quality.
- Complex synthetic demos can run in default CI because they do not call paid APIs, use networks,
  or require Docker.
- Registry-backed rollouts remain the target path for production-quality coding-agent data.
- New tiers or materially changed tier semantics require README, PLAN, and test updates.
