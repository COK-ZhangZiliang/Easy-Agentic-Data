# ADR 0003: Canonical Synthesis Paths

## Status

Accepted, superseding the earlier three-tier demo contract.

## Context

The repository previously exposed a calculator pipeline beside the scenario-bound headless coding
agent. The two paths used different task, trajectory, tool, verifier, and export models, which made
successful demos look equivalent even though only one path represented production data.

## Decision

All supported synthesis uses `HeadlessAgent`, scenario contracts, policy-governed tools,
deterministic evaluation, canonical traces, and trace-derived exports.

Two execution environments remain:

1. `complex_synthetic`: a deterministic `MemorySandbox` fixture for fast local validation.
2. `registry_backed`: fixed-revision workspaces executed in Docker for production trajectories.

The local path validates the contract without paid APIs or external repositories. The
registry-backed path is the production target. A new provider or worker may replace the model or
agent implementation, but it must emit the same observable trace and evaluation evidence.

## Consequences

- There is no separate lightweight task, runner, verifier, or exporter schema.
- Local smoke success proves contract wiring, not production data quality.
- Production readiness still requires materialization, hidden evaluation, replay, decontamination,
  human review, and measured pilot results.
