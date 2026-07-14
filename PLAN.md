# Easy Agentic Data Plan

This plan tracks only the work required to produce high-quality, executable agent training
trajectories. Git history records completed implementation details; this file records current
priorities, measurable gates, and deliberate deferrals.

## North Star

Produce a versioned dataset in which every training record can be traced to:

```text
licensed seed -> fixed workspace -> model/tool interaction -> hard verification -> canonical trace
```

Success means the trajectories are useful and reproducible, not merely numerous. Source-record
counts, CLI surface area, and generated intermediate JSON files are not product outcomes.

## Current State

Implemented foundations:

- public/hidden scenario contracts and deterministic IDs;
- fixed-revision workspace materialization and rootless Docker execution;
- headless coding agent, tool policy, budgets, and optional simulated user;
- append-only traces, schema validation, state hashing, and replay;
- deterministic evaluators and SFT, preference, RL, and analysis export contracts;
- persistent batch scheduling, recovery, budgets, and quality reports;
- registry imports, provenance/license checks, decontamination, and review sampling.

Useful corpus evidence already produced in the current workspace:

- frozen Gold-20 manifest `gold20_8757dfd30b43612a` across 8 repositories and 2 languages;
- 20/20 fixed revisions reproduce the same workspace tree across clean resets;
- 20/20 hidden patches apply and fail on the base tree, then pass after replaying retained,
  byte-verified private reference repairs through the production Docker sandbox; zero evaluator
  leaks, infrastructure failures, or unresolved contamination.

This is not a production dataset. The registry-backed execution path still needs direct derived
export wiring, measured real-model trajectories, human review, and an immutable release manifest.

## M0: One Canonical Pipeline

**Goal:** remove parallel demo architecture and make the repository explain one path:

**Status:** Complete.

```text
seed -> materialize -> HeadlessAgent -> EvaluationSuite -> Trace -> trace exporters
```

Completed work:

- [x] Removed the parallel calculator pipeline, duplicate runner/tool/verifier/export models, and
  mock provider.
- [x] Removed isolated compatibility and governance placeholders that were not connected to
  rollouts.
- [x] Reduced provider configuration to model-connection settings used by headless rollouts.
- [x] Replaced historical documentation and progress logs with concise current contracts.
- [x] Passed 242 default unit tests, 239 pytest tests, Ruff, and the deterministic local synthesis
  and replay smoke.

Exit gate:

- one documented execution architecture and no imports of removed legacy modules;
- `README.md` no more than 250 lines, `PLAN.md` no more than 160 lines, and `AGENTS.md` no more than
  130 lines;
- default tests and local synthesis smoke pass.

## M1: Gold-20 Executable Seed Set

**Goal:** freeze a small, diverse, fully verified corpus before collecting more data.

**Status:** Complete.

Tasks:

- [x] Curate and rehearse 19 fixed-revision hidden-test-patch tasks.
- [x] Add one more task, then freeze exactly 20 pilot seeds with content hashes.
- [x] Require licensed provenance, reproducible materialization, stable setup/health commands, and
  hidden evaluator isolation for every seed.
- [x] Run seed and scenario decontamination against benchmark and holdout registries.
- [x] Write a Gold-20 manifest containing scenario, workspace, evaluator, and source hashes.

Exit gate:

- 20/20 seeds materialize from fixed revisions;
- at least 8 repositories and 2 programming languages are represented;
- every hidden patch applies, fails on the original workspace, and passes with a validated repair;
- all base/fixed checks reproduce through the production Docker sandbox with the declared offline
  runtime identities and policies;
- zero evaluator leakage and zero unresolved contamination findings.

The previous 100-task curation queue remains an optional candidate pool. Completing all remaining
81 records is not a prerequisite for the pilot.

## M2: 40-Trajectory Quality Pilot

**Goal:** prove the complete data path with two independent rollouts for each Gold-20 seed.

Tasks:

- [ ] Persist SFT, preference, RL, and analysis exports directly from registry-backed evaluated
  traces instead of requiring a separate demo path.
- [ ] Run two rollouts per Gold-20 seed with one declared provider configuration and fixed budgets.
- [ ] Replay every trace and rerun successful outcomes from a clean workspace reset.
- [ ] Generate a quality report covering success, infrastructure failures, tool use, termination,
  reward, leakage, duplication, and cost.
- [ ] Perform stratified human review of 20 trajectories and quarantine every critical issue.

Exit gate:

- 40 canonical traces, all schema-valid and replayable;
- infrastructure-failure rate at or below 5%;
- zero hidden-content leaks and zero hard-verifier bypasses;
- every accepted success reproduces after a clean reset;
- at least 90% of the reviewed sample is acceptable and no critical issue remains;
- SFT contains only hard-verified successes and every preference pair has a positive deterministic
  margin.

## M3: Dataset v1

**Goal:** scale only after the pilot proves data quality.

Tasks:

- [ ] Expand to 100 verified seeds based on measured pilot gaps, not raw source availability.
- [ ] Generate at least 1,000 replayable rollouts with bounded provider and infrastructure budgets.
- [ ] Enforce corpus-level coverage, duplicate, contamination, reward, and failure-distribution
  gates.
- [ ] Human-review a stratified 5% sample and resolve or quarantine every critical issue.
- [ ] Freeze source snapshots, scenario IDs, model/prompt/tool versions, traces, derived exports,
  audit reports, and approval status in one immutable manifest.

Exit gate:

- at least 1,000 valid replayable trajectories;
- infrastructure-failure rate at or below 5% and no hard-verifier bypass;
- zero known credential, personal-data, evaluator, or benchmark-oracle leakage;
- at least 90% human-review acceptance with no unresolved critical finding;
- SFT, preference, RL, and analysis records all resolve back to canonical trace IDs.

## Deferred Until a Quality Gate Requires Them

- Additional source-collection connectors and large curation queues
- Distributed execution and new worker backends
- Aggressive concurrency, caching, or provider-specific optimization
- Desktop/GUI automation and unrestricted network tools
- Model training loops and training-framework-specific schemas in the core package

## Immediate Sequence

1. Wire registry-backed trace exports and run the 40-trajectory pilot.
2. Decide Dataset v1 scope from measured pilot evidence.
