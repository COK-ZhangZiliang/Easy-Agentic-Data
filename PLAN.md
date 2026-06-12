# Easy Agentic Data Implementation Plan

This document tracks the evolution of Easy Agentic Data from its current synthetic function-tool
pipeline into a headless, sandboxed agent trajectory factory. Update it in the same change that
completes, adds, removes, or materially changes a tracked task.

## Status Legend

- `[x]` Complete and verified
- `[ ]` Not started or incomplete
- `BLOCKED` Cannot proceed until the named dependency is resolved
- `DEFERRED` Intentionally moved outside the current roadmap

## Product Direction

The target system runs two models in a controlled scenario:

1. A simulated user model interacts with the agent without access to hidden answers or evaluators.
2. An agent model uses sandboxed tools to inspect and modify a reproducible working environment.
3. Deterministic evaluators inspect the final environment state.
4. The complete observable interaction is saved as replayable training data.

The initial domain is software engineering. General desktop automation and unrestricted computer
use are out of scope until the coding environment is safe, replayable, and measurably useful.

## Success Criteria

The first production-capable release must:

- Recreate the same initial workspace from a versioned scenario specification.
- Run every command and file mutation inside an isolated sandbox.
- Record an append-only event trace containing user messages, model messages, tool requests,
  policy decisions, tool results, state hashes, and verification results.
- Replay a completed trace without calling either model.
- Keep hidden tests, reference patches, and evaluator state unavailable to both the agent and the
  simulated user.
- Determine success primarily from executable environment state.
- Export successful and failed trajectories for SFT, preference optimization, and RL pipelines.
- Resume interrupted batch synthesis without duplicating completed rollouts.

## Current Baseline

- [x] Typed task, message, tool-event, trajectory, verification, and preference models
- [x] Hosted OpenAI-compatible LLM client
- [x] Local OpenAI-compatible LLM client with optional authentication
- [x] Prompt-safe LLM call audit ledger
- [x] Self-Instruct-style task generation
- [x] Evol-Instruct-style task mutation
- [x] Minimal function-tool agent loop
- [x] Structural, tool-execution, and semantic verification
- [x] Best-of-N SFT and preference exports
- [x] Dependency-free mock end-to-end pipeline
- [x] English-only documentation and code-comment policy

The current `AgentRunner` and `ToolRegistry` are prototypes. They do not yet provide workspace
isolation, scenario-bound environments, multi-turn simulated users, replay, or deterministic coding
task evaluation.

## Architecture Boundaries

The target package layout is:

```text
src/easy_agentic_data/
  agent/          # Agent loop, context construction, and stopping policies
  environments/   # Environment specifications, provisioning, reset, and snapshots
  sandbox/        # Isolated command and filesystem execution backends
  tools/          # Tool contracts, policy metadata, and capability packs
  simulation/     # Simulated user state and interaction policy
  seeds/          # Query seed and scenario registries
  traces/         # Append-only events, artifacts, replay, and state hashing
  evaluators/     # Tests, state comparison, policy checks, and reward construction
  synthesis/      # Rollout orchestration, scheduling, recovery, and selection
  exporters/      # SFT, preference, reward-model, and RL formats
  llm/            # Model-provider protocol adapters and observability
```

Rules:

- Model-provider code must not contain agent policy.
- Tools must execute through a sandbox and policy layer.
- Evaluators must not be visible to the agent or simulated user.
- Scenario metadata must not contain secrets.
- Raw traces are immutable; derived datasets reference their source trace IDs.

## P0: Canonical Contracts and Trace Replay

**Goal:** Define stable contracts before adding a privileged execution backend.

### Tasks

- [x] Add `QuerySeed`, `EnvironmentSpec`, `Scenario`, and `ScenarioInstance` models.
- [x] Separate public task context from hidden user and evaluator context.
- [x] Define versioned `TraceEvent` variants:
  `session_started`, `user_message`, `model_response`, `tool_requested`, `policy_decision`,
  `tool_started`, `tool_finished`, `workspace_diff`, `verification_result`, and
  `session_finished`.
- [x] Add content-addressed IDs for environment specifications, scenario instances, traces, and
  artifacts.
- [x] Define termination reasons such as success, agent stop, user stop, policy violation, timeout,
  token budget, tool budget, and infrastructure failure.
- [x] Implement append-only JSONL trace writing with crash-safe flush behavior.
- [x] Implement artifact references for large stdout, stderr, patches, and snapshots.
- [x] Implement trace loading, schema-version validation, and deterministic replay.
- [x] Add schema migration guidance for future event versions.
- [x] Add contract tests for round trips, truncated traces, unknown event versions, and replay.

### Deliverables

- Versioned scenario and event schemas
- A trace recorder independent of the current `Trajectory` export model
- A replay command that performs no model or tool calls

### Exit Criteria

- A fixture trace can be recorded, loaded, and replayed to the same terminal state hash.
- A trace truncated after any complete event remains readable up to that event.
- Hidden context never appears in public message or tool events.

## P1: Headless Coding Agent and Sandbox MVP

**Goal:** Run a useful coding agent without a UI and without host-level tool execution.

### Tasks

- [x] Define the sandbox protocol: `create`, `execute`, `read`, `write`, `diff`, `snapshot`,
  `restore`, and `destroy`.
- [x] Select and document rootless Docker as the first isolation backend.
- [x] Pin images by digest and run as a non-root user.
- [x] Disable network access by default.
- [x] Enforce CPU, memory, process, wall-time, output-size, and workspace-size limits.
- [x] Prohibit Docker socket mounts and arbitrary host-path mounts.
- [x] Add a policy engine with explicit allow, deny, and reason fields.
- [x] Implement the initial tool capability pack:
  `list_files`, `read_file`, `search_files`, `apply_patch`, `run_command`, `git_status`,
  `git_diff`, and `ask_user`.
- [x] Validate every tool argument against JSON Schema before execution.
- [x] Record command exit code, bounded stdout/stderr, duration, workspace diff, and state hash.
- [x] Implement the headless agent loop with token, turn, tool, and wall-clock budgets.
- [x] Add malformed tool-call repair with a strict retry limit.
- [x] Add a CLI command for running one scenario and writing one trace.
- [x] Add deterministic fixtures that require reading, editing, testing, and inspecting a diff.

### Deliverables

- Sandbox backend and policy engine
- Coding-oriented tool pack
- Headless single-scenario agent command

### Exit Criteria

- The agent can repair a deterministic fixture repository and pass its visible tests.
- Attempts to access undeclared host paths or the network are denied and recorded.
- Resetting the scenario restores the exact initial workspace hash.
- No coding tool executes directly in the orchestration process.

The container-level exit criteria are verified with Docker CLI 29.5.3 against a user-scoped Colima
runtime. The integration suite validates non-root execution, read-only root filesystem, disabled
networking, resource settings, workspace reset, Git diff, headless-agent repair, and trace replay.

## P2: Query Seed and Environment Registry

**Goal:** Maintain reusable, versioned query seeds and their reproducible working environments.

### Tasks

- [x] Define a Git-managed seed format with query, category, difficulty, constraints, provenance,
  license, split, and parent lineage.
- [x] Define environment fields for image digest, repository commit, fixture patch, working
  directory, setup commands, capability packs, limits, health checks, reset strategy, and
  evaluator references.
- [x] Add a local SQLite index for querying seeds, environments, scenarios, and rollout status.
- [x] Keep large repositories, images, snapshots, and traces outside Git using artifact references.
- [x] Implement registry validation for missing artifacts, mutable image tags, invalid commits,
  duplicate IDs, and train/evaluation overlap.
- [x] Add scenario materialization from a seed plus environment specification.
- [x] Add deterministic parameterization using an explicit random seed.
- [x] Add repository import from a fixed commit.
- [x] Add mutation-based coding task generation with a guaranteed failing test.
- [x] Add task import from issue/commit pairs without exposing the reference patch.
- [x] Add semantic and exact duplicate detection hooks.
- [x] Add CLI commands to list, validate, materialize, and inspect registry entries.

### Deliverables

- Versioned seed and environment registries
- SQLite discovery index
- At least two reproducible seed-source adapters

### Exit Criteria

- Every scenario can be recreated from registry metadata and content-addressed artifacts.
- Registry validation detects train/evaluation leakage by source and content hash.
- At least 20 fixture scenarios pass environment health checks after repeated resets.

## P3: Simulated User and Multi-Turn Interaction

**Goal:** Generate realistic user-agent conversations without answer leakage.

### Tasks

- [x] Define hidden `UserState`: goal, persona, known facts, unavailable facts, constraints, and
  patience policy.
- [x] Define the simulated-user observation boundary.
- [x] Ensure the user model cannot access hidden tests, evaluator code, reference patches, or raw
  agent system prompts.
- [x] Implement initial query generation from a query seed and parameterized environment context.
- [x] Implement responses to `ask_user` using only `UserState`.
- [x] Support clarification, correction, refusal, confirmation, and early-stop interactions.
- [x] Add user turn and token budgets.
- [x] Record user-state transitions without exposing hidden fields in training messages.
- [x] Add deterministic rule-based user fixtures for tests.
- [x] Add LLM-based user simulation behind the same protocol.
- [x] Add leakage probes and canary values to detect hidden-context exposure.
- [x] Measure interaction diversity, clarification rate, contradiction rate, and premature
  termination.

### Deliverables

- Rule-based and LLM-based simulated-user implementations
- Multi-turn conversation orchestrator
- Leakage tests and interaction metrics

### Exit Criteria

- The agent can request and use information that is available only through `ask_user`.
- Canary hidden answers do not appear in user messages unless explicitly marked as known facts.
- The same scenario supports deterministic test users and stochastic production users.

## P4: Deterministic Evaluation and Dataset Construction

**Goal:** Make final environment state, not model opinion, the primary training signal.

### Tasks

- [x] Define evaluator contracts with isolated execution and structured evidence.
- [x] Add hidden-test execution in a separate evaluator context.
- [x] Add required-state and forbidden-change evaluators.
- [x] Add policy-violation and sandbox-integrity evaluators.
- [x] Add final diff, file hash, test result, and environment state summaries.
- [x] Separate task failure from infrastructure failure.
- [x] Define a binary base reward from deterministic success.
- [x] Add diagnostic metrics for tool errors, retries, turns, tokens, latency, patch size, and user
  interactions.
- [x] Use diagnostic metrics only as tie-breakers until reward behavior is calibrated.
- [x] Run repeated rollouts per scenario and report pass@k and success variance.
- [x] Add contamination checks against hidden tests and reference patches.
- [x] Add SFT export from successful observable traces.
- [x] Add preference export from candidates with deterministic reward differences.
- [x] Add RL episode export with observations, actions, rewards, termination reason, and masks.
- [x] Preserve failed and policy-denied trajectories in a separate analysis dataset.

### Deliverables

- Isolated deterministic evaluator suite
- Reward and metrics report
- SFT, preference, and RL episode exporters

### Exit Criteria

- Evaluation can run without either generation model.
- A successful trace reproduces its success after environment reset.
- Preference pairs always have explainable, positive deterministic margins.
- Exported records link back to immutable source traces and scenario versions.

## P5: Batch Synthesis, Reliability, and Scale

**Goal:** Produce datasets reliably across many scenarios and model configurations.

### Tasks

- [x] Define rollout job states and idempotency keys.
- [x] Add persistent scheduling and checkpoint recovery.
- [x] Resume interrupted runs without repeating completed rollouts.
- [x] Add bounded retries for transient model, sandbox, and artifact-store failures.
- [x] Add provider-level rate limiting and exponential backoff.
- [x] Add model-call caching where sampling semantics permit it.
- [x] Add concurrency controls for model endpoints, sandboxes, and artifact writes.
- [x] Add per-run token, time, and monetary budgets with hard termination.
- [x] Add health monitoring for local model endpoints and sandbox workers.
- [x] Add dataset manifests with exact scenario, model, prompt, tool, image, evaluator, and exporter
  versions.
- [x] Add quality reports for coverage, duplicates, success rate, pass@k, tool usage, interaction
  depth, policy denials, and infrastructure failures.
- [x] Add sampled human-review queues and reviewer feedback ingestion.
- [x] Add a pluggable worker protocol after stabilizing the local scheduler contract.

### Deliverables

- Recoverable batch synthesis command
- Run-level budgets and observability
- Dataset quality and lineage reports

### Exit Criteria

- A terminated batch resumes without duplicate trace IDs.
- Re-running the same idempotent job does not overwrite or duplicate completed artifacts.
- Infrastructure failures are separately measurable and do not become negative training labels.
- A release manifest can reproduce the complete scenario and evaluation configuration.

## Deferred Work

- `DEFERRED` General desktop and GUI automation
- `DEFERRED` Unrestricted network browsing
- `DEFERRED` Real production business write APIs
- `DEFERRED` Built-in model training loops
- `DEFERRED` Distributed execution before the local scheduler and data contracts stabilize
- `DEFERRED` Storing or training on private hidden chain-of-thought

## Cross-Cutting Workstreams

### Safety

- [x] Threat-model sandbox escape, prompt injection, secret exposure, artifact poisoning, and
  denial-of-service paths before P1 completion.
- [x] Add adversarial scenarios for path traversal, symlink-style escapes, command expansion, oversized output,
  fork bombs, and network access.
- [x] Require explicit review before enabling a new side-effecting capability pack.

### Data Governance

- [x] Record source, license, permitted use, and provenance for every seed.
- [x] Define retention and deletion behavior for traces and artifacts.
- [x] Add redaction checks for credentials and personal data.
- [x] Maintain immutable train, validation, and evaluation split assignments.

### Compatibility

- [x] Define OpenAI-compatible chat and tool calling as the minimum coding-scenario capability.
- [x] Add a tool-calling capability probe for use before expensive rollouts.
- [x] Keep provider-specific chat-template handling outside canonical traces.

## Recommended Execution Order

1. Complete P0 before introducing a privileged sandbox.
2. Build P1 against small, deterministic fixture repositories.
3. Begin P2 once environment reset and state hashing are stable.
4. Add P3 only after hidden/public context separation is enforced structurally.
5. Complete deterministic P4 evaluation before scaling rollout volume.
6. Begin P5 after one local batch can be reproduced end to end.

P0 and sandbox threat modeling may proceed in parallel. Dataset exporters should be updated only
after the canonical event schema is stable enough to avoid repeated migrations.

## Key Risks and Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Sandbox escape or host mutation | Critical | Rootless isolation, no host socket mounts, deny-by-default policy, adversarial tests |
| Hidden-answer leakage | Invalid data | Separate contexts, canary probes, artifact ACLs, evaluator isolation |
| Environment drift | Irreproducible reward | Image digests, fixed commits, content hashes, health checks |
| Reward hacking | Misleading training signal | Deterministic final-state checks, forbidden-change rules, evidence retention |
| Simulated-user collapse | Unrealistic conversations | State-machine constraints, diversity metrics, sampled human review |
| Infrastructure failure mislabeled as task failure | Harmful negative examples | Separate termination classes and retry policy |
| Teacher and judge correlation | Systematic bias | Different models when possible, deterministic checks, human calibration |
| Trace volume and cost | Operational failure | Artifact references, compression, budgets, bounded output |
| Evaluation contamination | Inflated results | Immutable splits, source hashes, hidden-test isolation |

## Open Decisions

- [x] Use rootless Docker as the first sandbox backend.
- [x] Use Git-managed JSON plus a disposable SQLite discovery index.
- [x] Use standard-library dataclasses for the initial trace event schemas.
- [x] Use SHA-256 content-addressed local artifacts behind a replaceable store contract.
- [x] Support Python fixture repositories with command-configured unittest or pytest runners first.
- [x] Require valid OpenAI-style function tool calls from local coding-agent models.

Record decisions that affect multiple modules as ADRs under `docs/`.

## Progress Log

| Date | Change |
| --- | --- |
| 2026-06-12 | Created the initial roadmap for the headless sandboxed agent trajectory factory. |
| 2026-06-12 | Completed P0 canonical scenario contracts, content-addressed artifacts, append-only trace recording, strict loading, hidden-context guards, and event-only replay. |
| 2026-06-12 | Implemented P1-P5 library and CLI workflows with per-feature tests. |
| 2026-06-12 | Installed Docker CLI and Colima, then passed real container isolation and end-to-end headless coding-agent integration tests. |
