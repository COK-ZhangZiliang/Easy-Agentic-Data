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
- [x] Versioned scenarios, append-only traces, artifact storage, and deterministic replay
- [x] Rootless Docker sandbox with policy-governed coding tools
- [x] Query seed and environment registry with deterministic materialization
- [x] Rule-based and LLM-backed multi-turn user simulation
- [x] Deterministic evaluation and SFT, preference, RL, and analysis exports
- [x] Persistent batch scheduling, recovery, and supporting reliability utilities
- [x] DeepSeek V4 Flash and generic local OpenAI-compatible provider validation
- [x] Three-tier synthesis workflow separating smoke, complex synthetic, and registry-backed data
- [x] Real SWE-bench Lite seed preparation with fixed-commit repository cloning
- [x] Comprehensive seed metadata for task family, source method, train eligibility,
  contamination tags, verifier types, and coverage tags
- [x] Seed-library audit command for coverage and benchmark-contamination checks
- [x] Seed-library scale-up gates for trainable-pool coverage budgets and seed-level held-out
  overlap checks
- [x] Non-benchmark public issue/PR seed import with license allowlists and fixed-revision
  workspace requirements
- [x] Per-family verifier templates and seed-audit evidence requirements
- [x] Repository-grounded synthetic seed generation for test authoring, refactoring, dependency
  upgrades, migrations, docs/examples, security hardening, performance, CI/build repair, code
  review, and repo-understanding tasks
- [x] Scenario-level decontamination reports for held-out tests, reference artifacts, oracle hashes,
  and source instances
- [x] Stratified human-review queue generation by task family, difficulty, source method, and
  verifier type
- [x] Seed-corpus build gate that imports configured train and holdout sources, runs registry,
  seed, scenario, coverage, and review-queue checks, and freezes a manifest for pilot decisions
- [x] Repository allowlist audit and seed-corpus allowlist enforcement for train-source quarantine
- [x] Source collection plan and local export audit for public issue/PR records before registry
  import
- [x] First production seed-corpus policy and 10-repository allowlist candidate files with static
  coverage gates, repository-share readiness, and collection-plan validation
- [x] Public issue/PR source export command that turns a collection plan into auditable JSONL
  records
- [x] Resumable, shardable public issue/PR source export with partial-success summaries for
  rate-limited collection
- [x] Registry-backed smoke rollout coverage for every supported task family in
  `tests/test_seed_library_rollouts.py`

The core P0-P4 architecture and the P5 scheduler are implemented. Production integration remains
for independent resource gates, call caching, hard budget admission, quality-report commands, and
production seed-corpus population. The lightweight `AgentRunner` pipeline remains as a
dependency-free demonstration path, while
`HeadlessAgent` is the scenario-bound runtime for sandboxed coding trajectories.

## Architecture Boundaries

The current package layout is:

```text
src/easy_agentic_data/
  agent/               # Headless agent loop and stopping policies
  environments/        # Reproducible environment specifications
  sandbox/             # Docker and in-memory execution backends
  seeds/               # Query seed contracts
  traces/              # Append-only events, artifacts, replay, and state hashing
  llm/                 # Provider adapters, mocks, and observability
  coding_tools.py      # Policy-governed coding capability pack
  scenarios.py         # Public and hidden scenario contexts
  registry.py          # Git registry validation and SQLite discovery index
  real_seed_sources.py # Real seed download, repository clone, and registry preparation
  simulation.py        # Rule-based and LLM-backed simulated users
  synthesis_tiers.py   # Smoke, complex synthetic, and registry-backed synthesis workflows
  evaluation.py        # Deterministic evaluators and reward reports
  trace_exporters.py   # Canonical trace dataset exports
  batch.py             # Persistent scheduling, recovery, and quality reports
  pipeline.py          # Lightweight synthetic function-tool demonstration
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
- [x] Add external SWE-style JSON/JSONL import for paired query and workspace seed records.
- [x] Add SWE-bench Lite preparation that fetches real seed records and clones repositories at
  fixed commits before registry-backed agent runs.
- [x] Mark known benchmark imports as non-training seeds and attach contamination metadata.
- [x] Track task family, source construction method, verifier types, and coverage tags on each
  query seed.
- [x] Add a seed-library audit CLI for task coverage, source coverage, train eligibility, verifier
  coverage, and benchmark-contamination failures.
- [x] Execute environment setup commands before registry-backed agent runs so real workspaces can
  be initialized offline from prebuilt images.
- [x] Add semantic and exact duplicate detection hooks.
- [x] Add CLI commands to list, validate, materialize, and inspect registry entries.
- [x] Execute each environment `health_check` during materialization and repeated reset validation.

### Deliverables

- Versioned seed and environment registries
- SQLite discovery index
- At least two reproducible seed-source adapters

### Exit Criteria

- Every scenario can be recreated from registry metadata and content-addressed artifacts.
- Registry validation detects train/evaluation leakage by source and content hash.
- Twenty in-memory reset fixtures reproduce identical state, and registry-backed materialization
  runs environment health checks before accepting a workspace.

## P3: Simulated User and Multi-Turn Interaction

**Goal:** Generate realistic user-agent conversations without answer leakage.

### Tasks

- [x] Define hidden `UserState`: goal, persona, known facts, unavailable facts, constraints, and
  patience policy.
- [x] Add goal-component, disclosure-policy, stop-condition, and business-knowledge reference
  fields for profile-conditioned simulated users.
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
- [x] Measure simulator goal alignment, disclosure violations, unavailable-fact requests and
  leaks, and critical simulator errors.

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
- [x] Add a derived RL episode v1 export with explicit assistant action type, action mask, loss
  mask, and separated outcome versus turn reward components.
- [x] Preserve failed and policy-denied trajectories in a separate analysis dataset.
- [x] Add deterministic turn reward evidence for policy denials, tool execution, and user
  information-gathering actions.

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
- [x] Add provider HTTP retry backoff and a reusable rate-limiter primitive.
- [ ] Wire provider-level rate limiting into rollout workers.
- [ ] Wire model-call caching into deterministic or cache-safe sampling paths.
- [ ] Apply independent model, sandbox, and artifact concurrency gates in the scheduler.
- [ ] Enforce token, time, and monetary budgets before admitting additional work.
- [ ] Integrate local model endpoint and sandbox worker health checks into batch execution.
- [ ] Emit dataset manifests with exact scenario, model, prompt, tool, image, evaluator, and exporter
  versions.
- [x] Expose quality reports for coverage, duplicates, success rate, pass@k, tool usage, interaction
  depth, policy denials, and infrastructure failures.
- [x] Add reward variance, low-information rollout group, goal-type success, simulator-error, and
  goal-alignment diagnostics to batch quality reports.
- [x] Add scenario-level quality gates and token estimates for scale-up candidate selection.
- [x] Add a scale-readiness summary artifact before approving larger provider runs.
- [x] Add explicit smoke, complex synthetic, and registry-backed synthesis tiers.
- [ ] Add configurable human-review sampling and reviewer feedback ingestion.
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

## P6: Comprehensive Task Seed Library

**Goal:** Build a broad, auditable seed library for code-agent training without contaminating
held-out benchmarks.

### Tasks

- [x] Promote task family, source method, train eligibility, contamination tags, verifier types, and
  coverage tags into the query-seed contract.
- [x] Add seed-library audit reports for family distribution, source distribution, verifier
  distribution, coverage tags, and benchmark contamination.
- [x] Treat SWE-bench Lite and other known benchmark-style sources as validation or evaluation
  inputs by default rather than train seeds.
- [x] Add seed-level coverage budgets that fail scale-up when one task family, repository,
  language, or source method dominates the trainable pool.
- [x] Add seed-level decontamination checks comparing trainable seeds against held-out query text,
  provenance, source instance, and repository overlap.
- [x] Add non-benchmark public issue and PR importers with license allowlists and fixed-revision
  workspaces.
- [x] Add repository-grounded synthetic seed generators for test authoring, refactoring,
  dependency upgrades, migrations, docs/examples, security hardening, performance, CI/build repair,
  code review, and repo-understanding tasks.
- [x] Add per-family verifier templates and minimum evidence requirements.
- [x] Add scenario-level benchmark decontamination reports comparing trainable trajectories against
  held-out tests and reference artifacts.
- [x] Add sampled human-review queues stratified by task family, difficulty, source method, and
  verifier type.
- [x] Add registry-backed smoke rollout evidence for every supported task family, including
  trace-quality evaluation for repository-understanding tasks.

### Deliverables

- Multi-family seed adapters and generators
- Seed-library quality and contamination reports
- Train/dev/eval partition policy that keeps benchmarks measurable

### Exit Criteria

- Trainable seeds come only from licensed, non-benchmark, reproducible sources or verified
  repository-grounded synthesis.
- Each supported task family has executable verifier evidence and at least one end-to-end
  registry-backed rollout.
- Seed-library audits block scale-up when benchmark contamination, license gaps, verifier gaps, or
  severe coverage imbalance are detected.
- Seed-library decontamination blocks trainable seeds that duplicate held-out query text,
  provenance, or source instances.

## P7: Production Seed Corpus Population

**Goal:** Populate a production-scale, non-benchmark, auditable train seed registry that can drive
larger DeepSeek V4 Pro synthesis runs without invalidating held-out benchmark evaluation.

### Tasks

- [x] Define the first production seed-corpus target size, per-family minimum counts, per-language
  targets, and maximum repository/source shares before collecting data.
- [x] Build a repository allowlist from permissively licensed, active public repositories with
  reproducible Git history, stable test commands, and no benchmark overlap.
- [x] Add an automated public issue/PR export path from collection plans to auditable source JSONL.
- [x] Add resume, task sharding, sleep throttling, and partial-success summaries for public
  issue/PR export runs.
- [x] Add a source collection readiness gate that combines collection plans, export summaries, and
  audit outputs before registry import.
- [x] Validate a small public issue/PR registry import rehearsal and harden task-family inference
  against noisy PR body checklist terms.
- [x] Add a reusable registry import rehearsal gate that imports audited sources into a temporary
  registry, applies repository allowlists, runs registry validation, and enforces seed-audit policy
  before materialization.
- [x] Add a CI-specific source collection record contract for failed workflow runs, including fixed
  head revisions, public run URLs, labels, and `ci_commands` verifier evidence, while blocking
  accidental public issue/PR import of CI records.
- [x] Add a CI-specific registry importer that maps failed workflow runs to `ci_build` seeds,
  stores `ci_commands` as hidden verifier evidence, and keeps CI records out of the public
  issue/PR importer.
- [x] Add an optional import-rehearsal materialization gate that samples imported scenarios,
  materializes local `file://` workspaces, and can run hidden verifier commands before rollout.
- [x] Add a source-record split gate that routes mixed collection exports into issue/PR and CI
  shards before format-specific import rehearsals.
- [x] Add per-task collection export outcomes and a retry-plan gate that maps failed, skipped, and
  unselected source-collection tasks to explicit retry shards.
- [x] Add a production collection authentication gate so `collection-export` can require a
  configured GitHub token before making network requests.
- [x] Add a retry-run gate that consumes collection retry plans and executes single-task retry
  shards against the shared source JSONL with resume behavior.
- [x] Add a collection-summary gate that rebuilds one readiness-compatible export summary from
  the final source JSONL plus all shard and retry-run summaries, avoiding record double-counting
  while preserving unresolved task failures.
- [x] Add a collection-preflight gate that checks local plan validity, selected shard coverage,
  required GitHub authentication, and optional source/summary artifacts before any networked
  source export starts.
- [x] Add a collection-shards gate that writes a deterministic shard runbook with task offsets,
  source-type mix, output paths, and exact preflight/export arguments for production collection.
- [x] Add a collection-shard-status gate that reads shard schedules, preflight reports, export
  summaries, and source JSONL to report per-shard next actions before summary/readiness.
- [x] Add a clean accepted-output gate to `collection-audit` so quarantined public records can be
  excluded before readiness, split, and import-rehearsal stages while raw quarantine evidence is
  preserved.
- [ ] Collect public issue and PR records from allowlisted repositories, including title/body,
  labels, source URLs, fixed base commits, license, language, candidate verifier commands, and
  source-instance IDs.
- [ ] Reject or quarantine records with missing licenses, mutable revisions, missing source URIs,
  personal data, credentials, private URLs, or benchmark contamination signals.
- [ ] Generate repository-grounded synthetic tasks for under-covered task families using fixed
  repository snapshots and family-appropriate verifier evidence.
- [ ] Materialize the corpus into `runs/train-registry` or an explicitly configured external data
  root using the registry importer and repository synthesis generator.
- [ ] Build an evaluation/holdout registry from benchmark and curated non-train sources, keeping
  evaluator oracles out of trainable seed prompts and public traces.
- [ ] Run `registry seed-audit` with production coverage gates for all supported task families,
  verifier types, repository shares, source-method shares, language shares, and minimum trainable
  seed count.
- [ ] Run `registry scenario-audit` against the holdout registry to block hidden-test, reference
  artifact, oracle-hash, and source-instance overlap.
- [ ] Generate a stratified human-review queue and record reviewer decisions for sampled seeds
  before approving large provider spend.
- [ ] Run a small registry-backed pilot across all task families, then use batch quality reports,
  trace-logic audits, and scale-readiness checks before launching larger shards.
- [ ] Freeze a seed-corpus manifest with registry root, source snapshots, prompt/config versions,
  audit outputs, review sample path, and approved scale-up decision.

### Detailed Execution Plan

Current checkpoint:

- `examples/production-seed-corpus-policy.json` defines the first target as 1,000 train-eligible
  seeds, all supported task families, required verifier types, per-family minimums, source-method
  minimums, an 80% maximum language share, a 25% maximum family share, a 65% maximum source-method
  share, and a 10% maximum repository share.
- `examples/production-repository-allowlist.json` records the first ten public Python repository
  candidates with permissive licenses, public Git source URIs, collection sources, labels, stable
  commands, and source-evidence URLs.
- This checkpoint is not production approval. Ten repositories satisfy the first repository-share
  threshold for a 1,000-seed corpus, but the allowlist is still Python-only, the current clean
  authenticated export contains 107 accepted records rather than the 1,000-record target, and
  `scale_decision.approved` remains false.
- `registry collection-export` now turns issue, pull-request, and CI collection-plan tasks into
  normalized public source JSONL. Issue and PR records are trainable-source candidates for the
  public issue/PR importer, while CI records are trainable-source candidates for the public CI
  importer.
- `collection-export` supports task offsets, max-task shards, sleep throttling, resume without
  duplicate source-instance IDs, summary files, and partial-success mode for API rate limits.
- New export summaries include per-task outcomes, and `registry collection-retry-plan` turns
  failed, skipped, missing-outcome, and unselected source-collection tasks into exact retry shards
  with task IDs, repositories, source types, and `--task-offset N --max-tasks 1` arguments.
- `registry collection-retry-run` can now consume those retry plans and execute selected retry
  tasks against the shared source JSONL with resume enabled, producing a retry-run summary for
  follow-up audit and readiness decisions.
- `registry collection-summary` now rebuilds a single readiness-compatible export summary from
  the final source JSONL plus all collection-export and collection-retry-run summaries. The final
  JSONL is the record-count source of truth, while task outcomes from later retry summaries replace
  earlier failed shard outcomes so resolved rate-limit failures do not block a clean gate.
- `registry collection-audit --accepted-output` now writes a clean accepted-record JSONL while
  preserving raw quarantine evidence. Downstream readiness, split, and import-rehearsal gates can
  consume the clean JSONL so a quarantined public record cannot silently enter trainable sources.
- `registry collection-preflight` now checks the collection plan, selected shard, optional existing
  source JSONL, optional summary files, and required GitHub authentication before networked source
  export begins. It records only whether the named token environment variable is configured, never
  the token value.
- `registry collection-shards` now turns a production collection plan into a deterministic shard
  runbook with one preflight/export command pair per shard, stable output paths, source-type
  counts, and repository coverage for each shard.
- Production `collection-export` runs can now use `--require-github-token` with
  `--github-token-env GITHUB_TOKEN` to fail locally when authenticated collection is not
  configured, instead of silently falling back to anonymous GitHub API limits.
- `registry collection-readiness` now combines the collection plan, export summary, and collection
  audit into a registry-import gate with accepted-record, quarantine, source-type, clean-export,
  and full-plan coverage thresholds.
- `registry collection-split` can split mixed collection exports by normalized source type so the
  public issue/PR and public CI import rehearsals consume only compatible records.
- A small registry import rehearsal with the currently collected PR records passes registry
  validation and seed-audit family/verifier consistency after filtering noisy PR checklist terms
  out of task-family inference.
- `registry import-rehearsal` now makes that probe repeatable by creating a temporary registry from
  audited source JSONL, applying the repository allowlist, running registry validation, enforcing
  seed-audit policy, and returning a blocking exit code when import, quarantine, or coverage gates
  fail.
- `collection-export` can now export CI failure records as `public_ci` audit records with fixed
  workflow-run revisions and `ci_commands` evidence. These records are intentionally rejected by
  the public issue/PR importer; the `public_ci` importer now handles them as `ci_build` seeds with
  hidden-command verifier evidence.
- `registry import-rehearsal` can now optionally materialize sampled imported scenarios and run
  hidden verifier commands when records point at local `file://` workspaces. This gives production
  source import an explicit pre-rollout workspace proof instead of relying on schema validity alone.
- The latest anonymous GitHub API collection attempt covered and processed all 30 planned tasks
  after CI collection support. It added five PR records to the existing two-record probe, but 28
  GitHub requests still hit HTTP 403 rate limits. The current source collection remains a probe
  with 7 accepted PR records, no accepted issue or CI records, and no production import approval.
- The authenticated source collection run processed all 30 planned issue, pull-request, and CI
  tasks across eight shards. The first pass exposed local CA-bundle failures, and the rerun with
  `SSL_CERT_FILE` pointing at `certifi` produced 108 raw records with clean shard summaries and no
  unresolved export issues.
- The raw collection audit quarantined one public PR record because its body contained a local
  `127.0.0.1` URL. The clean accepted output contains 107 records: 8 `public_issue`, 49
  `public_pr`, and 50 `public_ci` records across all ten allowlisted repositories.
- Clean readiness passes a 100-record probe gate with all required source types, zero quarantine,
  no export issues, and full 30-task plan coverage. The production readiness gate still fails
  correctly because 107 accepted records is below the configured 1,000-record minimum.
- Clean issue/PR and CI import rehearsals now pass separately. Issue/PR import accepts 57 records
  after tightening docs-family inference so docs-labeled records without doctest or example-command
  evidence do not violate family-specific verifier requirements; CI import accepts 50 records as
  `ci_build` seeds with hidden-command evidence.
- `registry collection-shard-status` now reads the deterministic shard runbook plus local
  preflight reports, shard export summaries, and source JSONL, then assigns each shard a next
  action such as `run_preflight`, `run_export`, `resolve_preflight`, `plan_retry`, or
  `inspect_artifact` before merged-summary or readiness decisions.
- The next data gate is to expand or deepen non-benchmark public source collection until the clean
  accepted output meets the 1,000-record production minimum, then rerun clean readiness and
  import-rehearsal gates before any registry materialization or provider rollout.

Next milestone plan:

| Milestone | Inputs | Actions | Outputs | Exit gate |
| --- | --- | --- | --- | --- |
| P7.1 Source collection readiness | Production policy, repository allowlist, collection plan, shard export summaries, retry-run summaries, and collection audit outputs | Run all issue/PR/CI export shards, resume rate-limited shards, merge the final source JSONL and summaries, audit exported records, and summarize accepted, quarantined, skipped, duplicate, and failed tasks | Public source JSONL, merged export summary JSON, collection audit JSON, readiness summary JSON | Accepted source records meet the configured minimum, quarantine count is within budget, required source types are present, the merged summary covers all plan tasks, and all unresolved export failures are explicitly listed |
| P7.2 Registry import rehearsal | Audited public source JSONL, repository allowlist, fixed-revision rules, and license allowlist | Import records into a temporary train registry, reject records outside the allowlist, verify fixed commits, and run registry validation before creating scenarios | Temporary train registry, import manifest, rejected-record report, registry validation output | Every accepted seed has a stable `task_id`, fixed source revision, license-compatible provenance, train eligibility, contamination tags, task family, source method, verifier types, and coverage tags |
| P7.3 Synthetic coverage backfill | Seed-audit gaps from the import rehearsal and fixed repository snapshots | Generate repository-grounded synthetic tasks only for missing families, verifier types, languages, or difficulty bands; keep synthetic records separated by `source_method` | Synthetic source specs, generated train registry records, verifier-evidence report | Synthetic seeds improve audited coverage without replacing required real-source records or satisfying verifier requirements they do not actually exercise |
| P7.4 Holdout and decontamination | Benchmark registries, curated non-train sources, hidden evaluator metadata, and train registry candidates | Build the holdout registry, run seed and scenario audits, compare normalized prompts, provenance, source instances, hidden tests, reference artifacts, oracle hashes, and patch/test-patch hashes | Holdout registry, seed decontamination report, scenario decontamination report, quarantine report | No trainable scenario overlaps held-out evaluator oracles; every contamination hit is removed, relabeled non-train, or quarantined with an explicit reason |
| P7.5 Human-review queue | Candidate train registry, coverage audit, decontamination reports, and review sampling policy | Sample by task family, source method, difficulty, verifier type, repository, and language; present public query, provenance, verifier evidence, and risk checklist for review | Stratified review JSONL, reviewer decision summary, follow-up quarantine list | Reviewer decisions approve the sampled strata or quarantine every actionable source-quality, leakage, privacy, or verifier issue before provider spend increases |
| P7.6 Pilot rollout | Approved train registry slice, frozen prompt/config versions, DeepSeek V4 Pro provider config, and shard budget | Run a small registry-backed pilot across all supported task families, then inspect trace validity, tool success, reward distribution, hard-check failures, prompt leakage, and cost | Pilot traces, batch quality report, trace-logic audit, cost report, scale-readiness summary | Pilot quality and cost meet configured thresholds, hard verifier failures are not hidden by soft scores, and failures are assigned to task, model, verifier, or infrastructure categories |
| P7.7 Frozen corpus manifest | Final train registry, holdout registry, audit outputs, review outputs, pilot outputs, provider settings, and scale decision | Freeze all source snapshots, registry roots, prompt/config versions, audit paths, review decisions, pilot selection, and scale approval in one manifest | Versioned seed-corpus manifest and scale decision artifact | Larger shards can only launch from this immutable manifest; any source, prompt, verifier, or registry change creates a new manifest version |

Operational sequencing:

1. Complete public source collection before adding synthetic backfill so coverage gaps are measured
   against real non-benchmark evidence rather than assumed from the policy target.
2. Run import rehearsal before any paid rollout so malformed records, mutable revisions, license
   issues, and verifier-evidence gaps fail locally.
3. Build the holdout registry before scale-up so benchmark and curated evaluation sources remain
   useful for downstream quality measurement.
4. Treat anonymous API rate limits, skipped CI tasks, and partial exports as visible readiness
   issues rather than silent success.
5. Keep every generated source, registry, review, pilot, and manifest artifact under `runs/` or an
   explicitly configured external data root; do not commit production data outputs.

Immediate next execution plan:

1. **Authenticated source collection**
   - Run `collection-shards` once from the production collection plan to freeze shard offsets,
     per-shard summaries, and preflight/export command arguments before starting authenticated
     collection.
   - Run `collection-shard-status` from the shard schedule before and after each shard pass. Use
     its per-shard `next_action` field as the operational queue: `run_preflight` before local
     checks, `run_export` after a clean preflight, `resolve_preflight` for missing auth or invalid
     local inputs, `plan_retry` for partial or failed exports, and `inspect_artifact` for malformed
     run artifacts.
   - Run `collection-preflight` before each production shard so missing authentication, empty task
     selections, invalid plans, malformed source JSONL, or missing resume summaries fail before
     any networked GitHub request starts.
   - Run resumable `collection-export` shards with a GitHub token, fixed task offsets, conservative
     sleep throttling, and `--allow-partial` so rate-limited tasks remain visible in the summary.
   - Use `--require-github-token --github-token-env GITHUB_TOKEN` for production shards so missing
     authentication fails before any anonymous GitHub API request is made.
   - Run `collection-retry-plan` after every shard or combined summary so failed, skipped,
     missing-outcome, and not-yet-selected tasks are assigned to explicit single-task retry shards.
   - Run `collection-retry-run` over selected retry tasks so incomplete shards can be resumed from
     machine-readable retry metadata instead of hand-copied command fragments.
   - Run `collection-summary` over the final source JSONL, every shard summary, and every retry-run
     summary before any audit/readiness decision. Treat this merged summary as the only
     production `collection-readiness --export-summary` input.
   - Run `collection-audit --accepted-output` after raw collection so records with private/local
     URLs, non-public source links, mutable revisions, missing licenses, or other quarantine issues
     are preserved in the raw audit but excluded from the clean source JSONL used downstream.
   - Keep the combined source JSONL under `runs/seed-corpus-demo/` until readiness passes.
   - Exit gate: all 30 current plan tasks are represented in the merged summary, resolved retry
     attempts replace earlier failed shard outcomes, and every remaining failure is assigned to a
     task ID, repository, source type, retry status, and next action.

2. **Source audit and readiness**
   - Run `collection-audit` against the raw exported source JSONL and the checked-in allowlist,
     then rerun audit against the clean accepted output.
   - Run `collection-summary` and `collection-readiness` against the clean accepted output with the
     production minimum, zero quarantine budget, required `public_issue`, `public_pr`, and
     `public_ci` source types, clean export summaries, and full plan-task coverage.
   - Exit gate: readiness is true; otherwise the corpus remains a probe and cannot feed registry
     import or paid DeepSeek V4 Pro synthesis.

3. **Trainable issue/PR import rehearsal**
   - Use `collection-split` to route `public_issue` and `public_pr` source records away from
     `public_ci` records before invoking the public issue/PR importer.
   - Run `import-rehearsal` into a temporary registry with allowlist enforcement, registry
     validation, seed-audit gates, and explicit quarantine accounting.
   - Exit gate: imported issue/PR seeds have stable task IDs, fixed revisions, compatible licenses,
     source-instance IDs, task families, verifier types, train eligibility, and coverage tags.

4. **CI source import and materialization validation**
   - Use `collection-split` to route `public_ci` source records into a dedicated CI shard, then run
     `public_ci` import rehearsals separately from issue/PR shards so failed workflow runs are
     routed through the CI importer instead of the public issue/PR importer.
   - Use the import-rehearsal materialization gate on sampled CI scenarios from fixed source
     revisions and confirm their `ci_commands` run as hidden verifier commands in the target
     workspace.
   - Exit gate: CI records can be imported, materialized, reset, and verified without passing
     through the public issue/PR importer.

5. **Coverage backfill and holdout separation**
   - Use seed-audit gaps from real source import to decide which repository-grounded synthetic
     families are still needed.
   - Build or refresh the holdout registry before scale-up so benchmark and curated evaluation
     sources remain separate from trainable seeds.
   - Exit gate: seed and scenario decontamination pass for query text, provenance, source
     instances, hidden tests, reference artifacts, oracle hashes, and patch/test-patch hashes.

6. **Human review and small provider pilot**
   - Generate the stratified human-review queue from the candidate train registry and quarantine
     every actionable source-quality, privacy, leakage, or verifier issue.
   - Run a small DeepSeek V4 Pro registry-backed pilot only after source, import, coverage,
     decontamination, and review gates pass.
   - Exit gate: trace validity, tool success, reward distribution, hard-check failure handling,
     prompt leakage checks, and cost reports justify larger shards.

7. **Manifest freeze for scale-up**
   - Freeze the train registry, holdout registry, source snapshots, prompt/config versions, audit
     outputs, review decisions, pilot selection, provider settings, and scale decision in a
     versioned seed-corpus manifest.
   - Exit gate: large DeepSeek V4 Pro shards launch only from that immutable manifest; any source,
     prompt, verifier, or registry change creates a new manifest version.

### Deliverables

- Production train seed registry with licensed non-benchmark public and synthetic seeds
- Holdout/evaluation registry for contamination checks and downstream measurement
- Seed-corpus manifest, quality reports, decontamination reports, and review queue outputs
- Scale-up decision artifact for DeepSeek V4 Pro shard execution

### Exit Criteria

- The train registry satisfies the configured family, verifier, language, source, and repository
  coverage budgets.
- Every trainable seed traces to a license-compatible source, fixed revision, and reproducible
  workspace specification.
- Seed and scenario decontamination reports pass against all configured holdout and benchmark
  registries.
- Human review approves the sampled seed queue or quarantines every actionable issue.
- A pilot run produces enough successful registry-backed trajectories to justify larger shard
  synthesis under the configured budget gates.
- The release manifest can reproduce the exact seed registry, audit outputs, review sample, pilot
  selection, and scale-up decision.

## Deferred Work

- `DEFERRED` General desktop and GUI automation
- `DEFERRED` Unrestricted network browsing
- `DEFERRED` Real production business write APIs
- `DEFERRED` Built-in model training loops
- `DEFERRED` Distributed execution before the local scheduler and data contracts stabilize
- `DEFERRED` Storing or training on evaluator-only or hidden-context chain-of-thought

## Cross-Cutting Workstreams

### Safety

- [x] Threat-model sandbox escape, prompt injection, secret exposure, artifact poisoning, and
  denial-of-service paths before P1 completion.
- [x] Add adversarial scenarios for path traversal, command expansion, oversized output, and
  network access.
- [ ] Add explicit Docker integration tests for symlink traversal and process-exhaustion containment.
- [x] Require explicit review before enabling a new side-effecting capability pack.

### Data Governance

- [x] Record source, license, permitted use, and provenance for every seed.
- [x] Record task family, source construction method, training eligibility, contamination tags,
  verifier types, and coverage tags for seed-library audits.
- [x] Define retention and deletion behavior for traces and artifacts.
- [x] Add redaction checks for credentials and personal data.
- [x] Maintain immutable train, validation, and evaluation split assignments.

### Compatibility

- [x] Define OpenAI-compatible chat and tool calling as the minimum coding-scenario capability.
- [x] Add a tool-calling capability probe for use before expensive rollouts.
- [x] Keep provider-specific chat-template handling outside canonical traces.

## Implementation Sequence

The phases were completed in this order:

1. P0 established canonical contracts and replay before privileged execution.
2. P1 added the sandboxed coding agent against deterministic fixture repositories.
3. P2 added reproducible query and environment registries.
4. P3 added simulated users after hidden/public context separation was enforced.
5. P4 made deterministic environment evaluation the primary reward.
6. P5 added recoverable batch execution after local end-to-end reproducibility was established.

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
| 2026-06-15 | Validated DeepSeek V4 Flash with real structured generation, tool calls, semantic evaluation, dataset export, and a Docker coding-agent repair trajectory. |
| 2026-06-15 | Added provider request options, reasoning-context round trips, bounded HTTP retries, response validation, policy-filtered tool schemas, improved prompts, and opt-in live-provider tests. |
| 2026-06-15 | Synchronized implementation status and package layout, documented remaining P2/P5 integration gaps, rejected empty generation batches, and separated provider reasoning context from canonical datasets. |
| 2026-06-17 | Added RL episode action/loss-mask exports, deterministic turn rewards, simulator goal-state metrics, and batch reward-variance diagnostics. |
| 2026-06-17 | Added SWE-style query/workspace seed import from local JSON and JSONL sources into the scenario registry. |
| 2026-06-17 | Preserved assistant `reasoning_content` in trajectory, SFT, preference, and RL episode exports while keeping hidden-context reasoning out of agent records. |
| 2026-06-19 | Added three synthesis tiers plus a complex synthetic multi-tool trajectory generator with hidden evaluation and derived exports. |
| 2026-07-01 | Added a DeepSeek V4 Pro thinking config, batch quality-report CLI, scenario-level scale-up candidate selection, and token-estimate sharding for 50-trace pilot review. |
| 2026-07-01 | Added a no-side-effect batch dry run for shard selection previews before paid provider scale-up. |
| 2026-07-01 | Added an agent-stop-rate scale gate and a stricter DeepSeek V4 Pro scale queue based on full pilot trace review. |
| 2026-07-01 | Added a reusable trace-logic audit CLI for reviewing coherence, completion, and multi-step complexity before continuing scale-up. |
| 2026-07-01 | Wired trace-logic audit metrics into scale candidate selection and prepared an audit-strict DeepSeek V4 Pro scale queue. |
| 2026-07-01 | Added a scale-readiness summary CLI to combine candidate selection, cost estimate, shard status, trace audit, and continuation gates. |
| 2026-07-03 | Expanded the P7 production seed-corpus plan with milestone-level inputs, actions, outputs, and exit gates. |
| 2026-07-03 | Added a source collection readiness gate before production seed-corpus registry import. |
| 2026-07-03 | Verified a small public issue/PR registry import rehearsal and fixed noisy PR checklist task-family inference. |
| 2026-07-03 | Added a reusable registry import rehearsal gate for audited public source records. |
| 2026-07-03 | Added a CI source collection record contract while keeping CI records out of the issue/PR importer. |
| 2026-07-03 | Added a CI registry importer for public workflow failure records. |
| 2026-07-03 | Added an import-rehearsal materialization gate for sampled local workspaces. |
| 2026-07-03 | Added a source-record split gate for issue/PR and CI import-rehearsal shards. |
| 2026-07-03 | Added per-task source export outcomes and retry planning for incomplete collection shards. |
| 2026-07-03 | Added a GitHub token requirement gate for production source collection exports. |
| 2026-07-03 | Added a collection retry runner that executes retry-plan tasks as resumable single-task shards. |
| 2026-07-03 | Added a collection summary merge gate and updated the production source-collection plan to use merged summaries before readiness. |
| 2026-07-03 | Fixed retry planning for auth-gated source collection shards with selected tasks but no task outcomes. |
| 2026-07-03 | Added a source collection preflight gate for plan, shard, token, source, and summary checks before networked exports. |
| 2026-07-03 | Added a deterministic source collection shard schedule gate for production runbooks. |
| 2026-07-03 | Added a source collection shard status gate and updated the detailed production collection plan. |
| 2026-07-04 | Completed authenticated 30-task source collection, clean accepted-output filtering, readiness probes, and issue/PR plus CI import rehearsals. |
