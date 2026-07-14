<div align="center">

<img src="docs/assets/easy-agentic-data-icon.svg" alt="Easy Agentic Data icon" width="112">

# Easy Agentic Data

**Generate reproducible, executable, and traceable agent training trajectories.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-6B7280)](LICENSE)
[![Status](https://img.shields.io/badge/status-early%20development-F59E0B)](PLAN.md)

</div>

Easy Agentic Data turns a task and a reproducible workspace into a verified interaction trace for
SFT, preference optimization, reward modeling, or agent RL. The project owns the data contract
around the agent: task provenance, hidden evaluator isolation, sandbox policy, event recording,
deterministic verification, and derived exports.

```text
seed + fixed workspace
          |
          v
simulated user <-> headless agent <-> sandboxed coding tools
                                           |
                                           v
                              deterministic evaluation
                                           |
                                           v
                         canonical trace -> SFT / preference / RL
```

## Core Guarantees

- **Executable evidence**: workspace state, hidden tests, and policy checks determine success.
- **Reproducibility**: fixed source revisions, environment specifications, model settings, and
  random seeds are recorded.
- **Complete lineage**: every derived record points back to a canonical append-only trace.
- **Hidden-context isolation**: evaluator tests, reference patches, and private user facts never
  enter model prompts or public traces.
- **Provider neutrality**: hosted and local OpenAI-compatible models share one adapter contract.
- **Safe execution**: coding tools run through a deny-by-default policy inside a bounded sandbox.

The core package does not contain model trainers and does not use an LLM judge as a replacement
for executable verification.

## Quick Start

Run the deterministic local synthesis path. It uses the real headless agent, tool policy, trace,
evaluation, replay, and export contracts without Docker, network access, or a paid model:

```bash
PYTHONPATH=src python3 -m easy_agentic_data.cli synthesis complex-demo \
  --output runs/complex-synthetic-demo
```

The command writes:

- `trace.jsonl`: canonical append-only trajectory;
- `report.json`: deterministic evaluation and reward evidence;
- `sft.json`, `rl_episode.json`, and `analysis.json`: derived training/analysis views;
- `manifest.json`: IDs and artifact paths for the run.

Replay the trace without calling a model or executing tools:

```bash
PYTHONPATH=src python3 -m easy_agentic_data.cli replay \
  --trace runs/complex-synthetic-demo/trace.jsonl
```

Install the CLI and run the default tests:

```bash
python3 -m pip install -e .
PYTHONPATH=src python3 -m unittest discover -s tests
python3 -m ruff check .
```

## Canonical Workflow

### 1. Define or import a scenario

A scenario binds three contracts:

- `QuerySeed`: public task plus provenance, license, split, and optional hidden user state;
- `EnvironmentSpec`: fixed repository revision, image digest, setup, health checks, and limits;
- `HiddenEvaluatorContext`: hidden commands, test patches, required state, and forbidden changes.

The registry validates these contracts and materializes the same initial workspace for every
rollout. Benchmark-derived sources are non-training by default.

```bash
ead registry import \
  --root runs/registry \
  --source examples/public-issue-pr-seeds.jsonl \
  --format public-issue-pr \
  --source-name curated-public-sources \
  --train-eligible auto \
  --allow-train-license Apache-2.0

ead registry validate --root runs/registry
ead registry seed-audit --root runs/registry
```

Example source files use placeholder repositories. Production seeds must point to licensed,
public, fixed revisions and reproducible workspaces.

### 2. Run the headless agent

Provider configuration contains only model-connection settings. Secrets are read from the named
environment variable and never stored in the configuration file.

```bash
ead agent-run \
  --registry runs/registry \
  --scenario-id scenario_... \
  --config examples/local-openai-compatible.json \
  --trace runs/traces/trajectory.jsonl
```

Docker is the production isolation boundary. `MemorySandbox` exists only for deterministic local
tests and demos.

### 3. Verify and export

The evaluator operates after the agent stops and has access to hidden state that the agent never
sees. Hard failures keep a trajectory out of successful-training exports. Raw traces remain
immutable; exporters create new SFT, preference, RL, and analysis records.

Preference pairs require a positive deterministic reward margin. SFT exports include only
hard-verified successful trajectories. RL records retain actions, masks, rewards, and termination
reasons.

### 4. Pilot before scale

Use the persistent scheduler for repeated rollouts only after a small scenario set passes source,
materialization, verifier, leakage, and decontamination checks.

```bash
ead batch enqueue \
  --registry runs/registry \
  --database runs/jobs.sqlite3 \
  --model local-model \
  --config-hash config-v1 \
  --rollouts 2

ead batch run \
  --registry runs/registry \
  --database runs/jobs.sqlite3 \
  --config examples/local-openai-compatible.json \
  --trace-directory runs/traces

ead batch report \
  --database runs/jobs.sqlite3 \
  --trace-directory runs/traces \
  --output runs/quality-report.json
```

Scale decisions should use replay validity, executable success, infrastructure-failure rate,
hidden-content leakage, duplicate rate, coverage, reward distribution, cost, and stratified human
review. A large source collection is not evidence of trajectory quality.

## Data Quality Gates

A production trajectory is eligible only when:

1. the task has licensed provenance and a fixed, reproducible workspace;
2. public task text is separated from hidden user and evaluator context;
3. the trace is schema-valid, complete, and deterministically replayable;
4. all required hard evaluators pass after a clean reset;
5. no credential, personal data, benchmark oracle, or held-out evaluator content leaks;
6. the derived record preserves trace, scenario, model, prompt, tool, and verifier lineage;
7. corpus-level duplication, balance, contamination, and sampled human quality gates pass.

## Architecture

```text
src/easy_agentic_data/
├── seeds/, environments/, scenarios.py  # task and workspace contracts
├── registry.py                          # validation and materialization
├── agent/                               # headless interaction loop and budgets
├── coding_tools.py, policy.py           # sandboxed coding capabilities
├── sandbox/                             # Docker and in-memory backends
├── traces/                              # append-only events, artifacts, replay
├── evaluation.py                        # deterministic evaluators and rewards
├── trace_exporters.py                   # SFT, preference, RL, analysis views
├── simulation.py                        # optional multi-turn user simulation
├── batch.py                             # scheduling, recovery, quality reports
└── registry_sources.py, seed_corpus.py  # source adapters and corpus quality gates
```

The scenario, trace, and evaluator contracts are the source of truth. Model providers, agent
implementations, seed sources, and training frameworks are replaceable adapters.

## Provider Configuration

Use `examples/openai-compatible.json` for hosted APIs and
`examples/local-openai-compatible.json` for a local endpoint. Hosted providers fail fast when the
configured credential environment variable is missing. Local providers may set `api_key_env` to
`null`.

Paid live tests are opt-in:

```bash
EAD_RUN_LIVE_LLM_TESTS=1 EAD_RUN_DOCKER_TESTS=1 \
DEEPSEEK_API_KEY=... PYTHONPATH=src \
  python3 -m unittest tests.test_live_llm_integration -v
```

## Migration from the Legacy Demo

The calculator-oriented `ead run` path and its separate task, runner, verifier, and exporter
models were removed. Use `ead synthesis complex-demo` for a local contract smoke and `ead
agent-run` or `ead batch run` for registry-backed trajectories. Provider JSON files now contain
only an `llm` object; rollout selection, budgets, and output paths belong to the command that runs
the scenario. Canonical task and trajectory APIs are `Scenario` and append-only `Trace`.

## Documentation

- [PLAN.md](PLAN.md): active priorities and measurable exit gates
- [AGENTS.md](AGENTS.md): development, safety, and validation contract
- [Trace schema](docs/trace-schema.md): canonical event and migration rules
- [Threat model](docs/threat-model.md): protected assets and trust boundaries
- [Docker sandbox ADR](docs/adr-0001-docker-sandbox.md)
- [RL export ADR](docs/adr-0002-rl-episode-export.md)
- [Synthesis path ADR](docs/adr-0003-synthesis-tiers.md)
- [Gold-20 freeze manifest ADR](docs/adr-0004-gold20-freeze-manifest.md)

## Status

The canonical local trajectory path is implemented and tested. M1 is complete: the Gold-20
executable seed set and its metadata-only manifest are frozen with reproducibility, verifier,
decontamination, and production DockerSandbox evidence. M2 still requires registry-backed
rollouts, end-to-end export wiring, and a measured provider pilot. See [PLAN.md](PLAN.md) for the
current sequence.

## License

Licensed under the [Apache License 2.0](LICENSE).
