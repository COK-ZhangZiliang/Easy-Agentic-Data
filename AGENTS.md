# AGENTS.md

This file is the development contract for Easy Agentic Data. More specific directory rules may
add constraints, but may not weaken the safety, reproducibility, or data-lineage rules below.

## Mission

Build high-quality agent training trajectories from reproducible tasks, sandboxed execution,
complete traces, and executable verification.

The core pipeline is:

```text
seed -> reproducible workspace -> headless agent -> deterministic evaluation -> trace -> exports
```

The core package does not train models, require one model provider, use an LLM judge in place of
hard checks, or enable tools with external side effects by default.

## Non-Negotiable Contracts

- Target Python 3.10+ and keep the core runtime dependency-light.
- Use English for repository documentation, code comments, docstrings, developer-facing messages,
  commit messages, and pull request descriptions.
- Give every seed, scenario, trace, and derived training record a stable content-based ID.
- Pass and record a seed for every random process. Record model, parameters, usage, latency,
  retries, errors, prompt version, tool version, and environment version in run evidence.
- Keep hidden tests, reference patches, evaluator state, and private user facts outside model
  prompts and public traces.
- Treat executable environment checks as ground truth. A hard verifier failure must make the
  trajectory ineligible for successful-training exports.
- Preserve canonical traces. Selection and export create derived views instead of mutating raw
  trajectories.
- Record source, license, permitted use, split, and contamination metadata for external data.
  Personal data, credentials, and unlicensed data must not enter datasets.
- Read secrets only from environment variables or a secret manager. Never write credentials,
  private endpoints, authorization headers, or CA bundles into tracked files or run artifacts.
- Run tools through the sandbox and policy layer. New side-effecting capabilities require an
  explicit dry-run design, isolation boundary, and tests.
- Do not commit generated data under `runs/`, model weights, caches, or local environment files.

## Change Workflow

1. Read `README.md`, `PLAN.md`, this file, and the modules affected by the change.
2. Keep each change focused. Avoid unrelated formatting, dependency upgrades, or refactors.
3. Add or update tests with the implementation. A bug fix should first reproduce the failure.
4. Run the smallest relevant test set immediately after each functional slice.
5. Run the full default suite and local synthesis smoke before declaring the change complete.
6. Keep code, tests, examples, and documentation consistent. Update `PLAN.md` when scope, status,
   priorities, or exit gates change.

Do not swallow exceptions. Convert expected provider, tool, or infrastructure failures into
contextual states; let programming errors fail fast. Public interfaces should be typed,
deterministically serializable, and documented concisely.

Before adding a dependency, document its purpose, license, size, and why the standard library or
an existing dependency is insufficient. Set a reasonable lower version bound.

## Architecture Boundaries

| Area | Responsibility |
| --- | --- |
| `seeds/`, `environments/`, `scenarios.py` | Public/hidden task contracts and reproducible workspace specifications |
| `registry.py`, source adapters | Validation, provenance, fixed revisions, and scenario materialization |
| `llm/` | Provider protocol, authentication, retries, and response normalization only |
| `agent/`, `coding_tools.py`, `policy.py` | Interaction loop, budgets, sandboxed tools, and policy decisions |
| `sandbox/` | Isolated execution and bounded workspace access |
| `traces/` | Append-only event recording, artifacts, schema validation, and replay |
| `evaluation.py` | Independent deterministic evaluators and hard-failure reward gating |
| `trace_exporters.py` | Derived SFT, preference, RL, and analysis records |
| `batch.py` | Idempotent scheduling, recovery, budgets, and quality reports |

Provider adapters must not contain task or agent policy. The agent must not receive evaluator
payloads. The batch scheduler must not redefine trace or reward semantics.

## Validation

Run for every change:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
PYTHONPATH=src python3 -m unittest tests.test_synthesis_tiers -v
```

After installing development dependencies, also run:

```bash
python3 -m pytest
python3 -m ruff check .
```

When changing Docker, sandbox policy, workspace materialization, or the coding-agent path, run:

```bash
EAD_RUN_DOCKER_TESTS=1 PYTHONPATH=src \
  python3 -m unittest tests.test_docker_integration -v
```

Live provider tests are opt-in, must use environment-provided credentials, and must minimize paid
requests:

```bash
EAD_RUN_LIVE_LLM_TESTS=1 EAD_RUN_DOCKER_TESTS=1 \
DEEPSEEK_API_KEY=... PYTHONPATH=src \
  python3 -m unittest tests.test_live_llm_integration -v
```

## Git and Documentation

- Use Conventional Commits with one logical, testable change per commit.
- Never bypass checks with `--no-verify`, rewrite shared history, or include secrets/generated data.
- `README.md` is the user entry point, `PLAN.md` is the active roadmap, and `docs/` holds stable
  design decisions and detailed contracts.
- Add an ADR only for durable changes to data contracts, execution boundaries, or cross-module
  dependencies. Git history, not `PLAN.md`, is the archive for completed implementation details.
