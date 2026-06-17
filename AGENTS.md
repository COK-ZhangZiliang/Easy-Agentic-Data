# AGENTS.md

This file is the ongoing development contract for Easy Agentic Data. Every human contributor and
automated agent working in this repository must follow it. A directory may define a more specific
`AGENTS.md`, but local rules must not weaken the safety, testing, or data-lineage requirements in
this file.

## 1. Project Goals and Non-Goals

The project uses replaceable LLM APIs, task-generation strategies, and executable environments to
produce reproducible, verifiable, and traceable agent conversations and trajectories for SFT,
preference optimization, reward modeling, and agent RL post-training.

Current non-goals:

- Do not implement model trainers in the core package.
- Do not make one model provider or training framework a mandatory dependency.
- Do not use an LLM judge as a substitute for executable verification.
- Do not execute tools with external side effects by default.
- Do not commit generated files under `runs/`.

## 2. General Development Rules

1. Read the root `README.md`, `PLAN.md`, this file, and the relevant modules before changing code.
2. Keep changes narrowly scoped. Do not include unrelated refactors, formatting, or dependency
   upgrades.
3. Use English for all repository documentation, code comments, docstrings, user-facing developer
   messages, commit messages, and pull request descriptions.
4. Target Python 3.10+ and keep the core runtime dependency-light.
5. Define typed public data structures that serialize deterministically to JSON.
6. Every random process must accept and record a seed. Document irreproducible behavior in the run
   manifest.
7. Every LLM call must expose the model, parameters, usage, latency, retry count, and errors. Audit
   logs store prompt hashes by default unless a run explicitly permits sanitized prompt text.
8. Read secrets only from environment variables or a secret manager. Never place secrets in
   configuration files, logs, tests, or fixtures.
9. Never attempt to upload, commit, push, publish, or otherwise transmit API keys, access tokens,
   private API URLs, authorization headers, credentials, or other sensitive connection details.
   Use sanitized placeholders in all tracked files and user-visible artifacts.
10. Do not swallow exceptions. Convert recoverable model and tool failures into contextual states;
   fail fast on programming errors.
11. Before adding a dependency, document its purpose, license, size, and alternatives, then set a
    reasonable lower version bound.
12. Comments should explain constraints and reasoning rather than restate code. Public interfaces
    should have concise docstrings.
13. After completing each functional slice, run the smallest relevant test set immediately. Do not
    start the next slice or mark plan work complete until those tests pass. Run the full regression
    suite again at the end of each milestone.
14. If `REVIEW.md` exists at the repository root, read and follow its additional local
    development instructions. This file is intentionally excluded from version control.
15. After completing any code change and its relevant tests, perform the review workflow defined
    in `REVIEW.md`. Address every actionable finding and rerun affected tests before treating
    the change as complete. If the local file does not exist, do not invent a replacement command.

## 3. Data and Experiment Standards

- Export stages must not mutate raw trajectories in place. Create a new view or file for selected
  data.
- Every task requires a stable `task_id`; every trajectory requires a stable `trajectory_id`.
- Data artifacts must trace back to run configuration, prompt version, model, tool version, and
  verifier results.
- Prefer executable signals such as environment state, tests, or database state as ground truth.
  Model scores are soft signals.
- A hard-constraint failure must not receive a high reward through weighted averaging.
- Every preference pair needs an explicit chosen/rejected criterion and a positive margin.
- Do not expose hidden evaluation answers to generation models. Keep auditable separation between
  training and evaluation sources.
- Record the source, license, and permitted use of external data. Personal data and credentials
  must not enter datasets.
- Before releasing production data, inspect format validity, duplication, category distribution,
  difficulty distribution, tool success rate, reward distribution, sampled human quality, and
  possible evaluation contamination.

## 4. Module Guidance

### `models.py`

- Define cross-module data contracts without network, file, or business-execution logic.
- New fields must be backward compatible or include an explicit migration note.
- Derive IDs from stable semantic content, never from Python process hashes.

### `config.py`

- Configuration must be explicit, have defaults, and be serializable into a manifest.
- Secret configuration stores only environment-variable names.
- New settings require an example and boundary validation.

### `llm/`

- Provider adapters handle protocol details, authentication, timeouts, retries, and response
  normalization only.
- Hosted providers must fail fast when required credentials are absent. Local providers may omit
  authentication but must never invent or persist placeholder secrets.
- Do not embed task-generation or verifier prompts in provider adapters.
- Raw provider responses may have an optional audit path, but logs must be sanitized.

### `generation.py`

- Task generation and evolution must emit structured blueprints.
- Evolution must preserve solvability and introduce one explainable complexity dimension per round.
- Prompt changes are data-behavior changes and require tests plus a note in the commit description.

### `runner.py` and `tools.py`

- The runner manages interaction state and termination conditions only.
- Every tool declares a JSON Schema. The minimal implementation currently rejects malformed
  arguments through JSON-object parsing and handler signatures; add complete schema validation
  before introducing tools with side effects.
- Default tools must not access arbitrary files, networks, shells, or real business write APIs.
- Tools with side effects must be explicitly marked, isolated, and support dry runs.
- Write tool errors back as observations. Never represent them as successful output.

### `verification.py`

- Verifiers remain independent and return `passed`, `score`, and an auditable `reason`.
- Add deterministic verifiers before introducing an LLM judge.
- Reward aggregation changes require tests for hard failures, boundary scores, and stable ranking.

### `pipeline.py`

- The pipeline orchestrates stages, selection, and artifact persistence only.
- Stages should be replaceable and compatible with future checkpoint recovery. Do not embed
  business-specific tools in the pipeline.
- Preserve every candidate trajectory. Training exports are selected subsets.

### `exporters.py`

- Exporters must not mutate internal models.
- Every training format requires a small contract test.
- If a training framework's field semantics are uncertain, isolate them in a dedicated adapter
  instead of changing the core schema.

### `tests/`

- Unit tests must not call paid APIs, depend on public networks, or read user secrets.
- Use mocks or small, readable, sanitized recorded fixtures.
- Bug fixes should first add a test that reproduces the issue.

## 5. Testing and Definition of Done

Run at least:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
PYTHONPATH=src python3 -m easy_agentic_data.cli run --config examples/minimal.json
```

After installing development dependencies, also run `pytest` and `ruff check .`. Expand validation
for these changes:

- Data schema: serialization, compatibility, and export tests
- Provider: success, timeout, rate-limit, invalid-response, and sanitization tests
- Tool: argument boundaries, exceptions, and side-effect isolation tests
- Reward or selection: ranking, thresholds, ties, and hard-failure tests
- Concurrency or recovery: idempotency, duplicate execution, and partial-failure tests

When changing the Docker sandbox or coding-agent execution path, run:

```bash
EAD_RUN_DOCKER_TESTS=1 PYTHONPATH=src python3 -m unittest tests.test_docker_integration -v
```

This test requires a running Docker daemon and the pinned integration images documented in the
test module.

Live provider tests must be opt-in, use environment variables for credentials, minimize paid
requests, and clean up temporary artifacts. Run the DeepSeek pipeline and Docker-agent checks with:

```bash
EAD_RUN_LIVE_LLM_TESTS=1 EAD_RUN_DOCKER_TESTS=1 \
DEEPSEEK_API_KEY=... PYTHONPATH=src \
  python3 -m unittest tests.test_live_llm_integration -v
```

Never enable paid live tests in default CI or commit provider credentials, CA bundles, or live run
artifacts.

A change is done only when code, tests, examples, documentation, and configuration agree, and no
temporary artifacts or secrets are included.

## 6. Git Commit Rules

- Use branch names such as `codex/<topic>`, `feature/<topic>`, `fix/<topic>`, or `docs/<topic>`.
- Each commit should express one logical, testable change.
- When asked to commit a set of changes, separate them into multiple commits by functional scope
  instead of combining unrelated features, fixes, tests, or documentation into one commit.
- After completing user-requested commits, push them to the configured remote unless the user
  explicitly asks to keep them local or no remote is available.
- Use Conventional Commits:

```text
feat(generation): add blueprint diversity filter
fix(runner): preserve tool errors in trajectory
test(verification): cover hard-check reward gating
docs(architecture): explain environment boundary
```

- Recommended types are `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, and `perf`.
- Write subjects in English, imperative mood, and lowercase; omit the final period and keep them
  within 72 characters when practical.
- For behavioral changes, explain motivation, data impact, compatibility, and verification in the
  commit body.
- Do not commit `.env`, API keys, `runs/`, model weights, unlicensed data, or large generated
  artifacts.
- Do not bypass checks with `--no-verify` or rewrite history shared by other contributors.
- Pull request descriptions must cover the problem, solution, data or schema impact, test results,
  cost or safety impact, and rollback plan.

## 7. Architecture Decisions and Documentation

- Update `PLAN.md` in the same change when a tracked task is completed, added, removed, blocked, or
  materially re-scoped. Mark work complete only after its exit criteria and required tests pass.
- Add a short ADR under `docs/` when a change affects data contracts, the execution model, or a
  long-term dependency across two or more modules.
- Document the source, adopted ideas, and differences for every new generation strategy.
- The README serves first-time users, PLAN tracks milestones, AGENTS defines development
  constraints, and detailed research or design belongs under `docs/`.
- A disagreement between code and documentation is a defect and must be corrected in the same
  change.
