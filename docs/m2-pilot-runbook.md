# M2 Pilot Runbook

This runbook produces the evidence required by the M2 exit gate. It assumes the tracked Gold-20
manifest and retained private registry from M1. All outputs belong under ignored `runs/`; do not
commit traces, evaluator sidecars, human notes, credentials, or provider artifacts.

## 1. Preflight and freeze inputs

Finish implementation and documentation changes before creating the contract because the contract
hashes the pilot runtime, evaluator, exporter, and review implementations. Run the required local
gates first:

```bash
export PATH="/opt/homebrew/bin:$PATH"
export PYTHON_BIN="${PYTHON_BIN:-/usr/local/bin/python3.10}"
command -v docker
"$PYTHON_BIN" -c 'import sys; assert sys.version_info >= (3, 10), sys.version'
PYTHONPATH=src "$PYTHON_BIN" -m unittest discover -s tests
PYTHONPATH=src "$PYTHON_BIN" -m unittest tests.test_synthesis_tiers -v
"$PYTHON_BIN" -m pytest
"$PYTHON_BIN" -m ruff check .
EAD_RUN_DOCKER_TESTS=1 PYTHONPATH=src \
  "$PYTHON_BIN" -m unittest tests.test_docker_integration -v
```

Define one new artifact root and the exact M1/provider inputs. The price variables must come from
the provider billing schedule in force when the contract is created; retain that source outside
the public artifacts. The credential itself stays only in the environment named by the provider
configuration.

```bash
export M2_ROOT=runs/m2-pilot
export M2_MANIFEST=manifests/gold-20.json
export M2_REGISTRY=runs/gold-20/registry
export M2_CONFIG=examples/deepseek-v4-pro-thinking.json
export M2_INPUT_USD_PER_MILLION=REPLACE_WITH_CURRENT_PRICE
export M2_CACHED_INPUT_USD_PER_MILLION=REPLACE_WITH_CURRENT_PRICE
export M2_OUTPUT_USD_PER_MILLION=REPLACE_WITH_CURRENT_PRICE
export M2_MAX_TOTAL_COST_USD=REPLACE_WITH_APPROVED_BUDGET
mkdir -p "$M2_ROOT"
```

Confirm the configured credential environment variable is set without printing its value. Do not
put a credential, authorization header, private endpoint, or CA bundle in a tracked file.

Create the immutable contract with every budget explicit:

```bash
PYTHONPATH=src "$PYTHON_BIN" -m easy_agentic_data.cli pilot create-contract \
  --manifest "$M2_MANIFEST" \
  --registry "$M2_REGISTRY" \
  --config "$M2_CONFIG" \
  --output "$M2_ROOT/contract.json" \
  --rollout-seed 0 \
  --rollout-seed 1 \
  --max-agent-turns 20 \
  --max-agent-tool-calls 50 \
  --max-agent-tokens 300000 \
  --max-agent-seconds 1200 \
  --malformed-tool-retries 2 \
  --max-infrastructure-retries 2 \
  --max-workers 1 \
  --max-total-tokens 12000000 \
  --max-total-cost-usd "$M2_MAX_TOTAL_COST_USD" \
  --max-total-seconds 172800 \
  --input-usd-per-million-tokens "$M2_INPUT_USD_PER_MILLION" \
  --cached-input-usd-per-million-tokens "$M2_CACHED_INPUT_USD_PER_MILLION" \
  --output-usd-per-million-tokens "$M2_OUTPUT_USD_PER_MILLION"
```

Do not edit `contract.json`. If any bound input must change, choose a new artifact root and create a
new contract rather than mixing evidence from two run identities.

The two rollout seeds always bind scenario materialization and stable job IDs. They are forwarded
to the provider only when its config declares `seed_request_field`; otherwise the provider binding
must use deterministic sampling (`temperature: 0`), as the retained DeepSeek config does.

## 2. Enqueue and run exactly 40 jobs

```bash
PYTHONPATH=src "$PYTHON_BIN" -m easy_agentic_data.cli pilot enqueue \
  --contract "$M2_ROOT/contract.json" \
  --database "$M2_ROOT/jobs.sqlite3"

PYTHONPATH=src "$PYTHON_BIN" -m easy_agentic_data.cli pilot run \
  --contract "$M2_ROOT/contract.json" \
  --registry "$M2_REGISTRY" \
  --database "$M2_ROOT/jobs.sqlite3" \
  --config "$M2_CONFIG" \
  --trace-directory "$M2_ROOT/traces" \
  --dry-run

PYTHONPATH=src "$PYTHON_BIN" -m easy_agentic_data.cli pilot run \
  --contract "$M2_ROOT/contract.json" \
  --registry "$M2_REGISTRY" \
  --database "$M2_ROOT/jobs.sqlite3" \
  --config "$M2_CONFIG" \
  --trace-directory "$M2_ROOT/traces"
```

The same `pilot run` command may be resumed with the same contract, database, registry,
configuration, and trace directory. It validates runtime hashes before provider execution and
refuses to admit another job when the remaining token, cost, or time budget cannot cover it. Do not
copy canonical files from another run or promote files from `traces/.attempts/` manually.

Each attempt first writes a private staging set. The same strict cross-artifact validator used by
the final quality gate reconstructs its registry instance and provider prompts, then validates the
trace, patch, private evaluation, run evidence, usage, budgets, and scheduler outcome. Only then are
the sidecars promoted and the trace published last as the canonical commit marker.

Provider usage is recorded independently under `traces/.pilot-usage-ledger/`. Every admitted
scheduler attempt has an immutable attempt marker, one durable `call_started`/`call_completed` pair
per provider request, and a terminal record when usage is fully known. Resume reconciles the
scheduler's consumed totals to this ledger with absolute, idempotent updates before planning another
request. A started call without a matching completion has unknown billable usage and blocks dry-run
and resume for manual investigation. An interrupted attempt without a terminal is recovered only as
an infrastructure failure when every started call has a matching completion. If the scheduler marked
an attempt running before its ledger admission marker was created, the absence of canonical artifacts
proves that no provider request began and resume records a zero-call infrastructure failure. Recovery
conservatively charges elapsed wall time since admission, or the full per-attempt time reservation
when no admission timestamp exists. Symlinked or otherwise unsafe ledger and canonical artifact paths
always fail closed.

Before each request, the runtime reserves a conservative upper bound for the exact provider-visible
input and gives only the remaining per-agent token budget to `max_tokens`. The bound is recorded and
reconstructed during strict validation; reported input or output usage outside its reservation
prevents canonical publication. Job admission also reserves the full per-agent token ceiling and
prices it at the highest frozen token rate, so ordinary provider responses cannot cross the total
token or cost ceiling before the scheduler stops.

The terminal ledger record is committed before canonical publication. Therefore a publication
failure still consumes its recorded tokens and cost and remains an infrastructure failure, never a
successful trajectory. If publication completed but the process stopped before updating SQLite,
resume recovers the completed row without another provider attempt. Publication uses a
content-bound validation receipt and never overwrites files: exact matching sidecars may be reused
after an interrupted promotion, but mismatched, symbolic-link, or non-regular artifacts require a
manual audit and a new clean artifact root.

Each completed job must have all four bound artifacts:

```text
traces/<job_id>.jsonl
traces/candidate-patches/<job_id>.patch
traces/private-evaluations/<job_id>.json
traces/run-evidence/<job_id>.json
```

## 3. Reproduce every successful candidate

```bash
PYTHONPATH=src "$PYTHON_BIN" -m easy_agentic_data.cli pilot reproduce \
  --contract "$M2_ROOT/contract.json" \
  --registry "$M2_REGISTRY" \
  --database "$M2_ROOT/jobs.sqlite3" \
  --trace-directory "$M2_ROOT/traces" \
  --output "$M2_ROOT/reproduction.json"
```

The command reruns only successful candidate patches in another clean workspace. Keep
`reproduction.json` beside its generated `private-reproductions/` directory; later commands derive
that private path from the public artifact path and fail if the private evidence is absent or moved.

## 4. Build the review queue and collect human decisions

Generate a preliminary report containing the exact 40 review summaries:

```bash
PYTHONPATH=src "$PYTHON_BIN" -m easy_agentic_data.cli pilot quality-report \
  --contract "$M2_ROOT/contract.json" \
  --registry "$M2_REGISTRY" \
  --database "$M2_ROOT/jobs.sqlite3" \
  --trace-directory "$M2_ROOT/traces" \
  --reproduction "$M2_ROOT/reproduction.json" \
  --output "$M2_ROOT/pre-review-quality.json"
```

Exit status 2 is expected here: export and human-review gates are intentionally absent. Structural,
leakage, bypass, replay, reproduction, or budget failures are not expected and must be investigated
before review.

```bash
PYTHONPATH=src "$PYTHON_BIN" -m easy_agentic_data.cli pilot review-queue \
  --contract "$M2_ROOT/contract.json" \
  --registry "$M2_REGISTRY" \
  --quality-report "$M2_ROOT/pre-review-quality.json" \
  --output "$M2_ROOT/review-queue.json"
```

A human reviewer must inspect all 20 queued trajectories at each item's safe relative
`trace_path` under `$M2_ROOT/traces` and author exactly one JSON or JSONL decision per queued trace.
A canonical record has this shape:

```json
{
  "schema_version": "easy_agentic_data.trajectory_review_decision.v1",
  "trace_id": "trace_...",
  "reviewer_alias": "reviewer-01",
  "timestamp": "2026-07-14T12:00:00Z",
  "verdict": "acceptable",
  "issue_codes": [],
  "notes": "No material trajectory-quality issue found.",
  "quarantine": false
}
```

Valid verdicts are `acceptable`, `minor`, and `critical`; non-acceptable decisions require at least
one issue code. Every critical decision must set `quarantine` to `true`. At least 18 of the 20
decisions must be acceptable, and no issue code may mark a decision unresolved.

```bash
PYTHONPATH=src "$PYTHON_BIN" -m easy_agentic_data.cli pilot review-gate \
  --contract "$M2_ROOT/contract.json" \
  --registry "$M2_REGISTRY" \
  --queue "$M2_ROOT/review-queue.json" \
  --decisions "$M2_ROOT/review-decisions.jsonl" \
  --output "$M2_ROOT/review-gate.json" \
  --quarantine-output "$M2_ROOT/quarantine.json"
```

The command must exit zero. It embeds and revalidates the exact 20 human-authored decisions; it does
not generate verdicts or certify reviewer identity.

## 5. Generate quarantine-aware exports

```bash
PYTHONPATH=src "$PYTHON_BIN" -m easy_agentic_data.cli pilot export \
  --contract "$M2_ROOT/contract.json" \
  --registry "$M2_REGISTRY" \
  --database "$M2_ROOT/jobs.sqlite3" \
  --trace-directory "$M2_ROOT/traces" \
  --reproduction "$M2_ROOT/reproduction.json" \
  --review-queue "$M2_ROOT/review-queue.json" \
  --review-gate "$M2_ROOT/review-gate.json" \
  --quarantine "$M2_ROOT/quarantine.json" \
  --output-directory "$M2_ROOT/exports"
```

This writes `analysis.jsonl`, `rl.jsonl`, `sft.jsonl`, `preference.jsonl`, and `manifest.json`.
The manifest binds exact source trace IDs, quarantine IDs, reproduction hash, file bytes, counts,
skip reasons, and export gates.

## 6. Require the final quality gate

```bash
PYTHONPATH=src "$PYTHON_BIN" -m easy_agentic_data.cli pilot quality-report \
  --contract "$M2_ROOT/contract.json" \
  --registry "$M2_REGISTRY" \
  --database "$M2_ROOT/jobs.sqlite3" \
  --trace-directory "$M2_ROOT/traces" \
  --reproduction "$M2_ROOT/reproduction.json" \
  --export-manifest "$M2_ROOT/exports/manifest.json" \
  --review-gate "$M2_ROOT/review-gate.json" \
  --output "$M2_ROOT/final-quality.json"
```

Completion requires exit status zero, top-level `passed: true`, and every entry in `gates` equal to
`true`. The final command reloads canonical traces and private sidecars, recomputes replay and
contamination checks, independently executes every successful candidate again in fresh workspaces,
validates the new private reruns, and requires their semantics and content hashes to match the
declared reproduction. It also regenerates and byte-compares all exports, recomputes the review
queue and gate, and requires export quarantine to equal review quarantine. A partial report or a
manually edited pass flag is not M2 completion evidence.

Retain the full ignored `$M2_ROOT` evidence package. Update `PLAN.md` to `Complete` only after this
final gate passes and the measured counts and review outcome have been independently inspected.
