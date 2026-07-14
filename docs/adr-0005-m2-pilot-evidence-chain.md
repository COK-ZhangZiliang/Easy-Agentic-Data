# ADR 0005: M2 Pilot Evidence Chain

## Status

Accepted

## Context

M2 must demonstrate a measured 40-trajectory pilot, not merely provide code that could run one.
The pilot combines paid model calls, private evaluators, mutable workspaces, retries, human review,
and several derived datasets. Independently plausible JSON files are insufficient because they can
refer to different scenarios, configurations, candidate patches, traces, or review samples.

The implementation therefore needs one frozen run identity and a fail-closed chain from each model
request to the final quality decision. Hidden evaluator state must remain available to isolated
verification while staying out of model prompts, public traces, review summaries, and training
exports.

## Decision

### Freeze before execution

`ead pilot create-contract` creates one content-addressed `PilotRunContract`. It binds:

- the exact Gold-20 manifest and all 20 registry scenario hashes;
- two distinct rollout seeds, yielding exactly 40 stable job IDs;
- secret-free hashes of the endpoint, API path, request body, and provider configuration;
- per-agent and total turn, tool-call, token, time, retry, worker, and cost budgets;
- the USD input, cached-input, and output prices used for accounting; and
- hashes of the prompt, tool schemas, evaluator, environment/runtime, and exporter/review code.

Creating a contract is a freeze operation, not a mutable configuration step. A registry, provider
setting, budget, price, prompt, tool, evaluator, runtime, exporter, or review-code change requires a
new contract and a separate artifact root. Credentials remain environment-only and are never part
of the contract.

Before every provider request, the agent derives a conservative input-token upper bound from the
exact ASCII-escaped messages and tool schemas plus a fixed provider-framing margin. It subtracts
that bound from the remaining total-token budget before setting the provider output limit. The
observed call records the bound; strict validation reconstructs it and requires actual input and
output usage to stay within both reservations. The scheduler reserves the full per-agent token
budget and prices it at the most expensive frozen token rate before admitting a job. A provider
that violates a request bound is still accounted in the immutable ledger but cannot produce a
canonical trajectory.

### Separate execution from verification

Each job materializes the fixed source in an agent sandbox. The agent receives the public scenario
and policy-bounded tools but not hidden evaluator state. When the agent stops, the system captures
the complete candidate patch and verifies it in a second, freshly materialized sandbox:

1. rerun setup and health checks;
2. establish the same initial workspace hash;
3. apply only the captured candidate patch;
4. require the resulting state hash to match the agent workspace; and
5. apply and execute the private deterministic evaluators.

A canonical trace is published atomically only after terminal trace replay, clean-reset evaluation,
and leakage checks succeed. Before publication, the strict artifact loader also rebuilds the exact
registry instance and every provider-visible prompt, including tool-role messages, and checks each
request's message count, tool count, and prompt hash against run evidence. The adjacent candidate
patch, private evaluation, and secret-free run evidence are content-bound sidecars. They are
promoted first and the trace is published last as the commit marker. Failed attempts remain under
the private `.attempts` area and are not canonical trajectories. The scheduler accounts for usage,
cost, and elapsed time across retries and retries only infrastructure-classified failures.

Usage accounting has a separate immutable ledger. Each scheduler attempt durably records admission,
provider-call start, provider-call completion, and a terminal outcome using content-addressed
records. Provider-call start is persisted immediately before the request and completion immediately
after the normalized response is available. A started call without a completion makes billable usage
unknown and stops scheduling. Resume may synthesize only an infrastructure-failure terminal when
every started call has a durable completion, or when the scheduler admission precedes any ledger
marker and the absence of canonical artifacts proves that no provider call began. Completed receipts
determine token and cost totals; recovery charges wall time since the admission marker, or the full
per-attempt time reservation when no marker exists. Resume then reconciles the database using
absolute idempotent totals and can recover a terminal published attempt without another provider
request. A terminal attempt whose canonical publication failed remains accounted as an
infrastructure failure. Ledger, job, attempt, and canonical artifact paths must be regular,
non-symlink paths. Provider response identifiers and response hashes are provenance receipts and
uniqueness checks, not third-party cryptographic signatures.

Canonical publication requires an in-process validation receipt bound to the exact trace and all
sidecars. Promotion uses no-overwrite filesystem operations; exact matching interrupted sidecars are
safe to reuse, while conflicting or unsafe paths fail closed. The threat boundary assumes the
process and artifact root are access-controlled: arbitrary local filesystem writes by an actor with
the same permissions are outside this receipt's protection.

### Reproduce accepted successes independently

The first clean verifier run establishes the trajectory outcome. `ead pilot reproduce` performs a
second clean materialization for every successful trajectory and reapplies the exact candidate
patch. It requires the rerun evaluator signature and success state to match the original report.
The public reproduction artifact is bound to private per-job rerun reports in the sibling
`private-reproductions/` directory; a claimed success without that private evidence is rejected.
The final quality command does not merely trust those files: it independently repeats all
successful reruns in a temporary evidence root and requires the fresh semantics, private evidence,
and content hashes to match the declared reproduction.

This is reproduction of the agent candidate, not replay of the retained Gold-20 validated repair.
The M1 repair proves that each seed is solvable; it is never substituted for an M2 candidate.

### Keep review human-authored and bind quarantine to exports

An initial quality report over all 40 canonical trajectories produces leakage, verifier-bypass,
replay, reproduction, duplication, termination, tool-use, and lineage summaries. The deterministic
review selector chooses exactly 20 trajectories, prioritizing critical risks and then increasing
coverage across scenario, repository, language, outcome, and termination strata.

Review decisions are external human inputs. The CLI validates their schema, exact trace set,
uniqueness, timestamps, verdicts, issue codes, and explicit quarantine choices; it does not infer
decisions or prove the real-world identity of a reviewer. The self-contained review gate embeds all
20 canonical decisions and passes only when at least 18 are `acceptable`, no decision is unresolved,
and every `critical` decision is quarantined. Queue items bind the contract, stable job ID, and the
only permitted relative trace path, `<job_id>.jsonl`, so review evidence cannot redirect a reviewer
to another artifact. Export validates the queue, review gate, and content-addressed quarantine set
together before deriving records.

Exports are regenerated from the validated canonical traces, private evaluation lineage,
reproduction evidence, and quarantine set:

- analysis covers all canonical traces and marks quarantine/reproduction state;
- RL excludes quarantined and infrastructure-failed trajectories;
- SFT includes only non-quarantined, hard-verified, clean-reset-reproduced successes; and
- preference records require two eligible trajectories for one scenario and a positive
  deterministic reward margin.

The final quality report recomputes the review queue, validates the embedded review gate, validates
the exact export bytes against their source traces, and requires the review quarantine to equal the
export quarantine. Hand-edited summaries, manifests, counts, hashes, or pass flags do not satisfy
the gate.

### Define the only completion claim

M2 is complete only when the final contract-bound quality report has `passed: true` and every gate
is true. In particular, it requires exactly 40 canonical schema-valid replayable traces, at most 5%
infrastructure failures, zero hidden-content leaks, zero hard-verifier bypasses, reproduction of
every successful candidate, all total budgets respected, valid SFT and preference exports, a
passing 20-item human review, and exact review/export quarantine agreement.

Unit tests, Docker tests, a frozen contract, an enqueued database, a partial run, or a preliminary
quality report demonstrate infrastructure readiness only. None of them is measured M2 evidence.

## Consequences

- Pilot artifacts under `runs/` remain private and ignored; public documentation may describe
  schemas and hashes but must not publish private evaluator or reproduction payloads.
- Operators must retain the contract, scheduler database, canonical traces and sidecars,
  reproduction artifact and private reruns, human decisions, review gate, quarantine set, exports,
  and final quality report as one evidence package.
- Resuming with the same contract and database is supported; silently changing code or configuration
  is not. Version validation rejects drift before provider execution and before reproduction,
  export, quality reporting, or human-review derivation.
- The command order and operational checks are defined in
  [the M2 pilot runbook](m2-pilot-runbook.md).
