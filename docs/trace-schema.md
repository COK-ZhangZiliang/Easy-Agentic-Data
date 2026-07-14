# Trace Schema and Migration Policy

The canonical trace format is append-only JSONL. Each line contains one complete `TraceEvent`
envelope. Raw traces are immutable once written.

## Version 2 Envelope

```json
{
  "schema_version": 2,
  "event_id": "event_...",
  "session_id": "session_...",
  "sequence": 0,
  "event_type": "session_started",
  "timestamp": "2026-06-12T00:00:00+00:00",
  "payload": {}
}
```

`event_id` is derived from the schema version, session ID, sequence, event type, and payload. The
timestamp is intentionally excluded from the event ID so event identity does not change when a
recording system supplies an equivalent timestamp representation.

A trace ID is derived from the ordered, complete event envelopes loaded from a trace file. A
truncated final JSONL fragment is not part of the trace ID.

## Version 2 Events

- `session_started`
- `system_message`
- `user_message`
- `model_response`
- `tool_requested`
- `policy_decision`
- `tool_started`
- `tool_finished`
- `tool_message`
- `workspace_diff`
- `verification_result`
- `session_finished`

Every event type has required payload fields enforced by the loader. Unknown event types and schema
versions fail closed. Exactly one `system_message` must immediately follow `session_started` in any
non-empty interaction. Its payload contains the exact provider-visible system content and a stable
message ID. Replay and all training exports retain this message; prompt reconstruction reads it from
the trace rather than accepting an unbound external replacement.

Every assistant tool call must have a non-empty ID that is unique across the trace. A
`tool_message` consumes exactly one pending assistant call, must use the same tool name, and cannot
be duplicated. A new `model_response` or `session_finished` is invalid while any assistant tool call
is still awaiting a result. These constraints keep replay, SFT, preference, and RL views aligned
with the provider conversation.

`model_response` may include optional assistant `reasoning_content`. When present, it is treated as
part of the generated agent response and can be carried into SFT, preference, and RL episode
exports. This field remains subject to the same hidden-context and secret-exposure checks as the
visible assistant content. Evaluator-only reasoning and hidden-context reasoning must not be
recorded as agent `model_response` events.

## Public and Hidden Context

Trace events are public, observable interaction records. They must not contain:

- Simulated-user hidden goals or unavailable facts
- Hidden tests or evaluator configuration
- Reference answers or reference patches
- Secrets or provider credentials

`ScenarioInstance.public_view()` is the only scenario representation suitable for public traces.
`TraceRecorder` also rejects distinct hidden-context strings before writing an event. Event payload
validation independently rejects evaluator-only structures such as test patches, reference
artifacts, required or forbidden state, rubrics, and evaluator payload/state fields. Public verifier
projections remain valid because they contain only the verifier label, generic result, hashes,
counts, and other explicitly public summary fields.

## Truncation Behavior

Writers flush and synchronize every event after writing its newline. A reader may therefore recover
all complete events before a partially written final line. Invalid JSON in any complete line is an
error. Strict loading also treats a partial final line as an error.

## Migration Rules

1. Never reinterpret an existing field in place.
2. Add optional fields within the current version only when old readers can safely ignore them.
3. Increment `schema_version` for renamed fields, changed semantics, new required fields, or event
   ordering changes.
4. Implement lossless migrations as pure `vN -> vN+1` transformations that do not call models or
   tools. If a required value was never recorded, fail closed instead of inventing it.
5. Preserve the source trace and write migrated output to a new file with a new trace ID.
6. Add fixtures for every supported source version and tests for chained migration.
7. Reject unsupported older or newer versions rather than guessing their meaning.

## Version 1 Compatibility

Version 1 did not record the provider-visible system message and therefore cannot be migrated to
version 2 losslessly from the trace alone. The version 2 reader rejects version 1. Preserve the
original file and regenerate the trajectory from its reproducible scenario and bound provider
configuration. Supplying an unbound system string during loading is intentionally unsupported
because it would make prompt-lineage validation appear stronger than the source evidence.
