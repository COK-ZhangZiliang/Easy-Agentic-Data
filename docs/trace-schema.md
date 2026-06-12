# Trace Schema and Migration Policy

The canonical trace format is append-only JSONL. Each line contains one complete `TraceEvent`
envelope. Raw traces are immutable once written.

## Version 1 Envelope

```json
{
  "schema_version": 1,
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

## Version 1 Events

- `session_started`
- `user_message`
- `model_response`
- `tool_requested`
- `policy_decision`
- `tool_started`
- `tool_finished`
- `workspace_diff`
- `verification_result`
- `session_finished`

Every event type has required payload fields enforced by the loader. Unknown event types and schema
versions fail closed.

## Public and Hidden Context

Trace events are public, observable interaction records. They must not contain:

- Simulated-user hidden goals or unavailable facts
- Hidden tests or evaluator configuration
- Reference answers or reference patches
- Secrets or provider credentials

`ScenarioInstance.public_view()` is the only scenario representation suitable for public traces.
`TraceRecorder` also rejects distinct hidden-context strings before writing an event.

## Truncation Behavior

Writers flush and synchronize every event after writing its newline. A reader may therefore recover
all complete events before a partially written final line. Invalid JSON in any complete line is an
error. Strict loading also treats a partial final line as an error.

## Migration Rules

1. Never reinterpret an existing field in place.
2. Add optional fields within the current version only when old readers can safely ignore them.
3. Increment `schema_version` for renamed fields, changed semantics, new required fields, or event
   ordering changes.
4. Implement migrations as pure `vN -> vN+1` transformations that do not call models or tools.
5. Preserve the source trace and write migrated output to a new file with a new trace ID.
6. Add fixtures for every supported source version and tests for chained migration.
7. Reject versions newer than the current reader rather than guessing their meaning.

No migration is currently required because version 1 is the initial schema.

