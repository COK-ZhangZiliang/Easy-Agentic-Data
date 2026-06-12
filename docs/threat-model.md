# Sandbox and Data-Synthesis Threat Model

## Protected Assets

- Host filesystem and credentials
- Container runtime socket
- Hidden tests, reference patches, and evaluator state
- Other rollout workspaces
- Model endpoint credentials and budgets
- Immutable raw traces and dataset lineage

## Trust Boundaries

Model output, imported repositories, generated patches, shell commands, and artifact content are
untrusted. The orchestrator, policy engine, pinned sandbox image, and deterministic evaluator are
trusted components but still require auditable failure handling.

## Primary Threats and Controls

| Threat | Control |
| --- | --- |
| Path traversal and symlink escape | Relative-path validation, isolated volume, no host bind mounts |
| Runtime socket access | Never mount Docker or Podman sockets |
| Network exfiltration | `--network none` by default and policy rejection of network commands |
| Fork bomb or resource exhaustion | PID, CPU, memory, time, output, and workspace limits |
| Oversized or malicious output | Bounded capture and content-addressed artifacts |
| Prompt injection from repository files | Treat repository content as data; capability policy remains authoritative |
| Hidden-answer leakage | Separate context types, forbidden trace fields, canary checks, evaluator isolation |
| Secret leakage | Environment-only credentials, prompt-safe audit logs, redaction scans |
| Artifact poisoning | SHA-256 addressing and integrity verification |
| Trace tampering | Append-only create semantics, event IDs, sequence validation, trace IDs |
| Infrastructure failure mislabeled as task failure | Separate termination and scheduler statuses |

## Required Adversarial Tests

Before enabling a production capability pack, test path traversal, absolute paths, symlink escape,
network commands, runtime socket paths, oversized stdout/stderr, long-running commands, process
explosion, malformed tool arguments, and hidden-context canaries.

