# ADR 0001: Rootless Docker as the First Production Sandbox

- Status: Accepted
- Date: 2026-06-12

## Context

Coding-agent tools execute untrusted model-generated actions. Host subprocess execution does not
provide a sufficient security boundary. The first backend must be common, scriptable, resettable,
and capable of enforcing network and resource limits.

## Decision

Use rootless Docker as the first production sandbox backend.

For local macOS development, use the Docker CLI with a user-scoped Colima VM. The validated setup
uses Apple Virtualization Framework and keeps the Docker daemon inside the Linux VM.

- Images must be pinned by digest.
- Containers run as UID/GID `65532:65532`.
- Network access is disabled by default.
- The root filesystem is read-only.
- Workspaces use a dedicated named volume; arbitrary host bind mounts are prohibited.
- Docker and container-runtime sockets are never mounted.
- CPU, memory, PID, time, output, and workspace limits are mandatory.
- Each rollout recreates or restores its workspace before execution.

`MemorySandbox` exists only for deterministic unit tests and is not a production security boundary.

## Consequences

Docker must be installed and configured for rootless operation on synthesis workers. Machines
without Docker can run unit tests but cannot satisfy container integration acceptance tests.
Podman may be added later behind the same sandbox protocol.
