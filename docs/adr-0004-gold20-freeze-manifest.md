# ADR 0004: Gold-20 Freeze Manifest

## Status

Accepted

## Context

Milestone M1 requires a fixed set of 20 train-eligible repair tasks. A registry count alone cannot
prove that the tasks are reproducible, that their private verifiers exercise the reported defect,
or that a known repair satisfies the verifier. Copying evaluator payloads into a public manifest
would also violate the repository's hidden-state boundary.

## Decision

Gold-20 is frozen through `ead registry freeze-gold-20 --config ...`. The command accepts an exact
20-scenario registry and independently produced evidence artifacts. It fails closed unless all of
the following conditions hold:

- the scenario, seed, source-record, rehearsal, reset, and repair-validation sets are exactly equal;
- every source revision and content ID is fixed; source identity, canonical GitHub URL, public task
  text, workspace URI, image, setup recipe, and health recipe are bound across the source snapshot
  and registry; provenance is licensed for training and permitted use is declared;
- the corpus covers at least eight repositories and two languages;
- every environment has a content-addressed image, a nonempty health check, and at least two reset
  attempts with the same workspace tree hash; each attempt records zero setup and health-check
  exit codes for the declared recipes;
- every hidden test patch passes `git apply --check`, applies to the original source, and fails
  there; an exact 20-record private reference-repair bundle retains the repair bytes, recomputes
  their hashes, and binds each repair to the scenario, seed, environment, source instance, and
  revision; the repair then applies successfully, the same hidden test patch applies to the
  repaired source, and its test passes;
- `git apply --check` for the validated repair and for the hidden test patch on the repaired source
  is optional corroborating evidence; when either result is recorded, its exit code must be zero,
  while the corresponding successful applies and repaired test pass remain mandatory;
- repair validation names the same materialized tree, image, setup recipe, and health-check recipe;
- production `DockerSandbox` replay covers the exact 20 scenarios in both their base and repaired
  forms: all setup and health checks pass, each base hidden verification fails, each validated
  repair applies, and each repaired hidden verification passes; replay also proves the declared
  content-addressed image, resource limits, non-root user, disabled network, read-only root
  filesystem, writable workspace volume, bounded temporary filesystem, and absence of a mounted
  Docker socket;
- the holdout basis contains benchmark scenarios whose stored test patches match their declared
  hashes; seed- and scenario-level decontamination are recomputed with no unresolved findings;
  warning-only findings may be closed only by an exact audit/code/entry resolution with a written
  rationale, while error findings cannot be waived; and
- private evaluator strings do not occur in their own or any other Gold-20 public task projection.

The resulting `easy_agentic_data.gold20_manifest.v1` artifact is metadata-only. It records stable
IDs, public provenance metadata, coverage counts, exit-derived validation state, workspace tree
and canonical-origin hashes, private reference-repair bundle and per-repair hashes, and counts plus
hashes for resolved and unresolved audit findings. Public source URLs are validated against the
canonical GitHub issue or pull-request identity, then represented by a hash. Resolution rationales
and finding entry IDs are also represented only by hashes. The manifest does not record raw source
URLs, hidden test patches, hidden commands, reference repairs, output text, private evaluator
references, workspace paths, input artifact paths, or audit rationales. The corpus ID is derived
from the corpus, all evidence snapshots, the container replay artifact, both runtime build-spec
hashes, the holdout basis, and audit policy; creation time is informational and excluded from
identity. Only a valid freeze is atomically published to the configured manifest path; an invalid
attempt returns diagnostics without replacing the last valid manifest.

### Verifier runtime images

The production replay used two local, content-addressed verifier runtimes. Each runtime is paired
with a repository-tracked reconstruction specification:

- [`containers/gold20-python-runtime.Dockerfile`](../containers/gold20-python-runtime.Dockerfile)
  is paired with `sha256:6ff121957fe47796fa3d1d064d56279526b596e7379acfad0b18112d77bfe186`
  (88,963,618 bytes).
- [`containers/gold20-node-runtime.Dockerfile`](../containers/gold20-node-runtime.Dockerfile)
  is paired with `sha256:27e4fd2dedd97f542e3d3af73c61f9a6e5388e50985828a60d43a171ec3f0d90`
  (112,535,387 bytes).

The Python image contains the exact libraries needed by the selected Python workspaces and their
hidden verifiers. Installed sizes below are distribution-record file sizes measured in the
retained image; they exclude shared base-image and filesystem overhead.

| Package | Purpose in Gold-20 verifier | License | Installed KiB |
| --- | --- | --- | ---: |
| `anyio` | selected AnyIO repository API | MIT | 1,080.2 |
| `attrs` | selected attrs repository API | MIT | 431.8 |
| `certifi` | CA data required by the HTTP client dependency closure | MPL-2.0 | 239.9 |
| `charset-normalizer` | Requests character-set dependency | MIT | 1,372.1 |
| `click` | selected Click repository API | BSD-3-Clause | 873.3 |
| `h11` | HTTPX HTTP/1.1 protocol dependency | MIT | 181.1 |
| `httpcore` | HTTPX transport dependency | BSD-3-Clause | 611.4 |
| `httpx` | selected HTTPX repository API | BSD-3-Clause | 619.9 |
| `idna` | HTTP client hostname dependency | BSD-3-Clause | 479.7 |
| `iniconfig` | pytest configuration parser | MIT | 36.2 |
| `markdown-it-py` | Rich Markdown dependency | MIT | 488.7 |
| `mdurl` | Markdown URL parser dependency | MIT | 39.1 |
| `packaging` | pytest version and marker handling | Apache-2.0 OR BSD-2-Clause | 753.4 |
| `pluggy` | pytest plugin engine | MIT | 133.9 |
| `Pygments` | Rich syntax-highlighting dependency | BSD-2-Clause | 8,176.1 |
| `pytest` | hidden Python test runner | MIT | 2,929.0 |
| `requests` | selected Requests repository API | Apache-2.0 | 466.7 |
| `rich` | selected Rich repository API | MIT | 2,526.7 |
| `typing_extensions` | compatibility dependency for selected packages | PSF-2.0 | 341.3 |
| `urllib3` | Requests transport dependency | MIT | 873.6 |

The 20 pinned Python distributions occupy about 22.1 MiB by this measure. Both images also include
Git, licensed GPL-2.0-only, because the isolated evaluator must check and apply withheld test and
repair patches. The pinned Git package plus its core executables occupy about 28.7 MiB in the
Python image and 26.6 MiB in the Node image. The complete declared Python dependency closure is
version-pinned in the Dockerfile.

Python's standard library cannot reproduce these third-party project semantics, and the slim base
images do not contain the test runner, project dependencies, or Git needed for executable patch
verification. These packages belong only to the isolated verifier runtimes; they are not added as
dependencies of the Easy Agentic Data core package.

The recorded image IDs, sizes, declared runtime-spec hashes, and 20/20 base-fail/fixed-pass replay
were produced on `linux/arm64`. The replay records
`local_image_id_plus_declared_spec`: it proves which local image content ID executed and separately
binds the tracked specification, but it does not claim BuildKit or SLSA provenance proving that the
image was built from that specification. The pinned package versions and base image digests make
the specifications reviewable reconstruction recipes; package repositories and build provenance
are not frozen sufficiently for a byte-identical rebuild guarantee. These facts establish the
current private local pilot on one platform, not a portable multi-architecture image release. The
container evidence file hash and both tracked Dockerfile hashes are bound into the corpus ID, so
changing a runtime, declared specification, or replay evidence creates a different Gold-20
identity.

Gold-20 is a private local pilot, not a distributable dataset release. Its ignored evidence store
must retain the source-cache mirrors, hidden patches, reference-repair bundle, host and container
replay validation, and holdout registry named by the manifest hashes. The canonical public Git
origin, fixed revision, and materialized tree hash are all bound, but the current local materializer
consumes `file://` cache mirrors. The tracked manifest and runtime Dockerfiles therefore verify a
retained private corpus; they do not reconstruct that corpus on another machine. Portable
content-addressed packaging remains part of the immutable release work in M3.

## Consequences

- A validated repair is evidence that a verifier is satisfiable, while the hidden test patch
  defines the private behavior checked on arbitrary agent outputs; neither substitutes for the
  other.
- Materialization and repair validation remain separate executable processes. The freeze command
  verifies their recorded, content-bound evidence instead of executing untrusted repositories.
  Aggregate repair evidence must declare its schema, exact valid/invalid counts, and valid state;
  the private repair bytes are independently rehashed rather than trusted from that aggregate.
- Container replay remains a separate production-sandbox process. The freeze command requires an
  exact 20-record, 20-valid artifact and binds its producer, sandbox implementation, runtime build
  specs, scenario identities, image identities, materialized trees, evaluator hashes, and exit
  outcomes without publishing private verifier contents.
- Any task, source revision, evaluator, environment recipe, repair, or evidence change creates new
  hashes and therefore a new corpus identity.
- Incomplete evidence can be inspected, but it cannot produce a valid Gold-20 manifest.
