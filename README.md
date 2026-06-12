# Easy Agentic Data

A composable framework for synthesizing agent post-training data through LLM APIs. The first
release focuses on a complete data-generation loop:

`task blueprint -> difficulty evolution -> repeated agent rollouts -> tool execution -> layered verification -> best-of-N -> SFT/preference exports`

## Design Principles

- **Verification first**: Structure, tool execution, and semantic quality are checked separately.
  A failure cannot be hidden by other scores.
- **Environment feedback is ground truth**: Tool outputs remain in the complete trajectory and
  cannot be rewritten by a judge.
- **End-to-end traceability**: Tasks, trajectories, verification results, rewards, configuration,
  and timestamps are written to the run artifacts.
- **Composable strategies**: Task generation, evolution, runners, tools, verifiers, and exporters
  have explicit boundaries.
- **Training-framework independence**: The project emits standard JSONL instead of depending on a
  specific SFT, DPO, or online RL framework.

## Quick Start

The current implementation only requires the Python 3.10+ standard library.

```bash
PYTHONPATH=src python3 -m easy_agentic_data.cli run --config examples/minimal.json
PYTHONPATH=src python3 -m unittest discover -s tests
```

Artifacts are written to `runs/local-demo/`:

- `manifest.json`: Complete configuration and run statistics
- `llm_calls.jsonl`: Prompt hashes, parameters, usage, latency, and status without prompt text
- `tasks.jsonl`: Generated and evolved task blueprints
- `trajectories.jsonl`: Raw trajectories with tool events and verification results
- `sft.jsonl`: The highest-reward trajectory above the threshold for each task
- `preferences.jsonl`: Chosen/rejected pairs with a positive reward margin

Canonical append-only traces can be replayed without invoking a model or executing a tool:

```bash
PYTHONPATH=src python3 -m easy_agentic_data.cli replay --trace path/to/trace.jsonl
```

Scenario registry and sandboxed agent commands:

```bash
ead registry validate --root registry/
ead registry list --root registry/
ead agent-run --registry registry/ --scenario-id scenario_... \
  --config examples/local-openai-compatible.json --trace runs/agent.jsonl
ead batch enqueue --registry registry/ --database runs/jobs.sqlite3 \
  --model local-model --config-hash config-v1 --rollouts 4
ead batch run --registry registry/ --database runs/jobs.sqlite3 \
  --config examples/local-openai-compatible.json --trace-directory runs/traces
```

`agent-run` and `batch run` require rootless Docker. Unit tests use `MemorySandbox`, which is not a
production security boundary.

On macOS, the validated local runtime is Docker CLI with Colima:

```bash
brew install docker colima
colima start --cpu 2 --memory 4 --disk 20 --vm-type vz
EAD_RUN_DOCKER_TESTS=1 PYTHONPATH=src \
  python3 -m unittest tests.test_docker_integration -v
```

To use an OpenAI-compatible API:

```bash
export OPENAI_API_KEY=...
PYTHONPATH=src python3 -m easy_agentic_data.cli run \
  --config examples/openai-compatible.json
```

## Local LLM API

Use the `local_openai_compatible` provider for a local or private server that exposes an
OpenAI-compatible chat-completions endpoint, such as a vLLM, SGLang, llama.cpp, or compatible
Ollama deployment:

```bash
PYTHONPATH=src python3 -m easy_agentic_data.cli run \
  --config examples/local-openai-compatible.json
```

The relevant configuration is:

```json
{
  "provider": "local_openai_compatible",
  "model": "Qwen/Qwen3-8B",
  "base_url": "http://127.0.0.1:8000/v1",
  "api_key_env": null,
  "chat_completions_path": "/chat/completions"
}
```

Authentication is optional for this provider. Set `api_key_env` to an environment-variable name
when a local or private endpoint requires a Bearer token. `base_url` may target localhost or a
private network address, and `chat_completions_path` can be changed for a non-default route.

Agent rollouts send OpenAI function-tool schemas. The local model server and chat template must
support tool calling for tool-use trajectories to pass execution verification. Task generation and
semantic judging only require standard chat completions and valid JSON output.

The package can also be installed for development:

```bash
python3 -m pip install -e '.[dev]'
ead run --config examples/minimal.json
pytest
```

## Project Structure

```text
src/easy_agentic_data/
  config.py          # Run configuration
  models.py          # Task, message, trajectory, verification, and preference contracts
  seeds/             # Query seeds and public/hidden user context contracts
  environments/      # Reproducible environment specifications
  scenarios.py       # Scenario binding and materialized scenario instances
  traces/            # Versioned events, artifacts, append-only recording, and replay
  llm/               # Hosted/local OpenAI-compatible clients and the mock backend
  generation.py      # Self-Instruct-style generation and Evol-style mutation
  tools.py           # Tool registration, schema exposure, and controlled execution
  runner.py          # Multi-turn agent rollouts with tool calls
  verification.py    # Layered structural, execution, and semantic verification
  exporters.py       # Raw trajectory, SFT, and preference exports
  pipeline.py        # Local orchestration and best-of-N selection
```

See [docs/research-and-design.md](docs/research-and-design.md) for the research basis and
architecture decisions. See [PLAN.md](PLAN.md) for implementation milestones and progress, and
[AGENTS.md](AGENTS.md) for ongoing development rules.

## Current Scope

The first release provides a synchronous, single-process generation loop for hosted and locally
deployed OpenAI-compatible APIs.
Distributed scheduling, containerized environments, browser or code sandboxes, model-based
deduplication, standalone reward models, and training-framework adapters are planned milestones.
