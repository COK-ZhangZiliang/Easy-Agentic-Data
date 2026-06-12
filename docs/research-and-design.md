# Synthetic Agent Post-Training Data: Research and Initial Design

> Research date: June 12, 2026. This document focuses on using LLM APIs to generate tasks and
> interactive trajectories for SFT, preference optimization, and agent RL.

## 1. Major Approaches

| Approach | Representative work | Reusable strength | Primary risk |
| --- | --- | --- | --- |
| Instruction bootstrapping | Self-Instruct | Expands a small seed set with inexpensive format and similarity filtering | Teacher bias can be copied while task diversity narrows |
| Difficulty evolution | WizardLM / Evol-Instruct | Adds constraints and reasoning depth incrementally to create a curriculum | Longer prompts are not necessarily harder and may become unsolvable |
| Seedless self-synthesis | Magpie | Extracts diverse user requests from an aligned model's own distribution | Depends on model template behavior that an API may not expose |
| Agent teaching flows | AgentInstruct / MetaSynth | Uses multiple roles for generation, review, and domain specialization | Orchestration is expensive and related judges may share errors |
| Executable tool data | Toolformer / ToolBench / APIGen | Checks tool calls through real execution, improving reliability | API environments, permissions, and side effects are difficult to manage |
| Blueprint-to-dialogue generation | APIGen-MT | Builds grounded task blueprints before simulating and reviewing multi-turn interactions | Trajectory quality is bounded by the user simulator |
| Reasoning and reflection bootstrapping | STaR / self-reflection | Retains successful reasoning or introduces correction trajectories | Unmasked or unlabeled mistakes can contaminate training |
| AI feedback and preferences | Constitutional AI / UltraFeedback | Scales critiques, revisions, rubrics, and preference or reward labels | Judge bias, position bias, and self-preference require calibration |
| Environment-driven task synthesis | SWE-smith and related work | Creates verifiable tasks from real environments with tests or state transitions as objective reward | Environment construction is expensive and tasks may drift from real user demand |
| Multi-turn agent RL | RAGEN / StarPO-style work | Optimizes long-horizon decisions and emphasizes diverse initial states, sampling, and reward granularity | Sparse rewards, reward hacking, and credit assignment remain difficult |
| Distributed synthesis orchestration | Matrix and related systems | Splits generation workflows into message-driven nodes for higher throughput | Infrastructure complexity is unsuitable for the first release |

## 2. Combined Strategy Adopted by This Project

The initial framework combines useful properties instead of reproducing one paper:

1. **Blueprint first**: A task becomes a structured `Task` with explicit difficulty, constraints,
   expected tools, and an optional reference answer.
2. **Generation plus evolution**: Self-Instruct-style generation expands coverage. Evol-style
   mutation adds one verifiable complexity dimension at a time.
3. **Multiple rollouts**: Each task produces several candidates for best-of-N, rejection sampling,
   and preference construction.
4. **Real tool feedback**: Every call passes through `ToolRegistry`; outputs and errors remain
   unchanged in the trajectory.
5. **Layered verification**: Structural and execution checks precede semantic LLM evaluation. Any
   failed or broken verifier sets the reward to zero.
6. **Traceable selection**: All candidates are retained. Only the highest-reward candidate above
   the threshold is exported for SFT.
7. **Informative preferences only**: A chosen/rejected pair requires a positive reward difference.
8. **Training independence**: The generation layer emits generic JSONL that can feed SFT, DPO/IPO,
   reward modeling, or online RL adapters.

## 3. Initial Architecture

```text
Seed Topics / Raw Sources
          |
          v
Task Generator --> Evolver --> Structured Task Blueprints
                                      |
                                      v
                              N x Agent Runner
                                      |
                            Tool / Environment Events
                                      |
                                      v
             Structural -> Execution -> Semantic Verification
                                      |
                                      v
                         Reward + Best-of-N Selection
                              /                 \
                         SFT JSONL       Preference JSONL
```

Module boundaries:

- `llm` handles protocol adaptation, timeouts, and response normalization, not task strategy.
- `generation` creates or transforms task blueprints without executing them.
- `runner` manages model-environment interaction without deciding whether data is trainable.
- `tools` registers capabilities and isolates execution; tool failures become structured data.
- `verification` produces independent, explainable scores and aggregates trajectory reward.
- `pipeline` orchestrates stages, selection, and artifacts without implementing business tools.
- `exporters` maps canonical internal models to training-specific formats.

## 4. Data Quality and Risk Controls

- **Contamination control**: Record seed sources and licenses. Isolate evaluation sets, answers, and
  training sources at the hash level.
- **Diversity**: Measure coverage by domain, skill, difficulty, tool combination, and trajectory
  length, then apply semantic deduplication.
- **Correctness**: Prefer executable signals from environments, tests, or databases. Use an LLM
  judge only as a supplement.
- **Judge calibration**: Use multidimensional rubrics, order reversal, sampled human review, and
  heterogeneous model cross-checks.
- **Reward-hacking resistance**: Preserve each verifier result. Do not allow averaging to offset a
  hard-constraint failure.
- **Privacy and safety**: Do not place secrets, personal data, or content with uncertain licensing
  into prompts or artifacts.
- **Cost governance**: Record model, tokens, latency, and retries for every call. Production runs
  need budgets and termination conditions.

## 5. Roadmap

### M1: Reproducible Local Loop (Current)

- Hosted and local OpenAI-compatible backends plus a mock backend
- Task generation, evolution, tool trajectories, and three-layer verification
- Best-of-N, SFT JSONL, and preference JSONL
- Manifest, LLM call audit ledger, and foundational tests

### M2: Quality and Reliability

- Strict JSON Schema validation and automatic repair
- Content hashes plus embedding-based semantic deduplication
- Rate limits, exponential backoff, checkpoint recovery, call caching, and cost accounting
- Configurable verifier weights, rules, and a human-review sampling queue

### M3: Environment and Strategy Expansion

- Containerized code, browser, database, and business-simulation environments
- User simulators and APIGen-MT-style multi-turn blueprints
- Reflection and correction trajectories with erroneous-step masking
- AgentInstruct-style adapters from documents, code, and other raw sources

### M4: Scale and Training Integration

- Message-queue or Ray backends that preserve the local data contract
- Adapters for SFT, DPO/IPO, reward models, and agent RL frameworks
- Dataset versioning, lineage graphs, offline evaluation, and regression dashboards

## 6. Primary Sources

- Self-Instruct: https://arxiv.org/abs/2212.10560
- WizardLM / Evol-Instruct: https://arxiv.org/abs/2304.12244
- Toolformer: https://arxiv.org/abs/2302.04761
- ToolBench: https://arxiv.org/abs/2307.16789
- Magpie: https://arxiv.org/abs/2406.08464
- AgentInstruct: https://arxiv.org/abs/2407.03502
- APIGen: https://arxiv.org/abs/2406.18518
- APIGen-MT: https://arxiv.org/abs/2504.03601
- MetaSynth: https://arxiv.org/abs/2504.12563
- STaR: https://arxiv.org/abs/2203.14465
- Constitutional AI: https://arxiv.org/abs/2212.08073
- UltraFeedback: https://arxiv.org/abs/2310.01377
- SWE-smith: https://arxiv.org/abs/2504.21798
- RAGEN: https://arxiv.org/abs/2504.20073
- Matrix: https://arxiv.org/abs/2511.21686
