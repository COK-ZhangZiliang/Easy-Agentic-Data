# ADR 0002: RL Episode Export Contract

## Status

Accepted

## Context

Multi-turn agent reinforcement learning needs more than a replayable conversation transcript. A
trainer must know which steps are user or environment observations, which steps are assistant
actions, which tokens are eligible for loss, and how outcome and turn-level rewards attach to the
episode. The canonical trace format already records the observable event stream and remains the
source of truth for replay and audit.

## Decision

The project will derive `easy_agentic_data.rl_episode.v1` records from canonical traces instead of
changing trace schema version 1. The derived record contains ordered steps with role, step type,
assistant action type, loss mask, action mask, reward components, termination reason, and source
trace linkage. Token offsets remain optional in this contract and may be filled by tokenizer-aware
training adapters.

Turn-level rewards are deterministic first. The initial reward evidence covers policy denials,
tool execution success or failure, and information-gathering actions such as `ask_user`. Outcome
reward remains the primary success signal and is attached separately from turn reward.

## Consequences

- Existing trace replay remains backward compatible.
- Training frameworks can consume explicit action and loss masks without parsing free-form text.
- Reward shaping can evolve independently from the immutable trace.
- Tokenizer-specific span generation stays outside the core trace contract until a concrete
  training adapter needs it.
