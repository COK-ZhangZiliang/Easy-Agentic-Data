from __future__ import annotations

from typing import Any, Dict, Iterable, List

from easy_agentic_data.evaluation import EvaluationReport
from easy_agentic_data.traces import EventType, Trace, replay_trace


def trace_to_sft(trace: Trace, report: EvaluationReport) -> Dict[str, Any]:
    if not report.success:
        raise ValueError("Only successful traces can be exported for SFT")
    replay = replay_trace(trace)
    return {
        "id": f"sft_{trace.trace_id}",
        "trace_id": trace.trace_id,
        "scenario_instance_id": report.scenario_instance_id,
        "messages": replay.state.messages,
        "reward": report.reward,
    }


def traces_to_preference(
    chosen: Trace,
    chosen_report: EvaluationReport,
    rejected: Trace,
    rejected_report: EvaluationReport,
) -> Dict[str, Any]:
    margin = chosen_report.reward - rejected_report.reward
    if margin <= 0:
        raise ValueError("Preference pairs require a positive deterministic reward margin")
    return {
        "id": f"preference_{chosen.trace_id}_{rejected.trace_id}",
        "scenario_instance_id": chosen_report.scenario_instance_id,
        "chosen_trace_id": chosen.trace_id,
        "rejected_trace_id": rejected.trace_id,
        "chosen": replay_trace(chosen).state.messages,
        "rejected": replay_trace(rejected).state.messages,
        "margin": margin,
    }


def trace_to_rl_episode(trace: Trace, report: EvaluationReport) -> Dict[str, Any]:
    steps: List[Dict[str, Any]] = []
    for event in trace.events:
        if event.event_type in {
            EventType.USER_MESSAGE,
            EventType.MODEL_RESPONSE,
            EventType.TOOL_REQUESTED,
            EventType.TOOL_FINISHED,
        }:
            steps.append(
                {
                    "event_type": event.event_type.value,
                    "observation_or_action": event.payload,
                    "reward": 0,
                    "mask": 1 if event.event_type is EventType.MODEL_RESPONSE else 0,
                }
            )
    if steps:
        steps[-1]["reward"] = report.reward
    return {
        "id": f"episode_{trace.trace_id}",
        "trace_id": trace.trace_id,
        "scenario_instance_id": report.scenario_instance_id,
        "steps": steps,
        "termination_reason": replay_trace(trace).state.termination_reason,
        "success": report.success,
    }


def analysis_record(trace: Trace, report: EvaluationReport) -> Dict[str, Any]:
    return {
        "trace_id": trace.trace_id,
        "scenario_instance_id": report.scenario_instance_id,
        "success": report.success,
        "infrastructure_failure": report.infrastructure_failure,
        "results": [result.__dict__ for result in report.results],
        "metrics": report.metrics,
    }
