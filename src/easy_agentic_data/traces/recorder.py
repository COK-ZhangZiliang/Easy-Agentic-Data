from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from easy_agentic_data.models import stable_id
from easy_agentic_data.scenarios import ScenarioInstance
from easy_agentic_data.traces.events import EventType, TraceEvent


@dataclass(frozen=True)
class Trace:
    path: Path
    events: List[TraceEvent]
    trace_id: str
    truncated: bool = False

    @property
    def session_id(self) -> str:
        return self.events[0].session_id if self.events else ""


class TraceRecorder:
    """Append-only JSONL recorder that synchronizes every complete event to disk."""

    def __init__(
        self,
        path: str | Path,
        *,
        session_id: str,
        scenario_instance: Optional[ScenarioInstance] = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self._handle = self.path.open("x", encoding="utf-8")
        self._sequence = 0
        self._closed = False
        self._finished = False
        self._forbidden_values = (
            scenario_instance.sensitive_strings() if scenario_instance is not None else []
        )

    def __enter__(self) -> "TraceRecorder":
        return self

    def __exit__(self, *args: object) -> None:
        del args
        self.close()

    def record(self, event_type: EventType | str, payload: Dict[str, Any]) -> TraceEvent:
        if self._closed:
            raise RuntimeError("Cannot record an event after the trace recorder is closed")
        if self._finished:
            raise RuntimeError("Cannot record an event after session_finished")
        event_kind = EventType(event_type)
        self._check_hidden_context(payload)
        event = TraceEvent(
            session_id=self.session_id,
            sequence=self._sequence,
            event_type=event_kind,
            payload=payload,
        )
        encoded = json.dumps(event.to_dict(), ensure_ascii=True, sort_keys=True)
        self._handle.write(encoded)
        self._handle.write("\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._sequence += 1
        if event_kind is EventType.SESSION_FINISHED:
            self._finished = True
        return event

    def start(self, scenario_instance: ScenarioInstance) -> TraceEvent:
        if self._sequence != 0:
            raise RuntimeError("session_started must be the first trace event")
        self._forbidden_values = sorted(
            set(self._forbidden_values) | set(scenario_instance.sensitive_strings())
        )
        return self.record(
            EventType.SESSION_STARTED,
            {
                "scenario_instance_id": scenario_instance.instance_id,
                "scenario_id": scenario_instance.scenario_id,
                "environment_id": scenario_instance.environment_id,
                "initial_state_hash": scenario_instance.initial_state_hash,
                "public_task": scenario_instance.public_task.to_dict(),
                "random_seed": scenario_instance.random_seed,
                "parameters": scenario_instance.parameters,
            },
        )

    def close(self) -> None:
        if not self._closed:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()
            self._closed = True

    def _check_hidden_context(self, payload: Dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True)
        for value in self._forbidden_values:
            escaped = json.dumps(value, ensure_ascii=True)[1:-1]
            if escaped and escaped in encoded:
                raise ValueError("Trace event contains content from a hidden context")


def load_trace(
    path: str | Path,
    *,
    tolerate_truncated: bool = True,
) -> Trace:
    trace_path = Path(path)
    events: List[TraceEvent] = []
    truncated = False
    expected_session: Optional[str] = None

    with trace_path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            try:
                value = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                is_partial_tail = not line.endswith(b"\n") and handle.read(1) == b""
                if tolerate_truncated and is_partial_tail:
                    truncated = True
                    break
                raise ValueError(f"Invalid trace JSONL record after {len(events)} events") from exc

            event = TraceEvent.from_dict(value)
            if event.sequence != len(events):
                raise ValueError(
                    f"Invalid trace sequence {event.sequence}; expected {len(events)}"
                )
            if expected_session is None:
                expected_session = event.session_id
            elif event.session_id != expected_session:
                raise ValueError("Trace contains more than one session_id")
            events.append(event)

    _validate_event_order(events)
    trace_id = stable_id("trace", [event.to_dict() for event in events])
    return Trace(path=trace_path, events=events, trace_id=trace_id, truncated=truncated)


def _validate_event_order(events: Iterable[TraceEvent]) -> None:
    items = list(events)
    if not items:
        return
    if items[0].event_type is not EventType.SESSION_STARTED:
        raise ValueError("The first trace event must be session_started")
    finished_indexes = [
        index
        for index, event in enumerate(items)
        if event.event_type is EventType.SESSION_FINISHED
    ]
    if len(finished_indexes) > 1:
        raise ValueError("A trace can contain only one session_finished event")
    if finished_indexes and finished_indexes[0] != len(items) - 1:
        raise ValueError("session_finished must be the final trace event")
