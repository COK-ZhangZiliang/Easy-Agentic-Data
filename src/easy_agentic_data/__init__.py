"""Easy Agentic Data public package."""

from .models import Task, Trajectory
from .pipeline import SynthesisPipeline
from .scenarios import HiddenEvaluatorContext, Scenario, ScenarioInstance

__all__ = [
    "HiddenEvaluatorContext",
    "Scenario",
    "ScenarioInstance",
    "SynthesisPipeline",
    "Task",
    "Trajectory",
]
__version__ = "0.1.0"
