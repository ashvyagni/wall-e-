"""Planner interface shared by classical and LLM agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

from walle.contracts import (
    Action,
    AgentConfig,
    BeliefSummary,
    BudgetConfig,
    MapConfig,
    Observation,
    SensorsConfig,
)


class AgentView(BaseModel):
    """Redacted mission view handed to planners.

    Contains only planner-legal information: map geometry, budget, sensors,
    the agent's own id/start, and game-rule parameters (declare reach, total
    clue count). Never placements or seeds of hidden entities.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    mission_id: str
    map: MapConfig
    budget: BudgetConfig
    sensors: SensorsConfig
    agent: AgentConfig
    contact_radius: int = 1
    clue_count: int = 0


@dataclass
class Decision:
    action: Action
    planner: str
    reason: str
    options_considered: tuple[str, ...] = ()
    extras: dict[str, object] = field(default_factory=dict)


class Planner(Protocol):
    """A planner sees only Observations and its own belief state."""

    def act(self, obs: Observation) -> Decision: ...

    def belief_summary(self) -> BeliefSummary: ...
