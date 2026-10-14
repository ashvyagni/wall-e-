"""Frozen contract schemas for WALL-E and EVE's Signal (schema_version 1.0).

Single source of truth for all five contract documents:
MissionConfig, Observation, Action, TraceEvent, MissionReport.

Semantics are documented in contracts/SEMANTICS.md. Fixtures in
contracts/fixtures/*.json are validated against these models in Python
(tests/test_contracts.py) and against the zod mirror in TypeScript
(contracts/ts) so any breaking change fails CI in both languages.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1.0"

Pos = tuple[int, int]


def _pos_list(v: list[int] | tuple[int, int]) -> tuple[int, int]:
    if len(v) != 2:
        raise ValueError("position must be [x, y]")
    return (int(v[0]), int(v[1]))


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------------------
# Enums / literals
# ---------------------------------------------------------------------------


class TerminalState(StrEnum):
    EVE_LOCATED_SUCCESS = "EVE_LOCATED_SUCCESS"
    EVE_LOCATE_FAILED = "EVE_LOCATE_FAILED"
    RETURNED_NO_EVIDENCE = "RETURNED_NO_EVIDENCE"
    BATTERY_EXHAUSTED_ABORTED = "BATTERY_EXHAUSTED_ABORTED"
    TIMEOUT_ABORTED = "TIMEOUT_ABORTED"
    INVALID_ACTION_ABORTED = "INVALID_ACTION_ABORTED"


SUCCESS_TERMINALS = frozenset({TerminalState.EVE_LOCATED_SUCCESS})

ActionType = Literal["move", "scan", "recharge", "return_home", "declare_found"]
MoveDirection = Literal["north", "south", "east", "west"]
SignalStatus = Literal["ok", "degraded", "failed"]
PlannerKind = Literal["classical", "llm"]
ClueKind = Literal["footprint", "scorch_mark", "plant_sample", "waste_scrap", "signal_mirror"]


# ---------------------------------------------------------------------------
# 1. MissionConfig
# ---------------------------------------------------------------------------


class MapConfig(StrictModel):
    width: Annotated[int, Field(ge=2, le=512)]
    height: Annotated[int, Field(ge=2, le=512)]
    cell_size: Annotated[float, Field(gt=0)] = 1.0
    obstacles: tuple[Pos, ...] = ()
    home: Pos
    recharge_stations: tuple[Pos, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def _coerce_positions(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
        if isinstance(data, dict):
            for key in ("obstacles", "recharge_stations"):
                if key in data and data[key] is not None:
                    data[key] = [_pos_list(p) for p in data[key]]
            if "home" in data and data["home"] is not None:
                data["home"] = _pos_list(data["home"])
        return data

    @model_validator(mode="after")
    def _bounds(self) -> MapConfig:
        pts = [self.home, *self.recharge_stations, *self.obstacles]
        for x, y in pts:
            if not (0 <= x < self.width and 0 <= y < self.height):
                raise ValueError(f"position {(x, y)} outside map bounds")
        return self


class BudgetConfig(StrictModel):
    battery: Annotated[int, Field(gt=0, le=1_000_000)]
    max_ticks: Annotated[int, Field(gt=0, le=1_000_000)]
    action_costs: dict[str, Annotated[int, Field(ge=0)]] = Field(
        default_factory=lambda: {
            "move": 1,
            "scan": 2,
            "recharge": 0,
            "return_home": 0,
            "declare_found": 1,
        }
    )
    recharge_ticks: Annotated[int, Field(ge=1)] = 5
    reserve_margin: Annotated[int, Field(ge=0)] = 0

    @model_validator(mode="after")
    def _known_costs(self) -> BudgetConfig:
        known = {"move", "scan", "recharge", "return_home", "declare_found"}
        unknown = set(self.action_costs) - known
        if unknown:
            raise ValueError(f"unknown action cost keys: {sorted(unknown)}")
        missing = known - set(self.action_costs)
        if missing:
            raise ValueError(f"missing action cost keys: {sorted(missing)}")
        return self


class SignalSensorConfig(StrictModel):
    """Log-normal path-loss signal sensor.

    strength = clamp((reference_distance / max(d, reference_distance)) ^ path_loss_exponent
                     * reference_power, 0, 1), then log-normal noise in dB applied
    and squashed back to [0, 1]. bearing includes additive Gaussian noise (degrees).
    """

    model: Literal["log_normal_path_loss"] = "log_normal_path_loss"
    reference_distance: Annotated[float, Field(gt=0)] = 1.0
    reference_power: Annotated[float, Field(gt=0)] = 1.0
    path_loss_exponent: Annotated[float, Field(gt=0)] = 4.0
    noise_std_db: Annotated[float, Field(ge=0)] = 2.0
    bearing_noise_std_deg: Annotated[float, Field(ge=0)] = 5.0
    max_range: Annotated[int, Field(gt=0)] = 64
    scan_noise_scale: Annotated[float, Field(gt=0)] = 0.5


class SensorsConfig(StrictModel):
    signal: SignalSensorConfig = SignalSensorConfig()
    local_view_radius: Annotated[int, Field(ge=0, le=16)] = 3
    scan_radius: Annotated[int, Field(ge=1, le=32)] = 5
    failure_rate: Annotated[float, Field(ge=0, le=1)] = 0.0


class TargetConfig(StrictModel):
    """EVE's true placement. Evaluator-only: never exposed to planners."""

    placement: Pos | None = None  # explicit [x, y]; None => derived from seed
    contact_radius: Annotated[int, Field(ge=0, le=8)] = 1
    status: Literal["dormant", "active"] = "dormant"

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
        if isinstance(data, dict) and data.get("placement") is not None:
            data["placement"] = _pos_list(data["placement"])
        return data


class CluesConfig(StrictModel):
    """Clue placements/seeds. Evaluator-only: never exposed to planners."""

    placements: tuple[tuple[Pos, ClueKind], ...] = ()
    seed_count: Annotated[int, Field(ge=0, le=64)] = 4
    detection_radius: Annotated[int, Field(ge=1, le=16)] = 1
    # Seeded clues spawn within this chebyshev radius of EVE (the narrative
    # prior: "EVE left them behind"). Must be <= the planner's corroboration
    # radius (3) so any discovered clue corroborates a declare at true EVE.
    # Evaluator-only; never exposed to planners.
    spawn_radius: Annotated[int, Field(ge=0, le=64)] = 3

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
        if isinstance(data, dict) and data.get("placements"):
            data["placements"] = [
                (_pos_list(item[0]), item[1]) for item in data["placements"]
            ]
        return data


class DecoysConfig(StrictModel):
    """Spoofed signal sources. Evaluator-only: never exposed to planners."""

    count: Annotated[int, Field(ge=0, le=16)] = 0
    seed: int = 0
    placements: tuple[Pos, ...] = ()
    strength_floor: Annotated[float, Field(ge=0, le=1)] = 0.05
    # No decoy may sit within this chebyshev distance of any clue. This keeps
    # "a clue corroborates the declare cell" statistically valid: a MAP cell
    # next to a clue can then only be true EVE, never a decoy. Must be >
    # the planner's corroboration radius (3). Evaluator-only.
    min_clue_distance: Annotated[int, Field(ge=0, le=64)] = 4

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
        if isinstance(data, dict) and data.get("placements"):
            data["placements"] = [_pos_list(p) for p in data["placements"]]
        return data


class AgentConfig(StrictModel):
    id: str = "walle_0"
    start: Pos | None = None  # None => map.home

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
        if isinstance(data, dict) and data.get("start") is not None:
            data["start"] = _pos_list(data["start"])
        return data


class MissionConfig(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    mission_id: str
    seed: int
    map: MapConfig
    budget: BudgetConfig
    sensors: SensorsConfig = SensorsConfig()
    target: TargetConfig = TargetConfig()
    clues: CluesConfig = CluesConfig()
    decoys: DecoysConfig = DecoysConfig()
    agents: tuple[AgentConfig, ...] = (AgentConfig(),)
    planner: PlannerKind = "classical"

    @model_validator(mode="after")
    def _homes_free(self) -> MissionConfig:
        obs = set(self.map.obstacles)
        if self.map.home in obs:
            raise ValueError("home cell cannot be an obstacle")
        for st in self.map.recharge_stations:
            if st in obs:
                raise ValueError(f"recharge station {st} cannot be an obstacle")
        if not self.agents:
            raise ValueError("at least one agent required")
        return self

    def agent_view(self) -> dict[str, Any]:
        """Redacted config handed to planners: geometry, budget, sensors only.

        Strips every evaluator-only section (target, clues, decoys) including
        their seeds, so a planner cannot reconstruct ground truth. Game-rule
        parameters the agent must know to act (declare reach, whether clues
        exist in total) are included; never their locations or seeds.
        """
        return {
            "schema_version": self.schema_version,
            "mission_id": self.mission_id,
            "map": self.map.model_dump(mode="json"),
            "budget": self.budget.model_dump(mode="json"),
            "sensors": self.sensors.model_dump(mode="json"),
            "agent": self.agents[0].model_dump(mode="json"),
            "contact_radius": self.target.contact_radius,
            "clue_count": len(self.clues.placements) + self.clues.seed_count,
        }


# ---------------------------------------------------------------------------
# 2. Action
# ---------------------------------------------------------------------------


class Action(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    type: ActionType
    direction: MoveDirection | None = None  # required iff type == "move"
    position: Pos | None = None  # required iff type == "declare_found"

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
        if isinstance(data, dict) and data.get("position") is not None:
            data["position"] = _pos_list(data["position"])
        return data

    @model_validator(mode="after")
    def _shape(self) -> Action:
        if self.type == "move" and self.direction is None:
            raise ValueError("move requires direction")
        if self.type != "move" and self.direction is not None:
            raise ValueError(f"{self.type} must not carry direction")
        if self.type == "declare_found" and self.position is None:
            raise ValueError("declare_found requires position")
        if self.type != "declare_found" and self.position is not None:
            raise ValueError(f"{self.type} must not carry position")
        return self

    @classmethod
    def move(cls, direction: MoveDirection) -> Action:
        return cls(type="move", direction=direction)

    @classmethod
    def scan(cls) -> Action:
        return cls(type="scan")

    @classmethod
    def recharge(cls) -> Action:
        return cls(type="recharge")

    @classmethod
    def return_home(cls) -> Action:
        return cls(type="return_home")

    @classmethod
    def declare_found(cls, position: Pos) -> Action:
        return cls(type="declare_found", position=position)


# ---------------------------------------------------------------------------
# 3. Observation
# ---------------------------------------------------------------------------


class SignalReading(StrictModel):
    bearing_deg: float
    strength: Annotated[float, Field(ge=0, le=1)]
    measured_from: Pos
    timestamp_tick: int


class VisibleCell(StrictModel):
    pos: Pos
    kind: Literal["free", "obstacle", "recharge", "home"]

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
        if isinstance(data, dict) and "pos" in data:
            data["pos"] = _pos_list(data["pos"])
        return data


class ClueEvent(StrictModel):
    type: Literal["clue_discovered"] = "clue_discovered"
    clue_id: str
    kind: ClueKind
    pos: Pos
    description: str
    tick: int

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
        if isinstance(data, dict) and "pos" in data:
            data["pos"] = _pos_list(data["pos"])
        return data


def _default_sensor_status() -> dict[str, SignalStatus]:
    return {"signal": "ok"}


class Observation(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    tick: int
    agent_id: str
    position: Pos
    signal: SignalReading | None
    visible_cells: tuple[VisibleCell, ...]
    events: tuple[ClueEvent, ...] = ()
    battery: int
    max_battery: int
    max_ticks: int
    sensor_status: dict[str, SignalStatus] = Field(default_factory=_default_sensor_status)
    last_action_valid: bool = True
    last_action_error: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
        if isinstance(data, dict) and "position" in data and data["position"] is not None:
            data["position"] = _pos_list(data["position"])
        return data


# ---------------------------------------------------------------------------
# 4. TraceEvent
# ---------------------------------------------------------------------------


class BeliefSummary(StrictModel):
    map_pos: Pos
    confidence: Annotated[float, Field(ge=0, le=1)]
    entropy_bits: Annotated[float, Field(ge=0)]
    top_k: tuple[tuple[Pos, float], ...] = ()

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
        if isinstance(data, dict) and "map_pos" in data and data["map_pos"] is not None:
            data["map_pos"] = _pos_list(data["map_pos"])
        if isinstance(data, dict) and "top_k" in data:
            data["top_k"] = [( _pos_list(t[0]), t[1]) for t in data["top_k"]]
        return data


class DecisionRecord(StrictModel):
    planner: str
    action_type: str
    reason: str
    options_considered: tuple[str, ...] = ()
    extras: dict[str, Any] = Field(default_factory=dict)


class TraceEvent(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    seq: Annotated[int, Field(ge=0)]
    tick: int
    agent_id: str
    kind: Literal["initial", "step", "terminal"]
    action: Action | None = None
    observation: Observation | None = None
    belief: BeliefSummary | None = None
    decision: DecisionRecord | None = None
    battery: int
    terminal: TerminalState | None = None


# ---------------------------------------------------------------------------
# 5. MissionReport
# ---------------------------------------------------------------------------


class EvidenceItem(StrictModel):
    clue_id: str
    kind: ClueKind
    pos: Pos
    description: str
    tick: int

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
        if isinstance(data, dict) and "pos" in data:
            data["pos"] = _pos_list(data["pos"])
        return data


class InferenceSection(StrictModel):
    eve_location: Pos
    status: Literal["located", "estimated", "unknown"]
    confidence: Annotated[float, Field(ge=0, le=1)]
    recommended_next_step: str

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
        if isinstance(data, dict) and "eve_location" in data:
            data["eve_location"] = _pos_list(data["eve_location"])
        return data


class StrategyPhase(StrictModel):
    name: str
    start_tick: int
    end_tick: int
    note: str = ""


class StrategySummary(StrictModel):
    phases: tuple[StrategyPhase, ...] = ()
    recharges: int = 0
    scans: int = 0
    turn_back_tick: int | None = None


class EvaluationSection(StrictModel):
    """Evaluator-only ground truth. Never available to planners."""

    true_eve_pos: Pos
    localization_error: Annotated[float, Field(ge=0)]
    clues_total: int
    clues_found: int
    invalid_action_count: int
    failed_claims: int = 0

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
        if isinstance(data, dict) and "true_eve_pos" in data:
            data["true_eve_pos"] = _pos_list(data["true_eve_pos"])
        return data


class OutcomeSection(StrictModel):
    terminal: TerminalState
    successful: bool
    steps: int
    ticks_used: int
    battery_used: int
    battery_remaining: int


class MissionReport(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    mission_id: str
    seed: int
    planner: str
    generated_by: str = "walle"
    outcome: OutcomeSection
    evidence: tuple[EvidenceItem, ...] = ()
    inference: InferenceSection
    strategy: StrategySummary = StrategySummary()
    evaluation: EvaluationSection
    trace_path: str
    notes: tuple[str, ...] = ()


CONTRACT_MODELS: dict[str, type[BaseModel]] = {
    "MissionConfig": MissionConfig,
    "Observation": Observation,
    "Action": Action,
    "TraceEvent": TraceEvent,
    "MissionReport": MissionReport,
}
