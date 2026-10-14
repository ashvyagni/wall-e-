"""Semantic invariants from contracts/SEMANTICS.md (Hour-0 freeze)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from walle.contracts import (
    Action,
    MissionConfig,
    TerminalState,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "contracts" / "fixtures"


def _config() -> MissionConfig:
    return MissionConfig.model_validate(json.loads((FIXTURES / "mission_config.json").read_text()))


# --- action shape rules -----------------------------------------------------


def test_move_requires_direction() -> None:
    with pytest.raises(ValueError):
        Action(type="move")


def test_move_rejects_position() -> None:
    with pytest.raises(ValueError):
        Action(type="move", direction="north", position=(1, 1))


def test_declare_requires_position() -> None:
    with pytest.raises(ValueError):
        Action(type="declare_found")


def test_non_move_rejects_direction() -> None:
    with pytest.raises(ValueError):
        Action(type="scan", direction="north")


def test_action_helpers() -> None:
    assert Action.move("north").direction == "north"
    assert Action.declare_found((3, 4)).position == (3, 4)


# --- agent view redaction ---------------------------------------------------

def test_agent_view_strips_all_ground_truth() -> None:
    view = _config().agent_view()
    blob = json.dumps(view)
    for forbidden in ("target", "clues", "decoys", "planner", "seed", "17, 12", "[17, 12]"):
        assert forbidden not in blob, f"agent_view leaks {forbidden!r}"


def test_agent_view_keeps_geometry_budget_sensors() -> None:
    view = _config().agent_view()
    assert set(view) == {
        "schema_version",
        "mission_id",
        "map",
        "budget",
        "sensors",
        "agent",
        "contact_radius",
        "clue_count",
    }
    assert view["map"]["home"] == [1, 1]
    assert view["budget"]["battery"] == 220
    assert view["contact_radius"] == 1
    assert view["clue_count"] == 4


def test_agent_view_does_not_alias_original() -> None:
    cfg = _config()
    view = cfg.agent_view()
    view["budget"]["battery"] = 1
    assert cfg.budget.battery == 220


# --- terminal states --------------------------------------------------------


def test_terminal_enum_values_frozen() -> None:
    assert [t.value for t in TerminalState] == [
        "EVE_LOCATED_SUCCESS",
        "EVE_LOCATE_FAILED",
        "RETURNED_NO_EVIDENCE",
        "BATTERY_EXHAUSTED_ABORTED",
        "TIMEOUT_ABORTED",
        "INVALID_ACTION_ABORTED",
    ]


# --- config validation ------------------------------------------------------


def test_config_rejects_out_of_bounds_home() -> None:
    cfg = _config().model_dump(mode="json")
    cfg["map"]["home"] = [999, 999]
    with pytest.raises(ValidationError):
        MissionConfig.model_validate(cfg)


def test_config_rejects_missing_action_cost() -> None:
    cfg = _config().model_dump(mode="json")
    del cfg["budget"]["action_costs"]["scan"]
    with pytest.raises(ValidationError):
        MissionConfig.model_validate(cfg)


def test_config_rejects_unknown_action_cost() -> None:
    cfg = _config().model_dump(mode="json")
    cfg["budget"]["action_costs"]["warp"] = 1
    with pytest.raises(ValidationError):
        MissionConfig.model_validate(cfg)
