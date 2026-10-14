"""Contract tests: fixtures validate, schemas round-trip, exports are current."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from walle.contracts import (
    CONTRACT_MODELS,
    Action,
    MissionConfig,
    MissionReport,
    Observation,
    TraceEvent,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "contracts" / "fixtures"
SCHEMA_DIR = ROOT / "contracts" / "schema"

FIXTURE_MAP: dict[str, tuple[str, type]] = {
    "mission_config.json": ("MissionConfig", MissionConfig),
    "observation.json": ("Observation", Observation),
    "action_move.json": ("Action", Action),
    "action_declare.json": ("Action", Action),
    "trace_event.json": ("TraceEvent", TraceEvent),
    "mission_report.json": ("MissionReport", MissionReport),
}


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def test_all_fixtures_present() -> None:
    names = {p.name for p in FIXTURES.glob("*.json")}
    assert names == set(FIXTURE_MAP), f"fixture set drift: {names ^ set(FIXTURE_MAP)}"


@pytest.mark.parametrize("fname", sorted(FIXTURE_MAP))
def test_fixture_validates(fname: str) -> None:
    _, model = FIXTURE_MAP[fname]
    inst = model.model_validate(_load(fname))
    # round-trip: dump then re-validate produces identical JSON
    dumped = json.loads(json.dumps(inst.model_dump(mode="json")))
    inst2 = model.model_validate(dumped)
    assert inst2.model_dump(mode="json") == dumped


@pytest.mark.parametrize("fname", sorted(FIXTURE_MAP))
def test_fixture_rejects_unknown_fields(fname: str) -> None:
    _, model = FIXTURE_MAP[fname]
    data = _load(fname)
    data["__smuggled_field__"] = 1
    with pytest.raises(ValidationError):
        model.model_validate(data)


def test_schema_exports_are_current() -> None:
    for name, model in CONTRACT_MODELS.items():
        path = SCHEMA_DIR / f"{name}.schema.json"
        assert path.exists(), f"missing exported schema {path.name}"
        exported = json.loads(path.read_text())
        fresh = model.model_json_schema(by_alias=True)
        assert exported == fresh, f"{name} schema out of date — run scripts/export_schemas.py"


def test_schema_version_is_1_0_everywhere() -> None:
    for fname, (label, _model) in FIXTURE_MAP.items():
        if label in {"MissionConfig", "Observation", "Action", "TraceEvent", "MissionReport"}:
            data = _load(fname)
            assert data.get("schema_version") == "1.0", f"{fname} schema_version"
