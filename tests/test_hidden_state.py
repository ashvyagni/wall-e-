"""Hidden-state boundary tests (promised by contracts/SEMANTICS.md).

Three properties:
1. serialization: no truth-bearing keys ever appear in Observation or
   TraceEvent documents over a real mission;
2. clue-discovery ordering: a clue is invisible until proximity/scan
   reveals it, then appears exactly once with its discovery tick;
3. planner API signature: planners are constructed from the redacted
   AgentView and act on Observations only — a full MissionConfig is
   rejected.
"""

from __future__ import annotations

import inspect
import json
from typing import get_type_hints

import pytest
from pydantic import ValidationError

from walle.agent.base import AgentView
from walle.agent.classical import ClassicalPlanner
from walle.contracts import Action, Observation, TraceEvent
from walle.env.environment import Environment
from walle.mission import run_mission
from walle.report.build import MissionReport

from .test_env import load_cfg

# keys that only ever appear in evaluator-side documents (MissionConfig,
# MissionReport.evaluation). None may appear in planner-facing documents.
TRUTH_KEYS = frozenset(
    {
        "target",
        "clues",
        "decoys",
        "true_eve_pos",
        "eve_pos",
        "placement",
        "placements",
        "spawn_radius",
        "min_clue_distance",
        "evaluation",
    }
)


def walk_keys(obj: object, path: str = "") -> set[str]:
    keys: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.add(f"{path}.{k}" if path else str(k))
            keys |= walk_keys(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            keys |= walk_keys(item, path)
    return keys


def test_observation_and_trace_events_never_carry_truth_keys(tmp_path) -> None:
    cfg = load_cfg()
    env = Environment(cfg)
    planner = ClassicalPlanner(AgentView.model_validate(cfg.agent_view()))
    # walk a mission: sweep the map with scripted moves + scans
    dirs = ["north", "east", "south", "west"]
    for i in range(60):
        obs = env.observe()
        event = TraceEvent(
            seq=i,
            tick=obs.tick,
            agent_id=obs.agent_id,
            kind="step",
            action=Action.move(dirs[i % 4]),
            observation=obs,
            belief=planner.belief_summary(),
            decision=None,
            battery=obs.battery,
            terminal=None,
        )
        for doc in (obs, event):
            stray = walk_keys(doc.model_dump(mode="json")) & TRUTH_KEYS
            assert not stray, f"planner-facing document leaked truth keys: {stray}"
        if env.terminal is not None:
            break
        env.step(Action.move(dirs[i % 4]))


def test_clue_invisible_until_discovered_then_exactly_once() -> None:
    cfg = load_cfg()
    env = Environment(cfg)
    clue = env.world.clues[0]
    # park far away: no clue information may appear
    env.world.agent_pos = (cfg.map.home[0], cfg.map.home[1])
    obs = env.observe()
    assert not any(e.type == "clue_discovered" for e in obs.events)
    assert clue.clue_id not in json.dumps(obs.model_dump(mode="json"))
    # walk onto the clue cell: it must be revealed on that very step
    env.world.agent_pos = (clue.pos[0], clue.pos[1] - 1) if clue.pos[1] > 0 else (
        clue.pos[0],
        clue.pos[1] + 1,
    )
    res = env.step(Action.move("north") if clue.pos[1] > 0 else Action.move("south"))
    discovered = [e for e in res.observation.events if e.type == "clue_discovered"]
    assert len(discovered) == 1
    assert discovered[0].clue_id == clue.clue_id
    assert discovered[0].tick == res.observation.tick


def test_planner_api_rejects_full_mission_config() -> None:
    cfg = load_cfg()
    # signature: act takes only an Observation
    hints = get_type_hints(ClassicalPlanner.act)
    sig = inspect.signature(ClassicalPlanner.act)
    assert [n for n in sig.parameters if n != "self"] == ["obs"]
    assert hints["obs"] is Observation
    # construction: AgentView forbids the truth-bearing sections
    with pytest.raises(ValidationError):
        ClassicalPlanner(cfg)  # type: ignore[arg-type]
    # the redacted view itself may never grow truth fields
    assert not (set(AgentView.model_fields) & TRUTH_KEYS)


def test_agent_view_roundtrip_has_no_truth(tmp_path) -> None:
    cfg = load_cfg()
    view = AgentView.model_validate(cfg.agent_view())
    blob = json.dumps(view.model_dump(mode="json"))
    stray = walk_keys(view.model_dump(mode="json")) & TRUTH_KEYS
    assert not stray, f"AgentView leaked truth keys: {stray}"
    assert str(cfg.target.placement) not in blob
    for clue in cfg.clues.placements:
        assert str(clue[0]) not in blob


def test_full_run_reports_truth_only_in_evaluation(tmp_path) -> None:
    cfg = load_cfg()
    planner = ClassicalPlanner(AgentView.model_validate(cfg.agent_view()))
    result = run_mission(
        cfg,
        planner,
        tmp_path / "trace.jsonl",
        tmp_path / "report.json",
    )
    # planner-facing trace: no truth keys
    for line in (tmp_path / "trace.jsonl").read_text().splitlines():
        stray = walk_keys(json.loads(line)) & TRUTH_KEYS
        assert not stray, f"trace event leaked truth keys: {stray}"
    # evaluator-facing report: truth lives exactly in .evaluation
    report = MissionReport.model_validate_json((tmp_path / "report.json").read_text())
    ev = report.model_dump(mode="json")["evaluation"]
    assert ev["true_eve_pos"] == list(result.report.evaluation.true_eve_pos)
    # ... and nowhere else in the report: that path appears exactly once
    truth_paths = {
        k for k in walk_keys(report.model_dump(mode="json")) if k.endswith(".true_eve_pos")
    }
    assert truth_paths == {"evaluation.true_eve_pos"}
