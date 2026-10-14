"""LLM planner tests: cache determinism, provenance, replay, fallbacks.

No network: a scripted fake transport stands in for the API.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from walle.agent.base import AgentView
from walle.agent.llm import (
    PROMPT_VERSION,
    TOOL_SCHEMA_VERSION,
    LLMCache,
    LLMConfig,
    LLMPlanner,
    ReplayMissError,
    build_messages,
    build_system_prompt,
    cache_key,
)
from walle.contracts import Action, MissionConfig
from walle.env.environment import Environment
from walle.mission import run_mission

from .test_env import load_cfg
from .test_hidden_state import TRUTH_KEYS, walk_keys


def make_view(cfg: MissionConfig | None = None) -> AgentView:
    return AgentView.model_validate((cfg or load_cfg()).agent_view())


def tool_response(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {
        "content": None,
        "tool_calls": [{"name": name, "arguments_json": json.dumps(args)}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 10},
    }


def scripted_transport(script: list[dict[str, Any]], sink: list[int] | None = None):
    calls = {"n": 0}

    def transport(messages: list[dict[str, Any]], cfg: LLMConfig) -> dict[str, Any]:
        if sink is not None:
            sink.append(len(messages))
        idx = min(calls["n"], len(script) - 1)
        calls["n"] += 1
        return script[idx]

    transport.calls = calls  # type: ignore[attr-defined]
    return transport


def test_cache_key_is_deterministic_and_config_sensitive() -> None:
    cfg = load_cfg()
    view = make_view(cfg)
    env = Environment(cfg)
    obs = env.observe()
    msgs1 = build_messages(view, [], obs)
    msgs2 = build_messages(view, [], obs)
    c1 = LLMConfig(model="m", temperature=0.0)
    c2 = LLMConfig(model="m", temperature=0.7)
    assert cache_key(msgs1, c1) == cache_key(msgs2, c1)
    assert cache_key(msgs1, c1) != cache_key(msgs1, c2)


def test_prompt_and_messages_contain_no_truth() -> None:
    cfg = load_cfg()
    view = make_view(cfg)
    prompt = build_system_prompt(view)
    truth_values = [
        str(cfg.target.placement),
        *[str(list(p)) for p, _ in cfg.clues.placements],
        *[str(list(p)) for p in cfg.decoys.placements],
    ]
    for t in truth_values:
        assert t not in prompt, f"system prompt leaked {t}"
    env = Environment(cfg)
    msgs = build_messages(view, [], env.observe())
    blob = json.dumps(msgs)
    for t in truth_values:
        assert t not in blob, f"messages leaked {t}"
    stray = walk_keys(json.loads(blob)) & TRUTH_KEYS
    assert not stray


def test_live_record_then_replay_is_identical(tmp_path: Path) -> None:
    cfg = load_cfg()
    view = make_view(cfg)
    cache_path = tmp_path / "cache.jsonl"
    script = [
        tool_response("move", {"direction": "north", "reason": "explore north"}),
        tool_response("scan", {"reason": "gather evidence"}),
    ]
    cfg_live = LLMConfig(model="test-model", api_key="fake-key")

    live = LLMPlanner(
        view,
        cache_path=cache_path,
        config=cfg_live,
        live=True,
        transport=scripted_transport(script),
    )
    env = Environment(cfg)
    obs = env.observe()
    d1 = live.act(obs)
    d2 = live.act(obs)
    assert d1.action.type == "move" and d1.action.direction == "north"
    assert d1.reason.startswith("[live]")
    assert d2.action.type == "scan"
    assert live.stats.live_calls == 2
    assert cache_path.exists()

    # replay: no key, no transport — byte-identical decisions from cache
    replay = LLMPlanner(
        view, cache_path=cache_path, config=cfg_live, live=False, transport=None
    )
    r1 = replay.act(obs)
    r2 = replay.act(obs)
    assert r1.action == d1.action and r1.extras.get("replay") is True
    assert r2.action == d2.action
    assert replay.stats.replays == 2 and replay.stats.live_calls == 0


def test_replay_miss_raises_without_cache(tmp_path: Path) -> None:
    planner = LLMPlanner(
        make_view(),
        cache_path=tmp_path / "missing.jsonl",
        config=LLMConfig(api_key=None),
        live=False,
    )
    env = Environment(load_cfg())
    with pytest.raises(ReplayMissError, match="cache"):
        planner.act(env.observe())


def test_no_tool_call_falls_back_to_scan_and_records(tmp_path: Path) -> None:
    cfg = load_cfg()
    view = make_view(cfg)
    env = Environment(cfg)
    obs = env.observe()
    assert obs.battery >= cfg.budget.action_costs["scan"] + cfg.budget.reserve_margin
    empty = {"content": "hmm", "tool_calls": [], "usage": {}}
    planner = LLMPlanner(
        view,
        cache_path=tmp_path / "c.jsonl",
        config=LLMConfig(api_key="fake"),
        live=True,
        transport=scripted_transport([empty]),
    )
    d = planner.act(obs)
    assert d.action.type == "scan"
    assert d.extras.get("fallback") == "no_tool_call"
    entry = next(iter(planner.cache.entries.values()))
    assert entry.fallback == "no_tool_call"


def test_provenance_block_is_complete(tmp_path: Path) -> None:
    cfg = load_cfg()
    env = Environment(cfg)
    planner = LLMPlanner(
        make_view(cfg),
        cache_path=tmp_path / "c.jsonl",
        config=LLMConfig(model="test-model", api_key="fake"),
        live=True,
        transport=scripted_transport(
            [tool_response("scan", {"reason": "look around"})]
        ),
    )
    planner.act(env.observe())
    prov = next(iter(planner.cache.entries.values())).provenance
    required = {
        "provider",
        "model",
        "prompt_version",
        "tool_schema_version",
        "temperature",
        "seed",
        "max_tokens",
        "config_hash",
        "messages_sha256",
        "timestamp",
        "usage",
        "latency_ms",
    }
    assert required <= set(prov)
    assert prov["prompt_version"] == PROMPT_VERSION
    assert prov["tool_schema_version"] == TOOL_SCHEMA_VERSION
    assert prov["model"] == "test-model"


def test_full_run_replays_byte_identically(tmp_path: Path) -> None:
    cfg = load_cfg()
    view = make_view(cfg)
    cache_path = tmp_path / "c.jsonl"
    script = [
        tool_response("move", {"direction": "north", "reason": "n"}),
        tool_response("move", {"direction": "east", "reason": "e"}),
        tool_response("move", {"direction": "south", "reason": "s"}),
        tool_response("move", {"direction": "west", "reason": "w"}),
    ]
    live = LLMPlanner(
        view,
        cache_path=cache_path,
        config=LLMConfig(api_key="fake"),
        live=True,
        transport=scripted_transport(script),
    )
    r1 = run_mission(cfg, live, tmp_path / "t1.jsonl", tmp_path / "r1.json")

    replay = LLMPlanner(view, cache_path=cache_path, config=LLMConfig(), live=False)
    r2 = run_mission(cfg, replay, tmp_path / "t2.jsonl", tmp_path / "r2.json")

    assert r1.report.outcome.terminal == r2.report.outcome.terminal
    a1 = [
        (e.action, e.decision.action_type)
        for e in r1.events
        if e.decision is not None
    ]
    a2 = [
        (e.action, e.decision.action_type)
        for e in r2.events
        if e.decision is not None
    ]
    assert a1 == a2, "replay decisions must be byte-identical"
    assert replay.stats.live_calls == 0
    assert replay.stats.replays == len(a1)


def test_cache_roundtrip(tmp_path: Path) -> None:
    cfg = load_cfg()
    env = Environment(cfg)
    obs = env.observe()  # captured once: observe() draws fresh signal noise
    planner = LLMPlanner(
        make_view(cfg),
        cache_path=tmp_path / "c.jsonl",
        config=LLMConfig(api_key="fake"),
        live=True,
        transport=scripted_transport(
            [tool_response("declare_found", {"position": [1, 1], "reason": "test"})]
        ),
    )
    planner.act(obs)
    reloaded = LLMCache(tmp_path / "c.jsonl")
    assert len(reloaded.entries) == 1
    entry = next(iter(reloaded.entries.values()))
    assert entry.response["tool"] == "declare_found"

    # and the planner replays it into the same action
    replay = LLMPlanner(make_view(cfg), cache_path=tmp_path / "c.jsonl", live=False)
    d = replay.act(obs)
    assert d.action.type == "declare_found"
    assert isinstance(d.action, Action)
