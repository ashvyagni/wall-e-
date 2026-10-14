"""LLM ReAct planner with provenance-cached, credential-free replay.

Design (per the locked brief):
- OpenAI-compatible chat-completions tool calling, one tool call per tick.
- Every live call is recorded in a committed JSONL cache together with full
  provenance (provider, model id, prompt/tool-schema versions, decoding
  settings, seed, config hash, message hash, timestamp, usage, latency).
- Replay is the default: with no API key, decisions are served from the
  cache byte-identically; a cache miss raises ReplayMissError instead of
  inventing behavior.
- The planner sees only the redacted AgentView and Observations; prompts are
  built by pure functions so cache keys are deterministic.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from walle.agent.base import AgentView, Decision
from walle.belief.grid import BeliefGrid
from walle.contracts import Action, BeliefSummary, Observation

PROMPT_VERSION = "1.0"
TOOL_SCHEMA_VERSION = "1.0"
HISTORY_WINDOW = 16  # recent (user, assistant) pairs included in the prompt


class ReplayMissError(RuntimeError):
    """No cached decision for this prompt and live mode is off."""


# --------------------------------------------------------------------------
# Tool schemas (versioned — bump TOOL_SCHEMA_VERSION on any change)
# --------------------------------------------------------------------------

_REASON = {
    "type": "string",
    "description": "One-sentence justification for this action.",
}

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "move",
            "description": "Move one cell in a cardinal direction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["north", "south", "east", "west"],
                    },
                    "reason": _REASON,
                },
                "required": ["direction", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scan",
            "description": (
                "Pulse local sensors: detects clues within scan_radius, "
                "improves signal quality for this tick."
            ),
            "parameters": {
                "type": "object",
                "properties": {"reason": _REASON},
                "required": ["reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "declare_found",
            "description": (
                "Declare EVE's location. Must be within contact_radius of your "
                "position. Succeeds only if the cell is exactly right."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "position": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "reason": _REASON,
                },
                "required": ["position", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recharge",
            "description": "Recharge to full. Only valid on a recharge station.",
            "parameters": {
                "type": "object",
                "properties": {"reason": _REASON},
                "required": ["reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "return_home",
            "description": (
                "Macro: path autonomously back to home. Needs enough battery "
                "for the trip plus reserve."
            ),
            "parameters": {
                "type": "object",
                "properties": {"reason": _REASON},
                "required": ["reason"],
            },
        },
    },
]


def build_system_prompt(view: AgentView) -> str:
    """Pure: only planner-legal information from the redacted AgentView."""
    m = view.map
    b = view.budget
    obstacles = ", ".join(f"[{x},{y}]" for x, y in m.obstacles) or "none"
    stations = ", ".join(f"[{x},{y}]" for x, y in m.recharge_stations) or "none"
    costs = ", ".join(f"{k}={v}" for k, v in sorted(b.action_costs.items()))
    return (
        f"You are WALL-E, an autonomous search unit looking for EVE on a "
        f"{m.width}x{m.height} grid (x east, y north).\n"
        f"Map: home=[{m.home[0]},{m.home[1]}]; obstacles: {obstacles}; "
        f"recharge stations: {stations}.\n"
        f"Budget: battery={b.battery}, max_ticks={b.max_ticks}, action costs "
        f"({costs}), recharge takes {b.recharge_ticks} ticks, keep "
        f"{b.reserve_margin} battery in reserve.\n"
        f"Sensors: each tick you observe a signal (strength 0..1 and bearing "
        f"degrees) from the dominant transmitter, plus terrain within radius "
        f"{view.sensors.local_view_radius}. Signals may be spoofed by decoy "
        f"beacons — a strong reading is not proof of EVE.\n"
        f"Clues: {view.clue_count} clue(s) exist somewhere; you discover one "
        f"by reaching its neighborhood, or reliably by scanning (radius "
        f"{view.sensors.scan_radius}). Clues corroborate a declare — a signal "
        f"reading alone may be a decoy.\n"
        f"Declare: call declare_found with the cell [x,y] while within "
        f"{view.contact_radius} cells of it. Success requires the exact cell; "
        f"a wrong declare fails the mission.\n"
        f"Terminals: EVE located (success); returned home without evidence; "
        f"battery exhausted; tick limit; 10 consecutive invalid actions "
        f"aborts.\n"
        f"Rules: call exactly one tool every turn; never guess coordinates "
        f"without gathering evidence first; watch battery and return or "
        f"recharge before it runs out."
    )


def _obs_payload(obs: Observation) -> dict[str, Any]:
    """Compact, planner-legal observation for the prompt."""
    signal = None
    if obs.signal is not None:
        signal = {
            "strength": obs.signal.strength,
            "bearing_deg": obs.signal.bearing_deg,
        }
    visible = [f"{c.kind}:{c.pos[0]},{c.pos[1]}" for c in obs.visible_cells]
    events = [
        {"type": e.type, "kind": e.kind, "pos": list(e.pos)}
        for e in obs.events
    ]
    return {
        "tick": obs.tick,
        "position": list(obs.position),
        "battery": obs.battery,
        "signal": signal,
        "visible": visible,
        "events": events,
        "last_action": {"valid": obs.last_action_valid, "error": obs.last_action_error},
        "sensor_status": dict(obs.sensor_status),
    }


def build_messages(
    view: AgentView,
    history: list[dict[str, Any]],
    obs: Observation,
) -> list[dict[str, Any]]:
    """Pure message list: system + windowed history + current observation."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": build_system_prompt(view)}
    ]
    for turn in history[-HISTORY_WINDOW:]:
        messages.append({"role": "user", "content": json.dumps(turn["user"], sort_keys=True)})
        messages.append(turn["assistant"])
    messages.append({"role": "user", "content": json.dumps(_obs_payload(obs), sort_keys=True)})
    return messages


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "openai-compat"
    model: str = "gpt-4o-mini"
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.0
    seed: int = 0
    max_tokens: int = 700
    timeout_s: float = 120.0

    def config_hash(self) -> str:
        blob = _canonical(
            {
                "provider": self.provider,
                "model": self.model,
                "base_url": self.base_url,
                "temperature": self.temperature,
                "seed": self.seed,
                "max_tokens": self.max_tokens,
            }
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    @classmethod
    def from_env(cls) -> LLMConfig:
        return cls(
            provider=os.environ.get("WALLE_LLM_PROVIDER", "openai-compat"),
            model=os.environ.get("WALLE_LLM_MODEL")
            or os.environ.get("OPENAI_MODEL")
            or "gpt-4o-mini",
            base_url=os.environ.get("WALLE_LLM_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL"),
            api_key=os.environ.get("WALLE_LLM_API_KEY")
            or os.environ.get("OPENAI_API_KEY"),
            temperature=float(os.environ.get("WALLE_LLM_TEMPERATURE", "0")),
            seed=int(os.environ.get("WALLE_LLM_SEED", "0")),
        )


def cache_key(messages: list[dict[str, Any]], cfg: LLMConfig) -> str:
    blob = _canonical(
        {
            "messages": messages,
            "model": cfg.model,
            "temperature": cfg.temperature,
            "seed": cfg.seed,
            "max_tokens": cfg.max_tokens,
            "prompt_version": PROMPT_VERSION,
            "tool_schema_version": TOOL_SCHEMA_VERSION,
            "config_hash": cfg.config_hash(),
        }
    )
    return hashlib.sha256(blob.encode()).hexdigest()


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------


@dataclass
class CacheEntry:
    key: str
    request_messages: list[dict[str, Any]]
    response: dict[str, Any]
    provenance: dict[str, Any]
    fallback: str | None = None

    def to_json(self) -> str:
        return _canonical(
            {
                "key": self.key,
                "request_messages": self.request_messages,
                "response": self.response,
                "provenance": self.provenance,
                "fallback": self.fallback,
            }
        )

    @classmethod
    def from_json(cls, line: str) -> CacheEntry:
        d = json.loads(line)
        return cls(
            key=d["key"],
            request_messages=d["request_messages"],
            response=d["response"],
            provenance=d["provenance"],
            fallback=d.get("fallback"),
        )


class LLMCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.entries: dict[str, CacheEntry] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    entry = CacheEntry.from_json(line)
                    self.entries[entry.key] = entry

    def get(self, key: str) -> CacheEntry | None:
        return self.entries.get(key)

    def put(self, entry: CacheEntry) -> None:
        self.entries[entry.key] = entry
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(entry.to_json() + "\n")


# --------------------------------------------------------------------------
# Transport (OpenAI-compatible chat completions)
# --------------------------------------------------------------------------

Transport = Callable[[list[dict[str, Any]], LLMConfig], dict[str, Any]]
"""Returns {"content": str|None, "tool_calls": [{"name","arguments_json"}],
"usage": {...}}"""


def openai_transport(messages: list[dict[str, Any]], cfg: LLMConfig) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(
        base_url=cfg.base_url,
        api_key=cfg.api_key or "missing-key",
        timeout=cfg.timeout_s,
    )
    resp = client.chat.completions.create(
        model=cfg.model,
        messages=messages,  # type: ignore[arg-type]
        tools=TOOLS,  # type: ignore[arg-type]
        temperature=cfg.temperature,
        seed=cfg.seed,
        max_tokens=cfg.max_tokens,
    )
    choice = resp.choices[0].message
    tool_calls = [
        {"name": tc.function.name, "arguments_json": tc.function.arguments}
        for tc in (choice.tool_calls or [])
        if hasattr(tc, "function")
    ]
    usage: dict[str, Any] = {}
    if resp.usage is not None:
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
        }
    return {"content": choice.content, "tool_calls": tool_calls, "usage": usage}


# --------------------------------------------------------------------------
# Planner
# --------------------------------------------------------------------------


def _action_from_tool(name: str, args: dict[str, Any]) -> Action:
    if name == "move":
        return Action.move(args["direction"])
    if name == "scan":
        return Action.scan()
    if name == "declare_found":
        pos = args["position"]
        return Action.declare_found((int(pos[0]), int(pos[1])))
    if name == "recharge":
        return Action.recharge()
    if name == "return_home":
        return Action.return_home()
    raise ValueError(f"unknown tool: {name}")


@dataclass
class LLMStats:
    replays: int = 0
    live_calls: int = 0
    fallbacks: int = 0
    latency_ms: list[int] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMPlanner:
    planner_name = "llm"

    def __init__(
        self,
        view: AgentView,
        *,
        cache_path: Path,
        config: LLMConfig | None = None,
        live: bool = False,
        transport: Transport | None = None,
    ) -> None:
        if not isinstance(view, AgentView):
            view = AgentView.model_validate(view)
        self.view = view
        self.cfg = config or LLMConfig.from_env()
        self.live = live and bool(self.cfg.api_key or self.cfg.base_url)
        self._transport = transport
        self.cache = LLMCache(cache_path)
        self.belief = BeliefGrid(view.map, view.sensors, frozenset(view.map.obstacles))
        self.history: list[dict[str, Any]] = []
        self.stats = LLMStats()
        self._last_obs: Observation | None = None

    # -- belief (for trace + prompt context) --------------------------------

    def belief_summary(self) -> BeliefSummary:
        return self.belief.summary()

    # -- decision loop -------------------------------------------------------

    def act(self, obs: Observation) -> Decision:
        if obs.signal is not None or obs.sensor_status.get("signal") != "failed":
            self.belief.update_signal(obs)
        for event in obs.events:
            if event.type == "clue_discovered":
                self.belief.update_clue(event.pos)

        messages = build_messages(self.view, self.history, obs)
        key = cache_key(messages, self.cfg)
        entry = self.cache.get(key)

        if entry is not None:
            self.stats.replays += 1
            action, reason, fallback = self._action_from_entry(entry, obs)
            if fallback is not None:
                self.stats.fallbacks += 1
            self._remember(obs, action, reason)
            return Decision(
                action=action,
                planner=self.planner_name,
                reason=f"[replay] {reason}",
                extras={
                    "cache_key": key,
                    "replay": True,
                    "model": entry.provenance.get("model"),
                    "fallback": fallback,
                },
            )

        if not self.live:
            raise ReplayMissError(
                f"no cached decision for key {key[:12]} and live mode is off. "
                "Configure WALLE_LLM_API_KEY (or OPENAI_API_KEY) with --live, "
                f"or restore the committed cache {self.cache.path}."
            )

        entry = self._call_live(messages, key, obs)
        action, reason, fallback = self._action_from_entry(entry, obs)
        if fallback is not None:
            self.stats.fallbacks += 1
        self._remember(obs, action, reason)
        return Decision(
            action=action,
            planner=self.planner_name,
            reason=(f"[live] {reason}" if fallback is None else f"[live:{fallback}] {reason}"),
            extras={
                "cache_key": key,
                "replay": False,
                "model": entry.provenance.get("model"),
                "fallback": fallback,
            },
        )

    # -- internals -----------------------------------------------------------

    def _action_from_entry(
        self, entry: CacheEntry, obs: Observation
    ) -> tuple[Action, str, str | None]:
        resp = entry.response
        reason = str(resp.get("reason") or "no reason given")
        try:
            name = resp["tool"]
            args = dict(resp.get("args") or {})
            if "reason" in args:
                reason = str(args["reason"])
            action = _action_from_tool(name, args)
            return action, reason, entry.fallback
        except (KeyError, ValueError, TypeError, IndexError):
            # corrupt cache entry: fall back safely, honestly recorded
            return self._fallback_action(obs), reason, "corrupt_cache_entry"

    def _fallback_action(self, obs: Observation) -> Action:
        cost = self.view.budget.action_costs["scan"]
        reserve = self.view.budget.reserve_margin
        if obs.battery >= cost + reserve:
            return Action.scan()
        return Action.return_home()

    def _remember(self, obs: Observation, action: Action, reason: str) -> None:
        self._last_obs = obs
        payload = _obs_payload(obs)
        assistant = {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": action.type,
                        "arguments": json.dumps(
                            {
                                **(
                                    {"direction": action.direction}
                                    if action.type == "move" and action.direction
                                    else {}
                                ),
                                **(
                                    {"position": list(action.position)}
                                    if action.type == "declare_found" and action.position
                                    else {}
                                ),
                                "reason": reason,
                            },
                            sort_keys=True,
                        ),
                    },
                }
            ],
        }
        # histories store the user turn alongside for windowing
        self.history.append({"user": payload, "assistant": assistant})
        if len(self.history) > HISTORY_WINDOW * 2:
            self.history = self.history[-HISTORY_WINDOW * 2 :]

    def _call_live(self, messages: list[dict[str, Any]], key: str, obs: Observation) -> CacheEntry:
        transport = self._transport or openai_transport
        started = time.perf_counter()
        raw = transport(messages, self.cfg)
        latency = int((time.perf_counter() - started) * 1000)
        self.stats.live_calls += 1
        self.stats.latency_ms.append(latency)
        usage = raw.get("usage") or {}
        self.stats.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.stats.completion_tokens += int(usage.get("completion_tokens") or 0)

        response: dict[str, Any] = {"tool": None, "args": {}, "content": raw.get("content")}
        fallback: str | None = None
        tool_calls = raw.get("tool_calls") or []
        if tool_calls:
            tc = tool_calls[0]
            try:
                args = json.loads(tc.get("arguments_json") or "{}")
            except json.JSONDecodeError:
                args = {}
                fallback = "unparseable_arguments"
            name = tc.get("name")
            try:
                action = _action_from_tool(str(name), args)
            except (KeyError, ValueError, TypeError):
                action = self._fallback_action(obs)
                fallback = fallback or "invalid_tool_call"
            else:
                response = {"tool": name, "args": args, "content": raw.get("content")}
        else:
            action = self._fallback_action(obs)
            fallback = "no_tool_call"

        if response["tool"] is None:
            # record what the model actually returned alongside the fallback
            response = {
                "tool": action.type,
                "args": {"fallback": True},
                "content": raw.get("content"),
            }

        provenance = {
            "provider": self.cfg.provider,
            "base_url": (self.cfg.base_url or "https://api.openai.com/v1"),
            "model": self.cfg.model,
            "prompt_version": PROMPT_VERSION,
            "tool_schema_version": TOOL_SCHEMA_VERSION,
            "temperature": self.cfg.temperature,
            "seed": self.cfg.seed,
            "max_tokens": self.cfg.max_tokens,
            "config_hash": self.cfg.config_hash(),
            "messages_sha256": hashlib.sha256(
                _canonical(messages).encode()
            ).hexdigest()[:16],
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "usage": usage,
            "latency_ms": latency,
            "tick": obs.tick,
        }
        entry = CacheEntry(
            key=key,
            request_messages=messages,
            response=response,
            provenance=provenance,
            fallback=fallback,
        )
        self.cache.put(entry)
        return entry
