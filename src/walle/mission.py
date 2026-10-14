"""Mission runner: env <-> planner loop with tracing and reporting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from walle.agent.base import Planner
from walle.contracts import (
    DecisionRecord,
    MissionConfig,
    MissionReport,
    TerminalState,
    TraceEvent,
)
from walle.env.environment import Environment
from walle.report.build import build_report
from walle.trace.writer import TraceWriter

MAX_STEPS = 10_000


@dataclass
class MissionResult:
    report: MissionReport
    trace_path: Path
    report_path: Path
    events: list[TraceEvent]


def run_mission(
    cfg: MissionConfig,
    planner: Planner,
    trace_path: Path,
    report_path: Path,
) -> MissionResult:
    env = Environment(cfg)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    events: list[TraceEvent] = []
    seq = 0

    with TraceWriter(trace_path) as writer:
        # initial observation (action-less)
        obs = env.observe()
        writer.write(
            TraceEvent(
                seq=seq,
                tick=obs.tick,
                agent_id=obs.agent_id,
                kind="initial",
                action=None,
                observation=obs,
                belief=planner.belief_summary(),
                decision=None,
                battery=obs.battery,
                terminal=None,
            )
        )
        events.append(
            TraceEvent(
                seq=seq,
                tick=obs.tick,
                agent_id=obs.agent_id,
                kind="initial",
                action=None,
                observation=obs,
                belief=planner.belief_summary(),
                decision=None,
                battery=obs.battery,
                terminal=None,
            )
        )
        seq += 1

        steps = 0
        terminal: TerminalState | None = None
        while steps < MAX_STEPS:
            decision = planner.act(obs)
            result = env.step(decision.action)
            steps += 1
            terminal = result.terminal
            extras = dict(decision.extras)
            if result.executed_path and len(result.executed_path) > 1:
                extras["executed_path"] = [list(p) for p in result.executed_path]
            extras.update({k: v for k, v in result.extras.items()})

            event = TraceEvent(
                seq=seq,
                tick=result.observation.tick,
                agent_id=result.observation.agent_id,
                kind="terminal" if result.done else "step",
                action=decision.action,
                observation=result.observation,
                belief=planner.belief_summary(),
                decision=DecisionRecord(
                    planner=decision.planner,
                    action_type=decision.action.type,
                    reason=decision.reason,
                    options_considered=decision.options_considered,
                    extras=extras,
                ),
                battery=result.observation.battery,
                terminal=terminal,
            )
            writer.write(event)
            events.append(event)
            seq += 1
            obs = result.observation
            if result.done:
                break
        else:
            terminal = TerminalState.TIMEOUT_ABORTED

    assert terminal is not None
    belief = planner.belief_summary()
    strategy_stats: dict[str, int | None] = {
        "recharges": getattr(planner, "recharges", 0),
        "scans": getattr(planner, "scans", 0),
        "turn_back_tick": getattr(planner, "turn_back_tick", None),
    }
    report = build_report(
        cfg=cfg,
        env=env,
        terminal=terminal,
        trace_path=str(trace_path),
        steps=steps,
        planner_name=getattr(planner, "planner_name", type(planner).__name__),
        belief=belief,
        strategy_stats=strategy_stats,
    )
    report_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return MissionResult(
        report=report, trace_path=trace_path, report_path=report_path, events=events
    )


def planner_for(cfg: MissionConfig, kind: str | None = None, *, live: bool = False) -> Planner:
    from walle.agent.base import AgentView

    chosen = kind or cfg.planner
    if chosen == "classical":
        from walle.agent.classical import ClassicalPlanner

        return ClassicalPlanner(AgentView.model_validate(cfg.agent_view()))
    if chosen == "llm":
        from walle.agent.llm import LLMPlanner

        cache_path = Path("cache") / "llm" / f"{cfg.mission_id}.jsonl"
        return LLMPlanner(
            AgentView.model_validate(cfg.agent_view()),
            cache_path=cache_path,
            live=live,
        )
    raise ValueError(f"unknown planner: {chosen}")
