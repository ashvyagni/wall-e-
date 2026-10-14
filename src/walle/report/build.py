"""MissionReport construction from a finished run (planner-agnostic)."""

from __future__ import annotations

import math

from walle.contracts import (
    BeliefSummary,
    EvaluationSection,
    EvidenceItem,
    InferenceSection,
    MissionConfig,
    MissionReport,
    OutcomeSection,
    StrategyPhase,
    StrategySummary,
    TerminalState,
)
from walle.env.environment import Environment


def build_report(
    cfg: MissionConfig,
    env: Environment,
    terminal: TerminalState,
    trace_path: str,
    steps: int,
    planner_name: str,
    belief: BeliefSummary,
    strategy_stats: dict[str, int | None],
) -> MissionReport:
    w = env.world
    map_pos = belief.map_pos
    conf = belief.confidence

    successful = terminal == TerminalState.EVE_LOCATED_SUCCESS
    if successful:
        inference_status = "located"
        recommended = (
            "EVE located and confirmed by declare_found; return home with the "
            "evidence log and stand by for directive."
        )
    elif conf >= 0.3:
        inference_status = "estimated"
        recommended = (
            f"EVE not confirmed; best estimate {map_pos} at confidence {conf:.2f}. "
            "Recharge and re-approach along the bearing of strongest readings."
        )
    else:
        inference_status = "unknown"
        recommended = (
            "Signal did not converge within budget. Return home, recharge, and "
            "re-task with a wider sweep or a second sensor pass."
        )

    evidence = tuple(
        EvidenceItem(
            clue_id=c.clue_id,
            kind=c.kind,
            pos=c.pos,
            description=c.description,
            tick=c.discovered_tick,
        )
        for c in w.clues
        if c.discovered
    )

    err = math.hypot(map_pos[0] - w.eve_pos[0], map_pos[1] - w.eve_pos[1])

    turn_back = strategy_stats.get("turn_back_tick")
    phases: list[StrategyPhase] = [
        StrategyPhase(
            name="search",
            start_tick=0,
            end_tick=turn_back if turn_back is not None else w.tick,
            note="EIG-driven search with belief updates",
        )
    ]
    if turn_back is not None:
        phases.append(
            StrategyPhase(
                name="return",
                start_tick=turn_back,
                end_tick=w.tick,
                note="battery reserve triggered return-home",
            )
        )

    return MissionReport(
        mission_id=cfg.mission_id,
        seed=cfg.seed,
        planner=planner_name,
        outcome=OutcomeSection(
            terminal=terminal,
            successful=successful,
            steps=steps,
            ticks_used=w.tick,
            battery_used=cfg.budget.battery - w.battery,
            battery_remaining=w.battery,
        ),
        evidence=evidence,
        inference=InferenceSection(
            eve_location=map_pos,
            status=inference_status,  # type: ignore[arg-type]
            confidence=round(conf, 4),
            recommended_next_step=recommended,
        ),
        strategy=StrategySummary(
            phases=tuple(phases),
            recharges=int(strategy_stats.get("recharges") or 0),
            scans=int(strategy_stats.get("scans") or 0),
            turn_back_tick=turn_back,
        ),
        evaluation=EvaluationSection(
            true_eve_pos=w.eve_pos,
            localization_error=round(err, 3),
            clues_total=len(w.clues),
            clues_found=sum(1 for c in w.clues if c.discovered),
            invalid_action_count=w.invalid_total,
            failed_claims=w.failed_claims,
        ),
        trace_path=trace_path,
    )
