"""The mission environment: validates actions, executes them, emits observations.

Hidden-state boundary: `step()` accepts only a validated Action and returns only
an Observation. Ground truth (EVE, clues, decoys) never crosses this boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from walle.contracts import (
    Action,
    ClueEvent,
    MissionConfig,
    Observation,
    Pos,
    SignalStatus,
    TerminalState,
)
from walle.env import pathing, sensors
from walle.env.world import World, build_world

MAX_CONSECUTIVE_INVALID = 10


class InvalidReason(StrEnum):
    MOVE_BLOCKED = "move target is an obstacle or out of bounds"
    RECHARGE_NOT_AT_STATION = "recharge requires standing on a recharge station"
    DECLARE_OUT_OF_REACH = "declare_found position is out of reach"
    RETURN_INFEASIBLE = "battery cannot cover the return path plus reserve"
    UNKNOWN = "unknown action"


@dataclass
class StepResult:
    observation: Observation
    done: bool
    terminal: TerminalState | None = None
    executed_path: tuple[Pos, ...] = ()
    extras: dict[str, object] = field(default_factory=dict)


def _chebyshev(a: Pos, b: Pos) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


class Environment:
    """Deterministic mission environment. One agent, one mission."""

    def __init__(self, cfg: MissionConfig) -> None:
        self.cfg = cfg
        self.world: World = build_world(cfg)
        self.terminal: TerminalState | None = None
        self._obstacles = frozenset(cfg.map.obstacles)
        self._move_cost = cfg.budget.action_costs["move"]
        self._scan_cost = cfg.budget.action_costs["scan"]

    # ------------------------------------------------------------------ util

    @property
    def done(self) -> bool:
        return self.terminal is not None

    def _station_set(self) -> frozenset[Pos]:
        return frozenset(self.cfg.map.recharge_stations)

    def cost_to_home(self) -> int | None:
        path = pathing.shortest_path(
            self.world.agent_pos,
            self.cfg.map.home,
            self._obstacles,
            self.cfg.map.width,
            self.cfg.map.height,
        )
        if not path:
            return None
        return max(0, len(path) - 1) * self._move_cost

    # ------------------------------------------------------------ observation

    def _stamp_signal(self, action_type: str) -> Observation:
        w = self.world
        cfg = self.cfg

        # sensor failure roll (deterministic: consumes world rng)
        status: SignalStatus = "ok"
        failure_rate = cfg.sensors.failure_rate
        if failure_rate > 0:
            roll = w.rng.random()
            if roll < failure_rate * 0.5:
                status = "failed"
            elif roll < failure_rate:
                status = "degraded"

        signal = None
        if status == "failed":
            w.consecutive_signal_fail += 1
        else:
            signal = sensors.measure_signal(
                cfg,
                w.sources(),
                w.agent_pos,
                w.rng,
                scan=(action_type == "scan"),
                noise_scale=3.0 if status == "degraded" else 1.0,
            )
            if signal is None:
                w.consecutive_signal_fail += 1
            else:
                w.consecutive_signal_fail = 0
                signal = signal.model_copy(update={"timestamp_tick": w.tick})

        return Observation(
            tick=w.tick,
            agent_id=cfg.agents[0].id,
            position=w.agent_pos,
            signal=signal,
            visible_cells=sensors.visible_cells(cfg, w.agent_pos, self._obstacles),
            events=(),
            battery=w.battery,
            max_battery=cfg.budget.battery,
            max_ticks=cfg.budget.max_ticks,
            sensor_status={"signal": status},
        )

    def observe(self) -> Observation:
        return self._stamp_signal("move")

    # -------------------------------------------------------------- discovery

    def _check_clue_discovery(self, scan: bool) -> tuple[ClueEvent, ...]:
        w = self.world
        radius = (
            self.cfg.sensors.scan_radius if scan else self.cfg.clues.detection_radius
        )
        events = []
        for clue in w.undiscovered_clues():
            if _chebyshev(w.agent_pos, clue.pos) <= radius:
                clue.discovered = True
                clue.discovered_tick = w.tick
                events.append(clue.to_event(w.tick))
        return tuple(events)

    # --------------------------------------------------------------- validator

    def validate(self, action: Action) -> str | None:
        """Return an error message if the action is illegal right now, else None.

        Pure: never mutates state.
        """
        w = self.world
        cfg = self.cfg
        cost = cfg.budget.action_costs[action.type]

        if action.type == "move":
            assert action.direction is not None
            dx, dy = {"north": (0, 1), "south": (0, -1), "east": (1, 0), "west": (-1, 0)}[
                action.direction
            ]
            nxt = (w.agent_pos[0] + dx, w.agent_pos[1] + dy)
            if (
                not pathing.in_bounds(nxt, cfg.map.width, cfg.map.height)
                or nxt in self._obstacles
            ):
                return InvalidReason.MOVE_BLOCKED.value
        elif action.type == "recharge":
            if w.agent_pos not in self._station_set():
                return InvalidReason.RECHARGE_NOT_AT_STATION.value
        elif action.type == "declare_found":
            assert action.position is not None
            if _chebyshev(w.agent_pos, action.position) > w.contact_radius:
                return InvalidReason.DECLARE_OUT_OF_REACH.value
        elif action.type == "return_home":
            c = self.cost_to_home()
            if c is None:
                return InvalidReason.RETURN_INFEASIBLE.value
            if w.battery < c + cfg.budget.reserve_margin:
                return InvalidReason.RETURN_INFEASIBLE.value
        elif action.type == "scan":
            pass
        else:
            return InvalidReason.UNKNOWN.value

        # universal budget checks
        if cost > w.battery:
            return "insufficient battery for action cost"
        if w.tick + 1 > cfg.budget.max_ticks and action.type != "recharge":
            return "tick budget exhausted"
        return None

    # ------------------------------------------------------------------ step

    def step(self, action: Action) -> StepResult:
        if self.terminal is not None:
            raise RuntimeError(f"environment already terminal: {self.terminal}")

        w = self.world
        cfg = self.cfg

        error = self.validate(action)
        if error is not None:
            w.invalid_streak += 1
            w.invalid_total += 1
            if w.invalid_streak >= MAX_CONSECUTIVE_INVALID:
                self.terminal = TerminalState.INVALID_ACTION_ABORTED
                return StepResult(
                    observation=self._with_invalid(action, error),
                    done=True,
                    terminal=self.terminal,
                )
            return StepResult(
                observation=self._with_invalid(action, error),
                done=False,
            )

        w.invalid_streak = 0
        cost = cfg.budget.action_costs[action.type]
        executed: tuple[Pos, ...] = (w.agent_pos,)
        extras: dict[str, object] = {}

        # ---- execute -------------------------------------------------------
        if action.type == "move":
            assert action.direction is not None
            dx, dy = {"north": (0, 1), "south": (0, -1), "east": (1, 0), "west": (-1, 0)}[
                action.direction
            ]
            w.agent_pos = (w.agent_pos[0] + dx, w.agent_pos[1] + dy)
            w.battery -= cost
            w.tick += 1
            executed = (w.agent_pos,)

        elif action.type == "scan":
            w.battery -= cost
            w.tick += 1

        elif action.type == "recharge":
            w.battery = cfg.budget.battery
            w.tick += cfg.budget.recharge_ticks
            extras["recharged_ticks"] = cfg.budget.recharge_ticks

        elif action.type == "return_home":
            path = pathing.shortest_path(
                w.agent_pos, cfg.map.home, self._obstacles, cfg.map.width, cfg.map.height
            )
            step_cost = cfg.budget.action_costs["move"]
            travelled: list[Pos] = []
            for cell in path[1:]:
                if w.battery < step_cost:
                    w.battery = 0
                    self.terminal = TerminalState.BATTERY_EXHAUSTED_ABORTED
                    return StepResult(
                        observation=self._finalize(action_type=action.type),
                        done=True,
                        terminal=self.terminal,
                        executed_path=tuple([w.agent_pos, *travelled]),
                        extras=extras,
                    )
                w.battery -= step_cost
                w.tick += 1
                w.agent_pos = cell
                travelled.append(cell)
                if w.tick > cfg.budget.max_ticks:
                    self.terminal = TerminalState.TIMEOUT_ABORTED
                    return StepResult(
                        observation=self._finalize(action_type=action.type),
                        done=True,
                        terminal=self.terminal,
                        executed_path=tuple(travelled),
                        extras=extras,
                    )
            # arrived home (or path was empty)
            extras["returned_home"] = True
            self.terminal = TerminalState.RETURNED_NO_EVIDENCE
            return StepResult(
                observation=self._finalize(action_type=action.type),
                done=True,
                terminal=self.terminal,
                executed_path=tuple(travelled) if travelled else (w.agent_pos,),
                extras=extras,
            )

        elif action.type == "declare_found":
            assert action.position is not None
            w.battery -= cost
            w.tick += 1
            success = _chebyshev(action.position, w.eve_pos) <= w.contact_radius
            if success:
                self.terminal = TerminalState.EVE_LOCATED_SUCCESS
            else:
                w.failed_claims += 1
                self.terminal = TerminalState.EVE_LOCATE_FAILED
            return StepResult(
                observation=self._finalize(action_type=action.type),
                done=True,
                terminal=self.terminal,
                executed_path=(w.agent_pos,),
                extras={**extras, "declared_at": action.position},
            )

        # ---- post-execution bookkeeping ------------------------------------
        events = self._check_clue_discovery(scan=(action.type == "scan"))
        obs = self._finalize(action_type=action.type, events=events)

        if w.battery <= 0:
            # battery hit zero exactly; nothing further is possible
            if w.agent_pos == cfg.map.home:
                self.terminal = TerminalState.RETURNED_NO_EVIDENCE
            else:
                self.terminal = TerminalState.BATTERY_EXHAUSTED_ABORTED
            return StepResult(observation=obs, done=True, terminal=self.terminal, extras=extras)

        if w.tick >= cfg.budget.max_ticks:
            self.terminal = TerminalState.TIMEOUT_ABORTED
            return StepResult(observation=obs, done=True, terminal=self.terminal, extras=extras)

        return StepResult(
            observation=obs, done=False, executed_path=executed, extras=extras
        )

    # ---------------------------------------------------------------- helpers

    def _with_invalid(self, action: Action, error: str) -> Observation:
        base = self._stamp_signal(action.type)
        return base.model_copy(
            update={"last_action_valid": False, "last_action_error": error}
        )

    def _finalize(
        self, action_type: str, events: tuple[ClueEvent, ...] = ()
    ) -> Observation:
        obs = self._stamp_signal(action_type)
        if events:
            obs = obs.model_copy(update={"events": events})
        return obs
