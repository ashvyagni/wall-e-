"""Classical planner: myopic information-gain search over a Bayesian belief grid.

Each tick the planner:
1. updates its belief from the observation (signal + clue evidence)
2. handles safety first: declare at high confidence, recharge when needed,
   return home when the return leg is at risk
3. otherwise scores every reachable free cell by expected entropy reduction
   of a reading there (exact discrete EIG over strength x bearing bins),
   divided by path cost, plus a frontier bonus for unvisited territory
4. executes the first primitive step toward the best cell (or scans in place)
"""

from __future__ import annotations

import math
from collections import deque
from typing import cast

import numpy as np

from walle.agent.base import AgentView, Decision
from walle.belief.grid import BeliefGrid
from walle.contracts import (
    Action,
    BeliefSummary,
    ClueEvent,
    MoveDirection,
    Observation,
    Pos,
)
from walle.env import pathing

INVESTIGATE_CONFIDENCE = 0.5  # approach the MAP cell and verify before declaring
EVIDENCE_CONFIDENCE = 0.75  # required to declare with a corroborating clue
DECLARE_CONFIDENCE = 0.9  # required when there are no clues to corroborate
CORROBORATE_RADIUS = 3
_MAX_SCANS_PER_REGION = 6
_FRONTIER_BONUS = 0.35
_STRENGTH_BINS = 8
_BEARING_BINS = 12
_MIN_SCORE = 1e-6


class ClassicalPlanner:
    planner_name = "classical"

    def __init__(self, view: AgentView) -> None:
        self.view = view
        self.obstacles = frozenset(view.map.obstacles)
        self.belief = BeliefGrid(view.map, view.sensors, self.obstacles)
        self.visited: set[Pos] = {view.agent.start or view.map.home}
        self.discovered_clues: list[ClueEvent] = []
        self.recharges = 0
        self.scans = 0
        self.turn_back_tick: int | None = None
        self._move_cost = view.budget.action_costs["move"]
        self._scan_cost = view.budget.action_costs["scan"]
        self._declare_mode = False
        self._scanned: list[Pos] = []
        self._last_unconfirmed: Pos | None = None
        self._departed_map: Pos | None = None
        self._departed_conf: float = 0.0

    # ------------------------------------------------------------------ helpers

    def belief_summary(self) -> BeliefSummary:
        return self.belief.summary()

    def _dist_map(self, start: Pos) -> tuple[dict[Pos, int], dict[Pos, MoveDirection]]:
        """BFS from start: distance (moves) and first-step direction to every
        reachable free cell."""
        w, h = self.view.map.width, self.view.map.height
        dist: dict[Pos, int] = {start: 0}
        first: dict[Pos, MoveDirection] = {}
        queue: deque[Pos] = deque([start])
        while queue:
            cx, cy = queue.popleft()
            for direction, (dx, dy) in (
                ("north", (0, 1)),
                ("south", (0, -1)),
                ("east", (1, 0)),
                ("west", (-1, 0)),
            ):
                nxt = (cx + dx, cy + dy)
                if nxt in dist or not pathing.in_bounds(nxt, w, h) or nxt in self.obstacles:
                    continue
                dist[nxt] = dist[(cx, cy)] + 1
                first[nxt] = first.get((cx, cy), cast(MoveDirection, direction))
                queue.append(nxt)
        return dist, first

    def _eig_bits(self, agent_pos: Pos, *, scan: bool) -> float:
        """Exact discrete expected information gain (mutual information) in bits
        of taking a signal reading at `agent_pos`."""
        b = self.belief.prob
        s_cfg = self.view.sensors.signal
        scale = s_cfg.scan_noise_scale if scan else 1.0
        sigma_ln = max(math.log(10.0) * s_cfg.noise_std_db * scale / 20.0, 1e-3)
        bearing_std = max(s_cfg.bearing_noise_std_deg * scale, 1e-3)

        X, Y = np.meshgrid(
            np.arange(self.view.map.width, dtype=np.float64),
            np.arange(self.view.map.height, dtype=np.float64),
        )
        d = np.hypot(X - agent_pos[0], Y - agent_pos[1])
        pred_s = np.clip(
            s_cfg.reference_power
            * (s_cfg.reference_distance / np.maximum(d, s_cfg.reference_distance))
            ** s_cfg.path_loss_exponent,
            1e-9,
            1.0,
        )
        pred_b = np.degrees(np.arctan2(X - agent_pos[0], Y - agent_pos[1])) % 360.0

        s_edges = np.geomspace(1e-4, 1.0, _STRENGTH_BINS + 1)
        b_edges = np.linspace(0.0, 360.0, _BEARING_BINS + 1)
        s_centers = np.sqrt(s_edges[:-1] * s_edges[1:])
        b_centers = (b_edges[:-1] + b_edges[1:]) / 2.0

        # L[o, cell]: likelihood of bin o for each cell (flattened)
        pred_s_flat = pred_s.ravel()
        pred_b_flat = pred_b.ravel()
        # wrapped bearing difference for every (bin, cell) pair
        db = (b_centers[:, None] - pred_b_flat[None, :]) % 360.0
        db = np.minimum(db, 360.0 - db)
        lik_b = np.exp(-0.5 * (db / bearing_std) ** 2) / (bearing_std * math.sqrt(2 * math.pi))

        # strength bin likelihood via lognormal pdf at bin center (approx.)
        with np.errstate(divide="ignore"):
            ls = -(
                np.log(s_centers)[:, None]
                + np.log(sigma_ln)
                + 0.5 * math.log(2 * math.pi)
                + (np.log(s_centers)[:, None] - np.log(pred_s_flat)[None, :]) ** 2
                / (2 * sigma_ln**2)
            )
        lik_s = np.exp(np.clip(ls, -40, 0))

        # joint likelihood over (strength bin, bearing bin, cell)
        lik = lik_s[:, None, :] * lik_b[None, :, :]
        n_cells = b.size
        lik = lik.reshape(-1, n_cells)
        lik = np.maximum(lik, 1e-12)

        p_obs = lik @ b.ravel()  # (n_obs,)
        p_obs = np.maximum(p_obs, 1e-12)
        # I(X;O) = sum_o p(o) * sum_x b(x) L(o|x) log2( L(o|x) / p(o) )
        ratio = np.log2(lik / p_obs[:, None])
        mi = float((p_obs[:, None] * (b.ravel()[None, :] * lik) * ratio).sum())
        return max(mi, 0.0)

    # ------------------------------------------------------------------ policy

    def _escape_direction(
        self, pos: Pos, from_map: Pos, dist: dict[Pos, int], first: dict[Pos, MoveDirection]
    ) -> MoveDirection:
        """Move toward the highest-belief cell *away from* the current MAP cell.

        Staying inside the corroboration radius would just oscillate back into
        contact, so only cells outside that radius are escape targets.
        """
        target: Pos | None = None
        for (px, py), _p in self.belief.top_k(12):
            if (
                max(abs(px - from_map[0]), abs(py - from_map[1])) > CORROBORATE_RADIUS + 1
                and not self.belief.near_suspected((px, py))
            ):
                target = (px, py)
                break
        if target is not None:
            step = first.get(target)
            if step is not None:
                return step
        # fallback: first step toward the reachable cell farthest from the MAP
        fallback: MoveDirection = "north"
        best_d = -1
        for cell, _d in dist.items():
            if cell == pos:
                continue
            step2 = first.get(cell)
            if step2 is None:
                continue
            d_from_map = abs(cell[0] - from_map[0]) + abs(cell[1] - from_map[1])
            if d_from_map > best_d:
                best_d = d_from_map
                fallback = step2
        return fallback

    def _safety_action(
        self,
        obs: Observation,
        dist: dict[Pos, int],
        first: dict[Pos, MoveDirection],
    ) -> Decision | None:
        cfg = self.view
        w = cfg.budget
        pos = obs.position
        map_pos = self.belief.map_pos()
        conf = self.belief.confidence()
        cost_home = dist.get(cfg.map.home)
        reserve = w.reserve_margin

        # 1. investigate when the MAP cell is plausible: approach it and verify
        #    with clue corroboration (or very high confidence if no clues exist).
        #    While departed to gather fresh readings (corroborated but belief
        #    stuck), suspend the approach until the belief actually moves.
        if self._departed_map is not None:
            if (
                conf > self._departed_conf + 0.05
                or map_pos != self._departed_map
            ):
                self._departed_map = None
            else:
                self._declare_mode = False
        if conf >= INVESTIGATE_CONFIDENCE and self._departed_map is None:
            self._declare_mode = True
        if self._declare_mode:
            reach = max(abs(pos[0] - map_pos[0]), abs(pos[1] - map_pos[1]))
            if reach <= cfg.contact_radius:
                corroborated = any(
                    max(abs(c.pos[0] - map_pos[0]), abs(c.pos[1] - map_pos[1]))
                    <= CORROBORATE_RADIUS
                    for c in self.discovered_clues
                )
                no_clues_left = self.view.clue_count == 0
                if corroborated and conf >= EVIDENCE_CONFIDENCE:
                    return Decision(
                        action=Action.declare_found(map_pos),
                        planner=self.planner_name,
                        reason=(
                            f"confidence {conf:.2f} with a discovered clue within "
                            f"{CORROBORATE_RADIUS} cells of MAP cell {map_pos}; "
                            "corroborated, declaring"
                        ),
                        extras={"confidence": conf, "corroborated": True},
                    )
                if no_clues_left and conf >= DECLARE_CONFIDENCE:
                    return Decision(
                        action=Action.declare_found(map_pos),
                        planner=self.planner_name,
                        reason=(
                            f"belief confidence {conf:.2f} >= {DECLARE_CONFIDENCE}; "
                            "no clues configured; declaring at MAP cell "
                            f"{map_pos}"
                        ),
                        extras={"confidence": conf, "corroborated": False},
                    )
                # scan/hold: may reveal more clues and buys ticks for the belief
                # to concentrate on the source at hand (clues spawn near EVE, so
                # a real EVE cell always has one within scan range)
                region_scans = sum(
                    1
                    for s in self._scanned
                    if max(abs(map_pos[0] - s[0]), abs(map_pos[1] - s[1]))
                    <= CORROBORATE_RADIUS
                )
                if (
                    not no_clues_left
                    and region_scans < _MAX_SCANS_PER_REGION
                    and obs.battery >= self._scan_cost + reserve
                ):
                    self._scanned.append(map_pos)
                    self.scans += 1
                    if corroborated:
                        why = (
                            f"corroborated but confidence {conf:.2f} still below "
                            f"{EVIDENCE_CONFIDENCE}; re-observing at {map_pos}"
                        )
                    else:
                        why = (
                            f"reached plausible cell {map_pos} (conf {conf:.2f}) "
                            "without clue evidence; scanning for corroboration"
                        )
                    return Decision(
                        action=Action.scan(),
                        planner=self.planner_name,
                        reason=why,
                        extras={"confidence": conf, "scanning_at": list(map_pos)},
                    )
                # still uncorroborated after arrival/scans: a repeated failure in
                # the same region means the beacon is spoofed — reject it for
                # good. Only clue-bearing missions can corroborate, so clueless
                # missions never mask (there is nothing to fail against).
                if (
                    not corroborated
                    and not no_clues_left
                    and self._last_unconfirmed is not None
                    and max(
                        abs(map_pos[0] - self._last_unconfirmed[0]),
                        abs(map_pos[1] - self._last_unconfirmed[1]),
                    )
                    <= CORROBORATE_RADIUS
                ):
                    self.belief.suspect_spoof(map_pos)
                    self._declare_mode = False
                    self._last_unconfirmed = None
                    return Decision(
                        action=Action.move(
                            self._escape_direction(pos, map_pos, dist, first)
                        ),
                        planner=self.planner_name,
                        reason=(
                            f"region around {map_pos} again confident (conf {conf:.2f}) "
                            "but failed clue corroboration twice; rejecting it as a "
                            "spoofed beacon and resuming search"
                        ),
                        extras={"suspected_spoof_at": list(map_pos), "confidence": conf},
                    )
                if corroborated:
                    # corroborated but the belief is stuck (e.g. every nearby
                    # reading comes from a rejected source): break contact and
                    # let exploration gather readings from far enough away
                    self._departed_map = map_pos
                    self._departed_conf = conf
                    self._declare_mode = False
                    return None
                self._last_unconfirmed = map_pos
                self._declare_mode = False
                why = (
                    f"reached plausible cell {map_pos} (conf {conf:.2f}) with no "
                    f"clue within {CORROBORATE_RADIUS} cells; not declaring yet, "
                    "stepping away to gather independent evidence"
                )
                return Decision(
                    action=Action.move(self._escape_direction(pos, map_pos, dist, first)),
                    planner=self.planner_name,
                    reason=why,
                    extras={"confidence": conf, "map_pos": list(map_pos)},
                )
            d_map = dist.get(map_pos)
            if d_map is not None and d_map * self._move_cost <= obs.battery:
                step = first.get(map_pos)
                if step is not None:
                    return Decision(
                        action=Action.move(step),
                        planner=self.planner_name,
                        reason=f"approaching MAP cell {map_pos} to declare (conf {conf:.2f})",
                        extras={"target": list(map_pos), "confidence": conf},
                    )

        # 2. battery policy
        usable = obs.battery - (cost_home * self._move_cost if cost_home is not None else 10**9) - reserve
        min_cost = min(self._move_cost, self._scan_cost)
        if usable < min_cost:
            # need recharge or return
            stations = [
                (dist[s], s)
                for s in cfg.map.recharge_stations
                if s in dist
            ]
            stations.sort()
            for d_st, st in stations:
                # feasible round trip: to station + home + reserve
                st_dist, _st_first = self._dist_map(st)
                home_after = st_dist.get(cfg.map.home)
                if (
                    home_after is not None
                    and (d_st + home_after) * self._move_cost + reserve <= obs.battery
                ):
                    if pos in set(cfg.map.recharge_stations) and obs.battery < w.battery:
                        self.recharges += 1
                        return Decision(
                            action=Action.recharge(),
                            planner=self.planner_name,
                            reason=(
                                f"usable battery {usable} below min cost {min_cost}; "
                                f"recharging at {pos}"
                            ),
                        )
                    step = first.get(st)
                    if step is not None:
                        return Decision(
                            action=Action.move(step),
                            planner=self.planner_name,
                            reason=f"battery {obs.battery}: heading to recharge station {st}",
                            extras={"target": list(st)},
                        )
            if cost_home is not None and cost_home * self._move_cost + reserve <= obs.battery:
                self.turn_back_tick = obs.tick
                return Decision(
                    action=Action.return_home(),
                    planner=self.planner_name,
                    reason=(
                        f"battery {obs.battery} cannot fund recharge round-trip; "
                        f"turning back (return cost {cost_home * self._move_cost})"
                    ),
                )
            # stranded: reserve is already spent and no action above is
            # affordable — limp toward home so the EIG loop can never pick an
            # action the validator will reject (that aborts the mission).
            step = first.get(cfg.map.home)
            if step is None:
                # unreachable home: any reachable neighbor is still a legal move
                x, y = pos
                for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    nxt = first.get((x + dx, y + dy))
                    if nxt is not None:
                        step = nxt
                        break
            if step is None:
                step = "north"
            return Decision(
                action=Action.move(step),
                planner=self.planner_name,
                reason=(
                    f"battery {obs.battery} with reserve spent and no feasible "
                    "return/recharge; taking a minimal step"
                ),
                extras={"limp": True},
            )

        return None

    # ------------------------------------------------------------------ act

    def act(self, obs: Observation) -> Decision:
        cfg = self.view
        pos = obs.position

        # belief update (hidden evidence arrives only via this observation)
        if obs.signal is not None or obs.sensor_status.get("signal") != "failed":
            self.belief.update_signal(obs)
        for event in obs.events:
            if event.type == "clue_discovered":
                self.belief.update_clue(event.pos)
                self.discovered_clues.append(event)
        self.visited.add(pos)

        dist, first = self._dist_map(pos)

        safety = self._safety_action(obs, dist, first)
        if safety is not None:
            return safety

        # score candidates: scan in place vs move toward informative cells
        usable = obs.battery - cfg.budget.reserve_margin
        scan_eig = self._eig_bits(pos, scan=True)
        scan_score = scan_eig / max(self._scan_cost, 1)

        best_score = scan_score
        best_action: Action | None = Action.scan()
        best_label = f"scan in place (EIG {scan_eig:.3f} bits / cost {self._scan_cost})"

        visited = self.visited
        scan_r = cfg.sensors.scan_radius
        w, h = cfg.map.width, cfg.map.height

        candidates: list[tuple[float, str, Action, str]] = []
        if best_action is not None:
            candidates.append((scan_score, "scan", best_action, best_label))

        for (cx, cy), d in dist.items():
            if d == 0 or d * self._move_cost > usable:
                continue
            eig = self._eig_bits((cx, cy), scan=False)
            # frontier bonus: unvisited territory around the candidate
            unvisited = 0
            total = 0
            for dx in range(-scan_r, scan_r + 1):
                for dy in range(-scan_r, scan_r + 1):
                    if dx * dx + dy * dy > scan_r * scan_r:
                        continue
                    nx, ny = cx + dx, cy + dy
                    if not (0 <= nx < w and 0 <= ny < h):
                        continue
                    total += 1
                    if (nx, ny) not in visited:
                        unvisited += 1
            frontier = (unvisited / max(total, 1)) if total else 0.0
            score = eig / (d + 1) + _FRONTIER_BONUS * frontier / (d + 1)
            step = first.get((cx, cy))
            if step is None:
                continue
            candidates.append(
                (
                    score,
                    "move",
                    Action.move(step),
                    f"move toward ({cx},{cy}) d={d} EIG {eig:.3f} frontier {frontier:.2f}",
                )
            )

        candidates.sort(key=lambda t: t[0], reverse=True)
        if candidates:
            score, _kind, action, label = candidates[0]
            if best_action is None or score > best_score:
                best_action = action
                best_label = label
            options = tuple(c[3] for c in candidates[:3])
        else:
            options = (best_label,)

        assert best_action is not None
        if best_action.type == "scan":
            self.scans += 1

        return Decision(
            action=best_action,
            planner=self.planner_name,
            reason=f"best EIG-per-cost option: {best_label}",
            options_considered=options,
            extras={
                "confidence": round(self.belief.confidence(), 4),
                "entropy_bits": round(self.belief.entropy_bits(), 4),
            },
        )
