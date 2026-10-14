"""World state: seeded placement of EVE, clues, decoys; budget bookkeeping.

Everything in this module is env-internal ground truth. Nothing here may be
serialized into an Observation or any planner input (see contracts/SEMANTICS.md).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from walle.contracts import ClueEvent, ClueKind, MissionConfig, Pos

_CLUE_KINDS: tuple[ClueKind, ...] = (
    "footprint",
    "scorch_mark",
    "plant_sample",
    "waste_scrap",
    "signal_mirror",
)

_CLUE_DESCRIPTIONS: dict[ClueKind, str] = {
    "footprint": "Tread pattern matches EVE's hull unit.",
    "scorch_mark": "Fresh scorch mark on the hull plating — EVE passed through recently.",
    "plant_sample": "Sampled sprout with EVE's coolant residue on the leaves.",
    "waste_scrap": "Compact scrap stamped with an ejector serial near EVE's bay.",
    "signal_mirror": "Shard angled to catch the sun — a deliberate beacon fragment.",
}


@dataclass
class Clue:
    clue_id: str
    pos: Pos
    kind: ClueKind
    description: str
    discovered: bool = False
    discovered_tick: int = -1

    def to_event(self, tick: int) -> ClueEvent:
        return ClueEvent(
            clue_id=self.clue_id,
            kind=self.kind,
            pos=self.pos,
            description=self.description,
            tick=tick,
        )


@dataclass
class World:
    cfg: MissionConfig
    rng: random.Random
    eve_pos: Pos
    eve_status: str
    clues: list[Clue]
    decoys: list[Pos]
    contact_radius: int
    agent_pos: Pos
    battery: int
    tick: int = 0
    invalid_streak: int = 0
    invalid_total: int = 0
    failed_claims: int = 0
    consecutive_signal_fail: int = 0
    _free_cells: frozenset[Pos] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        obs = set(self.cfg.map.obstacles)
        free = {
            (x, y)
            for x in range(self.cfg.map.width)
            for y in range(self.cfg.map.height)
            if (x, y) not in obs
        }
        self._free_cells = frozenset(free)

    @property
    def obstacles(self) -> frozenset[Pos]:
        return frozenset(self.cfg.map.obstacles)

    @property
    def free_cells(self) -> frozenset[Pos]:
        return self._free_cells

    @property
    def max_ticks(self) -> int:
        return self.cfg.budget.max_ticks

    def sources(self) -> list[tuple[Pos, float]]:
        """(pos, floor) pairs for every signal-emitting source."""
        out = [(self.eve_pos, self.cfg.decoys.strength_floor)]
        out.extend((d, self.cfg.decoys.strength_floor) for d in self.decoys)
        return out

    def undiscovered_clues(self) -> list[Clue]:
        return [c for c in self.clues if not c.discovered]


def _seeded_positions(
    rng: random.Random,
    count: int,
    cfg: MissionConfig,
    taken: set[Pos],
    *,
    near: Pos | None = None,
    near_radius: int = 0,
    keep_away_from: tuple[Pos, ...] = (),
    keep_away_min: int = 0,
) -> list[Pos]:
    """Deterministic sample of free cells, avoiding home/taken cells.

    With ``near`` set, prefer cells within ``near_radius`` chebyshev distance
    of that position (the clue-vs-EVE proximity prior); unfilled slots fall
    back to uniform sampling over the whole map. ``keep_away`` rejects cells
    closer than ``keep_away_min`` to any of those positions (decoy-vs-clue
    exclusion) before sampling.
    """
    blocked = taken | {cfg.map.home, *cfg.map.obstacles, *cfg.map.recharge_stations}
    free = [
        (x, y)
        for x in range(cfg.map.width)
        for y in range(cfg.map.height)
        if (x, y) not in blocked
    ]
    if keep_away_from:
        free = [
            c
            for c in free
            if all(
                max(abs(c[0] - o[0]), abs(c[1] - o[1])) >= keep_away_min
                for o in keep_away_from
            )
        ]
    if near is not None:
        local = [
            c for c in free if max(abs(c[0] - near[0]), abs(c[1] - near[1])) <= near_radius
        ]
        rng.shuffle(local)
        picked = local[:count]
        if len(picked) >= count:
            return picked
        rest = [c for c in free if c not in set(picked)]
        rng.shuffle(rest)
        return picked + rest[: count - len(picked)]
    rng.shuffle(free)
    return free[:count]


def build_world(cfg: MissionConfig) -> World:
    """Create the env-internal world for a mission. Deterministic given cfg.seed."""
    rng = random.Random(cfg.seed)
    blocked: set[Pos] = set()

    # EVE placement
    if cfg.target.placement is not None:
        eve_pos = cfg.target.placement
    else:
        picks = _seeded_positions(rng, 1, cfg, blocked)
        if not picks:
            raise ValueError("map has no free cell for EVE placement")
        eve_pos = picks[0]
    if eve_pos in cfg.map.obstacles:
        raise ValueError(f"EVE placement {eve_pos} is an obstacle")
    if eve_pos == cfg.map.home:
        raise ValueError("EVE placement cannot be the home cell")
    blocked.add(eve_pos)

    # Clues: explicit placements first, then seeded ones up to seed_count
    clues: list[Clue] = []
    for i, (pos, kind) in enumerate(cfg.clues.placements):
        if pos in cfg.map.obstacles:
            raise ValueError(f"clue placement {pos} is an obstacle")
        if pos in blocked:
            raise ValueError(f"clue placement {pos} collides with another entity")
        blocked.add(pos)
        clues.append(
            Clue(
                clue_id=f"clue_{i + 1:02d}",
                pos=pos,
                kind=kind,
                description=_CLUE_DESCRIPTIONS[kind],
            )
        )
    if cfg.clues.seed_count > 0:
        seeded = _seeded_positions(
            rng,
            cfg.clues.seed_count,
            cfg,
            blocked,
            near=eve_pos,
            near_radius=cfg.clues.spawn_radius,
        )
        for j, pos in enumerate(seeded):
            kind = _CLUE_KINDS[(cfg.seed + j) % len(_CLUE_KINDS)]
            blocked.add(pos)
            clues.append(
                Clue(
                    clue_id=f"clue_seed_{j + 1:02d}",
                    pos=pos,
                    kind=kind,
                    description=_CLUE_DESCRIPTIONS[kind],
                )
            )

    # Decoys: explicit placements first (validated against clues), then seeded
    # with a hard exclusion zone around every clue so a declare cell backed by
    # an adjacent clue can never be a decoy.
    decoys: list[Pos] = list(cfg.decoys.placements)
    clue_cells = [c.pos for c in clues]
    for pos in decoys:
        if pos in cfg.map.obstacles:
            raise ValueError(f"decoy placement {pos} is an obstacle")
        too_close = [
            c
            for c in clue_cells
            if max(abs(pos[0] - c[0]), abs(pos[1] - c[1])) < cfg.decoys.min_clue_distance
        ]
        if too_close:
            raise ValueError(
                f"decoy placement {pos} violates decoys.min_clue_distance="
                f"{cfg.decoys.min_clue_distance}: too close to clue(s) {too_close}"
            )
        blocked.add(pos)
    if cfg.decoys.count > len(decoys):
        extra = _seeded_positions(
            rng,
            cfg.decoys.count - len(decoys),
            cfg,
            blocked,
            keep_away_from=tuple(clue_cells),
            keep_away_min=cfg.decoys.min_clue_distance,
        )
        if len(extra) < cfg.decoys.count - len(decoys):
            raise ValueError(
                "map has no free cell for a decoy that respects "
                f"decoys.min_clue_distance={cfg.decoys.min_clue_distance} from all clues"
            )
        for pos in extra:
            blocked.add(pos)
        decoys.extend(extra)
    decoys = decoys[: max(cfg.decoys.count, len(cfg.decoys.placements))]

    # Agent start
    start = cfg.agents[0].start if cfg.agents[0].start is not None else cfg.map.home
    if start in cfg.map.obstacles:
        raise ValueError(f"agent start {start} is an obstacle")

    return World(
        cfg=cfg,
        rng=rng,
        eve_pos=eve_pos,
        eve_status=cfg.target.status,
        clues=clues,
        decoys=decoys,
        contact_radius=cfg.target.contact_radius,
        agent_pos=start,
        battery=cfg.budget.battery,
    )
