"""Deterministic grid pathing (BFS with fixed tie-break: N < S < E < W)."""

from __future__ import annotations

from collections import deque
from itertools import pairwise

from walle.contracts import Pos

_NEIGHBOR_ORDER: tuple[tuple[str, Pos], ...] = (
    ("north", (0, 1)),
    ("south", (0, -1)),
    ("east", (1, 0)),
    ("west", (-1, 0)),
)

DIR_BY_DELTA: dict[Pos, str] = {delta: name for name, delta in _NEIGHBOR_ORDER}


def in_bounds(pos: Pos, width: int, height: int) -> bool:
    x, y = pos
    return 0 <= x < width and 0 <= y < height


def shortest_path(
    start: Pos,
    goal: Pos,
    obstacles: frozenset[Pos],
    width: int,
    height: int,
) -> tuple[Pos, ...]:
    """Return cell path start..goal inclusive; empty tuple if unreachable.

    Deterministic: neighbors expanded north, south, east, west in that order.
    """
    if start == goal:
        return (start,)
    if goal in obstacles or not in_bounds(goal, width, height):
        return ()
    prev: dict[Pos, Pos] = {}
    seen: set[Pos] = {start}
    queue: deque[Pos] = deque([start])
    while queue:
        cur = queue.popleft()
        cx, cy = cur
        for _name, (dx, dy) in _NEIGHBOR_ORDER:
            nxt = (cx + dx, cy + dy)
            if nxt in seen or not in_bounds(nxt, width, height) or nxt in obstacles:
                continue
            seen.add(nxt)
            prev[nxt] = cur
            if nxt == goal:
                path = [nxt]
                while path[-1] != start:
                    path.append(prev[path[-1]])
                path.reverse()
                return tuple(path)
            queue.append(nxt)
    return ()


def path_directions(path: tuple[Pos, ...]) -> list[str]:
    """Convert a cell path into move directions (excluding the first cell)."""
    dirs: list[str] = []
    for a, b in pairwise(path):
        delta = (b[0] - a[0], b[1] - a[1])
        dirs.append(DIR_BY_DELTA[delta])
    return dirs


def reachable_cells(
    start: Pos,
    obstacles: frozenset[Pos],
    width: int,
    height: int,
) -> set[Pos]:
    """All cells reachable from start (BFS flood fill, start included)."""
    seen: set[Pos] = {start}
    queue: deque[Pos] = deque([start])
    while queue:
        cx, cy = queue.popleft()
        for _name, (dx, dy) in _NEIGHBOR_ORDER:
            nxt = (cx + dx, cy + dy)
            if nxt in seen or not in_bounds(nxt, width, height) or nxt in obstacles:
                continue
            seen.add(nxt)
            queue.append(nxt)
    return seen
