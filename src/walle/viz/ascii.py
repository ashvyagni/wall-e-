"""ASCII visualization of a mission: final map, path, clues, terminals."""

from __future__ import annotations

from walle.contracts import MissionConfig, TraceEvent


def render_ascii(cfg: MissionConfig, events: list[TraceEvent]) -> str:
    w, h = cfg.map.width, cfg.map.height
    grid = [["." for _ in range(w)] for _ in range(h)]

    for ox, oy in cfg.map.obstacles:
        grid[oy][ox] = "#"
    hx, hy = cfg.map.home
    grid[hy][hx] = "H"
    for sx, sy in cfg.map.recharge_stations:
        grid[sy][sx] = "R"

    path: list[tuple[int, int]] = []
    clues: set[tuple[int, int]] = set()
    terminal = None
    for ev in events:
        if ev.observation is not None:
            path.append(ev.observation.position)
            for clue_ev in ev.observation.events:
                clues.add(clue_ev.pos)
        if ev.terminal is not None:
            terminal = ev.terminal

    for px, py in path:
        if grid[py][px] in {".", "o"}:
            grid[py][px] = "o"
    for cx, cy in clues:
        grid[cy][cx] = "c"
    if path:
        ax, ay = path[-1]
        grid[ay][ax] = "W"

    legend = (
        "legend: H=home R=recharge o=path c=clue W=WALL-E final #=obstacle .=unvisited\n"
        f"terminal: {terminal}\n"
    )
    rows = ["".join(row) for row in reversed(grid)]
    return legend + "\n".join(rows)
