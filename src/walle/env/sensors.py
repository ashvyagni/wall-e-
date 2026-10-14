"""Signal and view sensors (log-normal path loss, bearings, clue detection)."""

from __future__ import annotations

import math
import random
from typing import Literal

from walle.contracts import MissionConfig, Pos, SignalReading, VisibleCell

_EPS = 1e-9


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def euclid(a: Pos, b: Pos) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def noiseless_strength(cfg: MissionConfig, source: Pos, at: Pos) -> float:
    """Deterministic path-loss strength of a source as seen from `at`."""
    s = cfg.sensors.signal
    d = max(euclid(source, at), s.reference_distance)
    return _clamp01(s.reference_power * (s.reference_distance / d) ** s.path_loss_exponent)


def bearing_deg(from_pos: Pos, to_pos: Pos) -> float:
    """Clockwise-from-north bearing in [0, 360)."""
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    return math.degrees(math.atan2(dx, dy)) % 360.0


def _lognormal_factor(rng: random.Random, noise_std_db: float) -> float:
    """Multiplicative factor: dB-normal noise converted to linear domain."""
    if noise_std_db <= 0:
        return 1.0
    sigma_ln = math.log(10.0) * noise_std_db / 20.0
    return math.exp(rng.gauss(0.0, sigma_ln) - 0.5 * sigma_ln**2)


def _gauss_deg(rng: random.Random, std_deg: float) -> float:
    if std_deg <= 0:
        return 0.0
    return rng.gauss(0.0, std_deg)


def measure_signal(
    cfg: MissionConfig,
    sources: list[tuple[Pos, float]],
    agent_pos: Pos,
    rng: random.Random,
    *,
    scan: bool,
    noise_scale: float = 1.0,
) -> SignalReading | None:
    """Noisy signal reading from the dominant source (EVE or strongest decoy).

    `sources` are (position, strength_floor) pairs. Returns None when every
    source is beyond max_range.
    """
    s = cfg.sensors.signal
    base = s.scan_noise_scale if scan else 1.0
    noise_db = s.noise_std_db * base * noise_scale
    bearing_std = s.bearing_noise_std_deg * base * noise_scale

    best: tuple[float, Pos] | None = None
    for pos, _floor in sources:
        if euclid(pos, agent_pos) > s.max_range:
            continue
        true_s = noiseless_strength(cfg, pos, agent_pos)
        noisy = _clamp01(true_s * _lognormal_factor(rng, noise_db))
        if best is None or noisy > best[0]:
            best = (noisy, pos)

    if best is None:
        return None

    strength, src = best
    true_bearing = bearing_deg(agent_pos, src)
    noisy_bearing = (true_bearing + _gauss_deg(rng, bearing_std)) % 360.0
    return SignalReading(
        bearing_deg=round(noisy_bearing, 3),
        strength=round(strength, 6),
        measured_from=agent_pos,
        timestamp_tick=-1,  # stamped by the environment
    )


def visible_cells(
    cfg: MissionConfig,
    agent_pos: Pos,
    obstacles: frozenset[Pos],
) -> tuple[VisibleCell, ...]:
    """Terrain within local_view_radius (Euclidean circle). Terrain is known;
    this reports what the agent actually observes this tick."""
    r = cfg.sensors.local_view_radius
    cells: list[VisibleCell] = []
    w, h = cfg.map.width, cfg.map.height
    recharge = set(cfg.map.recharge_stations)
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            if dx * dx + dy * dy > r * r:
                continue
            x, y = agent_pos[0] + dx, agent_pos[1] + dy
            if not (0 <= x < w and 0 <= y < h):
                continue
            pos = (x, y)
            kind: Literal["free", "obstacle", "recharge", "home"]
            if pos in obstacles:
                kind = "obstacle"
            elif pos == cfg.map.home:
                kind = "home"
            elif pos in recharge:
                kind = "recharge"
            else:
                kind = "free"
            cells.append(VisibleCell(pos=pos, kind=kind))
    return tuple(cells)
