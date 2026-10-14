"""Bayesian belief grid over EVE's location.

Prior: uniform over free cells. Likelihood: log-normal path-loss strength plus
wrapped-normal bearing from the observation model. Evidence updates:
- signal readings (every observation that carries one)
- discovered clues (weak spatial boost: EVE left them behind)
Robustness: likelihood floor + small uniform mixture so a single spoofed
reading cannot fully hijack the posterior.
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np

from walle.contracts import (
    BeliefSummary,
    MapConfig,
    Observation,
    Pos,
    SensorsConfig,
)

_EPS = 1e-12
_MIX_ALPHA = 0.05  # uniform mixture weight (decoy/outlier robustness)
_LIK_FLOOR = 1e-6
_CLUE_BOOST = 1.5
_CLUE_RADIUS = 3


class BeliefGrid:
    """Grid posterior over EVE location. Indexing: prob[y, x]."""

    def __init__(
        self, map_cfg: MapConfig, sensors_cfg: SensorsConfig, obstacles: frozenset[Pos]
    ) -> None:
        self.map_cfg = map_cfg
        self.sensors_cfg = sensors_cfg
        self.width = map_cfg.width
        self.height = map_cfg.height
        self.prob = np.full((self.height, self.width), _EPS, dtype=np.float64)
        for x in range(self.width):
            for y in range(self.height):
                if (x, y) in obstacles:
                    self.prob[y, x] = 0.0
        total = self.prob.sum()
        if total <= 0:
            raise ValueError("belief prior has no mass (all cells are obstacles)")
        self.prob /= total
        self._sigma_ln = math.log(10.0) * sensors_cfg.signal.noise_std_db / 20.0
        self._bearing_std = max(sensors_cfg.signal.bearing_noise_std_deg, 1e-3)
        # spoof posterior: probability that the current dominant reading comes
        # from a decoy rather than EVE. Raised on uncorroborated arrivals,
        # decays every update. While high, likelihoods are tempered toward
        # uniform so a spoofed beacon cannot keep steering the belief.
        self.spoof_posterior: float = 0.0
        self.suspected_regions: list[Pos] = []

    def suspect_spoof(self, pos: Pos, radius: int = 3) -> None:
        """Reject a region as a spoofed beacon: mask it from all future
        evidence so the posterior can never re-converge there."""
        self.suspected_regions.append(pos)
        X, Y = self._coords()
        mask = np.maximum(np.abs(X - pos[0]), np.abs(Y - pos[1])) <= radius
        self.prob = np.where(mask, 0.0, self.prob)
        self.spoof_posterior = max(self.spoof_posterior, 0.8)
        self._normalize()

    def near_suspected(self, pos: Pos, radius: int = 3) -> bool:
        return any(
            max(abs(pos[0] - s[0]), abs(pos[1] - s[1])) <= radius
            for s in self.suspected_regions
        )

    def _mask_likelihood(self, lik: np.ndarray, at: Pos) -> np.ndarray:
        """Zero out likelihood inside every rejected (spoofed) region."""
        X, Y = self._coords()
        for s in self.suspected_regions:
            mask = np.maximum(np.abs(X - s[0]), np.abs(Y - s[1])) <= 3
            lik = np.where(mask, _LIK_FLOOR, lik)
        return lik

    # ------------------------------------------------------------------ coords

    def _coords(self) -> tuple[np.ndarray, np.ndarray]:
        xs = np.arange(self.width, dtype=np.float64)
        ys = np.arange(self.height, dtype=np.float64)
        X, Y = np.meshgrid(xs, ys)
        return cast(np.ndarray, X), cast(np.ndarray, Y)

    def _pred_strength(self, from_pos: Pos) -> np.ndarray:
        X, Y = self._coords()
        s_cfg = self.sensors_cfg.signal
        d = np.hypot(X - from_pos[0], Y - from_pos[1])
        d = np.maximum(d, s_cfg.reference_distance)
        return cast(
            np.ndarray,
            np.clip(
                s_cfg.reference_power
                * (s_cfg.reference_distance / d) ** s_cfg.path_loss_exponent,
                1e-9,
                1.0,
            ),
        )

    def _pred_bearing(self, from_pos: Pos) -> np.ndarray:
        X, Y = self._coords()
        dx = X - from_pos[0]
        dy = Y - from_pos[1]
        return cast(np.ndarray, np.degrees(np.arctan2(dx, dy)) % 360.0)

    # ------------------------------------------------------------------ updates

    def update_signal(self, obs: Observation) -> None:
        reading = obs.signal
        at = obs.position
        s_cfg = self.sensors_cfg.signal

        if reading is None:
            # No reading: cells close to the agent are unlikely to hide EVE
            # (they would have produced one); far cells are neutral.
            X, Y = self._coords()
            d = np.hypot(X - at[0], Y - at[1])
            lik = np.where(d > s_cfg.max_range, 1.0, 0.05)
        else:
            pred_s = self._pred_strength(at)
            sigma_ln = max(self._sigma_ln, 1e-3)
            s_obs = max(reading.strength, 1e-6)
            log_lik_s = (
                -np.log(s_obs)
                - np.log(sigma_ln)
                - 0.5 * math.log(2 * math.pi)
                - (np.log(s_obs) - np.log(pred_s)) ** 2 / (2 * sigma_ln**2)
            )
            lik_s = np.exp(np.clip(log_lik_s, -40, 0))

            pred_b = self._pred_bearing(at)
            db = _wrap_deg_grid(reading.bearing_deg - pred_b)
            lik_b = np.exp(-0.5 * (db / self._bearing_std) ** 2) / (
                self._bearing_std * math.sqrt(2 * math.pi)
            )

            d = np.hypot(*(self._coords()[0] - at[0], self._coords()[1] - at[1]))
            visible = d <= s_cfg.max_range
            lik = np.where(visible, lik_s * lik_b, _LIK_FLOOR)

        lik = np.maximum(lik, _LIK_FLOOR)
        # rejected spoof regions are filtered out of future evidence
        lik = self._mask_likelihood(lik, at)
        # robust mixture: uniform component guards against decoys/outliers
        lik = (1 - _MIX_ALPHA) * lik + _MIX_ALPHA * float(lik.mean())
        # spoof tempering: while a spoof posterior is high, the reading may not
        # be EVE at all — pull the likelihood toward uninformative uniform
        pi = self.spoof_posterior
        if pi > 0:
            lik = (1 - pi) * lik + pi * float(lik.mean())
            self.spoof_posterior = pi * 0.97  # decay per update
        self.prob = self.prob * lik
        self._normalize()

    def update_clue(self, pos: Pos, radius: int = _CLUE_RADIUS) -> None:
        """Weak evidence: EVE likely passed near a clue it left behind."""
        X, Y = self._coords()
        bump = np.where(
            np.maximum(np.abs(X - pos[0]), np.abs(Y - pos[1])) <= radius,
            _CLUE_BOOST,
            1.0,
        )
        self.prob = self.prob * bump
        self._normalize()

    def _normalize(self) -> None:
        total = float(self.prob.sum())
        if total <= _EPS:
            # numerical collapse: fall back to uniform over free cells
            mask = self.prob > 0
            self.prob = mask.astype(np.float64)
            total = float(self.prob.sum())
        self.prob /= total

    # ------------------------------------------------------------------ queries

    def entropy_bits(self) -> float:
        p = self.prob[self.prob > 0]
        return float(-(p * np.log2(p)).sum())

    def map_pos(self) -> Pos:
        idx = np.unravel_index(int(np.argmax(self.prob)), self.prob.shape)
        return (int(idx[1]), int(idx[0]))

    def confidence(self) -> float:
        return float(np.max(self.prob))

    def top_k(self, k: int = 5) -> list[tuple[Pos, float]]:
        flat = self.prob.ravel()
        order = np.argsort(flat)[::-1][:k]
        out: list[tuple[Pos, float]] = []
        for i in order:
            y, x = divmod(int(i), self.width)
            out.append(((x, y), float(flat[i])))
        return out

    def summary(self) -> BeliefSummary:
        return BeliefSummary(
            map_pos=self.map_pos(),
            confidence=round(self.confidence(), 6),
            entropy_bits=round(self.entropy_bits(), 6),
            top_k=tuple((p, round(v, 6)) for p, v in self.top_k(5)),
        )


def _wrap_deg_grid(a: np.ndarray) -> np.ndarray:
    return (a + 180.0) % 360.0 - 180.0
