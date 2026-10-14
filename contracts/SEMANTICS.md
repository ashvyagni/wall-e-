# Contract Semantics — schema_version 1.0

Frozen at Hour 0. Changes require: (1) bumping `schema_version`, (2) updating the zod
mirror in `contracts/ts`, (3) updating all fixtures in `contracts/fixtures/`, (4) both
language test suites passing.

## Documents

| Document | Purpose | Consumer |
|---|---|---|
| `MissionConfig` | Full mission definition (includes evaluator-only ground truth) | Environment, evaluator |
| `Action` | Single planner decision | Environment validator |
| `Observation` | Per-tick agent-visible percept | Planners (classical + LLM) |
| `TraceEvent` | Append-only mission trace record | Logging, replay UI, report |
| `MissionReport` | Final mission summary | Humans, grader |

## Coordinates and time

- **Grid:** integer `[x, y]`, origin bottom-left, `+x` east, `+y` north. A position is a
  cell center. `map.cell_size` (meters) exists for physics scaling; logical movement is
  always whole cells.
- **Movement:** 4-connected cardinal steps (`north`/`south`/`east`/`west`), exactly one
  cell per `move` action. Moving into an obstacle or out of bounds is invalid.
- **Timestamps:** `tick` is an integer simulation counter starting at `0` (mission start).
  Each accepted action advances the tick by `1`, except `recharge`, which advances by
  `budget.recharge_ticks` and `return_home`, which advances by one tick per traversed
  cell (deterministic shortest path, see below). Wall-clock time appears only in trace
  metadata written outside the frozen schema (cache/provenance files).

## Actions

| Action | Precondition | Effect | Battery cost |
|---|---|---|---|
| `move(dir)` | target cell free & in bounds | position += dir; tick += 1 | `action_costs.move` |
| `scan` | — | clue detection up to `sensors.scan_radius`; tick += 1 | `action_costs.scan` |
| `recharge` | agent on a `recharge_stations` cell | battery := full after `recharge_ticks`; tick += `recharge_ticks` | `action_costs.recharge` |
| `return_home` | battery ≥ shortest-path cost to `home` + `reserve_margin` | agent follows deterministic shortest path (BFS, tie-break: north < south < east < west order) to `home`, one tick + move cost per cell | `action_costs.return_home` (setup) + `action_costs.move` per cell |
| `declare_found(pos)` | `pos` within `contact_radius` (Chebyshev) of agent position | terminates mission: success iff `pos` within `contact_radius` of EVE's true cell | `action_costs.declare_found` |

`return_home` arriving at `home` (or a declared stand-down at home) with the mission not
otherwise terminated yields `RETURNED_NO_EVIDENCE`.

**Rejected actions** never mutate state: they produce a new tick-free observation with
`last_action_valid=false` and `last_action_error` set (battery unchanged, tick unchanged)
and increment the invalid-action counter. After 10 consecutive invalid actions (LLM) or
any invalid action that cannot be repaired (classical — should never happen), the mission
terminates `INVALID_ACTION_ABORTED`.

## Budget accounting

- `battery` is a non-negative integer. `tick` counts against `max_ticks`.
- When `battery < battery_needed_for_cheapest_legal_action`, or a `move` would reach 0
  battery while not at home and not adjacent-success, the mission terminates
  `BATTERY_EXHAUSTED_ABORTED`.
- When an action would push `tick` past `max_ticks`, the mission terminates
  `TIMEOUT_ABORTED`.
- `reserve_margin` is subtracted from usable battery when validating `return_home`
  feasibility (a planning safety margin, not a hard reserve).

## Signal observation

- True strength: `s = clamp(reference_power * (reference_distance / max(d, reference_distance)) ** path_loss_exponent, 0, 1)`
  where `d` is Euclidean cell distance to the strongest source (EVE or a decoy, whichever
  yields the higher observed strength).
- Noise: multiplicative log-normal via `noise_std_db` (dB) on `s`, clamped to `[0, 1]`;
  `scan` multiplies noise std by `scan_noise_scale` (sharper reading).
- `bearing_deg`: angle from `measured_from` to the dominant source, clockwise from north,
  plus Gaussian noise `bearing_noise_std_deg`, wrapped to `[0, 360)`.
- Sources beyond `max_range` produce no reading (`signal: null`).
- `sensor_status.signal == "degraded"` when noise is temporarily inflated (mission
  `failure_rate`), `"failed"` when the reading is withheld entirely. The observation
  always carries the status honestly.
- **The reading never contains distance or source coordinates.** `measured_from` is the
  agent's own cell, so the agent can invert bearing + strength only by inference.

## Hidden-state boundary

- **Planner-visible:** `MissionConfig.agent_view()` = map geometry (bounds, obstacles,
  home, recharge stations), `budget`, `sensors`, own agent id/start, plus game-rule
  parameters `target.contact_radius` and total `clue_count` (never where they are).
  Plus the per-tick `Observation`.
- **Planner-hidden (env-internal):** `target.placement`, `clues.*` (placements,
  seeds, and the `spawn_radius` proximity prior), `decoys.*` (placements, seeds,
  and the `min_clue_distance` exclusion), and any derivation thereof. Seeds are
  hidden because they would allow reconstruction of truth.
- **Clue placement prior (env-internal):** seeded clues spawn within
  `clues.spawn_radius` chebyshev cells of EVE ("EVE left them behind"), with
  `spawn_radius <= 3` so any discovered clue corroborates a declare at true EVE.
  The planner is not told the radius — it only uses the observed clue locations.
- **Decoy exclusion (env-internal):** no decoy is ever placed within
  `decoys.min_clue_distance` (> corroboration radius 3) of any clue. Together
  with the spawn prior this keeps corroboration sound: "a discovered clue within
  3 cells of the declare cell" can only be true EVE, never a decoy.
- The classical planner and the LLM planner receive byte-identical `Observation`
  documents. Enforcement: `tests/test_hidden_state.py` (serialization property tests,
  clue-discovery ordering, planner API signature).

## Map knowledge

Map geometry is fully known (no fog-of-war on terrain). Terrain of unvisited cells is
unknown only in the sense that **clues** reside there and are discovered exclusively by
proximity (`clues.detection_radius`, checked after every accepted action) or `scan`.
Unvisited cells therefore form a genuine frontier for evidence gathering.

## Terminal states

| State | Meaning | `successful` |
|---|---|---|
| `EVE_LOCATED_SUCCESS` | `declare_found` within `contact_radius` of true EVE | true |
| `EVE_LOCATE_FAILED` | `declare_found` at a wrong cell (within reach of agent) | false |
| `RETURNED_NO_EVIDENCE` | returned home / stood down without a successful claim | false |
| `BATTERY_EXHAUSTED_ABORTED` | battery could not continue or return | false |
| `TIMEOUT_ABORTED` | `max_ticks` reached | false |
| `INVALID_ACTION_ABORTED` | invalid-action limit reached | false |

## Belief summary (trace)

- `map_pos`: MAP cell of the belief distribution over EVE location.
- `confidence`: posterior probability at the MAP cell.
- `entropy_bits`: Shannon entropy of the belief distribution (base 2).
- `top_k`: up to 5 `(cell, probability)` pairs, descending.

## Report sections

- `outcome` / `evidence` / `inference` / `strategy`: agent-side narrative (evidence
  found, inferred location/status, confidence, recommended next step).
- `evaluation`: **evaluator-only** ground truth and derived metrics
  (`true_eve_pos`, `localization_error`, clue counts, invalid-action count). Present in
  the report file for grading; never serialized into any planner input.
