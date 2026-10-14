# Benchmark Protocol (pre-registered)

Frozen before any official run. Dev-loop results (seeds 1000–1049, run while
building the planner) are **excluded** from all reported numbers. Changes to
this file after a run require a new protocol version and invalidate prior runs.

**Protocol version:** 1.0
**Date frozen:** 2026-10-14

## 1. Subjects

| Subject   | Definition                                                        |
|-----------|-------------------------------------------------------------------|
| classical | `walle.agent.classical.ClassicalPlanner` via `planner_for()`      |
| llm       | LLM ReAct planner (when implemented), same `Planner` interface     |

Both receive byte-identical `Observation` documents and the redacted
`AgentView` only (enforced by `tests/test_hidden_state.py`).

## 2. Mission set

- Template: `configs/sample_basic.json`, with explicit placements stripped
  (seed-derived EVE, 4 clues within `clues.spawn_radius` of EVE, 1 decoy
  respecting `decoys.min_clue_distance`) — identical to `scripts/sweep.py`
  variant generation.
- Classical: seeds `bench/seeds_classical.json` (2000–2099, n = 100).
- LLM: seeds `bench/seeds_llm.json` (every 5th seed, n = 20) — a **paired
  subset**: every LLM mission has a classical counterpart with the same seed.
- Seed lists are committed before any run; no seed may be added, removed, or
  reordered after results are seen.

## 3. Metrics (per mission, all retained in the denominator)

1. `success` — terminal `EVE_LOCATED_SUCCESS` (exact cell).
2. `localization_error` — L2 distance in cells between declared/inferred
   position and true EVE (reported for all terminals; exact-cell = 0).
3. `steps` — decision steps to terminal.
4. `battery_used` — battery consumed to terminal.
5. `invalid_action_count` — validator rejections (abort at 10 consecutive).
6. LLM only: wall-clock inference latency per decision (mean, p95), total
   tokens and USD cost (provider usage records), from the provenance cache.

## 4. Statistical procedure

- Success rate: proportion + **Wilson 95% CI**.
- Continuous metrics (steps, error, battery): mean + **t-based 95% CI**;
  medians reported alongside (distributions are right-skewed).
- Failures (all non-success terminals, exceptions, timeouts) always remain in
  the denominator; exceptions additionally reported as their own count.
- LLM vs classical: **paired** comparison on the 20 shared seeds — report the
  2×2 outcome table (both succeed / classical only / LLM only / neither) and
  the paired success-rate difference with a bootstrap 95% CI (10 000
  resamples, fixed RNG seed 0).
- No outlier exclusion, no early stopping, no seed retries. A mission runs
  once; an exception counts as a failure with `terminal=EXCEPTION`.

## 5. Reproducibility

- Every run writes per-seed traces + reports under `runs/bench/<subject>/`
  and a machine-readable summary `reports/bench_<subject>_v1.json`
  containing: protocol version, git commit, seed list hash, planner kind,
  host/OS, wall time.
- LLM runs additionally store the provenance block (provider, model id,
  prompt/tool-schema version, seed, config hash, decoding settings,
  timestamp, usage) per decision in the cache, as required by the brief.
- The benchmark is fully offline for classical (no network); LLM requires
  the API and is never a gate for classical results.

## 6. Hypotheses (pre-registered)

- **H1 (classical):** success rate ≥ 0.85 on n = 100.
- **H2 (classical):** mean localization error over all terminals ≤ 2.0 cells.
- **H3 (LLM):** report-only — paired outcome table and difference vs
  classical; no pass threshold until the first LLM run establishes variance.

## 7. Change log

| Version | Change | Reason |
|---------|--------|--------|
| 1.0     | initial registration                | before any official run |
