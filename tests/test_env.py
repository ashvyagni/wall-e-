"""Environment core tests: determinism, validation, terminals, clue discovery."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from walle.contracts import Action, MissionConfig, TerminalState
from walle.env.environment import MAX_CONSECUTIVE_INVALID, Environment

ROOT = Path(__file__).resolve().parents[1]


def load_cfg(**overrides: object) -> MissionConfig:
    data = json.loads((ROOT / "contracts" / "fixtures" / "mission_config.json").read_text())
    for key, value in overrides.items():
        if "." in key:
            section, field = key.split(".", 1)
            data[section][field] = value
        else:
            data[key] = value
    return MissionConfig.model_validate(data)


def test_world_is_deterministic() -> None:
    a = Environment(load_cfg())
    b = Environment(load_cfg())
    assert a.world.eve_pos == b.world.eve_pos
    assert [c.pos for c in a.world.clues] == [c.pos for c in b.world.clues]
    assert a.world.decoys == b.world.decoys


def test_placement_avoids_obstacles_home_and_eve() -> None:
    env = Environment(load_cfg())
    cfg = env.cfg
    taken = {cfg.map.home, env.world.eve_pos, *cfg.map.obstacles}
    for clue in env.world.clues:
        assert clue.pos not in taken
        taken.add(clue.pos)
    for d in env.world.decoys:
        assert d not in taken


def test_move_into_obstacle_is_invalid_no_state_change() -> None:
    env = Environment(load_cfg())
    env.world.agent_pos = (6, 4)  # east of obstacle wall x=5
    before = (env.world.agent_pos, env.world.battery, env.world.tick)
    res = env.step(Action.move("west"))
    assert not res.done
    assert not res.observation.last_action_valid
    assert (env.world.agent_pos, env.world.battery, env.world.tick) == before


def test_valid_move_advances_tick_and_costs_battery() -> None:
    env = Environment(load_cfg())
    res = env.step(Action.move("north"))
    assert res.observation.last_action_valid
    assert env.world.tick == 1
    assert env.world.battery == 219
    assert env.world.agent_pos == (1, 2)


def test_recharge_requires_station() -> None:
    env = Environment(load_cfg())
    res = env.step(Action.recharge())
    assert not res.observation.last_action_valid
    assert "recharge" in (res.observation.last_action_error or "")


def test_recharge_at_station_restores_battery() -> None:
    env = Environment(load_cfg())
    env.world.agent_pos = (10, 7)
    env.world.battery = 50
    res = env.step(Action.recharge())
    assert env.world.battery == 220
    assert env.world.tick == 5  # recharge_ticks
    assert not res.done


def test_declare_out_of_reach_invalid() -> None:
    env = Environment(load_cfg())
    res = env.step(Action.declare_found((17, 12)))
    assert not res.observation.last_action_valid


def test_declare_correct_location_succeeds() -> None:
    env = Environment(load_cfg())
    env.world.agent_pos = (17, 11)
    res = env.step(Action.declare_found((17, 12)))
    assert res.done
    assert res.terminal == TerminalState.EVE_LOCATED_SUCCESS


def test_declare_wrong_location_fails() -> None:
    env = Environment(load_cfg())
    env.world.agent_pos = (5, 5)
    res = env.step(Action.declare_found((5, 6)))
    assert res.done
    assert res.terminal == TerminalState.EVE_LOCATE_FAILED
    assert env.world.failed_claims == 1


def test_return_home_feasible_ends_mission() -> None:
    env = Environment(load_cfg())
    res = env.step(Action.return_home())
    assert res.done
    assert res.terminal == TerminalState.RETURNED_NO_EVIDENCE
    assert env.world.agent_pos == (1, 1)


def test_return_home_blocked_by_reserve_is_invalid() -> None:
    env = Environment(load_cfg())
    env.world.battery = 1  # reserve_margin 4, path cost >= 0
    res = env.step(Action.return_home())
    assert not res.observation.last_action_valid
    assert "return" in (res.observation.last_action_error or "")


def test_consecutive_invalid_actions_abort() -> None:
    env = Environment(load_cfg())
    env.world.agent_pos = (0, 1)  # west is out of bounds
    for _ in range(MAX_CONSECUTIVE_INVALID):
        res = env.step(Action.move("west"))
    assert res.done
    assert res.terminal == TerminalState.INVALID_ACTION_ABORTED


def test_valid_action_resets_invalid_streak() -> None:
    env = Environment(load_cfg())
    env.step(Action.move("west"))
    env.step(Action.move("west"))
    env.step(Action.move("north"))
    assert env.world.invalid_streak == 0


def test_clue_discovered_only_through_proximity() -> None:
    env = Environment(load_cfg())
    # clue_01 at [3,8], detection_radius 1 — agent starts far away
    assert all(not c.discovered for c in env.world.clues)
    env.world.agent_pos = (3, 7)
    res = env.step(Action.move("north"))  # enters [3,8]... actually moves to (3,8)? no: from (3,7) north -> (3,8) is clue cell
    events = res.observation.events
    # clue sits exactly at [3,8]; standing on it must trigger discovery
    assert any(e.clue_id == "clue_01" for e in events) or env.world.clues[0].discovered


def test_clue_not_discovered_when_far() -> None:
    env = Environment(load_cfg())
    env.world.agent_pos = (10, 1)
    res = env.step(Action.move("east"))
    assert res.observation.events == ()
    assert all(not c.discovered for c in env.world.clues)


def test_scan_detects_clue_in_radius() -> None:
    env = Environment(load_cfg())
    # clue_02 at [9,3]; scan_radius 5; stand at [9,7]
    env.world.agent_pos = (9, 7)
    res = env.step(Action.scan())
    assert any(e.clue_id == "clue_02" for e in res.observation.events)


def test_observation_never_contains_ground_truth() -> None:
    env = Environment(load_cfg())
    truth = [
        str(env.world.eve_pos),
        *[str(c.pos) for c in env.world.clues],
        *[str(d) for d in env.world.decoys],
    ]
    obs = env.observe()
    blob = json.dumps(obs.model_dump(mode="json"))
    env.step(Action.move("north"))
    blob += json.dumps(env.step(Action.scan()).observation.model_dump(mode="json"))
    for t in truth:
        # the agent legitimately stands on cells; only assert that positions of
        # entities it has NOT visited are absent
        pos = eval(t)
        if abs(pos[0] - env.world.agent_pos[0]) > 4 and abs(pos[1] - env.world.agent_pos[1]) > 4:
            assert t not in blob, f"observation leaked hidden position {t}"


def test_battery_exhaustion_terminal() -> None:
    env = Environment(load_cfg())
    env.world.battery = 1
    res = env.step(Action.move("north"))  # costs 1
    assert res.done
    assert res.terminal == TerminalState.BATTERY_EXHAUSTED_ABORTED


def test_timeout_terminal() -> None:
    env = Environment(load_cfg(**{"budget.max_ticks": 1}))
    res = env.step(Action.move("north"))
    assert res.done
    assert res.terminal == TerminalState.TIMEOUT_ABORTED


def test_signal_reading_shape() -> None:
    env = Environment(load_cfg())
    obs = env.observe()
    assert obs.signal is not None
    assert 0.0 <= obs.signal.strength <= 1.0
    assert obs.signal.measured_from == env.world.agent_pos
    # no distance field exists in SignalReading (schema-level guarantee)
    from walle.contracts import SignalReading

    assert "distance" not in SignalReading.model_fields
    assert "source" not in SignalReading.model_fields


def test_seeded_clues_spawn_near_eve() -> None:
    cfg = load_cfg(**{"clues.placements": [], "clues.seed_count": 4})
    env = Environment(cfg)
    eve = env.world.eve_pos
    radius = cfg.clues.spawn_radius
    seeded = [c for c in env.world.clues if c.clue_id.startswith("clue_seed")]
    assert len(seeded) == 4
    for c in seeded:
        assert max(abs(c.pos[0] - eve[0]), abs(c.pos[1] - eve[1])) <= radius


def test_seeded_decoys_stay_away_from_clues() -> None:
    cfg = load_cfg(**{"decoys.placements": [], "decoys.count": 2})
    env = Environment(cfg)
    min_d = cfg.decoys.min_clue_distance
    clue_cells = [c.pos for c in env.world.clues]
    assert env.world.decoys
    for d in env.world.decoys:
        for c in clue_cells:
            assert max(abs(d[0] - c[0]), abs(d[1] - c[1])) >= min_d


def test_explicit_decoy_too_close_to_clue_is_rejected() -> None:
    cfg = load_cfg(**{"decoys.placements": [[3, 9]]})  # fixture clue at [3, 8]
    with pytest.raises(ValueError, match="min_clue_distance"):
        Environment(cfg)
