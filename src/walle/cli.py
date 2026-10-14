"""Command-line interface: run missions from config files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from walle.agent.llm import ReplayMissError
from walle.contracts import MissionConfig
from walle.mission import planner_for, run_mission
from walle.trace.writer import read_trace
from walle.viz.ascii import render_ascii


def load_config(path: Path) -> MissionConfig:
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text) if path.suffix in {".yaml", ".yml"} else json.loads(text)
    return MissionConfig.model_validate(data)


def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(Path(args.config))
    if args.planner:
        cfg = cfg.model_copy(update={"planner": args.planner})
    planner = planner_for(cfg, args.planner, live=bool(args.live))
    out_dir = Path(args.out)
    report_dir = Path(args.reports)
    trace_path = out_dir / f"{cfg.mission_id}.jsonl"
    report_path = report_dir / f"{cfg.mission_id}_report.json"

    try:
        result = run_mission(cfg, planner, trace_path, report_path)
    except ReplayMissError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    r = result.report
    print(f"mission   : {r.mission_id} (seed {r.seed}, planner {r.planner})")
    print(f"terminal  : {r.outcome.terminal}  successful={r.outcome.successful}")
    print(
        f"steps={r.outcome.steps} ticks={r.outcome.ticks_used} "
        f"battery {r.outcome.battery_used} used / {r.outcome.battery_remaining} left"
    )
    print(
        f"inference : {r.inference.eve_location} "
        f"({r.inference.status}, conf {r.inference.confidence:.2f})"
    )
    print(f"truth     : {r.evaluation.true_eve_pos}  error={r.evaluation.localization_error}")
    print(f"clues     : {r.evaluation.clues_found}/{r.evaluation.clues_total} found")
    print(f"trace     : {result.trace_path}")
    print(f"report    : {result.report_path}")
    if args.ascii:
        events = read_trace(result.trace_path)
        print()
        print(render_ascii(cfg, events))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="walle", description="WALL-E and EVE's Signal")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run a mission from a config file")
    run_p.add_argument("config", help="path to mission config (json or yaml)")
    run_p.add_argument("--planner", choices=["classical", "llm"], default=None)
    run_p.add_argument(
        "--live",
        action="store_true",
        help="allow the llm planner to call the API on cache misses "
        "(requires WALLE_LLM_API_KEY/OPENAI_API_KEY; default is replay-only)",
    )
    run_p.add_argument("--out", default="traces", help="trace output directory")
    run_p.add_argument("--reports", default="reports", help="report output directory")
    run_p.add_argument("--ascii", action="store_true", help="print ASCII map after run")
    run_p.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
