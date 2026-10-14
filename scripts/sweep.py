"""Multi-seed robustness sweep for the classical (and any) planner.

Generates N mission variants from a template config with seed-derived
entity placements (no explicit placements), runs each, and reports
success rate + terminal breakdown. Fully offline and deterministic.

Usage:
    PYTHONPATH=src python scripts/sweep.py --n 30 --template configs/sample_basic.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path

from walle.contracts import MissionConfig
from walle.mission import planner_for, run_mission


def make_variant(template: MissionConfig, seed: int) -> MissionConfig:
    data = template.model_dump(mode="json")
    data["seed"] = seed
    data["mission_id"] = f"{template.mission_id}_s{seed}"
    data["target"]["placement"] = None
    data["clues"]["placements"] = []
    data["clues"]["seed_count"] = max(data["clues"]["seed_count"], 4)
    data["decoys"]["placements"] = []
    return MissionConfig.model_validate(data)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--template", default="configs/sample_basic.json")
    ap.add_argument("--n", type=int, default=30, help="number of seeds")
    ap.add_argument("--seed-from", type=int, default=1000, help="first seed")
    ap.add_argument("--planner", default=None, help="override planner kind")
    ap.add_argument("--out", default="runs/sweep", help="output dir")
    ap.add_argument(
        "--seeds-file",
        default=None,
        help="JSON file with a {\"seeds\": [...]} list (overrides --n/--seed-from)",
    )
    ap.add_argument("--summary", default=None, help="also write summary JSON here")
    args = ap.parse_args()

    template = MissionConfig.model_validate_json(Path(args.template).read_text())
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.seeds_file:
        seed_list = json.loads(Path(args.seeds_file).read_text())["seeds"]
        seeds_sha = hashlib.sha256(Path(args.seeds_file).read_bytes()).hexdigest()[:16]
    else:
        seed_list = list(range(args.seed_from, args.seed_from + args.n))
        seeds_sha = "adhoc-" + ",".join(str(s) for s in seed_list[:3])
    n = len(seed_list)

    terminals: Counter[str] = Counter()
    successes = 0
    errors: list[float] = []
    steps: list[int] = []
    rows: list[dict] = []

    for seed in seed_list:
        cfg = make_variant(template, seed)
        try:
            planner = planner_for(cfg, args.planner)
            result = run_mission(
                cfg,
                planner,
                out_dir / f"{cfg.mission_id}.jsonl",
                out_dir / f"{cfg.mission_id}_report.json",
            )
        except Exception as exc:
            terminals[f"EXCEPTION:{type(exc).__name__}"] += 1
            rows.append({"seed": seed, "terminal": f"EXCEPTION: {exc}"})
            print(f"seed {seed}: EXCEPTION {exc}")
            continue

        outcome = result.report.outcome
        term = outcome.terminal
        ok = outcome.successful
        terminals[term] += 1
        successes += int(ok)
        errors.append(float(result.report.evaluation.localization_error))
        steps.append(outcome.steps)
        rows.append(
            {
                "seed": seed,
                "terminal": term,
                "success": ok,
                "steps": outcome.steps,
                "battery_used": outcome.battery_used,
                "localization_error": result.report.evaluation.localization_error,
            }
        )
        print(f"seed {seed}: {term} steps={outcome.steps}")

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"

    summary = {
        "n": n,
        "successes": successes,
        "success_rate": successes / n if n else 0.0,
        "terminals": dict(terminals),
        "mean_steps": sum(steps) / len(steps) if steps else None,
        "mean_error": sum(errors) / len(errors) if errors else None,
        "planner": args.planner or template.planner,
        "template": args.template,
        "protocol": "bench/PROTOCOL.md v1.0" if args.seeds_file else None,
        "seeds_sha256_16": seeds_sha,
        "git_commit": commit,
        "host": platform.platform(),
    }
    (out_dir / "sweep_summary.json").write_text(json.dumps(summary, indent=2))
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(json.dumps(summary, indent=2))
    print()
    print(json.dumps(summary, indent=2))
    return 0 if successes > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
