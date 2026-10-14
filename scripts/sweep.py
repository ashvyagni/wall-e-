"""Multi-seed robustness sweep for the classical (and any) planner.

Generates N mission variants from a template config with seed-derived
entity placements (no explicit placements), runs each, and reports
success rate + terminal breakdown. Fully offline and deterministic.

Usage:
    PYTHONPATH=src python scripts/sweep.py --n 30 --template configs/sample_basic.json
"""

from __future__ import annotations

import argparse
import json
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
    args = ap.parse_args()

    template = MissionConfig.model_validate_json(Path(args.template).read_text())
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    terminals: Counter[str] = Counter()
    successes = 0
    errors: list[float] = []
    steps: list[int] = []
    rows: list[dict] = []

    for i in range(args.n):
        seed = args.seed_from + i
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

    summary = {
        "n": args.n,
        "successes": successes,
        "success_rate": successes / args.n if args.n else 0.0,
        "terminals": dict(terminals),
        "mean_steps": sum(steps) / len(steps) if steps else None,
        "mean_error": sum(errors) / len(errors) if errors else None,
        "planner": args.planner or template.planner,
        "template": args.template,
    }
    (out_dir / "sweep_summary.json").write_text(json.dumps(summary, indent=2))
    print()
    print(json.dumps(summary, indent=2))
    return 0 if successes > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
