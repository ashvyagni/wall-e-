"""Export JSON Schemas for all contract documents to contracts/schema/."""

from __future__ import annotations

import json
from pathlib import Path

from walle.contracts import CONTRACT_MODELS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "contracts" / "schema"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, model in CONTRACT_MODELS.items():
        schema = model.model_json_schema(by_alias=True)
        path = OUT / f"{name}.schema.json"
        path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
