"""Append-only JSONL trace writer (TraceEvent schema v1.0)."""

from __future__ import annotations

import json
from pathlib import Path

from walle.contracts import TraceEvent


class TraceWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        self.count = 0

    def write(self, event: TraceEvent) -> None:
        self._fh.write(json.dumps(event.model_dump(mode="json"), ensure_ascii=False) + "\n")
        self._fh.flush()
        self.count += 1

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> TraceWriter:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def read_trace(path: Path) -> list[TraceEvent]:
    events: list[TraceEvent] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            events.append(TraceEvent.model_validate(json.loads(line)))
    return events
